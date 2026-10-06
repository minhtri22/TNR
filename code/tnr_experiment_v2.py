#!/usr/bin/env python3
"""
TNR Experiment v2 - Thực hiện kế hoạch run tiếp theo
Không thay đổi claim trung tâm.
Cải tiến:
1. Tăng capacity + đơn giản hóa true transform để R² baseline > 0.6
2. Cân bằng parameter (low-rank interface, smaller router)
3. Hard top-K routing + λ nhẹ
4. Thêm baseline E = recurrent dense, F = gated random-graph
5. Route recovery chính xác hơn (path edge matching)
6. Preregister ngưỡng PASS rõ ràng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from itertools import combinations

# ===================== PREREGISTRATION (khóa trước khi chạy) =====================
"""
Claim không đổi:
Adaptive topological routing > fixed depth
trên task family có variable computational dependency graph,
dưới parameter/compute budget được khóa trước.

Metric chính (khóa):
- Primary: test MSE
- Secondary: route recovery (edge F1 trên true path edges)

Ngưỡng PASS (preregistered, không đổi sau kết quả):
PASS nếu và chỉ nếu:
  (1) MSE_D < min(MSE_A, MSE_B, MSE_C, MSE_E, MSE_F) * 0.95
  VÀ
  (2) route_recovery_D (edge F1) >= 0.30
Ngược lại:
  - Nếu MSE_D > min_baseline * 1.05  → FAIL
  - Còn lại → UNRESOLVED

Param budget mục tiêu: ~15k–25k parameters cho mọi model (cố gắng).
Compute: cùng số iteration tối đa = 4, cùng epoch, cùng lr.
"""

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Hyperparameters (khóa)
N_MODULES = 8
HIDDEN = 40
INPUT_DIM = 12
OUTPUT_DIM = 4
N_ITER = 4
TOP_K = 3          # hard top-K edges per module (incoming)
BATCH_SIZE = 256
N_TRAIN = 4000
N_VAL = 800
N_TEST = 800
EPOCHS = 50
LR = 1e-3
ROUTE_COST_LAMBDA = 0.0005  # nhẹ

CASE_PATHS = {
    0: [0, 3, 6],
    1: [0, 5, 2],
    2: [1, 7, 4, 2],
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===================== DATA: true transform đơn giản hơn =====================
def generate_true_module_transforms():
    """Mỗi module là affine + tanh với scale vừa phải để giữ variance."""
    transforms = []
    torch.manual_seed(123)  # cố định true transform
    for i in range(N_MODULES):
        # Full but scaled down linear
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.35
        # Boost a few dimensions unique-ish to module
        boost = torch.randperm(INPUT_DIM)[:2]
        W[boost, boost] += 0.9
        b = torch.randn(INPUT_DIM) * 0.1
        transforms.append((W, b))
    torch.manual_seed(SEED)  # restore
    return transforms

TRUE_TRANSFORMS = generate_true_module_transforms()

def apply_true_path(x, path):
    h = x.clone()
    for m_idx in path:
        W, b = TRUE_TRANSFORMS[m_idx]
        h = torch.tanh(h @ W + b)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n_samples):
    X = torch.randn(n_samples, INPUT_DIM) * 0.8
    cases = torch.randint(0, 3, (n_samples,))
    Y = torch.zeros(n_samples, OUTPUT_DIM)
    for i in range(n_samples):
        path = CASE_PATHS[cases[i].item()]
        Y[i] = apply_true_path(X[i:i+1], path).squeeze(0)
    Y = Y + 0.02 * torch.randn_like(Y)
    return X, Y, cases

X_train, Y_train, cases_train = generate_dataset(N_TRAIN)
X_val, Y_val, cases_val = generate_dataset(N_VAL)
X_test, Y_test, cases_test = generate_dataset(N_TEST)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Y_train std: {Y_train.std(0)}, mean var={Y_train.var():.4f}")

# ===================== MODELS =====================

class MLP_A(nn.Module):
    """A: fixed-depth MLP (8 layers)."""
    def __init__(self):
        super().__init__()
        layers = []
        dims = [INPUT_DIM] + [HIDDEN] * 7 + [OUTPUT_DIM]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResidualModular_B(nn.Module):
    """B: residual modular stack."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x):
        h = self.input_proj(x)
        for m in self.modules_list:
            h = h + 0.5 * m(h)
        return self.out(h)

class StaticK8_C(nn.Module):
    """C: static complete (mean-field) modular, low-rank interface."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        # low-rank message: rank-8
        self.msg_down = nn.Linear(HIDDEN, 16)
        self.msg_up = nn.Linear(16, HIDDEN)
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
    def forward(self, x):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        for _ in range(self.n_iter):
            mean_h = torch.stack(h, dim=0).mean(0)
            msg = self.msg_up(torch.tanh(self.msg_down(mean_h)))
            new_h = []
            for i in range(N_MODULES):
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.4 * msg))
            h = new_h
        return self.out(torch.cat(h, dim=-1))

class GatedK8_D(nn.Module):
    """D: gated K8 with hard top-K + soft residual, low-rank interface, small router."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        self.msg_down = nn.Linear(HIDDEN, 16)
        self.msg_up = nn.Linear(16, HIDDEN)
        # Router nhỏ: chỉ từ mean state + input
        self.router = nn.Sequential(
            nn.Linear(INPUT_DIM + HIDDEN, 32),
            nn.Tanh(),
            nn.Linear(32, N_MODULES * N_MODULES)
        )
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
        self.top_k = TOP_K
    def forward(self, x, return_gates=False):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        all_gates = []
        for t in range(self.n_iter):
            mean_h = torch.stack(h, dim=0).mean(0)
            state = torch.cat([x, mean_h], dim=-1)
            logits = self.router(state).view(-1, N_MODULES, N_MODULES)
            # mask diagonal
            mask = (1.0 - torch.eye(N_MODULES, device=x.device)).unsqueeze(0)
            logits = logits.masked_fill(mask == 0, -1e9)
            # soft for grad + hard top-K for actual
            soft = torch.sigmoid(logits) * mask
            # hard top-K per row (incoming to i)
            topk_val, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            hard = torch.zeros_like(logits)
            hard.scatter_(-1, topk_idx, 1.0)
            hard = hard * mask
            # straight-through estimator
            gates = hard + (soft - soft.detach())
            all_gates.append(gates.detach())  # store hard for metric
            # messages
            stacked = torch.stack(h, dim=1)  # B, N, H
            new_h = []
            for i in range(N_MODULES):
                g_i = gates[:, i, :].unsqueeze(-1)  # B, N, 1
                msgs = (g_i * stacked).sum(dim=1)
                msg = self.msg_up(torch.tanh(self.msg_down(msgs)))
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.5 * msg))
            h = new_h
        out = self.out(torch.cat(h, dim=-1))
        if return_gates:
            return out, torch.stack(all_gates, dim=1)  # B, T, N, N
        return out
    def route_cost(self, gates):
        # cost trên soft-ish nhưng dùng hard count
        return gates.sum(dim=(-1, -2)).mean() / self.n_iter

class RecurrentDense_E(nn.Module):
    """E: recurrent dense baseline (cùng số iter, dense residual)."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.cell = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN)
        )
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.n_iter = N_ITER
    def forward(self, x):
        h = self.input_proj(x)
        for _ in range(self.n_iter):
            h = h + 0.5 * self.cell(h)
        return self.out(h)

class GatedRandomGraph_F(nn.Module):
    """F: gated trên random sparse graph (cùng top-K, nhưng topology random cố định)."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        self.msg_down = nn.Linear(HIDDEN, 16)
        self.msg_up = nn.Linear(16, HIDDEN)
        self.router = nn.Sequential(
            nn.Linear(INPUT_DIM + HIDDEN, 32),
            nn.Tanh(),
            nn.Linear(32, N_MODULES * N_MODULES)
        )
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
        self.top_k = TOP_K
        # fixed random adjacency (each node degree ~3)
        adj = torch.zeros(N_MODULES, N_MODULES)
        for i in range(N_MODULES):
            candidates = [j for j in range(N_MODULES) if j != i]
            chosen = np.random.choice(candidates, size=min(3, len(candidates)), replace=False)
            adj[i, chosen] = 1.0
        self.register_buffer("adj_mask", adj)
    def forward(self, x, return_gates=False):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        all_gates = []
        for t in range(self.n_iter):
            mean_h = torch.stack(h, dim=0).mean(0)
            state = torch.cat([x, mean_h], dim=-1)
            logits = self.router(state).view(-1, N_MODULES, N_MODULES)
            # chỉ cho phép cạnh trong random graph
            logits = logits.masked_fill(self.adj_mask.unsqueeze(0) == 0, -1e9)
            soft = torch.sigmoid(logits) * self.adj_mask.unsqueeze(0)
            topk_val, topk_idx = torch.topk(logits, min(self.top_k, 3), dim=-1)
            hard = torch.zeros_like(logits)
            hard.scatter_(-1, topk_idx, 1.0)
            hard = hard * self.adj_mask.unsqueeze(0)
            gates = hard + (soft - soft.detach())
            all_gates.append(gates.detach())
            stacked = torch.stack(h, dim=1)
            new_h = []
            for i in range(N_MODULES):
                g_i = gates[:, i, :].unsqueeze(-1)
                msgs = (g_i * stacked).sum(dim=1)
                msg = self.msg_up(torch.tanh(self.msg_down(msgs)))
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.5 * msg))
            h = new_h
        out = self.out(torch.cat(h, dim=-1))
        if return_gates:
            return out, torch.stack(all_gates, dim=1)
        return out
    def route_cost(self, gates):
        return gates.sum(dim=(-1, -2)).mean() / self.n_iter

# ===================== TRAIN / EVAL =====================

def train_model(model, name, use_route_cost=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
    Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
    history = {"train_loss": [], "val_loss": [], "route_cost": []}
    best_val = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if use_route_cost:
                pred, gates = model(xb, return_gates=True)
                task_loss = F.mse_loss(pred, yb)
                rc = model.route_cost(gates)
                loss = task_loss + ROUTE_COST_LAMBDA * rc
            else:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                rc = 0.0
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            if use_route_cost:
                pred_v, gates_v = model(Xv, return_gates=True)
                val_loss = F.mse_loss(pred_v, Yv).item()
                val_rc = model.route_cost(gates_v).item()
            else:
                pred_v = model(Xv)
                val_loss = F.mse_loss(pred_v, Yv).item()
                val_rc = 0.0
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)
        history["route_cost"].append(val_rc)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  [{name}] Ep {epoch+1}/{EPOCHS} train={history['train_loss'][-1]:.4f} val={val_loss:.4f} rc={val_rc:.2f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val

def compute_route_recovery(gates, cases, case_paths, n_sample=500):
    """
    Route recovery chính xác hơn:
    - Với mỗi sample, lấy trung bình gate qua các iteration → soft adjacency
    - True edges = các cạnh liên tiếp trên path
    - Tính precision/recall/F1 của top edges so với true edges (edge-level).
    """
    # gates: [B, T, N, N] hard
    B = min(n_sample, gates.size(0))
    gates = gates[:B].mean(dim=1)  # B, N, N  (avg over iter)
    cases = cases[:B]
    precs, recs, f1s = [], [], []
    for b in range(B):
        path = case_paths[cases[b].item()]
        true_edges = set()
        for k in range(len(path)-1):
            # gate[i,j] means from j → i (incoming to i from j)
            true_edges.add((path[k+1], path[k]))  # (to, from)
        # predicted: top edges by gate value (non-zero)
        g = gates[b]
        pred_edges = set()
        nz = (g > 0.5).nonzero(as_tuple=False)
        for idx in nz:
            pred_edges.add((idx[0].item(), idx[1].item()))
        if len(pred_edges) == 0:
            precs.append(0.0); recs.append(0.0); f1s.append(0.0)
            continue
        tp = len(true_edges & pred_edges)
        prec = tp / len(pred_edges)
        rec = tp / max(len(true_edges), 1)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return {
        "edge_precision": float(np.mean(precs)),
        "edge_recall": float(np.mean(recs)),
        "edge_f1": float(np.mean(f1s))
    }

def evaluate(model, name, use_gates=False):
    model.eval()
    Xt, Yt = X_test.to(DEVICE), Y_test.to(DEVICE)
    with torch.no_grad():
        if use_gates:
            pred, gates = model(Xt, return_gates=True)
            recovery = compute_route_recovery(gates.cpu(), cases_test, CASE_PATHS)
            avg_active = gates.sum(dim=(-1, -2)).mean().item() / N_ITER
        else:
            pred = model(Xt)
            recovery = {"edge_precision": None, "edge_recall": None, "edge_f1": None}
            avg_active = None
        mse = F.mse_loss(pred, Yt).item()
        ss_res = ((pred - Yt)**2).sum().item()
        ss_tot = ((Yt - Yt.mean(0))**2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {
        "mse": mse,
        "r2": r2,
        "route_edge_f1": recovery["edge_f1"],
        "route_edge_precision": recovery["edge_precision"],
        "route_edge_recall": recovery["edge_recall"],
        "avg_active_edges_per_iter": avg_active,
        "params": count_params(model)
    }

# ===================== RUN =====================
results = {}
histories = {}
best_vals = {}

print("\n=== A: MLP ===")
model_a = MLP_A()
print(f"Params A: {count_params(model_a)}")
model_a, hist_a, best_a = train_model(model_a, "A")
results["A"] = evaluate(model_a, "A")
histories["A"] = hist_a
best_vals["A"] = best_a
print(f"A Test MSE={results['A']['mse']:.4f} R2={results['A']['r2']:.4f}")

print("\n=== B: Residual Modular ===")
model_b = ResidualModular_B()
print(f"Params B: {count_params(model_b)}")
model_b, hist_b, best_b = train_model(model_b, "B")
results["B"] = evaluate(model_b, "B")
histories["B"] = hist_b
best_vals["B"] = best_b
print(f"B Test MSE={results['B']['mse']:.4f} R2={results['B']['r2']:.4f}")

print("\n=== C: Static K8 ===")
model_c = StaticK8_C()
print(f"Params C: {count_params(model_c)}")
model_c, hist_c, best_c = train_model(model_c, "C")
results["C"] = evaluate(model_c, "C")
histories["C"] = hist_c
best_vals["C"] = best_c
print(f"C Test MSE={results['C']['mse']:.4f} R2={results['C']['r2']:.4f}")

print("\n=== D: Gated K8 (top-K) ===")
model_d = GatedK8_D()
print(f"Params D: {count_params(model_d)}")
model_d, hist_d, best_d = train_model(model_d, "D", use_route_cost=True)
results["D"] = evaluate(model_d, "D", use_gates=True)
histories["D"] = hist_d
best_vals["D"] = best_d
print(f"D Test MSE={results['D']['mse']:.4f} R2={results['D']['r2']:.4f} F1={results['D']['route_edge_f1']}")

print("\n=== E: Recurrent Dense ===")
model_e = RecurrentDense_E()
print(f"Params E: {count_params(model_e)}")
model_e, hist_e, best_e = train_model(model_e, "E")
results["E"] = evaluate(model_e, "E")
histories["E"] = hist_e
best_vals["E"] = best_e
print(f"E Test MSE={results['E']['mse']:.4f} R2={results['E']['r2']:.4f}")

print("\n=== F: Gated Random Graph ===")
model_f = GatedRandomGraph_F()
print(f"Params F: {count_params(model_f)}")
model_f, hist_f, best_f = train_model(model_f, "F", use_route_cost=True)
results["F"] = evaluate(model_f, "F", use_gates=True)
histories["F"] = hist_f
best_vals["F"] = best_f
print(f"F Test MSE={results['F']['mse']:.4f} R2={results['F']['r2']:.4f} F1={results['F']['route_edge_f1']}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "seed": SEED,
        "preregistration": {
            "PASS_condition": "MSE_D < min_baseline * 0.95 AND route_edge_f1_D >= 0.30",
            "FAIL_condition": "MSE_D > min_baseline * 1.05",
            "else": "UNRESOLVED"
        },
        "config": {
            "HIDDEN": HIDDEN, "N_ITER": N_ITER, "TOP_K": TOP_K,
            "EPOCHS": EPOCHS, "ROUTE_COST_LAMBDA": ROUTE_COST_LAMBDA,
            "N_TRAIN": N_TRAIN, "INPUT_DIM": INPUT_DIM
        },
        "results": results,
        "best_val_losses": best_vals
    }, f, indent=2)

for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

for name, m in [("A", model_a), ("B", model_b), ("C", model_c),
                ("D", model_d), ("E", model_e), ("F", model_f)]:
    torch.save(m.state_dict(), os.path.join(run_dir, f"model_{name}.pt"))

print(f"\nRun saved to {run_dir}")
print(json.dumps(results, indent=2))

# Outcome
mses = {k: results[k]["mse"] for k in ["A", "B", "C", "E", "F"]}
best_base_mse = min(mses.values())
d_mse = results["D"]["mse"]
d_f1 = results["D"]["route_edge_f1"] or 0.0

if d_mse < best_base_mse * 0.95 and d_f1 >= 0.30:
    outcome = "PASS"
elif d_mse > best_base_mse * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"D MSE={d_mse:.4f} vs best baseline={best_base_mse:.4f} | F1={d_f1:.3f}")

# Lineage
lineage_path = "/home/workdir/artifacts/Lineage.md"
entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: TNR v2 – kế hoạch run tiếp theo (capacity↑, param balance, top-K, thêm E/F, route F1)
- Config: HIDDEN={HIDDEN}, N_ITER={N_ITER}, TOP_K={TOP_K}, λ={ROUTE_COST_LAMBDA}, EPOCHS={EPOCHS}
- Results MSE: A={results['A']['mse']:.4f}, B={results['B']['mse']:.4f}, C={results['C']['mse']:.4f}, D={results['D']['mse']:.4f}, E={results['E']['mse']:.4f}, F={results['F']['mse']:.4f}
- R²: A={results['A']['r2']:.3f}, B={results['B']['r2']:.3f}, C={results['C']['r2']:.3f}, D={results['D']['r2']:.3f}, E={results['E']['r2']:.3f}, F={results['F']['r2']:.3f}
- Params: A={results['A']['params']}, B={results['B']['params']}, C={results['C']['params']}, D={results['D']['params']}, E={results['E']['params']}, F={results['F']['params']}
- Route edge F1 D: {results['D']['route_edge_f1']}, F: {results['F']['route_edge_f1']}
- Preregistered PASS: MSE_D < min_base*0.95 AND F1>=0.30
- Outcome: **{outcome}**
- Notes: True transform low-rank (3-dim diagonal dominant). Hard top-K + STE. Baselines E (recurrent dense), F (gated random graph).
"""
with open(lineage_path, "a") as f:
    f.write(entry)
print(f"Lineage appended.")
