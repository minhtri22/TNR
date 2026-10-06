#!/usr/bin/env python3
"""
TNR Experiment v3 - Kế hoạch run tiếp theo (không đổi claim)
1. HIDDEN >= 48, N_ITER=5, data ~5k hướng R² baseline > 0.6
2. True transform compositional: mỗi module cộng dồn feature riêng
3. Soft gates + temperature annealing (thay pure hard top-K)
4. Cân bằng param E (recurrent wider)
5. Thêm trajectory recovery (sequence matching)
6. Giữ nguyên ngưỡng PASS preregistered
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

# ===================== PREREGISTRATION (giữ nguyên) =====================
"""
Claim không đổi.
PASS nếu và chỉ nếu:
  (1) MSE_D < min(MSE_A..F) * 0.95
  VÀ
  (2) route_edge_f1_D >= 0.30
FAIL nếu MSE_D > min_baseline * 1.05
else UNRESOLVED
"""

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_MODULES = 8
HIDDEN = 48
INPUT_DIM = 16
OUTPUT_DIM = 4
N_ITER = 5
TOP_K = 3
BATCH_SIZE = 128
N_TRAIN = 5000
N_VAL = 1000
N_TEST = 1000
EPOCHS = 30
LR = 1e-3
ROUTE_COST_LAMBDA = 0.001
TEMP_START = 2.0
TEMP_END = 0.5

CASE_PATHS = {
    0: [0, 3, 6],
    1: [0, 5, 2],
    2: [1, 7, 4, 2],
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===================== DATA: compositional true transform =====================
def generate_true_module_transforms():
    """Mỗi module cộng dồn một feature riêng (compositional additive + mild nonlinear)."""
    torch.manual_seed(123)
    transforms = []
    for i in range(N_MODULES):
        # Projection riêng cho module i
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.15
        # Unique boost: module i mạnh trên 2 chiều cố định theo i
        d1 = i % INPUT_DIM
        d2 = (i * 3 + 1) % INPUT_DIM
        W[d1, d1] += 1.1
        W[d2, d2] += 0.9
        b = torch.zeros(INPUT_DIM)
        b[d1] = 0.3 * (i % 3 - 1)
        transforms.append((W, b))
    torch.manual_seed(SEED)
    return transforms

TRUE_TRANSFORMS = generate_true_module_transforms()

def apply_true_path(x, path):
    h = x.clone()
    for m_idx in path:
        W, b = TRUE_TRANSFORMS[m_idx]
        # residual-style composition để giữ signal
        h = h + 0.6 * torch.tanh(h @ W + b)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n_samples):
    X = torch.randn(n_samples, INPUT_DIM) * 0.7
    cases = torch.randint(0, 3, (n_samples,))
    Y = torch.zeros(n_samples, OUTPUT_DIM)
    for i in range(n_samples):
        path = CASE_PATHS[cases[i].item()]
        Y[i] = apply_true_path(X[i:i+1], path).squeeze(0)
    Y = Y + 0.015 * torch.randn_like(Y)
    return X, Y, cases

X_train, Y_train, cases_train = generate_dataset(N_TRAIN)
X_val, Y_val, cases_val = generate_dataset(N_VAL)
X_test, Y_test, cases_test = generate_dataset(N_TEST)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Y_train std: {Y_train.std(0).tolist()}, var={Y_train.var():.4f}")

# ===================== MODELS =====================

class MLP_A(nn.Module):
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
            h = h + 0.4 * m(h)
        return self.out(h)

class StaticK8_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        self.msg_down = nn.Linear(HIDDEN, 20)
        self.msg_up = nn.Linear(20, HIDDEN)
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
    def forward(self, x):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        for _ in range(self.n_iter):
            mean_h = torch.stack(h, dim=0).mean(0)
            msg = self.msg_up(torch.tanh(self.msg_down(mean_h)))
            new_h = [torch.tanh(self.modules_list[i](h[i]) + 0.35 * msg) for i in range(N_MODULES)]
            h = new_h
        return self.out(torch.cat(h, dim=-1))

class GatedK8_D(nn.Module):
    """Soft gates + temperature annealing + light top-K encouragement."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        self.msg_down = nn.Linear(HIDDEN, 20)
        self.msg_up = nn.Linear(20, HIDDEN)
        self.router = nn.Sequential(
            nn.Linear(INPUT_DIM + HIDDEN, 48),
            nn.Tanh(),
            nn.Linear(48, N_MODULES * N_MODULES)
        )
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
        self.top_k = TOP_K
    def forward(self, x, return_gates=False, temperature=1.0):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        all_gates = []
        for t in range(self.n_iter):
            mean_h = torch.stack(h, dim=0).mean(0)
            state = torch.cat([x, mean_h], dim=-1)
            logits = self.router(state).view(-1, N_MODULES, N_MODULES)
            mask = (1.0 - torch.eye(N_MODULES, device=x.device)).unsqueeze(0)
            logits = logits.masked_fill(mask == 0, -1e9)
            # Soft with temperature
            soft = torch.softmax(logits / max(temperature, 0.1), dim=-1) * mask
            # Encourage sparsity via soft top-k mask (keep top-k mass)
            topk_val, topk_idx = torch.topk(soft, self.top_k, dim=-1)
            sparse_mask = torch.zeros_like(soft)
            sparse_mask.scatter_(-1, topk_idx, 1.0)
            gates = soft * sparse_mask  # soft values only on top-k positions
            # renormalize lightly
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
            all_gates.append(gates.detach())
            stacked = torch.stack(h, dim=1)
            new_h = []
            for i in range(N_MODULES):
                g_i = gates[:, i, :].unsqueeze(-1)
                msgs = (g_i * stacked).sum(dim=1)
                msg = self.msg_up(torch.tanh(self.msg_down(msgs)))
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.4 * msg))
            h = new_h
        out = self.out(torch.cat(h, dim=-1))
        if return_gates:
            return out, torch.stack(all_gates, dim=1)
        return out
    def route_cost(self, gates):
        return gates.sum(dim=(-1, -2)).mean() / self.n_iter

class RecurrentDense_E(nn.Module):
    """E cân bằng param hơn: wider + multi-step residual."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.cell1 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
        self.cell2 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.n_iter = N_ITER
    def forward(self, x):
        h = self.input_proj(x)
        for t in range(self.n_iter):
            h = h + 0.35 * self.cell1(h)
            if t % 2 == 1:
                h = h + 0.25 * self.cell2(h)
        return self.out(h)

class GatedRandomGraph_F(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_MODULES)
        ])
        self.msg_down = nn.Linear(HIDDEN, 20)
        self.msg_up = nn.Linear(20, HIDDEN)
        self.router = nn.Sequential(
            nn.Linear(INPUT_DIM + HIDDEN, 48),
            nn.Tanh(),
            nn.Linear(48, N_MODULES * N_MODULES)
        )
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
        self.top_k = TOP_K
        adj = torch.zeros(N_MODULES, N_MODULES)
        rng = np.random.RandomState(99)
        for i in range(N_MODULES):
            cands = [j for j in range(N_MODULES) if j != i]
            chosen = rng.choice(cands, size=3, replace=False)
            adj[i, chosen] = 1.0
        self.register_buffer("adj_mask", adj)
    def forward(self, x, return_gates=False, temperature=1.0):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        all_gates = []
        for t in range(self.n_iter):
            mean_h = torch.stack(h, dim=0).mean(0)
            state = torch.cat([x, mean_h], dim=-1)
            logits = self.router(state).view(-1, N_MODULES, N_MODULES)
            logits = logits.masked_fill(self.adj_mask.unsqueeze(0) == 0, -1e9)
            soft = torch.softmax(logits / max(temperature, 0.1), dim=-1) * self.adj_mask.unsqueeze(0)
            topk_val, topk_idx = torch.topk(soft, min(self.top_k, 3), dim=-1)
            sparse_mask = torch.zeros_like(soft)
            sparse_mask.scatter_(-1, topk_idx, 1.0)
            gates = soft * sparse_mask
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
            all_gates.append(gates.detach())
            stacked = torch.stack(h, dim=1)
            new_h = []
            for i in range(N_MODULES):
                g_i = gates[:, i, :].unsqueeze(-1)
                msgs = (g_i * stacked).sum(dim=1)
                msg = self.msg_up(torch.tanh(self.msg_down(msgs)))
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.4 * msg))
            h = new_h
        out = self.out(torch.cat(h, dim=-1))
        if return_gates:
            return out, torch.stack(all_gates, dim=1)
        return out
    def route_cost(self, gates):
        return gates.sum(dim=(-1, -2)).mean() / self.n_iter

# ===================== METRICS =====================

def compute_route_recovery(gates, cases, case_paths, n_sample=400):
    B = min(n_sample, gates.size(0))
    gates_avg = gates[:B].mean(dim=1)  # B, N, N
    cases = cases[:B]
    precs, recs, f1s = [], [], []
    traj_hits = []
    for b in range(B):
        path = case_paths[cases[b].item()]
        true_edges = set()
        for k in range(len(path)-1):
            true_edges.add((path[k+1], path[k]))  # (to, from)
        g = gates_avg[b]
        pred_edges = set()
        nz = (g > 0.15).nonzero(as_tuple=False)
        for idx in nz:
            pred_edges.add((idx[0].item(), idx[1].item()))
        if len(pred_edges) == 0:
            precs.append(0.0); recs.append(0.0); f1s.append(0.0)
        else:
            tp = len(true_edges & pred_edges)
            prec = tp / len(pred_edges)
            rec = tp / max(len(true_edges), 1)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            precs.append(prec); recs.append(rec); f1s.append(f1)
        # Trajectory recovery: top activated modules vs path modules
        node_act = g.sum(dim=1) + g.sum(dim=0)  # strength per node
        top_nodes = set(node_act.topk(min(len(path)+1, N_MODULES)).indices.tolist())
        path_nodes = set(path)
        traj_hits.append(len(top_nodes & path_nodes) / max(len(path_nodes), 1))
    return {
        "edge_precision": float(np.mean(precs)),
        "edge_recall": float(np.mean(recs)),
        "edge_f1": float(np.mean(f1s)),
        "traj_node_recall": float(np.mean(traj_hits))
    }

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
        # temperature annealing
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
        model.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if use_route_cost:
                pred, gates = model(xb, return_gates=True, temperature=temp)
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
                pred_v, gates_v = model(Xv, return_gates=True, temperature=TEMP_END)
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
            print(f"  [{name}] Ep {epoch+1}/{EPOCHS} train={history['train_loss'][-1]:.4f} val={val_loss:.4f} rc={val_rc:.2f} T={temp:.2f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val

def evaluate(model, name, use_gates=False):
    model.eval()
    Xt, Yt = X_test.to(DEVICE), Y_test.to(DEVICE)
    with torch.no_grad():
        if use_gates:
            pred, gates = model(Xt, return_gates=True, temperature=TEMP_END)
            recovery = compute_route_recovery(gates.cpu(), cases_test, CASE_PATHS)
            avg_active = (gates > 0.1).float().sum(dim=(-1, -2)).mean().item() / N_ITER
        else:
            pred = model(Xt)
            recovery = {"edge_precision": None, "edge_recall": None, "edge_f1": None, "traj_node_recall": None}
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
        "traj_node_recall": recovery["traj_node_recall"],
        "avg_active_edges_per_iter": avg_active,
        "params": count_params(model)
    }

# ===================== RUN =====================
results = {}
histories = {}
best_vals = {}

models_to_run = [
    ("A", MLP_A, False),
    ("B", ResidualModular_B, False),
    ("C", StaticK8_C, False),
    ("D", GatedK8_D, True),
    ("E", RecurrentDense_E, False),
    ("F", GatedRandomGraph_F, True),
]

for name, cls, use_rc in models_to_run:
    print(f"\n=== {name} ===")
    model = cls()
    print(f"Params {name}: {count_params(model)}")
    model, hist, best = train_model(model, name, use_route_cost=use_rc)
    results[name] = evaluate(model, name, use_gates=use_rc)
    histories[name] = hist
    best_vals[name] = best
    r = results[name]
    extra = f" F1={r['route_edge_f1']:.3f} traj={r['traj_node_recall']:.3f}" if r['route_edge_f1'] is not None else ""
    print(f"{name} Test MSE={r['mse']:.4f} R2={r['r2']:.4f}{extra}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "seed": SEED,
        "preregistration": {
            "PASS": "MSE_D < min_baseline*0.95 AND route_edge_f1_D >= 0.30",
            "FAIL": "MSE_D > min_baseline*1.05",
            "else": "UNRESOLVED"
        },
        "config": {
            "HIDDEN": HIDDEN, "N_ITER": N_ITER, "TOP_K": TOP_K,
            "EPOCHS": EPOCHS, "ROUTE_COST_LAMBDA": ROUTE_COST_LAMBDA,
            "N_TRAIN": N_TRAIN, "TEMP_START": TEMP_START, "TEMP_END": TEMP_END,
            "true_transform": "compositional residual additive unique dims"
        },
        "results": results,
        "best_val_losses": best_vals
    }, f, indent=2)

for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

# Note: models not saved to keep runtime short; metrics + history sufficient
print(f"\nRun saved to {run_dir}")
print(json.dumps({k: {kk: results[k][kk] for kk in ["mse","r2","route_edge_f1","traj_node_recall","params"]} for k in results}, indent=2))

# Outcome
mses = {k: results[k]["mse"] for k in ["A","B","C","E","F"]}
best_base = min(mses.values())
d_mse = results["D"]["mse"]
d_f1 = results["D"]["route_edge_f1"] or 0.0

if d_mse < best_base * 0.95 and d_f1 >= 0.30:
    outcome = "PASS"
elif d_mse > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"D MSE={d_mse:.4f} vs best_base={best_base:.4f} | F1={d_f1:.3f}")

lineage_path = "/home/workdir/artifacts/Lineage.md"
entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: TNR v3 – kế hoạch run tiếp theo (capacity↑ R² target, compositional transform, soft+temp, traj metric, E balanced)
- Config: HIDDEN={HIDDEN}, N_ITER={N_ITER}, TOP_K={TOP_K}, λ={ROUTE_COST_LAMBDA}, EPOCHS={EPOCHS}, T={TEMP_START}→{TEMP_END}
- Results MSE: A={results['A']['mse']:.4f}, B={results['B']['mse']:.4f}, C={results['C']['mse']:.4f}, D={results['D']['mse']:.4f}, E={results['E']['mse']:.4f}, F={results['F']['mse']:.4f}
- R²: A={results['A']['r2']:.3f}, B={results['B']['r2']:.3f}, C={results['C']['r2']:.3f}, D={results['D']['r2']:.3f}, E={results['E']['r2']:.3f}, F={results['F']['r2']:.3f}
- Params: A={results['A']['params']}, B={results['B']['params']}, C={results['C']['params']}, D={results['D']['params']}, E={results['E']['params']}, F={results['F']['params']}
- Route edge F1 D: {results['D']['route_edge_f1']}, traj_node_recall D: {results['D']['traj_node_recall']}
- Route edge F1 F: {results['F']['route_edge_f1']}, traj F: {results['F']['traj_node_recall']}
- Preregistered PASS: MSE_D < min_base*0.95 AND F1>=0.30
- Outcome: **{outcome}**
- Notes: Compositional residual true transform (unique dims per module). Soft gates + temperature annealing + soft top-K. Trajectory node recall added.
"""
with open(lineage_path, "a") as f:
    f.write(entry)
print("Lineage appended.")
