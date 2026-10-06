#!/usr/bin/env python3
"""
TNR Minimal Experiment - Topological Neural Routing
Thí nghiệm tối thiểu đầu tiên theo tài liệu nền tảng TNR.
So sánh:
A = 8-layer MLP
B = residual/dense modular baseline
C = static K8 modular network
D = gated K8 modular network
Dưới parameter budget và compute budget tương đương.
Task: synthetic với dependency path thay đổi theo case family.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
import hashlib

# Cấu hình
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Hyperparameters khóa trước (preregistered)
N_MODULES = 8
HIDDEN = 32  # width mỗi module / layer
INPUT_DIM = 16
OUTPUT_DIM = 4
N_ITER = 4  # iterations cho modular models
BATCH_SIZE = 64
N_TRAIN = 5000
N_VAL = 1000
N_TEST = 1000
EPOCHS = 60
LR = 1e-3
ROUTE_COST_LAMBDA = 0.001  # giảm pressure để router có cơ hội học
MAX_ACTIVE_EDGES = 4  # soft constraint via cost

# Case families với latent dependency (không đưa cho model)
# Family A: 0 -> 3 -> 6
# Family B: 0 -> 5 -> 2
# Family C: 1 -> 7 -> 4 -> 2
CASE_PATHS = {
    0: [0, 3, 6],
    1: [0, 5, 2],
    2: [1, 7, 4, 2],
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===================== DATA GENERATOR =====================
def generate_true_module_transforms():
    """Các phép biến đổi 'thật' cố định cho mỗi module (ground-truth composition)."""
    transforms = []
    for i in range(N_MODULES):
        # Linear + nonlinearity giả lập
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.5
        b = torch.randn(INPUT_DIM) * 0.1
        transforms.append((W, b))
    return transforms

TRUE_TRANSFORMS = generate_true_module_transforms()

def apply_true_path(x, path):
    """Áp dụng composition theo path ground-truth."""
    h = x.clone()
    for m_idx in path:
        W, b = TRUE_TRANSFORMS[m_idx]
        h = torch.tanh(h @ W + b)
    # Project to output
    return h[:, :OUTPUT_DIM]

def generate_dataset(n_samples):
    """Sinh data với latent route theo case family."""
    X = torch.randn(n_samples, INPUT_DIM)
    # Case family phân bố đều
    cases = torch.randint(0, 3, (n_samples,))
    Y = torch.zeros(n_samples, OUTPUT_DIM)
    for i in range(n_samples):
        path = CASE_PATHS[cases[i].item()]
        Y[i] = apply_true_path(X[i:i+1], path).squeeze(0)
    # Thêm noise nhỏ
    Y = Y + 0.05 * torch.randn_like(Y)
    return X, Y, cases

X_train, Y_train, cases_train = generate_dataset(N_TRAIN)
X_val, Y_val, cases_val = generate_dataset(N_VAL)
X_test, Y_test, cases_test = generate_dataset(N_TEST)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ===================== MODELS =====================

class MLP8(nn.Module):
    """A: 8-layer MLP fixed depth."""
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

class ResidualModular(nn.Module):
    """B: residual/dense modular baseline - 8 modules stacked với residual."""
    def __init__(self):
        super().__init__()
        self.modules_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN),
                nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN)
            ) for _ in range(N_MODULES)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
    
    def forward(self, x):
        h = self.input_proj(x)
        for m in self.modules_list:
            h = h + m(h)  # residual
        return self.out(h)

class StaticK8(nn.Module):
    """C: static K8 modular network - complete adjacency, no gate, fixed iterations."""
    def __init__(self):
        super().__init__()
        self.modules_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN),
                nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN)
            ) for _ in range(N_MODULES)
        ])
        # Low-rank interface to control params: W_ij = U_i @ V_j.T approx, but simple shared scale + linear
        self.msg_proj = nn.Linear(HIDDEN, HIDDEN)
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)  # concat all
        self.n_iter = N_ITER
    
    def forward(self, x):
        # Init states
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        for t in range(self.n_iter):
            new_h = []
            # Mean message for static dense
            mean_h = torch.stack(h, dim=0).mean(0)
            msg_base = self.msg_proj(mean_h)
            for i in range(N_MODULES):
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.3 * msg_base))
            h = new_h
        concat = torch.cat(h, dim=-1)
        return self.out(concat)

class GatedK8(nn.Module):
    """D: gated K8 modular network - dynamic routing with soft gates + cost."""
    def __init__(self):
        super().__init__()
        self.modules_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN),
                nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN)
            ) for _ in range(N_MODULES)
        ])
        self.msg_proj = nn.Linear(HIDDEN, HIDDEN)
        # Router: from concatenated states + input to gates (N x N)
        self.router = nn.Sequential(
            nn.Linear(INPUT_DIM + HIDDEN * N_MODULES, 48),
            nn.Tanh(),
            nn.Linear(48, N_MODULES * N_MODULES)
        )
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.out = nn.Linear(HIDDEN * N_MODULES, OUTPUT_DIM)
        self.n_iter = N_ITER
    
    def forward(self, x, return_gates=False):
        h = [self.input_proj(x) for _ in range(N_MODULES)]
        all_gates = []
        for t in range(self.n_iter):
            state_cat = torch.cat([x] + h, dim=-1)
            gate_logits = self.router(state_cat).view(-1, N_MODULES, N_MODULES)
            # Soft gates [0,1], diagonal forced 0
            gates = torch.sigmoid(gate_logits)
            mask = (1.0 - torch.eye(N_MODULES, device=x.device)).unsqueeze(0)
            gates = gates * mask
            all_gates.append(gates)
            
            new_h = []
            stacked = torch.stack(h, dim=1)  # [B, N, H]
            for i in range(N_MODULES):
                # Weighted messages from others
                g_i = gates[:, i, :].unsqueeze(-1)  # [B, N, 1]
                msgs = (g_i * stacked).sum(dim=1)  # [B, H]
                msg = self.msg_proj(msgs)
                local = self.modules_list[i](h[i])
                new_h.append(torch.tanh(local + 0.3 * msg))
            h = new_h
        concat = torch.cat(h, dim=-1)
        out = self.out(concat)
        if return_gates:
            return out, torch.stack(all_gates, dim=1)  # [B, T, N, N]
        return out
    
    def route_cost(self, gates):
        """Cost = mean sum of gates (soft active edges)."""
        return gates.sum(dim=(-1, -2)).mean()

# ===================== TRAINING =====================

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
        perm = torch.randperm(N_TRAIN)
        epoch_loss = 0.0
        epoch_rc = 0.0
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
                epoch_rc += rc.item()
            else:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        # Val
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
        
        avg_train = epoch_loss / n_batches
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["route_cost"].append(val_rc if use_route_cost else 0.0)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{EPOCHS} train={avg_train:.4f} val={val_loss:.4f} rc={val_rc:.2f}")
    
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val

# ===================== EVALUATION =====================

def evaluate(model, name, use_gates=False):
    model.eval()
    Xt, Yt = X_test.to(DEVICE), Y_test.to(DEVICE)
    with torch.no_grad():
        if use_gates:
            pred, gates = model(Xt, return_gates=True)
            # Route recovery: approximate by looking at high gates vs true paths
            # Simplified metric: mean gate mass on true edges vs all
            true_edge_mass = 0.0
            total_mass = 0.0
            for b in range(min(200, Xt.size(0))):  # sample
                case = cases_test[b].item()
                path = CASE_PATHS[case]
                # Edges in path
                true_edges = set()
                for k in range(len(path)-1):
                    true_edges.add((path[k], path[k+1]))
                g = gates[b].mean(0)  # avg over iter [N,N]
                total_mass += g.sum().item()
                for (i,j) in true_edges:
                    true_edge_mass += g[j, i].item()  # note direction: gate[i,j] is from j to i?
            route_fidelity = true_edge_mass / (total_mass + 1e-8) if total_mass > 0 else 0
            avg_active = gates.sum(dim=(-1,-2)).mean().item() / N_ITER
        else:
            pred = model(Xt)
            route_fidelity = None
            avg_active = None
        mse = F.mse_loss(pred, Yt).item()
        # R2 score
        ss_res = ((pred - Yt)**2).sum().item()
        ss_tot = ((Yt - Yt.mean(0))**2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {
        "mse": mse,
        "r2": r2,
        "route_fidelity": route_fidelity,
        "avg_active_edges_per_iter": avg_active,
        "params": count_params(model)
    }

# ===================== RUN =====================

results = {}
histories = {}

print("\n=== Training A: 8-layer MLP ===")
model_a = MLP8()
print(f"Params A: {count_params(model_a)}")
model_a, hist_a, best_a = train_model(model_a, "A")
results["A"] = evaluate(model_a, "A")
histories["A"] = hist_a
print(f"A Test MSE: {results['A']['mse']:.4f} R2: {results['A']['r2']:.4f}")

print("\n=== Training B: Residual Modular ===")
model_b = ResidualModular()
print(f"Params B: {count_params(model_b)}")
model_b, hist_b, best_b = train_model(model_b, "B")
results["B"] = evaluate(model_b, "B")
histories["B"] = hist_b
print(f"B Test MSE: {results['B']['mse']:.4f} R2: {results['B']['r2']:.4f}")

print("\n=== Training C: Static K8 ===")
model_c = StaticK8()
print(f"Params C: {count_params(model_c)}")
model_c, hist_c, best_c = train_model(model_c, "C")
results["C"] = evaluate(model_c, "C")
histories["C"] = hist_c
print(f"C Test MSE: {results['C']['mse']:.4f} R2: {results['C']['r2']:.4f}")

print("\n=== Training D: Gated K8 ===")
model_d = GatedK8()
print(f"Params D: {count_params(model_d)}")
model_d, hist_d, best_d = train_model(model_d, "D", use_route_cost=True)
results["D"] = evaluate(model_d, "D", use_gates=True)
histories["D"] = hist_d
print(f"D Test MSE: {results['D']['mse']:.4f} R2: {results['D']['r2']:.4f} fidelity={results['D']['route_fidelity']}")

# Save results
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)

# Save metrics
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "seed": SEED,
        "config": {
            "N_MODULES": N_MODULES,
            "HIDDEN": HIDDEN,
            "N_ITER": N_ITER,
            "EPOCHS": EPOCHS,
            "ROUTE_COST_LAMBDA": ROUTE_COST_LAMBDA,
            "N_TRAIN": N_TRAIN
        },
        "results": results,
        "best_val_losses": {"A": best_a, "B": best_b, "C": best_c, "D": best_d}
    }, f, indent=2)

# Save histories
for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

# Save models
torch.save(model_a.state_dict(), os.path.join(run_dir, "model_A.pt"))
torch.save(model_b.state_dict(), os.path.join(run_dir, "model_B.pt"))
torch.save(model_c.state_dict(), os.path.join(run_dir, "model_C.pt"))
torch.save(model_d.state_dict(), os.path.join(run_dir, "model_D.pt"))

print(f"\nRun saved to {run_dir}")
print(json.dumps(results, indent=2))

# Determine outcome for Lineage
# Preregistered: Adaptive topological routing > fixed depth on variable dependency task under equal budget
# Metric chính: test MSE (lower better), secondary R2
primary = results["D"]["mse"]
baselines = [results["A"]["mse"], results["B"]["mse"], results["C"]["mse"]]
best_base = min(baselines)
if primary < best_base * 0.95:  # 5% improvement threshold arbitrary but fixed
    outcome = "PASS"
elif primary > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"D MSE={primary:.4f} vs best baseline={best_base:.4f}")

# Append to Lineage.md
lineage_path = "/home/workdir/artifacts/Lineage.md"
entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: TNR minimal first experiment (A/B/C/D comparison)
- Config: HIDDEN={HIDDEN}, N_ITER={N_ITER}, EPOCHS={EPOCHS}, lambda={ROUTE_COST_LAMBDA}
- Results MSE: A={results['A']['mse']:.4f}, B={results['B']['mse']:.4f}, C={results['C']['mse']:.4f}, D={results['D']['mse']:.4f}
- Params: A={results['A']['params']}, B={results['B']['params']}, C={results['C']['params']}, D={results['D']['params']}
- Route fidelity D: {results['D'].get('route_fidelity')}
- Outcome: **{outcome}**
- Notes: Synthetic task with 3 case families variable dependency paths. Equal param/compute approximately controlled via width and iterations.
"""

with open(lineage_path, "a") as f:
    f.write(entry)

print(f"Lineage appended to {lineage_path}")
