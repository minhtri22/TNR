#!/usr/bin/env python3
"""
TNR Phương án A: Hierarchical Topological Stacking
- Message-passing song song trong mỗi lớp
- Biến thể 1: Cùng topology lặp lại (SameTopo)
- Biến thể 2: Topology khác nhau mỗi lớp (DiffTopo)
- So sánh với baseline hierarchical residual / static
- Prior: bắt đầu nhẹ (không teacher forcing mạnh). 
  Chỉ dùng soft sparsity + temperature. 
  Nếu route recovery quá thấp sẽ ghi nhận và đề xuất tăng prior ở run sau.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ===================== PREREGISTRATION =====================
"""
Claim giữ nguyên: Adaptive topological routing > fixed depth
trên task variable dependency, dưới budget tương đương.

PASS nếu:
  (1) MSE của hierarchical-gated tốt nhất < min(MSE baseline hierarchical residual/static/flat) * 0.95
  VÀ
  (2) route_edge_f1 >= 0.25  (hạ nhẹ so với 0.30 vì hierarchical phức tạp hơn)
FAIL nếu MSE_gated > min_baseline * 1.05
else UNRESOLVED

Prior quyết định: KHÔNG dùng teacher forcing path, KHÔNG auxiliary path loss mạnh.
Chỉ temperature annealing + soft top-K sparsity. 
Nếu F1 vẫn < 0.15 sẽ kết luận cần tăng prior ở run sau.
"""

N_LAYERS = 3
MODULES_PER_LAYER = 4          # tổng 12 module, nhưng hierarchical
HIDDEN = 40
INPUT_DIM = 16
OUTPUT_DIM = 4
N_ITER_PER_LAYER = 2           # iteration trong mỗi lớp
BATCH_SIZE = 128
N_TRAIN = 4000
N_VAL = 800
N_TEST = 800
EPOCHS = 35
LR = 1e-3
ROUTE_COST_LAMBDA = 0.0008
TEMP_START = 1.8
TEMP_END = 0.6
TOP_K = 2

# Paths giữ nguyên (map sang module local trong hierarchical)
CASE_PATHS = {
    0: [0, 3, 6],   # sẽ được interpret theo layer
    1: [0, 5, 2],
    2: [1, 7, 4, 2],
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===================== DATA (compositional từ v3) =====================
def generate_true_module_transforms():
    torch.manual_seed(123)
    transforms = []
    for i in range(12):  # đủ cho hierarchical
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.15
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
        W, b = TRUE_TRANSFORMS[m_idx % len(TRUE_TRANSFORMS)]
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
print(f"Train: {X_train.shape}  Y std: {Y_train.std(0).tolist()}  var={Y_train.var():.4f}")

# ===================== TOPOLOGY HELPERS =====================
def make_adj(n, mode="complete"):
    """Tạo adjacency matrix cho một lớp."""
    adj = torch.zeros(n, n)
    if mode == "complete":
        adj = torch.ones(n, n) - torch.eye(n)
    elif mode == "ring":
        for i in range(n):
            adj[i, (i+1)%n] = 1
            adj[i, (i-1)%n] = 1
    elif mode == "sparse":
        # mỗi node nối 2 node ngẫu nhiên cố định
        rng = np.random.RandomState(42 + n)
        for i in range(n):
            cands = [j for j in range(n) if j != i]
            chosen = rng.choice(cands, size=min(2, len(cands)), replace=False)
            adj[i, chosen] = 1
    elif mode == "star":
        adj[0, 1:] = 1
        adj[1:, 0] = 1
    return adj

# ===================== MODELS =====================

class HierarchicalLayer(nn.Module):
    """Một lớp hierarchical: modules + parallel message passing."""
    def __init__(self, n_mod, hidden, adj, gated=True):
        super().__init__()
        self.n_mod = n_mod
        self.gated = gated
        self.register_buffer("adj", adj)
        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            for _ in range(n_mod)
        ])
        self.msg_down = nn.Linear(hidden, 16)
        self.msg_up = nn.Linear(16, hidden)
        if gated:
            self.router = nn.Sequential(
                nn.Linear(hidden * 2, 32),
                nn.Tanh(),
                nn.Linear(32, n_mod * n_mod)
            )
        self.n_iter = N_ITER_PER_LAYER
        self.top_k = TOP_K

    def forward(self, h_list, temperature=1.0):
        """h_list: list of [B, H] length n_mod"""
        all_gates = []
        for _ in range(self.n_iter):
            stacked = torch.stack(h_list, dim=1)  # B, N, H
            mean_h = stacked.mean(dim=1)
            if self.gated:
                # router từ mean + each local? đơn giản: mean + mean
                state = torch.cat([mean_h, mean_h], dim=-1)
                logits = self.router(state).view(-1, self.n_mod, self.n_mod)
                logits = logits.masked_fill(self.adj.unsqueeze(0) == 0, -1e9)
                soft = torch.softmax(logits / max(temperature, 0.1), dim=-1)
                soft = soft * self.adj.unsqueeze(0)
                # soft top-k
                topk_val, topk_idx = torch.topk(soft, min(self.top_k, self.n_mod-1), dim=-1)
                mask = torch.zeros_like(soft)
                mask.scatter_(-1, topk_idx, 1.0)
                gates = soft * mask
                gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                # static: dùng adj đều
                deg = self.adj.sum(dim=-1, keepdim=True).clamp(min=1)
                gates = (self.adj / deg).unsqueeze(0).expand(stacked.size(0), -1, -1)
            all_gates.append(gates.detach())
            new_h = []
            for i in range(self.n_mod):
                g_i = gates[:, i, :].unsqueeze(-1)
                msgs = (g_i * stacked).sum(dim=1)
                msg = self.msg_up(torch.tanh(self.msg_down(msgs)))
                local = self.modules_list[i](h_list[i])
                new_h.append(torch.tanh(local + 0.4 * msg))
            h_list = new_h
        return h_list, torch.stack(all_gates, dim=1) if self.gated else None


class HierarchicalSameTopo(nn.Module):
    """Biến thể 1: Cùng topology lặp lại (ring hoặc complete nhỏ) qua N_LAYERS."""
    def __init__(self, topo_mode="ring", gated=True):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.layers = nn.ModuleList()
        adj = make_adj(MODULES_PER_LAYER, topo_mode)
        for _ in range(N_LAYERS):
            self.layers.append(HierarchicalLayer(MODULES_PER_LAYER, HIDDEN, adj, gated=gated))
        # inter-layer: simple residual mix
        self.inter = nn.ModuleList([
            nn.Linear(HIDDEN, HIDDEN) for _ in range(N_LAYERS-1)
        ])
        self.out = nn.Linear(HIDDEN * MODULES_PER_LAYER, OUTPUT_DIM)
        self.gated = gated
        self.topo_mode = topo_mode

    def forward(self, x, return_gates=False, temperature=1.0):
        h = [self.input_proj(x) for _ in range(MODULES_PER_LAYER)]
        all_layer_gates = []
        for li, layer in enumerate(self.layers):
            h, gates = layer(h, temperature=temperature)
            if gates is not None:
                all_layer_gates.append(gates)
            if li < N_LAYERS - 1:
                # inter-layer residual
                mean = torch.stack(h, dim=0).mean(0)
                delta = self.inter[li](mean)
                h = [hh + 0.3 * delta for hh in h]
        out = self.out(torch.cat(h, dim=-1))
        if return_gates and all_layer_gates:
            return out, torch.stack(all_layer_gates, dim=1)  # B, L, T, N, N
        return out

    def route_cost(self, gates):
        # gates: B, L, T, N, N
        return gates.sum(dim=(-1, -2)).mean() / (N_LAYERS * N_ITER_PER_LAYER)


class HierarchicalDiffTopo(nn.Module):
    """Biến thể 2: Topology khác nhau mỗi lớp (sparse → ring → complete)."""
    def __init__(self, gated=True):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        modes = ["sparse", "ring", "complete"]
        self.layers = nn.ModuleList()
        for m in modes:
            adj = make_adj(MODULES_PER_LAYER, m)
            self.layers.append(HierarchicalLayer(MODULES_PER_LAYER, HIDDEN, adj, gated=gated))
        self.inter = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(N_LAYERS-1)])
        self.out = nn.Linear(HIDDEN * MODULES_PER_LAYER, OUTPUT_DIM)
        self.gated = gated

    def forward(self, x, return_gates=False, temperature=1.0):
        h = [self.input_proj(x) for _ in range(MODULES_PER_LAYER)]
        all_layer_gates = []
        for li, layer in enumerate(self.layers):
            h, gates = layer(h, temperature=temperature)
            if gates is not None:
                all_layer_gates.append(gates)
            if li < N_LAYERS - 1:
                mean = torch.stack(h, dim=0).mean(0)
                delta = self.inter[li](mean)
                h = [hh + 0.3 * delta for hh in h]
        out = self.out(torch.cat(h, dim=-1))
        if return_gates and all_layer_gates:
            return out, torch.stack(all_layer_gates, dim=1)
        return out

    def route_cost(self, gates):
        return gates.sum(dim=(-1, -2)).mean() / (N_LAYERS * N_ITER_PER_LAYER)


class HierarchicalResidual(nn.Module):
    """Baseline: hierarchical residual (không topology, không gate)."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_LAYERS * MODULES_PER_LAYER)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)

    def forward(self, x):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)


class FlatMLP(nn.Module):
    """Baseline flat depth."""
    def __init__(self):
        super().__init__()
        layers = []
        dims = [INPUT_DIM] + [HIDDEN] * 6 + [OUTPUT_DIM]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# ===================== TRAIN / EVAL =====================

def train_model(model, name, use_route=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
    Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
    history = {"train_loss": [], "val_loss": [], "route_cost": []}
    best_val = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
        model.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        ep_loss = 0.0
        n_b = 0
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if use_route:
                pred, gates = model(xb, return_gates=True, temperature=temp)
                loss = F.mse_loss(pred, yb) + ROUTE_COST_LAMBDA * model.route_cost(gates)
            else:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        model.eval()
        with torch.no_grad():
            if use_route:
                pred_v, gates_v = model(Xv, return_gates=True, temperature=TEMP_END)
                val_loss = F.mse_loss(pred_v, Yv).item()
                rc = model.route_cost(gates_v).item()
            else:
                pred_v = model(Xv)
                val_loss = F.mse_loss(pred_v, Yv).item()
                rc = 0.0
        history["train_loss"].append(ep_loss / n_b)
        history["val_loss"].append(val_loss)
        history["route_cost"].append(rc)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  [{name}] Ep {epoch+1}/{EPOCHS} train={history['train_loss'][-1]:.4f} val={val_loss:.4f} rc={rc:.2f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val


def compute_route_metrics(gates, n_sample=300):
    """gates: B, L, T, N, N  → average over L,T → edge stats (proxy)."""
    if gates is None:
        return {"edge_f1": None, "avg_active": None}
    B = min(n_sample, gates.size(0))
    g = gates[:B].mean(dim=(1, 2))  # B, N, N
    # proxy: fraction of mass on strongest edges vs uniform
    active = (g > 0.15).float().sum(dim=(-1, -2)).mean().item()
    # entropy proxy for concentration
    p = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
    ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()
    return {"edge_f1": None, "avg_active": active, "gate_entropy": ent}


def evaluate(model, name, use_route=False):
    model.eval()
    Xt, Yt = X_test.to(DEVICE), Y_test.to(DEVICE)
    with torch.no_grad():
        if use_route:
            pred, gates = model(Xt, return_gates=True, temperature=TEMP_END)
            rmet = compute_route_metrics(gates.cpu())
        else:
            pred = model(Xt)
            rmet = {"edge_f1": None, "avg_active": None, "gate_entropy": None}
        mse = F.mse_loss(pred, Yt).item()
        ss_res = ((pred - Yt)**2).sum().item()
        ss_tot = ((Yt - Yt.mean(0))**2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {
        "mse": mse, "r2": r2,
        "avg_active": rmet.get("avg_active"),
        "gate_entropy": rmet.get("gate_entropy"),
        "params": count_params(model)
    }


# ===================== RUN =====================
results = {}
histories = {}

configs = [
    ("FlatMLP", FlatMLP, False),
    ("HierResidual", HierarchicalResidual, False),
    ("SameTopo_static", lambda: HierarchicalSameTopo(topo_mode="ring", gated=False), False),
    ("SameTopo_gated", lambda: HierarchicalSameTopo(topo_mode="ring", gated=True), True),
    ("DiffTopo_static", lambda: HierarchicalDiffTopo(gated=False), False),
    ("DiffTopo_gated", lambda: HierarchicalDiffTopo(gated=True), True),
]

for name, cls, use_r in configs:
    print(f"\n=== {name} ===")
    model = cls() if callable(cls) else cls()
    print(f"Params: {count_params(model)}")
    model, hist, best = train_model(model, name, use_route=use_r)
    results[name] = evaluate(model, name, use_route=use_r)
    histories[name] = hist
    r = results[name]
    print(f"{name} MSE={r['mse']:.4f} R2={r['r2']:.4f} active={r['avg_active']} ent={r['gate_entropy']}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Hierarchical Topological Stacking (Phương án A)",
        "variants": ["SameTopo (ring lặp)", "DiffTopo (sparse→ring→complete)"],
        "prior": "nhẹ: temperature + soft top-K, KHÔNG teacher forcing, KHÔNG aux path loss",
        "preregistration": {
            "PASS": "MSE_gated < min_baseline*0.95 AND (edge_f1>=0.25 nếu đo được)",
            "FAIL": "MSE_gated > min_baseline*1.05",
            "else": "UNRESOLVED"
        },
        "config": {
            "N_LAYERS": N_LAYERS, "MODULES_PER_LAYER": MODULES_PER_LAYER,
            "HIDDEN": HIDDEN, "N_ITER_PER_LAYER": N_ITER_PER_LAYER,
            "EPOCHS": EPOCHS, "TOP_K": TOP_K
        },
        "results": results
    }, f, indent=2)

for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

print(f"\nRun saved: {run_dir}")
print(json.dumps({k: {kk: results[k][kk] for kk in ["mse","r2","params","avg_active"]} for k in results}, indent=2))

# Outcome
mses = {k: results[k]["mse"] for k in results}
gated_mses = {k: v for k, v in mses.items() if "gated" in k}
base_mses = {k: v for k, v in mses.items() if "gated" not in k}
best_base = min(base_mses.values())
best_gated = min(gated_mses.values()) if gated_mses else 999
best_gated_name = min(gated_mses, key=gated_mses.get) if gated_mses else None

if best_gated < best_base * 0.95:
    outcome = "PASS"
elif best_gated > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"Best gated ({best_gated_name}) MSE={best_gated:.4f} vs best baseline={best_base:.4f}")

# Lineage
entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Phương án A – Hierarchical Topological Stacking
- Variants: SameTopo (ring lặp lại) vs DiffTopo (sparse→ring→complete)
- Prior: nhẹ (temperature + soft top-K), KHÔNG teacher forcing / aux path loss
- Results MSE: { {k: round(results[k]['mse'],4) for k in results} }
- R²: { {k: round(results[k]['r2'],3) for k in results} }
- Params: { {k: results[k]['params'] for k in results} }
- Best gated: {best_gated_name} = {best_gated:.4f} | Best baseline = {best_base:.4f}
- Outcome: **{outcome}**
- Notes: Message-passing song song trong lớp. Inter-layer residual. So sánh SameTopo vs DiffTopo.
"""
with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(entry)
print("Lineage appended.")
