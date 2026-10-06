#!/usr/bin/env python3
"""
TNR Hierarchical A + Prior tăng có kiểm soát
- Giữ DiffTopo hierarchical (sparse → ring → complete)
- So sánh 3 mức prior:
  P0: chỉ temperature + soft top-K (như run trước)
  P1: + auxiliary loss nhẹ khuyến khích gate mass tập trung (entropy penalty)
  P2: + soft curriculum target (soft path bias giảm dần theo epoch)
- Message-passing song song
- Không full teacher forcing
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
PASS nếu:
  MSE của P1 hoặc P2 < min(MSE P0, HierResidual, DiffTopo_static) * 0.95
  VÀ gate_entropy giảm rõ (tập trung hơn) so với P0
FAIL nếu MSE_P1/P2 > min_baseline * 1.05
else UNRESOLVED
"""

N_LAYERS = 3
MODULES_PER_LAYER = 4
HIDDEN = 40
INPUT_DIM = 16
OUTPUT_DIM = 4
N_ITER_PER_LAYER = 2
BATCH_SIZE = 128
N_TRAIN = 4000
N_VAL = 800
N_TEST = 800
EPOCHS = 35
LR = 1e-3
ROUTE_COST_LAMBDA = 0.0008
ENTROPY_LAMBDA = 0.01          # P1
CURRICULUM_LAMBDA_START = 0.05 # P2
TEMP_START, TEMP_END = 1.8, 0.6
TOP_K = 2

CASE_PATHS = {0: [0, 3, 6], 1: [0, 5, 2], 2: [1, 7, 4, 2]}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ===================== DATA =====================
def generate_true_module_transforms():
    torch.manual_seed(123)
    transforms = []
    for i in range(12):
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.15
        d1, d2 = i % INPUT_DIM, (i * 3 + 1) % INPUT_DIM
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
        W, b = TRUE_TRANSFORMS[m_idx % 12]
        h = h + 0.6 * torch.tanh(h @ W + b)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n):
    X = torch.randn(n, INPUT_DIM) * 0.7
    cases = torch.randint(0, 3, (n,))
    Y = torch.zeros(n, OUTPUT_DIM)
    for i in range(n):
        Y[i] = apply_true_path(X[i:i+1], CASE_PATHS[cases[i].item()]).squeeze(0)
    return X, Y + 0.015 * torch.randn_like(Y), cases

X_train, Y_train, cases_train = generate_dataset(N_TRAIN)
X_val, Y_val, cases_val = generate_dataset(N_VAL)
X_test, Y_test, cases_test = generate_dataset(N_TEST)
print(f"Train {X_train.shape}  Y var={Y_train.var():.4f}")

# ===================== TOPOLOGY + LAYER =====================
def make_adj(n, mode):
    adj = torch.zeros(n, n)
    if mode == "complete":
        adj = torch.ones(n, n) - torch.eye(n)
    elif mode == "ring":
        for i in range(n):
            adj[i, (i+1)%n] = 1
            adj[i, (i-1)%n] = 1
    elif mode == "sparse":
        rng = np.random.RandomState(42 + n)
        for i in range(n):
            cands = [j for j in range(n) if j != i]
            chosen = rng.choice(cands, size=min(2, len(cands)), replace=False)
            adj[i, chosen] = 1.0
    return adj

class HierarchicalLayer(nn.Module):
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
                nn.Linear(hidden * 2, 32), nn.Tanh(), nn.Linear(32, n_mod * n_mod)
            )
        self.n_iter = N_ITER_PER_LAYER
        self.top_k = TOP_K

    def forward(self, h_list, temperature=1.0):
        all_gates = []
        for _ in range(self.n_iter):
            stacked = torch.stack(h_list, dim=1)
            mean_h = stacked.mean(1)
            if self.gated:
                state = torch.cat([mean_h, mean_h], dim=-1)
                logits = self.router(state).view(-1, self.n_mod, self.n_mod)
                logits = logits.masked_fill(self.adj.unsqueeze(0) == 0, -1e9)
                soft = torch.softmax(logits / max(temperature, 0.1), dim=-1) * self.adj.unsqueeze(0)
                topk_val, topk_idx = torch.topk(soft, min(self.top_k, self.n_mod-1), dim=-1)
                mask = torch.zeros_like(soft)
                mask.scatter_(-1, topk_idx, 1.0)
                gates = soft * mask
                gates = gates / (gates.sum(-1, keepdim=True) + 1e-8)
            else:
                deg = self.adj.sum(-1, keepdim=True).clamp(min=1)
                gates = (self.adj / deg).unsqueeze(0).expand(stacked.size(0), -1, -1)
            all_gates.append(gates)
            new_h = []
            for i in range(self.n_mod):
                g_i = gates[:, i, :].unsqueeze(-1)
                msgs = (g_i * stacked).sum(1)
                msg = self.msg_up(torch.tanh(self.msg_down(msgs)))
                local = self.modules_list[i](h_list[i])
                new_h.append(torch.tanh(local + 0.4 * msg))
            h_list = new_h
        return h_list, torch.stack(all_gates, 1) if self.gated else None


class DiffTopoHier(nn.Module):
    def __init__(self, gated=True):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        modes = ["sparse", "ring", "complete"]
        self.layers = nn.ModuleList([
            HierarchicalLayer(MODULES_PER_LAYER, HIDDEN, make_adj(MODULES_PER_LAYER, m), gated)
            for m in modes
        ])
        self.inter = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(N_LAYERS-1)])
        self.out = nn.Linear(HIDDEN * MODULES_PER_LAYER, OUTPUT_DIM)
        self.gated = gated

    def forward(self, x, return_gates=False, temperature=1.0):
        h = [self.input_proj(x) for _ in range(MODULES_PER_LAYER)]
        all_g = []
        for li, layer in enumerate(self.layers):
            h, gates = layer(h, temperature)
            if gates is not None:
                all_g.append(gates)
            if li < N_LAYERS - 1:
                mean = torch.stack(h, 0).mean(0)
                delta = self.inter[li](mean)
                h = [hh + 0.3 * delta for hh in h]
        out = self.out(torch.cat(h, -1))
        if return_gates and all_g:
            return out, torch.stack(all_g, 1)  # B, L, T, N, N
        return out

    def route_cost(self, gates):
        return gates.sum((-1, -2)).mean() / (N_LAYERS * N_ITER_PER_LAYER)

    def entropy_cost(self, gates):
        # khuyến khích tập trung (thấp entropy)
        p = gates / (gates.sum(-1, keepdim=True) + 1e-8)
        ent = -(p * (p + 1e-8).log()).sum(-1).mean()
        return ent

    def curriculum_bias_loss(self, gates, cases, strength):
        """Soft bias: khuyến khích mass trên vài cạnh 'hợp lý' theo case (không hard target)."""
        if strength < 1e-6:
            return torch.tensor(0.0, device=gates.device)
        # proxy: với mỗi case, ưu tiên cạnh (0→1), (1→2) trong lớp đầu
        # đơn giản: tăng mass trên diagonal-ish của lớp 0
        g0 = gates[:, 0].mean(1)  # B, N, N  (avg over iter of layer 0)
        # target soft: mass cao hơn trên vài vị trí
        target = torch.zeros_like(g0)
        for b in range(g0.size(0)):
            c = cases[b].item() % 3
            target[b, (c+1) % 4, c % 4] = 1.0
            target[b, (c+2) % 4, (c+1) % 4] = 0.7
        target = target / (target.sum((-1, -2), keepdim=True) + 1e-8)
        # KL-like
        p = g0 / (g0.sum((-1, -2), keepdim=True) + 1e-8)
        loss = (target * (target + 1e-8).log() - target * (p + 1e-8).log()).sum((-1, -2)).mean()
        return strength * loss


class HierResidual(nn.Module):
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


# ===================== TRAIN =====================
def train_model(model, name, prior_level=0, cases_tr=None, cases_v=None):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
    Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
    if cases_tr is not None:
        cases_tr = cases_tr.to(DEVICE)
        cases_v = cases_v.to(DEVICE)
    history = {"train_loss": [], "val_loss": [], "entropy": []}
    best_val, best_state = float("inf"), None

    for epoch in range(EPOCHS):
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
        # curriculum strength giảm dần
        curr_str = CURRICULUM_LAMBDA_START * (1.0 - epoch / max(EPOCHS-1, 1)) if prior_level >= 2 else 0.0
        model.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        ep_loss, n_b = 0.0, 0
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if not getattr(model, "gated", False):
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            else:
                pred, gates = model(xb, return_gates=True, temperature=temp)
                loss = F.mse_loss(pred, yb) + ROUTE_COST_LAMBDA * model.route_cost(gates)
                if prior_level >= 1:
                    loss = loss + ENTROPY_LAMBDA * model.entropy_cost(gates)
                if prior_level >= 2 and cases_tr is not None:
                    loss = loss + model.curriculum_bias_loss(gates, cases_tr[idx], curr_str)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        model.eval()
        with torch.no_grad():
            if getattr(model, "gated", False):
                pred_v, gates_v = model(Xv, return_gates=True, temperature=TEMP_END)
                val_loss = F.mse_loss(pred_v, Yv).item()
                ent = model.entropy_cost(gates_v).item()
            else:
                pred_v = model(Xv)
                val_loss = F.mse_loss(pred_v, Yv).item()
                ent = 0.0
        history["train_loss"].append(ep_loss / n_b)
        history["val_loss"].append(val_loss)
        history["entropy"].append(ent)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  [{name}] Ep {epoch+1}/{EPOCHS} train={history['train_loss'][-1]:.4f} val={val_loss:.4f} ent={ent:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val


def evaluate(model, name, use_gates=False):
    model.eval()
    Xt, Yt = X_test.to(DEVICE), Y_test.to(DEVICE)
    with torch.no_grad():
        if use_gates:
            pred, gates = model(Xt, return_gates=True, temperature=TEMP_END)
            ent = model.entropy_cost(gates).item()
            active = (gates > 0.15).float().sum((-1, -2)).mean().item() / (N_LAYERS * N_ITER_PER_LAYER)
        else:
            pred = model(Xt)
            ent, active = None, None
        mse = F.mse_loss(pred, Yt).item()
        r2 = 1 - ((pred - Yt)**2).sum().item() / (((Yt - Yt.mean(0))**2).sum().item() + 1e-8)
    return {"mse": mse, "r2": r2, "entropy": ent, "avg_active": active, "params": count_params(model)}


# ===================== RUN =====================
results, histories = {}, {}

print("\n=== HierResidual baseline ===")
m = HierResidual()
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "HierResidual", prior_level=0)
results["HierResidual"] = evaluate(m, "HierResidual")
histories["HierResidual"] = h
print(f"MSE={results['HierResidual']['mse']:.4f} R2={results['HierResidual']['r2']:.4f}")

print("\n=== DiffTopo_static ===")
m = DiffTopoHier(gated=False)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "DiffTopo_static", prior_level=0)
results["DiffTopo_static"] = evaluate(m, "DiffTopo_static")
histories["DiffTopo_static"] = h
print(f"MSE={results['DiffTopo_static']['mse']:.4f} R2={results['DiffTopo_static']['r2']:.4f}")

for plevel, pname in [(0, "P0_light"), (1, "P1_entropy"), (2, "P2_curriculum")]:
    print(f"\n=== DiffTopo_gated {pname} ===")
    m = DiffTopoHier(gated=True)
    print(f"Params: {count_params(m)}")
    m, h, _ = train_model(m, pname, prior_level=plevel, cases_tr=cases_train, cases_v=cases_val)
    results[pname] = evaluate(m, pname, use_gates=True)
    histories[pname] = h
    r = results[pname]
    print(f"MSE={r['mse']:.4f} R2={r['r2']:.4f} ent={r['entropy']:.3f} active={r['avg_active']:.2f}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Hierarchical DiffTopo + prior levels P0/P1/P2",
        "prior_levels": {
            "P0": "temperature + soft top-K",
            "P1": "P0 + entropy penalty",
            "P2": "P1 + soft curriculum path bias (decaying)"
        },
        "preregistration": "PASS if P1/P2 MSE < min_baseline*0.95 AND entropy lower than P0",
        "results": results
    }, f, indent=2)
for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

print(f"\nSaved {run_dir}")
print(json.dumps({k: {kk: results[k][kk] for kk in ["mse","r2","entropy","params"]} for k in results}, indent=2))

# Outcome
base_mses = [results["HierResidual"]["mse"], results["DiffTopo_static"]["mse"], results["P0_light"]["mse"]]
best_base = min(base_mses)
best_prior = min(results["P1_entropy"]["mse"], results["P2_curriculum"]["mse"])
best_prior_name = "P1_entropy" if results["P1_entropy"]["mse"] <= results["P2_curriculum"]["mse"] else "P2_curriculum"
ent_p0 = results["P0_light"]["entropy"] or 99
ent_best = results[best_prior_name]["entropy"] or 99

if best_prior < best_base * 0.95 and ent_best < ent_p0 * 0.9:
    outcome = "PASS"
elif best_prior > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"Best prior ({best_prior_name}) MSE={best_prior:.4f} vs best_base={best_base:.4f}")
print(f"Entropy P0={ent_p0:.3f} → {best_prior_name}={ent_best:.3f}")

entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Hierarchical DiffTopo + prior P0/P1/P2
- P0: light (temp+topK) | P1: +entropy | P2: +curriculum soft bias
- Results MSE: HierRes={results['HierResidual']['mse']:.4f}, Static={results['DiffTopo_static']['mse']:.4f}, P0={results['P0_light']['mse']:.4f}, P1={results['P1_entropy']['mse']:.4f}, P2={results['P2_curriculum']['mse']:.4f}
- Entropy: P0={results['P0_light']['entropy']}, P1={results['P1_entropy']['entropy']}, P2={results['P2_curriculum']['entropy']}
- Outcome: **{outcome}**
- Notes: Tăng prior có kiểm soát trên DiffTopo hierarchical. Không full teacher forcing.
"""
with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(entry)
print("Lineage appended.")
