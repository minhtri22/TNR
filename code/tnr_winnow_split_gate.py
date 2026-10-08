#!/usr/bin/env python3
"""
Winnow Split+Gate (solo): không loại trấu.
  - Tách state thành nhánh nặng + nhánh nhẹ
  - Xử lý tách (optional mild)
  - Gate học được kết hợp có chọn lọc: h = g*heavy + (1-g)*light

So với Residual/MLP baseline trên multi-hop v2.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

SEEDS = [42, 7, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_OPS = 6
HIDDEN = 48
INPUT_DIM = 12
OUTPUT_DIM = 4
BATCH_SIZE = 128
N_TRAIN = 2800
N_VAL = 500
N_OOD = 700
EPOCHS = 25
LR = 1e-3
TOPK_RATIO = 0.5

TRAIN_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [0, 3, 1],
    3: [2, 5, 4], 4: [1, 0, 5], 5: [4, 2, 0],
}
OOD_PATHS = {
    0: [0, 2, 4, 1], 1: [1, 3, 5, 0], 2: [0, 3, 1, 5, 2],
    3: [2, 5, 4, 0], 4: [5, 1, 0, 3], 5: [4, 2, 1, 5, 0],
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def build_operators():
    torch.manual_seed(99)
    ops = []
    for i in range(N_OPS):
        W = torch.eye(INPUT_DIM) * 0.15
        perm = torch.randperm(INPUT_DIM)
        for r in range(INPUT_DIM):
            W[r, perm[r]] += 0.7 + 0.2 * ((i + r) % 3)
        a, b = i % INPUT_DIM, (i * 2 + 3) % INPUT_DIM
        W[a, b] += 0.55
        W[b, a] -= 0.25
        bias = torch.zeros(INPUT_DIM)
        bias[a] = 0.35 * (i % 4 - 1.5)
        gate_dims = [(i + k) % INPUT_DIM for k in range(3)]
        ops.append((W, bias, gate_dims))
    torch.manual_seed(42)
    return ops

OPERATORS = build_operators()

def apply_op(h, op_idx):
    W, bias, gate_dims = OPERATORS[op_idx % N_OPS]
    z = torch.tanh(h @ W + bias)
    out = h + 0.30 * z
    for d in gate_dims:
        out[:, d] = 0.35 * out[:, d] + 0.65 * z[:, d]
    return out

def apply_path(x, path):
    h = x.clone()
    for op in path:
        h = apply_op(h, op)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    X = torch.randn(n, INPUT_DIM, generator=g) * 0.75
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, OUTPUT_DIM)
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        Y[i] = apply_path(X[i:i+1], p).squeeze(0)
    Y = Y + 0.015 * torch.randn_like(Y)
    return X, Y


class SplitByMagnitude(nn.Module):
    """Tách heavy/light theo |h| top-k — không zero light, chỉ mask tách."""
    def __init__(self, dim, ratio=TOPK_RATIO):
        super().__init__()
        self.k = max(1, int(dim * ratio))

    def forward(self, h):
        score = h.abs()
        topk = torch.topk(score, self.k, dim=-1)
        heavy_mask = torch.zeros_like(h)
        heavy_mask.scatter_(-1, topk.indices, 1.0)
        light_mask = 1.0 - heavy_mask
        heavy = h * heavy_mask
        light = h * light_mask
        return heavy, light, heavy_mask.mean().item()


class SplitLearned(nn.Module):
    """Tách soft: mỗi dim có alpha in [0,1] — heavy=alpha*h, light=(1-alpha)*h."""
    def __init__(self, dim):
        super().__init__()
        self.split_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h):
        alpha = torch.sigmoid(self.split_net(h))  # 1 = heavy
        heavy = alpha * h
        light = (1 - alpha) * h
        return heavy, light, alpha.mean().item()


class GatedRecombine(nn.Module):
    """Gate kết hợp: out = g*heavy_proc + (1-g)*light_proc (+ optional residual h)."""
    def __init__(self, dim, use_residual=True):
        super().__init__()
        self.heavy_proc = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.light_proc = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.gate_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.use_residual = use_residual
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, h, heavy, light):
        hp = self.heavy_proc(heavy)
        lp = self.light_proc(light)
        g = torch.sigmoid(self.gate_net(torch.cat([hp, lp], dim=-1)))
        mixed = g * hp + (1 - g) * lp
        if self.use_residual:
            mixed = mixed + 0.3 * h
        return self.out_proj(mixed), g.mean().item()


class ResidualWithSplitGate(nn.Module):
    def __init__(self, split="mag", n_blocks=5):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        if split == "mag":
            self.splitter = SplitByMagnitude(HIDDEN)
        else:
            self.splitter = SplitLearned(HIDDEN)
        self.recombine = GatedRecombine(HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_heavy_frac = 0.5
        self.last_gate = 0.5

    def forward(self, x):
        h = self.input_proj(x)
        heavy, light, frac = self.splitter(h)
        self.last_heavy_frac = frac
        h2, gmean = self.recombine(h, heavy, light)
        self.last_gate = gmean
        h = h + 0.5 * h2  # inject split-gate path
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)


class ResidualBaseline(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)

    def forward(self, x):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, Y_train = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    X_val, Y_val = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    X_ood, Y_ood = generate_dataset(N_OOD, OOD_PATHS, seed+20)

    def train(model):
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
        Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
        best_val, best_state = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(Xtr[idx]), Ytr[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(model(Xv), Yv).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return model

    def evaluate(model, X, Y):
        model.eval()
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        with torch.no_grad():
            pred = model(X)
            mse = F.mse_loss(pred, Y).item()
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
        heavy_frac = getattr(model, "last_heavy_frac", 0.0)
        gate = getattr(model, "last_gate", 0.0)
        return mse, r2, heavy_frac, gate

    out = {"seed": seed}
    for name, model in [
        ("Res", ResidualBaseline()),
        ("SplitMag_Gate", ResidualWithSplitGate(split="mag")),
        ("SplitLearn_Gate", ResidualWithSplitGate(split="learn")),
    ]:
        model = train(model)
        ood_mse, ood_r2, hf, g = evaluate(model, X_ood, Y_ood)
        id_mse, id_r2, _, _ = evaluate(model, X_val, Y_val)
        out[f"{name}_OOD"] = ood_mse
        out[f"{name}_OOD_r2"] = ood_r2
        out[f"{name}_ID"] = id_mse
        out[f"{name}_heavy_frac"] = hf
        out[f"{name}_gate"] = g
        out[f"params_{name}"] = count_params(model)
        print(f"  {name}: OOD={ood_mse:.4f} R2={ood_r2:.3f} heavy_frac={hf:.2f} gate={g:.2f}")
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    all_results.append(run_seed(s))

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
for n in ["Res", "SplitMag_Gate", "SplitLearn_Gate"]:
    print(f"  {n}: OOD={ms(f'{n}_OOD')[0]:.4f}±{ms(f'{n}_OOD')[1]:.4f} R2={ms(f'{n}_OOD_r2')[0]:.3f} "
          f"heavy={ms(f'{n}_heavy_frac')[0]:.2f} gate={ms(f'{n}_gate')[0]:.2f}")

res_ood = ms("Res_OOD")[0]
mag_ood = ms("SplitMag_Gate_OOD")[0]
learn_ood = ms("SplitLearn_Gate_OOD")[0]
best = min([("Mag", mag_ood), ("Learn", learn_ood)], key=lambda x: x[1])
print(f"\nRes={res_ood:.4f} Mag={mag_ood:.4f} Learn={learn_ood:.4f}")

if best[1] < res_ood * 0.95:
    outcome = "PASS_solo"
elif best[1] > res_ood * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Winnow Split+Gate (keep both heavy and light, gated recombine)",
        "seeds": SEEDS,
        "summary": {
            "Res_OOD": list(ms("Res_OOD")),
            "Mag_OOD": list(ms("SplitMag_Gate_OOD")),
            "Learn_OOD": list(ms("SplitLearn_Gate_OOD")),
        },
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Split heavy/light + gated recombine** (không loại trấu) — solo
- OOD: Res={res_ood:.4f}, MagGate={mag_ood:.4f}, LearnGate={learn_ood:.4f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
