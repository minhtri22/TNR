#!/usr/bin/env python3
"""
Spiral SOLO — hiểu lớp xoáy ốc riêng, chưa gắn Causal/Full.

Giả thuyết người dùng:
  Xoay giúp phân loại / tách các "ý nghĩa" đang hút lẫn nhau (interference trong latent).

Thiết kế:
  - Baseline: Residual thuần
  - Spiral-only: 1 hoặc nhiều Spiral layer + head
  - Residual + Spiral inject (giống residual path + xoay)
  - Đo OOD MSE/R² + độ xoay trung bình |theta| + norm thay đổi

Task: multi-hop v2.5
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
        Y[i] = apply_path(X[i:i+1], path_dict[keys[cases[i].item()]]).squeeze(0)
    Y = Y + 0.015 * torch.randn_like(Y)
    return X, Y


class SpiralLayer(nn.Module):
    """
    Xoay theo cặp dims (Givens) + radial.
    Log |theta| trung bình để xem model có thực sự xoay không.
    """
    def __init__(self, dim):
        super().__init__()
        self.n_pairs = dim // 2
        self.theta_base = nn.Parameter(torch.zeros(self.n_pairs))
        self.theta_net = nn.Sequential(nn.Linear(dim, self.n_pairs), nn.Tanh())
        self.radial_net = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.last_theta_abs = 0.0
        self.last_delta_norm = 0.0

    def forward(self, h):
        B, D = h.shape
        n = self.n_pairs
        theta = 0.5 * self.theta_base + 0.5 * self.theta_net(h)
        self.last_theta_abs = theta.abs().mean().item()
        h_pairs = h[:, : 2 * n].view(B, n, 2)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x, y = h_pairs[:, :, 0], h_pairs[:, :, 1]
        x2 = cos_t * x - sin_t * y
        y2 = sin_t * x + cos_t * y
        rotated = torch.stack([x2, y2], dim=-1).view(B, 2 * n)
        if D > 2 * n:
            rotated = torch.cat([rotated, h[:, 2 * n :]], dim=-1)
        r = self.radial_net(h)
        out = h + 0.4 * rotated + 0.15 * (r * h) + 0.2 * self.mix(h)
        self.last_delta_norm = (out - h).norm(dim=-1).mean().item()
        return out


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


class SpiralOnly(nn.Module):
    """Chỉ xoáy (1 hoặc nhiều lớp) + head — không residual blocks."""
    def __init__(self, n_spiral=2):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.spirals = nn.ModuleList([SpiralLayer(HIDDEN) for _ in range(n_spiral)])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_theta = 0.0
        self.last_delta = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        thetas, deltas = [], []
        for s in self.spirals:
            h = s(h)
            thetas.append(s.last_theta_abs)
            deltas.append(s.last_delta_norm)
        self.last_theta = float(np.mean(thetas)) if thetas else 0.0
        self.last_delta = float(np.mean(deltas)) if deltas else 0.0
        return self.out(h)


class ResidualPlusSpiral(nn.Module):
    """Residual + chèn Spiral giữa / cuối (inject, không thay routing)."""
    def __init__(self, n_blocks=5, spiral_pos="mid"):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.spiral = SpiralLayer(HIDDEN)
        self.spiral_pos = spiral_pos
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_theta = 0.0
        self.last_delta = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        mid = len(self.blocks) // 2
        for i, b in enumerate(self.blocks):
            h = h + 0.4 * b(h)
            if self.spiral_pos == "mid" and i == mid:
                h = self.spiral(h)
                self.last_theta = self.spiral.last_theta_abs
                self.last_delta = self.spiral.last_delta_norm
        if self.spiral_pos == "end":
            h = self.spiral(h)
            self.last_theta = self.spiral.last_theta_abs
            self.last_delta = self.spiral.last_delta_norm
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
                val = F.mse_loss(model(Xv), Yv).item()
            if val < best_val:
                best_val = val
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
        theta = getattr(model, "last_theta", 0.0)
        delta = getattr(model, "last_delta", 0.0)
        return mse, r2, theta, delta

    configs = {
        "Res": ResidualBaseline(),
        "SpiralOnly_1": SpiralOnly(n_spiral=1),
        "SpiralOnly_2": SpiralOnly(n_spiral=2),
        "Res_Spiral_mid": ResidualPlusSpiral(spiral_pos="mid"),
        "Res_Spiral_end": ResidualPlusSpiral(spiral_pos="end"),
    }

    out = {"seed": seed}
    for name, model in configs.items():
        model = train(model)
        ood, r2, th, dn = evaluate(model, X_ood, Y_ood)
        out[f"{name}_OOD"] = ood
        out[f"{name}_R2"] = r2
        out[f"{name}_theta"] = th
        out[f"{name}_delta"] = dn
        out[f"params_{name}"] = count_params(model)
        print(f"  {name}: OOD={ood:.4f} R2={r2:.3f} |theta|={th:.3f} delta_norm={dn:.3f}")
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    all_results.append(run_seed(s))

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
names = ["Res", "SpiralOnly_1", "SpiralOnly_2", "Res_Spiral_mid", "Res_Spiral_end"]
for n in names:
    print(f"  {n:16s} OOD={ms(f'{n}_OOD')[0]:.4f}±{ms(f'{n}_OOD')[1]:.4f} R2={ms(f'{n}_R2')[0]:.3f} "
          f"|theta|={ms(f'{n}_theta')[0]:.3f} delta={ms(f'{n}_delta')[0]:.3f}")

res_ood = ms("Res_OOD")[0]
best_name = min([n for n in names if n != "Res"], key=lambda n: ms(f"{n}_OOD")[0])
best_ood = ms(f"{best_name}_OOD")[0]
best_theta = ms(f"{best_name}_theta")[0]

print(f"\nRes={res_ood:.4f} Best={best_name} {best_ood:.4f} |theta|={best_theta:.3f}")

if best_ood < res_ood * 0.95 and best_theta > 0.05:
    outcome = "PASS_solo"  # spiral helps and actually rotates
elif best_ood > res_ood * 1.05:
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
        "experiment": "Spiral SOLO — understand rotation without Causal",
        "hypothesis": "Spiral separates meanings that absorb each other",
        "seeds": SEEDS,
        "summary": {n: {"OOD": list(ms(f"{n}_OOD")), "theta": list(ms(f"{n}_theta")), "delta": list(ms(f"{n}_delta"))} for n in names},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Spiral solo** (chưa gắn Causal) — giả thuyết tách ý nghĩa hút lẫn nhau
- OOD: Res={res_ood:.4f}, Best={best_name}={best_ood:.4f}, |theta|={best_theta:.3f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
