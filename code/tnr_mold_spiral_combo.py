#!/usr/bin/env python3
"""
Combo solo: Nén khuôn hẹp + Spiral (xoay)
  So thứ tự: Mold→Spiral vs Spiral→Mold vs parallel gate
  Baseline: Res, MoldOnly, SpiralOnly
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
BOTTLE = 12
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


class MoldLayer(nn.Module):
    def __init__(self, dim, bottle):
        super().__init__()
        self.down = nn.Linear(dim, bottle)
        self.up = nn.Linear(bottle, dim)
        self.last_code_norm = 0.0

    def forward(self, h):
        code = torch.tanh(self.down(h))
        self.last_code_norm = code.norm(dim=-1).mean().item()
        return h + 0.5 * self.up(code)


class SpiralLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        n = dim // 2
        self.n_pairs = n
        self.theta_base = nn.Parameter(torch.zeros(n))
        self.theta_net = nn.Sequential(nn.Linear(dim, n), nn.Tanh())
        self.radial_net = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.last_theta_abs = 0.0

    def forward(self, h):
        B, D = h.shape
        n = self.n_pairs
        theta = 0.5 * self.theta_base + 0.5 * self.theta_net(h)
        self.last_theta_abs = theta.abs().mean().item()
        hp = h[:, : 2 * n].view(B, n, 2)
        c, s = torch.cos(theta), torch.sin(theta)
        x, y = hp[:, :, 0], hp[:, :, 1]
        rot = torch.stack([c * x - s * y, s * x + c * y], -1).view(B, 2 * n)
        if D > 2 * n:
            rot = torch.cat([rot, h[:, 2 * n :]], -1)
        r = self.radial_net(h)
        return h + 0.35 * rot + 0.15 * (r * h) + 0.2 * self.mix(h)


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


class MoldOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.m1 = MoldLayer(HIDDEN, BOTTLE)
        self.m2 = MoldLayer(HIDDEN, BOTTLE)
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_code = 0.0
        self.last_theta = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        h = self.m1(h)
        h = self.m2(h)
        self.last_code = (self.m1.last_code_norm + self.m2.last_code_norm) / 2
        return self.out(h)


class SpiralOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.s1 = SpiralLayer(HIDDEN)
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_code = 0.0
        self.last_theta = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        h = self.s1(h)
        self.last_theta = self.s1.last_theta_abs
        return self.out(h)


class MoldThenSpiral(nn.Module):
    """Nén trước → xoay sau."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.mold = MoldLayer(HIDDEN, BOTTLE)
        self.spiral = SpiralLayer(HIDDEN)
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_code = 0.0
        self.last_theta = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        h = self.mold(h)
        self.last_code = self.mold.last_code_norm
        h = self.spiral(h)
        self.last_theta = self.spiral.last_theta_abs
        return self.out(h)


class SpiralThenMold(nn.Module):
    """Xoay trước → nén sau."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.spiral = SpiralLayer(HIDDEN)
        self.mold = MoldLayer(HIDDEN, BOTTLE)
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_code = 0.0
        self.last_theta = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        h = self.spiral(h)
        self.last_theta = self.spiral.last_theta_abs
        h = self.mold(h)
        self.last_code = self.mold.last_code_norm
        return self.out(h)


class MoldSpiralParallel(nn.Module):
    """Song song mold + spiral, gate trộn."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.mold = MoldLayer(HIDDEN, BOTTLE)
        self.spiral = SpiralLayer(HIDDEN)
        self.gate = nn.Sequential(nn.Linear(HIDDEN, 2), nn.Softmax(dim=-1))
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_code = 0.0
        self.last_theta = 0.0
        self.last_gate_m = 0.5

    def forward(self, x):
        h = self.input_proj(x)
        hm = self.mold(h)
        hs = self.spiral(h)
        self.last_code = self.mold.last_code_norm
        self.last_theta = self.spiral.last_theta_abs
        g = self.gate(h)
        self.last_gate_m = g[:, 0].mean().item()
        mixed = g[:, 0:1] * hm + g[:, 1:2] * hs
        return self.out(h + 0.5 * mixed)


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
        code = getattr(model, "last_code", 0.0)
        theta = getattr(model, "last_theta", 0.0)
        return mse, r2, code, theta

    configs = {
        "Res": ResidualBaseline(),
        "MoldOnly": MoldOnly(),
        "SpiralOnly": SpiralOnly(),
        "Mold_then_Spiral": MoldThenSpiral(),
        "Spiral_then_Mold": SpiralThenMold(),
        "Parallel": MoldSpiralParallel(),
    }

    out = {"seed": seed}
    for name, model in configs.items():
        model = train(model)
        ood, r2, code, th = evaluate(model, X_ood, Y_ood)
        out[f"{name}_OOD"] = ood
        out[f"{name}_R2"] = r2
        out[f"{name}_code"] = code
        out[f"{name}_theta"] = th
        out[f"params_{name}"] = count_params(model)
        print(f"  {name}: OOD={ood:.4f} R2={r2:.3f} code={code:.2f} |θ|={th:.3f} p={count_params(model)}")
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    all_results.append(run_seed(s))

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
names = ["Res", "MoldOnly", "SpiralOnly", "Mold_then_Spiral", "Spiral_then_Mold", "Parallel"]
for n in names:
    print(f"  {n:18s} OOD={ms(f'{n}_OOD')[0]:.4f}±{ms(f'{n}_OOD')[1]:.4f} "
          f"|θ|={ms(f'{n}_theta')[0]:.3f} code={ms(f'{n}_code')[0]:.2f}")

res_ood = ms("Res_OOD")[0]
best_name = min(names, key=lambda n: ms(f"{n}_OOD")[0])
best_ood = ms(f"{best_name}_OOD")[0]
print(f"\nBest={best_name} {best_ood:.4f} vs Res {res_ood:.4f} ratio={best_ood/res_ood:.3f}")

if best_ood < res_ood * 0.95 and best_name != "Res":
    outcome = "PASS_solo"
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
        "experiment": "Mold + Spiral combo solo",
        "seeds": SEEDS,
        "summary": {n: list(ms(f"{n}_OOD")) for n in names},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Mold hẹp + Spiral combo solo** (Mold→S, S→Mold, Parallel)
- OOD: Res={res_ood:.4f}, Best={best_name}={best_ood:.4f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
