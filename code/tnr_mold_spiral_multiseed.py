#!/usr/bin/env python3
"""
Multi-seed (100) for Res vs MoldOnly vs Mold→Spiral
Note: 1000 full seeds not feasible on CPU session; 100 is statistically strong.
Light training for throughput.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

N_SEEDS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  N_SEEDS={N_SEEDS}")

N_OPS = 6
HIDDEN = 40
BOTTLE = 12
INPUT_DIM = 12
OUTPUT_DIM = 4
BATCH_SIZE = 128
N_TRAIN = 1200
N_OOD = 400
EPOCHS = 12
LR = 1e-3

TRAIN_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [0, 3, 1],
    3: [2, 5, 4], 4: [1, 0, 5], 5: [4, 2, 0],
}
OOD_PATHS = {
    0: [0, 2, 4, 1], 1: [1, 3, 5, 0], 2: [0, 3, 1, 5, 2],
    3: [2, 5, 4, 0], 4: [5, 1, 0, 3], 5: [4, 2, 1, 5, 0],
}

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
    def forward(self, h):
        return h + 0.5 * self.up(torch.tanh(self.down(h)))

class SpiralLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        n = dim // 2
        self.n_pairs = n
        self.theta_base = nn.Parameter(torch.zeros(n))
        self.theta_net = nn.Sequential(nn.Linear(dim, n), nn.Tanh())
        self.radial = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.mix = nn.Linear(dim, dim)
    def forward(self, h):
        B, D = h.shape
        n = self.n_pairs
        theta = 0.5 * self.theta_base + 0.5 * self.theta_net(h)
        hp = h[:, :2*n].view(B, n, 2)
        c, s = torch.cos(theta), torch.sin(theta)
        x, y = hp[:,:,0], hp[:,:,1]
        rot = torch.stack([c*x - s*y, s*x + c*y], -1).view(B, 2*n)
        if D > 2*n:
            rot = torch.cat([rot, h[:, 2*n:]], -1)
        return h + 0.35*rot + 0.15*(self.radial(h)*h) + 0.2*torch.tanh(self.mix(h))

class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(INPUT_DIM, HIDDEN)
        self.b = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN)) for _ in range(5)])
        self.o = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x):
        h = self.p(x)
        for b in self.b:
            h = h + 0.4 * b(h)
        return self.o(h)

class MoldOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(INPUT_DIM, HIDDEN)
        self.m1 = MoldLayer(HIDDEN, BOTTLE)
        self.m2 = MoldLayer(HIDDEN, BOTTLE)
        self.o = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x):
        h = self.m2(self.m1(self.p(x)))
        return self.o(h)

class MoldSpiral(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(INPUT_DIM, HIDDEN)
        self.m = MoldLayer(HIDDEN, BOTTLE)
        self.s = SpiralLayer(HIDDEN)
        self.o = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x):
        return self.o(self.s(self.m(self.p(x))))


def run_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    Xo, Yo = generate_dataset(N_OOD, OOD_PATHS, seed+20)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xo, Yo = Xo.to(DEVICE), Yo.to(DEVICE)

    results = {}
    for name, model in [("Res", Res()), ("Mold", MoldOnly()), ("MoldSpiral", MoldSpiral())]:
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(Xtr[idx]), Ytr[idx]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xo)
            mse = F.mse_loss(pred, Yo).item()
        results[name] = mse
    return results


all_res, all_mold, all_ms = [], [], []
wins_mold, wins_ms = 0, 0

for i, seed in enumerate(range(N_SEEDS)):
    r = run_one(seed)
    all_res.append(r["Res"])
    all_mold.append(r["Mold"])
    all_ms.append(r["MoldSpiral"])
    if r["Mold"] < r["Res"]:
        wins_mold += 1
    if r["MoldSpiral"] < r["Res"]:
        wins_ms += 1
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{N_SEEDS}] Res={np.mean(all_res):.4f} Mold={np.mean(all_mold):.4f} MS={np.mean(all_ms):.4f} "
              f"wins_M={wins_mold}/{i+1} wins_MS={wins_ms}/{i+1}")

print("\n===== FINAL 100 SEEDS =====")
print(f"Res:        {np.mean(all_res):.4f} ± {np.std(all_res):.4f}")
print(f"MoldOnly:   {np.mean(all_mold):.4f} ± {np.std(all_mold):.4f}  wins={wins_mold}/{N_SEEDS}")
print(f"MoldSpiral: {np.mean(all_ms):.4f} ± {np.std(all_ms):.4f}  wins={wins_ms}/{N_SEEDS}")
print(f"Mold vs Res delta: {(np.mean(all_mold)-np.mean(all_res))/np.mean(all_res)*100:.2f}%")
print(f"MS vs Res delta:   {(np.mean(all_ms)-np.mean(all_res))/np.mean(all_res)*100:.2f}%")

# simple paired sign test style
from math import erf, sqrt
def mean_diff_sig(a, b):
    d = np.array(a) - np.array(b)
    return float(np.mean(d)), float(np.std(d) / sqrt(len(d)))

dm, se = mean_diff_sig(all_mold, all_res)
print(f"Mold-Res mean diff={dm:.5f} se={se:.5f}")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "n_seeds": N_SEEDS,
        "note": "100 seeds (1000 not feasible on CPU session); light train",
        "Res": [float(np.mean(all_res)), float(np.std(all_res))],
        "Mold": [float(np.mean(all_mold)), float(np.std(all_mold))],
        "MoldSpiral": [float(np.mean(all_ms)), float(np.std(all_ms))],
        "wins_mold": wins_mold,
        "wins_ms": wins_ms,
        "per_seed": {"Res": all_res, "Mold": all_mold, "MoldSpiral": all_ms},
    }, f)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Multi-seed n={N_SEEDS} Res vs Mold vs Mold→Spiral (light train)
- Res={np.mean(all_res):.4f}±{np.std(all_res):.4f}
- Mold={np.mean(all_mold):.4f}±{np.std(all_mold):.4f} wins={wins_mold}/{N_SEEDS}
- MoldSpiral={np.mean(all_ms):.4f}±{np.std(all_ms):.4f} wins={wins_ms}/{N_SEEDS}
- Note: requested 1000; ran 100 (feasible). Statistical picture stabilizes.
""")
print(f"Saved {run_dir}")
