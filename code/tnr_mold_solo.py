#!/usr/bin/env python3
"""
Nén khuôn SOLO (mold compression / bottleneck)

Ý tưởng: ép state qua "khuôn" chiều thấp (bottleneck) rồi bung lại.
  - Ép thông tin qua nút thắt → chỉ giữ cấu trúc vừa khuôn
  - Khác winnow (tách nặng/nhẹ) và spiral (xoay)

Biến thể:
  - Mold fixed: Linear down → nonlinearity → Linear up
  - Mold residual: h + up(down(h))
  - Mold multi: nhiều bottleneck song song (nhiều khuôn) rồi gate trộn

So residual baseline trên multi-hop v2.5.
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
BOTTLENECK = 12          # khuôn hẹp
BOTTLENECK_WIDE = 24     # khuôn vừa
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
    """Nén khuôn: dim → bottleneck → dim."""
    def __init__(self, dim, bottle, residual=True):
        super().__init__()
        self.down = nn.Linear(dim, bottle)
        self.up = nn.Linear(bottle, dim)
        self.residual = residual
        self.last_compress_ratio = bottle / dim
        self.last_code_norm = 0.0

    def forward(self, h):
        code = torch.tanh(self.down(h))
        self.last_code_norm = code.norm(dim=-1).mean().item()
        recon = self.up(code)
        if self.residual:
            return h + 0.5 * recon
        return recon


class MultiMold(nn.Module):
    """Nhiều khuôn song song (bottle khác nhau) + gate trộn."""
    def __init__(self, dim, bottles=(8, 16, 24)):
        super().__init__()
        self.molds = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, b), nn.Tanh(), nn.Linear(b, dim))
            for b in bottles
        ])
        self.gate = nn.Sequential(nn.Linear(dim, len(bottles)), nn.Softmax(dim=-1))
        self.last_gate_entropy = 0.0

    def forward(self, h):
        outs = torch.stack([m(h) for m in self.molds], dim=1)  # B, M, D
        g = self.gate(h)  # B, M
        # entropy of gate (cao = dùng đều nhiều khuôn)
        self.last_gate_entropy = (-(g * (g + 1e-8).log()).sum(-1).mean()).item()
        mixed = (outs * g.unsqueeze(-1)).sum(1)
        return h + 0.5 * mixed


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
    def __init__(self, bottle=BOTTLENECK, n_mold=2, residual_mold=True):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.molds = nn.ModuleList([
            MoldLayer(HIDDEN, bottle, residual=residual_mold) for _ in range(n_mold)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_code_norm = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        norms = []
        for m in self.molds:
            h = m(h)
            norms.append(m.last_code_norm)
        self.last_code_norm = float(np.mean(norms)) if norms else 0.0
        return self.out(h)


class ResidualPlusMold(nn.Module):
    def __init__(self, n_blocks=5, bottle=BOTTLENECK, pos="mid", multi=False):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.multi = multi
        if multi:
            self.mold = MultiMold(HIDDEN)
        else:
            self.mold = MoldLayer(HIDDEN, bottle, residual=True)
        self.pos = pos
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_stat = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        mid = len(self.blocks) // 2
        for i, b in enumerate(self.blocks):
            h = h + 0.4 * b(h)
            if self.pos == "mid" and i == mid:
                h = self.mold(h)
                self.last_stat = getattr(self.mold, "last_code_norm",
                                         getattr(self.mold, "last_gate_entropy", 0.0))
        if self.pos == "end":
            h = self.mold(h)
            self.last_stat = getattr(self.mold, "last_code_norm",
                                     getattr(self.mold, "last_gate_entropy", 0.0))
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
        stat = getattr(model, "last_code_norm", getattr(model, "last_stat", 0.0))
        return mse, r2, stat

    configs = {
        "Res": ResidualBaseline(),
        "MoldOnly_narrow": MoldOnly(bottle=BOTTLENECK, n_mold=2),
        "MoldOnly_wide": MoldOnly(bottle=BOTTLENECK_WIDE, n_mold=2),
        "Res_Mold_mid": ResidualPlusMold(bottle=BOTTLENECK, pos="mid"),
        "Res_Mold_end": ResidualPlusMold(bottle=BOTTLENECK, pos="end"),
        "Res_MultiMold": ResidualPlusMold(pos="mid", multi=True),
    }

    out = {"seed": seed}
    for name, model in configs.items():
        model = train(model)
        ood, r2, st = evaluate(model, X_ood, Y_ood)
        out[f"{name}_OOD"] = ood
        out[f"{name}_R2"] = r2
        out[f"{name}_stat"] = st
        out[f"params_{name}"] = count_params(model)
        print(f"  {name}: OOD={ood:.4f} R2={r2:.3f} stat={st:.3f} params={count_params(model)}")
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    all_results.append(run_seed(s))

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
names = ["Res", "MoldOnly_narrow", "MoldOnly_wide", "Res_Mold_mid", "Res_Mold_end", "Res_MultiMold"]
for n in names:
    print(f"  {n:18s} OOD={ms(f'{n}_OOD')[0]:.4f}±{ms(f'{n}_OOD')[1]:.4f} R2={ms(f'{n}_R2')[0]:.3f} stat={ms(f'{n}_stat')[0]:.3f}")

res_ood = ms("Res_OOD")[0]
best_name = min([n for n in names if n != "Res"], key=lambda n: ms(f"{n}_OOD")[0])
best_ood = ms(f"{best_name}_OOD")[0]
print(f"\nRes={res_ood:.4f} Best={best_name} {best_ood:.4f} ratio={best_ood/res_ood:.3f}")

if best_ood < res_ood * 0.95:
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
        "experiment": "Mold compression (nen khuon) SOLO",
        "seeds": SEEDS,
        "summary": {n: {"OOD": list(ms(f"{n}_OOD")), "R2": list(ms(f"{n}_R2"))} for n in names},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Nén khuôn solo** (bottleneck mold) — chưa lắp TNR
- OOD: Res={res_ood:.4f}, Best={best_name}={best_ood:.4f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
