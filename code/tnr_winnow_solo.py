#!/usr/bin/env python3
"""
TNR: Thử Winnow (sàng thóc) RIÊNG — chưa lắp vào Causal/Spiral/Full.

Mục tiêu: hiểu winnow làm gì trên multi-hop v2.5
  - Baseline: Linear / MLP / Residual (không winnow)
  - Winnow-Mag: giữ top-k dims theo |h|
  - Winnow-Learned: soft gate học được theo dim
  - Winnow rồi mới Residual nhỏ

Preregister (solo, không claim routing):
  - Đo OOD MSE + R²
  - Winnow "có nghĩa" nếu OOD tốt hơn baseline cùng capacity vừa phải
    VÀ sparsity (tỷ lệ dim bị hạ) > 0.2
  - Nếu OOD xấu hơn baseline > 5% → FAIL (sàng làm mất tín hiệu)
  - else UNRESOLVED / quan sát
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
TOPK_RATIO = 0.5  # giữ 50% dims "nặng"

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
    paths_list = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        Y[i] = apply_path(X[i:i+1], p).squeeze(0)
        paths_list.append(p)
    Y = Y + 0.015 * torch.randn_like(Y)
    return X, Y, paths_list

# ===================== WINNOW MODULES (solo) =====================

class WinnowMag(nn.Module):
    """Sàng theo |activation|: giữ top-k dims, còn lại nhân 0 (hard) hoặc nhỏ."""
    def __init__(self, dim, ratio=TOPK_RATIO, hard=True):
        super().__init__()
        self.dim = dim
        self.k = max(1, int(dim * ratio))
        self.hard = hard

    def forward(self, h):
        # h: B, D
        score = h.abs()
        topk = torch.topk(score, self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk.indices, 1.0)
        if self.hard:
            out = h * mask
        else:
            out = h * (0.1 + 0.9 * mask)
        sparsity = 1.0 - mask.mean().item()
        return out, sparsity


class WinnowLearned(nn.Module):
    """Sàng học được: gate per-dim từ chính h (và optional global)."""
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h):
        logits = self.gate_net(h)
        gate = torch.sigmoid(logits)
        out = h * gate
        sparsity = (gate < 0.3).float().mean().item()  # fraction "bay đi"
        return out, sparsity


# ===================== MODELS =====================

class MLP(nn.Module):
    def __init__(self, winnow=None):
        super().__init__()
        self.winnow = winnow
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, OUTPUT_DIM),
        )
        self.last_sparsity = 0.0

    def forward(self, x):
        h = x
        if self.winnow is not None:
            h, sp = self.winnow(h)
            self.last_sparsity = sp
        return self.net(h)


class ResidualNet(nn.Module):
    def __init__(self, n_blocks=6, winnow=None, winnow_pos="input"):
        super().__init__()
        self.winnow = winnow
        self.winnow_pos = winnow_pos  # "input" | "mid"
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.last_sparsity = 0.0

    def forward(self, x):
        h = x
        if self.winnow is not None and self.winnow_pos == "input":
            h, sp = self.winnow(h)
            self.last_sparsity = sp
        h = self.input_proj(h)
        for i, b in enumerate(self.blocks):
            h = h + 0.4 * b(h)
            if self.winnow is not None and self.winnow_pos == "mid" and i == len(self.blocks)//2:
                h, sp = self.winnow(h)
                self.last_sparsity = sp
        return self.out(h)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, Y_train, _ = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    X_val, Y_val, _ = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    X_ood, Y_ood, _ = generate_dataset(N_OOD, OOD_PATHS, seed+20)

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
                pred = model(Xtr[idx])
                F.mse_loss(pred, Ytr[idx]).backward()
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
        sparsities = []
        with torch.no_grad():
            # multiple batches for sparsity avg
            for i in range(0, X.size(0), BATCH_SIZE):
                _ = model(X[i:i+BATCH_SIZE])
                if hasattr(model, "last_sparsity"):
                    sparsities.append(model.last_sparsity)
            pred = model(X)
            mse = F.mse_loss(pred, Y).item()
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
        sp = float(np.mean(sparsities)) if sparsities else 0.0
        return mse, r2, sp

    configs = {
        "MLP": MLP(winnow=None),
        "MLP_WinnowMag": MLP(winnow=WinnowMag(INPUT_DIM, hard=True)),
        "MLP_WinnowLearn": MLP(winnow=WinnowLearned(INPUT_DIM)),
        "Res": ResidualNet(winnow=None),
        "Res_WinnowMag_in": ResidualNet(winnow=WinnowMag(INPUT_DIM, hard=True), winnow_pos="input"),
        "Res_WinnowLearn_in": ResidualNet(winnow=WinnowLearned(INPUT_DIM), winnow_pos="input"),
        "Res_WinnowMag_mid": ResidualNet(winnow=WinnowMag(HIDDEN, hard=True), winnow_pos="mid"),
        "Res_WinnowLearn_mid": ResidualNet(winnow=WinnowLearned(HIDDEN), winnow_pos="mid"),
    }

    out = {"seed": seed}
    for name, model in configs.items():
        model = train(model)
        ood_mse, ood_r2, sp = evaluate(model, X_ood, Y_ood)
        id_mse, id_r2, _ = evaluate(model, X_val, Y_val)
        out[f"{name}_OOD"] = ood_mse
        out[f"{name}_OOD_r2"] = ood_r2
        out[f"{name}_ID"] = id_mse
        out[f"{name}_sp"] = sp
        out[f"params_{name}"] = count_params(model)
        print(f"  {name}: OOD={ood_mse:.4f} R2={ood_r2:.3f} sp={sp:.2f}")
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    all_results.append(run_seed(s))

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY (mean ± std) =====")
names = ["MLP", "MLP_WinnowMag", "MLP_WinnowLearn", "Res", "Res_WinnowMag_in",
         "Res_WinnowLearn_in", "Res_WinnowMag_mid", "Res_WinnowLearn_mid"]
for n in names:
    ood, ood_sd = ms(f"{n}_OOD")
    r2, _ = ms(f"{n}_OOD_r2")
    sp, _ = ms(f"{n}_sp")
    print(f"  {n:22s} OOD={ood:.4f}±{ood_sd:.4f}  R2={r2:.3f}  sparsity={sp:.2f}")

# Compare winnow vs no-winnow residual
res_ood = ms("Res_OOD")[0]
best_w_name = min(
    ["Res_WinnowMag_in", "Res_WinnowLearn_in", "Res_WinnowMag_mid", "Res_WinnowLearn_mid"],
    key=lambda n: ms(f"{n}_OOD")[0]
)
best_w_ood = ms(f"{best_w_name}_OOD")[0]
best_w_sp = ms(f"{best_w_name}_sp")[0]

print(f"\nRes OOD={res_ood:.4f}  BestWinnow={best_w_name} OOD={best_w_ood:.4f} sp={best_w_sp:.2f}")
print(f"Relative: {best_w_ood/res_ood:.4f}")

if best_w_ood < res_ood * 0.95 and best_w_sp > 0.2:
    outcome = "PASS_solo"  # winnow helps alone
elif best_w_ood > res_ood * 1.05:
    outcome = "FAIL"  # winnow hurts
else:
    outcome = "UNRESOLVED"

print(f"=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
summary = {n: {"OOD": list(ms(f"{n}_OOD")), "R2": list(ms(f"{n}_OOD_r2")), "sp": list(ms(f"{n}_sp"))} for n in names}
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Winnow SOLO (sàng thóc) — chưa lắp TNR",
        "seeds": SEEDS,
        "summary": summary,
        "per_seed": all_results,
        "outcome": outcome,
        "best_winnow": best_w_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Winnow solo** (sàng thóc) trên multi-hop v2.5 — chưa lắp Causal/Spiral
- Res OOD={res_ood:.4f}; Best winnow={best_w_name} OOD={best_w_ood:.4f} sparsity={best_w_sp:.2f}
- Outcome: **{outcome}**
- Notes: Mag=topk |h|; Learned=sigmoid gate. input vs mid position.
""")
print(f"Saved {run_dir}")
