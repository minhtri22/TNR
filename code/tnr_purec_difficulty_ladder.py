#!/usr/bin/env python3
"""
Phương án B tiếp — PureC cố định, chỉ biến độ khó task.

Giả thuyết (preregister):
  H1: Khi task đủ routing-critical (residual R² OOD trong 0.25–0.50),
      PureC OOD < 0.95 * HR_OOD VÀ traj >= 0.40 → PASS
  H2: Khi task quá dễ (HR R² > 0.70) → PureC không vượt (UNRESOLVED/hòa)
  H3: Khi task quá khó (HR R² < 0.15) → cả hai kém, traj vẫn đo được (UNRESOLVED)

Backbone: chỉ HierResidual vs PureC (CCC). Không add-in.
Ba mức độ khó: HARD / MID / EASY (operator + path length + noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

SEEDS = [42, 7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MOD_PER_LAYER = 4
N_OPS = 6
HIDDEN = 40
INPUT_DIM = 12
OUTPUT_DIM = 4
SEQ_LEN = 4
KV_DIM = 20
BATCH_SIZE = 128
N_TRAIN = 2000
N_VAL = 500
N_OOD = 700
EPOCHS = 14
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

# Path sets shared; difficulty mainly from operators + noise + OOD length
TRAIN_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [0, 3, 1],
    3: [2, 5, 4], 4: [1, 0, 5], 5: [4, 2, 0],
}
OOD_PATHS_SHORT = {  # easier OOD
    0: [0, 2, 4], 1: [1, 3, 5], 2: [0, 3, 1],
    3: [2, 5, 4], 4: [1, 0, 5], 5: [4, 2, 0],
}
OOD_PATHS_MID = {
    0: [0, 2, 4, 1], 1: [1, 3, 5, 0], 2: [0, 3, 1, 5],
    3: [2, 5, 4, 0], 4: [5, 1, 0, 3], 5: [4, 2, 1, 5],
}
OOD_PATHS_LONG = {
    0: [0, 2, 4, 1, 3], 1: [1, 3, 5, 0, 2], 2: [0, 3, 1, 5, 2],
    3: [2, 5, 4, 0, 1], 4: [5, 1, 0, 3, 4], 5: [4, 2, 1, 5, 0],
}

DIFFICULTY = {
    "EASY": dict(op_scale=0.35, residual_mix=0.50, noise=0.008, ood_paths=OOD_PATHS_SHORT),
    "MID":  dict(op_scale=0.55, residual_mix=0.30, noise=0.015, ood_paths=OOD_PATHS_MID),
    "HARD": dict(op_scale=0.85, residual_mix=0.15, noise=0.025, ood_paths=OOD_PATHS_LONG),
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def build_operators(op_scale=0.55):
    torch.manual_seed(99)
    ops = []
    for i in range(N_OPS):
        W = torch.eye(INPUT_DIM) * 0.15
        perm = torch.randperm(INPUT_DIM)
        for r in range(INPUT_DIM):
            W[r, perm[r]] += op_scale + 0.2 * ((i + r) % 3)
        a, b = i % INPUT_DIM, (i * 2 + 3) % INPUT_DIM
        W[a, b] += 0.4 * op_scale / 0.55
        W[b, a] -= 0.2 * op_scale / 0.55
        bias = torch.zeros(INPUT_DIM)
        bias[a] = 0.35 * (i % 4 - 1.5)
        gate_dims = [(i + k) % INPUT_DIM for k in range(3)]
        ops.append((W, bias, gate_dims))
    torch.manual_seed(42)
    return ops

def make_apply_path(ops, residual_mix):
    def apply_op(h, op_idx):
        W, bias, gate_dims = ops[op_idx % N_OPS]
        z = torch.tanh(h @ W + bias)
        out = h + residual_mix * z
        for d in gate_dims:
            out[:, d] = (1 - residual_mix - 0.2) * out[:, d] + (residual_mix + 0.2) * z[:, d]
        return out
    def apply_path(x, path):
        h = x.clone()
        for op in path:
            h = apply_op(h, op)
        return h[:, :OUTPUT_DIM]
    return apply_path

def generate_dataset(n, path_dict, apply_path, noise, seed_offset=0):
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
    Y = Y + noise * torch.randn_like(Y)
    return X, Y, paths_list

# ===================== MODELS: HR + PureC only =====================

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, KV_DIM)
        self.w_k = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(KV_DIM, 1)
        self.out = nn.Linear(dim, dim)
    def forward(self, query, keys, values, mask=None):
        q = self.w_q(query).unsqueeze(1)
        k = self.w_k(keys)
        score = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return self.out(ctx), attn

class CausalKVLayer(nn.Module):
    def __init__(self, n_mod=MOD_PER_LAYER, topo="ring"):
        super().__init__()
        self.n_mod = n_mod
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_mod)
        ])
        self.attn = AdditiveAttention(HIDDEN)
        self.controller = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, n_mod))
        self.kv_init = nn.Parameter(torch.randn(n_mod, HIDDEN) * 0.02)
        adj = torch.ones(n_mod, n_mod) if topo == "complete" else torch.zeros(n_mod, n_mod)
        if topo == "ring":
            for i in range(n_mod):
                adj[i, i] = adj[i, (i+1)%n_mod] = adj[i, (i-1)%n_mod] = 1
        elif topo == "sparse":
            rng = np.random.RandomState(42)
            for i in range(n_mod):
                adj[i, i] = 1
                cands = [j for j in range(n_mod) if j != i]
                adj[i, rng.choice(cands, min(2, len(cands)), replace=False)] = 1
        self.register_buffer("topo_mask", (adj.sum(0) > 0).float())

    def forward(self, h, temperature=1.0, return_traj=False):
        B = h.size(0)
        kv = self.kv_init.unsqueeze(0).expand(B, -1, -1).clone()
        written = torch.zeros(B, self.n_mod, device=h.device)
        traj = []
        for t in range(SEQ_LEN):
            attn_mask = written.clone() if t > 0 else torch.ones_like(written) * 0.15
            ctx, _ = self.attn(h, kv, kv, mask=attn_mask)
            logits = self.controller(ctx)
            logits = logits.masked_fill(self.topo_mask.unsqueeze(0) == 0, -1e9)
            soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            traj.append(gate.detach())
            updates = [self.mods[mi](h) * gate[:, mi:mi+1] for mi in range(self.n_mod)]
            h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            write = gate.unsqueeze(-1) * h.unsqueeze(1)
            kv = kv * (1 - gate.unsqueeze(-1)) + write
            written = torch.clamp(written + gate.detach(), 0, 1)
        if return_traj:
            return h, torch.stack(traj, 1)
        return h

class HierResidual(nn.Module):
    def __init__(self, n_blocks=12):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class PureC(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        n_mod = min(MOD_PER_LAYER, N_OPS)
        self.layers = nn.ModuleList([
            CausalKVLayer(n_mod=n_mod, topo="sparse"),
            CausalKVLayer(n_mod=n_mod, topo="ring"),
            CausalKVLayer(n_mod=n_mod, topo="complete"),
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)

    def forward(self, x, return_traj=False, temperature=1.0):
        h = self.input_proj(x)
        trajs = []
        for layer in self.layers:
            if return_traj:
                h, t = layer(h, temperature=temperature, return_traj=True)
                trajs.append(t)
            else:
                h = layer(h, temperature=temperature)
        out = self.out(h)
        if return_traj:
            return out, torch.stack(trajs, 1)
        return out

def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def traj_seq_recovery(traj, true_paths, n_sample=280):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [c % N_OPS for c in chosen]
        scores.append(lcs_length(pred_seq, true_paths[b]) / max(len(true_paths[b]), 1))
    return float(np.mean(scores))


def run_difficulty(diff_name, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ops = build_operators(cfg["op_scale"])
    apply_path = make_apply_path(ops, cfg["residual_mix"])
    X_train, Y_train, paths_train = generate_dataset(N_TRAIN, TRAIN_PATHS, apply_path, cfg["noise"], seed)
    X_val, Y_val, paths_val = generate_dataset(N_VAL, TRAIN_PATHS, apply_path, cfg["noise"], seed+10)
    X_ood, Y_ood, paths_ood = generate_dataset(N_OOD, cfg["ood_paths"], apply_path, cfg["noise"], seed+20)

    def train(model, use_traj=False):
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
        Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
        best_val, best_state = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred = model(Xtr[idx], temperature=temp) if use_traj else model(Xtr[idx])
                F.mse_loss(pred, Ytr[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_v = model(Xv, temperature=TEMP_END) if use_traj else model(Xv)
                val = F.mse_loss(pred_v, Yv).item()
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return model

    def evaluate(model, use_traj=False, X=None, Y=None, paths=None):
        model.eval()
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        with torch.no_grad():
            if use_traj:
                pred, traj = model(X, return_traj=True, temperature=TEMP_END)
                rec = traj_seq_recovery(traj.cpu(), paths)
            else:
                pred = model(X)
                rec = None
            mse = F.mse_loss(pred, Y).item()
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
        return mse, rec, r2

    hr = train(HierResidual(12))
    hr_ood, _, hr_r2 = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    pure = train(PureC(), use_traj=True)
    pc_ood, pc_traj, pc_r2 = evaluate(pure, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)

    return {
        "diff": diff_name, "seed": seed,
        "HR_OOD": hr_ood, "HR_R2": hr_r2,
        "PureC_OOD": pc_ood, "PureC_traj": pc_traj, "PureC_R2": pc_r2,
        "win": pc_ood < hr_ood,
    }


# ---- Run ladder ----
all_rows = []
print(f"Params HR={count_params(HierResidual(12))} PureC={count_params(PureC())}")

for diff_name, cfg in DIFFICULTY.items():
    print(f"\n########## DIFFICULTY: {diff_name} ##########")
    rows = []
    for s in SEEDS:
        r = run_difficulty(diff_name, cfg, s)
        rows.append(r)
        all_rows.append(r)
        print(f"  seed {s}: HR OOD={r['HR_OOD']:.4f} R2={r['HR_R2']:.3f} | "
              f"PureC OOD={r['PureC_OOD']:.4f} traj={r['PureC_traj']:.3f} win={r['win']}")

    hr_ood = np.mean([r["HR_OOD"] for r in rows])
    hr_r2 = np.mean([r["HR_R2"] for r in rows])
    pc_ood = np.mean([r["PureC_OOD"] for r in rows])
    pc_traj = np.mean([r["PureC_traj"] for r in rows])
    wins = sum(r["win"] for r in rows)

    # Outcome per difficulty (preregistered)
    if 0.25 <= hr_r2 <= 0.55 and pc_ood < hr_ood * 0.95 and pc_traj >= 0.40 and wins >= 2:
        outcome = "PASS"
    elif pc_ood > hr_ood * 1.05:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    print(f"  >> {diff_name}: HR_R2={hr_r2:.3f} HR_OOD={hr_ood:.4f} PureC_OOD={pc_ood:.4f} "
          f"traj={pc_traj:.3f} wins={wins}/3 → **{outcome}**")

# Summary table
print("\n===== LADDER SUMMARY =====")
for diff_name in DIFFICULTY:
    rows = [r for r in all_rows if r["diff"] == diff_name]
    hr_r2 = np.mean([r["HR_R2"] for r in rows])
    hr_ood = np.mean([r["HR_OOD"] for r in rows])
    pc_ood = np.mean([r["PureC_OOD"] for r in rows])
    pc_traj = np.mean([r["PureC_traj"] for r in rows])
    wins = sum(r["win"] for r in rows)
    if 0.25 <= hr_r2 <= 0.55 and pc_ood < hr_ood * 0.95 and pc_traj >= 0.40 and wins >= 2:
        outcome = "PASS"
    elif pc_ood > hr_ood * 1.05:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"
    print(f"  {diff_name:5s}  HR_R2={hr_r2:.3f}  OOD HR={hr_ood:.4f} PureC={pc_ood:.4f}  "
          f"traj={pc_traj:.3f}  wins={wins}/3  → {outcome}")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "PureC difficulty ladder (B) — no add-ins",
        "preregistration": {
            "PASS": "HR_R2 in [0.25,0.55] AND PureC_OOD < 0.95*HR AND traj>=0.40 AND wins>=2/3",
            "FAIL": "PureC_OOD > 1.05*HR",
            "else": "UNRESOLVED",
        },
        "seeds": SEEDS,
        "per_seed": all_rows,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **PureC difficulty ladder (B)** — chỉ HR vs PureC, 3 mức EASY/MID/HARD
- Không add-in. Preregister PASS: HR_R2∈[0.25,0.55] ∧ OOD<0.95×HR ∧ traj≥0.40 ∧ wins≥2/3
- Results in metrics.json / console summary
""")
print(f"Saved {run_dir}")
