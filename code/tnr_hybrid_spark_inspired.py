#!/usr/bin/env python3
"""
TNR Hybrid inspired by Spark-X2.5 Hybrid Attention
  Spark: 1 full-attention + 3 sliding-window attention
  TNR:   1 Full all-to-all  + 3 KV_causal (sliding causal)

Also compare:
  - 2:1  = Causal-Full-Causal (previous PASS-borderline)
  - 1:1  = Causal-Full-Causal-Full (4 layers, equal-ish)
  - HierResidual (param-matched as close as possible)

Preregistration (stricter, audit-informed):
  PASS if:
    mean OOD_best_hybrid < mean OOD_HierRes * 0.97
    AND wins >= 2/3 seeds
    AND param_ratio hybrid/HR <= 1.15
  else UNRESOLVED / FAIL
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

MOD_PER_LAYER = 4
HIDDEN = 36
INPUT_DIM = 16
OUTPUT_DIM = 4
SEQ_LEN = 4
KV_DIM = 18
BATCH_SIZE = 128
N_TRAIN = 2400
N_VAL = 500
N_OOD = 700
EPOCHS = 20
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.4

SHORT_PATHS = {0: [0, 2, 5], 1: [1, 4, 7], 2: [1, 5, 3]}
MID_PATHS   = {0: [0, 3, 6, 2], 1: [2, 6, 4, 0], 2: [0, 2, 5, 1]}
TRAIN_PATHS = {**SHORT_PATHS, **{k+10: v for k, v in MID_PATHS.items()}}
OOD_PATHS = {
    0: [0, 2, 5, 7, 3], 1: [1, 4, 7, 0, 6],
    2: [0, 3, 6, 2, 5, 1], 3: [2, 6, 4, 0, 7], 4: [1, 5, 3, 6, 2],
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def generate_true_transforms(n=12):
    torch.manual_seed(123)
    transforms = []
    for i in range(n):
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.12
        d1, d2 = i % INPUT_DIM, (i*3+1) % INPUT_DIM
        W[d1, d1] += 1.0
        W[d2, d2] += 0.85
        b = torch.zeros(INPUT_DIM)
        b[d1] = 0.25 * ((i % 5) - 2)
        transforms.append((W, b))
    return transforms

TRUE_TRANSFORMS = generate_true_transforms()

def apply_path(x, path):
    h = x.clone()
    for m_idx in path:
        W, b = TRUE_TRANSFORMS[m_idx % 12]
        h = h + 0.55 * torch.tanh(h @ W + b)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    X = torch.randn(n, INPUT_DIM, generator=g) * 0.7
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, OUTPUT_DIM)
    paths_list = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        Y[i] = apply_path(X[i:i+1], p).squeeze(0)
        paths_list.append(p)
    Y = Y + 0.012 * torch.randn_like(Y)
    return X, Y, paths_list

# ===================== BLOCKS =====================

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
    """Sliding-window style: sequential + causal KV (analog of sliding-window attention)."""
    def __init__(self, topo_mode="ring"):
        super().__init__()
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(MOD_PER_LAYER)
        ])
        self.attn = AdditiveAttention(HIDDEN)
        self.controller = nn.Sequential(nn.Linear(HIDDEN, 24), nn.Tanh(), nn.Linear(24, MOD_PER_LAYER))
        self.kv_init = nn.Parameter(torch.randn(MOD_PER_LAYER, HIDDEN) * 0.02)
        adj = torch.zeros(MOD_PER_LAYER, MOD_PER_LAYER)
        if topo_mode == "complete":
            adj = torch.ones(MOD_PER_LAYER, MOD_PER_LAYER)
        elif topo_mode == "ring":
            for i in range(MOD_PER_LAYER):
                adj[i, (i+1)%MOD_PER_LAYER] = adj[i, (i-1)%MOD_PER_LAYER] = adj[i, i] = 1
        else:
            rng = np.random.RandomState(42)
            for i in range(MOD_PER_LAYER):
                adj[i, i] = 1
                cands = [j for j in range(MOD_PER_LAYER) if j != i]
                adj[i, rng.choice(cands, 2, replace=False)] = 1
        self.register_buffer("topo_mask", (adj.sum(0) > 0).float())

    def forward(self, h, temperature=1.0, return_traj=False):
        B = h.size(0)
        kv = self.kv_init.unsqueeze(0).expand(B, -1, -1).clone()
        written = torch.zeros(B, MOD_PER_LAYER, device=h.device)
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
            updates = [self.mods[mi](h) * gate[:, mi:mi+1] for mi in range(MOD_PER_LAYER)]
            h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            write = gate.unsqueeze(-1) * h.unsqueeze(1)
            kv = kv * (1 - gate.unsqueeze(-1)) + write
            written = torch.clamp(written + gate.detach(), 0, 1)
        if return_traj:
            return h, torch.stack(traj, 1)
        return h


class FullAllToAllLayer(nn.Module):
    """Analog of full-attention: every module interacts with every module."""
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(MOD_PER_LAYER)
        ])
        self.msg_proj = nn.Linear(HIDDEN, HIDDEN // 2)
        self.msg_combine = nn.Linear(HIDDEN // 2 * MOD_PER_LAYER, HIDDEN)
        self.gate = nn.Sequential(nn.Linear(HIDDEN, MOD_PER_LAYER), nn.Sigmoid())

    def forward(self, h, temperature=1.0, return_traj=False):
        outs = [m(h) for m in self.mods]
        stacked = torch.stack(outs, dim=1)
        msgs = self.msg_proj(stacked)
        mixed = self.msg_combine(msgs.reshape(h.size(0), -1))
        g = self.gate(h)
        weighted = (stacked * g.unsqueeze(-1)).sum(1)
        h = h + 0.4 * torch.tanh(weighted + 0.3 * mixed)
        if return_traj:
            return h, g.unsqueeze(1).expand(-1, SEQ_LEN, -1)
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


class HybridStack(nn.Module):
    """
    pattern: list of 'C' (causal) or 'F' (full)
    Spark-inspired: C-C-F-C  (3 sliding + 1 full)
    2:1 style:      C-F-C
    1:1 style:      C-F-C-F
    """
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        layers = []
        topo_cycle = ["sparse", "ring", "complete", "ring"]
        ci = 0
        for p in pattern:
            if p == "C":
                layers.append(CausalKVLayer(topo_cycle[ci % len(topo_cycle)]))
                ci += 1
            else:
                layers.append(FullAllToAllLayer())
        self.layers = nn.ModuleList(layers)
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

def traj_seq_recovery(traj, true_paths, n_sample=250):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [(idx // SEQ_LEN) * MOD_PER_LAYER + c for idx, c in enumerate(chosen)]
        scores.append(lcs_length(pred_seq, true_paths[b]) / max(len(true_paths[b]), 1))
    return float(np.mean(scores))


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, Y_train, paths_train = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    X_val, Y_val, paths_val = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    X_ood, Y_ood, paths_ood = generate_dataset(N_OOD, OOD_PATHS, seed+20)

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
                xb, yb = Xtr[idx], Ytr[idx]
                opt.zero_grad()
                pred = model(xb, temperature=temp) if use_traj else model(xb)
                F.mse_loss(pred, yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_v = model(Xv, temperature=TEMP_END) if use_traj else model(Xv)
                val_loss = F.mse_loss(pred_v, Yv).item()
            if val_loss < best_val:
                best_val = val_loss
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
        return mse, rec

    out = {"seed": seed}

    # Residual — more blocks to approach param parity with hybrids
    hr = train(HierResidual(n_blocks=14))
    out["HR_OOD"], _ = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    out["HR_ID"], _ = evaluate(hr, X=X_val, Y=Y_val, paths=paths_val)
    out["params_hr"] = count_params(HierResidual(n_blocks=14))

    configs = {
        "Spark": "CCFC",   # 3 causal + 1 full (Spark-inspired)
        "R21": "CFC",      # 2:1 previous
        "R11": "CFCF",     # 1:1
    }
    for name, pattern in configs.items():
        m = train(HybridStack(pattern), use_traj=True)
        ood, traj = evaluate(m, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
        idm, _ = evaluate(m, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
        out[f"{name}_OOD"] = ood
        out[f"{name}_traj"] = traj
        out[f"{name}_ID"] = idm
        out[f"params_{name}"] = count_params(HybridStack(pattern))

    return out


# Param preview
print("Param preview:")
for name, pat in [("HR", None), ("Spark", "CCFC"), ("R21", "CFC"), ("R11", "CFCF")]:
    if name == "HR":
        print(f"  {name}: {count_params(HierResidual(n_blocks=14))}")
    else:
        print(f"  {name} ({pat}): {count_params(HybridStack(pat))}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR    OOD={r['HR_OOD']:.4f}")
    print(f"  Spark OOD={r['Spark_OOD']:.4f} traj={r['Spark_traj']:.3f}")
    print(f"  R21   OOD={r['R21_OOD']:.4f} traj={r['R21_traj']:.3f}")
    print(f"  R11   OOD={r['R11_OOD']:.4f} traj={r['R11_traj']:.3f}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
keys = ["HR_OOD", "Spark_OOD", "Spark_traj", "R21_OOD", "R21_traj", "R11_OOD", "R11_traj"]
for k in keys:
    mu, sd = ms(k)
    print(f"  {k}: {mu:.4f} ± {sd:.4f}")

hr_mu = ms("HR_OOD")[0]
best_name, best_mu = min(
    [("Spark", ms("Spark_OOD")[0]), ("R21", ms("R21_OOD")[0]), ("R11", ms("R11_OOD")[0])],
    key=lambda x: x[1]
)
ratio = all_results[0][f"params_{best_name}"] / all_results[0]["params_hr"]
wins = sum(1 for r in all_results if r[f"{best_name}_OOD"] < r["HR_OOD"])

print(f"\nBest hybrid: {best_name} OOD={best_mu:.4f} vs HR={hr_mu:.4f}")
print(f"Param ratio: {ratio:.2f}  Wins: {wins}/3")
print(f"Threshold 0.97*HR = {0.97*hr_mu:.4f}")

if best_mu < hr_mu * 0.97 and wins >= 2 and ratio <= 1.15:
    outcome = "PASS"
elif best_mu > hr_mu * 1.03:
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
        "experiment": "Spark-inspired Hybrid (1 full + 3 causal) vs 2:1 vs 1:1",
        "audit_note": "Stricter preregistration after PASS-borderline audit",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in keys},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
        "param_ratio": ratio,
        "wins": wins,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Spark-X2.5 inspired Hybrid (CCFC=3C+1F) vs R21(CFC) vs R11(CFCF)
- Audit prior PASS: razor-thin, param +36%, only 3 seeds → treated as fragile
- OOD mean: HR={hr_mu:.4f}, Spark={ms('Spark_OOD')[0]:.4f}, R21={ms('R21_OOD')[0]:.4f}, R11={ms('R11_OOD')[0]:.4f}
- Traj: Spark={ms('Spark_traj')[0]:.3f}, R21={ms('R21_traj')[0]:.3f}, R11={ms('R11_traj')[0]:.3f}
- Best={best_name}, ratio={ratio:.2f}, wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
