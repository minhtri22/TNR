#!/usr/bin/env python3
"""
TNR Phương án B — Task Multi-hop Routing-Critical v1

Task design (routing-critical):
  - Mỗi sample có latent path = chuỗi operator rời rạc (module ids).
  - Mỗi operator biến đổi feature theo quy tắc khác nhau (không chỉ residual cộng dồn mượt).
  - Intermediate state có semantic: sau mỗi hop, một "slot" được ghi.
  - Train: path length 3, families cố định.
  - OOD: path length 5 + permutation / composition chưa thấy.
  → Residual depth khó "đoán" đúng thứ tự operator; cần chọn đúng module theo thứ tự.

Backbone giữ từ lineage tốt nhất:
  - Pure KV_causal (3 lớp)
  - Spark-like CCFC (3 Causal + 1 Full)
  - HierResidual (baseline)

Preregistration:
  PASS nếu:
    mean OOD_MSE_best < mean OOD_HierRes * 0.95
    AND mean traj_recovery_OOD >= 0.40
    AND wins >= 2/3 seeds
  FAIL nếu mean OOD_best > mean OOD_HierRes * 1.05
  else UNRESOLVED
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
N_OPS = 6            # 6 operators distinct
HIDDEN = 40
INPUT_DIM = 12
OUTPUT_DIM = 4
SEQ_LEN = 4
KV_DIM = 20
BATCH_SIZE = 128
N_TRAIN = 2500
N_VAL = 600
N_OOD = 800
EPOCHS = 18
LR = 1e-3
TEMP_START, TEMP_END = 1.4, 0.35

# Train paths: length 3
TRAIN_PATHS = {
    0: [0, 2, 4],
    1: [1, 3, 5],
    2: [0, 3, 1],
    3: [2, 5, 4],
    4: [1, 0, 5],
}
# OOD: longer + new compositions
OOD_PATHS = {
    0: [0, 2, 4, 1, 3],       # extend
    1: [1, 3, 5, 0, 2],
    2: [0, 3, 1, 5, 4],       # reorder+extend
    3: [2, 5, 4, 0, 1],
    4: [5, 1, 0, 3, 2],       # new start + composition
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---- Distinct operators (not smooth residual) ----
def build_operators():
    torch.manual_seed(99)
    ops = []
    for i in range(N_OPS):
        # each op: select dims, nonlinear, scale, bias — different structure
        W = torch.zeros(INPUT_DIM, INPUT_DIM)
        # permutation-like + sparse transform
        perm = torch.randperm(INPUT_DIM)
        for r in range(INPUT_DIM):
            W[r, perm[r]] = 1.0 + 0.3 * ((i + r) % 3 - 1)
        # extra coupling unique to op
        a, b = i % INPUT_DIM, (i * 2 + 3) % INPUT_DIM
        W[a, b] += 0.8
        W[b, a] -= 0.4
        bias = torch.zeros(INPUT_DIM)
        bias[a] = 0.5 * (i % 4 - 1.5)
        gate_dims = [(i + k) % INPUT_DIM for k in range(3)]
        ops.append((W, bias, gate_dims))
    torch.manual_seed(42)
    return ops

OPERATORS = build_operators()

def apply_op(h, op_idx):
    W, bias, gate_dims = OPERATORS[op_idx % N_OPS]
    # hard-ish nonlinear path: not pure residual add
    z = torch.tanh(h @ W + bias)
    # write only to gated dims (slot semantic)
    out = h.clone()
    for d in gate_dims:
        out[:, d] = z[:, d]
    # also mild mix on all dims so path compounds
    out = out + 0.25 * z
    return out

def apply_path(x, path):
    h = x.clone()
    for op in path:
        h = apply_op(h, op)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    X = torch.randn(n, INPUT_DIM, generator=g) * 0.8
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, OUTPUT_DIM)
    paths_list = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        Y[i] = apply_path(X[i:i+1], p).squeeze(0)
        paths_list.append(p)
    Y = Y + 0.01 * torch.randn_like(Y)
    return X, Y, paths_list

# ===================== MODELS =====================

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


class FullLayer(nn.Module):
    def __init__(self, n_mod=MOD_PER_LAYER):
        super().__init__()
        self.n_mod = n_mod
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_mod)
        ])
        self.msg_proj = nn.Linear(HIDDEN, HIDDEN // 2)
        self.msg_combine = nn.Linear(HIDDEN // 2 * n_mod, HIDDEN)
        self.gate = nn.Sequential(nn.Linear(HIDDEN, n_mod), nn.Sigmoid())

    def forward(self, h, temperature=1.0, return_traj=False):
        outs = torch.stack([m(h) for m in self.mods], 1)
        msgs = self.msg_proj(outs).reshape(h.size(0), -1)
        mixed = self.msg_combine(msgs)
        g = self.gate(h)
        weighted = (outs * g.unsqueeze(-1)).sum(1)
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
    def __init__(self, pattern="CCC"):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        # Map modules to operators: use N_OPS modules if possible
        n_mod = min(MOD_PER_LAYER, N_OPS)
        layers = []
        topos = ["sparse", "ring", "complete", "ring"]
        ci = 0
        for p in pattern:
            if p == "C":
                layers.append(CausalKVLayer(n_mod=n_mod, topo=topos[ci % 4]))
                ci += 1
            else:
                layers.append(FullLayer(n_mod=n_mod))
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

def traj_seq_recovery(traj, true_paths, n_sample=300):
    """traj: B, n_layers, T, n_mod → flatten argmax sequence, LCS vs true path (op ids)."""
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    n_mod = traj.size(-1)
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        # map local module id to op space (mod n_mod, but true path uses 0..N_OPS-1)
        pred_seq = [c % N_OPS for c in chosen]
        scores.append(lcs_length(pred_seq, true_paths[b]) / max(len(true_paths[b]), 1))
    return float(np.mean(scores))


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, Y_train, paths_train = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    X_val, Y_val, paths_val = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    X_ood, Y_ood, paths_ood = generate_dataset(N_OOD, OOD_PATHS, seed+20)

    # Sanity: OOD should be harder if residual can't memorize path
    print(f"  Y_train var={Y_train.var():.4f} Y_ood var={Y_ood.var():.4f}")

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
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
        return mse, rec, r2

    out = {"seed": seed}

    hr = train(HierResidual(n_blocks=12))
    out["HR_OOD"], _, out["HR_OOD_r2"] = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    out["HR_ID"], _, _ = evaluate(hr, X=X_val, Y=Y_val, paths=paths_val)
    out["params_hr"] = count_params(HierResidual(n_blocks=12))

    for name, pattern in [("PureC", "CCC"), ("Spark", "CCFC")]:
        m = train(HybridStack(pattern), use_traj=True)
        ood, traj, r2 = evaluate(m, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
        idm, _, _ = evaluate(m, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
        out[f"{name}_OOD"] = ood
        out[f"{name}_traj"] = traj
        out[f"{name}_OOD_r2"] = r2
        out[f"{name}_ID"] = idm
        out[f"params_{name}"] = count_params(HybridStack(pattern))

    return out


print("Params preview:")
print(f"  HR={count_params(HierResidual(12))}")
print(f"  PureC={count_params(HybridStack('CCC'))}")
print(f"  Spark={count_params(HybridStack('CCFC'))}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR    OOD={r['HR_OOD']:.4f} R2={r['HR_OOD_r2']:.3f}")
    print(f"  PureC OOD={r['PureC_OOD']:.4f} traj={r['PureC_traj']:.3f} R2={r['PureC_OOD_r2']:.3f}")
    print(f"  Spark OOD={r['Spark_OOD']:.4f} traj={r['Spark_traj']:.3f} R2={r['Spark_OOD_r2']:.3f}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
for k in ["HR_OOD", "PureC_OOD", "PureC_traj", "Spark_OOD", "Spark_traj"]:
    mu, sd = ms(k)
    print(f"  {k}: {mu:.4f} ± {sd:.4f}")

hr_mu = ms("HR_OOD")[0]
best_name, best_mu = min([("PureC", ms("PureC_OOD")[0]), ("Spark", ms("Spark_OOD")[0])], key=lambda x: x[1])
best_traj = ms(f"{best_name}_traj")[0]
wins = sum(1 for r in all_results if r[f"{best_name}_OOD"] < r["HR_OOD"])

print(f"\nBest={best_name} OOD={best_mu:.4f} traj={best_traj:.3f} vs HR={hr_mu:.4f} wins={wins}/3")
print(f"Threshold 0.95*HR={0.95*hr_mu:.4f}")

if best_mu < hr_mu * 0.95 and best_traj >= 0.40 and wins >= 2:
    outcome = "PASS"
elif best_mu > hr_mu * 1.05:
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
        "experiment": "Phuong an B: Multi-hop routing-critical task v1",
        "task": "Distinct operators, path length 3 train / 5 OOD, slot-style writes",
        "preregistration": "PASS if OOD<0.95*HR AND traj>=0.40 AND wins>=2/3",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["HR_OOD","PureC_OOD","PureC_traj","Spark_OOD","Spark_traj"]},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phương án B** — Multi-hop routing-critical task v1
- Task: operator rời rạc (không residual mượt), train path len 3, OOD len 5 + composition mới
- Backbone: PureC (CCC), Spark (CCFC), HierResidual
- OOD: HR={hr_mu:.4f}, PureC={ms('PureC_OOD')[0]:.4f}, Spark={ms('Spark_OOD')[0]:.4f}
- Traj: PureC={ms('PureC_traj')[0]:.3f}, Spark={ms('Spark_traj')[0]:.3f}
- Best={best_name}, wins={wins}/3
- Outcome: **{outcome}**
- Notes: Preregister chặt (0.95× + traj≥0.40). Bắt đầu chuỗi PASS/FAIL để rút kiến trúc.
""")
print(f"Saved {run_dir}")
