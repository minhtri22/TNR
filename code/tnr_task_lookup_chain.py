#!/usr/bin/env python3
"""
Task MỚI: Lookup-Chain (routing-critical by construction)

Khác hẳn multi-hop residual-mix trước:
  - Input = [query_vec | key_bits]
  - Có bank K vectors cố định (keys) + K transforms (values)
  - Latent path = chuỗi index (i1, i2, ..., iL)
  - Mỗi bước: CHỌN đúng transform[index] áp vào state
  - Sai index → output lệch mạnh (hard selection, không soft residual)

Train: L=2, families cố định
OOD: L=3 hoặc L=4 + index sequence chưa thấy

Mục tiêu: residual depth khó "học hết" mọi composition → R² OOD ~0.25-0.55
Backbone: chỉ HR vs PureC
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

N_KEYS = 6              # bank size = number of selectable ops
STATE_DIM = 8
KEY_DIM = 6             # one-hot-ish hint (not full path)
INPUT_DIM = STATE_DIM + KEY_DIM   # 14
OUTPUT_DIM = 4
HIDDEN = 48
MOD_PER_LAYER = 4
SEQ_LEN = 4
KV_DIM = 24
BATCH_SIZE = 128
N_TRAIN = 3500
N_VAL = 600
N_OOD = 800
EPOCHS = 28
LR = 1e-3
TEMP_START, TEMP_END = 1.4, 0.35

# Train paths length 2
TRAIN_PATHS = {
    0: [0, 2], 1: [1, 3], 2: [2, 4],
    3: [3, 5], 4: [0, 4], 5: [1, 5],
    6: [5, 0], 7: [4, 1],
}
# OOD: length 3–4, new sequences
OOD_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [2, 4, 0],
    3: [3, 5, 1], 4: [0, 3, 1, 4], 5: [5, 2, 0, 3],
    6: [1, 4, 2], 7: [4, 0, 5, 2],
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def build_bank():
    """Fixed key embeddings + value transforms (not residual-friendly)."""
    torch.manual_seed(7)
    keys = F.normalize(torch.randn(N_KEYS, KEY_DIM), dim=-1)
    # Each transform: sparse signed permutation-like + bias
    transforms = []
    for i in range(N_KEYS):
        W = torch.zeros(STATE_DIM, STATE_DIM)
        perm = torch.randperm(STATE_DIM)
        for r in range(STATE_DIM):
            W[r, perm[r]] = 1.2 if (i + r) % 2 == 0 else -1.0
        # extra exclusive coupling
        a, b = i % STATE_DIM, (i * 3 + 1) % STATE_DIM
        W[a, b] += 1.5
        bvec = torch.zeros(STATE_DIM)
        bvec[a] = 0.8 * ((i % 3) - 1)
        transforms.append((W, bvec))
    torch.manual_seed(42)
    return keys, transforms

KEYS, TRANSFORMS = build_bank()

def apply_lookup_path(state, path):
    """Hard apply: only the selected transform each step."""
    h = state.clone()
    for idx in path:
        W, b = TRANSFORMS[idx % N_KEYS]
        # hard nonlinear — not a small residual
        h = torch.tanh(h @ W + b) * 1.1
    return h[:, :OUTPUT_DIM]

def make_input(state, path):
    """
    Input = [state | soft hint].
    Hint = average of key embeddings of path endpoints (NOT full path) —
    forces model to infer middle hops; residual can't just read the answer.
    """
    B = state.size(0)
    hint = torch.zeros(B, KEY_DIM)
    # only first and last index leaked weakly
    hint = hint + 0.5 * KEYS[path[0] % N_KEYS] + 0.5 * KEYS[path[-1] % N_KEYS]
    hint = hint + 0.05 * torch.randn_like(hint)
    return torch.cat([state, hint.expand(B, -1) if hint.dim()==1 else hint], dim=-1)

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    states = torch.randn(n, STATE_DIM, generator=g) * 0.8
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    X = torch.zeros(n, INPUT_DIM)
    Y = torch.zeros(n, OUTPUT_DIM)
    paths_list = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        paths_list.append(p)
        X[i] = make_input(states[i:i+1], p).squeeze(0)
        Y[i] = apply_lookup_path(states[i:i+1], p).squeeze(0)
    Y = Y + 0.02 * torch.randn_like(Y)
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

class HierResidual(nn.Module):
    def __init__(self, n_blocks=14):
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
        n_mod = min(MOD_PER_LAYER, N_KEYS)
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

def traj_seq_recovery(traj, true_paths, n_sample=300):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [c % N_KEYS for c in chosen]
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

    out = {"seed": seed}
    hr = train(HierResidual())
    out["HR_OOD"], _, out["HR_R2"] = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    out["HR_ID"], _, out["HR_ID_R2"] = evaluate(hr, X=X_val, Y=Y_val, paths=paths_val)

    pc = train(PureC(), use_traj=True)
    out["PC_OOD"], out["PC_traj"], out["PC_R2"] = evaluate(pc, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
    out["PC_ID"], _, out["PC_ID_R2"] = evaluate(pc, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
    out["win"] = out["PC_OOD"] < out["HR_OOD"]
    return out


print(f"Params HR={count_params(HierResidual())} PureC={count_params(PureC())}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR    OOD={r['HR_OOD']:.4f} R2={r['HR_R2']:.3f} | ID R2={r['HR_ID_R2']:.3f}")
    print(f"  PureC OOD={r['PC_OOD']:.4f} R2={r['PC_R2']:.3f} traj={r['PC_traj']:.3f} win={r['win']}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr_r2 = ms("HR_R2")[0]
hr_ood = ms("HR_OOD")[0]
pc_ood = ms("PC_OOD")[0]
pc_traj = ms("PC_traj")[0]
wins = sum(r["win"] for r in all_results)

print("\n===== SUMMARY =====")
print(f"  HR    OOD={hr_ood:.4f} R2={hr_r2:.3f}")
print(f"  PureC OOD={pc_ood:.4f} R2={ms('PC_R2')[0]:.3f} traj={pc_traj:.3f}")
print(f"  wins={wins}/3  band_ok={0.25 <= hr_r2 <= 0.55}")

# Preregister same as B
if 0.25 <= hr_r2 <= 0.55 and pc_ood < hr_ood * 0.95 and pc_traj >= 0.40 and wins >= 2:
    outcome = "PASS"
elif pc_ood > hr_ood * 1.05:
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
        "experiment": "NEW task Lookup-Chain — PureC vs HR only",
        "task": "Hard select transforms by latent path; weak endpoint hint only",
        "preregistration": "PASS if HR_R2 in [0.25,0.55] and PC_OOD<0.95*HR and traj>=0.40 and wins>=2/3",
        "seeds": SEEDS,
        "summary": {
            "HR_OOD": list(ms("HR_OOD")), "HR_R2": list(ms("HR_R2")),
            "PC_OOD": list(ms("PC_OOD")), "PC_traj": list(ms("PC_traj")), "PC_R2": list(ms("PC_R2")),
        },
        "per_seed": all_results,
        "outcome": outcome,
        "band_ok": 0.25 <= hr_r2 <= 0.55,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Task mới Lookup-Chain** (hard transform select, weak endpoint hint)
- Backbone: chỉ HR vs PureC
- HR OOD R²={hr_r2:.3f} (target band 0.25–0.55: {'YES' if 0.25<=hr_r2<=0.55 else 'NO'})
- OOD: HR={hr_ood:.4f}, PureC={pc_ood:.4f}, traj={pc_traj:.3f}, wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
