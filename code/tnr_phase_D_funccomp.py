#!/usr/bin/env python3
"""
Phase D — Function Composition (cùng class Lookup-Chain)

Task: compose atomic linear/tanh transforms by path of function ids.
Train short paths L=2; OOD longer L=3 and novel path templates.
Same architecture: PureC n_mod=6 λ=0 vs HierResidual.

Preregister:
  PASS: mean PC_OOD < 0.95 * HR_OOD AND traj >= 0.40 AND wins >= 2/3
  FAIL: PC_OOD > 1.05 * HR_OOD
  else UNRESOLVED

traj threshold 0.40 slightly softer than A (new task) but still meaningful.
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

N_FUNCS = 6
STATE_DIM = 8
HINT_DIM = 6
INPUT_DIM = STATE_DIM + HINT_DIM  # 14
OUTPUT_DIM = 4
HIDDEN = 48
N_MOD = 6
SEQ_LEN = 3
KV_DIM = 24
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 26
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

# Train paths L=2
TRAIN_PATHS = {
    0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4],
    4: [4, 5], 5: [5, 0], 6: [0, 3], 7: [2, 5],
}
# OOD: L=3 novel chains
OOD_PATHS = {
    0: [0, 1, 3], 1: [1, 2, 4], 2: [2, 3, 5],
    3: [3, 4, 0], 4: [4, 5, 1], 5: [5, 0, 2],
    6: [0, 2, 4], 7: [1, 3, 5],
}

def build_functions():
    """Atomic functions: distinct affine + nonlinearity."""
    torch.manual_seed(11)
    funcs = []
    for i in range(N_FUNCS):
        W = torch.randn(STATE_DIM, STATE_DIM) * 0.4
        # make each function more selective
        W = W + torch.eye(STATE_DIM) * (0.5 + 0.1 * (i % 3))
        b = torch.randn(STATE_DIM) * 0.3
        # rotate subspace
        angle = i * np.pi / N_FUNCS
        R = torch.eye(STATE_DIM)
        R[0, 0] = R[1, 1] = np.cos(angle)
        R[0, 1] = -np.sin(angle)
        R[1, 0] = np.sin(angle)
        W = R @ W
        funcs.append((W, b))
    torch.manual_seed(42)
    return funcs

FUNCS = build_functions()
FUNC_KEYS = F.normalize(torch.randn(N_FUNCS, HINT_DIM), dim=-1)

def apply_path(state, path):
    h = state.clone()
    for idx in path:
        W, b = FUNCS[idx % N_FUNCS]
        h = torch.tanh(h @ W.T + b)
    return h[:, :OUTPUT_DIM]

def make_input(state, path):
    B = state.size(0)
    # endpoint hint (weak)
    hint = 0.5 * FUNC_KEYS[path[0] % N_FUNCS] + 0.5 * FUNC_KEYS[path[-1] % N_FUNCS]
    hint = hint.unsqueeze(0).expand(B, -1) + 0.05 * torch.randn(B, HINT_DIM)
    return torch.cat([state, hint], dim=-1)

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    states = torch.randn(n, STATE_DIM, generator=g) * 0.9
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    X = torch.zeros(n, INPUT_DIM)
    Y = torch.zeros(n, OUTPUT_DIM)
    paths = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        paths.append(p)
        X[i] = make_input(states[i:i+1], p).squeeze(0)
        Y[i] = apply_path(states[i:i+1], p).squeeze(0)
    Y = Y + 0.01 * torch.randn_like(Y)
    return X, Y, paths

def path_to_soft_targets(paths, n_layers, seq_len, n_mod, device):
    B = len(paths)
    tgt = torch.zeros(B, n_layers, seq_len, n_mod, device=device)
    for b, p in enumerate(paths):
        for t, op in enumerate(p):
            tgt[b, t % n_layers, min(t, seq_len - 1), op % n_mod] = 1.0
    return tgt

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
    def __init__(self, n_mod=N_MOD, topo="ring"):
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
            traj.append(gate)
            updates = [self.mods[mi](h) * gate[:, mi:mi+1] for mi in range(self.n_mod)]
            h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            write = gate.unsqueeze(-1) * h.unsqueeze(1)
            kv = kv * (1 - gate.unsqueeze(-1)) + write
            written = torch.clamp(written + gate.detach(), 0, 1)
        stacked = torch.stack(traj, 1)
        if return_traj:
            return h, stacked
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
        self.layers = nn.ModuleList([
            CausalKVLayer(N_MOD, "sparse"),
            CausalKVLayer(N_MOD, "ring"),
            CausalKVLayer(N_MOD, "complete"),
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.n_layers = 3
        self.n_mod = N_MOD
    def forward(self, x, return_traj=False, temperature=1.0):
        h = self.input_proj(x)
        trajs = []
        for layer in self.layers:
            h, t = layer(h, temperature=temperature, return_traj=True)
            trajs.append(t)
        out = self.out(h)
        traj = torch.stack(trajs, 1)
        if return_traj:
            return out, traj
        return out

def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def traj_recovery(traj, true_paths, n_sample=280):
    B = min(n_sample, traj.size(0))
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred = [c % N_FUNCS for c in chosen]
        scores.append(lcs_length(pred, true_paths[b]) / max(len(true_paths[b]), 1))
    return float(np.mean(scores))


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, ptr = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    Xva, Yva, _ = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    Xood, Yood, pood = generate_dataset(N_OOD, OOD_PATHS, seed+20)

    def train_hr():
        model = HierResidual().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv, Yv = Xva.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(X[idx]), Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(Xv), Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_pc():
        model = PureC().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv, Yv = Xva.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred = model(X[idx], temperature=temp)
                F.mse_loss(pred, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(Xv, temperature=TEMP_END), Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    hr = train_hr()
    hr.eval()
    with torch.no_grad():
        out["HR_OOD"] = F.mse_loss(hr(Xood.to(DEVICE)), Yood.to(DEVICE)).item()

    pc = train_pc()
    pc.eval()
    with torch.no_grad():
        pred, traj = pc(Xood.to(DEVICE), return_traj=True, temperature=TEMP_END)
        out["PC_OOD"] = F.mse_loss(pred, Yood.to(DEVICE)).item()
        out["PC_traj"] = traj_recovery(traj.cpu(), pood)
    out["win"] = out["PC_OOD"] < out["HR_OOD"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR_OOD']:.4f}  PC={r['PC_OOD']:.4f} traj={r['PC_traj']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, pc = ms("HR_OOD")[0], ms("PC_OOD")[0]
tr = ms("PC_traj")[0]
wins = sum(r["win"] for r in all_results)

print("\n===== PHASE D SUMMARY =====")
print(f"HR OOD={hr:.4f}±{ms('HR_OOD')[1]:.4f}")
print(f"PC OOD={pc:.4f}±{ms('PC_OOD')[1]:.4f} traj={tr:.3f} wins={wins}/3")
print(f"ratio={pc/hr:.3f}")

if pc < hr * 0.95 and tr >= 0.40 and wins >= 2:
    outcome = "PASS"
elif pc > hr * 1.05:
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
        "experiment": "Phase D Function Composition",
        "seeds": SEEDS,
        "summary": {"HR": hr, "PC": [pc, tr], "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase D Function Composition** PureC n_mod=6 λ=0 vs HR
- HR OOD={hr:.4f} | PC OOD={pc:.4f} traj={tr:.3f} wins={wins}/3
- Outcome: **{outcome}**
""")

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write("""
## Lab close package 2026-10-09
- Report: reports/bao_cao_dong_goi_lab_phase_A_C.md
- Claim A FREEZE: PureC Lookup-Chain PASS
- Claim C BOUNDARY: Mini-SCAN no transfer
- Phase D started: Function Composition same class as A
""")
print(f"Saved {run_dir}")
