#!/usr/bin/env python3
"""
Curriculum refined: giữ OOD gain, bảo vệ traj.

Giả thuyết:
  Phase2 thay hẳn L3 → traj tụt (quên L2 routing).
  Phase2 MIX L2+L3 + λ_layout hơi tăng → giữ traj, vẫn cải OOD.

So sánh:
  BASE: L2-only full (PASS core protocol)
  CURR_replace: L2 then L3-only (đã UNRESOLVED — OOD tốt, traj kém)
  CURR_mix: L2 then mix(L2,L3) + λ_layout=0.20 phase2

Preregister PASS:
  CURR_mix_OOD < 0.95 * BASE_OOD
  AND CURR_mix_traj >= 0.45
  AND wins >= 2/3 vs BASE
FAIL nếu CURR_mix_OOD > 1.05 * BASE
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

N_KEYS = 6
STATE_DIM = 8
KEY_DIM = 6
INPUT_DIM = 14
OUTPUT_DIM = 4
HIDDEN = 48
MOD_PER_LAYER = 4
SEQ_LEN = 3
KV_DIM = 24
BATCH_SIZE = 128
N_TRAIN = 3000
N_VAL = 400
N_OOD = 600
EPOCHS_BASE = 24
EPOCHS_P1 = 14
EPOCHS_P2 = 12
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35
LAMBDA_P1 = 0.15
LAMBDA_P2 = 0.20  # slightly stronger layout in phase2

PATHS_L2 = {
    0: [0, 2], 1: [1, 3], 2: [2, 4], 3: [3, 5],
    4: [0, 4], 5: [1, 5], 6: [5, 0], 7: [4, 1],
}
PATHS_L3_TRAIN = {
    0: [0, 2, 1], 1: [1, 3, 0], 2: [2, 4, 3], 3: [3, 5, 2],
    4: [0, 4, 5], 5: [1, 5, 4], 6: [5, 0, 3], 7: [4, 1, 2],
}
OOD_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [2, 4, 0],
    3: [3, 5, 1], 4: [0, 3, 1], 5: [5, 2, 0],
    6: [1, 4, 2], 7: [4, 0, 5],
}

def build_bank():
    torch.manual_seed(7)
    keys = F.normalize(torch.randn(N_KEYS, KEY_DIM), dim=-1)
    transforms = []
    for i in range(N_KEYS):
        W = torch.zeros(STATE_DIM, STATE_DIM)
        perm = torch.randperm(STATE_DIM)
        for r in range(STATE_DIM):
            W[r, perm[r]] = 1.1 if (i + r) % 2 == 0 else -0.9
        a, b = i % STATE_DIM, (i * 3 + 1) % STATE_DIM
        W[a, b] += 1.2
        bvec = torch.zeros(STATE_DIM)
        bvec[a] = 0.6 * ((i % 3) - 1)
        transforms.append((W, bvec))
    torch.manual_seed(42)
    return keys, transforms

KEYS, TRANSFORMS = build_bank()

def apply_lookup_path(state, path):
    h = state.clone()
    for idx in path:
        W, b = TRANSFORMS[idx % N_KEYS]
        h = torch.tanh(h @ W + b) * 1.05
    return h[:, :OUTPUT_DIM]

def make_input(state, path):
    B = state.size(0)
    hint = 0.5 * KEYS[path[0] % N_KEYS] + 0.5 * KEYS[path[-1] % N_KEYS]
    hint = hint.unsqueeze(0).expand(B, -1) + 0.04 * torch.randn(B, KEY_DIM)
    return torch.cat([state, hint], dim=-1)

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
    Y = Y + 0.015 * torch.randn_like(Y)
    return X, Y, paths_list

def mix_datasets(X1, Y1, p1, X2, Y2, p2, ratio_l3=0.5):
    """Concatenate with ratio of L3 samples."""
    n3 = int(X2.size(0) * ratio_l3)
    n2 = X1.size(0)
    # take all L2 + portion L3
    X = torch.cat([X1, X2[:n3]], 0)
    Y = torch.cat([Y1, Y2[:n3]], 0)
    paths = p1 + p2[:n3]
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
        self.n_layers = 3
        self.n_mod = n_mod

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

def traj_seq_recovery(traj, true_paths, n_sample=280):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [c % N_KEYS for c in chosen]
        scores.append(lcs_length(pred_seq, true_paths[b]) / max(len(true_paths[b]), 1))
    return float(np.mean(scores))


def train_purec(model, X_train, Y_train, paths_train, X_val, Y_val, epochs, lr=LR, lam=LAMBDA_P1):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
    Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
    n = Xtr.size(0)
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(epochs - 1, 1))
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            batch_paths = [paths_train[j] for j in idx.cpu().tolist()]
            opt.zero_grad()
            pred, traj = model(xb, return_traj=True, temperature=temp)
            loss = F.mse_loss(pred, yb)
            tgt = path_to_soft_targets(batch_paths, model.n_layers, SEQ_LEN, model.n_mod, DEVICE)
            layout_loss = -(tgt * (traj.clamp(1e-8).log())).sum() / (tgt.sum() + 1e-8)
            loss = loss + lam * layout_loss
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(Xv, temperature=TEMP_END), Yv).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model

def eval_pc(model, X_ood, Y_ood, paths_ood):
    model.eval()
    X, Y = X_ood.to(DEVICE), Y_ood.to(DEVICE)
    with torch.no_grad():
        pred, traj = model(X, return_traj=True, temperature=TEMP_END)
        mse = F.mse_loss(pred, Y).item()
        rec = traj_seq_recovery(traj.cpu(), paths_ood)
    return mse, rec


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr2, Y_tr2, p_tr2 = generate_dataset(N_TRAIN, PATHS_L2, seed)
    X_va2, Y_va2, _ = generate_dataset(N_VAL, PATHS_L2, seed+10)
    X_tr3, Y_tr3, p_tr3 = generate_dataset(N_TRAIN, PATHS_L3_TRAIN, seed+5)
    X_va3, Y_va3, p_va3 = generate_dataset(N_VAL, PATHS_L3_TRAIN, seed+15)
    X_ood, Y_ood, p_ood = generate_dataset(N_OOD, OOD_PATHS, seed+20)

    # BASE
    base = train_purec(PureC(), X_tr2, Y_tr2, p_tr2, X_va2, Y_va2, EPOCHS_BASE, lam=LAMBDA_P1)
    base_ood, base_traj = eval_pc(base, X_ood, Y_ood, p_ood)

    # CURR_replace (L3 only phase2) — reference
    rep = train_purec(PureC(), X_tr2, Y_tr2, p_tr2, X_va2, Y_va2, EPOCHS_P1, lam=LAMBDA_P1)
    rep = train_purec(rep, X_tr3, Y_tr3, p_tr3, X_va3, Y_va3, EPOCHS_P2, lr=LR*0.4, lam=LAMBDA_P1)
    rep_ood, rep_traj = eval_pc(rep, X_ood, Y_ood, p_ood)

    # CURR_mix: phase2 = mix L2+L3, λ slightly higher
    X_mix, Y_mix, p_mix = mix_datasets(X_tr2, Y_tr2, p_tr2, X_tr3, Y_tr3, p_tr3, ratio_l3=0.5)
    X_vmix, Y_vmix, p_vmix = mix_datasets(X_va2, Y_va2, [[] for _ in range(len(Y_va2))], X_va3, Y_va3, p_va3, ratio_l3=0.5)
    # fix val paths for mix - regenerate simple
    X_vmix = torch.cat([X_va2, X_va3[:len(X_va3)//2]], 0)
    Y_vmix = torch.cat([Y_va2, Y_va3[:len(Y_va3)//2]], 0)
    p_vmix = [PATHS_L2[k] for k in list(PATHS_L2.keys())] * 50  # placeholder length fix below
    # simpler val: use L2 val for early stop stability
    mix = train_purec(PureC(), X_tr2, Y_tr2, p_tr2, X_va2, Y_va2, EPOCHS_P1, lam=LAMBDA_P1)
    mix = train_purec(mix, X_mix, Y_mix, p_mix, X_va2, Y_va2, EPOCHS_P2, lr=LR*0.4, lam=LAMBDA_P2)
    mix_ood, mix_traj = eval_pc(mix, X_ood, Y_ood, p_ood)

    return {
        "seed": seed,
        "BASE_OOD": base_ood, "BASE_traj": base_traj,
        "REP_OOD": rep_ood, "REP_traj": rep_traj,
        "MIX_OOD": mix_ood, "MIX_traj": mix_traj,
        "win_mix": mix_ood < base_ood,
    }


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  BASE OOD={r['BASE_OOD']:.4f} traj={r['BASE_traj']:.3f}", flush=True)
    print(f"  REP  OOD={r['REP_OOD']:.4f} traj={r['REP_traj']:.3f}", flush=True)
    print(f"  MIX  OOD={r['MIX_OOD']:.4f} traj={r['MIX_traj']:.3f} win={r['win_mix']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
for k in ["BASE_OOD", "BASE_traj", "REP_OOD", "REP_traj", "MIX_OOD", "MIX_traj"]:
    print(f"  {k}: {ms(k)[0]:.4f} ± {ms(k)[1]:.4f}")

base_ood, mix_ood = ms("BASE_OOD")[0], ms("MIX_OOD")[0]
base_t, mix_t = ms("BASE_traj")[0], ms("MIX_traj")[0]
wins = sum(r["win_mix"] for r in all_results)

print(f"\nMIX/BASE ratio={mix_ood/base_ood:.3f} wins={wins}/3")

if mix_ood < base_ood * 0.95 and mix_t >= 0.45 and wins >= 2:
    outcome = "PASS"
elif mix_ood > base_ood * 1.05:
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
        "experiment": "Curriculum MIX L2+L3 phase2 + stronger layout",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["BASE_OOD","BASE_traj","REP_OOD","REP_traj","MIX_OOD","MIX_traj"]},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Curriculum MIX** phase2 (L2+L3) + λ=0.20 vs BASE / REP
- BASE OOD={base_ood:.4f} traj={base_t:.3f}
- REP  OOD={ms('REP_OOD')[0]:.4f} traj={ms('REP_traj')[0]:.3f}
- MIX  OOD={mix_ood:.4f} traj={mix_t:.3f} wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
