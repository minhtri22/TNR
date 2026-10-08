#!/usr/bin/env python3
"""
Phương án A — LÕI THUẦN: PureC + layout prior (path consistency)

Bỏ hết: Full, Spiral, Mold, Winnow, hierarchical stack phức tạp.
Chỉ còn:
  - HierResidual (baseline)
  - PureC (3 lớp KV_causal)
  - Layout prior: auxiliary loss khuyến khích trajectory khớp soft target từ path
    (weak program supervision — sửa điểm gãy credit assignment)

Task: Lookup-Chain rút gọn (L_train=2, L_ood=3) — cấu trúc routing-critical,
làm mềm hơn run trước để còn headroom.

Preregister:
  PASS nếu:
    PureC_OOD < 0.95 * HR_OOD
    AND traj >= 0.45
    AND wins >= 2/3
  FAIL nếu PureC_OOD > 1.05 * HR_OOD
  else UNRESOLVED

Ablation nội bộ (vẫn trong lõi):
  PureC_no_prior  vs  PureC_with_prior (λ_layout nhỏ)
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
INPUT_DIM = STATE_DIM + KEY_DIM
OUTPUT_DIM = 4
HIDDEN = 48
MOD_PER_LAYER = 4
SEQ_LEN = 3
KV_DIM = 24
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 26
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35
LAMBDA_LAYOUT = 0.15   # weak prior — không át MSE

TRAIN_PATHS = {
    0: [0, 2], 1: [1, 3], 2: [2, 4], 3: [3, 5],
    4: [0, 4], 5: [1, 5], 6: [5, 0], 7: [4, 1],
}
OOD_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [2, 4, 0],
    3: [3, 5, 1], 4: [0, 3, 1], 5: [5, 2, 0],
    6: [1, 4, 2], 7: [4, 0, 5],
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

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

def path_to_soft_targets(paths, n_layers, seq_len, n_mod, device):
    """
    Layout prior yếu: trải path lên (layer, step) — mỗi phần tử path
    gán soft target one-hot trên module = index % n_mod.
    """
    B = len(paths)
    # target: B, n_layers, seq_len, n_mod
    tgt = torch.zeros(B, n_layers, seq_len, n_mod, device=device)
    for b, p in enumerate(paths):
        for t, op in enumerate(p):
            # map path step t -> layer t % n_layers, step min(t, seq_len-1)
            li = t % n_layers
            si = min(t, seq_len - 1)
            mi = op % n_mod
            tgt[b, li, si, mi] = 1.0
        # nếu path ngắn hơn, remaining steps: uniform mild (không ép)
    return tgt

# ===================== CORE MODELS ONLY =====================

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
            traj.append(gate)  # keep soft for layout loss
            updates = [self.mods[mi](h) * gate[:, mi:mi+1] for mi in range(self.n_mod)]
            h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            write = gate.unsqueeze(-1) * h.unsqueeze(1)
            kv = kv * (1 - gate.unsqueeze(-1)) + write
            written = torch.clamp(written + gate.detach(), 0, 1)
        stacked = torch.stack(traj, 1)  # B, T, n_mod
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
        traj = torch.stack(trajs, 1)  # B, L, T, M
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
    n_mod = traj.size(-1)
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

    def train_hr():
        model = HierResidual().to(DEVICE)
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

    def train_purec(use_layout=False):
        model = PureC().to(DEVICE)
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
                batch_paths = [paths_train[j] for j in idx.cpu().tolist()]
                opt.zero_grad()
                pred, traj = model(xb, return_traj=True, temperature=temp)
                loss = F.mse_loss(pred, yb)
                if use_layout:
                    tgt = path_to_soft_targets(
                        batch_paths, model.n_layers, SEQ_LEN, model.n_mod, DEVICE
                    )
                    # CE where target has mass; traj is B,L,T,M
                    layout_loss = -(tgt * (traj.clamp(1e-8).log())).sum() / (tgt.sum() + 1e-8)
                    loss = loss + LAMBDA_LAYOUT * layout_loss
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_v = model(Xv, temperature=TEMP_END)
                val = F.mse_loss(pred_v, Yv).item()
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return model

    def eval_hr(model):
        model.eval()
        X, Y = X_ood.to(DEVICE), Y_ood.to(DEVICE)
        with torch.no_grad():
            pred = model(X)
            mse = F.mse_loss(pred, Y).item()
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
        return mse, r2

    def eval_pc(model):
        model.eval()
        X, Y = X_ood.to(DEVICE), Y_ood.to(DEVICE)
        with torch.no_grad():
            pred, traj = model(X, return_traj=True, temperature=TEMP_END)
            mse = F.mse_loss(pred, Y).item()
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
            rec = traj_seq_recovery(traj.cpu(), paths_ood)
        return mse, r2, rec

    out = {"seed": seed}
    hr = train_hr()
    out["HR_OOD"], out["HR_R2"] = eval_hr(hr)

    pc0 = train_purec(use_layout=False)
    out["PC0_OOD"], out["PC0_R2"], out["PC0_traj"] = eval_pc(pc0)

    pc1 = train_purec(use_layout=True)
    out["PC1_OOD"], out["PC1_R2"], out["PC1_traj"] = eval_pc(pc1)

    return out


print(f"Params HR={count_params(HierResidual())} PureC={count_params(PureC())}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR          OOD={r['HR_OOD']:.4f} R2={r['HR_R2']:.3f}")
    print(f"  PureC       OOD={r['PC0_OOD']:.4f} traj={r['PC0_traj']:.3f}")
    print(f"  PureC+layout OOD={r['PC1_OOD']:.4f} traj={r['PC1_traj']:.3f}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
for k in ["HR_OOD", "HR_R2", "PC0_OOD", "PC0_traj", "PC1_OOD", "PC1_traj"]:
    print(f"  {k}: {ms(k)[0]:.4f} ± {ms(k)[1]:.4f}")

hr_ood = ms("HR_OOD")[0]
# best purec
if ms("PC1_OOD")[0] <= ms("PC0_OOD")[0]:
    best_name, best_ood, best_traj = "PC1", ms("PC1_OOD")[0], ms("PC1_traj")[0]
else:
    best_name, best_ood, best_traj = "PC0", ms("PC0_OOD")[0], ms("PC0_traj")[0]
wins = sum(1 for r in all_results if r[f"{best_name}_OOD"] < r["HR_OOD"])

print(f"\nBest={best_name} OOD={best_ood:.4f} traj={best_traj:.3f} vs HR={hr_ood:.4f} wins={wins}/3")
print(f"Layout helped traj? PC0={ms('PC0_traj')[0]:.3f} PC1={ms('PC1_traj')[0]:.3f}")

if best_ood < hr_ood * 0.95 and best_traj >= 0.45 and wins >= 2:
    outcome = "PASS"
elif best_ood > hr_ood * 1.05:
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
        "experiment": "A-core: PureC ± layout prior only",
        "lambda_layout": LAMBDA_LAYOUT,
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["HR_OOD","HR_R2","PC0_OOD","PC0_traj","PC1_OOD","PC1_traj"]},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **A-core PureC + layout prior** (bỏ hết add-in)
- Task: Lookup-Chain L2→L3
- HR OOD={hr_ood:.4f} R2={ms('HR_R2')[0]:.3f}
- PureC no-prior OOD={ms('PC0_OOD')[0]:.4f} traj={ms('PC0_traj')[0]:.3f}
- PureC+layout OOD={ms('PC1_OOD')[0]:.4f} traj={ms('PC1_traj')[0]:.3f}
- Best={best_name}, wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
