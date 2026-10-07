#!/usr/bin/env python3
"""
TNR: Xen kẽ KV_causal với 1 lớp Full all-to-all

Cấu hình:
  Layer 0: KV_causal (sparse topology + sliding causal KV)
  Layer 1: Full all-to-all (mọi module nối mọi module, parallel update)
  Layer 2: KV_causal (complete topology + sliding causal KV)

So sánh với:
  - HierResidual
  - KV_causal thuần (3 lớp causal, baseline tốt nhất trước đó)
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

N_LAYERS = 3
MOD_PER_LAYER = 4
HIDDEN = 36
INPUT_DIM = 16
OUTPUT_DIM = 4
SEQ_LEN = 5
KV_DIM = 20
BATCH_SIZE = 128
N_TRAIN = 2200
N_VAL = 600
N_OOD = 800
EPOCHS = 18
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

# ===================== BUILDING BLOCKS =====================

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


class HierResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_LAYERS * MOD_PER_LAYER)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)


class CausalKVLayer(nn.Module):
    """Một lớp KV_causal (sequential + sliding KV + causal mask)."""
    def __init__(self, topo_mode="sparse"):
        super().__init__()
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(MOD_PER_LAYER)
        ])
        self.attn = AdditiveAttention(HIDDEN)
        self.controller = nn.Sequential(nn.Linear(HIDDEN, 28), nn.Tanh(), nn.Linear(28, MOD_PER_LAYER))
        self.kv_init = nn.Parameter(torch.randn(MOD_PER_LAYER, HIDDEN) * 0.02)
        # topology mask
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
    """Một lớp full: mọi module chạy song song, message all-to-all, rồi residual mix."""
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(MOD_PER_LAYER)
        ])
        # low-rank message between modules
        self.msg_proj = nn.Linear(HIDDEN, HIDDEN // 2)
        self.msg_combine = nn.Linear(HIDDEN // 2 * MOD_PER_LAYER, HIDDEN)
        self.gate = nn.Sequential(nn.Linear(HIDDEN, MOD_PER_LAYER), nn.Sigmoid())

    def forward(self, h, temperature=1.0, return_traj=False):
        # every module processes h
        outs = [m(h) for m in self.mods]  # list of B,H
        stacked = torch.stack(outs, dim=1)  # B, M, H
        # all-to-all: each module receives messages from all
        msgs = self.msg_proj(stacked)  # B, M, H/2
        msgs_flat = msgs.reshape(h.size(0), -1)  # B, M*(H/2)
        mixed = self.msg_combine(msgs_flat)  # B, H
        g = self.gate(h)  # B, M
        # weighted residual from modules
        weighted = (stacked * g.unsqueeze(-1)).sum(1)  # B, H
        h = h + 0.4 * torch.tanh(weighted + 0.3 * mixed)
        if return_traj:
            # soft "participation" as pseudo-traj
            return h, g.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        return h


class KVCausalPure(nn.Module):
    """3 lớp KV_causal thuần (baseline)."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.layers = nn.ModuleList([
            CausalKVLayer("sparse"),
            CausalKVLayer("ring"),
            CausalKVLayer("complete"),
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


class KVInterleaveFull(nn.Module):
    """Xen kẽ: Causal - Full all-to-all - Causal."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.layer0 = CausalKVLayer("sparse")
        self.layer1 = FullAllToAllLayer()       # <-- full all-to-all
        self.layer2 = CausalKVLayer("complete")
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)

    def forward(self, x, return_traj=False, temperature=1.0):
        h = self.input_proj(x)
        trajs = []
        for layer in [self.layer0, self.layer1, self.layer2]:
            if return_traj:
                h, t = layer(h, temperature=temperature, return_traj=True)
                trajs.append(t)
            else:
                h = layer(h, temperature=temperature)
        out = self.out(h)
        if return_traj:
            return out, torch.stack(trajs, 1)
        return out


# ===================== METRICS & TRAIN =====================

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

    results = {}
    # Residual
    hr = train(HierResidual())
    results["HierRes_OOD"], _ = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    results["HierRes_ID"], _ = evaluate(hr, X=X_val, Y=Y_val, paths=paths_val)
    results["params_hr"] = count_params(HierResidual())

    # Pure causal
    pure = train(KVCausalPure(), use_traj=True)
    results["Pure_OOD"], results["Pure_traj"] = evaluate(pure, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
    results["Pure_ID"], _ = evaluate(pure, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
    results["params_pure"] = count_params(KVCausalPure())

    # Interleave
    inter = train(KVInterleaveFull(), use_traj=True)
    results["Inter_OOD"], results["Inter_traj"] = evaluate(inter, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
    results["Inter_ID"], _ = evaluate(inter, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
    results["params_inter"] = count_params(KVInterleaveFull())

    results["seed"] = seed
    return results


print(f"Params: HR={count_params(HierResidual())} Pure={count_params(KVCausalPure())} Inter={count_params(KVInterleaveFull())}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HierRes OOD={r['HierRes_OOD']:.4f}")
    print(f"  Pure    OOD={r['Pure_OOD']:.4f} traj={r['Pure_traj']:.3f}")
    print(f"  Inter   OOD={r['Inter_OOD']:.4f} traj={r['Inter_traj']:.3f}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
for k in ["HierRes_OOD", "Pure_OOD", "Pure_traj", "Inter_OOD", "Inter_traj"]:
    mu, sd = ms(k)
    print(f"  {k}: {mu:.4f} ± {sd:.4f}")

hr_mu = ms("HierRes_OOD")[0]
pure_mu = ms("Pure_OOD")[0]
inter_mu = ms("Inter_OOD")[0]
best_name = "Inter" if inter_mu <= pure_mu else "Pure"
best_mu = min(inter_mu, pure_mu)

if best_mu < hr_mu * 0.97:
    outcome = "PASS"
elif best_mu > hr_mu * 1.03:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} (best adaptive={best_name}) ===")
print(f"HierRes={hr_mu:.4f} Pure={pure_mu:.4f} Inter={inter_mu:.4f}")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Interleave KV_causal + 1 Full all-to-all layer",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["HierRes_OOD","Pure_OOD","Pure_traj","Inter_OOD","Inter_traj"]},
        "per_seed": all_results,
        "outcome": outcome,
        "params": {
            "hr": all_results[0]["params_hr"],
            "pure": all_results[0]["params_pure"],
            "inter": all_results[0]["params_inter"],
        }
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Xen kẽ KV_causal với 1 lớp Full all-to-all (Causal–Full–Causal)
- Seeds: {SEEDS}
- OOD: HierRes={hr_mu:.4f}, Pure={pure_mu:.4f}, Inter={inter_mu:.4f}
- Traj: Pure={ms('Pure_traj')[0]:.3f}, Inter={ms('Inter_traj')[0]:.3f}
- Outcome: **{outcome}** (best={best_name})
- Notes: Layer giữa = mọi module nối mọi module (parallel message). Hai lớp ngoài = KV_causal.
""")
print(f"Saved {run_dir}")
