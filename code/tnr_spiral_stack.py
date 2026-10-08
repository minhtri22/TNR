#!/usr/bin/env python3
"""
TNR: Pattern C-C-Spiral-C-C-F
  2 Causal + Spiral (xoay state) + 2 Causal + Full

So sánh với HierResidual, PureC (CCC) trên multi-hop v2.5 difficulty.
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
N_OPS = 6
HIDDEN = 40
INPUT_DIM = 12
OUTPUT_DIM = 4
SEQ_LEN = 4
KV_DIM = 20
BATCH_SIZE = 128
N_TRAIN = 2600
N_VAL = 500
N_OOD = 700
EPOCHS = 20
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

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

class SpiralLayer(nn.Module):
    """
    Xoay ốc trong latent space:
      - Học góc theta (scalar + feature-dependent)
      - Rotation theo từng cặp dims (2D Givens-style)
      - Radial term: kéo/đẩy nhẹ theo norm
    → lực có moment (xoay) + xuyên tâm (spiral)
    """
    def __init__(self, dim=HIDDEN):
        super().__init__()
        self.dim = dim
        n_pairs = dim // 2
        self.theta_base = nn.Parameter(torch.zeros(n_pairs))
        self.theta_net = nn.Sequential(nn.Linear(dim, n_pairs), nn.Tanh())
        self.radial_net = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.n_pairs = n_pairs

    def forward(self, h, temperature=1.0, return_traj=False):
        B, D = h.shape
        n = self.n_pairs
        theta = 0.5 * self.theta_base + 0.3 * self.theta_net(h)  # B, n_pairs
        # reshape pairs
        h_pairs = h[:, : 2 * n].view(B, n, 2)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x, y = h_pairs[:, :, 0], h_pairs[:, :, 1]
        x2 = cos_t * x - sin_t * y
        y2 = sin_t * x + cos_t * y
        rotated = torch.stack([x2, y2], dim=-1).view(B, 2 * n)
        if D > 2 * n:
            rotated = torch.cat([rotated, h[:, 2 * n :]], dim=-1)
        # radial spiral force
        r_gate = self.radial_net(h)  # B, 1
        radial = r_gate * h
        out = h + 0.35 * rotated + 0.15 * radial + 0.25 * self.mix(h)
        if return_traj:
            # pseudo-traj: soft participation from |theta|
            fake = torch.softmax(theta.abs(), dim=-1)
            if fake.size(-1) < MOD_PER_LAYER:
                pad = torch.zeros(B, MOD_PER_LAYER - fake.size(-1), device=h.device)
                fake = torch.cat([fake, pad], dim=-1)
            else:
                fake = fake[:, :MOD_PER_LAYER]
            return out, fake.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        return out

class FullLayer(nn.Module):
    def __init__(self, n_mod=MOD_PER_LAYER):
        super().__init__()
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
    """pattern chars: C=causal, S=spiral, F=full"""
    def __init__(self, pattern="CCSCCF"):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        n_mod = min(MOD_PER_LAYER, N_OPS)
        layers = []
        topos = ["sparse", "ring", "complete", "ring", "sparse"]
        ci = 0
        for p in pattern:
            if p == "C":
                layers.append(CausalKVLayer(n_mod=n_mod, topo=topos[ci % len(topos)]))
                ci += 1
            elif p == "S":
                layers.append(SpiralLayer(HIDDEN))
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

def traj_seq_recovery(traj, true_paths, n_sample=280):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [c % N_OPS for c in chosen]
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
            r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
        return mse, rec, r2

    out = {"seed": seed}
    hr = train(HierResidual(12))
    out["HR_OOD"], _, out["HR_OOD_r2"] = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    out["params_hr"] = count_params(HierResidual(12))

    for name, pattern in [("PureC", "CCC"), ("SpiralStack", "CCSCCF")]:
        m = train(HybridStack(pattern), use_traj=True)
        ood, traj, r2 = evaluate(m, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
        out[f"{name}_OOD"] = ood
        out[f"{name}_traj"] = traj
        out[f"{name}_OOD_r2"] = r2
        out[f"params_{name}"] = count_params(HybridStack(pattern))
    return out


print("Params HR", count_params(HierResidual(12)),
      "PureC", count_params(HybridStack("CCC")),
      "Spiral", count_params(HybridStack("CCSCCF")))

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR     OOD={r['HR_OOD']:.4f} R2={r['HR_OOD_r2']:.3f}")
    print(f"  PureC  OOD={r['PureC_OOD']:.4f} traj={r['PureC_traj']:.3f}")
    print(f"  Spiral OOD={r['SpiralStack_OOD']:.4f} traj={r['SpiralStack_traj']:.3f}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== SUMMARY =====")
for k in ["HR_OOD", "HR_OOD_r2", "PureC_OOD", "PureC_traj", "SpiralStack_OOD", "SpiralStack_traj"]:
    mu, sd = ms(k)
    print(f"  {k}: {mu:.4f} ± {sd:.4f}")

hr_mu = ms("HR_OOD")[0]
pure_mu = ms("PureC_OOD")[0]
spiral_mu = ms("SpiralStack_OOD")[0]
best_name = "SpiralStack" if spiral_mu <= pure_mu else "PureC"
best_mu = min(spiral_mu, pure_mu)
best_traj = ms(f"{best_name}_traj")[0]
wins = sum(1 for r in all_results if r[f"{best_name}_OOD"] < r["HR_OOD"])

print(f"\nBest={best_name} OOD={best_mu:.4f} traj={best_traj:.3f} vs HR={hr_mu:.4f} wins={wins}/3")

if best_mu < hr_mu * 0.95 and best_traj >= 0.40 and wins >= 2:
    outcome = "PASS"
elif best_mu > hr_mu * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"=== OUTCOME: {outcome} ===")
print(f"Spiral vs PureC: Spiral={spiral_mu:.4f} PureC={pure_mu:.4f}")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "C-C-Spiral-C-C-F vs PureC vs HR on multi-hop v2.5",
        "pattern": "CCSCCF",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["HR_OOD","PureC_OOD","PureC_traj","SpiralStack_OOD","SpiralStack_traj"]},
        "per_seed": all_results,
        "outcome": outcome,
        "best": best_name,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **C-C-Spiral-C-C-F** (xoay ốc state + 2C + F) trên multi-hop v2.5
- OOD: HR={hr_mu:.4f}, PureC={pure_mu:.4f}, Spiral={spiral_mu:.4f}
- Traj: PureC={ms('PureC_traj')[0]:.3f}, Spiral={ms('SpiralStack_traj')[0]:.3f}
- Best={best_name}, wins={wins}/3
- Outcome: **{outcome}**
- Notes: Spiral = Givens rotation theo cặp dims + radial force learnable.
""")
print(f"Saved {run_dir}")
