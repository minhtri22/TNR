#!/usr/bin/env python3
"""
KV_causal parity chặt (ratio ~1.0-1.1) + đủ 5 seeds
Giữ nguyên task & kiến trúc causal sliding KV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

SEEDS = [42, 7, 123, 99, 2024]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_LAYERS = 3
MOD_PER_LAYER = 4
# Cân bằng: tăng residual width, giữ KV vừa phải
HIDDEN_HR = 42          # residual rộng hơn
HIDDEN_KV = 32          # KV hẹp hơn
INPUT_DIM = 16
OUTPUT_DIM = 4
SEQ_LEN = 5
KV_DIM = 16
BATCH_SIZE = 128
N_TRAIN = 2600
N_VAL = 500
N_OOD = 700
EPOCHS = 22
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
        H = HIDDEN_HR
        self.input_proj = nn.Linear(INPUT_DIM, H)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, H))
            for _ in range(N_LAYERS * MOD_PER_LAYER)
        ])
        self.out = nn.Linear(H, OUTPUT_DIM)
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class SeqKVCausal(nn.Module):
    def __init__(self):
        super().__init__()
        H = HIDDEN_KV
        self.input_proj = nn.Linear(INPUT_DIM, H)
        self.mod_lists = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, H))
                for _ in range(MOD_PER_LAYER)
            ]) for _ in range(N_LAYERS)
        ])
        self.attns = nn.ModuleList([AdditiveAttention(H) for _ in range(N_LAYERS)])
        self.controllers = nn.ModuleList([
            nn.Sequential(nn.Linear(H, 24), nn.Tanh(), nn.Linear(24, MOD_PER_LAYER))
            for _ in range(N_LAYERS)
        ])
        self.topo_masks = self._make_masks()
        self.out = nn.Linear(H, OUTPUT_DIM)
        self.kv_init = nn.Parameter(torch.randn(N_LAYERS, MOD_PER_LAYER, H) * 0.02)

    def _make_masks(self):
        masks = []
        for mode in ["sparse", "ring", "complete"]:
            adj = torch.zeros(MOD_PER_LAYER, MOD_PER_LAYER)
            if mode == "complete":
                adj = torch.ones(MOD_PER_LAYER, MOD_PER_LAYER)
            elif mode == "ring":
                for i in range(MOD_PER_LAYER):
                    adj[i, (i+1)%MOD_PER_LAYER] = adj[i, (i-1)%MOD_PER_LAYER] = adj[i, i] = 1
            else:
                rng = np.random.RandomState(42)
                for i in range(MOD_PER_LAYER):
                    adj[i, i] = 1
                    cands = [j for j in range(MOD_PER_LAYER) if j != i]
                    adj[i, rng.choice(cands, 2, replace=False)] = 1
            masks.append(adj)
        return nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in masks])

    def forward(self, x, return_traj=False, temperature=1.0):
        B = x.size(0)
        h = self.input_proj(x)
        all_traj = []
        for li in range(N_LAYERS):
            kv = self.kv_init[li].unsqueeze(0).expand(B, -1, -1).clone()
            written = torch.zeros(B, MOD_PER_LAYER, device=x.device)
            mask_topo = (self.topo_masks[li].sum(0) > 0).float().to(x.device)
            traj_layer = []
            for t in range(SEQ_LEN):
                attn_mask = written.clone() if t > 0 else torch.ones_like(written) * 0.15
                ctx, _ = self.attns[li](h, kv, kv, mask=attn_mask)
                logits = self.controllers[li](ctx)
                logits = logits.masked_fill(mask_topo.unsqueeze(0) == 0, -1e9)
                soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
                if self.training:
                    hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                    gate = hard + (soft - soft.detach())
                else:
                    gate = soft
                traj_layer.append(gate.detach())
                updates = [self.mod_lists[li][mi](h) * gate[:, mi:mi+1] for mi in range(MOD_PER_LAYER)]
                h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
                write = gate.unsqueeze(-1) * h.unsqueeze(1)
                kv = kv * (1 - gate.unsqueeze(-1)) + write
                written = torch.clamp(written + gate.detach(), 0, 1)
            all_traj.append(torch.stack(traj_layer, 1))
        out = self.out(h)
        if return_traj:
            return out, torch.stack(all_traj, 1)
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
        pred_seq = [(idx//SEQ_LEN)*MOD_PER_LAYER + c for idx, c in enumerate(chosen)]
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

    hr = train(HierResidual())
    hr_ood, _ = evaluate(hr, X=X_ood, Y=Y_ood, paths=paths_ood)
    hr_id, _ = evaluate(hr, X=X_val, Y=Y_val, paths=paths_val)

    kv = train(SeqKVCausal(), use_traj=True)
    kv_ood, kv_rec = evaluate(kv, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood)
    kv_id, _ = evaluate(kv, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)

    return {
        "seed": seed,
        "params_hr": count_params(HierResidual()),
        "params_kv": count_params(SeqKVCausal()),
        "HierRes_OOD": hr_ood, "HierRes_ID": hr_id,
        "KV_OOD": kv_ood, "KV_ID": kv_id, "KV_traj": kv_rec,
        "win": kv_ood < hr_ood,
    }

# Print params once
print(f"Params HierRes={count_params(HierResidual())}  KV={count_params(SeqKVCausal())}  ratio={count_params(SeqKVCausal())/count_params(HierResidual()):.2f}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====")
    r = run_seed(s)
    all_results.append(r)
    print(f"  HierRes OOD={r['HierRes_OOD']:.4f}  KV OOD={r['KV_OOD']:.4f} traj={r['KV_traj']:.3f} win={r['win']}")

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr_mu, hr_sd = ms("HierRes_OOD")
kv_mu, kv_sd = ms("KV_OOD")
tr_mu, tr_sd = ms("KV_traj")
n_wins = sum(r["win"] for r in all_results)
ratio = all_results[0]["params_kv"] / all_results[0]["params_hr"]

print("\n===== SUMMARY 5 SEEDS =====")
print(f"  Params ratio: {ratio:.2f}")
print(f"  HierRes OOD: {hr_mu:.4f} ± {hr_sd:.4f}")
print(f"  KV_causal OOD: {kv_mu:.4f} ± {kv_sd:.4f}")
print(f"  Traj: {tr_mu:.3f} ± {tr_sd:.3f}")
print(f"  Wins: {n_wins}/5")

if kv_mu < hr_mu * 0.97 and tr_mu >= 0.33 and n_wins >= 4:
    outcome = "PASS"
elif kv_mu > hr_mu * 1.03:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"
print(f"\n=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id, "seeds": SEEDS,
        "params": {"hr": all_results[0]["params_hr"], "kv": all_results[0]["params_kv"], "ratio": ratio},
        "summary": {"HierRes_OOD": [hr_mu, hr_sd], "KV_OOD": [kv_mu, kv_sd], "traj": [tr_mu, tr_sd], "wins": n_wins},
        "per_seed": all_results, "outcome": outcome
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: KV_causal parity ratio~{ratio:.2f} + đủ 5 seeds
- Params: HR={all_results[0]['params_hr']}, KV={all_results[0]['params_kv']}
- OOD: HierRes={hr_mu:.4f}±{hr_sd:.4f}, KV={kv_mu:.4f}±{kv_sd:.4f}
- Traj: {tr_mu:.3f}±{tr_sd:.3f} | Wins: {n_wins}/5
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
