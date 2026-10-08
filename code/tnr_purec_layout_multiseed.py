#!/usr/bin/env python3
"""Multi-seed 10: PureC ± layout only (core A). Light train for throughput."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

N_SEEDS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} N_SEEDS={N_SEEDS}")

N_KEYS = 6
STATE_DIM = 8
KEY_DIM = 6
INPUT_DIM = 14
OUTPUT_DIM = 4
HIDDEN = 40
MOD_PER_LAYER = 4
SEQ_LEN = 3
KV_DIM = 20
BATCH_SIZE = 128
N_TRAIN = 2000
N_OOD = 500
EPOCHS = 16
LR = 1e-3
TEMP_START, TEMP_END = 1.2, 0.35
LAMBDA_LAYOUT = 0.15

TRAIN_PATHS = {
    0: [0, 2], 1: [1, 3], 2: [2, 4], 3: [3, 5],
    4: [0, 4], 5: [1, 5], 6: [5, 0], 7: [4, 1],
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
    return keys, transforms

KEYS, TRANSFORMS = build_bank()

def apply_path(state, path):
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

def gen(n, path_dict, off=0):
    g = torch.Generator().manual_seed(42 + off)
    states = torch.randn(n, STATE_DIM, generator=g) * 0.8
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
    Y = Y + 0.015 * torch.randn_like(Y)
    return X, Y, paths

class Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wq = nn.Linear(dim, KV_DIM)
        self.wk = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(KV_DIM, 1)
        self.out = nn.Linear(dim, dim)
    def forward(self, q, k, v, mask=None):
        qq = self.wq(q).unsqueeze(1)
        kk = self.wk(k)
        sc = self.v(torch.tanh(qq + kk)).squeeze(-1)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, -1e9)
        a = F.softmax(sc, dim=-1)
        return self.out(torch.bmm(a.unsqueeze(1), v).squeeze(1)), a

class CausalLayer(nn.Module):
    def __init__(self, n_mod=4, topo="ring"):
        super().__init__()
        self.n_mod = n_mod
        self.mods = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN)) for _ in range(n_mod)])
        self.attn = Attn(HIDDEN)
        self.ctrl = nn.Sequential(nn.Linear(HIDDEN, 24), nn.Tanh(), nn.Linear(24, n_mod))
        self.kv0 = nn.Parameter(torch.randn(n_mod, HIDDEN) * 0.02)
        adj = torch.zeros(n_mod, n_mod)
        if topo == "complete":
            adj = torch.ones(n_mod, n_mod)
        elif topo == "ring":
            for i in range(n_mod):
                adj[i, i] = adj[i, (i+1)%n_mod] = adj[i, (i-1)%n_mod] = 1
        else:
            rng = np.random.RandomState(42)
            for i in range(n_mod):
                adj[i, i] = 1
                c = [j for j in range(n_mod) if j != i]
                adj[i, rng.choice(c, min(2, len(c)), replace=False)] = 1
        self.register_buffer("mask", (adj.sum(0) > 0).float())

    def forward(self, h, temp=1.0):
        B = h.size(0)
        kv = self.kv0.unsqueeze(0).expand(B, -1, -1).clone()
        written = torch.zeros(B, self.n_mod, device=h.device)
        traj = []
        for t in range(SEQ_LEN):
            am = written.clone() if t > 0 else torch.ones_like(written) * 0.15
            ctx, _ = self.attn(h, kv, kv, mask=am)
            logits = self.ctrl(ctx)
            logits = logits.masked_fill(self.mask.unsqueeze(0) == 0, -1e9)
            soft = F.softmax(logits / max(temp, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temp, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            traj.append(gate)
            ups = [self.mods[i](h) * gate[:, i:i+1] for i in range(self.n_mod)]
            h = h + 0.5 * torch.tanh(torch.stack(ups, 0).sum(0))
            kv = kv * (1 - gate.unsqueeze(-1)) + gate.unsqueeze(-1) * h.unsqueeze(1)
            written = torch.clamp(written + gate.detach(), 0, 1)
        return h, torch.stack(traj, 1)

class HR(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(INPUT_DIM, HIDDEN)
        self.b = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN)) for _ in range(10)])
        self.o = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x):
        h = self.p(x)
        for b in self.b:
            h = h + 0.4 * b(h)
        return self.o(h)

class PureC(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(INPUT_DIM, HIDDEN)
        self.layers = nn.ModuleList([
            CausalLayer(4, "sparse"), CausalLayer(4, "ring"), CausalLayer(4, "complete")
        ])
        self.o = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.n_layers, self.n_mod = 3, 4
    def forward(self, x, temp=1.0):
        h = self.p(x)
        trajs = []
        for L in self.layers:
            h, t = L(h, temp)
            trajs.append(t)
        return self.o(h), torch.stack(trajs, 1)

def path_tgt(paths, nL, T, M, device):
    B = len(paths)
    tgt = torch.zeros(B, nL, T, M, device=device)
    for b, p in enumerate(paths):
        for t, op in enumerate(p):
            tgt[b, t % nL, min(t, T-1), op % M] = 1.0
    return tgt

def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def traj_rec(traj, paths, n=200):
    B = min(n, traj.size(0))
    sc = []
    for b in range(B):
        ch = traj[b].argmax(-1).view(-1).tolist()
        sc.append(lcs([c % N_KEYS for c in ch], paths[b]) / max(len(paths[b]), 1))
    return float(np.mean(sc))

def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, ptr = gen(N_TRAIN, TRAIN_PATHS, seed)
    Xo, Yo, po = gen(N_OOD, OOD_PATHS, seed+20)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xo, Yo = Xo.to(DEVICE), Yo.to(DEVICE)

    def train_hr():
        m = HR().to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        for ep in range(EPOCHS):
            m.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(m(Xtr[idx]), Ytr[idx]).backward()
                opt.step()
        return m

    def train_pc(layout=False):
        m = PureC().to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        for ep in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (ep / max(EPOCHS-1, 1))
            m.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                bp = [ptr[j] for j in idx.cpu().tolist()]
                opt.zero_grad()
                pred, traj = m(Xtr[idx], temp)
                loss = F.mse_loss(pred, Ytr[idx])
                if layout:
                    tgt = path_tgt(bp, m.n_layers, SEQ_LEN, m.n_mod, DEVICE)
                    loss = loss + LAMBDA_LAYOUT * (-(tgt * traj.clamp(1e-8).log()).sum() / (tgt.sum() + 1e-8))
                loss.backward()
                opt.step()
        return m

    def eval_hr(m):
        m.eval()
        with torch.no_grad():
            pred = m(Xo)
            return F.mse_loss(pred, Yo).item()

    def eval_pc(m):
        m.eval()
        with torch.no_grad():
            pred, traj = m(Xo, TEMP_END)
            mse = F.mse_loss(pred, Yo).item()
            rec = traj_rec(traj.cpu(), po)
        return mse, rec

    hr = train_hr()
    hr_ood = eval_hr(hr)
    pc0 = train_pc(False)
    pc0_ood, pc0_t = eval_pc(pc0)
    pc1 = train_pc(True)
    pc1_ood, pc1_t = eval_pc(pc1)
    return {"hr": hr_ood, "pc0": pc0_ood, "pc0_t": pc0_t, "pc1": pc1_ood, "pc1_t": pc1_t}

hrs, p0s, p1s, t0s, t1s = [], [], [], [], []
w0 = w1 = 0
for i, s in enumerate(range(N_SEEDS)):
    r = run_seed(s)
    hrs.append(r["hr"]); p0s.append(r["pc0"]); p1s.append(r["pc1"])
    t0s.append(r["pc0_t"]); t1s.append(r["pc1_t"])
    if r["pc0"] < r["hr"]: w0 += 1
    if r["pc1"] < r["hr"]: w1 += 1
    print(f"  [{i+1}/{N_SEEDS}] HR={r['hr']:.4f} PC0={r['pc0']:.4f} t={r['pc0_t']:.2f} | PC1={r['pc1']:.4f} t={r['pc1_t']:.2f}")

print("\n===== 10-SEED SUMMARY =====")
print(f"HR:   {np.mean(hrs):.4f} ± {np.std(hrs):.4f}")
print(f"PC0:  {np.mean(p0s):.4f} ± {np.std(p0s):.4f}  traj={np.mean(t0s):.3f}  wins={w0}/{N_SEEDS}  ratio={np.mean(p0s)/np.mean(hrs):.3f}")
print(f"PC1:  {np.mean(p1s):.4f} ± {np.std(p1s):.4f}  traj={np.mean(t1s):.3f}  wins={w1}/{N_SEEDS}  ratio={np.mean(p1s)/np.mean(hrs):.3f}")

best = "PC1" if np.mean(p1s) <= np.mean(p0s) else "PC0"
bo = min(np.mean(p0s), np.mean(p1s))
bt = np.mean(t1s) if best == "PC1" else np.mean(t0s)
bw = w1 if best == "PC1" else w0
hr_m = np.mean(hrs)

if bo < hr_m * 0.95 and bt >= 0.45 and bw >= 7:
    outcome = "PASS"
elif bo > hr_m * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"Best={best} → {outcome}")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "n_seeds": N_SEEDS,
        "HR": [float(np.mean(hrs)), float(np.std(hrs))],
        "PC0": [float(np.mean(p0s)), float(np.std(p0s)), float(np.mean(t0s)), w0],
        "PC1": [float(np.mean(p1s)), float(np.std(p1s)), float(np.mean(t1s)), w1],
        "outcome": outcome,
        "per_seed": {"hr": hrs, "pc0": p0s, "pc1": p1s, "t0": t0s, "t1": t1s},
    }, f)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **A-core multi-seed n=10** PureC ± layout only
- HR={np.mean(hrs):.4f}±{np.std(hrs):.4f}
- PC0={np.mean(p0s):.4f} traj={np.mean(t0s):.3f} wins={w0}/10
- PC1={np.mean(p1s):.4f} traj={np.mean(t1s):.3f} wins={w1}/10
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
