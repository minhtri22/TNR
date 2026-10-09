#!/usr/bin/env python3
"""
Củng cố E2: MatrixComp trên OOD path dài hơn (L=4)

Train: L=2 (như E2)
OOD: L=4 novel chains
N_STEPS model = 4 (đủ depth)

Preregister:
  PASS: MC_OOD < 0.95*HR AND wins >= 2/3
  FAIL: MC_OOD > 1.05*HR
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

N_OPS = 6
STATE_DIM = 8
HINT_DIM = 6
INPUT_DIM = STATE_DIM + HINT_DIM
OUTPUT_DIM = 4
HIDDEN = 48
N_MOD = 6
N_STEPS = 4
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 26
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

TRAIN_PATHS = {
    0: [0, 2], 1: [1, 3], 2: [2, 4], 3: [3, 5],
    4: [4, 0], 5: [5, 1], 6: [0, 4], 7: [1, 5],
}
OOD_L4 = {
    0: [0, 2, 4, 1], 1: [1, 3, 5, 0], 2: [2, 4, 1, 3],
    3: [3, 5, 0, 2], 4: [4, 0, 2, 5], 5: [5, 1, 3, 4],
    6: [0, 3, 1, 5], 7: [2, 5, 4, 0],
}

def build_ops():
    torch.manual_seed(19)
    ops = []
    for i in range(N_OPS):
        W = torch.eye(STATE_DIM)
        a, b = i % (STATE_DIM - 1), (i % (STATE_DIM - 1) + 1) % STATE_DIM
        theta = (i + 1) * np.pi / 7
        c, s = np.cos(theta), np.sin(theta)
        W[a, a], W[a, b], W[b, a], W[b, b] = c, -s, s, c
        scale_ax = (i * 2 + 3) % STATE_DIM
        W[scale_ax, scale_ax] = 1.15 if i % 2 == 0 else 0.85
        bias = torch.zeros(STATE_DIM)
        bias[a] = 0.15 * ((i % 3) - 1)
        ops.append((W, bias))
    torch.manual_seed(42)
    return ops

OPS = build_ops()
OP_KEYS = F.normalize(torch.randn(N_OPS, HINT_DIM), dim=-1)

def apply_path(state, path):
    h = state.clone()
    for idx in path:
        W, b = OPS[idx % N_OPS]
        h = torch.tanh(h @ W.T + b)
    return h[:, :OUTPUT_DIM]

def make_input(state, path):
    B = state.size(0)
    hint = 0.5 * OP_KEYS[path[0] % N_OPS] + 0.5 * OP_KEYS[path[-1] % N_OPS]
    hint = hint.unsqueeze(0).expand(B, -1) + 0.05 * torch.randn(B, HINT_DIM)
    return torch.cat([state, hint], dim=-1)

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    states = torch.randn(n, STATE_DIM, generator=g) * 0.9
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    X = torch.zeros(n, INPUT_DIM)
    Y = torch.zeros(n, OUTPUT_DIM)
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        X[i] = make_input(states[i:i+1], p).squeeze(0)
        Y[i] = apply_path(states[i:i+1], p).squeeze(0)
    Y = Y + 0.01 * torch.randn_like(Y)
    return X, Y

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

class MatrixComposer(nn.Module):
    def __init__(self, n_mod=N_MOD, n_steps=N_STEPS):
        super().__init__()
        self.n_mod = n_mod
        self.n_steps = n_steps
        W0 = torch.eye(STATE_DIM).unsqueeze(0).expand(n_mod, -1, -1).clone()
        W0 = W0 + torch.randn_like(W0) * 0.05
        self.W = nn.Parameter(W0)
        self.b = nn.Parameter(torch.zeros(n_mod, STATE_DIM))
        self.ctrl = nn.Sequential(nn.Linear(INPUT_DIM, 64), nn.Tanh(), nn.Linear(64, n_mod))
        self.step_embed = nn.Embedding(n_steps, n_mod)
        self.out = nn.Linear(STATE_DIM, OUTPUT_DIM)

    def forward(self, x, temperature=1.0):
        h = x[:, :STATE_DIM]
        for t in range(self.n_steps):
            logits = self.ctrl(x) + self.step_embed.weight[t].unsqueeze(0)
            soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            h_exp = h.unsqueeze(1).expand(-1, self.n_mod, -1)
            applied = torch.tanh(torch.einsum("bmd,mde->bme", h_exp, self.W) + self.b.unsqueeze(0))
            h = (applied * gate.unsqueeze(-1)).sum(1)
        return self.out(h)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    Xva, Yva = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    Xood, Yood = generate_dataset(N_OOD, OOD_L4, seed+20)

    def train(model, use_temp=False):
        model = model.to(DEVICE)
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
                pred = model(X[idx], temperature=temp) if use_temp else model(X[idx])
                F.mse_loss(pred, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(Xv, temperature=TEMP_END) if use_temp else model(Xv)
                val = F.mse_loss(pred, Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def eval_ood(model, use_temp=False):
        model.eval()
        with torch.no_grad():
            pred = model(Xood.to(DEVICE), temperature=TEMP_END) if use_temp else model(Xood.to(DEVICE))
            return F.mse_loss(pred, Yood.to(DEVICE)).item()

    out = {"seed": seed}
    hr = train(HierResidual())
    out["HR"] = eval_ood(hr)
    mc = train(MatrixComposer(), use_temp=True)
    out["MC"] = eval_ood(mc, use_temp=True)
    out["win"] = out["MC"] < out["HR"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} MC={r['MC']:.4f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, mc = ms("HR")[0], ms("MC")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== E2 LONG PATH L=4 SUMMARY =====")
print(f"HR={hr:.4f} MC={mc:.4f} ratio={mc/hr:.3f} wins={wins}/3")

if mc < hr * 0.95 and wins >= 2:
    outcome = "PASS"
elif mc > hr * 1.05:
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
        "experiment": "E2 strengthen long path L=4",
        "seeds": SEEDS,
        "summary": {"HR": hr, "MC": mc, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **E2 long-path L=4** MatrixComp vs HR
- HR={hr:.4f} MC={mc:.4f} ratio={mc/hr:.3f} wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
