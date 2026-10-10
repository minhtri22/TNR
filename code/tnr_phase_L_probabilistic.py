#!/usr/bin/env python3
"""
Phase L — Probabilistic composition (họ 5)

Task: ambiguous skill composition trên point-mass-lite (2D position only).
  Hai chương trình plausible cho cùng hint bag; target = E[outcome | posterior].
  Train: quan sát (state0, bag_hint, noisy final pos từ true program).
  OOD: bag trùng nhiều programs hơn / chương trình novel length.

Giả thuyết L1 — Mixture over programs:
  q(k) = softmax(score(state, hint, program_k))
  pred = Σ_k q(k) * Execute(program_k, state0)
vs
  Residual: map (state, hint) → pos (point estimate)

TRUE programs = fixed skill sequences; Execute = deterministic dynamics.

Preregister:
  PASS: Mix OOD MSE < 0.95*HR AND wins >= 2/3
  FAIL: Mix > 1.05*HR
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

STATE_DIM = 2  # x, y only (no velocity — simpler probabilistic)
N_SKILLS = 4
HINT_DIM = 8
HIDDEN = 64
DT = 1.0
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 28
LR = 1e-3

# skills = unit steps
def build_skills():
    return torch.tensor([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    ], dtype=torch.float32)

SKILLS = build_skills()
SKILL_KEYS = F.normalize(torch.randn(N_SKILLS, HINT_DIM), dim=-1)

# Candidate programs (library)
TRAIN_PROGS = [
    [0, 2], [1, 3], [0, 1], [2, 3],
    [0, 0], [2, 2], [1, 1], [3, 3],
]
OOD_PROGS = [
    [0, 2, 1], [1, 3, 0], [0, 1, 2], [2, 3, 1],
    [0, 0, 2], [2, 2, 0], [1, 1, 3], [3, 3, 1],
]

def bag_hint(prog):
    h = torch.zeros(HINT_DIM)
    for k in prog:
        h = h + SKILL_KEYS[k % N_SKILLS]
    return h / max(len(prog), 1)

def execute(state0, prog):
    s = state0.clone()
    for k in prog:
        s = s + DT * SKILLS[k % N_SKILLS].unsqueeze(0).expand(s.size(0), -1)
    return s

def make_batch(n, prog_list, seed, noise=0.05):
    g = torch.Generator().manual_seed(42 + seed)
    state0 = torch.randn(n, STATE_DIM, generator=g) * 0.5
    idx = torch.randint(0, len(prog_list), (n,), generator=g)
    Y = torch.zeros(n, STATE_DIM)
    hints = torch.zeros(n, HINT_DIM)
    for i in range(n):
        prog = prog_list[idx[i].item()]
        Y[i] = execute(state0[i:i+1], prog).squeeze(0)
        hints[i] = bag_hint(prog)
    Y = Y + noise * torch.randn(n, STATE_DIM, generator=g)
    hints = hints + 0.05 * torch.randn_like(hints)
    return state0, hints, Y, idx

class HierResidual(nn.Module):
    def __init__(self, in_dim, n_blocks=8):
        super().__init__()
        self.proj = nn.Linear(in_dim, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, STATE_DIM)
    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class MixtureOverPrograms(nn.Module):
    """
    Soft posterior over a fixed program library; predictive mean under execute.
    Library = TRAIN_PROGS at train; at OOD we expand library to TRAIN+OOD
    so mixture can put mass on novel programs (open library for fair OOD).
    """
    def __init__(self, library):
        super().__init__()
        self.library = library  # list of programs
        self.score = nn.Sequential(
            nn.Linear(STATE_DIM + HINT_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, len(library)),
        )

    def set_library(self, library):
        # rebuild last layer if size changes — for OOD we keep train library only
        # (harder: cannot name novel progs) OR expand. Use train library only = conservative.
        pass

    def forward(self, state0, hint):
        ctx = torch.cat([state0, hint], dim=-1)
        logits = self.score(ctx)
        q = F.softmax(logits, dim=-1)  # B, K
        # execute all programs
        outs = []
        for prog in self.library:
            outs.append(execute(state0, prog))
        stack = torch.stack(outs, dim=1)  # B, K, 2
        pred = (q.unsqueeze(-1) * stack).sum(1)
        return pred, q


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s0_tr, h_tr, Ytr, _ = make_batch(N_TRAIN, TRAIN_PROGS, seed)
    s0_va, h_va, Yva, _ = make_batch(N_VAL, TRAIN_PROGS, seed + 10)
    s0_ood, h_ood, Yood, _ = make_batch(N_OOD, OOD_PROGS, seed + 20)
    Xtr = torch.cat([s0_tr, h_tr], -1)
    Xva = torch.cat([s0_va, h_va], -1)
    Xood = torch.cat([s0_ood, h_ood], -1)
    in_dim = Xtr.size(1)

    def train_hr():
        model = HierResidual(in_dim).to(DEVICE)
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

    def train_mix(library):
        model = MixtureOverPrograms(library).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        S0, H, Y = s0_tr.to(DEVICE), h_tr.to(DEVICE), Ytr.to(DEVICE)
        S0v, Hv, Yv = s0_va.to(DEVICE), h_va.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred, q = model(S0[idx], H[idx])
                # MSE + mild entropy (encourage calibrated mixture under noise)
                loss = F.mse_loss(pred, Y[idx]) - 0.01 * (q * (q + 1e-8).log()).sum(-1).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred, _ = model(S0v, Hv)
                val = F.mse_loss(pred, Yv).item()
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
        out["HR"] = F.mse_loss(hr(Xood.to(DEVICE)), Yood.to(DEVICE)).item()

    # Library = train only (cannot enumerate OOD progs by name) — hard transfer
    mix = train_mix(TRAIN_PROGS)
    mix.eval()
    with torch.no_grad():
        pred, _ = mix(s0_ood.to(DEVICE), h_ood.to(DEVICE))
        out["MIX"] = F.mse_loss(pred, Yood.to(DEVICE)).item()

    # Upper: library includes OOD programs (oracle library)
    mix2 = train_mix(TRAIN_PROGS + OOD_PROGS)
    mix2.eval()
    with torch.no_grad():
        pred2, _ = mix2(s0_ood.to(DEVICE), h_ood.to(DEVICE))
        out["MIX_full"] = F.mse_loss(pred2, Yood.to(DEVICE)).item()

    out["win"] = out["MIX"] < out["HR"]
    out["win_full"] = out["MIX_full"] < out["HR"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} MIX={r['MIX']:.4f} MIX_full={r['MIX_full']:.4f} "
          f"win={r['win']} win_full={r['win_full']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, mix, full = ms("HR")[0], ms("MIX")[0], ms("MIX_full")[0]
wins = sum(r["win"] for r in all_results)
wins_f = sum(r["win_full"] for r in all_results)
print("\n===== L PROBABILISTIC SUMMARY =====")
print(f"HR={hr:.4f} MIX(train-lib)={mix:.4f} MIX_full={full:.4f}")
print(f"ratio_mix={mix/hr:.3f} wins={wins}/3 wins_full={wins_f}/3")

# Primary claim: MIX with train library only (fair OOD)
if mix < hr * 0.95 and wins >= 2:
    outcome = "PASS"
elif mix > hr * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"
print(f"=== OUTCOME (train-lib OOD): {outcome} ===")
print(f"=== FULL-lib (oracle library) wins_full={wins_f}/3 ratio={full/hr:.3f} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase L Mixture over programs",
        "seeds": SEEDS,
        "summary": {"HR": hr, "MIX": mix, "MIX_full": full, "wins": wins, "wins_full": wins_f},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase L Probabilistic** mixture over programs vs residual
- HR={hr:.4f} MIX={mix:.4f} MIX_full={full:.4f} wins={wins}/3 wins_full={wins_f}/3
- Outcome (train-lib): **{outcome}**
""")
print(f"Saved {run_dir}")
