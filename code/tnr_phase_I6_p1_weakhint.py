#!/usr/bin/env python3
"""
Phase I6 = P1: routing-critical control (weak hint)

Đổi so với I2–I5:
  Ordered skill-key hint  →  BAG hint (tổng keys, không thứ tự)
  → Residual khó copy sequence; router phải suy order / composition.

Cùng dynamics, TRUE skills frozen, controller learned (Gumbel).
So sánh HR vs SkillComp.

Preregister:
  PASS: SC < 0.95*HR AND wins >= 2/3
  FAIL: SC > 1.05*HR
  else UNRESOLVED

Nếu FAIL → đóng control (shortcut + OOD path cứng cả khi hint yếu).
Nếu PASS → cửa cho P2/P3.
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

STATE_DIM = 4
N_SKILLS = 6
HINT_DIM = 8
HIDDEN = 64
HORIZON_TRAIN = 2
HORIZON_OOD = 3
DT = 0.4
DAMP = 0.85
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 30
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

def build_true_skills():
    torch.manual_seed(21)
    angles = np.linspace(0, 2 * np.pi, N_SKILLS, endpoint=False)
    skills = []
    for i, a in enumerate(angles):
        mag = 0.8 + 0.15 * (i % 3)
        skills.append(torch.tensor([mag * np.cos(a), mag * np.sin(a)], dtype=torch.float32))
    torch.manual_seed(42)
    return torch.stack(skills)

TRUE_SKILLS = build_true_skills()
SKILL_KEYS = F.normalize(torch.randn(N_SKILLS, HINT_DIM), dim=-1)

TRAIN_SEQS = {
    0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4],
    4: [4, 5], 5: [5, 0], 6: [0, 3], 7: [2, 5],
}
OOD_SEQS = {
    0: [0, 1, 3], 1: [1, 2, 4], 2: [2, 3, 5],
    3: [3, 4, 0], 4: [4, 5, 1], 5: [5, 0, 2],
    6: [0, 2, 4], 7: [1, 3, 5],
}

def step_dynamics(state, impulse):
    x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    vx = DAMP * vx + impulse[:, 0]
    vy = DAMP * vy + impulse[:, 1]
    x = x + DT * vx
    y = y + DT * vy
    return torch.stack([x, y, vx, vy], dim=-1)

def rollout(state0, skill_ids):
    s = state0.clone()
    for k in skill_ids:
        imp = TRUE_SKILLS[k % N_SKILLS].unsqueeze(0).expand(s.size(0), -1)
        s = step_dynamics(s, imp)
    return s[:, :2]

def bag_hint(seq):
    """Sum of skill keys — NO order information."""
    h = torch.zeros(HINT_DIM)
    for k in seq:
        h = h + SKILL_KEYS[k % N_SKILLS]
    h = h / max(len(seq), 1)
    return h

def make_batch(n, seq_dict, seed):
    g = torch.Generator().manual_seed(42 + seed)
    state0 = torch.randn(n, STATE_DIM, generator=g) * 0.5
    keys = list(seq_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, 2)
    hints = torch.zeros(n, HINT_DIM)
    for i in range(n):
        seq = seq_dict[keys[cases[i].item()]]
        Y[i] = rollout(state0[i:i+1], seq).squeeze(0)
        hints[i] = bag_hint(seq)
    Y = Y + 0.02 * torch.randn(n, 2, generator=g)
    hints = hints + 0.08 * torch.randn_like(hints)  # stronger noise than ordered
    X = torch.cat([state0, hints], dim=-1)
    return X, Y, state0

class HierResidual(nn.Module):
    def __init__(self, in_dim, n_blocks=10):
        super().__init__()
        self.proj = nn.Linear(in_dim, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, 2)
    def forward(self, x, **kwargs):
        h = self.proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class SkillComposer(nn.Module):
    def __init__(self, in_dim, n_steps=HORIZON_OOD):
        super().__init__()
        self.n_steps = n_steps
        self.register_buffer("skills", TRUE_SKILLS.clone())
        self.ctrl = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, N_SKILLS))
        self.step_embed = nn.Embedding(n_steps, N_SKILLS)

    def forward(self, x, state0, temperature=1.0):
        h = state0.clone()
        for t in range(self.n_steps):
            logits = self.ctrl(x) + self.step_embed.weight[t].unsqueeze(0)
            soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            impulse = gate @ self.skills
            h = step_dynamics(h, impulse)
        return h[:, :2]


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, s0_tr = make_batch(N_TRAIN, TRAIN_SEQS, seed)
    Xva, Yva, s0_va = make_batch(N_VAL, TRAIN_SEQS, seed + 10)
    Xood, Yood, s0_ood = make_batch(N_OOD, OOD_SEQS, seed + 20)
    in_dim = Xtr.size(1)

    def train(model, use_sc=False):
        model = model.to(DEVICE)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
        X, Y, S0 = Xtr.to(DEVICE), Ytr.to(DEVICE), s0_tr.to(DEVICE)
        Xv, Yv, S0v = Xva.to(DEVICE), Yva.to(DEVICE), s0_va.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS - 1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred = model(X[idx], S0[idx], temperature=temp) if use_sc else model(X[idx])
                F.mse_loss(pred, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(Xv, S0v, temperature=TEMP_END) if use_sc else model(Xv)
                val = F.mse_loss(pred, Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def eval_ood(model, use_sc=False):
        model.eval()
        with torch.no_grad():
            X, Y, S0 = Xood.to(DEVICE), Yood.to(DEVICE), s0_ood.to(DEVICE)
            pred = model(X, S0, temperature=TEMP_END) if use_sc else model(X)
            return F.mse_loss(pred, Y).item()

    out = {"seed": seed}
    hr = train(HierResidual(in_dim))
    out["HR"] = eval_ood(hr)
    sc = train(SkillComposer(in_dim), use_sc=True)
    out["SC"] = eval_ood(sc, use_sc=True)
    out["win"] = out["SC"] < out["HR"]
    return out


print("P1: bag hint (no order) — routing-critical control")
all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} SC={r['SC']:.4f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, sc = ms("HR")[0], ms("SC")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== P1 WEAK-HINT SUMMARY =====")
print(f"HR={hr:.4f}±{ms('HR')[1]:.4f}")
print(f"SC={sc:.4f}±{ms('SC')[1]:.4f} ratio={sc/hr:.3f} wins={wins}/3")

if sc < hr * 0.95 and wins >= 2:
    outcome = "PASS"
elif sc > hr * 1.05:
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
        "experiment": "P1/I6 routing-critical bag hint",
        "seeds": SEEDS,
        "summary": {"HR": hr, "SC": sc, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **P1/I6** bag hint (no order) routing-critical control
- HR={hr:.4f} SC={sc:.4f} ratio={sc/hr:.3f} wins={wins}/3
- Outcome: **{outcome}**
- Gate: PASS→P2/P3; FAIL→đóng control
""")
print(f"Saved {run_dir}")
