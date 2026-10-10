#!/usr/bin/env python3
"""
Phase I — Continuous control / skill composition (họ 3)

Môi trường:
  Point-mass 2D: state (x,y,vx,vy), horizon T steps.
  Mỗi skill k = impulse (ax, ay) cố định + damping.
  Goal: đạt target position sau chuỗi skill.

Train: skill sequences length L=2
OOD:  novel sequences length L=3

Giả thuyết I1 — Explicit skill library + sequential apply:
  Học bank skill (Δv parameters) + Gumbel chọn skill theo bước
  (cùng tinh thần E2 matrix compose, nhưng trên dynamics)
Baseline: HierResidual flat policy state+goal → force mỗi bước

Preregister:
  PASS: SkillComp OOD MSE < 0.95 * HR AND wins >= 2/3
  FAIL: SkillComp OOD > 1.05 * HR
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

STATE_DIM = 4   # x,y,vx,vy
GOAL_DIM = 2    # target x,y
N_SKILLS = 6
HIDDEN = 64
HORIZON_TRAIN = 2
HORIZON_OOD = 3
DT = 0.4
DAMP = 0.85
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 28
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

# Fixed true skills (impulse library) — model must rediscover composition
def build_true_skills():
    torch.manual_seed(21)
    # 6 impulses in different directions / magnitudes
    angles = np.linspace(0, 2 * np.pi, N_SKILLS, endpoint=False)
    skills = []
    for i, a in enumerate(angles):
        mag = 0.8 + 0.15 * (i % 3)
        skills.append(torch.tensor([mag * np.cos(a), mag * np.sin(a)], dtype=torch.float32))
    torch.manual_seed(42)
    return skills

TRUE_SKILLS = build_true_skills()

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
    """state: B,4 ; impulse: B,2 → next state"""
    x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    vx = DAMP * vx + impulse[:, 0]
    vy = DAMP * vy + impulse[:, 1]
    x = x + DT * vx
    y = y + DT * vy
    return torch.stack([x, y, vx, vy], dim=-1)

def rollout(state0, skill_ids):
    """Apply sequence of true skills; return final (x,y)."""
    s = state0.clone()
    for k in skill_ids:
        imp = TRUE_SKILLS[k % N_SKILLS].unsqueeze(0).expand(s.size(0), -1)
        s = step_dynamics(s, imp)
    return s[:, :2]

def make_batch(n, seq_dict, seed):
    g = torch.Generator().manual_seed(42 + seed)
    state0 = torch.randn(n, STATE_DIM, generator=g) * 0.5
    keys = list(seq_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    goals = torch.zeros(n, GOAL_DIM)
    seqs = []
    for i in range(n):
        seq = seq_dict[keys[cases[i].item()]]
        seqs.append(seq)
        goals[i] = rollout(state0[i:i+1], seq).squeeze(0)
    goals = goals + 0.02 * torch.randn(n, GOAL_DIM, generator=g)
    # input: state0 + goal
    X = torch.cat([state0, goals], dim=-1)  # B, 6
    Y = goals  # predict final position (same as goal with noise — learn dynamics+seq)
    # Actually task: given state0 and *hint of which skills* weakly, reach goal
    # Stronger: given state0 + skill-endpoint hint embedding
    return X, Y, seqs, state0

def skill_hint(seqs, dim=8):
    """Weak bag hint of skills used (not order)."""
    torch.manual_seed(99)
    keys = F.normalize(torch.randn(N_SKILLS, dim), dim=-1)
    torch.manual_seed(42)
    B = len(seqs)
    H = torch.zeros(B, dim)
    for i, seq in enumerate(seqs):
        for k in seq:
            H[i] = H[i] + keys[k % N_SKILLS]
        H[i] = H[i] / max(len(seq), 1)
    return H + 0.05 * torch.randn(B, dim)

# --- Models ---

class HierResidual(nn.Module):
    """Flat residual: (state, goal, hint) → predicted final pos."""
    def __init__(self, in_dim, n_blocks=10):
        super().__init__()
        self.proj = nn.Linear(in_dim, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, GOAL_DIM)
    def forward(self, x, **kwargs):
        h = self.proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class SkillComposer(nn.Module):
    """
    Explicit skill library: each skill = learnable impulse (2,).
    Controller picks skill per step (Gumbel); apply dynamics in STATE space.
    """
    def __init__(self, in_dim, n_skills=N_SKILLS, n_steps=HORIZON_OOD):
        super().__init__()
        self.n_skills = n_skills
        self.n_steps = n_steps
        # learnable skill bank (impulses)
        self.skills = nn.Parameter(torch.randn(n_skills, 2) * 0.3)
        self.ctrl = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, n_skills),
        )
        self.step_embed = nn.Embedding(n_steps, n_skills)
        # optional residual refine
        self.refine = nn.Linear(STATE_DIM + GOAL_DIM, GOAL_DIM)

    def forward(self, x, state0, temperature=1.0):
        # x: B, in_dim includes state0[:4] already or we use state0
        B = state0.size(0)
        h = state0.clone()
        goal = x[:, STATE_DIM:STATE_DIM + GOAL_DIM]
        for t in range(self.n_steps):
            logits = self.ctrl(x) + self.step_embed.weight[t].unsqueeze(0)
            soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            # mixed impulse
            impulse = gate @ self.skills  # B, 2
            h = step_dynamics(h, impulse)
        pred = h[:, :2]
        # light residual to goal features
        pred = pred + 0.1 * self.refine(torch.cat([h, goal], dim=-1))
        return pred


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, seq_tr, s0_tr = make_batch(N_TRAIN, TRAIN_SEQS, seed)
    hint_tr = skill_hint(seq_tr)
    Xtr_full = torch.cat([Xtr, hint_tr], dim=-1)
    in_dim = Xtr_full.size(1)

    Xva, Yva, seq_va, s0_va = make_batch(N_VAL, TRAIN_SEQS, seed + 10)
    hint_va = skill_hint(seq_va)
    Xva_full = torch.cat([Xva, hint_va], dim=-1)

    Xood, Yood, seq_ood, s0_ood = make_batch(N_OOD, OOD_SEQS, seed + 20)
    hint_ood = skill_hint(seq_ood)
    Xood_full = torch.cat([Xood, hint_ood], dim=-1)

    def train(model, use_temp=False, use_state0=False):
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X = Xtr_full.to(DEVICE)
        Y = Ytr.to(DEVICE)
        S0 = s0_tr.to(DEVICE)
        Xv = Xva_full.to(DEVICE)
        Yv = Yva.to(DEVICE)
        S0v = s0_va.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS - 1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                opt.zero_grad()
                if use_state0:
                    pred = model(X[idx], S0[idx], temperature=temp) if use_temp else model(X[idx], S0[idx])
                else:
                    pred = model(X[idx])
                F.mse_loss(pred, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                if use_state0:
                    pred = model(Xv, S0v, temperature=TEMP_END) if use_temp else model(Xv, S0v)
                else:
                    pred = model(Xv)
                val = F.mse_loss(pred, Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def eval_ood(model, use_temp=False, use_state0=False):
        model.eval()
        with torch.no_grad():
            X = Xood_full.to(DEVICE)
            Y = Yood.to(DEVICE)
            S0 = s0_ood.to(DEVICE)
            if use_state0:
                pred = model(X, S0, temperature=TEMP_END) if use_temp else model(X, S0)
            else:
                pred = model(X)
            return F.mse_loss(pred, Y).item()

    out = {"seed": seed}
    hr = train(HierResidual(in_dim))
    out["HR"] = eval_ood(hr)

    sc = train(SkillComposer(in_dim, n_steps=HORIZON_OOD), use_temp=True, use_state0=True)
    out["SC"] = eval_ood(sc, use_temp=True, use_state0=True)
    out["win"] = out["SC"] < out["HR"]
    return out


print("Building Phase I control experiment...")
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
print("\n===== PHASE I CONTROL SUMMARY =====")
print(f"HR={hr:.4f} SC={sc:.4f} ratio={sc/hr:.3f} wins={wins}/3")

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
        "experiment": "Phase I Continuous control skill composition",
        "seeds": SEEDS,
        "summary": {"HR": hr, "SC": sc, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase I Control** skill library + sequential dynamics vs HR
- HR={hr:.4f} SC={sc:.4f} ratio={sc/hr:.3f} wins={wins}/3
- Outcome: **{outcome}**
- Family (3) continuous control — one bias: explicit skill compose in state space
""")
print(f"Saved {run_dir}")
