#!/usr/bin/env python3
"""
Phase I4 — Control upper bound: teacher-forced / oracle skill IDs

Giả thuyết:
  I2/I3 FAIL vì Gumbel controller không chọn đúng skill.
  I4: đưa đúng skill sequence (oracle) vào model; chỉ apply dynamics
      với TRUE skills theo id — không học routing.
  → Nếu MSE ≈ noise floor: dynamics+skills đúng, routing là bottleneck.
  → Nếu vẫn kém HR: residual đang shortcut, task chưa routing-critical.

Cùng data I2/I3. Oracle model không dùng Gumbel.

Preregister:
  PASS oracle nếu OOD MSE < 0.95*HR AND wins >= 2/3
  (upper bound: composition đúng khi biết path)
  FAIL nếu oracle > 1.05*HR
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

STATE_DIM = 4
N_SKILLS = 6
HINT_DIM = 8
HIDDEN = 64
HORIZON_OOD = 3
DT = 0.4
DAMP = 0.85
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 30
LR = 1e-3

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

def ordered_hint(seq, max_len=3):
    parts = []
    for t in range(max_len):
        if t < len(seq):
            parts.append(SKILL_KEYS[seq[t] % N_SKILLS])
        else:
            parts.append(torch.zeros(HINT_DIM))
    return torch.cat(parts, dim=0)

def make_batch(n, seq_dict, seed, max_len=3):
    g = torch.Generator().manual_seed(42 + seed)
    state0 = torch.randn(n, STATE_DIM, generator=g) * 0.5
    keys = list(seq_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, 2)
    hints = torch.zeros(n, max_len * HINT_DIM)
    seq_ids = torch.zeros(n, max_len, dtype=torch.long)
    for i in range(n):
        seq = seq_dict[keys[cases[i].item()]]
        Y[i] = rollout(state0[i:i+1], seq).squeeze(0)
        hints[i] = ordered_hint(seq, max_len)
        for t, k in enumerate(seq):
            if t < max_len:
                seq_ids[i, t] = k % N_SKILLS
        # pad remaining steps with last skill (or 0) — oracle uses real length via mask
        for t in range(len(seq), max_len):
            seq_ids[i, t] = -1  # mask
    Y = Y + 0.02 * torch.randn(n, 2, generator=g)
    hints = hints + 0.05 * torch.randn_like(hints)
    X = torch.cat([state0, hints], dim=-1)
    return X, Y, state0, seq_ids

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

class OracleSkillRollout(nn.Module):
    """
    No learned routing: apply TRUE skills by oracle sequence ids.
    Optional tiny learnable scale on impulses (default frozen pure oracle).
    """
    def __init__(self, learn_scale=False):
        super().__init__()
        self.register_buffer("skills", TRUE_SKILLS.clone())
        self.learn_scale = learn_scale
        if learn_scale:
            self.scale = nn.Parameter(torch.ones(1))

    def forward(self, state0, seq_ids):
        # seq_ids: B, T with -1 = skip
        h = state0.clone()
        B, T = seq_ids.shape
        for t in range(T):
            ids = seq_ids[:, t]
            active = ids >= 0
            if not active.any():
                continue
            # gather skills
            safe_ids = ids.clamp(min=0)
            impulse = self.skills[safe_ids]
            if self.learn_scale:
                impulse = impulse * self.scale
            # only update active rows
            h_next = step_dynamics(h, impulse)
            h = torch.where(active.unsqueeze(-1), h_next, h)
        return h[:, :2]


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, s0_tr, seq_tr = make_batch(N_TRAIN, TRAIN_SEQS, seed)
    Xva, Yva, s0_va, seq_va = make_batch(N_VAL, TRAIN_SEQS, seed + 10)
    Xood, Yood, s0_ood, seq_ood = make_batch(N_OOD, OOD_SEQS, seed + 20)
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

    out = {"seed": seed}
    hr = train_hr()
    hr.eval()
    with torch.no_grad():
        out["HR"] = F.mse_loss(hr(Xood.to(DEVICE)), Yood.to(DEVICE)).item()

    # Oracle: no training needed (pure dynamics)
    oracle = OracleSkillRollout(learn_scale=False).to(DEVICE)
    oracle.eval()
    with torch.no_grad():
        pred = oracle(s0_ood.to(DEVICE), seq_ood.to(DEVICE))
        out["ORACLE"] = F.mse_loss(pred, Yood.to(DEVICE)).item()
        # noise floor check on train
        pred_tr = oracle(s0_tr.to(DEVICE), seq_tr.to(DEVICE))
        out["ORACLE_tr"] = F.mse_loss(pred_tr, Ytr.to(DEVICE)).item()

    out["win"] = out["ORACLE"] < out["HR"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} ORACLE={r['ORACLE']:.4f} (train={r['ORACLE_tr']:.4f}) win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, ora = ms("HR")[0], ms("ORACLE")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== I4 ORACLE SUMMARY =====")
print(f"HR={hr:.4f} ORACLE={ora:.4f} ratio={ora/hr:.3f} wins={wins}/3")
print(f"ORACLE train={ms('ORACLE_tr')[0]:.4f}")

if ora < hr * 0.95 and wins >= 2:
    outcome = "PASS"
elif ora > hr * 1.05:
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
        "experiment": "Phase I4 Oracle skill sequence upper bound",
        "seeds": SEEDS,
        "summary": {"HR": hr, "ORACLE": ora, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase I4 Oracle** true skills + true sequence (no learned routing)
- HR={hr:.4f} ORACLE={ora:.4f} ratio={ora/hr:.3f} wins={wins}/3
- Outcome: **{outcome}**
- Closes control diagnosis: oracle vs residual on same task
""")
print(f"Saved {run_dir}")
