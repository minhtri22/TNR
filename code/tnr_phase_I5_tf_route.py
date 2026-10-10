#!/usr/bin/env python3
"""
Phase I5 — Control: teacher-forced routing train → free OOD

Giả thuyết:
  I3/I4: oracle path PASS, Gumbel end-to-end FAIL.
  I5: train controller với CE vào true skill ids (teacher force);
      OOD dùng argmax/soft routing tự do + TRUE frozen skills.
  → Nếu OOD PASS: routing học được khi có label path.
  → Nếu FAIL: free OOD routing vẫn không generalize (cần oracle cả test).

Preregister:
  PASS: SC_OOD < 0.95*HR AND wins >= 2/3
  FAIL: SC_OOD > 1.05*HR
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
HORIZON = 3
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
    skills = [torch.tensor([ (0.8+0.15*(i%3))*np.cos(a), (0.8+0.15*(i%3))*np.sin(a) ], dtype=torch.float32)
              for i, a in enumerate(angles)]
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
    return torch.stack([x + DT*vx, y + DT*vy, vx, vy], dim=-1)

def rollout(state0, skill_ids):
    s = state0.clone()
    for k in skill_ids:
        s = step_dynamics(s, TRUE_SKILLS[k%N_SKILLS].unsqueeze(0).expand(s.size(0), -1))
    return s[:, :2]

def ordered_hint(seq, max_len=3):
    parts = []
    for t in range(max_len):
        parts.append(SKILL_KEYS[seq[t]%N_SKILLS] if t < len(seq) else torch.zeros(HINT_DIM))
    return torch.cat(parts, 0)

def make_batch(n, seq_dict, seed):
    g = torch.Generator().manual_seed(42 + seed)
    state0 = torch.randn(n, STATE_DIM, generator=g) * 0.5
    keys = list(seq_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, 2)
    hints = torch.zeros(n, HORIZON * HINT_DIM)
    seq_ids = torch.full((n, HORIZON), -1, dtype=torch.long)
    for i in range(n):
        seq = seq_dict[keys[cases[i].item()]]
        Y[i] = rollout(state0[i:i+1], seq).squeeze(0)
        hints[i] = ordered_hint(seq, HORIZON)
        for t, k in enumerate(seq):
            if t < HORIZON:
                seq_ids[i, t] = k % N_SKILLS
    Y = Y + 0.02 * torch.randn(n, 2, generator=g)
    hints = hints + 0.05 * torch.randn_like(hints)
    X = torch.cat([state0, hints], -1)
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
    def forward(self, x, **kw):
        h = self.proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class TFSkillComposer(nn.Module):
    """Frozen true skills; controller trained with CE on true ids; free argmax at eval."""
    def __init__(self, in_dim):
        super().__init__()
        self.register_buffer("skills", TRUE_SKILLS.clone())
        self.ctrl = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, N_SKILLS))
        self.step_embed = nn.Embedding(HORIZON, N_SKILLS)

    def logits_at(self, x, t):
        return self.ctrl(x) + self.step_embed.weight[t].unsqueeze(0)

    def forward(self, x, state0, seq_ids=None, teacher_force=False):
        h = state0.clone()
        route_logits = []
        for t in range(HORIZON):
            logits = self.logits_at(x, t)
            route_logits.append(logits)
            if teacher_force and seq_ids is not None:
                ids = seq_ids[:, t]
                active = ids >= 0
                safe = ids.clamp(min=0)
                impulse = self.skills[safe]
                h_next = step_dynamics(h, impulse)
                h = torch.where(active.unsqueeze(-1), h_next, h)
            else:
                # free routing
                gate = F.softmax(logits, dim=-1)
                impulse = gate @ self.skills
                # skip if this step unused in short seq: still apply (train L=2 uses T=0,1)
                h = step_dynamics(h, impulse)
        return h[:, :2], route_logits


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

    def train_tf():
        model = TFSkillComposer(in_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y, S0, SID = Xtr.to(DEVICE), Ytr.to(DEVICE), s0_tr.to(DEVICE), seq_tr.to(DEVICE)
        Xv, Yv, S0v, SIDv = Xva.to(DEVICE), Yva.to(DEVICE), s0_va.to(DEVICE), seq_va.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred, route_logits = model(X[idx], S0[idx], SID[idx], teacher_force=True)
                loss_mse = F.mse_loss(pred, Y[idx])
                loss_ce = 0
                n_ce = 0
                for t, logits in enumerate(route_logits):
                    ids = SID[idx, t]
                    active = ids >= 0
                    if active.any():
                        loss_ce = loss_ce + F.cross_entropy(logits[active], ids[active])
                        n_ce += 1
                loss = loss_mse + 0.5 * (loss_ce / max(n_ce, 1))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred, _ = model(Xv, S0v, SIDv, teacher_force=True)
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

    tf = train_tf()
    tf.eval()
    with torch.no_grad():
        # free OOD (no teacher force)
        pred, _ = tf(Xood.to(DEVICE), s0_ood.to(DEVICE), teacher_force=False)
        out["SC_free"] = F.mse_loss(pred, Yood.to(DEVICE)).item()
        # oracle-style TF on OOD (upper)
        pred_tf, _ = tf(Xood.to(DEVICE), s0_ood.to(DEVICE), seq_ood.to(DEVICE), teacher_force=True)
        out["SC_tf"] = F.mse_loss(pred_tf, Yood.to(DEVICE)).item()
    out["win"] = out["SC_free"] < out["HR"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} SC_free={r['SC_free']:.4f} SC_tf={r['SC_tf']:.4f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, free, tfm = ms("HR")[0], ms("SC_free")[0], ms("SC_tf")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== I5 TF-ROUTE SUMMARY =====")
print(f"HR={hr:.4f} SC_free={free:.4f} SC_tf={tfm:.4f}")
print(f"free/HR={free/hr:.3f} wins={wins}/3")

if free < hr * 0.95 and wins >= 2:
    outcome = "PASS"
elif free > hr * 1.05:
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
        "experiment": "Phase I5 TF train free OOD routing",
        "seeds": SEEDS,
        "summary": {"HR": hr, "SC_free": free, "SC_tf": tfm, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase I5** teacher-forced route train → free OOD
- HR={hr:.4f} SC_free={free:.4f} SC_tf={tfm:.4f} wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
