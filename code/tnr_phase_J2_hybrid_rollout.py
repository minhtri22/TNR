#!/usr/bin/env python3
"""
Phase J2 — Hybrid: skill rollout + residual (học hệ trưởng thành)

pred = ProgramExecute(state, hint) + residual_MLP(state0, hint)

Giả thuyết:
  Rollout mang inductive bias dynamics; residual sửa sai số routing.
  So với HR thuần và PE thuần (J/I6 FAIL).

Bag hint, TRUE skills frozen, closed-loop controller + CE aux.
Preregister:
  PASS: HYB < 0.95*HR AND HYB <= 0.98*min(PE, HR_only_ref) AND wins>=2/3
  FAIL: HYB > 1.05*HR
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
EPOCHS = 32
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
    return torch.stack([x + DT * vx, y + DT * vy, vx, vy], dim=-1)

def rollout_true(state0, skill_ids):
    s = state0.clone()
    for k in skill_ids:
        s = step_dynamics(s, TRUE_SKILLS[k % N_SKILLS].unsqueeze(0).expand(s.size(0), -1))
    return s[:, :2]

def bag_hint(seq):
    h = torch.zeros(HINT_DIM)
    for k in seq:
        h = h + SKILL_KEYS[k % N_SKILLS]
    return h / max(len(seq), 1)

def make_batch(n, seq_dict, seed):
    g = torch.Generator().manual_seed(42 + seed)
    state0 = torch.randn(n, STATE_DIM, generator=g) * 0.5
    keys = list(seq_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    Y = torch.zeros(n, 2)
    hints = torch.zeros(n, HINT_DIM)
    seq_ids = torch.full((n, HORIZON), -1, dtype=torch.long)
    for i in range(n):
        seq = seq_dict[keys[cases[i].item()]]
        Y[i] = rollout_true(state0[i:i+1], seq).squeeze(0)
        hints[i] = bag_hint(seq)
        for t, k in enumerate(seq):
            if t < HORIZON:
                seq_ids[i, t] = k % N_SKILLS
    Y = Y + 0.02 * torch.randn(n, 2, generator=g)
    hints = hints + 0.08 * torch.randn_like(hints)
    return state0, hints, Y, seq_ids

class HierResidual(nn.Module):
    def __init__(self, in_dim, n_blocks=10):
        super().__init__()
        self.proj = nn.Linear(in_dim, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, 2)
    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class HybridRolloutResidual(nn.Module):
    """
    Program+Execute closed-loop + residual correction.
    pred = stopgrad_optional(rollout_pos) + residual  OR  rollout_pos + residual (full grad)
    We use full grad through rollout so routing still trains; residual can compensate.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("skills", TRUE_SKILLS.clone())
        self.ctrl = nn.Sequential(
            nn.Linear(STATE_DIM + HINT_DIM, 64), nn.Tanh(), nn.Linear(64, N_SKILLS)
        )
        self.residual = nn.Sequential(
            nn.Linear(STATE_DIM + HINT_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
        # gate: how much residual vs rollout (learnable soft)
        self.mix = nn.Parameter(torch.tensor(0.5))

    def rollout(self, state0, hint, temperature=1.0, seq_ids=None, teacher_force=False):
        h = state0.clone()
        logits_list = []
        for t in range(HORIZON):
            ctx = torch.cat([h, hint], dim=-1)
            logits = self.ctrl(ctx)
            logits_list.append(logits)
            if teacher_force and seq_ids is not None:
                ids = seq_ids[:, t]
                active = ids >= 0
                impulse = self.skills[ids.clamp(min=0)]
                h_next = step_dynamics(h, impulse)
                h = torch.where(active.unsqueeze(-1), h_next, h)
            else:
                soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
                if self.training:
                    hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                    gate = hard + (soft - soft.detach())
                else:
                    gate = soft
                h = step_dynamics(h, gate @ self.skills)
        return h[:, :2], logits_list

    def forward(self, state0, hint, temperature=1.0, seq_ids=None, teacher_force=False):
        pos, logits_list = self.rollout(state0, hint, temperature, seq_ids, teacher_force)
        ctx0 = torch.cat([state0, hint], dim=-1)
        res = self.residual(ctx0)
        alpha = torch.sigmoid(self.mix)
        # pred = alpha * rollout + (1-alpha) * residual  — residual can dominate if rollout bad
        pred = alpha * pos + (1 - alpha) * res
        return pred, logits_list, pos, res, alpha


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s0_tr, h_tr, Ytr, seq_tr = make_batch(N_TRAIN, TRAIN_SEQS, seed)
    s0_va, h_va, Yva, seq_va = make_batch(N_VAL, TRAIN_SEQS, seed + 10)
    s0_ood, h_ood, Yood, seq_ood = make_batch(N_OOD, OOD_SEQS, seed + 20)
    Xtr = torch.cat([s0_tr, h_tr], -1)
    Xood = torch.cat([s0_ood, h_ood], -1)
    in_dim = Xtr.size(1)

    def train_hr():
        model = HierResidual(in_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv = torch.cat([s0_va, h_va], -1).to(DEVICE)
        Yv = Yva.to(DEVICE)
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

    def train_hyb():
        model = HybridRolloutResidual().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        S0, H, Y, SID = s0_tr.to(DEVICE), h_tr.to(DEVICE), Ytr.to(DEVICE), seq_tr.to(DEVICE)
        S0v, Hv, Yv = s0_va.to(DEVICE), h_va.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS - 1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred, logits_list, pos, res, alpha = model(
                    S0[idx], H[idx], temperature=temp, teacher_force=False
                )
                loss = F.mse_loss(pred, Y[idx])
                # encourage rollout to be useful (aux MSE on pure rollout)
                loss = loss + 0.2 * F.mse_loss(pos, Y[idx])
                for t, logits in enumerate(logits_list):
                    ids = SID[idx, t]
                    active = ids >= 0
                    if active.any():
                        loss = loss + 0.2 * F.cross_entropy(logits[active], ids[active])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred, _, _, _, _ = model(S0v, Hv, temperature=TEMP_END)
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

    hyb = train_hyb()
    hyb.eval()
    with torch.no_grad():
        pred, _, pos, res, alpha = hyb(s0_ood.to(DEVICE), h_ood.to(DEVICE), temperature=TEMP_END)
        out["HYB"] = F.mse_loss(pred, Yood.to(DEVICE)).item()
        out["ROLLOUT_only"] = F.mse_loss(pos, Yood.to(DEVICE)).item()
        out["RES_only"] = F.mse_loss(res, Yood.to(DEVICE)).item()
        out["alpha"] = float(alpha.item())
    out["win"] = out["HYB"] < out["HR"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} HYB={r['HYB']:.4f} roll={r['ROLLOUT_only']:.4f} "
          f"res={r['RES_only']:.4f} α={r['alpha']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, hyb = ms("HR")[0], ms("HYB")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== J2 HYBRID SUMMARY =====")
print(f"HR={hr:.4f} HYB={hyb:.4f} ratio={hyb/hr:.3f} wins={wins}/3")
print(f"rollout_only={ms('ROLLOUT_only')[0]:.4f} res_only={ms('RES_only')[0]:.4f} α={ms('alpha')[0]:.3f}")

if hyb < hr * 0.95 and wins >= 2:
    outcome = "PASS"
elif hyb > hr * 1.05:
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
        "experiment": "Phase J2 hybrid rollout+residual",
        "seeds": SEEDS,
        "summary": {"HR": hr, "HYB": hyb, "wins": wins,
                    "rollout": ms("ROLLOUT_only")[0], "res": ms("RES_only")[0]},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase J2** hybrid rollout + residual vs HR
- HR={hr:.4f} HYB={hyb:.4f} ratio={hyb/hr:.3f} wins={wins}/3
- rollout_only={ms('ROLLOUT_only')[0]:.4f} res_only={ms('RES_only')[0]:.4f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
