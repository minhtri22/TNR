#!/usr/bin/env python3
"""
Phase L2 — Probabilistic: mixture + Gaussian NLL (calibrated)

Giả thuyết:
  L FAIL vì tối ưu MSE point-estimate → residual đủ.
  L2: pred mean từ mixture; học log-variance; loss = Gaussian NLL.
  Đo OOD bằng NLL và MSE (preregister chính: MSE để so HR công bằng).

Preregister (MSE, cùng L):
  PASS: Mix MSE < 0.95*HR AND wins >= 2/3
  FAIL: Mix MSE > 1.05*HR
  else UNRESOLVED
Ghi thêm NLL so sánh (không dùng để đổi PASS nếu MSE FAIL).
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

STATE_DIM = 2
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

SKILLS = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]], dtype=torch.float32)
SKILL_KEYS = F.normalize(torch.randn(N_SKILLS, HINT_DIM), dim=-1)

TRAIN_PROGS = [[0,2],[1,3],[0,1],[2,3],[0,0],[2,2],[1,1],[3,3]]
OOD_PROGS = [[0,2,1],[1,3,0],[0,1,2],[2,3,1],[0,0,2],[2,2,0],[1,1,3],[3,3,1]]

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
    return state0, hints, Y

def gaussian_nll(mean, log_var, y):
    # log_var: B,2 or B,1
    var = log_var.exp().clamp(min=1e-4)
    return 0.5 * (((y - mean) ** 2) / var + log_var + np.log(2 * np.pi)).sum(-1).mean()

class HierResidual(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(8)
        ])
        self.out = nn.Linear(HIDDEN, STATE_DIM)
        self.log_var = nn.Linear(HIDDEN, STATE_DIM)
    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h), self.log_var(h)

class MixtureNLL(nn.Module):
    def __init__(self, library):
        super().__init__()
        self.library = library
        self.score = nn.Sequential(
            nn.Linear(STATE_DIM + HINT_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, len(library)),
        )
        self.log_var = nn.Sequential(
            nn.Linear(STATE_DIM + HINT_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, STATE_DIM),
        )
    def forward(self, state0, hint):
        ctx = torch.cat([state0, hint], dim=-1)
        q = F.softmax(self.score(ctx), dim=-1)
        outs = [execute(state0, prog) for prog in self.library]
        stack = torch.stack(outs, dim=1)
        mean = (q.unsqueeze(-1) * stack).sum(1)
        log_var = self.log_var(ctx)
        return mean, log_var, q


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s0_tr, h_tr, Ytr = make_batch(N_TRAIN, TRAIN_PROGS, seed)
    s0_va, h_va, Yva = make_batch(N_VAL, TRAIN_PROGS, seed + 10)
    s0_ood, h_ood, Yood = make_batch(N_OOD, OOD_PROGS, seed + 20)
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
                mean, lv = model(X[idx])
                gaussian_nll(mean, lv, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                mean, lv = model(Xv)
                val = gaussian_nll(mean, lv, Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_mix():
        model = MixtureNLL(TRAIN_PROGS + OOD_PROGS).to(DEVICE)  # full lib upper for NLL
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
                mean, lv, q = model(S0[idx], H[idx])
                loss = gaussian_nll(mean, lv, Y[idx])
                loss = loss - 0.01 * (q * (q + 1e-8).log()).sum(-1).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                mean, lv, _ = model(S0v, Hv)
                val = gaussian_nll(mean, lv, Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    hr = train_hr()
    mix = train_mix()
    hr.eval()
    mix.eval()
    with torch.no_grad():
        hm, hlv = hr(Xood.to(DEVICE))
        mm, mlv, _ = mix(s0_ood.to(DEVICE), h_ood.to(DEVICE))
        Y = Yood.to(DEVICE)
        out["HR_mse"] = F.mse_loss(hm, Y).item()
        out["MIX_mse"] = F.mse_loss(mm, Y).item()
        out["HR_nll"] = gaussian_nll(hm, hlv, Y).item()
        out["MIX_nll"] = gaussian_nll(mm, mlv, Y).item()
    out["win"] = out["MIX_mse"] < out["HR_mse"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR mse={r['HR_mse']:.4f} nll={r['HR_nll']:.4f} | "
          f"MIX mse={r['MIX_mse']:.4f} nll={r['MIX_nll']:.4f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr_m, mix_m = ms("HR_mse")[0], ms("MIX_mse")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== L2 NLL SUMMARY =====")
print(f"HR mse={hr_m:.4f} MIX mse={mix_m:.4f} ratio={mix_m/hr_m:.3f} wins={wins}/3")
print(f"HR nll={ms('HR_nll')[0]:.4f} MIX nll={ms('MIX_nll')[0]:.4f}")

if mix_m < hr_m * 0.95 and wins >= 2:
    outcome = "PASS"
elif mix_m > hr_m * 1.05:
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
        "experiment": "Phase L2 mixture + Gaussian NLL",
        "seeds": SEEDS,
        "summary": {"HR_mse": hr_m, "MIX_mse": mix_m, "wins": wins,
                    "HR_nll": ms("HR_nll")[0], "MIX_nll": ms("MIX_nll")[0]},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase L2** mixture + Gaussian NLL vs residual NLL
- HR mse={hr_m:.4f} MIX mse={mix_m:.4f} ratio={mix_m/hr_m:.3f} wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
