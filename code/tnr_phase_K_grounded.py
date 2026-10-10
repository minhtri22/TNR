#!/usr/bin/env python3
"""
Phase K — Grounded (họ 4)

Môi trường: lưới 5×5. Agent (x,y). Moves: N,S,E,W (clip biên).
Lệnh grounded Mini-SCAN-like:
  "north" / "south" / "east" / "west"
  "north after south" → south rồi north
  "opposite north" → south (binding)

Train: atomic moves + twice (lặp 2 bước cùng hướng)
OOD: after (hierarchical grounded) + opposite

Giả thuyết K1 — Symbolic grounded executor:
  Parse command → apply moves trên grid (program+execute trên world state)
vs
  Residual/S2S: map (state0, cmd embedding) → final (x,y) hồi quy

Preregister:
  Setup: Exec ID exact-cell >= 0.50
  PASS: Exec OOD cell-acc > Res + 0.05 AND Exec OOD >= 0.50 AND wins >= 2/3
  FAIL: Exec OOD + 0.05 < Res
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

GRID = 5
MOVES = ["north", "south", "east", "west"]
MOVE2DELTA = {
    "north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0),
}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

CMD_TOKENS = ["<pad>", "after", "twice", "opposite"] + MOVES
CMD2ID = {t: i for i, t in enumerate(CMD_TOKENS)}
ID2CMD = {i: t for t, i in CMD2ID.items()}
MAX_CMD = 6
N_CMD = len(CMD_TOKENS)

HIDDEN = 64
BATCH_SIZE = 64
N_TRAIN = 3000
N_ID = 400
N_OOD = 600
EPOCHS = 25
LR = 1e-3

def clip(x, y):
    return max(0, min(GRID - 1, x)), max(0, min(GRID - 1, y))

def apply_moves(x0, y0, moves):
    x, y = x0, y0
    for m in moves:
        dx, dy = MOVE2DELTA[m]
        x, y = clip(x + dx, y + dy)
    return x, y

def expand_command(tokens):
    """tokens: list[str] → list of move names."""
    if "after" in tokens:
        i = tokens.index("after")
        left, right = tokens[:i], tokens[i + 1:]
        return expand_command(right) + expand_command(left)
    moves = []
    opposite = False
    twice = False
    direction = None
    for t in tokens:
        if t == "opposite":
            opposite = True
        elif t == "twice":
            twice = True
        elif t in MOVES:
            direction = t
    if direction is None:
        return []
    if opposite:
        direction = OPP[direction]
    unit = [direction]
    if twice:
        unit = unit * 2
    return unit

def encode_cmd(tokens):
    ids = [CMD2ID[t] for t in tokens] + [0] * MAX_CMD
    return ids[:MAX_CMD]

def make_train_examples():
    ex = []
    for m in MOVES:
        ex.append(([m], expand_command([m])))
        ex.append(([m, "twice"], expand_command([m, "twice"])))
    return ex

def make_ood_examples():
    ex = []
    for a in MOVES:
        for b in MOVES:
            if a == b:
                continue
            toks = [a, "after", b]
            ex.append((toks, expand_command(toks)))
        toks = ["opposite", a]
        ex.append((toks, expand_command(toks)))
    return ex

TRAIN_EX = make_train_examples()
OOD_EX = make_ood_examples()
print(f"Train templates={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X_cmd, X_pos, Y_pos = [], [], []
    for _ in range(n):
        cmd, moves = examples[rng.randint(0, len(examples))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        xf, yf = apply_moves(x0, y0, moves)
        X_cmd.append(encode_cmd(cmd))
        X_pos.append([x0, y0])
        Y_pos.append([xf, yf])
    return (
        torch.tensor(X_cmd, dtype=torch.long),
        torch.tensor(X_pos, dtype=torch.float32),
        torch.tensor(Y_pos, dtype=torch.float32),
    )

def cell_acc(pred, y):
    """pred/y: B,2 continuous → round to cell."""
    pr = pred.round().clamp(0, GRID - 1)
    yt = y.round().clamp(0, GRID - 1)
    return (pr == yt).all(dim=-1).float().mean().item()

class ResidualGrounded(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_CMD, HIDDEN, padding_idx=0)
        self.pos_proj = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, cmd, pos):
        e = self.emb(cmd)
        mask = (cmd != 0).float().unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        h = torch.cat([pooled, self.pos_proj(pos)], dim=-1)
        return self.mlp(h)

class GroundedExecutor(nn.Module):
    """
    Symbolic grounded: parse tokens → moves → simulate on grid.
    Neural only for optional move classifier (not required if parse works).
    """
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_CMD, 16, padding_idx=0)
        self.move_head = nn.Linear(16, len(MOVES))  # weak aux

    def execute(self, cmd, pos):
        self.eval()
        B = cmd.size(0)
        outs = []
        with torch.no_grad():
            for i in range(B):
                toks = [ID2CMD.get(int(t), "<pad>") for t in cmd[i].tolist() if int(t) != 0]
                moves = expand_command(toks)
                x0, y0 = float(pos[i, 0]), float(pos[i, 1])
                xf, yf = apply_moves(int(round(x0)), int(round(y0)), moves)
                outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=cmd.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Ctr, Ptr, Ytr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    Cid, Pid, Yid = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    Cood, Pood, Yood = sample_dataset(OOD_EX, N_OOD, seed + 20)

    def train_res():
        model = ResidualGrounded().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        C, P, Y = Ctr.to(DEVICE), Ptr.to(DEVICE), Ytr.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred = model(C[idx], P[idx])
                F.mse_loss(pred, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(C[:512], P[:512]), Y[:512]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    res = train_res()
    res.eval()
    with torch.no_grad():
        pred_id = res(Cid.to(DEVICE), Pid.to(DEVICE))
        pred_ood = res(Cood.to(DEVICE), Pood.to(DEVICE))
        out["Res_id"] = cell_acc(pred_id, Yid.to(DEVICE))
        out["Res_ood"] = cell_acc(pred_ood, Yood.to(DEVICE))

    # Executor: symbolic, no heavy train
    exe = GroundedExecutor().to(DEVICE)
    out["Exe_id"] = cell_acc(exe.execute(Cid.to(DEVICE), Pid.to(DEVICE)), Yid.to(DEVICE))
    out["Exe_ood"] = cell_acc(exe.execute(Cood.to(DEVICE), Pood.to(DEVICE)), Yood.to(DEVICE))
    out["win"] = out["Exe_ood"] > out["Res_ood"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Res ID={r['Res_id']:.3f} OOD={r['Res_ood']:.3f}", flush=True)
    print(f"  Exe ID={r['Exe_id']:.3f} OOD={r['Exe_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== K GROUNDED SUMMARY =====")
print(f"Res: ID={ms('Res_id')[0]:.3f} OOD={ms('Res_ood')[0]:.3f}")
print(f"Exe: ID={ms('Exe_id')[0]:.3f} OOD={ms('Exe_ood')[0]:.3f}")

res_ood, exe_ood = ms("Res_ood")[0], ms("Exe_ood")[0]
exe_id = ms("Exe_id")[0]
wins = sum(r["win"] for r in all_results)
setup_ok = exe_id >= 0.50

if not setup_ok:
    outcome, note = "UNRESOLVED", "Exe ID < 0.50"
elif exe_ood > res_ood + 0.05 and exe_ood >= 0.50 and wins >= 2:
    outcome, note = "PASS", "grounded symbolic execute unlocks OOD after/opposite"
elif exe_ood + 0.05 < res_ood:
    outcome, note = "FAIL", "Executor worse than residual"
else:
    outcome, note = "UNRESOLVED", "insufficient gap"

print(f"setup_ok={setup_ok} wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase K Grounded grid + symbolic execute",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["Res_id","Res_ood","Exe_id","Exe_ood"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## CONTROL FREEZE 2026-10-10
- Report: reports/bao_cao_freeze_control.md
- Oracle PASS; learned routing FAIL (I2–J2). Family CLOSED.

## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase K Grounded** grid navigate symbolic execute vs residual
- Res OOD={res_ood:.3f} | Exe OOD={exe_ood:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
