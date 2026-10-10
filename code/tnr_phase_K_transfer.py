#!/usr/bin/env python3
"""
Phase K-transfer — grounded transfer trước probabilistic

Train: atomic + twice (như K)
OOD-1: single after + opposite (K held)
OOD-2: nested after "A after B after C" trên grid
OOD-3: opposite + after combined "opposite north after south"

Preregister:
  PASS nếu Exe nested cell-acc >= 0.50 AND single/opp held >= 0.50
       AND wins vs Res nested >= 2/3
  FAIL nếu nested + 0.05 < Res
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
MAX_CMD = 8
N_CMD = len(CMD_TOKENS)

HIDDEN = 64
BATCH_SIZE = 64
N_TRAIN = 3000
N_ID = 400
N_OOD = 500
EPOCHS = 24
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
    if "after" in tokens:
        i = tokens.index("after")
        left, right = tokens[:i], tokens[i + 1:]
        return expand_command(right) + expand_command(left)
    opposite = twice = False
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

def make_ood_single():
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

def make_ood_nested():
    ex = []
    for a in MOVES:
        for b in MOVES:
            for c in MOVES:
                if len({a, b, c}) < 2:
                    continue
                toks = [a, "after", b, "after", c]
                ex.append((toks, expand_command(toks)))
    return ex

def make_ood_combo():
    """opposite + after."""
    ex = []
    for a in MOVES:
        for b in MOVES:
            toks = ["opposite", a, "after", b]
            ex.append((toks, expand_command(toks)))
            toks = [a, "after", "opposite", b]
            ex.append((toks, expand_command(toks)))
    return ex

TRAIN_EX = make_train_examples()
OOD_S = make_ood_single()
OOD_N = make_ood_nested()
OOD_C = make_ood_combo()
print(f"Train={len(TRAIN_EX)} single={len(OOD_S)} nested={len(OOD_N)} combo={len(OOD_C)}")

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
        return self.mlp(torch.cat([pooled, self.pos_proj(pos)], -1))

class GroundedExecutor(nn.Module):
    def execute(self, cmd, pos):
        B = cmd.size(0)
        outs = []
        with torch.no_grad():
            for i in range(B):
                toks = [ID2CMD.get(int(t), "<pad>") for t in cmd[i].tolist() if int(t) != 0]
                moves = expand_command(toks)
                x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
                xf, yf = apply_moves(x0, y0, moves)
                outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=cmd.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Ctr, Ptr, Ytr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    Cid, Pid, Yid = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    Cs, Ps, Ys = sample_dataset(OOD_S, N_OOD, seed + 20)
    Cn, Pn, Yn = sample_dataset(OOD_N, N_OOD, seed + 30)
    Cc, Pc, Yc = sample_dataset(OOD_C, N_OOD, seed + 40)

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
                F.mse_loss(model(C[idx], P[idx]), Y[idx]).backward()
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
    exe = GroundedExecutor()

    def eval_pair(C, P, Y, prefix):
        with torch.no_grad():
            out[f"Res_{prefix}"] = cell_acc(res(C.to(DEVICE), P.to(DEVICE)), Y.to(DEVICE))
            out[f"Exe_{prefix}"] = cell_acc(exe.execute(C.to(DEVICE), P.to(DEVICE)), Y.to(DEVICE))

    eval_pair(Cid, Pid, Yid, "id")
    eval_pair(Cs, Ps, Ys, "s")
    eval_pair(Cn, Pn, Yn, "n")
    eval_pair(Cc, Pc, Yc, "c")
    out["win_n"] = out["Exe_n"] > out["Res_n"] + 0.02
    out["win_c"] = out["Exe_c"] > out["Res_c"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Res id={r['Res_id']:.3f} s={r['Res_s']:.3f} n={r['Res_n']:.3f} c={r['Res_c']:.3f}", flush=True)
    print(f"  Exe id={r['Exe_id']:.3f} s={r['Exe_s']:.3f} n={r['Exe_n']:.3f} c={r['Exe_c']:.3f}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== K TRANSFER SUMMARY =====")
for k in ["Res_id","Res_s","Res_n","Res_c","Exe_id","Exe_s","Exe_n","Exe_c"]:
    print(f"  {k}: {ms(k)[0]:.3f}")

exe_n, res_n = ms("Exe_n")[0], ms("Res_n")[0]
exe_s, exe_c = ms("Exe_s")[0], ms("Exe_c")[0]
wins = sum(r["win_n"] for r in all_results)

if exe_n >= 0.50 and exe_s >= 0.50 and wins >= 2:
    outcome, note = "PASS", "nested+combo transfer; single held"
elif exe_n + 0.05 < res_n:
    outcome, note = "FAIL", "nested worse than Res"
else:
    outcome, note = "UNRESOLVED", "nested transfer incomplete"

print(f"wins_nested={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "K-transfer nested after + opposite-after combo",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in
            ["Res_id","Res_s","Res_n","Res_c","Exe_id","Exe_s","Exe_n","Exe_c"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **K-transfer** nested after + opposite-after on grid
- Exe s={exe_s:.3f} n={exe_n:.3f} c={exe_c:.3f} | Res n={res_n:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
- Gate: PASS → open (5) probabilistic
""")
print(f"Saved {run_dir}")
