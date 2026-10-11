#!/usr/bin/env python3
"""
Phase O4b — Different OOD for state-mid sequential multi-turn

Train: sequential cmds, pairs from TRAIN_DIR_PAIRS only, paraphrase set A
OOD-1: held-out direction pairs (never as (m1,m2) in train)
OOD-2: all pairs but paraphrase set B only
OOD-3: held-out pairs + paraphrase B (hardest)

Preregister (primary = OOD-3 hard):
  PASS: Prog_hard > Flat_hard + 0.05 AND Prog_hard >= 0.45 AND wins >= 2/3
  FAIL: Prog_hard + 0.05 < Flat_hard
  else UNRESOLVED
Also report OOD-1, OOD-2.
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
MOVE2DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE2ID = {m: i for i, m in enumerate(MOVES)}

# Hold out some direction pairs for OOD
ALL_PAIRS = [(a, b) for a in MOVES for b in MOVES]
OOD_PAIRS = [("north", "east"), ("south", "west"), ("east", "south"), ("west", "north"),
             ("north", "north"), ("east", "east")]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in OOD_PAIRS]

WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "head", "step", "travel", "proceed",
         "north", "south", "east", "west",
         "twice", "two", "times", "again", "double",
         "opposite", "reverse", "other_way", "inverse",
         "then", "next", "after", "continue", "also", "followed", "by",
         "please", "the", "direction", "once"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_TURN = 10
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3500
N_ID = 400
N_OOD = 500
EPOCHS = 20
LR = 1e-3

def clip(x, y):
    return max(0, min(GRID - 1, x)), max(0, min(GRID - 1, y))

def apply_moves(x0, y0, moves):
    x, y = x0, y0
    for m in moves:
        dx, dy = MOVE2DELTA[m]
        x, y = clip(x + dx, y + dy)
    return x, y

def expand(direction, twice=False, opposite=False):
    d = OPP[direction] if opposite else direction
    return [d] * (2 if twice else 1)

def tok(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_TURN)[:MAX_TURN]

def phr(m, twice=False, opposite=False, set_b=False):
    if opposite:
        return (f"proceed inverse {m}" if set_b else f"go opposite {m}"), m, False, True
    if twice:
        return (f"travel {m} double" if set_b else f"go {m} twice"), m, True, False
    return (f"step {m}" if set_b else f"go {m}"), m, False, False

def sample(n, seed, pairs, set_b=False, allow_twice=True):
    rng = np.random.RandomState(seed)
    T1, T2, Pos, Y = [], [], [], []
    labs = {k: [] for k in ["d1", "tw1", "op1", "d2", "tw2", "op2"]}
    for _ in range(n):
        m1, m2 = pairs[rng.randint(0, len(pairs))]
        tw1 = allow_twice and rng.rand() < 0.25
        tw2 = allow_twice and rng.rand() < 0.25
        op2 = rng.rand() < 0.2
        if op2:
            tw2 = False
        p1, d1, t1, o1 = phr(m1, tw1, False, set_b)
        p2, d2, t2, o2 = phr(m2, tw2, op2, set_b)
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        xm, ym = apply_moves(x0, y0, expand(d1, t1, o1))
        xf, yf = apply_moves(xm, ym, expand(d2, t2, o2))
        T1.append(tok(p1))
        T2.append(tok(p2))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["d1"].append(MOVE2ID[d1])
        labs["tw1"].append(1 if t1 else 0)
        labs["op1"].append(1 if o1 else 0)
        labs["d2"].append(MOVE2ID[d2])
        labs["tw2"].append(1 if t2 else 0)
        labs["op2"].append(1 if o2 else 0)
    out = {
        "t1": torch.tensor(T1, dtype=torch.long),
        "t2": torch.tensor(T2, dtype=torch.long),
        "pos": torch.tensor(Pos, dtype=torch.float32),
        "y": torch.tensor(Y, dtype=torch.float32),
    }
    for k, v in labs.items():
        out[k] = torch.tensor(v, dtype=torch.long)
    return out

def cell_acc(pred, y):
    pr = pred.round().clamp(0, GRID - 1)
    yt = y.round().clamp(0, GRID - 1)
    return (pr == yt).all(-1).float().mean().item()

class TurnEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_WORD, HIDDEN, padding_idx=0)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(HIDDEN * 2, HIDDEN)
    def forward(self, nl):
        e = self.emb(nl)
        lengths = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))

class FlatSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2 = TurnEnc(), TurnEnc()
        self.pos = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, t1, t2, pos):
        return self.mlp(torch.cat([self.e1(t1), self.e2(t2), self.pos(pos)], -1))

class StateMidProg(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2 = TurnEnc(), TurnEnc()
        self.d1 = nn.Linear(HIDDEN, 4)
        self.tw1 = nn.Linear(HIDDEN, 2)
        self.op1 = nn.Linear(HIDDEN, 2)
        self.d2 = nn.Linear(HIDDEN, 4)
        self.tw2 = nn.Linear(HIDDEN, 2)
        self.op2 = nn.Linear(HIDDEN, 2)
    def logits(self, t1, t2):
        h1, h2 = self.e1(t1), self.e2(t2)
        return {
            "d1": self.d1(h1), "tw1": self.tw1(h1), "op1": self.op1(h1),
            "d2": self.d2(h2), "tw2": self.tw2(h2), "op2": self.op2(h2),
        }
    def execute_from_logits(self, lg, pos):
        outs = []
        for i in range(pos.size(0)):
            d1 = MOVES[int(lg["d1"][i].argmax())]
            t1 = bool(int(lg["tw1"][i].argmax()))
            o1 = bool(int(lg["op1"][i].argmax()))
            d2 = MOVES[int(lg["d2"][i].argmax())]
            t2 = bool(int(lg["tw2"][i].argmax()))
            o2 = bool(int(lg["op2"][i].argmax()))
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xm, ym = apply_moves(x0, y0, expand(d1, t1, o1))
            xf, yf = apply_moves(xm, ym, expand(d2, t2, o2))
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=pos.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample(N_TRAIN, seed, TRAIN_PAIRS, set_b=False)
    idd = sample(N_ID, seed + 15, TRAIN_PAIRS, set_b=False)
    ood_pair = sample(N_OOD, seed + 20, OOD_PAIRS, set_b=False)       # OOD-1
    ood_phr = sample(N_OOD, seed + 30, TRAIN_PAIRS, set_b=True)       # OOD-2
    ood_hard = sample(N_OOD, seed + 40, OOD_PAIRS, set_b=True)        # OOD-3

    def train_flat():
        model = FlatSeq().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t1, t2, pos, y = tr["t1"].to(DEVICE), tr["t2"].to(DEVICE), tr["pos"].to(DEVICE), tr["y"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(t1[idx], t2[idx], pos[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(t1[:400], t2[:400], pos[:400]), y[:400]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = StateMidProg().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t1, t2 = tr["t1"].to(DEVICE), tr["t2"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total = 0
            nb = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                lg = model.logits(t1[idx], t2[idx])
                loss = (
                    F.cross_entropy(lg["d1"], tr["d1"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["tw1"], tr["tw1"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["op1"], tr["op1"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["d2"], tr["d2"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["tw2"], tr["tw2"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["op2"], tr["op2"].to(DEVICE)[idx])
                )
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            val = total / max(nb, 1)
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    flat = train_flat()
    prog = train_prog()
    flat.eval()
    prog.eval()
    with torch.no_grad():
        for name, data in [("id", idd), ("pair", ood_pair), ("phr", ood_phr), ("hard", ood_hard)]:
            t1, t2, pos, y = data["t1"].to(DEVICE), data["t2"].to(DEVICE), data["pos"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(t1, t2, pos), y)
            out[f"Prog_{name}"] = cell_acc(prog.execute_from_logits(prog.logits(t1, t2), pos), y)
    out["win"] = out["Prog_hard"] > out["Flat_hard"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat id={r['Flat_id']:.3f} pair={r['Flat_pair']:.3f} phr={r['Flat_phr']:.3f} hard={r['Flat_hard']:.3f}", flush=True)
    print(f"  Prog id={r['Prog_id']:.3f} pair={r['Prog_pair']:.3f} phr={r['Prog_phr']:.3f} hard={r['Prog_hard']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== O4b OOD TRANSFER SUMMARY =====")
for k in ["Flat_id","Flat_pair","Flat_phr","Flat_hard","Prog_id","Prog_pair","Prog_phr","Prog_hard"]:
    print(f"  {k}: {ms(k)[0]:.3f}")

fh, ph = ms("Flat_hard")[0], ms("Prog_hard")[0]
wins = sum(r["win"] for r in all_results)

if ph > fh + 0.05 and ph >= 0.45 and wins >= 2:
    outcome, note = "PASS", "hard OOD (held-out pairs + new phrasing) held"
elif ph + 0.05 < fh:
    outcome, note = "FAIL", "flat better on hard OOD"
else:
    outcome, note = "UNRESOLVED", "hard OOD gap insufficient"

print(f"wins={wins}/3 === OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "O4b different OOD pair+phrasing transfer",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in
            ["Flat_id","Flat_pair","Flat_phr","Flat_hard","Prog_id","Prog_pair","Prog_phr","Prog_hard"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **O4b** state-mid with different OOD (held-out pairs + phrasing)
- Flat hard={fh:.3f} | Prog hard={ph:.3f} pair Prog={ms('Prog_pair')[0]:.3f} phr={ms('Prog_phr')[0]:.3f}
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
