#!/usr/bin/env python3
"""
Phase O3 — Multi-turn with explicit REVISE ops

Bias: turn2 maps to revise operators, not full re-parse of joint string.
  REVISE ops: KEEP | SET_DIR(d) | SET_TWICE | SET_OPP | SET_DIR_TWICE(d)
Program = base from turn1 slots + apply revise from turn2 → execute.

Train: single-turn + multi-turn with revise labels
OOD: novel paraphrase for turn2 corrections

Preregister:
  PASS: Prog OOD > Flat OOD + 0.05 AND Prog OOD >= 0.45 AND wins >= 2/3
  FAIL: Prog + 0.05 < Flat
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
MOVE2DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE2ID = {m: i for i, m in enumerate(MOVES)}

# Revise op ids
# 0 KEEP, 1 SET_TWICE, 2 SET_OPP, 3 SET_DIR (need dir), 4 SET_DIR_TWICE
REVISE_OPS = ["KEEP", "SET_TWICE", "SET_OPP", "SET_DIR", "SET_DIR_TWICE"]
ROP2ID = {r: i for i, r in enumerate(REVISE_OPS)}
N_ROP = len(REVISE_OPS)

WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "head", "step", "travel",
         "north", "south", "east", "west",
         "twice", "two", "times", "again",
         "opposite", "reverse", "other_way",
         "after", "then", "following",
         "no", "actually", "instead", "correction", "wait", "cancel",
         "please", "the", "direction", "once", "keep", "same"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_TURN = 10
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 2800
N_ID = 400
N_OOD = 600
EPOCHS = 18
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

def apply_revise(base_dir, base_twice, base_opp, rop, rev_dir):
    """Apply revise op to base slots → final slots."""
    d, t, o = base_dir, base_twice, base_opp
    if rop == "KEEP":
        pass
    elif rop == "SET_TWICE":
        t, o = True, False
    elif rop == "SET_OPP":
        t, o = False, True
    elif rop == "SET_DIR":
        d, t, o = rev_dir, False, False
    elif rop == "SET_DIR_TWICE":
        d, t, o = rev_dir, True, False
    return d, t, o

def tok(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_TURN)[:MAX_TURN]

def make_examples(ood=False):
    """(t1, t2, base_dir, base_twice, base_opp, rop, rev_dir, final_dir, final_twice, final_opp)"""
    ex = []
    # Single-turn: t2 empty, rop=KEEP
    for m in MOVES:
        t1s = [f"go {m}", f"walk {m}"] if not ood else [f"step {m}", f"travel {m}"]
        for t1 in t1s:
            ex.append((t1, "", m, False, False, "KEEP", m, m, False, False))
        t1s = [f"go {m} twice"] if not ood else [f"walk {m} two times"]
        for t1 in t1s:
            ex.append((t1, "", m, True, False, "KEEP", m, m, True, False))
        t1s = [f"go opposite {m}"] if not ood else [f"walk reverse {m}"]
        for t1 in t1s:
            ex.append((t1, "", m, False, True, "KEEP", m, m, False, True))

    # Multi-turn corrections
    for m in MOVES:
        t1 = f"go {m}" if not ood else f"step {m}"
        # SET_OPP
        t2 = "no opposite instead" if not ood else "wait reverse correction"
        fd, ft, fo = apply_revise(m, False, False, "SET_OPP", m)
        ex.append((t1, t2, m, False, False, "SET_OPP", m, fd, ft, fo))
        # SET_TWICE
        t2 = "no twice instead" if not ood else "actually two times"
        fd, ft, fo = apply_revise(m, False, False, "SET_TWICE", m)
        ex.append((t1, t2, m, False, False, "SET_TWICE", m, fd, ft, fo))
        # SET_DIR
        for m2 in MOVES:
            if m2 == m:
                continue
            t2 = f"no go {m2} instead" if not ood else f"wait step {m2} correction"
            fd, ft, fo = apply_revise(m, False, False, "SET_DIR", m2)
            ex.append((t1, t2, m, False, False, "SET_DIR", m2, fd, ft, fo))
        # SET_DIR_TWICE
        for m2 in MOVES:
            if m2 == m:
                continue
            t2 = f"no go {m2} twice instead" if not ood else f"wait step {m2} two times"
            fd, ft, fo = apply_revise(m, False, False, "SET_DIR_TWICE", m2)
            ex.append((t1, t2, m, False, False, "SET_DIR_TWICE", m2, fd, ft, fo))
    return ex

TRAIN_EX = make_examples(False)
OOD_EX = make_examples(True)
print(f"Train={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    T1, T2, Pos, Y = [], [], [], []
    labs = {k: [] for k in ["base_dir", "base_twice", "base_opp", "rop", "rev_dir"]}
    for _ in range(n):
        (t1, t2, bd, bt, bo, rop, rd, fd, ft, fo) = examples[rng.randint(0, len(examples))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(fd, ft, fo)
        xf, yf = apply_moves(x0, y0, moves)
        T1.append(tok(t1))
        T2.append(tok(t2 if t2 else "keep same"))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["base_dir"].append(MOVE2ID[bd])
        labs["base_twice"].append(1 if bt else 0)
        labs["base_opp"].append(1 if bo else 0)
        labs["rop"].append(ROP2ID[rop])
        labs["rev_dir"].append(MOVE2ID[rd])
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

class FlatMT(nn.Module):
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

class ReviseProg(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2 = TurnEnc(), TurnEnc()
        # turn1 → base slots
        self.base_dir = nn.Linear(HIDDEN, 4)
        self.base_twice = nn.Linear(HIDDEN, 2)
        self.base_opp = nn.Linear(HIDDEN, 2)
        # turn2 → revise op + optional dir
        self.rop = nn.Linear(HIDDEN, N_ROP)
        self.rev_dir = nn.Linear(HIDDEN, 4)

    def forward_slots(self, t1, t2):
        h1, h2 = self.e1(t1), self.e2(t2)
        return {
            "base_dir": self.base_dir(h1),
            "base_twice": self.base_twice(h1),
            "base_opp": self.base_opp(h1),
            "rop": self.rop(h2),
            "rev_dir": self.rev_dir(h2),
        }

    def execute_from_logits(self, logits, pos):
        outs = []
        for i in range(pos.size(0)):
            bd = MOVES[int(logits["base_dir"][i].argmax())]
            bt = bool(int(logits["base_twice"][i].argmax()))
            bo = bool(int(logits["base_opp"][i].argmax()))
            rop = REVISE_OPS[int(logits["rop"][i].argmax())]
            rd = MOVES[int(logits["rev_dir"][i].argmax())]
            fd, ft, fo = apply_revise(bd, bt, bo, rop, rd)
            moves = expand(fd, ft, fo)
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xf, yf = apply_moves(x0, y0, moves)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=pos.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    idd = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    ood = sample_dataset(OOD_EX, N_OOD, seed + 20)

    def train_flat():
        model = FlatMT().to(DEVICE)
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
        model = ReviseProg().to(DEVICE)
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
                lg = model.forward_slots(t1[idx], t2[idx])
                loss = (
                    F.cross_entropy(lg["base_dir"], tr["base_dir"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["base_twice"], tr["base_twice"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["base_opp"], tr["base_opp"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["rop"], tr["rop"].to(DEVICE)[idx])
                    + F.cross_entropy(lg["rev_dir"], tr["rev_dir"].to(DEVICE)[idx])
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
        for name, data in [("id", idd), ("ood", ood)]:
            t1, t2, pos, y = data["t1"].to(DEVICE), data["t2"].to(DEVICE), data["pos"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(t1, t2, pos), y)
            out[f"Prog_{name}"] = cell_acc(
                prog.execute_from_logits(prog.forward_slots(t1, t2), pos), y
            )
    out["win"] = out["Prog_ood"] > out["Flat_ood"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat ID={r['Flat_id']:.3f} OOD={r['Flat_ood']:.3f}", flush=True)
    print(f"  Prog ID={r['Prog_id']:.3f} OOD={r['Prog_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

fod, pod = ms("Flat_ood")[0], ms("Prog_ood")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== O3 REVISE SUMMARY =====")
print(f"Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.45 and wins >= 2:
    outcome, note = "PASS", "explicit REVISE ops unlock multi-turn"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better"
else:
    outcome, note = "UNRESOLVED", "gap insufficient"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase O3 explicit REVISE ops multi-turn",
        "seeds": SEEDS,
        "summary": {"Flat_ood": fod, "Prog_ood": pod, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase O3** multi-turn explicit REVISE ops (KEEP/SET_TWICE/SET_OPP/SET_DIR)
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
