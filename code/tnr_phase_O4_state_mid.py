#!/usr/bin/env python3
"""
Phase O4 — Multi-turn: execute turn1 → intermediate state → turn2 as fresh command

Protocol:
  Turn1 NL → program1 → execute from start → state_mid
  Turn2 NL → program2 → execute from state_mid → final

Vs Flat: encode (t1,t2,pos0) → final cell (no intermediate execute)

Train/OOD: sequential two commands (not only revise-from-same-start).
OOD: novel paraphrases.

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

WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "head", "step", "travel",
         "north", "south", "east", "west",
         "twice", "two", "times", "again",
         "opposite", "reverse", "other_way",
         "then", "next", "after", "continue", "also",
         "please", "the", "direction", "once"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_TURN = 10
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3500
N_ID = 400
N_OOD = 600
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

def cmd_phr(m, twice=False, opposite=False, ood=False):
    if opposite:
        return (f"walk reverse {m}" if ood else f"go opposite {m}"), m, False, True
    if twice:
        return (f"walk {m} two times" if ood else f"go {m} twice"), m, True, False
    return (f"step {m}" if ood else f"go {m}"), m, False, False

def make_examples(ood=False):
    """Two sequential commands; final = apply both from start."""
    ex = []
    for m1 in MOVES:
        for m2 in MOVES:
            for t1_tw in [False, True]:
                for t2_tw in [False, True]:
                    # limit combos
                    if t1_tw and t2_tw:
                        continue
                    p1, d1, tw1, op1 = cmd_phr(m1, t1_tw, False, ood)
                    p2, d2, tw2, op2 = cmd_phr(m2, t2_tw, False, ood)
                    # connector
                    conn = "next" if ood else "then"
                    # store as sequential
                    ex.append((p1, p2, d1, tw1, op1, d2, tw2, op2))
            # opposite second
            p1, d1, tw1, op1 = cmd_phr(m1, False, False, ood)
            p2, d2, tw2, op2 = cmd_phr(m2, False, True, ood)
            ex.append((p1, p2, d1, tw1, op1, d2, tw2, op2))
    return ex

TRAIN_EX = make_examples(False)
OOD_EX = make_examples(True)
print(f"Train={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    T1, T2, Pos, Y = [], [], [], []
    labs = {k: [] for k in ["d1", "tw1", "op1", "d2", "tw2", "op2"]}
    for _ in range(n):
        p1, p2, d1, tw1, op1, d2, tw2, op2 = examples[rng.randint(0, len(examples))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        # sequential execute
        xm, ym = apply_moves(x0, y0, expand(d1, tw1, op1))
        xf, yf = apply_moves(xm, ym, expand(d2, tw2, op2))
        T1.append(tok(p1))
        T2.append(tok(p2))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["d1"].append(MOVE2ID[d1])
        labs["tw1"].append(1 if tw1 else 0)
        labs["op1"].append(1 if op1 else 0)
        labs["d2"].append(MOVE2ID[d2])
        labs["tw2"].append(1 if tw2 else 0)
        labs["op2"].append(1 if op2 else 0)
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
    """Independent program heads per turn; execute sequentially with mid state."""
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
            tw1 = bool(int(lg["tw1"][i].argmax()))
            op1 = bool(int(lg["op1"][i].argmax()))
            d2 = MOVES[int(lg["d2"][i].argmax())]
            tw2 = bool(int(lg["tw2"][i].argmax()))
            op2 = bool(int(lg["op2"][i].argmax()))
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xm, ym = apply_moves(x0, y0, expand(d1, tw1, op1))
            xf, yf = apply_moves(xm, ym, expand(d2, tw2, op2))
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=pos.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    idd = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    ood = sample_dataset(OOD_EX, N_OOD, seed + 20)

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
        for name, data in [("id", idd), ("ood", ood)]:
            t1, t2, pos, y = data["t1"].to(DEVICE), data["t2"].to(DEVICE), data["pos"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(t1, t2, pos), y)
            out[f"Prog_{name}"] = cell_acc(
                prog.execute_from_logits(prog.logits(t1, t2), pos), y
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
print("\n===== O4 STATE-MID SUMMARY =====")
print(f"Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.45 and wins >= 2:
    outcome, note = "PASS", "sequential execute via mid-state beats flat"
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
        "experiment": "Phase O4 state-after-turn1 sequential execute",
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
- Experiment: **Phase O4** state-after-turn1 sequential program execute
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
