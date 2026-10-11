#!/usr/bin/env python3
"""
Phase O2 — Multi-turn: separate turn2 encoder (một bias)

O UNRESOLVED: joint encode t1+t2 không tách rõ.
O2: encode turn1, encode turn2 riêng; fuse → program slots → execute.
So sánh Flat joint vs Prog split-encoder.

Cùng data O (train + OOD paraphrase multi-turn).
Preregister giữ O:
  PASS: Prog OOD > Flat OOD + 0.05 AND Prog OOD >= 0.45 AND wins >= 2/3
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

WORDS = ["<pad>", "<unk>", "<sep>",
         "go", "move", "walk", "head", "step", "travel",
         "north", "south", "east", "west",
         "twice", "two", "times", "again",
         "opposite", "reverse", "other_way",
         "after", "then", "following",
         "no", "actually", "instead", "correction", "wait",
         "please", "the", "direction", "once", "cancel"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_TURN = 10
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3500
N_ID = 400
N_OOD = 600
EPOCHS = 22
LR = 1e-3

def clip(x, y):
    return max(0, min(GRID - 1, x)), max(0, min(GRID - 1, y))

def apply_moves(x0, y0, moves):
    x, y = x0, y0
    for m in moves:
        dx, dy = MOVE2DELTA[m]
        x, y = clip(x + dx, y + dy)
    return x, y

def expand(direction, twice=False, opposite=False, after_dir=None):
    moves = []
    if after_dir is not None:
        moves.extend(expand(after_dir, False, False, None))
    d = OPP[direction] if opposite else direction
    moves.extend([d] * (2 if twice else 1))
    return moves

def tok(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_TURN)[:MAX_TURN]

def make_examples(ood=False):
    ex = []
    for m in MOVES:
        phrs = [f"go {m}", f"walk {m}"] if not ood else [f"step {m}", f"travel {m}"]
        for p in phrs:
            ex.append((p, "", m, False, False, None))  # single: t2 empty
        phrs = [f"go {m} twice"] if not ood else [f"walk {m} two times"]
        for p in phrs:
            ex.append((p, "", m, True, False, None))
        phrs = [f"go opposite {m}"] if not ood else [f"walk reverse {m}"]
        for p in phrs:
            ex.append((p, "", m, False, True, None))
        for a in MOVES:
            if a == m:
                continue
            phrs = [f"go {m} after {a}"] if not ood else [f"walk {m} then {a}"]
            for p in phrs:
                ex.append((p, "", m, False, False, a))
        t1 = f"go {m}" if not ood else f"step {m}"
        t2 = "no opposite instead" if not ood else "wait reverse correction"
        ex.append((t1, t2, m, False, True, None))
        t2 = "no twice instead" if not ood else "actually two times"
        ex.append((t1, t2, m, True, False, None))
        for m2 in MOVES:
            if m2 == m:
                continue
            t2 = f"no go {m2} instead" if not ood else f"wait step {m2} correction"
            ex.append((t1, t2, m2, False, False, None))
    return ex

TRAIN_EX = make_examples(False)
OOD_EX = make_examples(True)
print(f"Train={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    T1, T2, Pos, Y = [], [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    for _ in range(n):
        t1, t2, direction, twice, opposite, after_dir = examples[rng.randint(0, len(examples))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(direction, twice, opposite, after_dir)
        xf, yf = apply_moves(x0, y0, moves)
        T1.append(tok(t1))
        T2.append(tok(t2 if t2 else "please"))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["dir"].append(MOVE2ID[direction])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opposite else 0)
        labs["has_after"].append(1 if after_dir else 0)
        labs["after"].append(MOVE2ID[after_dir] if after_dir else 0)
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

class TurnEncoder(nn.Module):
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

class FlatSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = TurnEncoder()
        self.e2 = TurnEncoder()
        self.pos_proj = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, t1, t2, pos):
        h = torch.cat([self.e1(t1), self.e2(t2), self.pos_proj(pos)], -1)
        return self.mlp(h)

class ProgSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = TurnEncoder()
        self.e2 = TurnEncoder()
        self.fuse = nn.Linear(HIDDEN * 2, HIDDEN)
        self.dir_head = nn.Linear(HIDDEN, 4)
        self.twice_head = nn.Linear(HIDDEN, 2)
        self.opp_head = nn.Linear(HIDDEN, 2)
        self.has_after_head = nn.Linear(HIDDEN, 2)
        self.after_head = nn.Linear(HIDDEN, 4)

    def program_logits(self, t1, t2):
        h = torch.tanh(self.fuse(torch.cat([self.e1(t1), self.e2(t2)], -1)))
        return {
            "dir": self.dir_head(h), "twice": self.twice_head(h),
            "opp": self.opp_head(h), "has_after": self.has_after_head(h),
            "after": self.after_head(h),
        }

    def execute_from_logits(self, logits, pos):
        outs = []
        for i in range(pos.size(0)):
            d = MOVES[int(logits["dir"][i].argmax())]
            t = bool(int(logits["twice"][i].argmax()))
            o = bool(int(logits["opp"][i].argmax()))
            ad = MOVES[int(logits["after"][i].argmax())] if int(logits["has_after"][i].argmax()) == 1 else None
            moves = expand(d, t, o, ad)
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
        model = FlatSplit().to(DEVICE)
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
        model = ProgSplit().to(DEVICE)
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
                logits = model.program_logits(t1[idx], t2[idx])
                loss = (
                    F.cross_entropy(logits["dir"], tr["dir"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["twice"], tr["twice"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["opp"], tr["opp"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["has_after"], tr["has_after"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["after"], tr["after"].to(DEVICE)[idx])
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
                prog.execute_from_logits(prog.program_logits(t1, t2), pos), y
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
print("\n===== O2 TURN-SPLIT SUMMARY =====")
print(f"Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.45 and wins >= 2:
    outcome, note = "PASS", "split turn encoders help multi-turn"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better"
else:
    outcome, note = "UNRESOLVED", "still insufficient gap"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase O2 split turn encoders",
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
- Experiment: **Phase O2** multi-turn split turn1/turn2 encoders
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
