#!/usr/bin/env python3
"""
Phase O — Multi-turn (hội thoại / sửa lệnh)

State: grid agent. Turn1: NL command → should reach cell A.
Turn2: correction NL ("no, go opposite", "actually twice", "then east") 
       applied as revise program from same start OR continue from turn1 result.

Protocol (một giả thuyết):
  Structured: encode turn1+turn2 → program slots → execute from start
  Flat: encode turn1+turn2 + pos → final cell

Train: single-turn full structures (M2-style) + multi-turn corrections
OOD: novel paraphrase corrections + composition corrections

Preregister:
  PASS: Prog OOD cell-acc > Flat OOD + 0.05 AND Prog OOD >= 0.45 AND wins >= 2/3
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

WORDS = ["<pad>", "<unk>", "<sep>",
         "go", "move", "walk", "head", "step", "travel",
         "north", "south", "east", "west",
         "twice", "two", "times", "again",
         "opposite", "reverse", "other_way",
         "after", "then", "following",
         "no", "actually", "instead", "correction", "wait",
         "please", "the", "direction", "once", "cancel"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 20
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3000
N_ID = 400
N_OOD = 700
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

def expand(direction, twice=False, opposite=False, after_dir=None):
    moves = []
    if after_dir is not None:
        moves.extend(expand(after_dir, False, False, None))
    d = OPP[direction] if opposite else direction
    moves.extend([d] * (2 if twice else 1))
    return moves

def tokenize(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def join_turns(t1, t2):
    return t1 + " <sep> " + t2

# Train single + multi-turn corrections
def make_examples(ood=False):
    """Returns list of (nl, direction, twice, opposite, after_dir)."""
    ex = []
    # single turn
    for m in MOVES:
        phrs = [f"go {m}", f"walk {m}"] if not ood else [f"step {m}", f"travel {m}"]
        for p in phrs:
            ex.append((p, m, False, False, None))
        phrs = [f"go {m} twice"] if not ood else [f"walk {m} two times"]
        for p in phrs:
            ex.append((p, m, True, False, None))
        phrs = [f"go opposite {m}"] if not ood else [f"walk reverse {m}"]
        for p in phrs:
            ex.append((p, m, False, True, None))
        for a in MOVES:
            if a == m:
                continue
            phrs = [f"go {m} after {a}"] if not ood else [f"walk {m} then {a}"]
            for p in phrs:
                ex.append((p, m, False, False, a))
    # multi-turn: turn1 wrong direction atomic, turn2 correction
    for m in MOVES:
        t1 = f"go {m}" if not ood else f"step {m}"
        # correction to opposite
        t2 = "no opposite instead" if not ood else "wait reverse correction"
        ex.append((join_turns(t1, t2), m, False, True, None))
        # correction to twice
        t2 = "no twice instead" if not ood else "actually two times"
        ex.append((join_turns(t1, t2), m, True, False, None))
        # correction to different direction
        for m2 in MOVES:
            if m2 == m:
                continue
            t2 = f"no go {m2} instead" if not ood else f"wait step {m2} correction"
            ex.append((join_turns(t1, t2), m2, False, False, None))
    return ex

TRAIN_EX = make_examples(False)
OOD_EX = make_examples(True)
print(f"Train ex={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X_nl, X_pos, Y = [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    for _ in range(n):
        nl, direction, twice, opposite, after_dir = examples[rng.randint(0, len(examples))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(direction, twice, opposite, after_dir)
        xf, yf = apply_moves(x0, y0, moves)
        X_nl.append(tokenize(nl))
        X_pos.append([x0, y0])
        Y.append([xf, yf])
        labs["dir"].append(MOVE2ID[direction])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opposite else 0)
        labs["has_after"].append(1 if after_dir else 0)
        labs["after"].append(MOVE2ID[after_dir] if after_dir else 0)
    out = {
        "nl": torch.tensor(X_nl, dtype=torch.long),
        "pos": torch.tensor(X_pos, dtype=torch.float32),
        "y": torch.tensor(Y, dtype=torch.float32),
    }
    for k, v in labs.items():
        out[k] = torch.tensor(v, dtype=torch.long)
    return out

def cell_acc(pred, y):
    pr = pred.round().clamp(0, GRID - 1)
    yt = y.round().clamp(0, GRID - 1)
    return (pr == yt).all(-1).float().mean().item()

class NLEncoder(nn.Module):
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
        self.enc = NLEncoder()
        self.pos_proj = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, nl, pos):
        return self.mlp(torch.cat([self.enc(nl), self.pos_proj(pos)], -1))

class ProgMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.dir_head = nn.Linear(HIDDEN, 4)
        self.twice_head = nn.Linear(HIDDEN, 2)
        self.opp_head = nn.Linear(HIDDEN, 2)
        self.has_after_head = nn.Linear(HIDDEN, 2)
        self.after_head = nn.Linear(HIDDEN, 4)

    def program_logits(self, nl):
        h = self.enc(nl)
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
        model = FlatMT().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, pos, y = tr["nl"].to(DEVICE), tr["pos"].to(DEVICE), tr["y"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(nl[idx], pos[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(nl[:512], pos[:512]), y[:512]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ProgMT().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl = tr["nl"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total = 0
            nb = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                logits = model.program_logits(nl[idx])
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
            nl, pos, y = data["nl"].to(DEVICE), data["pos"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(nl, pos), y)
            out[f"Prog_{name}"] = cell_acc(
                prog.execute_from_logits(prog.program_logits(nl), pos), y
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
print("\n===== O MULTI-TURN SUMMARY =====")
print(f"Flat: ID={ms('Flat_id')[0]:.3f} OOD={fod:.3f}")
print(f"Prog: ID={ms('Prog_id')[0]:.3f} OOD={pod:.3f}")
print(f"wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.45 and wins >= 2:
    outcome, note = "PASS", "multi-turn correction program>flat"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better multi-turn"
else:
    outcome, note = "UNRESOLVED", "gap insufficient"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase O multi-turn correction",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["Flat_id","Flat_ood","Prog_id","Prog_ood"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase O Multi-turn** command + correction → program→execute
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
