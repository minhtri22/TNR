#!/usr/bin/env python3
"""
Phase ProdNL — Production-like free-form NL

Train: broad paraphrase bank (varied syntax)
OOD free-form:
  - longer sentences with distractors
  - mild typos / character noise on keywords
  - clause order variation
  - held-out surface forms never in train templates

Task: grid navigation (atomic / twice / opposite / after) + optional if-branch subset
Metric: cell-acc
Prog: NL→slots→execute vs Flat NL+pos→cell

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
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

GRID = 5
MOVES = ["north", "south", "east", "west"]
MOVE2DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE2ID = {m: i for i, m in enumerate(MOVES)}

# Larger production-ish vocab
WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "step", "travel", "head", "proceed", "advance", "please",
         "north", "south", "east", "west", "up", "down", "left", "right",
         "twice", "two", "times", "again", "double", "once",
         "opposite", "reverse", "other_way", "inverse", "contrary",
         "after", "then", "following", "before", "prior", "next",
         "the", "direction", "agent", "should", "kindly", "now", "just",
         "towards", "forward", "bit", "cells", "squares",
         "hey", "um", "actually", "basically", "can", "you", "make", "me",
         "i", "want", "need", "to", "a", "of"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 22
N_WORD = len(WORDS)
HIDDEN = 128
BATCH_SIZE = 64
N_TRAIN = 2800
N_ID = 400
N_OOD = 600
EPOCHS = 14
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
    return (ids + [0] * MAX_NL)[:MAX_NL]

def mild_typo(word, rng):
    if len(word) < 4 or rng.rand() > 0.35:
        return word
    chars = list(word)
    i = rng.randint(1, len(chars) - 1)
    # swap or drop
    if rng.rand() < 0.5 and i + 1 < len(chars):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    else:
        chars.pop(i)
    return "".join(chars)

# Train templates (structured but varied)
def train_phrases(m, twice=False, opposite=False, after=None):
    base = []
    if after is None and not twice and not opposite:
        base = [
            f"go {m}", f"please walk {m}", f"move {m} once",
            f"agent should head {m}", f"travel {m}",
            f"kindly proceed {m}", f"i want to go {m}",
        ]
    elif twice and not opposite and after is None:
        base = [
            f"go {m} twice", f"walk {m} two times", f"please move {m} again",
            f"agent should step {m} double", f"i need to go {m} twice",
        ]
    elif opposite and after is None:
        base = [
            f"go opposite {m}", f"walk reverse {m}", f"please head other_way {m}",
            f"move inverse {m}", f"agent should go contrary {m}",
        ]
    else:
        base = [
            f"go {m} after {after}", f"walk {m} then {after}",
            f"please move {m} following {after}", f"agent should go {m} after {after}",
        ]
    return base

# Free-form OOD: distractors, reordering, typos applied at sample time
def freeform_shell(core, rng):
    prefixes = [
        "hey can you", "um actually please", "basically i want you to",
        "now just", "kindly", "", "hey agent",
    ]
    suffixes = [
        "thanks", "please", "now", "a bit", "forward", "",
    ]
    pre = prefixes[rng.randint(0, len(prefixes))]
    suf = suffixes[rng.randint(0, len(suffixes))]
    words = (pre + " " + core + " " + suf).split()
    # typo on content words sometimes
    words = [mild_typo(w, rng) if w in MOVES or w in ("twice", "opposite", "after", "go", "walk") else w for w in words]
    return " ".join(w for w in words if w)

def sample_train(n, seed):
    rng = np.random.RandomState(seed)
    X, Pos, Y = [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    for _ in range(n):
        mode = rng.randint(0, 4)
        m = MOVES[rng.randint(0, 4)]
        twice = opposite = False
        after = None
        if mode == 0:
            phrs = train_phrases(m)
        elif mode == 1:
            twice = True
            phrs = train_phrases(m, twice=True)
        elif mode == 2:
            opposite = True
            phrs = train_phrases(m, opposite=True)
        else:
            after = MOVES[rng.randint(0, 4)]
            while after == m:
                after = MOVES[rng.randint(0, 4)]
            phrs = train_phrases(m, after=after)
        phr = phrs[rng.randint(0, len(phrs))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(m, twice, opposite, after)
        xf, yf = apply_moves(x0, y0, moves)
        X.append(tok(phr))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["dir"].append(MOVE2ID[m])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opposite else 0)
        labs["has_after"].append(1 if after else 0)
        labs["after"].append(MOVE2ID[after] if after else 0)
    out = {
        "nl": torch.tensor(X, dtype=torch.long),
        "pos": torch.tensor(Pos, dtype=torch.float32),
        "y": torch.tensor(Y, dtype=torch.float32),
    }
    for k, v in labs.items():
        out[k] = torch.tensor(v, dtype=torch.long)
    return out

def sample_ood_free(n, seed):
    """Free-form OOD never using train template strings exactly."""
    rng = np.random.RandomState(seed)
    X, Pos, Y = [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    # cores not in train list
    cores_atomic = ["advance {m}", "head towards {m}", "make me go {m}", "need to walk {m}"]
    cores_twice = ["go {m} two times please", "double step {m}", "walk {m} twice now"]
    cores_opp = ["reverse direction {m}", "other_way from {m}", "inverse of {m}"]
    cores_after = ["{m} after going {a}", "first {a} next {m}", "following {a} go {m}"]
    for _ in range(n):
        mode = rng.randint(0, 4)
        m = MOVES[rng.randint(0, 4)]
        twice = opposite = False
        after = None
        if mode == 0:
            core = cores_atomic[rng.randint(0, len(cores_atomic))].format(m=m)
        elif mode == 1:
            twice = True
            core = cores_twice[rng.randint(0, len(cores_twice))].format(m=m)
        elif mode == 2:
            opposite = True
            core = cores_opp[rng.randint(0, len(cores_opp))].format(m=m)
        else:
            after = MOVES[rng.randint(0, 4)]
            while after == m:
                after = MOVES[rng.randint(0, 4)]
            core = cores_after[rng.randint(0, len(cores_after))].format(m=m, a=after)
        phr = freeform_shell(core, rng)
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(m, twice, opposite, after)
        xf, yf = apply_moves(x0, y0, moves)
        X.append(tok(phr))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["dir"].append(MOVE2ID[m])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opposite else 0)
        labs["has_after"].append(1 if after else 0)
        labs["after"].append(MOVE2ID[after] if after else 0)
    out = {
        "nl": torch.tensor(X, dtype=torch.long),
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

class NLEnc(nn.Module):
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

class FlatP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEnc()
        self.pos = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, nl, pos):
        return self.mlp(torch.cat([self.enc(nl), self.pos(pos)], -1))

class ProgP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEnc()
        self.dir_h = nn.Linear(HIDDEN, 4)
        self.tw_h = nn.Linear(HIDDEN, 2)
        self.op_h = nn.Linear(HIDDEN, 2)
        self.has_h = nn.Linear(HIDDEN, 2)
        self.af_h = nn.Linear(HIDDEN, 4)
    def logits(self, nl):
        h = self.enc(nl)
        return {
            "dir": self.dir_h(h), "twice": self.tw_h(h), "opp": self.op_h(h),
            "has_after": self.has_h(h), "after": self.af_h(h),
        }
    def execute(self, lg, pos):
        outs = []
        for i in range(pos.size(0)):
            d = MOVES[int(lg["dir"][i].argmax())]
            t = bool(int(lg["twice"][i].argmax()))
            o = bool(int(lg["opp"][i].argmax()))
            ad = MOVES[int(lg["after"][i].argmax())] if int(lg["has_after"][i].argmax()) == 1 else None
            moves = expand(d, t, o, ad)
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xf, yf = apply_moves(x0, y0, moves)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_train(N_TRAIN, seed)
    idd = sample_train(N_ID, seed + 15)
    ood = sample_ood_free(N_OOD, seed + 20)

    def train_flat():
        model = FlatP()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, pos, y = tr["nl"], tr["pos"], tr["y"]
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(nl[idx], pos[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(nl[:400], pos[:400]), y[:400]).item()
            if val < best:
                best = val
                st = {k: v.clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ProgP()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl = tr["nl"]
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN)
            total = 0
            nb = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                lg = model.logits(nl[idx])
                loss = (
                    F.cross_entropy(lg["dir"], tr["dir"][idx])
                    + F.cross_entropy(lg["twice"], tr["twice"][idx])
                    + F.cross_entropy(lg["opp"], tr["opp"][idx])
                    + F.cross_entropy(lg["has_after"], tr["has_after"][idx])
                    + F.cross_entropy(lg["after"], tr["after"][idx])
                )
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            val = total / max(nb, 1)
            if val < best:
                best = val
                st = {k: v.clone() for k, v in model.state_dict().items()}
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
            nl, pos, y = data["nl"], data["pos"], data["y"]
            out[f"Flat_{name}"] = cell_acc(flat(nl, pos), y)
            out[f"Prog_{name}"] = cell_acc(prog.execute(prog.logits(nl), pos), y)
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
print("\n===== ProdNL FREEFORM SUMMARY =====")
print(f"Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.45 and wins >= 2:
    outcome, note = "PASS", "free-form NL still favors program→execute"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better on free-form"
else:
    outcome, note = "UNRESOLVED", "free-form gap insufficient"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase ProdNL free-form production-like NL",
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
- Experiment: **ProdNL** free-form production-like NL (distractors, typos, novel cores)
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
