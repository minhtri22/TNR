#!/usr/bin/env python3
"""
Phase CF — Control-flow on agent ladder (if / conditional branch)

Grid agent. Observation: wall ahead (binary) from facing direction.
Commands:
  atomic moves
  "if blocked go DIR else go DIR2"
  "if clear go DIR else go DIR2"

Train: atomics + if-templates set A
OOD: novel paraphrase if-templates + held-out direction combos in branches

Pipelines:
  Prog: NL → slots (cond, then_dir, else_dir) → execute with branch on wall
  Flat: NL + pos + wall_flag → final cell

Preregister:
  PASS: Prog OOD > Flat OOD + 0.05 AND Prog OOD >= 0.50 AND wins >= 2/3
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
MOVE2ID = {m: i for i, m in enumerate(MOVES)}
# facing for wall check: same as intended first dir in atomic; for if uses probe dir

WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "step", "travel", "head",
         "north", "south", "east", "west",
         "if", "blocked", "clear", "wall", "open", "path",
         "then", "else", "otherwise", "otherwise_go",
         "when", "unless", "please", "the", "is", "ahead"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 16
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

def step(x, y, d):
    dx, dy = MOVE2DELTA[d]
    return clip(x + dx, y + dy)

def wall_ahead(x, y, probe_dir):
    """Blocked if step would hit boundary (toy wall = edge)."""
    nx, ny = step(x, y, probe_dir)
    return (nx, ny) == (x, y)  # no movement => blocked by edge

def execute_atomic(x, y, d):
    return step(x, y, d)

def execute_if(x, y, cond_blocked, then_d, else_d, probe_d):
    blocked = wall_ahead(x, y, probe_d)
    if cond_blocked:
        take = then_d if blocked else else_d
    else:  # if clear
        take = then_d if not blocked else else_d
    return step(x, y, take)

def tok(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def make_examples(ood=False):
    """Returns list of dicts with nl, kind, fields, final executor args."""
    ex = []
    # atomic
    for m in MOVES:
        phrs = [f"go {m}", f"walk {m}"] if not ood else [f"step {m}", f"travel {m}"]
        for p in phrs:
            ex.append({"nl": p, "kind": "atomic", "dir": m, "cond_blocked": 0,
                       "then_d": m, "else_d": m, "probe": m})
    # if blocked ... else ...
    pairs = [(a, b) for a in MOVES for b in MOVES if a != b]
    # hold out some pairs for OOD structure
    ood_pairs = [("north", "east"), ("south", "west"), ("east", "south")]
    train_pairs = [p for p in pairs if p not in ood_pairs]
    use_pairs = ood_pairs if ood else train_pairs
    for then_d, else_d in use_pairs:
        probe = then_d  # probe same as then branch intent
        if not ood:
            phrs = [
                f"if blocked go {then_d} else go {else_d}",
                f"if wall ahead go {then_d} else go {else_d}",
            ]
        else:
            phrs = [
                f"when blocked walk {then_d} otherwise {else_d}",
                f"if path is wall head {then_d} else {else_d}",
            ]
        for p in phrs:
            ex.append({"nl": p, "kind": "if", "dir": then_d, "cond_blocked": 1,
                       "then_d": then_d, "else_d": else_d, "probe": probe})
        # if clear
        if not ood:
            phrs = [f"if clear go {then_d} else go {else_d}"]
        else:
            phrs = [f"when open go {then_d} otherwise_go {else_d}"]
        for p in phrs:
            ex.append({"nl": p, "kind": "if", "dir": then_d, "cond_blocked": 0,
                       "then_d": then_d, "else_d": else_d, "probe": probe})
    return ex

TRAIN_EX = make_examples(False)
OOD_EX = make_examples(True)
print(f"Train={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X_nl, Pos, Wall, Y = [], [], [], []
    labs = {k: [] for k in ["kind", "cond", "then", "else_d", "probe"]}
    for _ in range(n):
        e = examples[rng.randint(0, len(examples))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        probe = e["probe"]
        w = 1.0 if wall_ahead(x0, y0, probe) else 0.0
        if e["kind"] == "atomic":
            xf, yf = execute_atomic(x0, y0, e["dir"])
            kind_id = 0
        else:
            xf, yf = execute_if(x0, y0, bool(e["cond_blocked"]), e["then_d"], e["else_d"], probe)
            kind_id = 1
        X_nl.append(tok(e["nl"]))
        Pos.append([x0, y0])
        Wall.append([w])
        Y.append([xf, yf])
        labs["kind"].append(kind_id)
        labs["cond"].append(int(e["cond_blocked"]))
        labs["then"].append(MOVE2ID[e["then_d"]])
        labs["else_d"].append(MOVE2ID[e["else_d"]])
        labs["probe"].append(MOVE2ID[e["probe"]])
    out = {
        "nl": torch.tensor(X_nl, dtype=torch.long),
        "pos": torch.tensor(Pos, dtype=torch.float32),
        "wall": torch.tensor(Wall, dtype=torch.float32),
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

class FlatCF(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEnc()
        self.pos = nn.Linear(3, HIDDEN)  # x,y,wall
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, nl, pos, wall):
        st = torch.cat([pos, wall], -1)
        return self.mlp(torch.cat([self.enc(nl), self.pos(st)], -1))

class ProgCF(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEnc()
        self.kind = nn.Linear(HIDDEN, 2)
        self.cond = nn.Linear(HIDDEN, 2)
        self.then_d = nn.Linear(HIDDEN, 4)
        self.else_d = nn.Linear(HIDDEN, 4)
        self.probe = nn.Linear(HIDDEN, 4)

    def logits(self, nl):
        h = self.enc(nl)
        return {
            "kind": self.kind(h), "cond": self.cond(h),
            "then": self.then_d(h), "else_d": self.else_d(h), "probe": self.probe(h),
        }

    def execute(self, lg, pos):
        outs = []
        for i in range(pos.size(0)):
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            kind = int(lg["kind"][i].argmax())
            then_d = MOVES[int(lg["then"][i].argmax())]
            else_d = MOVES[int(lg["else_d"][i].argmax())]
            probe = MOVES[int(lg["probe"][i].argmax())]
            cond_b = bool(int(lg["cond"][i].argmax()))
            if kind == 0:
                xf, yf = execute_atomic(x0, y0, then_d)
            else:
                xf, yf = execute_if(x0, y0, cond_b, then_d, else_d, probe)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, dtype=torch.float32, device=pos.device)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    idd = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    ood = sample_dataset(OOD_EX, N_OOD, seed + 20)

    def train_flat():
        model = FlatCF().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, pos, wall, y = tr["nl"], tr["pos"], tr["wall"], tr["y"]
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(nl[idx], pos[idx], wall[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(nl[:400], pos[:400], wall[:400]), y[:400]).item()
            if val < best:
                best = val
                st = {k: v.clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ProgCF().to(DEVICE)
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
                    F.cross_entropy(lg["kind"], tr["kind"][idx])
                    + F.cross_entropy(lg["cond"], tr["cond"][idx])
                    + F.cross_entropy(lg["then"], tr["then"][idx])
                    + F.cross_entropy(lg["else_d"], tr["else_d"][idx])
                    + F.cross_entropy(lg["probe"], tr["probe"][idx])
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
            nl, pos, wall, y = data["nl"], data["pos"], data["wall"], data["y"]
            out[f"Flat_{name}"] = cell_acc(flat(nl, pos, wall), y)
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
print("\n===== CF CONTROL-FLOW SUMMARY =====")
print(f"Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.50 and wins >= 2:
    outcome, note = "PASS", "if/branch program→execute beats flat"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better on control-flow"
else:
    outcome, note = "UNRESOLVED", "insufficient gap"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase CF control-flow if/branch",
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
- Experiment: **Phase CF** control-flow if/blocked branch on grid
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
