#!/usr/bin/env python3
"""
Phase M3 — NL bridge: noise / OOV / longer sentences

Train: paraphrase set A (như M2), clean
OOD-A: paraphrase set B (M2 held)
OOD-B: noise — filler words, OOV tokens, longer phrasing
OOD-C: word dropout / shuffle mild on B

Pipeline: NL→program slots→grid execute vs Flat NL→pos

Preregister (primary = OOD-B noise):
  PASS: Prog_noise > Flat_noise + 0.05 AND Prog_noise >= 0.50 AND wins >= 2/3
  FAIL: Prog_noise + 0.05 < Flat_noise
  else UNRESOLVED
Also report paraphrase-B held.
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
         "go", "move", "walk", "head", "step", "travel", "proceed",
         "north", "south", "east", "west", "up", "down", "left", "right",
         "twice", "two", "times", "again", "double",
         "opposite", "opposite_of", "reverse", "other_way", "inverse",
         "after", "then", "following", "before", "prior_to",
         "the", "direction", "once", "please",
         # fillers seen in train sometimes
         "now", "kindly", "agent", "should"]
# OOV only in noise OOD (map to <unk>)
OOV_FILLERS = ["xyzzy", "foobar", "qwerty", "bla", "umm", "actually", "basically"]

W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 18
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 4500
N_ID = 400
N_OOD = 600
EPOCHS = 28
LR = 1e-3

def clip(x, y):
    return max(0, min(GRID - 1, x)), max(0, min(GRID - 1, y))

def apply_moves(x0, y0, moves):
    x, y = x0, y0
    for m in moves:
        dx, dy = MOVE2DELTA[m]
        x, y = clip(x + dx, y + dy)
    return x, y

def expand_program(direction, twice=False, opposite=False, after_dir=None):
    moves = []
    if after_dir is not None:
        moves.extend(expand_program(after_dir, False, False, None))
    d = OPP[direction] if opposite else direction
    moves.extend([d] * (2 if twice else 1))
    return moves

def phr_atomic(m, set_b=False):
    if not set_b:
        return [f"go {m}", f"move {m}", f"walk {m}", f"head {m}"]
    return [f"step {m}", f"travel {m}", f"proceed {m}", f"please go {m}"]

def phr_twice(m, set_b=False):
    if not set_b:
        return [f"go {m} twice", f"move {m} two times", f"walk {m} again"]
    return [f"step {m} twice", f"travel {m} double", f"proceed {m} two times"]

def phr_opp(m, set_b=False):
    if not set_b:
        return [f"go opposite {m}", f"move opposite_of {m}", f"walk reverse {m}"]
    return [f"step inverse {m}", f"travel other_way {m}", f"proceed reverse {m}"]

def phr_after(m, a, set_b=False):
    if not set_b:
        return [f"go {m} after {a}", f"move {m} then {a}", f"walk {m} following {a}"]
    return [f"step {m} after {a}", f"travel {m} before {a}", f"proceed {m} prior_to {a}"]

def build_templates(set_b=False):
    ex = []
    for m in MOVES:
        for phr in phr_atomic(m, set_b):
            ex.append((phr, m, False, False, None))
        for phr in phr_twice(m, set_b):
            ex.append((phr, m, True, False, None))
        for phr in phr_opp(m, set_b):
            ex.append((phr, m, False, True, None))
        for a in MOVES:
            if a == m:
                continue
            for phr in phr_after(m, a, set_b):
                ex.append((phr, m, False, False, a))
    return ex

def noise_phrase(phr, rng):
    """Longer + OOV fillers + mild duplication."""
    words = phr.split()
    out = []
    # prefix fillers
    for _ in range(rng.randint(1, 3)):
        if rng.rand() < 0.5:
            out.append(OOV_FILLERS[rng.randint(0, len(OOV_FILLERS))])
        else:
            out.append(["now", "kindly", "please", "agent"][rng.randint(0, 4)])
    out.extend(words)
    # infix OOV
    if rng.rand() < 0.7 and len(out) > 2:
        pos = rng.randint(1, len(out))
        out.insert(pos, OOV_FILLERS[rng.randint(0, len(OOV_FILLERS))])
    # suffix
    if rng.rand() < 0.5:
        out.append(OOV_FILLERS[rng.randint(0, len(OOV_FILLERS))])
    return " ".join(out)

TRAIN_TPL = build_templates(False)
OOD_B_TPL = build_templates(True)
print(f"Train={len(TRAIN_TPL)} OOD_B={len(OOD_B_TPL)}")

def tokenize(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def sample_dataset(templates, n, seed, noise=False):
    rng = np.random.RandomState(seed)
    X_nl, X_pos, Y_pos = [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    for _ in range(n):
        phr, direction, twice, opposite, after_dir = templates[rng.randint(0, len(templates))]
        if noise:
            phr = noise_phrase(phr, rng)
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand_program(direction, twice, opposite, after_dir)
        xf, yf = apply_moves(x0, y0, moves)
        X_nl.append(tokenize(phr))
        X_pos.append([x0, y0])
        Y_pos.append([xf, yf])
        labs["dir"].append(MOVE2ID[direction])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opposite else 0)
        labs["has_after"].append(1 if after_dir else 0)
        labs["after"].append(MOVE2ID[after_dir] if after_dir else 0)
    out = {
        "nl": torch.tensor(X_nl, dtype=torch.long),
        "pos": torch.tensor(X_pos, dtype=torch.float32),
        "y": torch.tensor(Y_pos, dtype=torch.float32),
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

class FlatNL2Pos(nn.Module):
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

class NL2ProgramExecute(nn.Module):
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
            moves = expand_program(d, t, o, ad)
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xf, yf = apply_moves(x0, y0, moves)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=pos.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_dataset(TRAIN_TPL, N_TRAIN, seed, noise=False)
    idd = sample_dataset(TRAIN_TPL, N_ID, seed + 15, noise=False)
    ood_b = sample_dataset(OOD_B_TPL, N_OOD, seed + 20, noise=False)
    ood_n = sample_dataset(OOD_B_TPL, N_OOD, seed + 30, noise=True)

    def train_flat():
        model = FlatNL2Pos().to(DEVICE)
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
        model = NL2ProgramExecute().to(DEVICE)
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
        for name, data in [("id", idd), ("para", ood_b), ("noise", ood_n)]:
            nl, pos, y = data["nl"].to(DEVICE), data["pos"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(nl, pos), y)
            out[f"Prog_{name}"] = cell_acc(
                prog.execute_from_logits(prog.program_logits(nl), pos), y
            )
    out["win"] = out["Prog_noise"] > out["Flat_noise"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat id={r['Flat_id']:.3f} para={r['Flat_para']:.3f} noise={r['Flat_noise']:.3f}", flush=True)
    print(f"  Prog id={r['Prog_id']:.3f} para={r['Prog_para']:.3f} noise={r['Prog_noise']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== M3 NOISE SUMMARY =====")
for k in ["Flat_id","Flat_para","Flat_noise","Prog_id","Prog_para","Prog_noise"]:
    print(f"  {k}: {ms(k)[0]:.3f}")

fn, pn = ms("Flat_noise")[0], ms("Prog_noise")[0]
wins = sum(r["win"] for r in all_results)
if pn > fn + 0.05 and pn >= 0.50 and wins >= 2:
    outcome, note = "PASS", "Prog robust to NL noise/OOV vs Flat"
elif pn + 0.05 < fn:
    outcome, note = "FAIL", "Flat better under noise"
else:
    outcome, note = "UNRESOLVED", "insufficient gap under noise"

print(f"wins={wins}/3 === OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase M3 NL noise OOV longer",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in
            ["Flat_id","Flat_para","Flat_noise","Prog_id","Prog_para","Prog_noise"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase M3** NL noise/OOV/longer → program→execute
- Flat noise={fn:.3f} | Prog noise={pn:.3f} para Prog={ms('Prog_para')[0]:.3f}
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
