#!/usr/bin/env python3
"""
Phase P — Perception mock (grid → “pixels”)

Thay pos (x,y) số bằng local grid patch (one-hot / occupancy) quanh agent.
NL → program slots → execute; perception encoder đọc patch.

Vs Flat: pixels + NL → final cell residual.

Train/OOD: cùng M2-style structures; OOD paraphrase set B.
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

GRID = 5
PATCH = 5  # full grid as "image" for mock perception
MOVES = ["north", "south", "east", "west"]
MOVE2DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE2ID = {m: i for i, m in enumerate(MOVES)}

WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "head", "step", "travel", "proceed",
         "north", "south", "east", "west",
         "twice", "two", "times", "again", "double",
         "opposite", "opposite_of", "reverse", "other_way", "inverse",
         "after", "then", "following", "before", "prior_to",
         "the", "direction", "once", "please"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 14
N_WORD = len(WORDS)
HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 4000
N_ID = 400
N_OOD = 600
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

def expand(direction, twice=False, opposite=False, after_dir=None):
    moves = []
    if after_dir is not None:
        moves.extend(expand(after_dir, False, False, None))
    d = OPP[direction] if opposite else direction
    moves.extend([d] * (2 if twice else 1))
    return moves

def grid_pixels(x, y):
    """Mock perception: GRID×GRID image, agent=1, empty=0."""
    img = np.zeros((GRID, GRID), dtype=np.float32)
    img[y, x] = 1.0  # row=y, col=x
    return img

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

TRAIN_TPL = build_templates(False)
OOD_TPL = build_templates(True)
print(f"Train={len(TRAIN_TPL)} OOD={len(OOD_TPL)}")

def tokenize(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def sample_dataset(templates, n, seed):
    rng = np.random.RandomState(seed)
    X_nl, X_img, Y = [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    for _ in range(n):
        phr, direction, twice, opposite, after_dir = templates[rng.randint(0, len(templates))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(direction, twice, opposite, after_dir)
        xf, yf = apply_moves(x0, y0, moves)
        X_nl.append(tokenize(phr))
        X_img.append(grid_pixels(x0, y0))
        Y.append([xf, yf])
        labs["dir"].append(MOVE2ID[direction])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opposite else 0)
        labs["has_after"].append(1 if after_dir else 0)
        labs["after"].append(MOVE2ID[after_dir] if after_dir else 0)
    out = {
        "nl": torch.tensor(X_nl, dtype=torch.long),
        "img": torch.tensor(np.array(X_img), dtype=torch.float32).unsqueeze(1),  # B,1,G,G
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

class VisionEncoder(nn.Module):
    """Mock CNN on grid pixels."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, HIDDEN)
    def forward(self, img):
        h = self.conv(img).view(img.size(0), -1)
        return self.proj(h)

class FlatPerc(nn.Module):
    def __init__(self):
        super().__init__()
        self.nl = NLEncoder()
        self.vis = VisionEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, nl, img):
        return self.mlp(torch.cat([self.nl(nl), self.vis(img)], -1))

class ProgPerc(nn.Module):
    """Vision only to read start cell for execute; program from NL only (structured)."""
    def __init__(self):
        super().__init__()
        self.nl = NLEncoder()
        self.vis = VisionEncoder()
        self.start_head = nn.Linear(HIDDEN, GRID * GRID)  # predict start cell from pixels
        self.dir_head = nn.Linear(HIDDEN, 4)
        self.twice_head = nn.Linear(HIDDEN, 2)
        self.opp_head = nn.Linear(HIDDEN, 2)
        self.has_after_head = nn.Linear(HIDDEN, 2)
        self.after_head = nn.Linear(HIDDEN, 4)

    def program_logits(self, nl):
        h = self.nl(nl)
        return {
            "dir": self.dir_head(h), "twice": self.twice_head(h),
            "opp": self.opp_head(h), "has_after": self.has_after_head(h),
            "after": self.after_head(h),
        }

    def start_from_img(self, img):
        logits = self.start_head(self.vis(img))
        idx = logits.argmax(-1)
        y = idx // GRID
        x = idx % GRID
        return x, y, logits

    def execute_from_logits(self, logits, img):
        x_t, y_t, _ = self.start_from_img(img)
        outs = []
        for i in range(img.size(0)):
            d = MOVES[int(logits["dir"][i].argmax())]
            t = bool(int(logits["twice"][i].argmax()))
            o = bool(int(logits["opp"][i].argmax()))
            ad = MOVES[int(logits["after"][i].argmax())] if int(logits["has_after"][i].argmax()) == 1 else None
            moves = expand(d, t, o, ad)
            x0, y0 = int(x_t[i]), int(y_t[i])
            xf, yf = apply_moves(x0, y0, moves)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=img.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_dataset(TRAIN_TPL, N_TRAIN, seed)
    idd = sample_dataset(TRAIN_TPL, N_ID, seed + 15)
    ood = sample_dataset(OOD_TPL, N_OOD, seed + 20)

    # start cell labels for aux
    def start_labels(img):
        # img: B,1,G,G — find argmax
        flat = img.view(img.size(0), -1)
        return flat.argmax(-1)

    def train_flat():
        model = FlatPerc().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, img, y = tr["nl"].to(DEVICE), tr["img"].to(DEVICE), tr["y"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(nl[idx], img[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(nl[:400], img[:400]), y[:400]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ProgPerc().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, img = tr["nl"].to(DEVICE), tr["img"].to(DEVICE)
        start_lab = start_labels(img)
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
                _, _, start_lg = model.start_from_img(img[idx])
                loss = (
                    F.cross_entropy(logits["dir"], tr["dir"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["twice"], tr["twice"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["opp"], tr["opp"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["has_after"], tr["has_after"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["after"], tr["after"].to(DEVICE)[idx])
                    + F.cross_entropy(start_lg, start_lab[idx])
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
            nl, img, y = data["nl"].to(DEVICE), data["img"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(nl, img), y)
            out[f"Prog_{name}"] = cell_acc(
                prog.execute_from_logits(prog.program_logits(nl), img), y
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
print("\n===== P PERCEPTION SUMMARY =====")
print(f"Flat: ID={ms('Flat_id')[0]:.3f} OOD={fod:.3f}")
print(f"Prog: ID={ms('Prog_id')[0]:.3f} OOD={pod:.3f}")
print(f"wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.50 and wins >= 2:
    outcome, note = "PASS", "pixels→start + NL→program→execute beats flat"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better with perception"
else:
    outcome, note = "UNRESOLVED", "insufficient gap"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase P perception mock grid pixels",
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
- Experiment: **Phase P Perception** grid pixels → start + NL→program→execute
- Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
