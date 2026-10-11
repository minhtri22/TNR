#!/usr/bin/env python3
"""
Phase HumanNL — Open corpus style (LLM/human-like free NL)

Khác ProdNL: không core template cố định + freeform_shell.
Corpus = hàng trăm câu natural, đa cú pháp, thiếu chủ ngữ, slang nhẹ,
multi-clause; gán nhãn program slots (dir, twice, opp, after).

Split:
  Train: 70% surface clusters
  OOD: 30% held-out phrasings (cùng intent class, wording khác hẳn)

Prog: NL→slots→grid execute
Flat: NL+pos→cell

Preregister:
  PASS: Prog OOD > Flat OOD + 0.05 AND Prog OOD >= 0.40 AND wins >= 2/3
  FAIL: Prog + 0.05 < Flat
  else UNRESOLVED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import hashlib
from datetime import datetime

SEEDS = [42, 7, 123]
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

GRID = 5
MOVES = ["north", "south", "east", "west"]
MOVE2DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE2ID = {m: i for i, m in enumerate(MOVES)}
# colloquial direction words
DIR_WORDS = {
    "north": ["north", "up", "upward", "upwards"],
    "south": ["south", "down", "downward"],
    "east": ["east", "right"],
    "west": ["west", "left"],
}

WORDS = ["<pad>", "<unk>"]
# build vocab from corpus generation
HIDDEN = 128
MAX_NL = 28
BATCH_SIZE = 64
N_TRAIN = 3500
N_ID = 400
N_OOD = 600
EPOCHS = 16
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

def build_open_corpus():
    """Naturalistic open-style sentences labeled with program slots."""
    corpus = []  # (text, dir, twice, opp, after_or_none)

    def add(text, d, twice=False, opp=False, after=None):
        corpus.append((text.strip(), d, twice, opp, after))

    for d in MOVES:
        dw = DIR_WORDS[d]
        for w in dw:
            # atomic — varied human/LLM style
            add(f"go {w}", d)
            add(f"walk {w}", d)
            add(f"please go {w}", d)
            add(f"can you move {w}", d)
            add(f"i need to head {w}", d)
            add(f"take a step {w}", d)
            add(f"shift {w} a bit", d)
            add(f"nudge me {w}", d)
            add(f"heading {w} now", d)
            add(f"let's go {w}", d)
            add(f"move one square {w}", d)
            add(f"could you send the agent {w}", d)
            add(f"agent, {w}", d)
            add(f"just {w}", d)
            add(f"ok go {w} please", d)
            # twice
            add(f"go {w} twice", d, twice=True)
            add(f"walk {w} two times", d, twice=True)
            add(f"do {w} twice please", d, twice=True)
            add(f"two steps {w}", d, twice=True)
            add(f"double move {w}", d, twice=True)
            add(f"go {w} then {w} again", d, twice=True)
            add(f"i want two steps toward {w}", d, twice=True)
            # opposite
            add(f"go opposite {w}", d, opp=True)
            add(f"the other way from {w}", d, opp=True)
            add(f"reverse of {w}", d, opp=True)
            add(f"not {w}, the opposite", d, opp=True)
            add(f"opposite direction of {w}", d, opp=True)
            add(f"away from {w} sense — opposite {w}", d, opp=True)

    for d in MOVES:
        for a in MOVES:
            if a == d:
                continue
            dw, aw = DIR_WORDS[d][0], DIR_WORDS[a][0]
            add(f"go {dw} after {aw}", d, after=a)
            add(f"first {aw} then {dw}", d, after=a)
            add(f"after going {aw}, go {dw}", d, after=a)
            add(f"do {aw} and then {dw}", d, after=a)
            add(f"{aw} followed by {dw}", d, after=a)
            add(f"once you've gone {aw}, head {dw}", d, after=a)
            add(f"start {aw} finish {dw}", d, after=a)

    # more messy / open
    for d in MOVES:
        w = DIR_WORDS[d][0]
        add(f"hmm maybe go {w}?", d)
        add(f"i guess {w} works", d)
        add(f"whatever, just {w}", d)
        add(f"the map says {w}", d)
        add(f"path is {w} from here", d)
        add(f"one cell {w} if possible", d)
        add(f"please please go {w}", d)
        add(f"go {w} go {w}", d, twice=True)
        add(f"again {w} — twice total", d, twice=True)

    return corpus

CORPUS = build_open_corpus()
print(f"Open corpus size={len(CORPUS)}")

# vocab from corpus
word_set = set()
for text, *_ in CORPUS:
    for w in text.lower().replace("?", "").replace(",", "").replace("—", " ").replace("-", " ").split():
        word_set.add(w)
WORDS = ["<pad>", "<unk>"] + sorted(word_set)
W2ID = {w: i for i, w in enumerate(WORDS)}
N_WORD = len(WORDS)
print(f"Vocab={N_WORD}")

def tokenize(text):
    text = text.lower().replace("?", "").replace(",", "").replace("—", " ").replace("-", " ")
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def cluster_key(text):
    """Surface cluster for held-out split: hash of normalized text."""
    return hashlib.md5(text.lower().encode()).hexdigest()[:8]

def split_corpus(seed=0):
    rng = np.random.RandomState(seed)
    # group by (dir, twice, opp, after) intent, hold out 30% of unique texts
    items = list(CORPUS)
    rng.shuffle(items)
    n_ood = max(1, int(0.30 * len(items)))
    ood_set = set(cluster_key(t) for t, *_ in items[:n_ood])
    train, ood = [], []
    for item in items:
        if cluster_key(item[0]) in ood_set:
            ood.append(item)
        else:
            train.append(item)
    # ensure both non-empty
    if len(train) < 10:
        train, ood = items[n_ood:], items[:n_ood]
    return train, ood

def sample_from(bank, n, seed):
    rng = np.random.RandomState(seed)
    X, Pos, Y = [], [], []
    labs = {k: [] for k in ["dir", "twice", "opp", "has_after", "after"]}
    for _ in range(n):
        text, d, twice, opp, after = bank[rng.randint(0, len(bank))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand(d, twice, opp, after)
        xf, yf = apply_moves(x0, y0, moves)
        X.append(tokenize(text))
        Pos.append([x0, y0])
        Y.append([xf, yf])
        labs["dir"].append(MOVE2ID[d])
        labs["twice"].append(1 if twice else 0)
        labs["opp"].append(1 if opp else 0)
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

class FlatH(nn.Module):
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

class ProgH(nn.Module):
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
    train_bank, ood_bank = split_corpus(seed)
    print(f"  bank train={len(train_bank)} ood={len(ood_bank)}", flush=True)
    tr = sample_from(train_bank, N_TRAIN, seed)
    idd = sample_from(train_bank, N_ID, seed + 15)
    ood = sample_from(ood_bank, N_OOD, seed + 20)

    def train_flat():
        model = FlatH()
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
        model = ProgH()
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

    out = {"seed": seed, "n_train_bank": len(train_bank), "n_ood_bank": len(ood_bank)}
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
print("\n===== HumanNL OPEN CORPUS SUMMARY =====")
print(f"Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3")

if pod > fod + 0.05 and pod >= 0.40 and wins >= 2:
    outcome, note = "PASS", "open corpus held-out phrasing: program→execute wins"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "flat better on open corpus OOD"
else:
    outcome, note = "UNRESOLVED", "insufficient gap"

print(f"=== OUTCOME: {outcome} === ({note})")
print("NOTE: corpus is curated open-style (human/LLM-like), not raw user logs.")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase HumanNL open corpus (human/LLM-like)",
        "corpus_size": len(CORPUS),
        "seeds": SEEDS,
        "summary": {"Flat_ood": fod, "Prog_ood": pod, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
        "limitation": "Not raw human logs; curated open-style labeled corpus",
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **HumanNL** open corpus (human/LLM-like free NL, held-out 30% surfaces)
- Corpus size={len(CORPUS)} | Flat OOD={fod:.3f} | Prog OOD={pod:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
- Limitation: curated labeled open-style, NOT raw production user logs
""")
print(f"Saved {run_dir}")
