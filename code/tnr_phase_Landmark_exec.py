#!/usr/bin/env python3
"""Landmark-Exec: NL → GOTO/FACE slots → greedy path to fixed landmark cells.
Lineage: PASS Flat~0.31 Prog~0.71 wins 3/3
Mock landmarks on a small grid (no full SAIL map required).
"""
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
H, MAX_NL, EPOCHS, BS, LR = 96, 20, 12, 64, 1e-3
N_TRAIN, N_OOD = 2500, 400

# Fixed landmark cells on 5x5 mock
LANDMARKS = {
    "easel": (0, 0),
    "sofa": (0, 4),
    "bar": (4, 0),
    "hatrack": (4, 4),
    "lamp": (2, 2),
}
LM_NAMES = sorted(LANDMARKS.keys())
LM2ID = {n: i for i, n in enumerate(LM_NAMES)}
N_LM = len(LM_NAMES)
FACE2ID = {"north": 0, "east": 1, "south": 2, "west": 3}
ID2FACE = {v: k for k, v in FACE2ID.items()}
DELTA = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

TRAIN_PHRASE = {
    "easel": ["go to the easel", "walk to easel"],
    "sofa": ["go to the sofa", "walk to sofa"],
    "bar": ["go to the bar", "walk to bar"],
    "hatrack": ["go to the hatrack", "walk to hatrack"],
    "lamp": ["go to the lamp", "walk to lamp"],
}
OOD_PHRASE = {
    "easel": ["please proceed to the easel", "make your way to easel"],
    "sofa": ["please proceed to the sofa", "make your way to sofa"],
    "bar": ["please proceed to the bar", "make your way to bar"],
    "hatrack": ["please proceed to the hatrack", "make your way to hatrack"],
    "lamp": ["please proceed to the lamp", "make your way to lamp"],
}

words = set()
for d in (TRAIN_PHRASE, OOD_PHRASE):
    for phrases in d.values():
        for p in phrases:
            for w in p.lower().split():
                words.add(w)
WORDS = ["<pad>", "<unk>"] + sorted(words)
W2ID = {w: i for i, w in enumerate(WORDS)}
N_WORD = len(WORDS)


def tok(text):
    ids = [W2ID.get(w, 1) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]


def path_to(x, y, tx, ty, max_steps=6):
    for _ in range(max_steps):
        if (x, y) == (tx, ty):
            break
        # greedy manhattan on open grid
        if abs(tx - x) >= abs(ty - y):
            x = x + (1 if tx > x else -1 if tx < x else 0)
        else:
            y = y + (1 if ty > y else -1 if ty < y else 0)
        x = max(0, min(4, x))
        y = max(0, min(4, y))
    return x, y


def sample(n, seed, ood=False):
    rng = np.random.RandomState(seed)
    phrases = OOD_PHRASE if ood else TRAIN_PHRASE
    X, Pos, Y, labs = [], [], [], []
    for _ in range(n):
        name = LM_NAMES[rng.randint(0, N_LM)]
        text = phrases[name][rng.randint(0, len(phrases[name]))]
        x0, y0 = rng.randint(0, 5), rng.randint(0, 5)
        tx, ty = LANDMARKS[name]
        xf, yf = path_to(x0, y0, tx, ty)
        X.append(tok(text))
        Pos.append([float(x0), float(y0)])
        Y.append([float(xf), float(yf)])
        labs.append(LM2ID[name])
    return {
        "nl": torch.tensor(X, dtype=torch.long),
        "pos": torch.tensor(Pos, dtype=torch.float32),
        "y": torch.tensor(Y, dtype=torch.float32),
        "lm": torch.tensor(labs, dtype=torch.long),
    }


def cell_acc(pred, y):
    return (pred.round() == y.round()).all(-1).float().mean().item()


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_WORD, H, padding_idx=0)
        self.lstm = nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(H * 2, H)

    def forward(self, nl):
        e = self.emb(nl)
        lens = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))


class Flat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.sp = nn.Linear(2, H)
        self.mlp = nn.Sequential(nn.Linear(H * 2, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 2))

    def forward(self, nl, pos):
        return self.mlp(torch.cat([self.enc(nl), self.sp(pos)], -1))


class Prog(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.lm = nn.Linear(H, N_LM)

    def logits(self, nl):
        return self.lm(self.enc(nl))

    def execute(self, lm_ids, pos):
        outs = []
        for i in range(pos.size(0)):
            name = LM_NAMES[int(lm_ids[i])]
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            tx, ty = LANDMARKS[name]
            xf, yf = path_to(x0, y0, tx, ty)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample(N_TRAIN, seed, False)
    ood = sample(N_OOD, seed + 20, True)

    flat = Flat()
    opt = torch.optim.Adam(flat.parameters(), lr=LR)
    best, st = 1e9, None
    for ep in range(EPOCHS):
        flat.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BS):
            idx = perm[i : i + BS]
            opt.zero_grad()
            F.mse_loss(flat(tr["nl"][idx], tr["pos"][idx]), tr["y"][idx]).backward()
            opt.step()
        flat.eval()
        with torch.no_grad():
            val = F.mse_loss(flat(tr["nl"][:200], tr["pos"][:200]), tr["y"][:200]).item()
        if val < best:
            best, st = val, {k: v.clone() for k, v in flat.state_dict().items()}
    if st:
        flat.load_state_dict(st)

    prog = Prog()
    opt = torch.optim.Adam(prog.parameters(), lr=LR)
    best, st = 1e9, None
    for ep in range(EPOCHS):
        prog.train()
        perm = torch.randperm(N_TRAIN)
        tot = nb = 0
        for i in range(0, N_TRAIN, BS):
            idx = perm[i : i + BS]
            opt.zero_grad()
            loss = F.cross_entropy(prog.logits(tr["nl"][idx]), tr["lm"][idx])
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if tot / max(nb, 1) < best:
            best, st = tot / max(nb, 1), {k: v.clone() for k, v in prog.state_dict().items()}
    if st:
        prog.load_state_dict(st)

    flat.eval()
    prog.eval()
    with torch.no_grad():
        fe = cell_acc(flat(ood["nl"], ood["pos"]), ood["y"])
        lm_ids = prog.logits(ood["nl"]).argmax(-1)
        pe = cell_acc(prog.execute(lm_ids, ood["pos"]), ood["y"])
    return dict(Flat=fe, Prog=pe, win=pe > fe + 0.02)


if __name__ == "__main__":
    res = []
    for s in SEEDS:
        r = run_seed(s)
        res.append(r)
        print(f"SEED {s}: Flat={r['Flat']:.3f} Prog={r['Prog']:.3f} win={r['win']}")
    fe = float(np.mean([r["Flat"] for r in res]))
    pe = float(np.mean([r["Prog"] for r in res]))
    wins = sum(r["win"] for r in res)
    out = "PASS" if pe > fe + 0.05 and pe >= 0.35 and wins >= 2 else "UNRESOLVED"
    print(f"SUMMARY Flat={fe:.3f} Prog={pe:.3f} wins={wins}/3 {out}")
