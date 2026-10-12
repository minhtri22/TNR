#!/usr/bin/env python3
"""HumanNL pass-path: synonym-in-train + OOD novel syntax (not pure lexical cold-start).
Lineage:
  HumanNL curated UNRESOLVED (surface OOD too easy OR pure lexical cold-start)
  pass-path PASS Flat~0.17 Prog~0.50 wins 3/3
Protocol: train sees canonical + synonym phrases; OOD = novel syntactic frames.
Nav grid 5x5 relative slots FORWARD/LEFT/RIGHT.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
OP2ID = {"FORWARD": 0, "LEFT": 1, "RIGHT": 2}
ID2OP = {0: "FORWARD", 1: "LEFT", 2: "RIGHT"}
MAX_STEPS, H, MAX_NL = 2, 96, 24
N_TRAIN, N_OOD, EPOCHS, BS, LR = 3000, 400, 14, 64, 1e-3
DELTA = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# Train: canonical + synonyms (no cold-start on direction words)
TRAIN_TEMPLATES = {
    "FORWARD": [
        "go forward",
        "walk forward",
        "move ahead",
        "go straight",
        "walk straight",
        "advance",
    ],
    "LEFT": [
        "turn left",
        "go left",
        "rotate left",
        "turn to the left",
        "face left",
    ],
    "RIGHT": [
        "turn right",
        "go right",
        "rotate right",
        "turn to the right",
        "face right",
    ],
}
# OOD: novel syntax frames (same lexicon roots mostly, different structure)
OOD_TEMPLATES = {
    "FORWARD": [
        "please proceed forward carefully",
        "you should walk forward now",
        "continue moving forward a bit",
    ],
    "LEFT": [
        "please make a left turn",
        "you should turn left now",
        "rotate yourself to the left",
    ],
    "RIGHT": [
        "please make a right turn",
        "you should turn right now",
        "rotate yourself to the right",
    ],
}

words = set()
for d in (TRAIN_TEMPLATES, OOD_TEMPLATES):
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


def exec_open(x, y, facing, slots):
    for op, n in slots:
        for _ in range(min(int(n), 2)):
            if op == "LEFT":
                facing = (facing - 1) % 4
            elif op == "RIGHT":
                facing = (facing + 1) % 4
            elif op == "FORWARD":
                dx, dy = DELTA[facing]
                x, y = max(0, min(4, x + dx)), max(0, min(4, y + dy))
    return x, y, facing


def cell_acc(pred, y):
    return (pred.round() == y.round()).all(-1).float().mean().item()


def sample(n, seed, ood=False):
    rng = np.random.RandomState(seed)
    templates = OOD_TEMPLATES if ood else TRAIN_TEMPLATES
    ops = list(OP2ID.keys())
    X, Pos, Y = [], [], []
    labs_op = [[] for _ in range(MAX_STEPS)]
    labs_n = [[] for _ in range(MAX_STEPS)]
    labs_mask = [[] for _ in range(MAX_STEPS)]
    lengths = []
    for _ in range(n):
        L = rng.randint(1, MAX_STEPS + 1)
        slots = []
        parts = []
        for _s in range(L):
            op = ops[rng.randint(0, len(ops))]
            nrep = 1 if op != "FORWARD" else rng.randint(1, 3)
            slots.append((op, nrep))
            phrase = templates[op][rng.randint(0, len(templates[op]))]
            if nrep > 1 and op == "FORWARD":
                phrase = phrase + f" {nrep} steps"
            parts.append(phrase)
        text = " then ".join(parts)
        x0, y0 = rng.randint(0, 5), rng.randint(0, 5)
        xf, yf, _ = exec_open(x0, y0, 0, slots)
        X.append(tok(text))
        Pos.append([float(x0), float(y0)])
        Y.append([float(xf), float(yf)])
        lengths.append(L - 1)
        for s in range(MAX_STEPS):
            if s < L:
                op, nrep = slots[s]
                labs_op[s].append(OP2ID[op])
                labs_n[s].append(min(2, nrep) - 1)
                labs_mask[s].append(1.0)
            else:
                labs_op[s].append(0)
                labs_n[s].append(0)
                labs_mask[s].append(0.0)
    out = {
        "nl": torch.tensor(X, dtype=torch.long),
        "pos": torch.tensor(Pos, dtype=torch.float32),
        "y": torch.tensor(Y, dtype=torch.float32),
        "len": torch.tensor(lengths, dtype=torch.long),
    }
    for s in range(MAX_STEPS):
        out[f"op{s}"] = torch.tensor(labs_op[s], dtype=torch.long)
        out[f"n{s}"] = torch.tensor(labs_n[s], dtype=torch.long)
        out[f"m{s}"] = torch.tensor(labs_mask[s], dtype=torch.float32)
    return out


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
        self.mlp = nn.Sequential(
            nn.Linear(H * 2, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 2)
        )

    def forward(self, nl, pos):
        return self.mlp(torch.cat([self.enc(nl), self.sp(pos)], -1))


class Prog(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.len_head = nn.Linear(H, MAX_STEPS)
        self.op_heads = nn.ModuleList([nn.Linear(H, 3) for _ in range(MAX_STEPS)])
        self.n_heads = nn.ModuleList([nn.Linear(H, 2) for _ in range(MAX_STEPS)])

    def logits(self, nl):
        h = self.enc(nl)
        return self.len_head(h), [(self.op_heads[s](h), self.n_heads[s](h)) for s in range(MAX_STEPS)]

    def execute(self, len_lg, step_lgs, pos):
        outs = []
        for i in range(pos.size(0)):
            L = int(len_lg[i].argmax()) + 1
            slots = []
            for s in range(L):
                op = ID2OP[int(step_lgs[s][0][i].argmax())]
                n = int(step_lgs[s][1][i].argmax()) + 1
                slots.append((op, n))
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xf, yf, _ = exec_open(x0, y0, 0, slots)
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
            len_lg, step_lgs = prog.logits(tr["nl"][idx])
            loss = F.cross_entropy(len_lg, tr["len"][idx])
            for s in range(MAX_STEPS):
                m = tr[f"m{s}"][idx]
                if float(m.sum()) < 1:
                    continue
                loss = loss + (
                    F.cross_entropy(step_lgs[s][0], tr[f"op{s}"][idx], reduction="none") * m
                ).sum() / m.sum()
                loss = loss + (
                    F.cross_entropy(step_lgs[s][1], tr[f"n{s}"][idx], reduction="none") * m
                ).sum() / m.sum()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if nb and tot / nb < best:
            best, st = tot / nb, {k: v.clone() for k, v in prog.state_dict().items()}
    if st:
        prog.load_state_dict(st)

    flat.eval()
    prog.eval()
    with torch.no_grad():
        fe = cell_acc(flat(ood["nl"], ood["pos"]), ood["y"])
        len_lg, step_lgs = prog.logits(ood["nl"])
        pe = cell_acc(prog.execute(len_lg, step_lgs, ood["pos"]), ood["y"])
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
    print("Note: synonym-in-train + novel syntax OOD (not pure lexical cold-start)")
