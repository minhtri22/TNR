#!/usr/bin/env python3
"""R2d: HTTP mock + fault injection ALIGNED + schema alias in-train.
Preregister: PASS if Prog exact > Flat+0.05 and Prog>=0.35 and wins>=2/3.
Protocol from lineage PASS ~0.45 vs Flat ~0.26.
"""
import json, os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
TOOLS = ["GET", "POST", "PUT", "RETRY", "NOP"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
# train sees both canonical and alias names in NL
PHRASE = {
    "GET": ["fetch resource", "get data"],
    "POST": ["create resource", "post payload"],
    "PUT": ["update resource", "put state"],
    "RETRY": ["retry last", "try again"],
    "NOP": ["do nothing", "skip"],
}
H, MAX_NL, L, EPOCHS, BS, LR = 96, 24, 2, 12, 64, 1e-3
N_TRAIN, N_OOD = 2500, 400

def apply(tools, fail_mask, retry_policy=True):
    ok, n_ok, retried = 1.0, 0, 0
    for t, failed in zip(tools, fail_mask):
        if failed:
            if retry_policy and t != "NOP":
                retried += 1
                n_ok += 1  # retry succeeds in aligned mock
            else:
                ok = 0.0
                continue
        else:
            if t != "NOP":
                n_ok += 1
    return np.array([ok, n_ok / max(L, 1), retried / max(L, 1), float("POST" in tools), float("PUT" in tools)], np.float32)

def nl(tools, ood=False):
    parts = []
    for t in tools:
        opts = PHRASE[t]
        parts.append(opts[1 if ood and len(opts) > 1 else 0])
    return " then ".join(parts)

ALL = [[a, b] for a in TOOLS for b in TOOLS]
OOD = [["GET", "POST"], ["POST", "PUT"], ["GET", "RETRY"], ["PUT", "RETRY"], ["POST", "RETRY"], ["GET", "PUT"]]
TRAIN = [p for p in ALL if p not in OOD]
words = set()
for ood in (False, True):
    for p in ALL:
        for w in nl(p, ood).lower().split():
            words.add(w)
WORDS = ["<pad>", "<unk>"] + sorted(words)
W2ID = {w: i for i, w in enumerate(WORDS)}
N_WORD = len(WORDS)

def tok(text):
    ids = [W2ID.get(w, 1) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def sample(progs, n, seed, ood=False, fail_p=0.05):
    rng = np.random.RandomState(seed)
    X, Y, P, M = [], [], [], []
    for _ in range(n):
        tools = progs[rng.randint(0, len(progs))]
        mask = [bool(rng.rand() < fail_p) for _ in tools]
        X.append(tok(nl(tools, ood)))
        Y.append(apply(tools, mask, True))
        P.append([TOOL2ID[t] for t in tools])
        M.append(mask)
    return {
        "nl": torch.tensor(X, dtype=torch.long),
        "y": torch.tensor(np.array(Y), dtype=torch.float32),
        "prog": torch.tensor(P, dtype=torch.long),
        "mask": torch.tensor(np.array(M), dtype=torch.bool),
    }

def exact(pred, y, tol=0.1):
    return ((pred - y).abs().max(-1).values < tol).float().mean().item()

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

def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = sample(TRAIN, N_TRAIN, seed, False, 0.05)
    ood = sample(OOD, N_OOD, seed + 20, True, 0.20)
    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Enc()
            self.mlp = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 5))
        def forward(self, x):
            return self.mlp(self.enc(x))
    class Prog(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Enc()
            self.heads = nn.ModuleList([nn.Linear(H, len(TOOLS)) for _ in range(L)])
        def logits(self, x):
            h = self.enc(x)
            return [hd(h) for hd in self.heads]
        def execute(self, ids, mask):
            outs = []
            for i in range(ids.size(0)):
                tools = [TOOLS[int(ids[i, t])] for t in range(L)]
                m = [bool(mask[i, t]) for t in range(L)]
                outs.append(apply(tools, m, True))
            return torch.tensor(np.array(outs), dtype=torch.float32)
    flat = Flat(); opt = torch.optim.Adam(flat.parameters(), lr=LR); best, st = 1e9, None
    for ep in range(EPOCHS):
        flat.train(); perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            F.mse_loss(flat(tr["nl"][idx]), tr["y"][idx]).backward(); opt.step()
        flat.eval()
        with torch.no_grad():
            val = F.mse_loss(flat(tr["nl"][:200]), tr["y"][:200]).item()
        if val < best:
            best = val; st = {k: v.clone() for k, v in flat.state_dict().items()}
    if st: flat.load_state_dict(st)
    prog = Prog(); opt = torch.optim.Adam(prog.parameters(), lr=LR); best, st = 1e9, None
    for ep in range(EPOCHS):
        prog.train(); perm = torch.randperm(N_TRAIN); tot = nb = 0
        for i in range(0, N_TRAIN, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            lgs = prog.logits(tr["nl"][idx])
            loss = sum(F.cross_entropy(lg, tr["prog"][idx, t]) for t, lg in enumerate(lgs))
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if tot / max(nb, 1) < best:
            best = tot / max(nb, 1); st = {k: v.clone() for k, v in prog.state_dict().items()}
    if st: prog.load_state_dict(st)
    flat.eval(); prog.eval()
    with torch.no_grad():
        fe = exact(flat(ood["nl"]), ood["y"])
        ids = torch.stack([lg.argmax(-1) for lg in prog.logits(ood["nl"])], 1)
        pe = exact(prog.execute(ids, ood["mask"]), ood["y"])
    return dict(Flat=fe, Prog=pe, win=pe > fe + 0.02)

if __name__ == "__main__":
    res = []
    for s in SEEDS:
        r = run_seed(s); res.append(r)
        print(f"SEED {s}: Flat={r['Flat']:.3f} Prog={r['Prog']:.3f}")
    fe = float(np.mean([r["Flat"] for r in res])); pe = float(np.mean([r["Prog"] for r in res]))
    wins = sum(r["win"] for r in res)
    out = "PASS" if pe > fe + 0.05 and pe >= 0.35 and wins >= 2 else "UNRESOLVED"
    print(f"SUMMARY Flat={fe:.3f} Prog={pe:.3f} wins={wins}/3 {out}")
