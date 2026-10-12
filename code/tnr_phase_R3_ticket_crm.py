#!/usr/bin/env python3
"""R3: narrow production Ticket/CRM domain — NL→tool program→execute exact.
Lineage: PASS Prog~0.52 vs Flat 0.00.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
TOOLS = ["CREATE", "ASSIGN", "SET_STATUS", "ADD_NOTE", "CLOSE", "NOP"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
H, MAX_NL, L, EPOCHS, BS, LR = 128, 24, 2, 16, 64, 1e-3
N_TRAIN, N_OOD = 3500, 500

def apply(tools):
    exists = status = assignee = notes = closed = 0
    for t in tools:
        if t == "CREATE": exists, status, closed = 1, 0, 0
        elif t == "ASSIGN" and exists and not closed: assignee = (assignee + 1) % 3
        elif t == "SET_STATUS" and exists and not closed: status = min(2, status + 1)
        elif t == "ADD_NOTE" and exists and not closed: notes = min(3, notes + 1)
        elif t == "CLOSE" and exists: closed, status = 1, 2
    return np.array([exists, status / 2.0, assignee / 2.0, notes / 3.0, closed], np.float32)

def nl(tools, ood=False):
    m = {
        "CREATE": ("open a new ticket", "please create ticket"),
        "ASSIGN": ("assign to next agent", "hand off to teammate"),
        "SET_STATUS": ("advance status", "move status forward"),
        "ADD_NOTE": ("add a note", "leave a comment"),
        "CLOSE": ("close the ticket", "mark resolved"),
        "NOP": ("do nothing", "skip"),
    }
    return " then ".join(m[t][1 if ood else 0] for t in tools)

ALL = [[a, b] for a in TOOLS for b in TOOLS]
OOD = [["CREATE", "CLOSE"], ["CREATE", "ASSIGN"], ["ASSIGN", "CLOSE"],
       ["CREATE", "ADD_NOTE"], ["SET_STATUS", "CLOSE"], ["ADD_NOTE", "CLOSE"]]
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

def sample(progs, n, seed, ood=False):
    rng = np.random.RandomState(seed)
    X, Y, P = [], [], []
    for _ in range(n):
        tools = progs[rng.randint(0, len(progs))]
        X.append(tok(nl(tools, ood))); Y.append(apply(tools)); P.append([TOOL2ID[t] for t in tools])
    return {
        "nl": torch.tensor(X, dtype=torch.long),
        "y": torch.tensor(np.array(Y), dtype=torch.float32),
        "prog": torch.tensor(P, dtype=torch.long),
    }

def exact(pred, y, tol=0.08):
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
    tr, ood = sample(TRAIN, N_TRAIN, seed, False), sample(OOD, N_OOD, seed + 20, True)
    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Enc(); self.mlp = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 5))
        def forward(self, x): return self.mlp(self.enc(x))
    class Prog(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Enc(); self.heads = nn.ModuleList([nn.Linear(H, len(TOOLS)) for _ in range(L)])
        def logits(self, x):
            h = self.enc(x); return [hd(h) for hd in self.heads]
        def execute(self, ids):
            outs = []
            for i in range(ids.size(0)):
                tools = [TOOLS[int(ids[i, t])] for t in range(L)]
                outs.append(apply(tools))
            return torch.tensor(np.array(outs), dtype=torch.float32)
    flat = Flat(); opt = torch.optim.Adam(flat.parameters(), lr=LR); best, st = 1e9, None
    for ep in range(EPOCHS):
        flat.train(); perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            F.mse_loss(flat(tr["nl"][idx]), tr["y"][idx]).backward(); opt.step()
        flat.eval()
        with torch.no_grad():
            val = F.mse_loss(flat(tr["nl"][:300]), tr["y"][:300]).item()
        if val < best: best, st = val, {k: v.clone() for k, v in flat.state_dict().items()}
    if st: flat.load_state_dict(st)
    prog = Prog(); opt = torch.optim.Adam(prog.parameters(), lr=LR); best, st = 1e9, None
    for ep in range(EPOCHS):
        prog.train(); perm = torch.randperm(N_TRAIN); tot = nb = 0
        for i in range(0, N_TRAIN, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            lgs = prog.logits(tr["nl"][idx])
            loss = sum(F.cross_entropy(lg, tr["prog"][idx, t]) for t, lg in enumerate(lgs))
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if tot / max(nb, 1) < best: best, st = tot / max(nb, 1), {k: v.clone() for k, v in prog.state_dict().items()}
    if st: prog.load_state_dict(st)
    flat.eval(); prog.eval()
    with torch.no_grad():
        fe = exact(flat(ood["nl"]), ood["y"])
        ids = torch.stack([lg.argmax(-1) for lg in prog.logits(ood["nl"])], 1)
        pe = exact(prog.execute(ids), ood["y"])
    return dict(Flat=fe, Prog=pe, win=pe > fe + 0.02)

if __name__ == "__main__":
    res = [run_seed(s) for s in SEEDS]
    for s, r in zip(SEEDS, res):
        print(f"SEED {s}: Flat={r['Flat']:.3f} Prog={r['Prog']:.3f}")
    fe = float(np.mean([r["Flat"] for r in res])); pe = float(np.mean([r["Prog"] for r in res]))
    wins = sum(r["win"] for r in res)
    out = "PASS" if pe > fe + 0.05 and pe >= 0.35 and wins >= 2 else "UNRESOLVED"
    print(f"SUMMARY Flat={fe:.3f} Prog={pe:.3f} wins={wins}/3 {out}")
