#!/usr/bin/env python3
"""2-ben tools stress (structure OOD, same lexicon): B5 L=3, B1 file domain, B4 confirm, B6 fault.
Lineage: all PASS under structure-OOD protocol.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
H, MAX_NL, EPOCHS, BS, LR = 96, 28, 12, 64, 1e-3
N_TRAIN, N_OOD = 2000, 300
EXACT_TOL = 0.1

def exact_acc(pred, y):
    return ((pred - y).abs().max(-1).values < EXACT_TOL).float().mean().item()

class Enc(nn.Module):
    def __init__(self, nw):
        super().__init__()
        self.emb = nn.Embedding(nw, H, padding_idx=0)
        self.lstm = nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(H * 2, H)
    def forward(self, nl):
        e = self.emb(nl)
        lens = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))

def run_suite(name, tools, apply_fn, phrase, train_progs, ood_progs, L):
    TID = {t: i for i, t in enumerate(tools)}; NT = len(tools)
    def nl(tp): return " then ".join(phrase[t] for t in tp)
    words = set()
    for p in train_progs + ood_progs:
        for w in nl(p).lower().split(): words.add(w)
    WORDS = ["<pad>", "<unk>"] + sorted(words); W2ID = {w: i for i, w in enumerate(WORDS)}; NW = len(WORDS)
    def tok(text):
        ids = [W2ID.get(w, 1) for w in text.lower().split()]
        return (ids + [0] * MAX_NL)[:MAX_NL]
    def sample(progs, n, seed):
        rng = np.random.RandomState(seed); X, Y, prog = [], [], []
        for _ in range(n):
            tp = list(progs[rng.randint(0, len(progs))])
            while len(tp) < L: tp.append("NOP" if "NOP" in tools else tools[0])
            tp = tp[:L]
            X.append(tok(nl(tp))); Y.append(apply_fn(tp)); prog.append([TID[t] for t in tp])
        return {"nl": torch.tensor(X, dtype=torch.long), "y": torch.tensor(np.array(Y), dtype=torch.float32),
                "prog": torch.tensor(prog, dtype=torch.long)}
    res = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tr, ood = sample(train_progs, N_TRAIN, seed), sample(ood_progs, N_OOD, seed + 20)
        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc(NW); self.mlp = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 5))
            def forward(self, x): return self.mlp(self.enc(x))
        class Prog(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc(NW); self.heads = nn.ModuleList([nn.Linear(H, NT) for _ in range(L)])
            def logits(self, x):
                h = self.enc(x); return [hd(h) for hd in self.heads]
            def execute(self, ids):
                outs = []
                for i in range(ids.size(0)):
                    tp = [tools[int(ids[i, t])] for t in range(L)]; outs.append(apply_fn(tp))
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
            fe = exact_acc(flat(ood["nl"]), ood["y"])
            ids = torch.stack([lg.argmax(-1) for lg in prog.logits(ood["nl"])], 1)
            pe = exact_acc(prog.execute(ids), ood["y"])
        res.append(dict(Flat=fe, Prog=pe)); print(f"  {name} {seed}: F={fe:.3f} P={pe:.3f}")
    fe = float(np.mean([r["Flat"] for r in res])); pe = float(np.mean([r["Prog"] for r in res]))
    print(f"{name} SUMMARY Flat={fe:.3f} Prog={pe:.3f}")
    return fe, pe

if __name__ == "__main__":
    TOOLS = ["CREATE", "ASSIGN", "SET_STATUS", "ADD_NOTE", "CLOSE", "NOP"]
    def apply_ticket(tools):
        exists = status = assignee = notes = closed = 0
        for t in tools:
            if t == "CREATE": exists, status, closed = 1, 0, 0
            elif t == "ASSIGN" and exists and not closed: assignee = (assignee + 1) % 3
            elif t == "SET_STATUS" and exists and not closed: status = min(2, status + 1)
            elif t == "ADD_NOTE" and exists and not closed: notes = min(3, notes + 1)
            elif t == "CLOSE" and exists: closed, status = 1, 2
        return np.array([exists, status / 2., assignee / 2., notes / 3., closed], np.float32)
    phrase = {t: f"do {t.lower()}" for t in TOOLS}
    OOD3 = [["CREATE", "ASSIGN", "CLOSE"], ["CREATE", "ADD_NOTE", "CLOSE"], ["CREATE", "SET_STATUS", "CLOSE"],
            ["CREATE", "ASSIGN", "ADD_NOTE"], ["ASSIGN", "SET_STATUS", "CLOSE"], ["CREATE", "ADD_NOTE", "SET_STATUS"]]
    TRAIN3 = []
    for a in TOOLS: TRAIN3.append([a, "NOP", "NOP"])
    for a in TOOLS:
        for b in TOOLS: TRAIN3.append([a, b, "NOP"])
    for a in TOOLS:
        for b in TOOLS:
            for c in TOOLS:
                if [a, b, c] not in OOD3: TRAIN3.append([a, b, c])
    print("===== B5 L=3 =====")
    run_suite("B5", TOOLS, apply_ticket, phrase, TRAIN3, OOD3, 3)

    FTOOLS = ["MKDIR", "WRITE", "READ", "DELETE", "LIST", "NOP"]
    def apply_file(tools):
        has_dir = has_file = n_w = deleted = listed = 0
        for t in tools:
            if t == "MKDIR": has_dir = 1
            elif t == "WRITE": has_file, n_w = 1, min(3, n_w + 1)
            elif t == "DELETE" and has_file: deleted, has_file = 1, 0
            elif t == "LIST": listed = 1
        return np.array([has_dir, has_file, n_w / 3., deleted, listed], np.float32)
    phrase_f = {t: f"do {t.lower()}" for t in FTOOLS}
    OODF = [["MKDIR", "WRITE"], ["WRITE", "DELETE"], ["MKDIR", "LIST"], ["WRITE", "READ"], ["DELETE", "LIST"], ["MKDIR", "DELETE"]]
    TRAINF = []
    for a in FTOOLS: TRAINF.append([a, "NOP"])
    for a in FTOOLS:
        for b in FTOOLS:
            if [a, b] not in OODF: TRAINF.append([a, b])
    print("===== B1 file =====")
    run_suite("B1", FTOOLS, apply_file, phrase_f, TRAINF, OODF, 2)
