#!/usr/bin/env python3
"""HumanNL-SAIL: import SAIL human corpus, auto-annotate slots, train Prog vs Flat.
Lineage: PASS (import + annotate + eval on auto slots)
Data: data/human_nl/sail_auto_annotated_v2.jsonl (or v1 fallback)
Slots are relative FORWARD/LEFT/RIGHT with optional repeat count.
"""
import re, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
OP2ID = {"FORWARD": 0, "LEFT": 1, "RIGHT": 2}
ID2OP = {0: "FORWARD", 1: "LEFT", 2: "RIGHT"}
MAX_STEPS, H, MAX_NL = 3, 96, 24
N_TRAIN, N_OOD, EPOCHS, BS, LR = 2000, 300, 12, 64, 1e-3
DELTA = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}


def exec_open(x, y, facing, slots):
    for op, n in slots:
        for _ in range(min(int(n), 3)):
            if op == "LEFT":
                facing = (facing - 1) % 4
            elif op == "RIGHT":
                facing = (facing + 1) % 4
            elif op == "FORWARD":
                dx, dy = DELTA[facing]
                x, y = x + dx, y + dy
    return x, y, facing


def normalize_slots(raw):
    """Accept [['LEFT',2]] or ['LEFT','LEFT'] style."""
    if not raw:
        return []
    if isinstance(raw[0], (list, tuple)):
        out = []
        for item in raw:
            if len(item) >= 2:
                op, n = str(item[0]).upper(), int(item[1])
            else:
                op, n = str(item[0]).upper(), 1
            if op in OP2ID:
                out.append((op, max(1, min(3, n))))
        return out[:MAX_STEPS]
    # list of op strings
    slots = []
    for op in raw:
        op = str(op).upper()
        if op not in OP2ID:
            continue
        if slots and slots[-1][0] == op:
            slots[-1] = (op, min(3, slots[-1][1] + 1))
        else:
            slots.append((op, 1))
    return slots[:MAX_STEPS]


def cell_acc(pred, y):
    return (pred.round() == y.round()).all(-1).float().mean().item()


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "human_nl")
    candidates = [
        os.path.join(base, "sail_auto_annotated_v2.jsonl"),
        os.path.join(base, "sail_auto_annotated.jsonl"),
        os.path.join(base, "sail_trace_gold.jsonl"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("No SAIL annotated jsonl found under data/human_nl/")
    rows = [json.loads(l) for l in open(path)]
    bank = []
    for a in rows:
        slots = normalize_slots(a.get("slots") or [])
        if not slots:
            continue
        bank.append({"text": a["text"], "slots": slots, "start": a.get("start") or {}})
    print(f"source={os.path.basename(path)} bank={len(bank)}")

    words = set()
    for a in bank:
        for w in re.findall(r"[a-z0-9]+", a["text"].lower()):
            words.add(w)
    WORDS = ["<pad>", "<unk>"] + sorted(words)
    W2ID = {w: i for i, w in enumerate(WORDS)}
    N_WORD = len(WORDS)

    def tok(text):
        ids = [W2ID.get(w, 1) for w in re.findall(r"[a-z0-9]+", text.lower())]
        return (ids + [0] * MAX_NL)[:MAX_NL]

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(bank))
    n_tr = int(0.75 * len(bank))
    train_bank = [bank[i] for i in idx[:n_tr]]
    ood_bank = [bank[i] for i in idx[n_tr:]]

    def sample(bnk, n, seed):
        rng = np.random.RandomState(seed)
        X, Pos, Y = [], [], []
        labs_op = [[] for _ in range(MAX_STEPS)]
        labs_n = [[] for _ in range(MAX_STEPS)]
        labs_mask = [[] for _ in range(MAX_STEPS)]
        lengths = []
        for _ in range(n):
            a = bnk[rng.randint(0, len(bnk))]
            st = a["start"] if isinstance(a["start"], dict) else {}
            x0, y0 = int(st.get("x", 2)), int(st.get("y", 7))
            xf, yf, _ = exec_open(x0, y0, 0, a["slots"])
            X.append(tok(a["text"]))
            Pos.append([float(x0), float(y0)])
            Y.append([float(xf), float(yf)])
            L = len(a["slots"])
            lengths.append(L - 1)
            for s in range(MAX_STEPS):
                if s < L:
                    op, nrep = a["slots"][s]
                    labs_op[s].append(OP2ID[op])
                    labs_n[s].append(min(3, nrep) - 1)
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
            self.n_heads = nn.ModuleList([nn.Linear(H, 3) for _ in range(MAX_STEPS)])

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

    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr = sample(train_bank, N_TRAIN, seed)
        ood = sample(ood_bank, N_OOD, seed + 20)

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
        results.append(dict(Flat=fe, Prog=pe, win=pe > fe + 0.02))
        print(f"SEED {seed}: Flat={fe:.3f} Prog={pe:.3f}")

    fe = float(np.mean([r["Flat"] for r in results]))
    pe = float(np.mean([r["Prog"] for r in results]))
    wins = sum(r["win"] for r in results)
    out = "PASS" if pe > fe + 0.05 and pe >= 0.35 and wins >= 2 else "UNRESOLVED"
    print(f"SUMMARY Flat={fe:.3f} Prog={pe:.3f} wins={wins}/3 {out}")


if __name__ == "__main__":
    main()
