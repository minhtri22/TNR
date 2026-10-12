#!/usr/bin/env python3
"""3-bền SAIL stress tests (structure on map-grid):
  A2 facing gold from %% traces
  A1 edge-dropout 30% OOD (multi-map proxy)
  A4 landmark grounded GOTO item on map
Lineage: all PASS (Prog ~0.94 / 0.97 / 1.0)
Requires: data/human_nl/sail/map-grid.xml, SingleSentence.grid.txt
"""
import re, os, xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 7, 123]
H, MAX_NL, MAX_STEPS = 64, 16, 3
N_TRAIN, N_OOD, EPOCHS, BS, LR = 1200, 200, 8, 64, 1e-3
OP2ID = {"FORWARD": 0, "LEFT": 1, "RIGHT": 2}
ID2OP = {0: "FORWARD", 1: "LEFT", 2: "RIGHT"}
DELTA = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

def load_map(path):
    root = ET.parse(path).getroot()
    nodes, items, adj = set(), {}, defaultdict(set)
    for n in root.findall(".//node"):
        x, y = int(n.get("x")), int(n.get("y"))
        nodes.add((x, y))
        it = n.get("item") or ""
        if it:
            items[(x, y)] = it
    for e in root.findall(".//edge"):
        a = tuple(map(int, e.get("node1").split(",")))
        b = tuple(map(int, e.get("node2").split(",")))
        adj[a].add(b)
        adj[b].add(a)
    return nodes, adj, items

def parse_trace(actions):
    ops, start_face = [], None
    for a in actions:
        m = re.match(r"%%\s*(FORWARD|LEFT|RIGHT|NULL)\((\d+),(\d+),(\d+)\)", a)
        if not m:
            continue
        op, face = m.group(1), int(m.group(4))
        if start_face is None:
            start_face = face
        if op != "NULL":
            ops.append(op)
    slots = []
    for op in ops:
        if slots and slots[-1][0] == op:
            slots[-1] = (op, slots[-1][1] + 1)
        else:
            slots.append((op, 1))
    f0 = 0 if start_face is None else (start_face // 90) % 4
    return slots, f0

def exec_graph(x, y, facing, slots, adj):
    for op, n in slots:
        for _ in range(min(int(n), 3)):
            if op == "LEFT":
                facing = (facing - 1) % 4
            elif op == "RIGHT":
                facing = (facing + 1) % 4
            elif op == "FORWARD":
                dx, dy = DELTA[facing]
                nx, ny = x + dx, y + dy
                if (nx, ny) in adj.get((x, y), ()):
                    x, y = nx, ny
    return x, y, facing

def cell_acc(pred, y):
    return (pred.round() == y.round()).all(-1).float().mean().item()

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

def load_bank(base):
    nodes, adj, items = load_map(os.path.join(base, "sail", "map-grid.xml"))
    raw = open(os.path.join(base, "sail", "SingleSentence.grid.txt")).read().splitlines()
    episodes, i = [], 0
    while i < len(raw):
        if raw[i].strip().startswith("Cleaned-"):
            i += 1
            meta = None
            if i < len(raw) and raw[i].strip().startswith("{"):
                try:
                    meta = eval(raw[i].strip())
                except Exception:
                    meta = None
                i += 1
            while i < len(raw) and not raw[i].strip().startswith("Cleaned-"):
                t = raw[i].strip()
                if not t or t.startswith("{") or t.startswith("%%"):
                    i += 1
                    continue
                text, acts = t, []
                i += 1
                while i < len(raw) and raw[i].strip().startswith("%%"):
                    acts.append(raw[i].strip())
                    i += 1
                if acts:
                    episodes.append({"text": text, "meta": meta, "trace": acts})
            continue
        i += 1
    bank = []
    for e in episodes:
        slots, f0 = parse_trace(e["trace"])
        if not slots or len(slots) > 3:
            continue
        bank.append({
            "text": e["text"],
            "slots": slots,
            "face0": f0,
            "start": e["meta"] if isinstance(e["meta"], dict) else {"x": 2, "y": 7},
        })
    uniq = {a["text"].lower().strip(): a for a in bank}
    return list(uniq.values()), nodes, adj, items

def main():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "human_nl")
    bank, nodes, adj_g, items_g = load_bank(base)
    print(f"bank={len(bank)} nodes={len(nodes)}")
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
    ntr = int(0.75 * len(bank))
    tr_b = [bank[i] for i in idx[:ntr]]
    ood_b = [bank[i] for i in idx[ntr:]]

    def sample(bnk, n, seed, adj):
        rng = np.random.RandomState(seed)
        X, Pos, Face, Y = [], [], [], []
        Lops = [[] for _ in range(MAX_STEPS)]
        Ln = [[] for _ in range(MAX_STEPS)]
        Lm = [[] for _ in range(MAX_STEPS)]
        lengths = []
        for _ in range(n):
            a = bnk[rng.randint(0, len(bnk))]
            st = a["start"]
            x0, y0 = int(st.get("x", 2)), int(st.get("y", 7))
            f0 = int(a.get("face0", 0)) % 4
            xf, yf, _ = exec_graph(x0, y0, f0, a["slots"], adj)
            X.append(tok(a["text"]))
            Pos.append([float(x0), float(y0)])
            Face.append([f0 / 3.0])
            Y.append([float(xf), float(yf)])
            L = len(a["slots"])
            lengths.append(L - 1)
            for s in range(MAX_STEPS):
                if s < L:
                    op, nr = a["slots"][s]
                    Lops[s].append(OP2ID[op])
                    Ln[s].append(min(3, nr) - 1)
                    Lm[s].append(1.0)
                else:
                    Lops[s].append(0)
                    Ln[s].append(0)
                    Lm[s].append(0.0)
        out = {
            "nl": torch.tensor(X, dtype=torch.long),
            "pos": torch.tensor(Pos, dtype=torch.float32),
            "face": torch.tensor(Face, dtype=torch.float32),
            "y": torch.tensor(Y, dtype=torch.float32),
            "len": torch.tensor(lengths, dtype=torch.long),
        }
        for s in range(MAX_STEPS):
            out[f"op{s}"] = torch.tensor(Lops[s], dtype=torch.long)
            out[f"n{s}"] = torch.tensor(Ln[s], dtype=torch.long)
            out[f"m{s}"] = torch.tensor(Lm[s], dtype=torch.float32)
        return out

    def train_eval_nav(seed, adj_train, adj_ood, label):
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr = sample(tr_b, N_TRAIN, seed, adj_train)
        ood = sample(ood_b, N_OOD, seed + 20, adj_ood)

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc(N_WORD)
                self.sp = nn.Linear(3, H)
                self.mlp = nn.Sequential(nn.Linear(H * 2, H), nn.Tanh(), nn.Linear(H, 2))
            def forward(self, nl, pos, face):
                return self.mlp(torch.cat([self.enc(nl), self.sp(torch.cat([pos, face], -1))], -1))

        class Prog(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc(N_WORD)
                self.len_h = nn.Linear(H, MAX_STEPS)
                self.op_h = nn.ModuleList([nn.Linear(H, 3) for _ in range(MAX_STEPS)])
                self.n_h = nn.ModuleList([nn.Linear(H, 3) for _ in range(MAX_STEPS)])
            def logits(self, nl):
                h = self.enc(nl)
                return self.len_h(h), [(self.op_h[s](h), self.n_h[s](h)) for s in range(MAX_STEPS)]

        flat = Flat()
        opt = torch.optim.Adam(flat.parameters(), lr=LR)
        best, st = 1e9, None
        for ep in range(EPOCHS):
            flat.train()
            perm = torch.randperm(N_TRAIN)
            for i in range(0, N_TRAIN, BS):
                idx = perm[i : i + BS]
                opt.zero_grad()
                F.mse_loss(flat(tr["nl"][idx], tr["pos"][idx], tr["face"][idx]), tr["y"][idx]).backward()
                opt.step()
            flat.eval()
            with torch.no_grad():
                val = F.mse_loss(flat(tr["nl"][:150], tr["pos"][:150], tr["face"][:150]), tr["y"][:150]).item()
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
                len_lg, steps = prog.logits(tr["nl"][idx])
                loss = F.cross_entropy(len_lg, tr["len"][idx])
                for s in range(MAX_STEPS):
                    m = tr[f"m{s}"][idx]
                    if float(m.sum()) < 1:
                        continue
                    loss = loss + (F.cross_entropy(steps[s][0], tr[f"op{s}"][idx], reduction="none") * m).sum() / m.sum()
                    loss = loss + (F.cross_entropy(steps[s][1], tr[f"n{s}"][idx], reduction="none") * m).sum() / m.sum()
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
            fe = cell_acc(flat(ood["nl"], ood["pos"], ood["face"]), ood["y"])
            len_lg, steps = prog.logits(ood["nl"])
            outs = []
            for i in range(ood["pos"].size(0)):
                L = int(len_lg[i].argmax()) + 1
                slots = [(ID2OP[int(steps[s][0][i].argmax())], int(steps[s][1][i].argmax()) + 1) for s in range(L)]
                x0, y0 = int(round(float(ood["pos"][i, 0]))), int(round(float(ood["pos"][i, 1])))
                f0 = int(round(float(ood["face"][i, 0]) * 3)) % 4
                xf, yf, _ = exec_graph(x0, y0, f0, slots, adj_ood)
                outs.append([float(xf), float(yf)])
            pe = cell_acc(torch.tensor(outs), ood["y"])
        return dict(Flat=fe, Prog=pe, win=pe > fe + 0.02)

    results = {}
    # A2
    print("===== A2 facing =====")
    res = [train_eval_nav(s, adj_g, adj_g, "A2") for s in SEEDS]
    for s, r in zip(SEEDS, res):
        print(f"  {s}: F={r['Flat']:.3f} P={r['Prog']:.3f}")
    fe = float(np.mean([r["Flat"] for r in res]))
    pe = float(np.mean([r["Prog"] for r in res]))
    wins = sum(r["win"] for r in res)
    results["A2"] = dict(Flat=fe, Prog=pe, wins=wins)
    print("A2", results["A2"])

    # A1 edge dropout
    print("===== A1 edge-dropout =====")
    res = []
    for s in SEEDS:
        rng = np.random.RandomState(s + 99)
        edges = []
        for a, nbs in adj_g.items():
            for b in nbs:
                if a < b:
                    edges.append((a, b))
        keep = set()
        for a, b in edges:
            if rng.rand() > 0.3:
                keep.add(frozenset([a, b]))
        adj_ood = defaultdict(set)
        for e in keep:
            a, b = tuple(e)
            adj_ood[a].add(b)
            adj_ood[b].add(a)
        r = train_eval_nav(s, adj_g, adj_ood, "A1")
        res.append(r)
        print(f"  {s}: F={r['Flat']:.3f} P={r['Prog']:.3f}")
    fe = float(np.mean([r["Flat"] for r in res]))
    pe = float(np.mean([r["Prog"] for r in res]))
    wins = sum(r["win"] for r in res)
    results["A1"] = dict(Flat=fe, Prog=pe, wins=wins)
    print("A1", results["A1"])

    # A4 landmark
    print("===== A4 landmark grounded =====")
    LM_LIST = sorted(set(items_g.values()))
    LM2ID = {n: i for i, n in enumerate(LM_LIST)}
    N_LM = len(LM_LIST)
    global_words = set(WORDS)
    for name in LM_LIST:
        for phrase in [
            f"go to the {name}",
            f"walk to the {name}",
            f"find the {name}",
            f"please proceed to {name}",
            f"make your way to the {name}",
        ]:
            for w in phrase.split():
                global_words.add(w)
    WORDS2 = ["<pad>", "<unk>"] + sorted(w for w in global_words if w not in ("<pad>", "<unk>"))
    W2ID2 = {w: i for i, w in enumerate(WORDS2)}
    N_WORD2 = len(WORDS2)

    def tok2(text):
        ids = [W2ID2.get(w, 1) for w in re.findall(r"[a-z0-9]+", text.lower())]
        return (ids + [0] * MAX_NL)[:MAX_NL]

    def nearest(x, y, name):
        cands = [p for p, it in items_g.items() if it == name]
        return min(cands, key=lambda p: abs(p[0] - x) + abs(p[1] - y)) if cands else (x, y)

    def path_toward(x, y, tx, ty, adj, ms=5):
        for _ in range(ms):
            if (x, y) == (tx, ty):
                break
            nbs = list(adj.get((x, y), []))
            if not nbs:
                break
            nbs.sort(key=lambda p: abs(p[0] - tx) + abs(p[1] - ty))
            x, y = nbs[0]
        return x, y

    lm_train, lm_ood = [], []
    for name in LM_LIST:
        lm_train += [(f"go to the {name}", name), (f"walk to the {name}", name), (f"find the {name}", name)]
        lm_ood += [(f"please proceed to {name}", name), (f"make your way to the {name}", name)]

    def sample_lm(bnk, n, seed):
        rng = np.random.RandomState(seed)
        node_list = list(nodes)
        X, Pos, Y, labs = [], [], [], []
        for _ in range(n):
            text, name = bnk[rng.randint(0, len(bnk))]
            x0, y0 = node_list[rng.randint(0, len(node_list))]
            tx, ty = nearest(x0, y0, name)
            xf, yf = path_toward(x0, y0, tx, ty, adj_g)
            X.append(tok2(text))
            Pos.append([float(x0), float(y0)])
            Y.append([float(xf), float(yf)])
            labs.append(LM2ID[name])
        return {
            "nl": torch.tensor(X, dtype=torch.long),
            "pos": torch.tensor(Pos, dtype=torch.float32),
            "y": torch.tensor(Y, dtype=torch.float32),
            "lm": torch.tensor(labs, dtype=torch.long),
        }

    res = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        tr = sample_lm(lm_train, 1500, s)
        ood = sample_lm(lm_ood, 250, s + 20)

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc(N_WORD2)
                self.sp = nn.Linear(2, H)
                self.mlp = nn.Sequential(nn.Linear(H * 2, H), nn.Tanh(), nn.Linear(H, 2))
            def forward(self, nl, pos):
                return self.mlp(torch.cat([self.enc(nl), self.sp(pos)], -1))

        class Prog(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc(N_WORD2)
                self.lm = nn.Linear(H, N_LM)
            def logits(self, nl):
                return self.lm(self.enc(nl))

        flat = Flat()
        opt = torch.optim.Adam(flat.parameters(), lr=LR)
        best, st = 1e9, None
        for ep in range(EPOCHS):
            flat.train()
            perm = torch.randperm(1500)
            for i in range(0, 1500, BS):
                idx = perm[i : i + BS]
                opt.zero_grad()
                F.mse_loss(flat(tr["nl"][idx], tr["pos"][idx]), tr["y"][idx]).backward()
                opt.step()
            flat.eval()
            with torch.no_grad():
                val = F.mse_loss(flat(tr["nl"][:150], tr["pos"][:150]), tr["y"][:150]).item()
            if val < best:
                best, st = val, {k: v.clone() for k, v in flat.state_dict().items()}
        if st:
            flat.load_state_dict(st)
        prog = Prog()
        opt = torch.optim.Adam(prog.parameters(), lr=LR)
        best, st = 1e9, None
        for ep in range(EPOCHS):
            prog.train()
            perm = torch.randperm(1500)
            tot = nb = 0
            for i in range(0, 1500, BS):
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
            pred = prog.logits(ood["nl"]).argmax(-1)
            outs = []
            for i in range(ood["pos"].size(0)):
                name = LM_LIST[int(pred[i])]
                x0, y0 = int(round(float(ood["pos"][i, 0]))), int(round(float(ood["pos"][i, 1])))
                tx, ty = nearest(x0, y0, name)
                xf, yf = path_toward(x0, y0, tx, ty, adj_g)
                outs.append([float(xf), float(yf)])
            pe = cell_acc(torch.tensor(outs), ood["y"])
        res.append(dict(Flat=fe, Prog=pe, win=pe > fe + 0.02))
        print(f"  {s}: F={fe:.3f} P={pe:.3f}")
    fe = float(np.mean([r["Flat"] for r in res]))
    pe = float(np.mean([r["Prog"] for r in res]))
    wins = sum(r["win"] for r in res)
    results["A4"] = dict(Flat=fe, Prog=pe, wins=wins)
    print("A4", results["A4"])
    print("===== 3-BEN =====")
    for k, v in results.items():
        print(k, v)

if __name__ == "__main__":
    main()
