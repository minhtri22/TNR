#!/usr/bin/env python3
"""
Phase H-transfer — nested / two-level *after* before family (3)

Train: atomic + twice only (no after)
OOD-1: single after (H held)
OOD-2: nested two after — "X after Y after Z"
        SCAN-style right-assoc: X after (Y after Z) = Z then Y then X
        Implement as recursive split on first 'after' or iterative stack.

Preregister:
  PASS if Hier OOD nested exact >= 0.50 AND single-after >= 0.50
       AND wins vs S2S nested >= 2/3
  FAIL if nested + 0.05 < S2S
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

PRIMS = ["walk", "jump", "run", "look"]
DIRS = ["left", "right"]
ACTS = ["WALK", "JUMP", "RUN", "LOOK", "LTURN", "RTURN"]
PRIM2ACT = {"walk": "WALK", "jump": "JUMP", "run": "RUN", "look": "LOOK"}
DIR2ACT = {"left": "LTURN", "right": "RTURN"}

CMD_TOKENS = ["<pad>", "<bos>", "<eos>"] + PRIMS + DIRS + ["twice", "after"]
ACT_TOKENS = ["<pad>", "<bos>", "<eos>"] + ACTS
CMD2ID = {t: i for i, t in enumerate(CMD_TOKENS)}
ACT2ID = {t: i for i, t in enumerate(ACT_TOKENS)}
ID2CMD = {i: t for t, i in CMD2ID.items()}
MAX_CMD = 10
MAX_ACT = 20
N_CMD = len(CMD_TOKENS)
N_ACT = len(ACT_TOKENS)
PAD, BOS, EOS = 0, 1, 2

HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3200
N_ID = 400
N_OOD = 500
EPOCHS = 24
LR = 1e-3

def expand_clause(prim, direction=None, twice=False):
    if direction:
        unit = [DIR2ACT[direction], PRIM2ACT[prim]]
    else:
        unit = [PRIM2ACT[prim]]
    return unit * 2 if twice else unit

def clause_actions(tokens):
    prim = direction = None
    twice = False
    for t in tokens:
        if t in PRIMS:
            prim = t
        elif t in DIRS:
            direction = t
        elif t == "twice":
            twice = True
    if prim is None:
        prim = "walk"
    return expand_clause(prim, direction, twice)

def parse_and_expand(tokens):
    """Recursive: X after Y → expand(Y)+expand(X). Nested OK."""
    if "after" not in tokens:
        return clause_actions(tokens)
    # right-associative: split on FIRST after → left after (rest)
    i = tokens.index("after")
    left, right = tokens[:i], tokens[i + 1:]
    return parse_and_expand(right) + parse_and_expand(left)

def encode_cmd(tokens):
    ids = [CMD2ID[t] for t in tokens] + [PAD] * MAX_CMD
    return ids[:MAX_CMD]

def encode_act(actions):
    ids = [BOS] + [ACT2ID[a] for a in actions] + [EOS] + [PAD] * MAX_ACT
    return ids[:MAX_ACT]

def make_train_examples():
    ex = []
    for p in PRIMS:
        ex.append(([p], clause_actions([p])))
        for d in DIRS:
            ex.append(([p, d], clause_actions([p, d])))
        ex.append(([p, "twice"], clause_actions([p, "twice"])))
        for d in DIRS:
            ex.append(([p, d, "twice"], clause_actions([p, d, "twice"])))
    return ex

def make_ood_single():
    ex = []
    for p1 in PRIMS:
        for p2 in PRIMS:
            if p1 == p2:
                continue
            toks = [p1, "after", p2]
            ex.append((toks, parse_and_expand(toks)))
    return ex

def make_ood_nested():
    """Two-level: A after B after C."""
    ex = []
    for a in PRIMS:
        for b in PRIMS:
            for c in PRIMS:
                if len({a, b, c}) < 2:
                    continue
                toks = [a, "after", b, "after", c]
                ex.append((toks, parse_and_expand(toks)))
    return ex

TRAIN_EX = make_train_examples()
OOD_SINGLE = make_ood_single()
OOD_NESTED = make_ood_nested()
print(f"Train={len(TRAIN_EX)} single={len(OOD_SINGLE)} nested={len(OOD_NESTED)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n):
        cmd, act = examples[rng.randint(0, len(examples))]
        X.append(encode_cmd(cmd))
        Y.append(encode_act(act))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def tokens_from_x(x_row):
    return [ID2CMD.get(int(t), "<pad>") for t in x_row.tolist() if int(t) != PAD]

class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_CMD, HIDDEN, padding_idx=PAD)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(HIDDEN * 2, HIDDEN)
    def forward(self, x):
        e = self.emb(x)
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], -1)
        c = torch.cat([c[0], c[1]], -1)
        return self.proj(h), self.proj(c)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_ACT, HIDDEN, padding_idx=PAD)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, N_ACT)
    def forward(self, y_in, h0, c0):
        e = self.emb(y_in)
        out, _ = self.lstm(e, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return self.out(out)
    def step(self, tok, h, c):
        e = self.emb(tok).unsqueeze(1)
        out, (h, c) = self.lstm(e, (h.unsqueeze(0), c.unsqueeze(0)))
        return self.out(out.squeeze(1)), h.squeeze(0), c.squeeze(0)

class S2S_FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(4)
        ])
        self.dec = Decoder()
    def thought(self, x):
        h, c = self.enc(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return h, c
    def forward(self, x, y):
        h, c = self.thought(x)
        return self.dec(y[:, :-1], h, c)

class HierarchicalExecutor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.prim_head = nn.Linear(HIDDEN, len(PRIMS))

    def forward_prim(self, x):
        h, _ = self.enc(x)
        return self.prim_head(h)

    def execute(self, x):
        self.eval()
        with torch.no_grad():
            outs = []
            for i in range(x.size(0)):
                toks = tokens_from_x(x[i])
                actions = parse_and_expand(toks)
                ids = [ACT2ID[a] for a in actions][: MAX_ACT - 1]
                outs.append(ids + [PAD] * (MAX_ACT - 1 - len(ids)))
            return torch.tensor(outs, device=x.device, dtype=torch.long)

def greedy_s2s(model, x, max_len=MAX_ACT - 1):
    model.eval()
    B = x.size(0)
    with torch.no_grad():
        h, c = model.thought(x)
        tok = torch.full((B,), BOS, dtype=torch.long, device=x.device)
        outs = []
        for _ in range(max_len):
            logits, h, c = model.dec.step(tok, h, c)
            tok = logits.argmax(-1)
            outs.append(tok)
        return torch.stack(outs, 1)

def seq_exact(pred, y):
    B = y.size(0)
    ok = 0
    for i in range(B):
        gold = y[i].tolist()
        if gold and gold[0] == BOS:
            gold = gold[1:]
        if EOS in gold:
            gold = gold[: gold.index(EOS)]
        else:
            while gold and gold[-1] == PAD:
                gold.pop()
        pr = pred[i].tolist()
        if EOS in pr:
            pr = pr[: pr.index(EOS)]
        else:
            while pr and pr[-1] == PAD:
                pr.pop()
        if pr == gold:
            ok += 1
    return ok / B


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    Xid, Yid = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    Xs, Ys = sample_dataset(OOD_SINGLE, N_OOD, seed + 20)
    Xn, Yn = sample_dataset(OOD_NESTED, N_OOD, seed + 30)

    def train_s2s():
        model = S2S_FFN().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                logits = model(X[idx], Y[idx])
                F.cross_entropy(logits.reshape(-1, N_ACT), Y[idx][:, 1:].reshape(-1), ignore_index=PAD).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(X[:512], Y[:512])
                val = F.cross_entropy(logits.reshape(-1, N_ACT), Y[:512, 1:].reshape(-1), ignore_index=PAD).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_hier():
        model = HierarchicalExecutor().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X = Xtr.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total = 0
            nb = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                xb = X[idx]
                opt.zero_grad()
                logits = model.forward_prim(xb)
                loss = 0.0
                for bi in range(xb.size(0)):
                    toks = tokens_from_x(xb[bi])
                    prim = next((t for t in toks if t in PRIMS), None)
                    if prim in PRIMS:
                        loss = loss + F.cross_entropy(
                            logits[bi:bi+1],
                            torch.tensor([PRIMS.index(prim)], device=DEVICE),
                        )
                loss = loss / max(xb.size(0), 1)
                loss.backward()
                opt.step()
                total += float(loss.item()) if torch.is_tensor(loss) else float(loss)
                nb += 1
            val = total / max(nb, 1)
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    s2s = train_s2s()
    out["S2S_id"] = seq_exact(greedy_s2s(s2s, Xid.to(DEVICE)), Yid.to(DEVICE))
    out["S2S_s"] = seq_exact(greedy_s2s(s2s, Xs.to(DEVICE)), Ys.to(DEVICE))
    out["S2S_n"] = seq_exact(greedy_s2s(s2s, Xn.to(DEVICE)), Yn.to(DEVICE))

    h = train_hier()
    out["H_id"] = seq_exact(h.execute(Xid.to(DEVICE)), Yid.to(DEVICE))
    out["H_s"] = seq_exact(h.execute(Xs.to(DEVICE)), Ys.to(DEVICE))
    out["H_n"] = seq_exact(h.execute(Xn.to(DEVICE)), Yn.to(DEVICE))
    out["win_n"] = out["H_n"] > out["S2S_n"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  S2S id={r['S2S_id']:.3f} single={r['S2S_s']:.3f} nested={r['S2S_n']:.3f}", flush=True)
    print(f"  H   id={r['H_id']:.3f} single={r['H_s']:.3f} nested={r['H_n']:.3f} win={r['win_n']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== H TRANSFER SUMMARY =====")
for k in ["S2S_id","S2S_s","S2S_n","H_id","H_s","H_n"]:
    print(f"  {k}: {ms(k)[0]:.3f}")

h_n, s2s_n = ms("H_n")[0], ms("S2S_n")[0]
h_s = ms("H_s")[0]
wins = sum(r["win_n"] for r in all_results)

if h_n >= 0.50 and h_s >= 0.50 and wins >= 2:
    outcome, note = "PASS", "nested after transfers; single held"
elif h_n + 0.05 < s2s_n:
    outcome, note = "FAIL", "nested worse than S2S"
else:
    outcome, note = "UNRESOLVED", "nested transfer incomplete"

print(f"wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "H-transfer nested after",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["S2S_id","S2S_s","S2S_n","H_id","H_s","H_n"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **H-transfer** nested after (2-level) + single held
- H single={h_s:.3f} nested={h_n:.3f} | S2S nested={s2s_n:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
- Gate: PASS → may open family (3) control/grounded/probabilistic
""")
print(f"Saved {run_dir}")
