#!/usr/bin/env python3
"""
Phase G — Relational / binding (họ tiếp sau 3 claim freeze)

Task: Mini-SCAN *opposite* (systematic relational).
  opposite left  = right turn + prim
  opposite right = left turn + prim
Train: prim, prim+dir, prim+twice, prim+dir+twice (NO opposite)
OOD: prim+dir+opposite

Giả thuyết G1 — Slot binding:
  Neural base unit + *symbolic opposite* trên direction token
  (mirror của F2: rule trên vocabulary, không học seq2seq thuần)
  opposite(left)=right, opposite(right)=left rồi emit turn+prim.

So sánh:
  S2S_FFN (seq2seq) — baseline C/F (OOD systematic thường = 0)
  Binder: neural prim/dir heads + symbolic opposite rewrite

Preregister:
  Setup: Binder ID >= 0.50
  PASS: Binder OOD exact > S2S + 0.05 AND Binder OOD >= 0.50 AND wins >= 2/3
  FAIL: Binder OOD + 0.05 < S2S (setup ok)
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
OPP_DIR = {"left": "right", "right": "left"}

CMD_TOKENS = ["<pad>", "<bos>", "<eos>"] + PRIMS + DIRS + ["twice", "opposite"]
ACT_TOKENS = ["<pad>", "<bos>", "<eos>"] + ACTS
CMD2ID = {t: i for i, t in enumerate(CMD_TOKENS)}
ACT2ID = {t: i for i, t in enumerate(ACT_TOKENS)}
ID2CMD = {i: t for t, i in CMD2ID.items()}
MAX_CMD = 5
MAX_ACT = 10
N_CMD = len(CMD_TOKENS)
N_ACT = len(ACT_TOKENS)
PAD, BOS, EOS = 0, 1, 2

HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3500
N_VAL = 400
N_ID = 400
N_OOD = 600
EPOCHS = 28
LR = 1e-3

def expand_command(prim, direction=None, mod=None):
    if direction:
        d = OPP_DIR[direction] if mod == "opposite" else direction
        turn = DIR2ACT[d]
        if mod == "twice":
            return [turn, PRIM2ACT[prim]] * 2
        return [turn, PRIM2ACT[prim]]
    if mod == "twice":
        return [PRIM2ACT[prim]] * 2
    return [PRIM2ACT[prim]]

def encode_cmd(tokens):
    ids = [CMD2ID[t] for t in tokens] + [PAD] * MAX_CMD
    return ids[:MAX_CMD]

def encode_act(actions):
    ids = [BOS] + [ACT2ID[a] for a in actions] + [EOS] + [PAD] * MAX_ACT
    return ids[:MAX_ACT]

def make_train_examples():
    ex = []
    for p in PRIMS:
        ex.append(([p], expand_command(p)))
        for d in DIRS:
            ex.append(([p, d], expand_command(p, d)))
        ex.append(([p, "twice"], expand_command(p, None, "twice")))
        for d in DIRS:
            ex.append(([p, d, "twice"], expand_command(p, d, "twice")))
    return ex

def make_ood_examples():
    ex = []
    for p in PRIMS:
        for d in DIRS:
            ex.append(([p, d, "opposite"], expand_command(p, d, "opposite")))
    return ex

TRAIN_EX = make_train_examples()
OOD_EX = make_ood_examples()
print(f"Train={len(TRAIN_EX)} OOD opposite={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n):
        cmd, act = examples[rng.randint(0, len(examples))]
        X.append(encode_cmd(cmd))
        Y.append(encode_act(act))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def parse_cmd_tokens(x_row):
    """Extract prim, dir, flags from token ids."""
    toks = [ID2CMD.get(int(t), "<pad>") for t in x_row.tolist() if int(t) != PAD]
    prim = None
    direction = None
    twice = False
    opposite = False
    for t in toks:
        if t in PRIMS:
            prim = t
        elif t in DIRS:
            direction = t
        elif t == "twice":
            twice = True
        elif t == "opposite":
            opposite = True
    return prim, direction, twice, opposite

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

class RelationalBinder(nn.Module):
    """
    Slot-style: predict prim & dir as classifications; symbolic opposite + twice.
    Binding: opposite rewrites dir slot before execution.
    """
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.prim_head = nn.Linear(HIDDEN, len(PRIMS))
        self.dir_head = nn.Linear(HIDDEN, len(DIRS) + 1)  # + none
        # no neural opposite — symbolic

    def forward_slots(self, x):
        h, _ = self.enc(x)
        return self.prim_head(h), self.dir_head(h)

    def execute(self, x):
        self.eval()
        with torch.no_grad():
            prim_logits, dir_logits = self.forward_slots(x)
            prim_ids = prim_logits.argmax(-1)
            dir_ids = dir_logits.argmax(-1)
            B = x.size(0)
            outs = []
            for i in range(B):
                prim, direction, twice, opposite = parse_cmd_tokens(x[i])
                # neural slots (fallback if parse misses)
                p_name = PRIMS[int(prim_ids[i].item())] if prim is None else prim
                if direction is None and int(dir_ids[i].item()) < len(DIRS):
                    direction = DIRS[int(dir_ids[i].item())]
                # symbolic binding
                if opposite and direction in OPP_DIR:
                    direction = OPP_DIR[direction]
                actions = expand_command(p_name, direction, "twice" if twice else None)
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
    Xva, Yva = sample_dataset(TRAIN_EX, N_VAL, seed + 10)
    Xid, Yid = sample_dataset(TRAIN_EX, N_ID, seed + 15)
    Xood, Yood = sample_dataset(OOD_EX, N_OOD, seed + 20)

    def train_s2s():
        model = S2S_FFN().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv, Yv = Xva.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                logits = model(X[idx], Y[idx])
                loss = F.cross_entropy(logits.reshape(-1, N_ACT), Y[idx][:, 1:].reshape(-1), ignore_index=PAD)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(Xv, Yv)
                val = F.cross_entropy(logits.reshape(-1, N_ACT), Yv[:, 1:].reshape(-1), ignore_index=PAD).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_binder():
        model = RelationalBinder().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total = 0
            n = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                xb = X[idx]
                opt.zero_grad()
                prim_logits, dir_logits = model.forward_slots(xb)
                loss = 0
                for bi in range(xb.size(0)):
                    prim, direction, twice, opposite = parse_cmd_tokens(xb[bi])
                    if prim in PRIMS:
                        loss = loss + F.cross_entropy(
                            prim_logits[bi:bi+1],
                            torch.tensor([PRIMS.index(prim)], device=DEVICE),
                        )
                    if direction in DIRS:
                        loss = loss + F.cross_entropy(
                            dir_logits[bi:bi+1],
                            torch.tensor([DIRS.index(direction)], device=DEVICE),
                        )
                    else:
                        loss = loss + F.cross_entropy(
                            dir_logits[bi:bi+1],
                            torch.tensor([len(DIRS)], device=DEVICE),  # none
                        )
                loss = loss / max(xb.size(0), 1)
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            val = total / max(n, 1)
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    s2s = train_s2s()
    out["S2S_id"] = seq_exact(greedy_s2s(s2s, Xid.to(DEVICE)), Yid.to(DEVICE))
    out["S2S_ood"] = seq_exact(greedy_s2s(s2s, Xood.to(DEVICE)), Yood.to(DEVICE))

    bd = train_binder()
    out["BD_id"] = seq_exact(bd.execute(Xid.to(DEVICE)), Yid.to(DEVICE))
    out["BD_ood"] = seq_exact(bd.execute(Xood.to(DEVICE)), Yood.to(DEVICE))
    out["win"] = out["BD_ood"] > out["S2S_ood"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  S2S ID={r['S2S_id']:.3f} OOD={r['S2S_ood']:.3f}", flush=True)
    print(f"  BD  ID={r['BD_id']:.3f} OOD={r['BD_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== G BINDING SUMMARY =====")
print(f"S2S: ID={ms('S2S_id')[0]:.3f} OOD={ms('S2S_ood')[0]:.3f}")
print(f"BD:  ID={ms('BD_id')[0]:.3f} OOD={ms('BD_ood')[0]:.3f}")

s2s_id, bd_id = ms("S2S_id")[0], ms("BD_id")[0]
s2s_ood, bd_ood = ms("S2S_ood")[0], ms("BD_ood")[0]
wins = sum(r["win"] for r in all_results)
setup_ok = bd_id >= 0.50

if not setup_ok:
    outcome, note = "UNRESOLVED", f"BD ID={bd_id:.3f}<0.50"
elif bd_ood > s2s_ood + 0.05 and bd_ood >= 0.50 and wins >= 2:
    outcome, note = "PASS", "symbolic opposite binding unlocks OOD"
elif bd_ood + 0.05 < s2s_ood:
    outcome, note = "FAIL", "Binder worse than S2S"
else:
    outcome, note = "UNRESOLVED", "insufficient gap"

print(f"setup_ok={setup_ok} wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase G Relational binding opposite",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["S2S_id","S2S_ood","BD_id","BD_ood"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase G Binding** symbolic opposite + neural slots vs S2S
- S2S ID={s2s_id:.3f} OOD={s2s_ood:.3f} | BD ID={bd_id:.3f} OOD={bd_ood:.3f}
- Outcome: **{outcome}** ({note})
- Note: opens family (1) after lab freeze of 3 claims — separate phase
""")
print(f"Saved {run_dir}")
