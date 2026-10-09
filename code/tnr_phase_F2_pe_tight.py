#!/usr/bin/env python3
"""
Phase F2 — Siết ProgramExecutor (một giả thuyết)

Giả thuyết:
  F UNRESOLVED vì multiplicity học yếu + base unit lỏng.
  F2: *symbolic repeat* — m đọc cứng từ token (twice→2, thrice→3, else→1)
       chỉ học neural base unit (chuỗi action gốc trước khi lặp).
  → Upper bound productivity: nếu base học đúng, thrice OOD phải PASS.

Preregister:
  Setup: PE ID exact >= 0.50
  PASS: PE OOD exact > S2S + 0.05 AND PE OOD >= 0.50 AND wins >= 2/3
  FAIL: PE OOD + 0.05 < S2S (and ID setup ok)
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

CMD_TOKENS = ["<pad>", "<bos>", "<eos>"] + PRIMS + DIRS + ["twice", "thrice"]
ACT_TOKENS = ["<pad>", "<bos>", "<eos>"] + ACTS
CMD2ID = {t: i for i, t in enumerate(CMD_TOKENS)}
ACT2ID = {t: i for i, t in enumerate(ACT_TOKENS)}
MAX_CMD = 5
MAX_ACT = 12
L_BASE = 4
N_CMD = len(CMD_TOKENS)
N_ACT = len(ACT_TOKENS)
PAD, BOS, EOS = 0, 1, 2
TWICE_ID = CMD2ID["twice"]
THRICE_ID = CMD2ID["thrice"]

HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 3500
N_VAL = 400
N_ID = 400
N_OOD = 600
EPOCHS = 30
LR = 1e-3

def expand_command(prim, direction=None, mod=None):
    if direction:
        turn = DIR2ACT[direction]
        if mod == "twice":
            return [turn, PRIM2ACT[prim]] * 2
        if mod == "thrice":
            return [turn, PRIM2ACT[prim]] * 3
        return [turn, PRIM2ACT[prim]]
    if mod == "twice":
        return [PRIM2ACT[prim]] * 2
    if mod == "thrice":
        return [PRIM2ACT[prim]] * 3
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
        ex.append(([p, "thrice"], expand_command(p, None, "thrice")))
        for d in DIRS:
            ex.append(([p, d, "thrice"], expand_command(p, d, "thrice")))
    return ex

TRAIN_EX = make_train_examples()
OOD_EX = make_ood_examples()
print(f"Train={len(TRAIN_EX)} OOD={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n):
        cmd, act = examples[rng.randint(0, len(examples))]
        X.append(encode_cmd(cmd))
        Y.append(encode_act(act))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def symbolic_mult_from_x(x):
    """Hard rule: thrice→3, twice→2, else→1. x: B,L"""
    has_thrice = (x == THRICE_ID).any(1)
    has_twice = (x == TWICE_ID).any(1)
    m = torch.ones(x.size(0), dtype=torch.long, device=x.device)
    m = torch.where(has_twice, torch.full_like(m, 2), m)
    m = torch.where(has_thrice, torch.full_like(m, 3), m)
    return m

def gold_base_unit(y_row, m):
    """Extract base action unit from gold sequence given multiplicity m."""
    gold = y_row.tolist()
    if gold and gold[0] == BOS:
        gold = gold[1:]
    if EOS in gold:
        gold = gold[: gold.index(EOS)]
    else:
        while gold and gold[-1] == PAD:
            gold.pop()
    if m < 1:
        m = 1
    unit_len = max(len(gold) // m, 1) if gold else 1
    unit = gold[:unit_len] if gold else [ACT2ID["WALK"]]
    return unit[:L_BASE]

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
        return self.proj(h)

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
        self.enc_emb = nn.Embedding(N_CMD, HIDDEN, padding_idx=PAD)
        self.enc_lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(HIDDEN * 2, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(4)
        ])
        self.dec = Decoder()
    def thought(self, x):
        e = self.enc_emb(x)
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.enc_lstm(packed)
        h = self.proj(torch.cat([h[0], h[1]], -1))
        c = self.proj(torch.cat([c[0], c[1]], -1))
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return h, c
    def forward(self, x, y):
        h, c = self.thought(x)
        return self.dec(y[:, :-1], h, c)

class ProgramExecutorF2(nn.Module):
    """Neural base unit only; multiplicity is symbolic from tokens."""
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.base_head = nn.Linear(HIDDEN, L_BASE * N_ACT)
        self.len_head = nn.Linear(HIDDEN, L_BASE)  # soft base length

    def base_logits(self, x):
        h = self.enc(x)
        return self.base_head(h).view(x.size(0), L_BASE, N_ACT), self.len_head(h)

    def execute(self, x):
        """Greedy base + symbolic m."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.base_logits(x)
            base_ids = logits.argmax(-1)  # B, L_BASE
            m = symbolic_mult_from_x(x)
            B = x.size(0)
            outs = []
            for i in range(B):
                unit = []
                for a in base_ids[i].tolist():
                    if a in (PAD, EOS, BOS):
                        if unit:
                            break
                        continue
                    unit.append(a)
                if not unit:
                    unit = [ACT2ID["WALK"]]
                seq = (unit * int(m[i].item()))[: MAX_ACT - 1]
                outs.append(seq + [PAD] * (MAX_ACT - 1 - len(seq)))
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

    def train_pe():
        model = ProgramExecutorF2().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv, Yv = Xva.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total_loss = 0
            n_batch = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                xb, yb = X[idx], Y[idx]
                opt.zero_grad()
                base_logits, _ = model.base_logits(xb)
                m = symbolic_mult_from_x(xb)
                loss = 0
                for bi in range(xb.size(0)):
                    unit = gold_base_unit(yb[bi], int(m[bi].item()))
                    for t, a in enumerate(unit):
                        loss = loss + F.cross_entropy(
                            base_logits[bi, t:t+1],
                            torch.tensor([a], device=DEVICE),
                        )
                    # encourage PAD after unit
                    for t in range(len(unit), L_BASE):
                        loss = loss + 0.3 * F.cross_entropy(
                            base_logits[bi, t:t+1],
                            torch.tensor([PAD], device=DEVICE),
                        )
                loss = loss / max(xb.size(0), 1)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batch += 1
            # val: base CE
            model.eval()
            with torch.no_grad():
                base_logits, _ = model.base_logits(Xv)
                m = symbolic_mult_from_x(Xv)
                val = 0
                for bi in range(Xv.size(0)):
                    unit = gold_base_unit(Yv[bi], int(m[bi].item()))
                    for t, a in enumerate(unit):
                        val += F.cross_entropy(
                            base_logits[bi, t:t+1],
                            torch.tensor([a], device=DEVICE),
                        ).item()
                val /= max(Xv.size(0), 1)
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

    pe = train_pe()
    out["PE_id"] = seq_exact(pe.execute(Xid.to(DEVICE)), Yid.to(DEVICE))
    out["PE_ood"] = seq_exact(pe.execute(Xood.to(DEVICE)), Yood.to(DEVICE))
    out["win"] = out["PE_ood"] > out["S2S_ood"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  S2S ID={r['S2S_id']:.3f} OOD={r['S2S_ood']:.3f}", flush=True)
    print(f"  PE  ID={r['PE_id']:.3f} OOD={r['PE_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== F2 SUMMARY =====")
print(f"S2S: ID={ms('S2S_id')[0]:.3f} OOD={ms('S2S_ood')[0]:.3f}")
print(f"PE:  ID={ms('PE_id')[0]:.3f} OOD={ms('PE_ood')[0]:.3f}")

s2s_id, pe_id = ms("S2S_id")[0], ms("PE_id")[0]
s2s_ood, pe_ood = ms("S2S_ood")[0], ms("PE_ood")[0]
wins = sum(r["win"] for r in all_results)
setup_ok = pe_id >= 0.50

if not setup_ok:
    outcome, note = "UNRESOLVED", f"PE ID={pe_id:.3f} < 0.50"
elif pe_ood > s2s_ood + 0.05 and pe_ood >= 0.50 and wins >= 2:
    outcome, note = "PASS", "symbolic repeat unlocks thrice"
elif pe_ood + 0.05 < s2s_ood:
    outcome, note = "FAIL", "PE worse"
else:
    outcome, note = "UNRESOLVED", "gap insufficient or OOD < 0.50"

print(f"setup_ok={setup_ok} wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "F2 neural base + symbolic multiplicity",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["S2S_id","S2S_ood","PE_id","PE_ood"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **F2** neural base + symbolic m from twice/thrice tokens
- S2S ID={s2s_id:.3f} OOD={s2s_ood:.3f} | PE ID={pe_id:.3f} OOD={pe_ood:.3f}
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
