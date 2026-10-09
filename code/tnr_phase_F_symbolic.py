#!/usr/bin/env python3
"""
Phase F — Symbolic (một inductive bias đúng lớp)

Giả thuyết:
  SCAN productivity (twice→thrice) cần bias *lặp chương trình*, không phải thought-vector.
  ProgramExecutor:
    - Encoder LSTM command
    - Predict base action sequence (length ≤ L_base) via Gumbel slots
    - Predict multiplicity m ∈ {1,2,3} (soft)
    - Execute: tile base actions × m  → action sequence

So sánh trên thrice-holdout (như C soft OOD):
  S2S_FFN: LSTM enc+dec (đã exact OOD=0)
  ProgramExec: explicit repeat bias

Preregister:
  Setup: ID exact >= 0.50
  PASS: PE OOD exact > S2S + 0.05 AND wins >= 2/3
  FAIL: PE OOD + 0.05 < S2S
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
L_BASE = 4  # max base program length before repeat
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
TEMP_START, TEMP_END = 1.2, 0.4

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
print(f"Train={len(TRAIN_EX)} OOD thrice={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n):
        cmd, act = examples[rng.randint(0, len(examples))]
        X.append(encode_cmd(cmd))
        Y.append(encode_act(act))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

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
        out, (h, c) = self.lstm(e, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return self.out(out)
    def step(self, tok, h, c):
        e = self.emb(tok).unsqueeze(1)
        out, (h, c) = self.lstm(e, (h.unsqueeze(0), c.unsqueeze(0)))
        return self.out(out.squeeze(1)), h.squeeze(0), c.squeeze(0)

class S2S_FFN(nn.Module):
    def __init__(self, n_blocks=4):
        super().__init__()
        self.enc = LSTMEncoder()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.dec = Decoder()
    def thought(self, x, **kwargs):
        h, c = self.enc(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return h, c
    def forward(self, x, y, **kwargs):
        h, c = self.thought(x)
        return self.dec(y[:, :-1], h, c)

class ProgramExecutor(nn.Module):
    """
    Bias symbolic: base program + multiplicity.
    - base_logits: L_BASE slots × N_ACT (which primitive actions form the unit)
    - mult_logits: 3-way (m=1,2,3)
    Execute: take argmax/soft base sequence until EOS-ish, tile × m, pad to MAX_ACT-1
    """
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.base_head = nn.Linear(HIDDEN, L_BASE * N_ACT)
        self.mult_head = nn.Linear(HIDDEN, 3)  # m=1,2,3
        self.stop_head = nn.Linear(HIDDEN, L_BASE)  # soft length of base

    def forward(self, x, y=None, temperature=1.0):
        h, _ = self.enc(x)
        B = x.size(0)
        base_logits = self.base_head(h).view(B, L_BASE, N_ACT)
        mult_logits = self.mult_head(h)
        stop_logits = self.stop_head(h)  # B, L_BASE

        # soft base actions
        if self.training:
            base_soft = F.gumbel_softmax(base_logits, tau=max(temperature, 0.1), hard=True, dim=-1)
            mult_soft = F.gumbel_softmax(mult_logits, tau=max(temperature, 0.1), hard=True, dim=-1)
        else:
            base_soft = F.softmax(base_logits / max(temperature, 0.1), dim=-1)
            mult_soft = F.softmax(mult_logits / max(temperature, 0.1), dim=-1)

        # expected multiplicity 1,2,3
        m_vals = torch.tensor([1.0, 2.0, 3.0], device=x.device)
        m = (mult_soft * m_vals).sum(-1)  # B

        # For training loss we compare executed sequence distribution to y
        # Build soft output sequence: for simplicity use hard execute in forward for CE via teacher path
        return base_logits, mult_logits, stop_logits, m

    def execute_greedy(self, x, temperature=0.4):
        self.eval()
        with torch.no_grad():
            h, _ = self.enc(x)
            B = x.size(0)
            base_logits = self.base_head(h).view(B, L_BASE, N_ACT)
            mult_logits = self.mult_head(h)
            base_ids = base_logits.argmax(-1)  # B, L_BASE
            mult_id = mult_logits.argmax(-1)  # 0,1,2 → m=1,2,3
            m = mult_id + 1
            outs = []
            for i in range(B):
                base = base_ids[i].tolist()
                # trim pads / keep until repeated pad
                unit = []
                for a in base:
                    if a == PAD or a == EOS:
                        break
                    if a == BOS:
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
        model = ProgramExecutor().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv, Yv = Xva.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS - 1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                base_logits, mult_logits, stop_logits, m = model(X[idx], temperature=temp)
                # Supervise multiplicity from y length heuristic + base from first unit
                # Gold mult: estimate from sequence length vs unit
                yb = Y[idx]
                # Derive gold multiplicity roughly from presence of twice/thrice in x
                # x tokens: twice id, thrice id
                twice_id = CMD2ID["twice"]
                thrice_id = CMD2ID["thrice"]
                has_twice = (X[idx] == twice_id).any(1).long()
                has_thrice = (X[idx] == thrice_id).any(1).long()
                gold_mult = has_twice * 1 + has_thrice * 2  # 0→m1, 1→m2, 2→m3 index
                # gold base: take gold actions / mult
                loss_mult = F.cross_entropy(mult_logits, gold_mult)
                # base: teacher — first unit of gold sequence
                loss_base = 0
                for bi in range(idx.size(0)):
                    gold = yb[bi].tolist()
                    if gold[0] == BOS:
                        gold = gold[1:]
                    if EOS in gold:
                        gold = gold[: gold.index(EOS)]
                    m_i = int(gold_mult[bi].item()) + 1
                    unit_len = max(len(gold) // m_i, 1)
                    unit = gold[:unit_len]
                    for t, a in enumerate(unit[:L_BASE]):
                        loss_base = loss_base + F.cross_entropy(
                            base_logits[bi, t:t+1], torch.tensor([a], device=DEVICE)
                        )
                loss_base = loss_base / max(idx.size(0), 1)
                loss = loss_mult + loss_base
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                # val = mult accuracy proxy
                _, mult_logits, _, _ = model(Xv, temperature=TEMP_END)
                has_twice = (Xv == CMD2ID["twice"]).any(1).long()
                has_thrice = (Xv == CMD2ID["thrice"]).any(1).long()
                gold_mult = has_twice * 1 + has_thrice * 2
                val = F.cross_entropy(mult_logits, gold_mult).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    s2s = train_s2s()
    pred = greedy_s2s(s2s, Xid.to(DEVICE))
    out["S2S_id"] = seq_exact(pred, Yid.to(DEVICE))
    pred = greedy_s2s(s2s, Xood.to(DEVICE))
    out["S2S_ood"] = seq_exact(pred, Yood.to(DEVICE))

    pe = train_pe()
    pred = pe.execute_greedy(Xid.to(DEVICE))
    out["PE_id"] = seq_exact(pred, Yid.to(DEVICE))
    pred = pe.execute_greedy(Xood.to(DEVICE))
    out["PE_ood"] = seq_exact(pred, Yood.to(DEVICE))
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

print("\n===== F SYMBOLIC SUMMARY =====")
print(f"S2S: ID={ms('S2S_id')[0]:.3f} OOD={ms('S2S_ood')[0]:.3f}")
print(f"PE:  ID={ms('PE_id')[0]:.3f} OOD={ms('PE_ood')[0]:.3f}")

s2s_id, pe_id = ms("S2S_id")[0], ms("PE_id")[0]
s2s_ood, pe_ood = ms("S2S_ood")[0], ms("PE_ood")[0]
wins = sum(r["win"] for r in all_results)
setup_ok = max(s2s_id, pe_id) >= 0.50

if not setup_ok:
    outcome, note = "UNRESOLVED", "ID < 0.50"
elif pe_ood > s2s_ood + 0.05 and wins >= 2:
    outcome, note = "PASS", "ProgramExecutor wins thrice productivity"
elif pe_ood + 0.05 < s2s_ood:
    outcome, note = "FAIL", "PE worse than S2S"
else:
    outcome, note = "UNRESOLVED", "no clear OOD gap"

print(f"setup_ok={setup_ok} wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase F Symbolic ProgramExecutor vs S2S",
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
- Experiment: **Phase F Symbolic** ProgramExecutor (base+multiplicity) vs S2S
- S2S ID={s2s_id:.3f} OOD={s2s_ood:.3f} | PE ID={pe_id:.3f} OOD={pe_ood:.3f}
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
