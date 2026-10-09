#!/usr/bin/env python3
"""
C — Mini-SCAN + sequential encoder (fix comparison)

All models share LSTM command encoder.
  - LSTM_FFN: LSTM + residual FFN stack + action head
  - LSTM_PureC: LSTM + PureC blocks (n_mod=6, λ=0) + action head
  - Transformer-lite: small Transformer encoder + head

Preregister:
  PASS if PureC OOD exact > best_baseline + 0.05 AND wins >= 2/3
  FAIL if PureC OOD exact + 0.05 < min baselines
  else UNRESOLVED

Also report ID exact to verify models learn train split.
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
MODS = ["twice", "thrice", "opposite", "around"]
ACTS = ["WALK", "JUMP", "RUN", "LOOK", "LTURN", "RTURN"]
PRIM2ACT = {"walk": "WALK", "jump": "JUMP", "run": "RUN", "look": "LOOK"}
DIR2ACT = {"left": "LTURN", "right": "RTURN"}

CMD_TOKENS = ["<pad>"] + PRIMS + DIRS + MODS
ACT_TOKENS = ["<pad>"] + ACTS
CMD2ID = {t: i for i, t in enumerate(CMD_TOKENS)}
ACT2ID = {t: i for i, t in enumerate(ACT_TOKENS)}
MAX_CMD = 4
MAX_ACT = 12
N_CMD = len(CMD_TOKENS)
N_ACT = len(ACT_TOKENS)

HIDDEN = 64
N_MOD = 6
SEQ_LEN = 4
KV_DIM = 32
BATCH_SIZE = 64
N_TRAIN = 5000
N_VAL = 600
N_OOD = 800
N_ID_EVAL = 600
EPOCHS = 35
LR = 1e-3
TEMP_START, TEMP_END = 1.2, 0.35

def expand_command(prim, direction=None, mod=None):
    if direction:
        turn = DIR2ACT[direction]
        if mod == "opposite":
            turn = "RTURN" if direction == "left" else "LTURN"
            return [turn, PRIM2ACT[prim]]
        if mod == "around":
            return [turn, PRIM2ACT[prim]] * 4
        if mod == "twice":
            return [turn, PRIM2ACT[prim]] * 2
        if mod == "thrice":
            return [turn, PRIM2ACT[prim]] * 3
        return [turn, PRIM2ACT[prim]]
    if mod == "twice":
        return [PRIM2ACT[prim]] * 2
    if mod == "thrice":
        return [PRIM2ACT[prim]] * 3
    if mod == "around":
        return [PRIM2ACT[prim]] * 4
    return [PRIM2ACT[prim]]

def encode_cmd(tokens):
    ids = [CMD2ID[t] for t in tokens] + [0] * MAX_CMD
    return ids[:MAX_CMD]

def encode_act(actions):
    ids = [ACT2ID[a] for a in actions] + [0] * MAX_ACT
    return ids[:MAX_ACT]

def make_train_examples():
    ex = []
    for p in PRIMS:
        ex.append(([p], expand_command(p)))
        for d in DIRS:
            ex.append(([p, d], expand_command(p, d)))
        for m in ["twice", "thrice"]:
            ex.append(([p, m], expand_command(p, None, m)))
        for d in DIRS:
            for m in ["twice", "thrice"]:
                ex.append(([p, d, m], expand_command(p, d, m)))
    return ex

def make_ood_examples():
    ex = []
    for p in PRIMS:
        for d in DIRS:
            ex.append(([p, d, "opposite"], expand_command(p, d, "opposite")))
            ex.append(([p, d, "around"], expand_command(p, d, "around")))
        ex.append(([p, "around"], expand_command(p, None, "around")))
    return ex

TRAIN_EX = make_train_examples()
OOD_EX = make_ood_examples()
print(f"Train templates={len(TRAIN_EX)} OOD templates={len(OOD_EX)}")

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n):
        cmd, act = examples[rng.randint(0, len(examples))]
        X.append(encode_cmd(cmd))
        Y.append(encode_act(act))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# ---- Shared LSTM encoder ----
class LSTMEncoder(nn.Module):
    def __init__(self, emb_dim=HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(N_CMD, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(emb_dim * 2, HIDDEN)
    def forward(self, x):
        e = self.emb(x)
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        # h: 2, B, emb
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.proj(h)

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, KV_DIM)
        self.w_k = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(KV_DIM, 1)
        self.out = nn.Linear(dim, dim)
    def forward(self, query, keys, values, mask=None):
        q = self.w_q(query).unsqueeze(1)
        k = self.w_k(keys)
        score = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return self.out(ctx), attn

class CausalKVLayer(nn.Module):
    def __init__(self, n_mod=N_MOD, topo="ring"):
        super().__init__()
        self.n_mod = n_mod
        self.mods = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_mod)
        ])
        self.attn = AdditiveAttention(HIDDEN)
        self.controller = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, n_mod))
        self.kv_init = nn.Parameter(torch.randn(n_mod, HIDDEN) * 0.02)
        adj = torch.ones(n_mod, n_mod) if topo == "complete" else torch.zeros(n_mod, n_mod)
        if topo == "ring":
            for i in range(n_mod):
                adj[i, i] = adj[i, (i+1)%n_mod] = adj[i, (i-1)%n_mod] = 1
        elif topo == "sparse":
            rng = np.random.RandomState(42)
            for i in range(n_mod):
                adj[i, i] = 1
                cands = [j for j in range(n_mod) if j != i]
                adj[i, rng.choice(cands, min(2, len(cands)), replace=False)] = 1
        self.register_buffer("topo_mask", (adj.sum(0) > 0).float())

    def forward(self, h, temperature=1.0):
        B = h.size(0)
        kv = self.kv_init.unsqueeze(0).expand(B, -1, -1).clone()
        written = torch.zeros(B, self.n_mod, device=h.device)
        for t in range(SEQ_LEN):
            attn_mask = written.clone() if t > 0 else torch.ones_like(written) * 0.15
            ctx, _ = self.attn(h, kv, kv, mask=attn_mask)
            logits = self.controller(ctx)
            logits = logits.masked_fill(self.topo_mask.unsqueeze(0) == 0, -1e9)
            soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            updates = [self.mods[mi](h) * gate[:, mi:mi+1] for mi in range(self.n_mod)]
            h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            write = gate.unsqueeze(-1) * h.unsqueeze(1)
            kv = kv * (1 - gate.unsqueeze(-1)) + write
            written = torch.clamp(written + gate.detach(), 0, 1)
        return h

class LSTM_FFN(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        self.enc = LSTMEncoder()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(HIDDEN, MAX_ACT * N_ACT)
    def forward(self, x, **kwargs):
        h = self.enc(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.head(h).view(-1, MAX_ACT, N_ACT)

class LSTM_PureC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.layers = nn.ModuleList([
            CausalKVLayer(N_MOD, "sparse"),
            CausalKVLayer(N_MOD, "ring"),
            CausalKVLayer(N_MOD, "complete"),
        ])
        self.head = nn.Linear(HIDDEN, MAX_ACT * N_ACT)
    def forward(self, x, temperature=1.0, **kwargs):
        h = self.enc(x)
        for layer in self.layers:
            h = layer(h, temperature=temperature)
        return self.head(h).view(-1, MAX_ACT, N_ACT)

class TransformerLite(nn.Module):
    def __init__(self, n_layers=3, nhead=4):
        super().__init__()
        self.emb = nn.Embedding(N_CMD, HIDDEN, padding_idx=0)
        self.pos = nn.Embedding(MAX_CMD, HIDDEN)
        layer = nn.TransformerEncoderLayer(d_model=HIDDEN, nhead=nhead, dim_feedforward=HIDDEN*2,
                                            batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(HIDDEN, MAX_ACT * N_ACT)
    def forward(self, x, **kwargs):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        pad_mask = (x == 0)
        h = self.tr(h, src_key_padding_mask=pad_mask)
        # mean over non-pad
        mask = (~pad_mask).float().unsqueeze(-1)
        h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(h).view(-1, MAX_ACT, N_ACT)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def exact_and_tok(logits, y):
    pred = logits.argmax(-1)
    B = y.size(0)
    ok = 0
    tok_c = tok_t = 0
    for i in range(B):
        yi = y[i].tolist()
        pi = pred[i].tolist()
        while yi and yi[-1] == 0:
            yi.pop()
        while pi and pi[-1] == 0:
            pi.pop()
        if yi == pi:
            ok += 1
        for a, b in zip(yi, pi):
            tok_t += 1
            if a == b:
                tok_c += 1
        tok_t += abs(len(yi) - len(pi))
    return ok / B, tok_c / max(tok_t, 1)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    Xva, Yva = sample_dataset(TRAIN_EX, N_VAL, seed + 10)
    Xid, Yid = sample_dataset(TRAIN_EX, N_ID_EVAL, seed + 15)
    Xood, Yood = sample_dataset(OOD_EX, N_OOD, seed + 20)

    def train(model, use_temp=False):
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        Xtr_d, Ytr_d = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xva_d, Yva_d = Xva.to(DEVICE), Yva.to(DEVICE)
        best_val, best_state = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS - 1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                logits = model(Xtr_d[idx], temperature=temp) if use_temp else model(Xtr_d[idx])
                loss = F.cross_entropy(logits.reshape(-1, N_ACT), Ytr_d[idx].reshape(-1), ignore_index=0)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(Xva_d, temperature=TEMP_END) if use_temp else model(Xva_d)
                val = F.cross_entropy(logits.reshape(-1, N_ACT), Yva_d.reshape(-1), ignore_index=0).item()
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return model

    def evaluate(model, X, Y, use_temp=False):
        model.eval()
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        with torch.no_grad():
            logits = model(X, temperature=TEMP_END) if use_temp else model(X)
            return exact_and_tok(logits, Y)

    out = {"seed": seed}
    ffn = train(LSTM_FFN())
    out["FFN_id_exact"], out["FFN_id_tok"] = evaluate(ffn, Xid, Yid)
    out["FFN_ood_exact"], out["FFN_ood_tok"] = evaluate(ffn, Xood, Yood)

    tr = train(TransformerLite())
    out["TR_id_exact"], out["TR_id_tok"] = evaluate(tr, Xid, Yid)
    out["TR_ood_exact"], out["TR_ood_tok"] = evaluate(tr, Xood, Yood)

    pc = train(LSTM_PureC(), use_temp=True)
    out["PC_id_exact"], out["PC_id_tok"] = evaluate(pc, Xid, Yid, use_temp=True)
    out["PC_ood_exact"], out["PC_ood_tok"] = evaluate(pc, Xood, Yood, use_temp=True)

    best_base = max(out["FFN_ood_exact"], out["TR_ood_exact"])
    out["win"] = out["PC_ood_exact"] > best_base + 0.02
    return out


print(f"Params FFN={count_params(LSTM_FFN())} TR={count_params(TransformerLite())} PureC={count_params(LSTM_PureC())}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  FFN ID={r['FFN_id_exact']:.3f} OOD={r['FFN_ood_exact']:.3f}", flush=True)
    print(f"  TR  ID={r['TR_id_exact']:.3f} OOD={r['TR_ood_exact']:.3f}", flush=True)
    print(f"  PC  ID={r['PC_id_exact']:.3f} OOD={r['PC_ood_exact']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== C SEQ SUMMARY =====")
for n in ["FFN", "TR", "PC"]:
    print(f"  {n}: ID exact={ms(f'{n}_id_exact')[0]:.3f}  OOD exact={ms(f'{n}_ood_exact')[0]:.3f}  "
          f"OOD tok={ms(f'{n}_ood_tok')[0]:.3f}")

pc = ms("PC_ood_exact")[0]
ffn = ms("FFN_ood_exact")[0]
tr = ms("TR_ood_exact")[0]
best_base = max(ffn, tr)
wins = sum(r["win"] for r in all_results)

if pc > best_base + 0.05 and wins >= 2:
    outcome = "PASS"
elif pc + 0.05 < min(ffn, tr):
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"PC={pc:.3f} best_base={best_base:.3f} wins={wins}/3")
print(f"=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "C Mini-SCAN sequential encoder",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in
            ["FFN_id_exact","FFN_ood_exact","TR_id_exact","TR_ood_exact",
             "PC_id_exact","PC_ood_exact","PC_ood_tok"]},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **C Mini-SCAN + LSTM/Transformer sequential encoder**
- FFN OOD exact={ffn:.3f} | TR={tr:.3f} | PC={pc:.3f} wins={wins}/3
- ID learning: FFN={ms('FFN_id_exact')[0]:.3f} TR={ms('TR_id_exact')[0]:.3f} PC={ms('PC_id_exact')[0]:.3f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
