#!/usr/bin/env python3
"""
C soft OOD — thrice held-out productivity split (seq2seq)

Train: prim, prim dir, prim twice, prim dir twice  (NO thrice)
OOD:  prim thrice, prim dir thrice

Setup gate: ID exact >= 0.50
PASS: PC OOD exact > FFN + 0.05 and wins >= 2/3
FAIL: PC OOD + 0.05 < FFN
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
N_CMD = len(CMD_TOKENS)
N_ACT = len(ACT_TOKENS)
PAD, BOS, EOS = 0, 1, 2

HIDDEN = 96
N_MOD = 6
SEQ_LEN = 3
KV_DIM = 32
BATCH_SIZE = 64
N_TRAIN = 3500
N_VAL = 400
N_ID = 400
N_OOD = 600
EPOCHS = 24
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
    """thrice held out — productivity of count modifier."""
    ex = []
    for p in PRIMS:
        ex.append(([p, "thrice"], expand_command(p, None, "thrice")))
        for d in DIRS:
            ex.append(([p, d, "thrice"], expand_command(p, d, "thrice")))
    return ex

TRAIN_EX = make_train_examples()
OOD_EX = make_ood_examples()
print(f"Train templates={len(TRAIN_EX)} OOD(thrice)={len(OOD_EX)}")

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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_ACT, HIDDEN, padding_idx=PAD)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, N_ACT)
    def forward(self, y_in, h0, c0):
        e = self.emb(y_in)
        out, (h, c) = self.lstm(e, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return self.out(out), h.squeeze(0), c.squeeze(0)
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
        logits, _, _ = self.dec(y[:, :-1], h, c)
        return logits

class S2S_PureC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LSTMEncoder()
        self.layers = nn.ModuleList([
            CausalKVLayer(N_MOD, "sparse"),
            CausalKVLayer(N_MOD, "ring"),
            CausalKVLayer(N_MOD, "complete"),
        ])
        self.dec = Decoder()
    def thought(self, x, temperature=1.0):
        h, c = self.enc(x)
        for layer in self.layers:
            h = layer(h, temperature=temperature)
        return h, c
    def forward(self, x, y, temperature=1.0, **kwargs):
        h, c = self.thought(x, temperature=temperature)
        logits, _, _ = self.dec(y[:, :-1], h, c)
        return logits

def greedy_decode(model, x, use_temp=False, max_len=MAX_ACT - 1):
    model.eval()
    B = x.size(0)
    with torch.no_grad():
        h, c = model.thought(x, temperature=TEMP_END) if use_temp else model.thought(x)
        tok = torch.full((B,), BOS, dtype=torch.long, device=x.device)
        outs = []
        for _ in range(max_len):
            logits, h, c = model.dec.step(tok, h, c)
            tok = logits.argmax(-1)
            outs.append(tok)
        return torch.stack(outs, 1)

def seq_exact_tok(pred, y):
    B = y.size(0)
    ok = tok_c = tok_t = 0
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
        for a, b in zip(pr, gold):
            tok_t += 1
            if a == b:
                tok_c += 1
        tok_t += abs(len(pr) - len(gold))
    return ok / B, tok_c / max(tok_t, 1)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    Xva, Yva = sample_dataset(TRAIN_EX, N_VAL, seed + 10)
    Xid, Yid = sample_dataset(TRAIN_EX, N_ID, seed + 15)
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
                logits = model(Xtr_d[idx], Ytr_d[idx], temperature=temp) if use_temp else model(Xtr_d[idx], Ytr_d[idx])
                loss = F.cross_entropy(logits.reshape(-1, N_ACT), Ytr_d[idx][:, 1:].reshape(-1), ignore_index=PAD)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(Xva_d, Yva_d, temperature=TEMP_END) if use_temp else model(Xva_d, Yva_d)
                val = F.cross_entropy(logits.reshape(-1, N_ACT), Yva_d[:, 1:].reshape(-1), ignore_index=PAD).item()
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return model

    def evaluate(model, X, Y, use_temp=False):
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        pred = greedy_decode(model, X, use_temp=use_temp)
        return seq_exact_tok(pred, Y)

    out = {"seed": seed}
    ffn = train(S2S_FFN())
    out["FFN_id"], out["FFN_id_tok"] = evaluate(ffn, Xid, Yid)
    out["FFN_ood"], out["FFN_ood_tok"] = evaluate(ffn, Xood, Yood)
    pc = train(S2S_PureC(), use_temp=True)
    out["PC_id"], out["PC_id_tok"] = evaluate(pc, Xid, Yid, use_temp=True)
    out["PC_ood"], out["PC_ood_tok"] = evaluate(pc, Xood, Yood, use_temp=True)
    out["win"] = out["PC_ood"] > out["FFN_ood"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  FFN ID={r['FFN_id']:.3f} OOD={r['FFN_ood']:.3f} tok={r['FFN_ood_tok']:.3f}", flush=True)
    print(f"  PC  ID={r['PC_id']:.3f} OOD={r['PC_ood']:.3f} tok={r['PC_ood_tok']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== C SOFT OOD SUMMARY =====")
print(f"FFN: ID={ms('FFN_id')[0]:.3f} OOD={ms('FFN_ood')[0]:.3f} tok={ms('FFN_ood_tok')[0]:.3f}")
print(f"PC:  ID={ms('PC_id')[0]:.3f} OOD={ms('PC_ood')[0]:.3f} tok={ms('PC_ood_tok')[0]:.3f}")

ffn_id, pc_id = ms("FFN_id")[0], ms("PC_id")[0]
ffn_ood, pc_ood = ms("FFN_ood")[0], ms("PC_ood")[0]
wins = sum(r["win"] for r in all_results)
setup_ok = max(ffn_id, pc_id) >= 0.50

if not setup_ok:
    outcome, note = "UNRESOLVED", "ID < 0.50"
elif pc_ood > ffn_ood + 0.05 and wins >= 2:
    outcome, note = "PASS", "PureC > FFN on thrice OOD"
elif pc_ood + 0.05 < ffn_ood:
    outcome, note = "FAIL", "PureC < FFN on thrice OOD"
else:
    outcome, note = "UNRESOLVED", "no clear gap on thrice productivity"

print(f"setup_ok={setup_ok} wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "C soft OOD thrice-holdout seq2seq",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["FFN_id","FFN_ood","PC_id","PC_ood","FFN_ood_tok","PC_ood_tok"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **C soft OOD** thrice held-out, seq2seq FFN vs PureC
- FFN ID={ffn_id:.3f} OOD={ffn_ood:.3f} | PC ID={pc_id:.3f} OOD={pc_ood:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
