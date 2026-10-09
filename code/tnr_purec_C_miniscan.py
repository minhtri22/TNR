#!/usr/bin/env python3
"""
C — Benchmark chuẩn: Mini-SCAN (compositional generalization)

Inspired by SCAN (Lake & Baroni): command composition → action sequence.
Split: train short/simple compositions; OOD longer / novel templates.

Architecture locked from A:
  PureC n_mod=6, λ=0, KV_causal
Baselines: HierResidual, Bag+MLP (no routing)

Metric: action-sequence accuracy (exact match) + token accuracy
Preregister:
  PASS nếu PureC OOD exact-acc > max(HR, Bag) + 5pp  AND wins >= 2/3 seeds
  FAIL nếu PureC OOD exact-acc + 5pp < min(HR, Bag)
  else UNRESOLVED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

SEEDS = [42, 7, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---- Mini-SCAN vocabulary ----
# commands: PRIM [DIR] [MOD]
PRIMS = ["walk", "jump", "run", "look"]
DIRS = ["left", "right"]
MODS = ["twice", "thrice", "opposite", "around"]
# actions
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
N_TRAIN = 4000
N_VAL = 500
N_OOD = 800
EPOCHS = 30
LR = 1e-3
TEMP_START, TEMP_END = 1.2, 0.35

def expand_command(prim, direction=None, mod=None):
    """Deterministic Mini-SCAN semantics → action list."""
    base = []
    if direction:
        turn = DIR2ACT[direction]
        if mod == "opposite":
            # opposite left = right, etc.
            turn = "RTURN" if direction == "left" else "LTURN"
            base = [turn, PRIM2ACT[prim]]
        elif mod == "around":
            # around left = LTURN + prim x4 (simplified)
            base = [turn, PRIM2ACT[prim]] * 4
        elif mod == "twice":
            base = [turn, PRIM2ACT[prim]] * 2
        elif mod == "thrice":
            base = [turn, PRIM2ACT[prim]] * 3
        else:
            base = [turn, PRIM2ACT[prim]]
    else:
        if mod == "twice":
            base = [PRIM2ACT[prim]] * 2
        elif mod == "thrice":
            base = [PRIM2ACT[prim]] * 3
        elif mod == "around":
            base = [PRIM2ACT[prim]] * 4
        else:
            base = [PRIM2ACT[prim]]
    return base[:MAX_ACT]

def encode_cmd(tokens):
    ids = [CMD2ID[t] for t in tokens]
    ids = ids + [0] * (MAX_CMD - len(ids))
    return ids[:MAX_CMD]

def encode_act(actions):
    ids = [ACT2ID[a] for a in actions]
    ids = ids + [0] * (MAX_ACT - len(ids))
    return ids[:MAX_ACT]

def make_train_examples():
    """Short / simple: prim; prim dir; prim twice/thrice; prim dir twice."""
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
    """OOD: opposite, around, and longer combos not in train template set."""
    ex = []
    for p in PRIMS:
        for d in DIRS:
            ex.append(([p, d, "opposite"], expand_command(p, d, "opposite")))
            ex.append(([p, d, "around"], expand_command(p, d, "around")))
        ex.append(([p, "around"], expand_command(p, None, "around")))
        # novel order-ish: thrice after dir already in train; add opposite twice-like length
        for d in DIRS:
            ex.append(([p, d, "opposite"], expand_command(p, d, "opposite")))
    return ex

def sample_dataset(examples, n, seed):
    rng = np.random.RandomState(seed)
    X, Y, meta = [], [], []
    for i in range(n):
        cmd, act = examples[rng.randint(0, len(examples))]
        X.append(encode_cmd(cmd))
        Y.append(encode_act(act))
        meta.append((cmd, act))
    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long),
        meta,
    )

TRAIN_EX = make_train_examples()
OOD_EX = make_ood_examples()
print(f"Train templates: {len(TRAIN_EX)}  OOD templates: {len(OOD_EX)}")

# ---- Models ----

class CmdEncoder(nn.Module):
    def __init__(self, emb_dim=HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(N_CMD, emb_dim, padding_idx=0)
    def forward(self, x):
        # x: B, MAX_CMD
        e = self.emb(x)
        mask = (x != 0).float().unsqueeze(-1)
        summed = (e * mask).sum(1)
        denom = mask.sum(1).clamp(min=1.0)
        return summed / denom

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

class HierResidual(nn.Module):
    def __init__(self, n_blocks=8):
        super().__init__()
        self.enc = CmdEncoder(HIDDEN)
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

class BagMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CmdEncoder(HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 2), nn.Tanh(),
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, MAX_ACT * N_ACT),
        )
    def forward(self, x, **kwargs):
        h = self.enc(x)
        return self.mlp(h).view(-1, MAX_ACT, N_ACT)

class PureC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CmdEncoder(HIDDEN)
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

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def exact_match(logits, y):
    pred = logits.argmax(-1)
    # ignore pad positions in target for exact: full sequence including pads must match
    # better: match non-pad prefix
    B = y.size(0)
    ok = 0
    tok_correct = 0
    tok_total = 0
    for i in range(B):
        yi = y[i].tolist()
        pi = pred[i].tolist()
        # trim trailing pads
        while yi and yi[-1] == 0:
            yi.pop()
        while pi and pi[-1] == 0:
            pi.pop()
        if yi == pi:
            ok += 1
        L = max(len(yi), 1)
        for a, b in zip(yi, pi):
            tok_total += 1
            if a == b:
                tok_correct += 1
        # penalize length mismatch lightly in token metric
        tok_total += abs(len(yi) - len(pi))
    return ok / B, tok_correct / max(tok_total, 1)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, _ = sample_dataset(TRAIN_EX, N_TRAIN, seed)
    Xva, Yva, _ = sample_dataset(TRAIN_EX, N_VAL, seed + 10)
    Xood, Yood, _ = sample_dataset(OOD_EX, N_OOD, seed + 20)

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
                if use_temp:
                    logits = model(Xtr_d[idx], temperature=temp)
                else:
                    logits = model(Xtr_d[idx])
                loss = F.cross_entropy(logits.reshape(-1, N_ACT), Ytr_d[idx].reshape(-1), ignore_index=0)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                if use_temp:
                    logits = model(Xva_d, temperature=TEMP_END)
                else:
                    logits = model(Xva_d)
                val = F.cross_entropy(logits.reshape(-1, N_ACT), Yva_d.reshape(-1), ignore_index=0).item()
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return model

    def evaluate(model, use_temp=False):
        model.eval()
        X, Y = Xood.to(DEVICE), Yood.to(DEVICE)
        with torch.no_grad():
            if use_temp:
                logits = model(X, temperature=TEMP_END)
            else:
                logits = model(X)
            exact, tok = exact_match(logits, Y)
        return exact, tok

    out = {"seed": seed}
    bag = train(BagMLP())
    out["Bag_exact"], out["Bag_tok"] = evaluate(bag)
    hr = train(HierResidual())
    out["HR_exact"], out["HR_tok"] = evaluate(hr)
    pc = train(PureC(), use_temp=True)
    out["PC_exact"], out["PC_tok"] = evaluate(pc, use_temp=True)
    best_base = max(out["Bag_exact"], out["HR_exact"])
    out["win"] = out["PC_exact"] > best_base + 0.02
    return out


print(f"Params Bag={count_params(BagMLP())} HR={count_params(HierResidual())} PureC={count_params(PureC())}")

all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Bag exact={r['Bag_exact']:.3f} tok={r['Bag_tok']:.3f}", flush=True)
    print(f"  HR  exact={r['HR_exact']:.3f} tok={r['HR_tok']:.3f}", flush=True)
    print(f"  PC  exact={r['PC_exact']:.3f} tok={r['PC_tok']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== C Mini-SCAN SUMMARY =====")
for name in ["Bag", "HR", "PC"]:
    print(f"  {name}: exact={ms(f'{name}_exact')[0]:.3f}±{ms(f'{name}_exact')[1]:.3f}  "
          f"tok={ms(f'{name}_tok')[0]:.3f}")

pc_e = ms("PC_exact")[0]
bag_e = ms("Bag_exact")[0]
hr_e = ms("HR_exact")[0]
best_base = max(bag_e, hr_e)
wins = sum(r["win"] for r in all_results)

if pc_e > best_base + 0.05 and wins >= 2:
    outcome = "PASS"
elif pc_e + 0.05 < min(bag_e, hr_e):
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"PC={pc_e:.3f} best_base={best_base:.3f} wins={wins}/3")
print(f"=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "C Mini-SCAN compositional benchmark",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in
            ["Bag_exact","Bag_tok","HR_exact","HR_tok","PC_exact","PC_tok"]},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **C Mini-SCAN** PureC n_mod=6 λ=0 vs HR vs Bag
- Bag exact={bag_e:.3f} | HR={hr_e:.3f} | PC={pc_e:.3f} wins={wins}/3
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
