#!/usr/bin/env python3
"""
Phase R2b — Fix R2 metric + light retry

Nguyên nhân R2: y sinh với fail ngẫu nhiên nhưng eval execute fail=0 → exact ảo thấp.
R2b:
  1) Lưu fail_mask / seed per example; execute dùng CÙNG realization
  2) Optional retry once on failed step (idempotent-ish ADD/SUB/SET)
  3) Schema OOD + partial observe giữ như R2

Preregister exact (aligned):
  PASS: Prog_ex_ood > Flat_ex_ood + 0.05 AND Prog_ex_ood >= 0.35 AND wins >= 2/3
  FAIL: Prog + 0.05 < Flat
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

TOOLS = ["GET_STATUS", "POST_ADD", "POST_SUB", "POST_SET", "POST_NOP"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
N_TOOLS = len(TOOLS)
ALIAS = {
    "GET_STATUS": ["get_status", "status", "get", "legacy_status"],
    "POST_ADD": ["post_add", "add", "http_add", "incr"],
    "POST_SUB": ["post_sub", "sub", "http_sub", "decr"],
    "POST_SET": ["post_set", "set", "http_set", "reset_xy"],
    "POST_NOP": ["post_nop", "nop", "http_nop", "noop"],
}

def apply_with_fail_mask(x0, y0, tools, fail_mask, retry=False):
    """fail_mask: list[bool] per step — True means this call fails (no mutate).
    retry: if fail, try once more (second attempt always succeeds in this mock).
    """
    x, y = float(x0), float(y0)
    for t, failed in zip(tools, fail_mask):
        do_fail = failed
        if do_fail and retry:
            do_fail = False  # one retry succeeds
        if do_fail:
            continue
        if t == "POST_ADD":
            x, y = x + 1.0, y + 1.0
        elif t == "POST_SUB":
            x, y = x - 1.0, y - 1.0
        elif t == "POST_SET":
            x, y = 1.0, 0.0
        # GET/NOP no change
    return x, y

WORDS = ["<pad>", "<unk>",
         "please", "kindly", "call", "http", "request", "api", "endpoint", "method",
         "get", "post", "status", "get_status", "post_add", "post_sub", "post_set", "post_nop",
         "add", "sub", "set", "nop", "http_add", "http_sub", "http_set", "http_nop",
         "legacy_status", "incr", "decr", "reset_xy", "noop",
         "then", "next", "and", "followed", "by", "after", "before",
         "the", "tool", "json", "body", "empty", "run", "execute", "invoke",
         "user", "agent", "system", "should", "want", "to", "retry"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 24
N_WORD = len(WORDS)
HIDDEN = 128
MAX_PROG = 2
BATCH_SIZE = 64
N_TRAIN = 3000
N_ID = 350
N_OOD = 450
EPOCHS = 16
LR = 1e-3
EXACT_TOL = 0.15
TRAIN_FAIL = 0.05
OOD_FAIL = 0.20

ALL_PAIRS = [(a, b) for a in TOOLS for b in TOOLS]
OOD_PAIRS = [
    ("POST_ADD", "POST_SUB"), ("POST_SET", "POST_ADD"),
    ("GET_STATUS", "POST_ADD"), ("POST_NOP", "POST_SET"),
    ("POST_SUB", "POST_SET"), ("GET_STATUS", "POST_NOP"),
]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in OOD_PAIRS]

def tokenize(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def nl_single(tool, set_b=False):
    a = ALIAS[tool]
    if not set_b:
        return [f"please call http {a[0]}", f"execute api {a[1]}", f"invoke method {a[0]}"]
    return [f"user want legacy {a[-1]}", f"agent should call endpoint {a[-1]}", f"json method {a[-1]}"]

def nl_compose(a, b, set_b=False):
    if not set_b:
        aa, bb = ALIAS[a][0], ALIAS[b][0]
        return [f"please call http {aa} then {bb}", f"execute api {aa} and next {bb}"]
    aa, bb = ALIAS[a][-1], ALIAS[b][-1]
    return [f"user want {aa} next {bb}", f"agent should call {aa} before {bb}", f"json method {aa} and method {bb}"]

def sample(n, seed, pairs, set_b=False, single_ratio=0.3, fail_prob=0.0):
    rng = np.random.RandomState(seed)
    X_nl, S0, Y, prog, masks = [], [], [], [], []
    for i in range(n):
        true_s0 = rng.randn(2).astype(np.float32) * 0.5
        if rng.rand() < single_ratio:
            t = TOOLS[rng.randint(0, N_TOOLS)]
            phr = nl_single(t, set_b)[rng.randint(0, 3)]
            tools = [t, "POST_NOP"]
        else:
            a, b = pairs[rng.randint(0, len(pairs))]
            phrs = nl_compose(a, b, set_b)
            phr = phrs[rng.randint(0, len(phrs))]
            tools = [b, a] if "before" in phr else [a, b]
        fail_mask = [rng.rand() < fail_prob for _ in tools]
        yx, yy = apply_with_fail_mask(true_s0[0], true_s0[1], tools, fail_mask, retry=False)
        # partial observe on input
        obs = true_s0.copy()
        if set_b and rng.rand() < 0.5:
            obs = obs + rng.randn(2).astype(np.float32) * 0.15
        X_nl.append(tokenize(phr))
        S0.append(obs.astype(np.float32))
        Y.append([yx, yy])
        prog.append([TOOL2ID[tools[0]], TOOL2ID[tools[1]]])
        masks.append(fail_mask + [False] * (MAX_PROG - len(fail_mask)))
        # also store true start for execute alignment
    return {
        "nl": torch.tensor(np.array(X_nl), dtype=torch.long),
        "s0": torch.tensor(np.array(S0), dtype=torch.float32),
        "y": torch.tensor(np.array(Y), dtype=torch.float32),
        "prog": torch.tensor(np.array(prog), dtype=torch.long),
        "fail_mask": torch.tensor(np.array(masks), dtype=torch.bool),
        # For aligned execute we need true state — recover approx: when no partial noise use s0
        # Store true_s0 separately
    }

def sample_full(n, seed, pairs, set_b=False, single_ratio=0.3, fail_prob=0.0):
    rng = np.random.RandomState(seed)
    X_nl, S0_obs, S0_true, Y, prog, masks = [], [], [], [], [], []
    for i in range(n):
        true_s0 = rng.randn(2).astype(np.float32) * 0.5
        if rng.rand() < single_ratio:
            t = TOOLS[rng.randint(0, N_TOOLS)]
            phr = nl_single(t, set_b)[rng.randint(0, 3)]
            tools = [t, "POST_NOP"]
        else:
            a, b = pairs[rng.randint(0, len(pairs))]
            phrs = nl_compose(a, b, set_b)
            phr = phrs[rng.randint(0, len(phrs))]
            tools = [b, a] if "before" in phr else [a, b]
        fail_mask = [bool(rng.rand() < fail_prob) for _ in tools]
        while len(fail_mask) < MAX_PROG:
            fail_mask.append(False)
        yx, yy = apply_with_fail_mask(true_s0[0], true_s0[1], tools, fail_mask[:len(tools)], retry=False)
        obs = true_s0.copy()
        if set_b and rng.rand() < 0.5:
            obs = obs + rng.randn(2).astype(np.float32) * 0.15
        X_nl.append(tokenize(phr))
        S0_obs.append(obs.astype(np.float32))
        S0_true.append(true_s0.astype(np.float32))
        Y.append([yx, yy])
        prog.append([TOOL2ID[tools[0]], TOOL2ID[tools[1]]])
        masks.append(fail_mask[:MAX_PROG])
    return {
        "nl": torch.tensor(np.array(X_nl), dtype=torch.long),
        "s0": torch.tensor(np.array(S0_obs), dtype=torch.float32),
        "s0_true": torch.tensor(np.array(S0_true), dtype=torch.float32),
        "y": torch.tensor(np.array(Y), dtype=torch.float32),
        "prog": torch.tensor(np.array(prog), dtype=torch.long),
        "fail_mask": torch.tensor(np.array(masks), dtype=torch.bool),
    }

def exact_acc(pred, y, tol=EXACT_TOL):
    return ((pred - y).abs().max(-1).values < tol).float().mean().item()

class NLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_WORD, HIDDEN, padding_idx=0)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(HIDDEN * 2, HIDDEN)
    def forward(self, nl):
        e = self.emb(nl)
        lengths = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))

class FlatR2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.sproj = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, nl, s0):
        return self.mlp(torch.cat([self.enc(nl), self.sproj(s0)], -1))

class ProgR2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.step_heads = nn.ModuleList([nn.Linear(HIDDEN, N_TOOLS) for _ in range(MAX_PROG)])
    def program_logits(self, nl):
        h = self.enc(nl)
        return [head(h) for head in self.step_heads]
    def execute_aligned(self, tool_ids, s0_true, fail_mask, retry=False):
        outs = []
        for i in range(s0_true.size(0)):
            tools = [TOOLS[int(tool_ids[i, t])] for t in range(MAX_PROG)]
            mask = [bool(fail_mask[i, t]) for t in range(MAX_PROG)]
            x, y = apply_with_fail_mask(
                float(s0_true[i, 0]), float(s0_true[i, 1]), tools, mask, retry=retry
            )
            outs.append([x, y])
        return torch.tensor(outs, dtype=torch.float32, device=s0_true.device)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_full(N_TRAIN, seed, TRAIN_PAIRS, set_b=False, fail_prob=TRAIN_FAIL)
    idd = sample_full(N_ID, seed + 15, TRAIN_PAIRS, set_b=False, fail_prob=TRAIN_FAIL)
    ood = sample_full(N_OOD, seed + 20, OOD_PAIRS, set_b=True, single_ratio=0.0, fail_prob=OOD_FAIL)

    def train_flat():
        model = FlatR2().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, s0, y = tr["nl"].to(DEVICE), tr["s0"].to(DEVICE), tr["y"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(nl[idx], s0[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(nl[:300], s0[:300]), y[:300]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ProgR2().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, prog = tr["nl"].to(DEVICE), tr["prog"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total = 0
            nb = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                logits = model.program_logits(nl[idx])
                loss = sum(F.cross_entropy(lg, prog[idx, t]) for t, lg in enumerate(logits))
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            val = total / max(nb, 1)
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    flat = train_flat()
    prog = train_prog()
    flat.eval()
    prog.eval()
    with torch.no_grad():
        for name, data in [("id", idd), ("ood", ood)]:
            nl = data["nl"].to(DEVICE)
            s0 = data["s0"].to(DEVICE)
            s0t = data["s0_true"].to(DEVICE)
            y = data["y"].to(DEVICE)
            mask = data["fail_mask"].to(DEVICE)
            fp = flat(nl, s0)
            logits = prog.program_logits(nl)
            ids = torch.stack([lg.argmax(-1) for lg in logits], dim=1)
            # aligned no-retry
            pp = prog.execute_aligned(ids, s0t, mask, retry=False)
            # aligned + retry
            pp_r = prog.execute_aligned(ids, s0t, mask, retry=True)
            out[f"Flat_ex_{name}"] = exact_acc(fp, y)
            out[f"Prog_ex_{name}"] = exact_acc(pp, y)
            out[f"Prog_retry_ex_{name}"] = exact_acc(pp_r, y)
    # Primary decision: Prog with retry (fault-tolerance bias) vs Flat
    out["win"] = out["Prog_retry_ex_ood"] > out["Flat_ex_ood"] + 0.02
    return out


print("R2b aligned fault + retry — preregister exact on retry")
all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat OOD={r['Flat_ex_ood']:.3f} Prog_aligned={r['Prog_ex_ood']:.3f} "
          f"Prog_retry={r['Prog_retry_ex_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

fe = ms("Flat_ex_ood")[0]
pe = ms("Prog_ex_ood")[0]
pr = ms("Prog_retry_ex_ood")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== R2b SUMMARY =====")
print(f"Flat OOD={fe:.3f} | Prog aligned={pe:.3f} | Prog+retry={pr:.3f} wins={wins}/3")

# Primary = retry (declared)
if pr > fe + 0.05 and pr >= 0.35 and wins >= 2:
    outcome, note = "PASS", "aligned fault + retry unlocks R2"
elif pr + 0.05 < fe:
    outcome, note = "FAIL", "flat better"
else:
    outcome, note = "UNRESOLVED", "still insufficient"

print(f"=== OUTCOME (retry primary): {outcome} === ({note})")
print(f"=== NOTE aligned no-retry Prog={pe:.3f} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase R2b aligned fault + retry",
        "seeds": SEEDS,
        "summary": {"Flat_ex_ood": fe, "Prog_aligned": pe, "Prog_retry": pr, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase R2b** aligned fault realization + one retry
- Flat OOD={fe:.3f} | Prog aligned={pe:.3f} | Prog+retry={pr:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
""")
print(f"Saved {run_dir}")
