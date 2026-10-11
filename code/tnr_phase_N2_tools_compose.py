#!/usr/bin/env python3
"""
Phase N2 — Tools bank with composition IN train; OOD = novel pairs / paraphrases

N FAIL: single-tool train only.
N2 train: single + two-tool with template set A
OOD: two-tool template set B and/or held-out tool pairs
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

TOOLS = ["ADD", "SUB", "MUL", "SET", "NOP"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
N_TOOLS = len(TOOLS)
NAME = {"ADD": "add", "SUB": "sub", "MUL": "mul", "SET": "set", "NOP": "nop"}

def apply_tool(state, tool):
    x, y = state[:, 0], state[:, 1]
    if tool == "ADD":
        return torch.stack([x + 1.0, y + 1.0], -1)
    if tool == "SUB":
        return torch.stack([x - 1.0, y - 1.0], -1)
    if tool == "MUL":
        return torch.stack([x * 1.5, y * 1.5], -1)
    if tool == "SET":
        return torch.stack([torch.ones_like(x), torch.zeros_like(y)], -1)
    return state.clone()

def apply_program(state0, tool_names):
    s = state0.clone()
    for t in tool_names:
        s = apply_tool(s, t)
    return s

WORDS = ["<pad>", "<unk>",
         "please", "run", "call", "use", "invoke", "do", "execute",
         "add", "sub", "subtract", "mul", "multiply", "set", "reset", "nop", "nothing", "skip",
         "then", "after", "followed_by", "and_then", "next", "before",
         "the", "tool", "operation", "once"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 14
N_WORD = len(WORDS)
HIDDEN = 96
MAX_PROG = 2
BATCH_SIZE = 64
N_TRAIN = 4500
N_ID = 400
N_OOD = 700
EPOCHS = 28
LR = 1e-3

# Train pairs: all except a held-out set for OOD
ALL_PAIRS = [(a, b) for a in TOOLS for b in TOOLS]
OOD_PAIRS = [("ADD", "MUL"), ("SUB", "SET"), ("MUL", "ADD"), ("SET", "SUB"), ("NOP", "ADD"), ("MUL", "MUL")]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in OOD_PAIRS]

TRAIN_COMPOSE_TPL = [
    "{a} then {b}",
    "run {a} then {b}",
    "call {a} and_then {b}",
]
OOD_COMPOSE_TPL = [
    "{a} followed_by {b}",
    "execute {a} next {b}",
    "invoke {a} after {b}",  # after => b then a — handle carefully
]

def tokenize(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def sample(n, seed, pairs, templates, single_ratio=0.35, after_semantics=False):
    rng = np.random.RandomState(seed)
    X_nl, S0, Y, prog = [], [], [], []
    single_phr = {
        "ADD": ["run add", "call add", "use add"],
        "SUB": ["run sub", "call sub", "use subtract"],
        "MUL": ["run mul", "call mul", "use multiply"],
        "SET": ["run set", "call set", "use reset"],
        "NOP": ["run nop", "call nop", "use skip"],
    }
    for _ in range(n):
        s0 = rng.randn(2).astype(np.float32) * 0.8
        if rng.rand() < single_ratio:
            t = TOOLS[rng.randint(0, N_TOOLS)]
            phr = single_phr[t][rng.randint(0, len(single_phr[t]))]
            tools = [t, "NOP"]
        else:
            a, b = pairs[rng.randint(0, len(pairs))]
            tpl = templates[rng.randint(0, len(templates))]
            phr = tpl.format(a=NAME[a], b=NAME[b])
            if "after" in tpl:
                tools = [b, a]
            else:
                tools = [a, b]
        y = apply_program(torch.tensor(s0).unsqueeze(0), tools).squeeze(0).numpy()
        X_nl.append(tokenize(phr))
        S0.append(s0)
        Y.append(y)
        prog.append([TOOL2ID[tools[0]], TOOL2ID[tools[1]]])
    return {
        "nl": torch.tensor(np.array(X_nl), dtype=torch.long),
        "s0": torch.tensor(np.array(S0), dtype=torch.float32),
        "y": torch.tensor(np.array(Y), dtype=torch.float32),
        "prog": torch.tensor(np.array(prog), dtype=torch.long),
    }

def exact_acc(pred, y, tol=0.15):
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

class FlatTools(nn.Module):
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

class ToolProgramExecute(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.step_heads = nn.ModuleList([nn.Linear(HIDDEN, N_TOOLS) for _ in range(MAX_PROG)])

    def program_logits(self, nl):
        h = self.enc(nl)
        return [head(h) for head in self.step_heads]

    def execute(self, tool_ids, s0):
        s = s0.clone()
        for t in range(MAX_PROG):
            outs = [apply_tool(s[i:i+1], TOOLS[int(tool_ids[i, t])]) for i in range(s.size(0))]
            s = torch.cat(outs, 0)
        return s

    def forward_execute_greedy(self, nl, s0):
        logits = self.program_logits(nl)
        ids = torch.stack([lg.argmax(-1) for lg in logits], dim=1)
        return self.execute(ids, s0), ids


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample(N_TRAIN, seed, TRAIN_PAIRS, TRAIN_COMPOSE_TPL)
    idd = sample(N_ID, seed + 15, TRAIN_PAIRS, TRAIN_COMPOSE_TPL)
    # OOD: held-out pairs + OOD templates
    ood = sample(N_OOD, seed + 20, OOD_PAIRS, OOD_COMPOSE_TPL, single_ratio=0.0)

    def train_flat():
        model = FlatTools().to(DEVICE)
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
                val = F.mse_loss(model(nl[:512], s0[:512]), y[:512]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ToolProgramExecute().to(DEVICE)
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
            nl, s0, y = data["nl"].to(DEVICE), data["s0"].to(DEVICE), data["y"].to(DEVICE)
            fp = flat(nl, s0)
            pp, _ = prog.forward_execute_greedy(nl, s0)
            out[f"Flat_mse_{name}"] = F.mse_loss(fp, y).item()
            out[f"Prog_mse_{name}"] = F.mse_loss(pp, y).item()
            out[f"Flat_ex_{name}"] = exact_acc(fp, y)
            out[f"Prog_ex_{name}"] = exact_acc(pp, y)
    out["win"] = out["Prog_mse_ood"] < out["Flat_mse_ood"]
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat OOD mse={r['Flat_mse_ood']:.4f} ex={r['Flat_ex_ood']:.3f}", flush=True)
    print(f"  Prog OOD mse={r['Prog_mse_ood']:.4f} ex={r['Prog_ex_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

fm, pm = ms("Flat_mse_ood")[0], ms("Prog_mse_ood")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== N2 TOOLS COMPOSE SUMMARY =====")
print(f"Flat OOD mse={fm:.4f} ex={ms('Flat_ex_ood')[0]:.3f}")
print(f"Prog OOD mse={pm:.4f} ex={ms('Prog_ex_ood')[0]:.3f}")
print(f"ratio={pm/fm:.3f} wins={wins}/3")

if pm < fm * 0.95 and wins >= 2:
    outcome = "PASS"
elif pm > fm * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"
print(f"=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase N2 tools compose in train, OOD novel pairs+phrasing",
        "seeds": SEEDS,
        "summary": {"Flat_mse_ood": fm, "Prog_mse_ood": pm, "wins": wins,
                    "Prog_ex": ms("Prog_ex_ood")[0], "Flat_ex": ms("Flat_ex_ood")[0]},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase N2** tools bank composition in train; OOD novel pairs
- Flat OOD mse={fm:.4f} | Prog OOD mse={pm:.4f} ex={ms('Prog_ex_ood')[0]:.3f}
- Outcome: **{outcome}**
""")
print(f"Saved {run_dir}")
