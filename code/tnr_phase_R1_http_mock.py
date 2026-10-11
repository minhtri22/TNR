#!/usr/bin/env python3
"""
Phase R1 — Real track: mock HTTP / JSON tool API

Tools simulated as HTTP-style calls:
  GET_STATUS, POST_ADD, POST_SUB, POST_SET, POST_NOP
Each call mutates a server-side register state (like remote API).

NL → tool sequence → execute via mock HTTP client
vs Flat: NL + state → final state

Train: single + two-call compositions (template A)
OOD: held-out tool pairs + JSON-ish / HTTP-ish phrasing

Preregister (exact primary):
  PASS: Prog_ex_ood > Flat_ex_ood + 0.05 AND Prog_ex_ood >= 0.35 AND wins >= 2/3
  FAIL: Prog_ex + 0.05 < Flat_ex
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

# HTTP-style tool names
TOOLS = ["GET_STATUS", "POST_ADD", "POST_SUB", "POST_SET", "POST_NOP"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
N_TOOLS = len(TOOLS)
# short tokens in NL
ALIAS = {
    "GET_STATUS": ["get_status", "status", "get"],
    "POST_ADD": ["post_add", "add", "http_add"],
    "POST_SUB": ["post_sub", "sub", "http_sub"],
    "POST_SET": ["post_set", "set", "http_set"],
    "POST_NOP": ["post_nop", "nop", "http_nop"],
}

class MockHTTPServer:
    """Minimal stateful mock: register (x,y), methods mutate like REST."""
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)
        self.calls = []

    def call(self, tool):
        self.calls.append(tool)
        if tool == "GET_STATUS":
            return {"ok": True, "x": self.x, "y": self.y}
        if tool == "POST_ADD":
            self.x += 1.0
            self.y += 1.0
            return {"ok": True, "x": self.x, "y": self.y}
        if tool == "POST_SUB":
            self.x -= 1.0
            self.y -= 1.0
            return {"ok": True, "x": self.x, "y": self.y}
        if tool == "POST_SET":
            self.x, self.y = 1.0, 0.0
            return {"ok": True, "x": self.x, "y": self.y}
        if tool == "POST_NOP":
            return {"ok": True, "x": self.x, "y": self.y}
        return {"ok": False}

def apply_program(state0, tools):
    """state0: (B,2) or numpy; run mock server per sample."""
    if isinstance(state0, torch.Tensor):
        out = []
        for i in range(state0.size(0)):
            srv = MockHTTPServer(float(state0[i, 0]), float(state0[i, 1]))
            for t in tools:
                srv.call(t)
            out.append([srv.x, srv.y])
        return torch.tensor(out, dtype=torch.float32, device=state0.device)
    srv = MockHTTPServer(float(state0[0]), float(state0[1]))
    for t in tools:
        srv.call(t)
    return np.array([srv.x, srv.y], dtype=np.float32)

WORDS = ["<pad>", "<unk>",
         "please", "kindly", "call", "http", "request", "api", "endpoint", "method",
         "get", "post", "status", "get_status", "post_add", "post_sub", "post_set", "post_nop",
         "add", "sub", "set", "nop", "http_add", "http_sub", "http_set", "http_nop",
         "then", "next", "and", "followed", "by", "after", "before",
         "the", "tool", "json", "body", "empty", "run", "execute", "invoke",
         "user", "agent", "system", "should", "want", "to"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 24
N_WORD = len(WORDS)
HIDDEN = 128
MAX_PROG = 2
BATCH_SIZE = 64
N_TRAIN = 2800
N_ID = 400
N_OOD = 400
EPOCHS = 16
LR = 1e-3
EXACT_TOL = 0.15

ALL_PAIRS = [(a, b) for a in TOOLS for b in TOOLS]
OOD_PAIRS = [
    ("POST_ADD", "POST_SUB"), ("POST_SET", "POST_ADD"),
    ("GET_STATUS", "POST_ADD"), ("POST_MUL_PLACEHOLDER", "POST_NOP"),  # will filter
]
# fix OOD pairs to valid only
OOD_PAIRS = [
    ("POST_ADD", "POST_SUB"), ("POST_SET", "POST_ADD"),
    ("GET_STATUS", "POST_ADD"), ("POST_NOP", "POST_SET"),
    ("POST_SUB", "POST_SET"), ("GET_STATUS", "POST_NOP"),
]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in OOD_PAIRS]

def tokenize(text):
    ids = [W2ID.get(w, W2ID["<unk>"]) for w in text.lower().replace("/", " ").replace("{}", " ").split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]

def nl_single(tool, set_b=False):
    a = ALIAS[tool]
    if not set_b:
        return [
            f"please call http {a[0]}",
            f"execute api {a[1] if len(a)>1 else a[0]}",
            f"invoke method {a[0]}",
        ]
    return [
        f"user want to post request {a[-1]}",
        f"agent should call endpoint {a[0]}",
        f"json body empty method {a[1] if len(a)>1 else a[0]}",
    ]

def nl_compose(a, b, set_b=False):
    aa, bb = ALIAS[a][0], ALIAS[b][0]
    if not set_b:
        return [
            f"please call http {aa} then {bb}",
            f"execute api {aa} and next {bb}",
            f"invoke method {aa} followed by {bb}",
        ]
    return [
        f"user want endpoint {aa} next endpoint {bb}",
        f"agent should call {aa} before {bb}",  # before => b then a
        f"json method {aa} and method {bb}",
    ]

def sample(n, seed, pairs, set_b=False, single_ratio=0.3):
    rng = np.random.RandomState(seed)
    X_nl, S0, Y, prog = [], [], [], []
    for _ in range(n):
        s0 = rng.randn(2).astype(np.float32) * 0.5
        if rng.rand() < single_ratio:
            t = TOOLS[rng.randint(0, N_TOOLS)]
            phr = nl_single(t, set_b)[rng.randint(0, 3)]
            tools = [t, "POST_NOP"]
        else:
            a, b = pairs[rng.randint(0, len(pairs))]
            phrs = nl_compose(a, b, set_b)
            phr = phrs[rng.randint(0, len(phrs))]
            if "before" in phr:
                tools = [b, a]
            else:
                tools = [a, b]
        y = apply_program(s0, tools)
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

class FlatR1(nn.Module):
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

class ProgR1(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.step_heads = nn.ModuleList([nn.Linear(HIDDEN, N_TOOLS) for _ in range(MAX_PROG)])
    def program_logits(self, nl):
        h = self.enc(nl)
        return [head(h) for head in self.step_heads]
    def execute(self, tool_ids, s0):
        tools_batch = []
        for i in range(s0.size(0)):
            tools_batch.append([TOOLS[int(tool_ids[i, t])] for t in range(MAX_PROG)])
        outs = []
        for i, tools in enumerate(tools_batch):
            y = apply_program(s0[i].detach().cpu().numpy(), tools)
            outs.append(y)
        return torch.tensor(np.array(outs), dtype=torch.float32, device=s0.device)
    def forward_execute_greedy(self, nl, s0):
        logits = self.program_logits(nl)
        ids = torch.stack([lg.argmax(-1) for lg in logits], dim=1)
        return self.execute(ids, s0), ids


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample(N_TRAIN, seed, TRAIN_PAIRS, set_b=False)
    idd = sample(N_ID, seed + 15, TRAIN_PAIRS, set_b=False)
    ood = sample(N_OOD, seed + 20, OOD_PAIRS, set_b=True, single_ratio=0.0)

    def train_flat():
        model = FlatR1().to(DEVICE)
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
                val = F.mse_loss(model(nl[:400], s0[:400]), y[:400]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = ProgR1().to(DEVICE)
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
            out[f"Flat_ex_{name}"] = exact_acc(fp, y)
            out[f"Prog_ex_{name}"] = exact_acc(pp, y)
    out["win"] = out["Prog_ex_ood"] > out["Flat_ex_ood"] + 0.02
    return out


print("R1 Mock HTTP — preregister exact")
all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat OOD ex={r['Flat_ex_ood']:.3f} | Prog OOD ex={r['Prog_ex_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

fe, pe = ms("Flat_ex_ood")[0], ms("Prog_ex_ood")[0]
wins = sum(r["win"] for r in all_results)
print("\n===== R1 HTTP MOCK SUMMARY =====")
print(f"Flat OOD exact={fe:.3f} | Prog OOD exact={pe:.3f} wins={wins}/3")

if pe > fe + 0.05 and pe >= 0.35 and wins >= 2:
    outcome, note = "PASS", "mock HTTP program→execute exact"
elif pe + 0.05 < fe:
    outcome, note = "FAIL", "flat better"
else:
    outcome, note = "UNRESOLVED", "gap/threshold"

print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase R1 mock HTTP JSON tools",
        "seeds": SEEDS,
        "summary": {"Flat_ex_ood": fe, "Prog_ex_ood": pe, "wins": wins},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase R1** mock HTTP/JSON tools NL→program→execute
- Flat OOD exact={fe:.3f} | Prog OOD exact={pe:.3f} wins={wins}/3
- Outcome: **{outcome}** ({note})
- Control-flow deferred by plan (after R1)
""")
print(f"Saved {run_dir}")
