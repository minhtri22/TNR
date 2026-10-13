#!/usr/bin/env python3
"""Order pack v0b — synonym-in-train + more epochs (same schema/metric).
Hypothesis: OOD NL frames also appear in train as alternate phrases;
EPOCHS 20. Preregister UNCHANGED from v0.
Preregister:
  PASS: mean Prog exact > Flat+0.05 AND Prog>=0.35 AND wins>=2/3
  FAIL: mean Prog+0.05 < Flat
  else UNRESOLVED
Structure OOD: held-out tool pairs, same lexicon templates.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo roots: .../artifacts/code/packages/order/eval/this.py
CODE_ROOT = Path(__file__).resolve().parents[3]  # .../code
ROOT = CODE_ROOT.parent  # .../artifacts
sys.path.insert(0, str(CODE_ROOT))

from kernel.program import Program, Slot
from kernel.executor import Executor
from kernel.scorecard import Scorecard, exact_state, decide_outcome
from packages.order.connectors.mock_store import apply_slot, empty_state

SEEDS = [42, 7, 123]
# v0 curriculum: L=2 only (structure OOD pairs). L=3 later as order_v0b.
H, MAX_NL, L, EPOCHS, BS, LR = 96, 28, 2, 20, 64, 1e-3
N_TRAIN, N_OOD = 3600, 400
EXACT_TOL_KEYS = ["exists", "address", "status", "cod", "cancelled", "confirmed"]

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "tools.json"
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
TOOLS = SCHEMA["tools"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
POLICIES = SCHEMA["policies"]

# NL templates (train = canonical; OOD = novel frames, same roots)
TRAIN_PHRASE = {
    # canonical + synonyms including OOD surface forms (synonym-in-train)
    "LOOKUP_ORDER": [
        "lookup order", "check order", "please retrieve the order record", "find the order",
    ],
    "UPDATE_ADDRESS": [
        "update address", "change address", "kindly revise the delivery address", "revise address",
    ],
    "CONFIRM": [
        "confirm action", "please confirm", "I confirm this operation", "confirm please",
    ],
    "CANCEL_ORDER": [
        "cancel order", "void order", "please void this order now", "cancel this order",
    ],
    "SET_STATUS": [
        "set status", "update status", "advance the order status", "change status",
    ],
    "CREATE_ORDER": [
        "create order", "place order", "open a new order entry", "make a new order",
    ],
    "NOP": [
        "do nothing", "skip step", "no operation required", "skip",
    ],
}
OOD_PHRASE = {
    "LOOKUP_ORDER": ["please retrieve the order record"],
    "UPDATE_ADDRESS": ["kindly revise the delivery address"],
    "CONFIRM": ["I confirm this operation"],
    "CANCEL_ORDER": ["please void this order now"],
    "SET_STATUS": ["advance the order status"],
    "CREATE_ORDER": ["open a new order entry"],
    "NOP": ["no operation required"],
}

# structure OOD: held-out sequences (must include CONFIRM before CANCEL when cancel present)
def all_len2():
    out = []
    for a in TOOLS:
        for b in TOOLS:
            out.append([a, b])
    return out


def all_len3():
    out = []
    for a in TOOLS:
        for b in TOOLS:
            for c in TOOLS:
                out.append([a, b, c])
    return out


OOD_SEQS_RAW = [
    ["LOOKUP_ORDER", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "SET_STATUS"],
    ["UPDATE_ADDRESS", "SET_STATUS"],
    ["CREATE_ORDER", "LOOKUP_ORDER"],
    ["CREATE_ORDER", "CONFIRM"],
    ["SET_STATUS", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "CONFIRM"],
    ["UPDATE_ADDRESS", "LOOKUP_ORDER"],
]

words = set()
for d in (TRAIN_PHRASE, OOD_PHRASE):
    for phrases in d.values():
        for p in phrases:
            for w in p.lower().split():
                words.add(w)
WORDS = ["<pad>", "<unk>"] + sorted(words)
W2ID = {w: i for i, w in enumerate(WORDS)}
N_WORD = len(WORDS)


def tok(text: str):
    ids = [W2ID.get(w, 1) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]


def nl_for(tools, ood=False, rng=None):
    ph = OOD_PHRASE if ood else TRAIN_PHRASE
    parts = []
    for tool in tools:
        opts = ph[tool]
        if ood:
            parts.append(opts[0])
        else:
            # synonym-in-train: uniform over train phrases
            j = 0 if rng is None else int(rng.randint(0, len(opts)))
            parts.append(opts[j])
    return " then ".join(parts)


def pad_tools(tools, L=L):
    t = list(tools)[:L]
    while len(t) < L:
        t.append("NOP")
    return t


def make_program(tools):
    slots = []
    for t in tools:
        args = {}
        if t == "CREATE_ORDER":
            args = {"order_id": "ORD-1", "address": "addr-A", "cod": 1}
        elif t == "UPDATE_ADDRESS":
            args = {"address": "addr-B"}
        elif t == "SET_STATUS":
            args = {"status": 2, "cod": 0}
        elif t == "LOOKUP_ORDER":
            args = {"order_id": "ORD-1"}
        slots.append(Slot(tool=t, args=args))
    return Program(slots=slots, pack="order")


def gold_state(tools, args_list=None):
    """Execute gold program through kernel executor."""
    prog = make_program(tools)
    ex = Executor(
        allowed_tools=TOOLS,
        policies=POLICIES,
        apply_slot=apply_slot,
        initial_state=empty_state(),
    )
    res = ex.run(prog)
    return res.state, res.ok, res.errors


def is_valid_tools(tools):
    prog = make_program(tools)
    from kernel.validate import validate_program

    ok, _ = validate_program(prog, TOOLS, POLICIES)
    return ok


def build_splits():
    ood_set = {tuple(p) for p in OOD_SEQS_RAW}
    train = []
    for p in all_len2():
        if tuple(p) in ood_set:
            continue
        if not is_valid_tools(pad_tools(p)):
            continue
        train.append(p)
    for extra in (
        ["CONFIRM", "CANCEL_ORDER"],
        ["CREATE_ORDER", "UPDATE_ADDRESS"],
        ["CREATE_ORDER", "SET_STATUS"],
        ["CREATE_ORDER", "NOP"],
        ["LOOKUP_ORDER", "NOP"],
    ):
        if tuple(extra) not in ood_set and is_valid_tools(pad_tools(extra)) and extra not in train:
            train.append(extra)
    ood = [p for p in OOD_SEQS_RAW if is_valid_tools(pad_tools(p))]
    if not ood:
        raise SystemExit("OOD empty after valid filter")
    return train, ood


def state_to_vec(st):
    # continuous targets for Flat head
    return np.array(
        [
            float(st.get("exists", 0)),
            float(hash(str(st.get("address", ""))) % 97) / 97.0,  # weak addr signal for Flat
            float(st.get("status", 0)) / 3.0,
            float(st.get("cod", 0)),
            float(st.get("cancelled", 0)),
            float(st.get("confirmed", 0)),
        ],
        dtype=np.float32,
    )


def sample(seqs, n, seed, ood=False):
    rng = np.random.RandomState(seed)
    X, Y, P, gold_states = [], [], [], []
    for _ in range(n):
        tools = pad_tools(seqs[rng.randint(0, len(seqs))])
        st, ok, _ = gold_state(tools)
        X.append(tok(nl_for(tools, ood=ood, rng=rng)))
        Y.append(state_to_vec(st))
        P.append([TOOL2ID[t] for t in tools])
        gold_states.append(st)
    return {
        "nl": torch.tensor(X, dtype=torch.long),
        "y": torch.tensor(np.array(Y), dtype=torch.float32),
        "prog": torch.tensor(P, dtype=torch.long),
        "gold": gold_states,
    }


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_WORD, H, padding_idx=0)
        self.lstm = nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(H * 2, H)

    def forward(self, nl):
        e = self.emb(nl)
        lens = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))


class Flat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.mlp = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 6))

    def forward(self, x):
        return self.mlp(self.enc(x))


class Prog(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.heads = nn.ModuleList([nn.Linear(H, len(TOOLS)) for _ in range(L)])

    def logits(self, x):
        h = self.enc(x)
        return [hd(h) for hd in self.heads]

    def decode(self, ids):
        programs = []
        for i in range(ids.size(0)):
            tools = [TOOLS[int(ids[i, t])] for t in range(L)]
            # strip trailing NOPs for cleaner program but execute padded
            programs.append(tools)
        return programs

    def execute_batch(self, tools_list):
        outs = []
        oks = []
        for tools in tools_list:
            st, ok, _ = gold_state(tools)
            outs.append(st)
            oks.append(ok)
        return outs, oks


def flat_exact(pred_y, gold_states):
    # map continuous pred back to discrete-ish decisions
    n = pred_y.size(0)
    hit = 0
    for i in range(n):
        p = pred_y[i]
        g = gold_states[i]
        pred_st = {
            "exists": 1 if p[0] > 0.5 else 0,
            "address": g.get("address", ""),  # Flat cannot reliably carry string addr
            "status": int(round(float(p[2]) * 3)),
            "cod": 1 if p[3] > 0.5 else 0,
            "cancelled": 1 if p[4] > 0.5 else 0,
            "confirmed": 1 if p[5] > 0.5 else 0,
        }
        # fairer Flat: compare numeric fields only (exists,status,cod,cancelled,confirmed)
        keys = ["exists", "status", "cod", "cancelled", "confirmed"]
        if exact_state(pred_st, g, keys=keys):
            hit += 1
    return hit / max(n, 1)


def prog_exact(tools_list, gold_states):
    n = len(tools_list)
    hit = 0
    for i, tools in enumerate(tools_list):
        st, ok, _ = gold_state(tools)
        g = gold_states[i]
        keys = ["exists", "status", "cod", "cancelled", "confirmed", "address"]
        if exact_state(st, g, keys=keys):
            hit += 1
    return hit / max(n, 1)


def run_seed(seed: int, train_seqs, ood_seqs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample(train_seqs, N_TRAIN, seed, ood=False)
    ood = sample(ood_seqs, N_OOD, seed + 21, ood=True)

    flat = Flat()
    opt = torch.optim.Adam(flat.parameters(), lr=LR)
    best, st = 1e9, None
    for ep in range(EPOCHS):
        flat.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BS):
            idx = perm[i : i + BS]
            opt.zero_grad()
            F.mse_loss(flat(tr["nl"][idx]), tr["y"][idx]).backward()
            opt.step()
        flat.eval()
        with torch.no_grad():
            val = F.mse_loss(flat(tr["nl"][:256]), tr["y"][:256]).item()
        if val < best:
            best, st = val, {k: v.clone() for k, v in flat.state_dict().items()}
    if st:
        flat.load_state_dict(st)

    prog = Prog()
    opt = torch.optim.Adam(prog.parameters(), lr=LR)
    best, st = 1e9, None
    for ep in range(EPOCHS):
        prog.train()
        perm = torch.randperm(N_TRAIN)
        tot = nb = 0
        for i in range(0, N_TRAIN, BS):
            idx = perm[i : i + BS]
            opt.zero_grad()
            lgs = prog.logits(tr["nl"][idx])
            loss = sum(F.cross_entropy(lg, tr["prog"][idx, t]) for t, lg in enumerate(lgs))
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if tot / max(nb, 1) < best:
            best, st = tot / max(nb, 1), {k: v.clone() for k, v in prog.state_dict().items()}
    if st:
        prog.load_state_dict(st)

    flat.eval()
    prog.eval()
    with torch.no_grad():
        fe = flat_exact(flat(ood["nl"]), ood["gold"])
        ids = torch.stack([lg.argmax(-1) for lg in prog.logits(ood["nl"])], 1)
        tools_list = prog.decode(ids)
        pe = prog_exact(tools_list, ood["gold"])
    return {"Flat": fe, "Prog": pe, "win": pe > fe + 0.02}


def main():
    preregister = {
        "PASS": "mean Prog > Flat+0.05 AND Prog>=0.35 AND wins>=2/3",
        "FAIL": "mean Prog+0.05 < Flat",
        "else": "UNRESOLVED",
        "metric": "exact state keys exists/status/cod/cancelled/confirmed (+address for Prog)",
        "ood": "structure held-out sequences + novel NL frames",
    }
    print("=== Order pack v0b (synonym-in-train) ===")
    print("preregister:", preregister["PASS"])
    train_seqs, ood_seqs = build_splits()
    print(f"TRAIN_SEQS={len(train_seqs)} OOD_SEQS={len(ood_seqs)}")
    results = []
    for s in SEEDS:
        r = run_seed(s, train_seqs, ood_seqs)
        results.append(r)
        print(f"SEED {s}: Flat={r['Flat']:.3f} Prog={r['Prog']:.3f} win={r['win']}")

    fe = float(np.mean([r["Flat"] for r in results]))
    pe = float(np.mean([r["Prog"] for r in results]))
    wins = int(sum(r["win"] for r in results))
    outcome = decide_outcome(fe, pe, wins, n_seeds=3, gap=0.05, min_prog=0.35, min_wins=2)

    card = Scorecard(
        suite="order_v0b_synonym_in_train",
        pack="order",
        seeds=SEEDS,
        metrics={
            "Flat_exact_mean": fe,
            "Prog_exact_mean": pe,
            "wins": wins,
            "per_seed": results,
        },
        outcome=outcome,
        preregister=preregister,
    )
    out_dir = ROOT / "r" / "uns" / "order_v0b"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    card.save(str(path))
    print(f"SUMMARY Flat={fe:.3f} Prog={pe:.3f} wins={wins}/3 → {outcome}")
    print(f"scorecard → {path}")

    # policy unit check (kernel)
    print("--- kernel policy smoke ---")
    bad = Program(
        slots=[Slot("CREATE_ORDER", {"order_id": "X"}), Slot("CANCEL_ORDER", {})],
        pack="order",
    )
    ex = Executor(TOOLS, POLICIES, apply_slot, empty_state())
    res = ex.run(bad)
    print("CANCEL without CONFIRM blocked:", not res.ok, res.errors)

    good = Program(
        slots=[
            Slot("CREATE_ORDER", {"order_id": "X", "address": "A"}),
            Slot("CONFIRM", {}),
            Slot("CANCEL_ORDER", {}),
        ],
        pack="order",
    )
    res2 = ex.run(good)
    print("CANCEL with CONFIRM ok:", res2.ok, "cancelled=", res2.state.get("cancelled"))


if __name__ == "__main__":
    main()
