#!/usr/bin/env python3
"""order_oracle_http_v1 — teacher-forced / oracle tools on HTTP (Shopee).

Hypothesis: with GOLD tool sequences (no NL decode), MarketplaceHttpClient
achieves high exact state — proving execute path is sound; e2e_v1 FAIL was
dominated by NL→program error, not HTTP/adapter.

Preregister (oracle path only — separate from e2e NL bar):
  PASS: mean oracle HTTP exact >= 0.90 on structure-held-out sequences
        AND CANCEL gold path shaped CANCELLED
        AND v0b prior PASS
  FAIL: mean oracle HTTP exact < 0.80
  else UNRESOLVED

Also REPORT (not pass/fail gate): NL→Prog tool-match rate on same OOD
to quantify NL gap without conflating execute failures.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from kernel.program import Program, Slot
from kernel.executor import Executor
from packages.order.connectors.file_store import apply_slot, empty_state
from packages.order.connectors.http_marketplace_mock import (
    MarketplaceHttpClient,
    start_marketplace_server,
)
from packages.order.connectors.adapters_marketplace import to_shopee
from packages.order.connectors import http_marketplace_mock as hmm

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "tools.json").read_text(encoding="utf-8")
)
TOOLS = SCHEMA["tools"]
TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
POLICIES = SCHEMA["policies"]

SEEDS = [42, 7, 123]
H, MAX_NL, L, EPOCHS, BS, LR = 96, 28, 2, 14, 64, 1e-3
N_TRAIN, N_OOD = 2400, 160

TRAIN_PHRASE = {
    "LOOKUP_ORDER": ["lookup order", "check order", "please retrieve the order record", "find the order"],
    "UPDATE_ADDRESS": ["update address", "change address", "kindly revise the delivery address", "revise address"],
    "CONFIRM": ["confirm action", "please confirm", "I confirm this operation", "confirm please"],
    "CANCEL_ORDER": ["cancel order", "void order", "please void this order now", "cancel this order"],
    "SET_STATUS": ["set status", "update status", "advance the order status", "change status"],
    "CREATE_ORDER": ["create order", "place order", "open a new order entry", "make a new order"],
    "NOP": ["do nothing", "skip step", "no operation required", "skip"],
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
OOD_SEQS = [
    ["LOOKUP_ORDER", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "SET_STATUS"],
    ["UPDATE_ADDRESS", "SET_STATUS"],
    ["CREATE_ORDER", "LOOKUP_ORDER"],
    ["CREATE_ORDER", "CONFIRM"],
    ["SET_STATUS", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "CONFIRM"],
    ["UPDATE_ADDRESS", "LOOKUP_ORDER"],
]
OOD_SET = {tuple(p) for p in OOD_SEQS}


def all_len2():
    return [[a, b] for a in TOOLS for b in TOOLS]


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


def is_valid_tools(tools):
    from kernel.validate import validate_program

    ok, _ = validate_program(make_program(tools), TOOLS, POLICIES)
    return ok


def gold_state_local(tools):
    return Executor(TOOLS, POLICIES, apply_slot, empty_state()).run(make_program(tools)).state


def build_splits():
    train = []
    for p in all_len2():
        if tuple(p) in OOD_SET:
            continue
        if is_valid_tools(pad_tools(p)):
            train.append(p)
    for extra in (
        ["CONFIRM", "CANCEL_ORDER"],
        ["CREATE_ORDER", "UPDATE_ADDRESS"],
        ["CREATE_ORDER", "SET_STATUS"],
        ["CREATE_ORDER", "NOP"],
        ["LOOKUP_ORDER", "NOP"],
    ):
        if tuple(extra) not in OOD_SET and is_valid_tools(pad_tools(extra)) and extra not in train:
            train.append(extra)
    ood = [p for p in OOD_SEQS if is_valid_tools(pad_tools(p))]
    return train, ood


words = set()
for d in (TRAIN_PHRASE, OOD_PHRASE):
    for phrases in d.values():
        for phr in phrases:
            for w in phr.lower().split():
                words.add(w)
WORDS = ["<pad>", "<unk>"] + sorted(words)
W2ID = {w: i for i, w in enumerate(WORDS)}


def tok(text):
    ids = [W2ID.get(w, 1) for w in text.lower().split()]
    return (ids + [0] * MAX_NL)[:MAX_NL]


def nl_for(tools, ood=False):
    ph = OOD_PHRASE if ood else TRAIN_PHRASE
    return " then ".join(ph[t][0] for t in tools)


def exact_core(a, b):
    keys = ["exists", "status", "cod", "cancelled", "confirmed"]
    return all(int(a.get(k, 0)) == int(b.get(k, 0)) for k in keys)


def exec_http(client, tools, oid="ORD-1"):
    st = empty_state()
    for t in tools:
        args = {}
        if t == "CREATE_ORDER":
            args = {"order_id": oid, "address": "addr-A", "cod": 1}
        elif t == "UPDATE_ADDRESS":
            args = {"address": "addr-B"}
        elif t == "SET_STATUS":
            args = {"status": 2, "cod": 0}
        elif t == "LOOKUP_ORDER":
            args = {"order_id": oid}
        try:
            st = client.apply(t, args, order_id=st.get("order_id") or oid)
        except RuntimeError:
            break
    return st


def oracle_http_exact(client, seqs, n, seed):
    """Teacher-forced: gold tools only → HTTP."""
    rng = np.random.RandomState(seed)
    hit = 0
    for i in range(n):
        tools = pad_tools(seqs[rng.randint(0, len(seqs))])
        # isolate per-example store key
        oid = f"O{seed}-{i}"
        tools_oid = list(tools)
        # rewrite CREATE/LOOKUP ids via exec with oid
        gold = gold_state_local(tools)
        # map gold to same oid semantics: recompute gold with oid in CREATE
        st = empty_state()
        for t in tools:
            args = {}
            if t == "CREATE_ORDER":
                args = {"order_id": oid, "address": "addr-A", "cod": 1}
            elif t == "UPDATE_ADDRESS":
                args = {"address": "addr-B"}
            elif t == "SET_STATUS":
                args = {"status": 2, "cod": 0}
            elif t == "LOOKUP_ORDER":
                args = {"order_id": oid}
            st = apply_slot(t, args, st)
        gold = st
        hmm._Handler.store.pop(oid, None)
        pred = empty_state()
        for t in tools:
            args = {}
            if t == "CREATE_ORDER":
                args = {"order_id": oid, "address": "addr-A", "cod": 1}
            elif t == "UPDATE_ADDRESS":
                args = {"address": "addr-B"}
            elif t == "SET_STATUS":
                args = {"status": 2, "cod": 0}
            elif t == "LOOKUP_ORDER":
                args = {"order_id": oid}
            try:
                pred = client.apply(t, args, order_id=pred.get("order_id") or oid)
            except RuntimeError:
                break
        if exact_core(pred, gold):
            hit += 1
    return hit / max(n, 1)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(WORDS), H, padding_idx=0)
        self.lstm = nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(H * 2, H)

    def forward(self, nl):
        e = self.emb(nl)
        lens = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))


class Prog(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.heads = nn.ModuleList([nn.Linear(H, len(TOOLS)) for _ in range(L)])

    def logits(self, x):
        h = self.enc(x)
        return [hd(h) for hd in self.heads]


def nl_tool_match(train_seqs, ood_seqs, seed):
    """Report-only: NL→Prog exact tool sequence rate on OOD (no HTTP)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    # train data
    X, P = [], []
    for _ in range(N_TRAIN):
        tools = pad_tools(train_seqs[rng.randint(0, len(train_seqs))])
        X.append(tok(nl_for(tools, ood=False)))
        P.append([TOOL2ID[t] for t in tools])
    X = torch.tensor(X, dtype=torch.long)
    P = torch.tensor(P, dtype=torch.long)
    prog = Prog()
    opt = torch.optim.Adam(prog.parameters(), lr=LR)
    for _ in range(EPOCHS):
        prog.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BS):
            idx = perm[i : i + BS]
            opt.zero_grad()
            lgs = prog.logits(X[idx])
            loss = sum(F.cross_entropy(lg, P[idx, t]) for t, lg in enumerate(lgs))
            loss.backward()
            opt.step()
    prog.eval()
    # OOD tool match
    hit = total = 0
    for _ in range(N_OOD):
        tools = pad_tools(ood_seqs[rng.randint(0, len(ood_seqs))])
        nl = torch.tensor([tok(nl_for(tools, ood=True))], dtype=torch.long)
        with torch.no_grad():
            pred = [TOOLS[int(lg.argmax(-1).item())] for lg in prog.logits(nl)]
        total += 1
        if pred == tools:
            hit += 1
    return hit / max(total, 1)


def main():
    print("=== order_oracle_http_v1: teacher-forced tools → HTTP ===")
    train_seqs, ood_seqs = build_splits()
    print(f"TRAIN={len(train_seqs)} OOD={len(ood_seqs)}")
    server, base = start_marketplace_server()
    client = MarketplaceHttpClient(base, shape="shopee")
    oracle_scores = []
    nl_scores = []
    try:
        for s in SEEDS:
            hmm._Handler.store = {}
            oe = oracle_http_exact(client, ood_seqs, N_OOD, s)
            oracle_scores.append(oe)
            print(f"SEED {s}: oracle_HTTP_exact={oe:.3f}")
        # cancel gold path
        hmm._Handler.store = {}
        st = empty_state()
        for t, args in [
            ("CREATE_ORDER", {"order_id": "CX", "address": "A", "cod": 1}),
            ("CONFIRM", {}),
            ("CANCEL_ORDER", {}),
        ]:
            st = client.apply(t, args, order_id=st.get("order_id") or "CX")
        shaped = to_shopee(st)
        cancel_ok = int(st.get("cancelled", 0)) == 1 and shaped.get("order_status") == "CANCELLED"
        print(f"gold_cancel_shaped={shaped.get('order_status')} ok={cancel_ok}")

        # report-only NL gap (1 seed for budget)
        print("NL tool-match (report-only, seed=42)...")
        nl_m = nl_tool_match(train_seqs, ood_seqs, 42)
        nl_scores.append(nl_m)
        print(f"NL_tool_match_ood={nl_m:.3f}")
    finally:
        server.shutdown()

    mean_o = float(np.mean(oracle_scores))
    v0b_path = ARTIFACTS / "r" / "uns" / "order_v0b" / "metrics.json"
    v0b = json.loads(v0b_path.read_text()) if v0b_path.exists() else {}
    v0b_ok = v0b.get("outcome") == "PASS"

    if mean_o >= 0.90 and cancel_ok and v0b_ok:
        outcome = "PASS"
    elif mean_o < 0.80:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_oracle_http_v1",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "oracle HTTP exact>=0.90 AND CANCELLED shaped AND v0b PASS",
            "FAIL": "oracle HTTP exact<0.80",
            "UNRESOLVED": "else",
            "report_only": "NL tool-match OOD (not a gate)",
        },
        "metrics": {
            "oracle_HTTP_exact_mean": mean_o,
            "oracle_per_seed": oracle_scores,
            "cancel_shaped_ok": cancel_ok,
            "NL_tool_match_ood_seed42": nl_scores[0] if nl_scores else None,
        },
        "v0b": {"pass": v0b_ok, "outcome": v0b.get("outcome")},
        "interpretation": "Separates execute (oracle) from NL decode (report-only)",
    }
    out = ARTIFACTS / "r" / "uns" / "order_oracle_http_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUMMARY oracle_HTTP={mean_o:.3f} cancel={cancel_ok} v0b={v0b_ok} → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
