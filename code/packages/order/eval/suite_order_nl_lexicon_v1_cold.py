#!/usr/bin/env python3
"""order_nl_lexicon_v1_cold — lexicon on TRUE cold-start NL (outside dictionary).

Hypothesis: fixed lexicon fails when OOD phrases share no listed patterns
(same spirit as order_v0c). Expect FAIL or UNRESOLVED — controlled negative.

Preregister:
  PASS: tool_match>=0.90 (unexpected for pure cold-start)
  FAIL: tool_match mean < 0.70
  else UNRESOLVED
Does not change v0b / lexicon_v1 PASS baselines.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from packages.order.connectors.file_store import apply_slot, empty_state
from packages.order.connectors.http_marketplace_mock import (
    MarketplaceHttpClient,
    start_marketplace_server,
)
from packages.order.connectors import http_marketplace_mock as hmm

# Same lexicon as v1 (canonical + synonym listed forms only)
LEXICON = [
    (r"retrieve the order record|lookup order|check order|find the order", "LOOKUP_ORDER"),
    (r"revise the delivery address|update address|change address|revise address", "UPDATE_ADDRESS"),
    (r"confirm this operation|confirm action|please confirm|confirm please|i confirm", "CONFIRM"),
    (r"void this order now|cancel order|void order|cancel this order", "CANCEL_ORDER"),
    (r"advance the order status|set status|update status|change status", "SET_STATUS"),
    (r"open a new order entry|create order|place order|make a new order", "CREATE_ORDER"),
    (r"no operation required|do nothing|skip step|skip", "NOP"),
]

# Cold-start phrases — disjoint from lexicon patterns (order_v0c style)
COLD_PHRASE = {
    "LOOKUP_ORDER": ["retrieve shipment identifier now"],
    "UPDATE_ADDRESS": ["relocate delivery destination"],
    "CONFIRM": ["authorize the transaction explicitly"],
    "CANCEL_ORDER": ["abort purchase immediately"],
    "SET_STATUS": ["progress fulfillment phase"],
    "CREATE_ORDER": ["register merchandise request"],
    "NOP": ["idle without mutation"],
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
    ["CONFIRM", "CANCEL_ORDER"],
]

SEEDS = [42, 7, 123]
N_OOD = 80
L = 2


def pad_tools(tools):
    t = list(tools)[:L]
    while len(t) < L:
        t.append("NOP")
    return t


def nl_for(tools):
    return " then ".join(COLD_PHRASE[t][0] for t in tools)


def lexicon_parse(text: str) -> list:
    parts = re.split(r"\s+then\s+|[.;]", text, flags=re.I)
    tools = []
    for part in parts:
        part = part.strip().lower()
        if not part:
            continue
        hit = None
        for pat, tool in LEXICON:
            if re.search(pat, part, flags=re.I):
                hit = tool
                break
        tools.append(hit if hit else "NOP")
    return pad_tools(tools if tools else ["NOP"])


def exact_core(a, b):
    keys = ["exists", "status", "cod", "cancelled", "confirmed"]
    return all(int(a.get(k, 0)) == int(b.get(k, 0)) for k in keys)


def gold_state(tools, oid):
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
    return st


def exec_http(client, tools, oid):
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


def eval_seed(seed, client):
    rng = np.random.RandomState(seed)
    match = http_hit = 0
    for i in range(N_OOD):
        gold_tools = pad_tools(OOD_SEQS[rng.randint(0, len(OOD_SEQS))])
        text = nl_for(gold_tools)
        pred = lexicon_parse(text)
        if pred == gold_tools:
            match += 1
        oid = f"C{seed}-{i}"
        hmm._Handler.store.pop(oid, None)
        g = gold_state(gold_tools, oid)
        p = exec_http(client, pred, oid)
        if exact_core(p, g):
            http_hit += 1
    return {"tool_match": match / N_OOD, "http_exact": http_hit / N_OOD}


def main():
    print("=== order_nl_lexicon_v1_cold: cold-start NL ===")
    server, base = start_marketplace_server()
    client = MarketplaceHttpClient(base, shape="shopee")
    results = []
    try:
        for s in SEEDS:
            hmm._Handler.store = {}
            r = eval_seed(s, client)
            results.append(r)
            print(f"SEED {s}: tool_match={r['tool_match']:.3f} http_exact={r['http_exact']:.3f}")
    finally:
        server.shutdown()

    tm = float(np.mean([r["tool_match"] for r in results]))
    he = float(np.mean([r["http_exact"] for r in results]))
    if tm >= 0.90:
        outcome = "PASS"
    elif tm < 0.70:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_nl_lexicon_v1_cold",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "tool_match>=0.90 (unexpected)",
            "FAIL": "tool_match<0.70",
            "UNRESOLVED": "else",
        },
        "metrics": {"tool_match_mean": tm, "http_exact_mean": he, "per_seed": results},
        "note": "Controlled negative: lexicon without listed surface forms",
    }
    out = ARTIFACTS / "r" / "uns" / "order_nl_lexicon_v1_cold"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUMMARY tool_match={tm:.3f} http_exact={he:.3f} → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
