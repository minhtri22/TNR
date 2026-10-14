#!/usr/bin/env python3
"""order_nl_lexicon_en_v1 — full EN lexicon file + gate (before VI).

Hypothesis: en_v1.json patterns cover schema tools + listed paraphrases;
tool_match and HTTP exact stay >= 0.90 on held-out structure sequences
drawn from listed EN phrases only.

Preregister:
  PASS: tool_match_mean>=0.90 AND http_exact_mean>=0.90 AND v0b PASS
  FAIL: tool_match_mean<0.70
  else UNRESOLVED
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from packages.order.lexicon import compile_patterns, load_lexicon, parse_nl
from packages.order.connectors.file_store import apply_slot, empty_state
from packages.order.connectors.http_marketplace_mock import (
    MarketplaceHttpClient,
    start_marketplace_server,
)
from packages.order.connectors import http_marketplace_mock as hmm
from packages.order.connectors.adapters_marketplace import to_shopee

LEX = load_lexicon("en", "v1")
COMPILED = compile_patterns(LEX)
TOOLS = list(LEX["tools"].keys())

# Build phrase pools from lexicon file (all listed forms)
PHRASES = {t: list(LEX["tools"][t]["patterns"]) for t in TOOLS}

SEQS = [
    ["LOOKUP_ORDER", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "SET_STATUS"],
    ["UPDATE_ADDRESS", "SET_STATUS"],
    ["CREATE_ORDER", "LOOKUP_ORDER"],
    ["CREATE_ORDER", "CONFIRM"],
    ["SET_STATUS", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "CONFIRM"],
    ["UPDATE_ADDRESS", "LOOKUP_ORDER"],
    ["CONFIRM", "CANCEL_ORDER"],
    ["CREATE_ORDER", "SET_STATUS"],
    ["CREATE_ORDER", "UPDATE_ADDRESS"],
    ["LOOKUP_ORDER", "NOP"],
]

SEEDS = [42, 7, 123]
N_OOD = 100
L = 2


def nl_for(tools, rng):
    parts = []
    for t in tools:
        opts = PHRASES[t]
        parts.append(opts[int(rng.randint(0, len(opts)))])
    return " then ".join(parts)


def pad_tools(tools):
    t = list(tools)[:L]
    while len(t) < L:
        t.append("NOP")
    return t


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
        gold = pad_tools(SEQS[rng.randint(0, len(SEQS))])
        text = nl_for(gold, rng)
        pred = parse_nl(text, COMPILED, max_slots=L)
        if pred == gold:
            match += 1
        oid = f"EN{seed}-{i}"
        hmm._Handler.store.pop(oid, None)
        g = gold_state(gold, oid)
        p = exec_http(client, pred, oid)
        if exact_core(p, g):
            http_hit += 1
    return {"tool_match": match / N_OOD, "http_exact": http_hit / N_OOD}


def main():
    print("=== order_nl_lexicon_en_v1: EN lexicon file gate ===")
    print(f"patterns loaded: {sum(len(v) for v in PHRASES.values())} over {len(TOOLS)} tools")
    server, base = start_marketplace_server()
    client = MarketplaceHttpClient(base, shape="shopee")
    results = []
    try:
        for s in SEEDS:
            hmm._Handler.store = {}
            r = eval_seed(s, client)
            results.append(r)
            print(f"SEED {s}: tool_match={r['tool_match']:.3f} http_exact={r['http_exact']:.3f}")
        hmm._Handler.store = {}
        st = empty_state()
        for t, args in [
            ("CREATE_ORDER", {"order_id": "CX", "address": "A", "cod": 1}),
            ("CONFIRM", {}),
            ("CANCEL_ORDER", {}),
        ]:
            st = client.apply(t, args, order_id=st.get("order_id") or "CX")
        cancel_ok = to_shopee(st).get("order_status") == "CANCELLED"
    finally:
        server.shutdown()

    tm = float(np.mean([r["tool_match"] for r in results]))
    he = float(np.mean([r["http_exact"] for r in results]))
    v0b_path = ARTIFACTS / "r" / "uns" / "order_v0b" / "metrics.json"
    v0b = json.loads(v0b_path.read_text()) if v0b_path.exists() else {}
    v0b_ok = v0b.get("outcome") == "PASS"

    if tm >= 0.90 and he >= 0.90 and v0b_ok and cancel_ok:
        outcome = "PASS"
    elif tm < 0.70:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_nl_lexicon_en_v1",
        "pack": "order",
        "lang": "en",
        "outcome": outcome,
        "preregister": {
            "PASS": "tool_match>=0.90 AND http_exact>=0.90 AND v0b PASS",
            "FAIL": "tool_match<0.70",
            "UNRESOLVED": "else",
        },
        "metrics": {
            "tool_match_mean": tm,
            "http_exact_mean": he,
            "per_seed": results,
            "n_patterns": sum(len(v) for v in PHRASES.values()),
            "cancel_shaped_ok": cancel_ok,
        },
        "v0b": {"pass": v0b_ok, "outcome": v0b.get("outcome")},
        "lexicon_file": "packages/order/lexicon/en_v1.json",
    }
    out = ARTIFACTS / "r" / "uns" / "order_nl_lexicon_en_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUMMARY tool_match={tm:.3f} http_exact={he:.3f} → {outcome}")
    print(f"scorecard → {path}")
    if outcome == "PASS":
        print("EN lexicon PASS — Vietnamese lexicon may proceed as next hypothesis.")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
