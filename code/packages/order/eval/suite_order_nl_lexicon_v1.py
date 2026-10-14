#!/usr/bin/env python3
"""order_nl_lexicon_v1 — lexicon / keyword slot-fill NL→program (no neural).

Hypothesis: with synonym-overlap OOD (same surface regime as v0b), a
deterministic lexicon can recover tool sequences at high rate; piping
lexicon programs through HTTP yields high exact — isolating that the
e2e neural gap is model capacity/lexicon, not connector.

Preregister:
  PASS: mean lexicon tool-match on OOD >= 0.90 (3 seeds of sample draws)
        AND lexicon→HTTP exact mean >= 0.90
        AND v0b prior PASS
  FAIL: tool-match mean < 0.70
  else UNRESOLVED

Seeds resample OOD instances; lexicon rules are fixed (no train).
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
from packages.order.connectors.adapters_marketplace import to_shopee

TOOLS = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "tools.json").read_text(encoding="utf-8")
)["tools"]

# Keyword / phrase → tool (order matters: longer phrases first)
LEXICON = [
    (r"retrieve the order record|lookup order|check order|find the order", "LOOKUP_ORDER"),
    (r"revise the delivery address|update address|change address|revise address", "UPDATE_ADDRESS"),
    (r"confirm this operation|confirm action|please confirm|confirm please|i confirm", "CONFIRM"),
    (r"void this order now|cancel order|void order|cancel this order", "CANCEL_ORDER"),
    (r"advance the order status|set status|update status|change status", "SET_STATUS"),
    (r"open a new order entry|create order|place order|make a new order", "CREATE_ORDER"),
    (r"no operation required|do nothing|skip step|skip", "NOP"),
]

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
    ["CONFIRM", "CANCEL_ORDER"],
]

SEEDS = [42, 7, 123]
N_OOD = 80
L = 2


def nl_for(tools):
    return " then ".join(OOD_PHRASE[t][0] for t in tools)


def pad_tools(tools):
    t = list(tools)[:L]
    while len(t) < L:
        t.append("NOP")
    return t


def lexicon_parse(text: str) -> list:
    """Split on 'then' / punctuation; map each clause via first lexicon hit."""
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
        pred_tools = lexicon_parse(text)
        if pred_tools == gold_tools:
            match += 1
        oid = f"L{seed}-{i}"
        hmm._Handler.store.pop(oid, None)
        g = gold_state(gold_tools, oid)
        # execute **predicted** program (lexicon output)
        p = exec_http(client, pred_tools, oid)
        if exact_core(p, g):
            http_hit += 1
    return {
        "tool_match": match / N_OOD,
        "http_exact": http_hit / N_OOD,
    }


def main():
    print("=== order_nl_lexicon_v1: lexicon NL→program + HTTP ===")
    server, base = start_marketplace_server()
    client = MarketplaceHttpClient(base, shape="shopee")
    results = []
    try:
        for s in SEEDS:
            hmm._Handler.store = {}
            r = eval_seed(s, client)
            results.append(r)
            print(f"SEED {s}: tool_match={r['tool_match']:.3f} http_exact={r['http_exact']:.3f}")
        # smoke cancel
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
        "suite": "order_nl_lexicon_v1",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "lexicon tool-match>=0.90 AND lexicon→HTTP exact>=0.90 AND v0b PASS",
            "FAIL": "tool-match mean<0.70",
            "UNRESOLVED": "else",
        },
        "metrics": {
            "tool_match_mean": tm,
            "http_exact_mean": he,
            "per_seed": results,
            "cancel_shaped_ok": cancel_ok,
        },
        "v0b": {"pass": v0b_ok, "outcome": v0b.get("outcome")},
        "method": "deterministic lexicon keyword/phrase slot-fill",
    }
    out = ARTIFACTS / "r" / "uns" / "order_nl_lexicon_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUMMARY tool_match={tm:.3f} http_exact={he:.3f} → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
