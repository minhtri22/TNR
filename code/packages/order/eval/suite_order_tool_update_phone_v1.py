#!/usr/bin/env python3
"""order_tool_update_phone_v1 — new schema tool UPDATE_PHONE + lexicon + HTTP.

Hypothesis: UPDATE_PHONE behaves like UPDATE_ADDRESS (no CONFIRM required);
EN/VI lexicon maps phrases; HTTP exact on phone field; prior gates still green.

Preregister:
  PASS if all of:
    (1) kernel: CREATE→UPDATE_PHONE sets phone; cancelled blocks update
    (2) lexicon EN+VI tool_match on sequences containing UPDATE_PHONE >= 0.90
    (3) HTTP exact (phone in state) >= 0.90
    (4) v0b prior PASS
    (5) regression: CANCEL still requires CONFIRM
  FAIL if (1) or (5) fail or tool_match < 0.70
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

from kernel.program import Program, Slot
from kernel.executor import Executor
from packages.order.lexicon import compile_patterns, load_lexicon, parse_nl
from packages.order.connectors.file_store import apply_slot, empty_state
from packages.order.connectors.http_marketplace_mock import (
    MarketplaceHttpClient,
    start_marketplace_server,
)
from packages.order.connectors import http_marketplace_mock as hmm

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "tools.json").read_text()
)
TOOLS = SCHEMA["tools"]
POLICIES = SCHEMA["policies"]
assert "UPDATE_PHONE" in TOOLS

SEEDS = [42, 7, 123]
N = 60
L = 2


def kernel_smoke() -> dict:
    ex = Executor(TOOLS, POLICIES, apply_slot, empty_state())
    # phone update
    prog = Program(
        slots=[
            Slot("CREATE_ORDER", {"order_id": "P1", "address": "A", "phone": "0900"}),
            Slot("UPDATE_PHONE", {"phone": "0911"}),
        ],
        pack="order",
    )
    r = ex.run(prog)
    phone_ok = r.ok and r.state.get("phone") == "0911"
    # cancel still needs confirm
    bad = Program(
        slots=[Slot("CREATE_ORDER", {"order_id": "P2"}), Slot("CANCEL_ORDER", {})],
        pack="order",
    )
    rb = ex.run(bad)
    cancel_blocked = not rb.ok
    # cancelled blocks phone update
    st = empty_state()
    st = apply_slot("CREATE_ORDER", {"order_id": "P3", "address": "A"}, st)
    st = apply_slot("CONFIRM", {}, st)
    st = apply_slot("CANCEL_ORDER", {}, st)
    st2 = apply_slot("UPDATE_PHONE", {"phone": "0999"}, st)
    blocked_after_cancel = st2.get("phone", "") != "0999"
    return {
        "phone_update_ok": phone_ok,
        "cancel_still_blocked": cancel_blocked,
        "no_phone_after_cancel": blocked_after_cancel,
        "pass": phone_ok and cancel_blocked and blocked_after_cancel,
    }


def lexicon_http(lang: str, client: MarketplaceHttpClient) -> dict:
    lex = load_lexicon(lang, "v1")
    compiled = compile_patterns(lex)
    phrases = {t: list(lex["tools"][t]["patterns"]) for t in lex["tools"]}
    seqs = [
        ["CREATE_ORDER", "UPDATE_PHONE"],
        ["LOOKUP_ORDER", "UPDATE_PHONE"],
        ["UPDATE_PHONE", "UPDATE_ADDRESS"],
        ["UPDATE_ADDRESS", "UPDATE_PHONE"],
        ["CREATE_ORDER", "UPDATE_ADDRESS"],
    ]
    match = http_hit = 0
    rng = np.random.RandomState(0 if lang == "en" else 1)
    for i in range(N):
        gold = list(seqs[rng.randint(0, len(seqs))])
        while len(gold) < L:
            gold.append("NOP")
        gold = gold[:L]
        sep = " then " if lang == "en" else " rồi "
        text = sep.join(phrases[t][int(rng.randint(0, len(phrases[t])))] for t in gold)
        if lang == "vi":
            text = text.replace(" rồi ", " then ")
        pred = parse_nl(text, compiled, max_slots=L)
        if pred == gold:
            match += 1
        oid = f"PH-{lang}-{i}"
        hmm._Handler.store.pop(oid, None)
        # gold state
        gst = empty_state()
        for t in gold:
            args = {}
            if t == "CREATE_ORDER":
                args = {"order_id": oid, "address": "A", "phone": "0900", "cod": 0}
            elif t == "UPDATE_PHONE":
                args = {"phone": "0911"}
            elif t == "UPDATE_ADDRESS":
                args = {"address": "B"}
            elif t == "LOOKUP_ORDER":
                args = {"order_id": oid}
            gst = apply_slot(t, args, gst)
        pst = empty_state()
        for t in pred:
            args = {}
            if t == "CREATE_ORDER":
                args = {"order_id": oid, "address": "A", "phone": "0900", "cod": 0}
            elif t == "UPDATE_PHONE":
                args = {"phone": "0911"}
            elif t == "UPDATE_ADDRESS":
                args = {"address": "B"}
            elif t == "LOOKUP_ORDER":
                args = {"order_id": oid}
            try:
                pst = client.apply(t, args, order_id=pst.get("order_id") or oid)
            except RuntimeError:
                break
        keys = ["exists", "status", "cod", "cancelled", "confirmed", "phone"]
        if all(str(pst.get(k, "")) == str(gst.get(k, "")) or int(pst.get(k, 0) or 0) == int(gst.get(k, 0) or 0) for k in keys):
            # careful: phone string compare
            if str(pst.get("phone", "")) == str(gst.get("phone", "")) and int(pst.get("exists", 0)) == int(gst.get("exists", 0)):
                if int(pst.get("cancelled", 0)) == int(gst.get("cancelled", 0)):
                    http_hit += 1
    return {
        "lang": lang,
        "tool_match": match / N,
        "http_exact": http_hit / N,
    }


def main():
    print("=== order_tool_update_phone_v1 ===")
    ks = kernel_smoke()
    print("kernel:", ks)
    server, base = start_marketplace_server()
    try:
        client = MarketplaceHttpClient(base, shape="shopee")
        en = lexicon_http("en", client)
        hmm._Handler.store = {}
        vi = lexicon_http("vi", client)
        print("en:", en)
        print("vi:", vi)
    finally:
        server.shutdown()

    v0b_path = ARTIFACTS / "r" / "uns" / "order_v0b" / "metrics.json"
    v0b = json.loads(v0b_path.read_text()) if v0b_path.exists() else {}
    v0b_ok = v0b.get("outcome") == "PASS"

    tm = (en["tool_match"] + vi["tool_match"]) / 2
    he = (en["http_exact"] + vi["http_exact"]) / 2
    if ks["pass"] and tm >= 0.90 and he >= 0.90 and v0b_ok:
        outcome = "PASS"
    elif not ks["pass"] or tm < 0.70:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_tool_update_phone_v1",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "kernel phone+cancel policy AND lexicon EN/VI match>=0.90 AND HTTP>=0.90 AND v0b",
            "FAIL": "kernel fail OR match<0.70",
            "UNRESOLVED": "else",
        },
        "kernel": ks,
        "en": en,
        "vi": vi,
        "v0b": {"pass": v0b_ok},
    }
    out = ARTIFACTS / "r" / "uns" / "order_tool_update_phone_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTCOME → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
