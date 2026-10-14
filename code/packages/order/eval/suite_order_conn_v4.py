#!/usr/bin/env python3
"""order_conn_v4 — Shopee/Sapo-shaped HTTP adapters + v0b gate.

Preregister:
  PASS: adapter roundtrip exact AND HTTP policy+CRUD (both shapes) AND v0b PASS
  FAIL: any adapter/HTTP smoke fail
  UNRESOLVED: connector ok but v0b not PASS
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from kernel.program import Program, Slot
from kernel.executor import Executor
from packages.order.connectors.adapters_marketplace import (
    from_sapo,
    from_shopee,
    roundtrip_ok,
    to_sapo,
    to_shopee,
)
from packages.order.connectors.file_store import apply_slot, empty_state
from packages.order.connectors.http_marketplace_mock import (
    MarketplaceHttpClient,
    start_marketplace_server,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "tools.json"
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
TOOLS = SCHEMA["tools"]
POLICIES = SCHEMA["policies"]


def adapter_unit() -> dict:
    samples = []
    # build states via apply_slot
    st = empty_state()
    st = apply_slot("CREATE_ORDER", {"order_id": "ORD-S", "address": "Q1-HCM", "cod": 1}, st)
    samples.append(st)
    st2 = apply_slot("CONFIRM", {}, dict(st))
    st2 = apply_slot("CANCEL_ORDER", {}, st2)
    samples.append(st2)
    st3 = apply_slot("CREATE_ORDER", {"order_id": "ORD-T", "address": "HN", "cod": 0}, empty_state())
    st3 = apply_slot("SET_STATUS", {"status": 2}, st3)
    samples.append(st3)

    ok_s = all(roundtrip_ok(s, "shopee") for s in samples)
    ok_p = all(roundtrip_ok(s, "sapo") for s in samples)
    # shape field presence
    sh = to_shopee(samples[0])
    sa = to_sapo(samples[0])
    shape_ok = (
        "order_sn" in sh
        and "order_status" in sh
        and "recipient_address" in sh
        and "code" in sa
        and "shipping_address" in sa
        and "status" in sa
    )
    return {
        "shopee_roundtrip": ok_s,
        "sapo_roundtrip": ok_p,
        "shape_fields": shape_ok,
        "shopee_sample": sh,
        "sapo_sample": sa,
        "pass": ok_s and ok_p and shape_ok,
    }


def http_shape_smoke(shape: str) -> dict:
    server, base = start_marketplace_server()
    try:
        client = MarketplaceHttpClient(base, shape=shape)
        ex = Executor(TOOLS, POLICIES, apply_slot, empty_state())
        bad = Program(
            slots=[Slot("CREATE_ORDER", {"order_id": f"{shape}-1", "address": "A"}), Slot("CANCEL_ORDER", {})],
            pack="order",
        )
        blocked = (not ex.run(bad).ok)

        st = empty_state()
        for slot in [
            Slot("CREATE_ORDER", {"order_id": f"{shape}-1", "address": "A", "cod": 1}),
            Slot("CONFIRM", {}),
            Slot("CANCEL_ORDER", {}),
        ]:
            st = client.apply(slot.tool, slot.args, order_id=st.get("order_id") or f"{shape}-1")
        remote = client.get_internal(f"{shape}-1")
        shaped = client.get_shaped(f"{shape}-1")
        ok = int(remote.get("cancelled", 0)) == 1 and blocked
        return {
            "shape": shape,
            "blocked_without_confirm": blocked,
            "remote_cancelled": int(remote.get("cancelled", 0)),
            "shaped_status": shaped.get("order_status") or shaped.get("status"),
            "pass": ok,
            "shaped": shaped,
        }
    finally:
        server.shutdown()


def crud_both() -> dict:
    server, base = start_marketplace_server()
    try:
        results = {}
        for shape in ("shopee", "sapo"):
            c = MarketplaceHttpClient(base, shape=shape)
            oid = f"C-{shape}"
            c.apply("CREATE_ORDER", {"order_id": oid, "address": "DN", "cod": 1})
            c.apply("UPDATE_ADDRESS", {"address": "HN"}, order_id=oid)
            c.apply("SET_STATUS", {"status": 2, "cod": 0}, order_id=oid)
            st = c.get_internal(oid)
            results[shape] = {
                "address": st.get("address"),
                "status": st.get("status"),
                "cod": st.get("cod"),
                "ok": st.get("address") == "HN" and int(st.get("status", -1)) == 2 and int(st.get("cod", -1)) == 0,
            }
        return {"pass": all(v["ok"] for v in results.values()), "detail": results}
    finally:
        server.shutdown()


def v0b_gate() -> dict:
    p = ARTIFACTS / "r" / "uns" / "order_v0b" / "metrics.json"
    if not p.exists():
        return {"pass": False, "outcome": "MISSING"}
    m = json.loads(p.read_text(encoding="utf-8"))
    return {"pass": m.get("outcome") == "PASS", "outcome": m.get("outcome"), "metrics": m.get("metrics")}


def main():
    print("=== order_conn_v4: Shopee/Sapo adapter + HTTP + v0b ===")
    ad = adapter_unit()
    print("adapter:", {k: ad[k] for k in ad if k not in ("shopee_sample", "sapo_sample")})
    print("  shopee sample:", ad["shopee_sample"])
    print("  sapo sample:", ad["sapo_sample"])
    hs = http_shape_smoke("shopee")
    hp = http_shape_smoke("sapo")
    print("http_shopee:", {k: hs[k] for k in hs if k != "shaped"})
    print("http_sapo:", {k: hp[k] for k in hp if k != "shaped"})
    crud = crud_both()
    print("crud:", crud)
    v0b = v0b_gate()
    print("v0b:", {k: v0b[k] for k in ("pass", "outcome")})

    ok = ad["pass"] and hs["pass"] and hp["pass"] and crud["pass"]
    if ok and v0b["pass"]:
        outcome = "PASS"
    elif not ok:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_conn_v4_marketplace_adapter",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "adapter roundtrip + HTTP both shapes + CRUD + v0b PASS",
            "FAIL": "adapter or HTTP smoke fail",
            "UNRESOLVED": "connector ok but v0b not PASS",
        },
        "adapter": {k: ad[k] for k in ad if k not in ("shopee_sample", "sapo_sample")},
        "http_shopee": {k: hs[k] for k in hs if k != "shaped"},
        "http_sapo": {k: hp[k] for k in hp if k != "shaped"},
        "crud": crud,
        "v0b": {k: v0b[k] for k in ("pass", "outcome")},
    }
    out = ARTIFACTS / "r" / "uns" / "order_conn_v4"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTCOME → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
