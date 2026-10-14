#!/usr/bin/env python3
"""order_conn_v3 — HTTP mock connector (sàn/OMS-like) + v0b gate.

Preregister:
  PASS: HTTP policy+CRUD smoke AND v0b PASS
  FAIL: HTTP policy or CRUD fail
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
from packages.order.connectors.http_mock import HttpOrderClient, start_server
from packages.order.connectors.file_store import apply_slot, empty_state

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "tools.json"
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
TOOLS = SCHEMA["tools"]
POLICIES = SCHEMA["policies"]


def policy_smoke(client: HttpOrderClient) -> dict:
    ex = Executor(TOOLS, POLICIES, apply_slot, empty_state())
    bad = Program(
        slots=[Slot("CREATE_ORDER", {"order_id": "H1", "address": "A"}), Slot("CANCEL_ORDER", {})],
        pack="order",
    )
    res_bad = ex.run(bad)
    blocked = (not res_bad.ok) and any("CONFIRM" in e for e in res_bad.errors)

    good = Program(
        slots=[
            Slot("CREATE_ORDER", {"order_id": "H1", "address": "A", "cod": 1}),
            Slot("CONFIRM", {}),
            Slot("CANCEL_ORDER", {}),
        ],
        pack="order",
    )
    res_good = ex.run(good)
    st = empty_state()
    for slot in good.slots:
        st = client.apply(slot.tool, slot.args, order_id=st.get("order_id") or "H1")
    remote = client.get("H1")
    ok_persist = (
        res_good.ok
        and int(res_good.state.get("cancelled", 0)) == 1
        and int(remote.get("cancelled", 0)) == 1
    )
    return {
        "blocked_without_confirm": blocked,
        "cancel_with_confirm_ok": ok_persist,
        "remote_cancelled": int(remote.get("cancelled", 0)),
        "pass": blocked and ok_persist,
    }


def crud_smoke(client: HttpOrderClient) -> dict:
    client.apply("CREATE_ORDER", {"order_id": "H200", "address": "DN", "cod": 1})
    client.apply("LOOKUP_ORDER", {"order_id": "H200"}, order_id="H200")
    client.apply("UPDATE_ADDRESS", {"address": "HN"}, order_id="H200")
    client.apply("SET_STATUS", {"status": 2, "cod": 0}, order_id="H200")
    remote = client.get("H200")
    ok = (
        remote.get("address") == "HN"
        and int(remote.get("status", -1)) == 2
        and int(remote.get("cod", -1)) == 0
        and int(remote.get("cancelled", 1)) == 0
        and int(remote.get("exists", 0)) == 1
    )
    return {"pass": ok, "remote": remote}


def v0b_gate() -> dict:
    metrics_path = ARTIFACTS / "r" / "uns" / "order_v0b" / "metrics.json"
    if not metrics_path.exists():
        return {"pass": False, "outcome": "MISSING", "mode": "prior_metrics"}
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)
    return {
        "mode": "prior_metrics",
        "outcome": m.get("outcome"),
        "metrics": m.get("metrics"),
        "pass": m.get("outcome") == "PASS",
    }


def main():
    print("=== order_conn_v3: HTTP mock connector + v0b gate ===")
    server, base = start_server()
    try:
        client = HttpOrderClient(base)
        print("base_url:", base)
        pol = policy_smoke(client)
        crud = crud_smoke(client)
        print("policy:", pol)
        print("crud:", crud)
    finally:
        server.shutdown()

    v0b = v0b_gate()
    print("v0b:", {k: v0b[k] for k in ("mode", "outcome", "pass")})

    if pol["pass"] and crud["pass"] and v0b["pass"]:
        outcome = "PASS"
    elif not (pol["pass"] and crud["pass"]):
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_conn_v3_http_mock",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "HTTP policy+CRUD AND v0b PASS",
            "FAIL": "HTTP policy or CRUD fail",
            "UNRESOLVED": "connector ok but v0b not PASS",
        },
        "policy": pol,
        "crud": crud,
        "v0b": v0b,
        "base_url_ephemeral": True,
    }
    out_dir = ARTIFACTS / "r" / "uns" / "order_conn_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"OUTCOME → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
