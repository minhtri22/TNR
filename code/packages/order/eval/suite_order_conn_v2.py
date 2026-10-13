#!/usr/bin/env python3
"""order_conn_v2 — CSV/Sheet-shaped connector + same gate as conn_v1 + v0b.

Preregister (unchanged bar):
  PASS: CSV policy+CRUD smoke AND v0b baseline PASS
  FAIL: CSV policy or CRUD fail
  UNRESOLVED: connector ok but v0b not PASS
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from kernel.program import Program, Slot
from kernel.executor import Executor
from packages.order.connectors.csv_store import CsvOrderStore
from packages.order.connectors.file_store import apply_slot, empty_state

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "tools.json"
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
TOOLS = SCHEMA["tools"]
POLICIES = SCHEMA["policies"]


def policy_smoke(path: str) -> dict:
    store = CsvOrderStore(path)
    ex = Executor(TOOLS, POLICIES, apply_slot, empty_state())
    bad = Program(
        slots=[Slot("CREATE_ORDER", {"order_id": "S1", "address": "A"}), Slot("CANCEL_ORDER", {})],
        pack="order",
    )
    res_bad = ex.run(bad)
    blocked = (not res_bad.ok) and any("CONFIRM" in e for e in res_bad.errors)

    good = Program(
        slots=[
            Slot("CREATE_ORDER", {"order_id": "S1", "address": "A", "cod": 1}),
            Slot("CONFIRM", {}),
            Slot("CANCEL_ORDER", {}),
        ],
        pack="order",
    )
    res_good = ex.run(good)
    st = empty_state()
    for slot in good.slots:
        st = store.apply(slot.tool, slot.args, order_id=st.get("order_id") or "S1")
    disk = store.get("S1")
    ok_persist = (
        res_good.ok
        and int(res_good.state.get("cancelled", 0)) == 1
        and int(disk.get("cancelled", 0)) == 1
    )
    return {
        "blocked_without_confirm": blocked,
        "cancel_with_confirm_ok": ok_persist,
        "disk_cancelled": int(disk.get("cancelled", 0)),
        "pass": blocked and ok_persist,
    }


def crud_smoke(path: str) -> dict:
    store = CsvOrderStore(path)
    store.apply("CREATE_ORDER", {"order_id": "C200", "address": "DN", "cod": 1})
    store.apply("LOOKUP_ORDER", {"order_id": "C200"}, order_id="C200")
    store.apply("UPDATE_ADDRESS", {"address": "HN"}, order_id="C200")
    store.apply("SET_STATUS", {"status": 2, "cod": 0}, order_id="C200")
    disk = store.get("C200")
    ok = (
        disk.get("address") == "HN"
        and int(disk.get("status", -1)) == 2
        and int(disk.get("cod", -1)) == 0
        and int(disk.get("cancelled", 1)) == 0
        and int(disk.get("exists", 0)) == 1
    )
    # file is real CSV with header
    text = Path(path).read_text(encoding="utf-8")
    has_header = text.splitlines()[0].startswith("order_id")
    return {"pass": ok and has_header, "disk": disk, "csv_header_ok": has_header}


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
    print("=== order_conn_v2: CSV/Sheet connector + v0b gate ===")
    with tempfile.TemporaryDirectory() as td:
        pol = policy_smoke(os.path.join(td, "orders_policy.csv"))
        crud = crud_smoke(os.path.join(td, "orders_crud.csv"))
        print("policy:", pol)
        print("crud:", crud)
    v0b = v0b_gate()
    print("v0b:", {k: v0b[k] for k in ("mode", "outcome", "pass")})

    if pol["pass"] and crud["pass"] and v0b["pass"]:
        outcome = "PASS"
    elif not (pol["pass"] and crud["pass"]):
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_conn_v2_csv_sheet",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "CSV policy+CRUD AND v0b PASS",
            "FAIL": "CSV policy or CRUD fail",
            "UNRESOLVED": "connector ok but v0b not PASS",
        },
        "policy": pol,
        "crud": crud,
        "v0b": v0b,
    }
    out_dir = ARTIFACTS / "r" / "uns" / "order_conn_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"OUTCOME → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
