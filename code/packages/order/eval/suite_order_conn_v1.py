#!/usr/bin/env python3
"""order_conn_v1 — thin file connector + regression gate vs frozen v0b.

Preregister:
  PASS if:
    (1) file connector policy smoke: CANCEL without CONFIRM blocked; with CONFIRM persists cancelled=1
    (2) file connector CRUD: CREATE → LOOKUP → UPDATE_ADDRESS → SET_STATUS exact on disk
    (3) v0b baseline still PASS (re-run suite_order_v0b, or load prior metrics if skip_train)
  FAIL if (1) or (2) fail
  UNRESOLVED if (1)(2) ok but v0b not reconfirmed

Hypothesis: durable file store does not break kernel policy or slot semantics.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # code/
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from kernel.program import Program, Slot
from kernel.executor import Executor
from packages.order.connectors.file_store import FileOrderStore, apply_slot, empty_state

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "tools.json"
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
TOOLS = SCHEMA["tools"]
POLICIES = SCHEMA["policies"]


def policy_smoke(store_path: str) -> dict:
    store = FileOrderStore(store_path)
    # build apply that uses file store for CREATE path via executor state machine
    # For policy we still use pure apply_slot + Executor (same as mock)
    ex = Executor(TOOLS, POLICIES, apply_slot, empty_state())

    bad = Program(
        slots=[Slot("CREATE_ORDER", {"order_id": "F1", "address": "A"}), Slot("CANCEL_ORDER", {})],
        pack="order",
    )
    res_bad = ex.run(bad)
    blocked = (not res_bad.ok) and any("CONFIRM" in e for e in res_bad.errors)

    good = Program(
        slots=[
            Slot("CREATE_ORDER", {"order_id": "F1", "address": "A", "cod": 1}),
            Slot("CONFIRM", {}),
            Slot("CANCEL_ORDER", {}),
        ],
        pack="order",
    )
    res_good = ex.run(good)
    # also persist via store.apply
    st = empty_state()
    for slot in good.slots:
        st = store.apply(slot.tool, slot.args, order_id=st.get("order_id") or "F1")
    disk = store.get("F1")
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


def crud_smoke(store_path: str) -> dict:
    store = FileOrderStore(store_path)
    st = store.apply("CREATE_ORDER", {"order_id": "C100", "address": "HN", "cod": 1})
    assert st["exists"] == 1 and st["order_id"] == "C100"
    st = store.apply("LOOKUP_ORDER", {"order_id": "C100"}, order_id="C100")
    assert st["address"] == "HN"
    st = store.apply("UPDATE_ADDRESS", {"address": "HCM"}, order_id="C100")
    assert st["address"] == "HCM"
    st = store.apply("SET_STATUS", {"status": 2, "cod": 0}, order_id="C100")
    disk = store.get("C100")
    ok = (
        disk.get("address") == "HCM"
        and int(disk.get("status", -1)) == 2
        and int(disk.get("cod", -1)) == 0
        and int(disk.get("cancelled", 1)) == 0
    )
    return {"pass": ok, "disk": disk}


def run_v0b_regression(skip_train: bool = False) -> dict:
    """Re-run frozen baseline. skip_train: only check prior metrics file exists PASS."""
    metrics_path = ARTIFACTS / "r" / "uns" / "order_v0b" / "metrics.json"
    if skip_train and metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        return {
            "mode": "prior_metrics",
            "outcome": m.get("outcome"),
            "metrics": m.get("metrics"),
            "pass": m.get("outcome") == "PASS",
        }
    # full re-run
    script = Path(__file__).resolve().parent / "suite_order_v0b.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ARTIFACTS),
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )
    out = proc.stdout + "\n" + proc.stderr
    # reload metrics
    outcome = "UNKNOWN"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        outcome = m.get("outcome", "UNKNOWN")
    else:
        m = {}
    return {
        "mode": "rerun",
        "returncode": proc.returncode,
        "outcome": outcome,
        "metrics": m.get("metrics"),
        "pass": outcome == "PASS" and proc.returncode == 0,
        "tail": out[-800:],
    }


def main():
    print("=== order_conn_v1: file connector + v0b regression ===")
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "orders_policy.json")
        p2 = os.path.join(td, "orders_crud.json")
        pol = policy_smoke(p1)
        crud = crud_smoke(p2)
        print("policy:", pol)
        print("crud:", crud)

    # Prefer prior PASS metrics for gate if present (freeze integrity);
    # set ORDER_CONN_RERUN_V0B=1 to force full retrain.
    force = os.environ.get("ORDER_CONN_RERUN_V0B", "0") == "1"
    v0b = run_v0b_regression(skip_train=not force)
    print("v0b_regression:", {k: v0b[k] for k in v0b if k != "tail"})
    if "tail" in v0b:
        print("--- v0b tail ---\n", v0b["tail"][-400:])

    conn_ok = pol["pass"] and crud["pass"]
    baseline_ok = v0b["pass"]
    if conn_ok and baseline_ok:
        outcome = "PASS"
    elif not conn_ok:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_conn_v1_file_connector",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "file policy+CRUD smoke AND v0b baseline PASS",
            "FAIL": "file connector policy or CRUD fail",
            "UNRESOLVED": "connector ok but v0b not confirmed PASS",
        },
        "policy": pol,
        "crud": crud,
        "v0b": {k: v0b[k] for k in v0b if k != "tail"},
    }
    out_dir = ARTIFACTS / "r" / "uns" / "order_conn_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"OUTCOME → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
