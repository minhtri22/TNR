"""Thin file connector — same slot semantics as mock_store, persists to JSON.

Hypothesis (order_conn_v1): program→execute + CONFIRM policy work unchanged
when side-effects go through a durable file-backed store instead of RAM-only.
"""
from __future__ import annotations
import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

# Reuse identical empty state shape as mock (regression compatibility)
def empty_state() -> Dict[str, Any]:
    return {
        "exists": 0,
        "order_id": "",
        "address": "",
        "status": 0,
        "cod": 0,
        "cancelled": 0,
        "confirmed": 0,
    }


def apply_slot(tool: str, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Pure state transition — identical rules to mock_store."""
    st = dict(state)

    if tool == "NOP":
        return st

    if tool == "CREATE_ORDER":
        oid = str(args.get("order_id", "ORD-NEW"))
        addr = str(args.get("address", "addr-default"))
        st.update(
            {
                "exists": 1,
                "order_id": oid,
                "address": addr,
                "status": 0,
                "cod": int(args.get("cod", 0)),
                "cancelled": 0,
                "confirmed": 0,
            }
        )
        return st

    if tool == "LOOKUP_ORDER":
        if "order_id" in args and args["order_id"]:
            st["order_id"] = str(args["order_id"])
            if not st.get("exists"):
                st["exists"] = 1
                st["address"] = st.get("address") or "addr-lookup"
        return st

    if tool == "UPDATE_ADDRESS":
        if st.get("exists") and not st.get("cancelled"):
            st["address"] = str(args.get("address", st.get("address", "")))
        return st

    if tool == "CONFIRM":
        st["confirmed"] = 1
        return st

    if tool == "CANCEL_ORDER":
        if not st.get("confirmed"):
            raise RuntimeError("CANCEL_ORDER requires CONFIRM")
        if st.get("exists") and not st.get("cancelled"):
            st["cancelled"] = 1
            st["status"] = 0
        return st

    if tool == "SET_STATUS":
        if st.get("exists") and not st.get("cancelled"):
            st["status"] = int(args.get("status", st.get("status", 0)))
            if "cod" in args:
                st["cod"] = int(args["cod"])
        return st

    raise ValueError(f"unknown tool: {tool}")


class FileOrderStore:
    """Multi-order JSON file. Thread-safe thin persistence layer."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"orders": {}})

    def _read(self) -> Dict[str, Any]:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def get(self, order_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            st = data.get("orders", {}).get(order_id)
            if st is None:
                return empty_state()
            return dict(st)

    def put(self, order_id: str, state: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data.setdefault("orders", {})[order_id] = dict(state)
            self._write(data)

    def apply(self, tool: str, args: Dict[str, Any], order_id: Optional[str] = None) -> Dict[str, Any]:
        """Load → apply_slot → persist. Returns new state."""
        oid = order_id or str(args.get("order_id", ""))
        with self._lock:
            data = self._read()
            orders = data.setdefault("orders", {})
            st = dict(orders.get(oid, empty_state())) if oid else empty_state()
            if tool == "LOOKUP_ORDER" and oid and oid in orders:
                st = dict(orders[oid])
            st = apply_slot(tool, args, st)
            # persist under order_id from state
            key = str(st.get("order_id") or oid or "UNKNOWN")
            if st.get("exists"):
                orders[key] = dict(st)
                self._write(data)
            return dict(st)
