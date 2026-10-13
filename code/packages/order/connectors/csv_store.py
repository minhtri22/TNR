"""CSV / Sheet-shaped order store — same slot semantics as mock/file.

Hypothesis (order_conn_v2): durable tabular rows (CSV) do not break
program→execute or CONFIRM policy; still passes conn gate + v0b freeze.
"""
from __future__ import annotations
import csv
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_store import apply_slot, empty_state  # shared pure transitions

FIELDNAMES = [
    "order_id",
    "exists",
    "address",
    "status",
    "cod",
    "cancelled",
    "confirmed",
]


def _row_to_state(row: Dict[str, str]) -> Dict[str, Any]:
    st = empty_state()
    st["order_id"] = str(row.get("order_id", ""))
    st["exists"] = int(row.get("exists", 0) or 0)
    st["address"] = str(row.get("address", ""))
    st["status"] = int(row.get("status", 0) or 0)
    st["cod"] = int(row.get("cod", 0) or 0)
    st["cancelled"] = int(row.get("cancelled", 0) or 0)
    st["confirmed"] = int(row.get("confirmed", 0) or 0)
    return st


def _state_to_row(st: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "order_id": str(st.get("order_id", "")),
        "exists": int(st.get("exists", 0)),
        "address": str(st.get("address", "")),
        "status": int(st.get("status", 0)),
        "cod": int(st.get("cod", 0)),
        "cancelled": int(st.get("cancelled", 0)),
        "confirmed": int(st.get("confirmed", 0)),
    }


class CsvOrderStore:
    """One row per order_id — Sheet-shaped, human-readable."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_rows([])

    def _read_rows(self) -> List[Dict[str, str]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDNAMES})
        os.replace(tmp, self.path)

    def get(self, order_id: str) -> Dict[str, Any]:
        with self._lock:
            for row in self._read_rows():
                if str(row.get("order_id", "")) == str(order_id):
                    return _row_to_state(row)
            return empty_state()

    def put(self, order_id: str, state: Dict[str, Any]) -> None:
        with self._lock:
            rows = self._read_rows()
            out: List[Dict[str, Any]] = []
            found = False
            row = _state_to_row(state)
            row["order_id"] = str(order_id)
            for r in rows:
                if str(r.get("order_id", "")) == str(order_id):
                    out.append(row)
                    found = True
                else:
                    out.append({k: r.get(k, "") for k in FIELDNAMES})
            if not found:
                out.append(row)
            self._write_rows(out)

    def apply(self, tool: str, args: Dict[str, Any], order_id: Optional[str] = None) -> Dict[str, Any]:
        oid = order_id or str(args.get("order_id", ""))
        with self._lock:
            rows = self._read_rows()
            by_id = {str(r.get("order_id", "")): r for r in rows if r.get("order_id")}
            if tool == "LOOKUP_ORDER" and oid and oid in by_id:
                st = _row_to_state(by_id[oid])
            elif oid and oid in by_id:
                st = _row_to_state(by_id[oid])
            else:
                st = empty_state()
            st = apply_slot(tool, args, st)
            key = str(st.get("order_id") or oid or "UNKNOWN")
            if st.get("exists"):
                by_id[key] = _state_to_row(st)
                by_id[key]["order_id"] = key
                self._write_rows(list(by_id.values()))
            return dict(st)
