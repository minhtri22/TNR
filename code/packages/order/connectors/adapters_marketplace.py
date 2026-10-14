"""Map internal order state ↔ Shopee / Sapo-shaped payloads (mock only).

Hypothesis (order_conn_v4): external API field names do not change kernel
slot semantics; adapters are pure and reversible enough for exact checks.
"""
from __future__ import annotations
from typing import Any, Dict

from .file_store import empty_state

# Internal status: 0 new, 1 confirmed, 2 shipped, 3 done
# Shopee-like order_status strings (simplified subset)
_SHOPEE_STATUS = {
    0: "UNPAID",
    1: "READY_TO_SHIP",
    2: "SHIPPED",
    3: "COMPLETED",
}
_SHOPEE_STATUS_INV = {v: k for k, v in _SHOPEE_STATUS.items()}
_SHOPEE_STATUS_INV["CANCELLED"] = -1  # signal cancel

# Sapo-like
_SAPO_STATUS = {
    0: "pending",
    1: "confirmed",
    2: "delivering",
    3: "completed",
}
_SAPO_STATUS_INV = {v: k for k, v in _SAPO_STATUS.items()}
_SAPO_STATUS_INV["cancelled"] = -1


def to_shopee(st: Dict[str, Any]) -> Dict[str, Any]:
    """Internal → Shopee-like order object."""
    if not st.get("exists"):
        return {"error": "order_not_found"}
    status = "CANCELLED" if st.get("cancelled") else _SHOPEE_STATUS.get(int(st.get("status", 0)), "UNPAID")
    return {
        "order_sn": str(st.get("order_id", "")),
        "order_status": status,
        "recipient_address": {
            "full_address": str(st.get("address", "")),
        },
        "cod": bool(int(st.get("cod", 0))),
        "cod_amount": 10000 if int(st.get("cod", 0)) else 0,
        # lab-only echo (not real Shopee)
        "_lab_confirmed": int(st.get("confirmed", 0)),
    }


def from_shopee(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Shopee-like → internal state."""
    if payload.get("error") == "order_not_found":
        return empty_state()
    st = empty_state()
    st["exists"] = 1
    st["order_id"] = str(payload.get("order_sn", ""))
    addr = payload.get("recipient_address") or {}
    st["address"] = str(addr.get("full_address", ""))
    ost = str(payload.get("order_status", "UNPAID"))
    if ost == "CANCELLED":
        st["cancelled"] = 1
        st["status"] = 0
    else:
        st["cancelled"] = 0
        st["status"] = int(_SHOPEE_STATUS_INV.get(ost, 0))
    st["cod"] = 1 if payload.get("cod") or int(payload.get("cod_amount", 0) or 0) > 0 else 0
    st["confirmed"] = int(payload.get("_lab_confirmed", 0))
    return st


def to_sapo(st: Dict[str, Any]) -> Dict[str, Any]:
    """Internal → Sapo-like order object."""
    if not st.get("exists"):
        return {"error": "not_found"}
    status = "cancelled" if st.get("cancelled") else _SAPO_STATUS.get(int(st.get("status", 0)), "pending")
    return {
        "id": abs(hash(str(st.get("order_id", "")))) % 10_000_000,
        "code": str(st.get("order_id", "")),
        "status": status,
        "shipping_address": str(st.get("address", "")),
        "cod_amount": 10000.0 if int(st.get("cod", 0)) else 0.0,
        "_lab_confirmed": int(st.get("confirmed", 0)),
    }


def from_sapo(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sapo-like → internal state."""
    if payload.get("error") == "not_found":
        return empty_state()
    st = empty_state()
    st["exists"] = 1
    st["order_id"] = str(payload.get("code", ""))
    st["address"] = str(payload.get("shipping_address", ""))
    ost = str(payload.get("status", "pending"))
    if ost == "cancelled":
        st["cancelled"] = 1
        st["status"] = 0
    else:
        st["cancelled"] = 0
        st["status"] = int(_SAPO_STATUS_INV.get(ost, 0))
    st["cod"] = 1 if float(payload.get("cod_amount", 0) or 0) > 0 else 0
    st["confirmed"] = int(payload.get("_lab_confirmed", 0))
    return st


def roundtrip_ok(st: Dict[str, Any], kind: str) -> bool:
    """Exact on core keys after to→from."""
    keys = ["exists", "order_id", "address", "status", "cod", "cancelled", "confirmed"]
    if kind == "shopee":
        back = from_shopee(to_shopee(st))
    elif kind == "sapo":
        back = from_sapo(to_sapo(st))
    else:
        raise ValueError(kind)
    return all(back.get(k) == st.get(k) for k in keys)
