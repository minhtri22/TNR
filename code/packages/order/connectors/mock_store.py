"""Mock order store — in-memory side-effects for Order pack v0."""
from __future__ import annotations
from typing import Any, Dict


def empty_state() -> Dict[str, Any]:
    return {
        "exists": 0,
        "order_id": "",
        "address": "",
        "status": 0,  # 0 new, 1 confirmed, 2 shipped, 3 done
        "cod": 0,
        "cancelled": 0,
        "confirmed": 0,  # CONFIRM slot seen for destructive ops
    }


def apply_slot(tool: str, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
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
        # read-only; may set order_id from args if provided for tracking
        if "order_id" in args and args["order_id"]:
            st["order_id"] = str(args["order_id"])
            if not st.get("exists"):
                # mock: looking up unknown still marks id for downstream
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
        # policy layer already requires CONFIRM in program;
        # runtime also refuses without confirmed flag
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
