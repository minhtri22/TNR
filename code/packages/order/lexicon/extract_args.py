"""Extract order_id and phone from NL (controlled regex — not open NER)."""
from __future__ import annotations
import re
from typing import Any, Dict, Optional

_ORDER_ID_PATTERNS = [
    re.compile(r"\b(ORD-?[A-Za-z0-9][A-Za-z0-9_-]{1,})\b", re.I),
    re.compile(r"\b(SPX[A-Za-z0-9_-]{3,})\b", re.I),
    re.compile(r"\b(DH[A-Za-z0-9_-]{3,})\b", re.I),
    re.compile(r"#\s*([A-Za-z0-9][A-Za-z0-9_-]{2,})\b"),
    re.compile(
        r"(?:order(?:\s*id)?|mã\s*đơn|ma\s*don)\s*[:=#]?\s*#?\s*"
        r"([A-Za-z]*\d[A-Za-z0-9_-]{1,}|\d[A-Za-z0-9_-]{2,}|"
        r"ORD-?[A-Za-z0-9_-]+|SPX[A-Za-z0-9_-]+|DH[A-Za-z0-9_-]+)",
        re.I,
    ),
    re.compile(
        r"(?:đơn|don)\s+(?:mã\s*)?#?\s*"
        r"(ORD-?[A-Za-z0-9_-]+|SPX[A-Za-z0-9_-]+|DH[A-Za-z0-9_-]+|[A-Za-z]*\d[A-Za-z0-9_-]+)",
        re.I,
    ),
]

_PHONE_PATTERNS = [
    re.compile(r"(?:\+?84|0)(?:3|5|7|8|9)\d{8}\b"),
    re.compile(r"\b0\d{9,10}\b"),
    re.compile(r"\+\d{10,14}\b"),
    re.compile(
        r"(?:phone|sđt|sdt|số(?:\s*điện\s*thoại)?|so\s*dien\s*thoai|mobile|tel)\s*[:=]?\s*"
        r"(\+?\d[\d\s-]{8,14}\d)",
        re.I,
    ),
]

_STOP = {"order", "orders", "please", "status", "phone", "number", "don", "đơn"}


def _norm_phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw)
    if digits.startswith("+84"):
        digits = "0" + digits[3:]
    elif digits.startswith("84") and len(digits) >= 11 and not digits.startswith("0"):
        digits = "0" + digits[2:]
    return digits


def _valid_order_id(tok: str) -> bool:
    t = tok.strip()
    if not t or t.lower() in _STOP or len(t) < 3:
        return False
    if re.match(r"^(ORD|SPX|DH)", t, re.I):
        return True
    return bool(re.search(r"\d", t))


def extract_order_id(text: str) -> Optional[str]:
    for cre in _ORDER_ID_PATTERNS:
        m = cre.search(text)
        if m and _valid_order_id(m.group(1)):
            return m.group(1)
    return None


def extract_phone(text: str) -> Optional[str]:
    for cre in _PHONE_PATTERNS:
        m = cre.search(text)
        if not m:
            continue
        raw = m.group(1) if m.lastindex else m.group(0)
        ph = _norm_phone(raw)
        if len(re.sub(r"\D", "", ph)) >= 9:
            return ph
    return None


def extract_args(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    oid = extract_order_id(text)
    if oid:
        out["order_id"] = oid
    phone = extract_phone(text)
    if phone:
        out["phone"] = phone
    return out


def merge_args_for_tools(tools: list, text: str) -> list:
    args = extract_args(text)
    slots = []
    for t in tools:
        a: Dict[str, Any] = {}
        if t in ("LOOKUP_ORDER", "CREATE_ORDER") and "order_id" in args:
            a["order_id"] = args["order_id"]
        if t in ("UPDATE_PHONE", "CREATE_ORDER") and "phone" in args:
            a["phone"] = args["phone"]
        slots.append({"tool": t, "args": a})
    return slots
