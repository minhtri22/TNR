#!/usr/bin/env python3
"""order_nl_args_v1 — extract order_id + phone from NL (regex).

Hypothesis: controlled patterns recover ids/phones on labeled EN/VI phrases
without changing the 8 tools or lexicon PASS bars.

Preregister:
  PASS if:
    - order_id accuracy >= 0.90 on labeled set
    - phone accuracy >= 0.90 on labeled set
    - merge LOOKUP/UPDATE_PHONE with extract → HTTP state matches gold args
      on >= 0.90 of cases
    - v0b prior PASS
    - en_v1 + vi_v1 lexicon regression still PASS (prior metrics or quick re-run)
  FAIL if order_id or phone accuracy < 0.70
  else UNRESOLVED
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent
sys.path.insert(0, str(ROOT))

from packages.order.lexicon.extract_args import extract_args, extract_order_id, extract_phone
from packages.order.connectors.file_store import apply_slot, empty_state
from packages.order.connectors.http_marketplace_mock import (
    MarketplaceHttpClient,
    start_marketplace_server,
)
from packages.order.connectors import http_marketplace_mock as hmm

# Labeled cases: text → expected {order_id?, phone?}
LABELED = [
    # EN order_id
    {"text": "lookup order ORD-1001", "order_id": "ORD-1001", "phone": None},
    {"text": "check order SPX99881 please", "order_id": "SPX99881", "phone": None},
    {"text": "cancel order #A12B3", "order_id": "A12B3", "phone": None},
    {"text": "order id: DH7788 status", "order_id": "DH7788", "phone": None},
    {"text": "find the order ORD-XYZ-9", "order_id": "ORD-XYZ-9", "phone": None},
    # EN phone
    {"text": "update phone 0912345678", "order_id": None, "phone": "0912345678"},
    {"text": "change phone number +84912345678", "order_id": None, "phone": "0912345678"},
    {"text": "new phone 0987654321 for delivery", "order_id": None, "phone": "0987654321"},
    {"text": "phone: 0901122334", "order_id": None, "phone": "0901122334"},
    # EN both
    {"text": "update phone 0911222333 for order ORD-55", "order_id": "ORD-55", "phone": "0911222333"},
    {"text": "order SPX1 phone 0888777666", "order_id": "SPX1", "phone": "0888777666"},
    # VI order_id
    {"text": "tra đơn ORD-2002", "order_id": "ORD-2002", "phone": None},
    {"text": "kiểm tra đơn SPX11223", "order_id": "SPX11223", "phone": None},
    {"text": "hủy đơn mã đơn DH9001", "order_id": "DH9001", "phone": None},
    {"text": "xem đơn #B99", "order_id": "B99", "phone": None},
    # VI phone
    {"text": "đổi số điện thoại 0912345678", "order_id": None, "phone": "0912345678"},
    {"text": "sửa sđt 0909988776", "order_id": None, "phone": "0909988776"},
    {"text": "cập nhật số 0888111222", "order_id": None, "phone": "0888111222"},
    {"text": "đổi sđt +84 912 345 678", "order_id": None, "phone": "0912345678"},
    # VI both
    {"text": "đổi sđt 0912000111 cho đơn ORD-77", "order_id": "ORD-77", "phone": "0912000111"},
    {"text": "tra đơn SPX9 số 0977111222", "order_id": "SPX9", "phone": "0977111222"},
    # negatives (must not hallucinate)
    {"text": "lookup order please", "order_id": None, "phone": None},
    {"text": "đổi địa chỉ giao hàng", "order_id": None, "phone": None},
    {"text": "xác nhận hủy", "order_id": None, "phone": None},
]


def score_extract(cases):
    oid_ok = oid_n = 0
    ph_ok = ph_n = 0
    for c in cases:
        text = c["text"]
        got = extract_args(text)
        if c["order_id"] is not None:
            oid_n += 1
            if got.get("order_id") == c["order_id"]:
                oid_ok += 1
        else:
            # true negative: must not invent order_id
            oid_n += 1
            if "order_id" not in got:
                oid_ok += 1
        if c["phone"] is not None:
            ph_n += 1
            if got.get("phone") == c["phone"]:
                ph_ok += 1
        else:
            ph_n += 1
            if "phone" not in got:
                ph_ok += 1
    return {
        "order_id_acc": oid_ok / max(oid_n, 1),
        "phone_acc": ph_ok / max(ph_n, 1),
        "n": len(cases),
        "oid_ok": oid_ok,
        "ph_ok": ph_ok,
    }


def http_args_path(client) -> dict:
    """CREATE + LOOKUP/UPDATE_PHONE with extracted args → state match."""
    cases = [
        ("create and set phone", "create order ORD-H1 phone 0911000222", "ORD-H1", "0911000222"),
        ("vi phone order", "tạo đơn ORD-H2 số 0922000333", "ORD-H2", "0922000333"),
        ("update phone only", "update phone 0933000444 for order ORD-H3", "ORD-H3", "0933000444"),
        ("vi update", "đổi sđt 0944000555 cho đơn ORD-H4", "ORD-H4", "0944000555"),
    ]
    hit = 0
    details = []
    for name, text, oid, phone in cases:
        hmm._Handler.store.pop(oid, None)
        args = extract_args(text)
        # ensure create exists then update phone
        st = client.apply(
            "CREATE_ORDER",
            {"order_id": oid, "address": "A", "phone": args.get("phone", "000")},
            order_id=oid,
        )
        if "phone" in args:
            st = client.apply("UPDATE_PHONE", {"phone": args["phone"]}, order_id=oid)
        st = client.apply("LOOKUP_ORDER", {"order_id": oid}, order_id=oid)
        ok = st.get("order_id") == oid and str(st.get("phone", "")) == phone
        if ok:
            hit += 1
        details.append({"name": name, "ok": ok, "got": {"order_id": st.get("order_id"), "phone": st.get("phone")}, "args": args})
    return {"http_args_exact": hit / len(cases), "detail": details}


def prior_pass(name: str) -> bool:
    p = ARTIFACTS / "r" / "uns" / name / "metrics.json"
    if not p.exists():
        return False
    return json.loads(p.read_text()).get("outcome") == "PASS"


def main():
    print("=== order_nl_args_v1: extract order_id + phone ===")
    sc = score_extract(LABELED)
    print("extract:", sc)

    server, base = start_marketplace_server()
    try:
        client = MarketplaceHttpClient(base, shape="shopee")
        http = http_args_path(client)
        print("http:", {k: http[k] for k in http if k != "detail"})
        for d in http["detail"]:
            print(" ", d)
    finally:
        server.shutdown()

    v0b = prior_pass("order_v0b")
    en = prior_pass("order_nl_lexicon_en_v1")
    vi = prior_pass("order_nl_lexicon_vi_v1")
    print(f"regression priors: v0b={v0b} en_lex={en} vi_lex={vi}")

    oid_a, ph_a, http_a = sc["order_id_acc"], sc["phone_acc"], http["http_args_exact"]
    if oid_a >= 0.90 and ph_a >= 0.90 and http_a >= 0.90 and v0b and en and vi:
        outcome = "PASS"
    elif oid_a < 0.70 or ph_a < 0.70:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    card = {
        "suite": "order_nl_args_v1",
        "pack": "order",
        "outcome": outcome,
        "preregister": {
            "PASS": "order_id>=0.90 phone>=0.90 http_args>=0.90 AND v0b+en_lex+vi_lex prior PASS",
            "FAIL": "order_id or phone <0.70",
            "UNRESOLVED": "else",
        },
        "metrics": {
            "order_id_acc": oid_a,
            "phone_acc": ph_a,
            "http_args_exact": http_a,
            "extract_detail": sc,
            "http_detail": http["detail"],
        },
        "regression": {"v0b": v0b, "en_lexicon": en, "vi_lexicon": vi},
        "note": "8 tools unchanged; extract is additive",
    }
    out = ARTIFACTS / "r" / "uns" / "order_nl_args_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTCOME → {outcome}")
    print(f"scorecard → {path}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
