#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tnr_p1_vdr1 import CONFIRMATORY_WORLD_SEEDS, materialize_split, verify_model_contract, verify_route_contract
from run_p1_confirmatory import verify_execution_lock

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    lock = verify_execution_lock()
    verify_route_contract()
    verify_model_contract()

    try:
        materialize_split(CONFIRMATORY_WORLD_SEEDS[0], "TEST")
        raise AssertionError("TEST firewall failed")
    except PermissionError:
        pass

    evidence = json.loads((ROOT / "evidence" / "p1_static_preflight.json").read_text(encoding="utf-8"))
    if evidence["status"] != "PASS" or evidence["test_materialized"] is not False:
        raise AssertionError("Preflight evidence invalid")

    report = {
        "qa": "TNR_P1_EXECUTION_LOCK_INDEPENDENT_QA",
        "status": "PASS",
        "execution_lock_status": lock["status"],
        "locked_file_count": len(lock["locked_files_sha256"]),
        "lock_sha256": sha256_file(ROOT / "configs" / "p1_execution_lock.json"),
        "test_materialized": False,
        "test_firewall": "PASS",
        "route_contract": "PASS",
        "model_contract": "PASS",
    }
    (ROOT / "evidence" / "p1_execution_lock_qa.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
