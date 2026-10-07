#!/usr/bin/env python3
"""Điểm vào xác nhận P1 dùng môi trường TNR trên ổ D."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
if not SITE_PACKAGES.exists():
    raise RuntimeError("TNR .venv is missing")
sys.path.insert(0, str(SITE_PACKAGES))
runpy.run_path(str(ROOT / "scripts" / "run_p1_confirmatory.py"), run_name="__main__")
