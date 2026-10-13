"""Unified scorecard for domain packs. Same format for order/ticket/..."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import time


def exact_state(pred: Dict[str, Any], gold: Dict[str, Any], keys: Optional[List[str]] = None) -> bool:
    """Exact match on selected state keys (float tol 1e-6 for numbers)."""
    ks = keys if keys is not None else list(gold.keys())
    for k in ks:
        if k not in pred or k not in gold:
            return False
        a, b = pred[k], gold[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-6:
                return False
        else:
            if a != b:
                return False
    return True


def decide_outcome(
    mean_flat: float,
    mean_prog: float,
    wins: int,
    n_seeds: int = 3,
    gap: float = 0.05,
    min_prog: float = 0.35,
    min_wins: int = 2,
) -> str:
    if mean_prog > mean_flat + gap and mean_prog >= min_prog and wins >= min_wins:
        return "PASS"
    if mean_prog + gap < mean_flat:
        return "FAIL"
    return "UNRESOLVED"


@dataclass
class Scorecard:
    suite: str
    pack: str
    seeds: List[int]
    metrics: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "UNRESOLVED"
    preregister: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "pack": self.pack,
            "seeds": self.seeds,
            "metrics": self.metrics,
            "outcome": self.outcome,
            "preregister": self.preregister,
            "ts": self.ts,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
