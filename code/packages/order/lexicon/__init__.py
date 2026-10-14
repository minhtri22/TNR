"""Order pack lexicons (controlled NL → tool)."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

_DIR = Path(__file__).resolve().parent


def load_lexicon(lang: str = "en", version: str = "v1") -> Dict:
    path = _DIR / f"{lang}_{version}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compile_patterns(lex: Dict) -> List[Tuple[re.Pattern, str]]:
    """Longer patterns first to prefer specific matches."""
    items: List[Tuple[str, str]] = []
    for tool, spec in lex.get("tools", {}).items():
        for pat in spec.get("patterns", []):
            items.append((pat, tool))
    items.sort(key=lambda x: len(x[0]), reverse=True)
    return [(re.compile(re.escape(p), re.I), tool) for p, tool in items]


def parse_nl(text: str, compiled: List[Tuple[re.Pattern, str]], max_slots: int = 2) -> List[str]:
    parts = re.split(r"\s+then\s+|[.;]", text, flags=re.I)
    tools: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        hit = None
        for cre, tool in compiled:
            if cre.search(part):
                hit = tool
                break
        tools.append(hit if hit else "NOP")
    while len(tools) < max_slots:
        tools.append("NOP")
    return tools[:max_slots]
