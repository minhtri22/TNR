"""Validate program against pack schema + policies before execute."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .program import Program


def validate_program(
    program: Program,
    allowed_tools: List[str],
    policies: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Return (ok, errors). Does not execute side-effects."""
    errors: List[str] = []
    if not program.slots:
        errors.append("empty_program")
        return False, errors

    for i, slot in enumerate(program.slots):
        if slot.tool not in allowed_tools:
            errors.append(f"slot[{i}]: unknown_tool:{slot.tool}")
            continue
        # policy: tool requires confirm somewhere before/at this index
        need_confirm = policies.get("require_confirm_before", {})
        if slot.tool in need_confirm:
            # confirm must appear at or before this slot
            conf_tool = need_confirm[slot.tool]
            prior = [s.tool for s in program.slots[: i + 1]]
            if conf_tool not in prior:
                errors.append(f"slot[{i}]: missing_{conf_tool}_before_{slot.tool}")

    max_len = int(policies.get("max_slots", 8))
    if len(program.slots) > max_len:
        errors.append(f"too_many_slots:{len(program.slots)}>{max_len}")

    return len(errors) == 0, errors
