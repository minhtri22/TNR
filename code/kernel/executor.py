"""Executor: run validated program via pack connector. Kernel does not know domain state."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .program import Program
from .validate import validate_program
from .audit import AuditLog, AuditEvent


@dataclass
class ExecutionResult:
    ok: bool
    state: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    events: List[AuditEvent] = field(default_factory=list)


class Executor:
    def __init__(
        self,
        allowed_tools: List[str],
        policies: Dict[str, Any],
        apply_slot: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        initial_state: Optional[Dict[str, Any]] = None,
        audit: Optional[AuditLog] = None,
    ):
        self.allowed_tools = allowed_tools
        self.policies = policies
        self.apply_slot = apply_slot
        self.initial_state = initial_state or {}
        self.audit = audit or AuditLog()

    def run(self, program: Program, state: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        ok, errs = validate_program(program, self.allowed_tools, self.policies)
        st = dict(state if state is not None else self.initial_state)
        events: List[AuditEvent] = []

        if not ok:
            # hard fail before any side-effect
            for e in errs:
                ev = self.audit.record(
                    tool="VALIDATE",
                    args={},
                    status="blocked",
                    detail=e,
                    state_snapshot=dict(st),
                )
                events.append(ev)
            return ExecutionResult(ok=False, state=st, errors=errs, events=events)

        for slot in program.slots:
            try:
                st = self.apply_slot(slot.tool, slot.args, st)
                ev = self.audit.record(
                    tool=slot.tool,
                    args=dict(slot.args),
                    status="ok",
                    detail="",
                    state_snapshot=dict(st),
                )
                events.append(ev)
            except Exception as ex:  # connector / policy runtime
                ev = self.audit.record(
                    tool=slot.tool,
                    args=dict(slot.args),
                    status="error",
                    detail=str(ex),
                    state_snapshot=dict(st),
                )
                events.append(ev)
                return ExecutionResult(ok=False, state=st, errors=[str(ex)], events=events)

        return ExecutionResult(ok=True, state=st, errors=[], events=events)
