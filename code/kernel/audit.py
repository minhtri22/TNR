"""Append-only audit log of slot executions."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json


@dataclass
class AuditEvent:
    ts: str
    tool: str
    args: Dict[str, Any]
    status: str  # ok | error | blocked
    detail: str
    state_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "detail": self.detail,
            "state": self.state_snapshot,
        }


class AuditLog:
    def __init__(self):
        self.events: List[AuditEvent] = []

    def record(
        self,
        tool: str,
        args: Dict[str, Any],
        status: str,
        detail: str = "",
        state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        ev = AuditEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            tool=tool,
            args=args,
            status=status,
            detail=detail,
            state_snapshot=state_snapshot or {},
        )
        self.events.append(ev)
        return ev

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in self.events)
