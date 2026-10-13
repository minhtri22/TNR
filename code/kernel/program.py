"""Program = ordered list of slots (tool_id + args). Domain-agnostic structure."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Slot:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass
class Program:
    slots: List[Slot] = field(default_factory=list)
    pack: str = ""  # domain pack id, e.g. "order"

    def to_dict(self) -> Dict[str, Any]:
        return {"pack": self.pack, "slots": [s.to_dict() for s in self.slots]}

    def tools(self) -> List[str]:
        return [s.tool for s in self.slots]

    def __len__(self) -> int:
        return len(self.slots)
