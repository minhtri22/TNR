"""TNR kernel: program → validate → execute → audit → scorecard.
Domain-agnostic. Domain packs supply schema, policies, connectors, eval suites.
"""
from .program import Program, Slot
from .validate import validate_program
from .executor import Executor, ExecutionResult
from .audit import AuditLog
from .scorecard import Scorecard, exact_state, decide_outcome

__all__ = [
    "Program",
    "Slot",
    "validate_program",
    "Executor",
    "ExecutionResult",
    "AuditLog",
    "Scorecard",
    "exact_state",
    "decide_outcome",
]
