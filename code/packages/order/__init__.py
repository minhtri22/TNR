"""Order domain pack v0 — schema, mock connector, policies."""
from .connectors.mock_store import apply_slot, empty_state

__all__ = ["apply_slot", "empty_state"]
