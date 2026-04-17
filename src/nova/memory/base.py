"""Memory store contracts."""

from __future__ import annotations

from typing import Any, Protocol

from nova.types import MemoryEvent, RetrievalHit


class MemoryStore(Protocol):
    """Minimal memory store contract for Phase 1."""

    def add(self, event: MemoryEvent) -> None:
        """Persist one normalized memory event."""

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        """Return normalized retrieval hits for a query."""

    def stats(self) -> dict[str, Any]:
        """Return simple store statistics."""
