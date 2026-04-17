"""Memory routing and retrieval logic."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.types import MemoryEvent, RetrievalHit


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BasicMemoryEventFactory:
    """Create normalized memory events from validated conversation turns."""

    def from_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_text: str,
        final_answer: str,
    ) -> list[MemoryEvent]:
        timestamp = utc_now_iso()
        events: list[MemoryEvent] = [
            MemoryEvent(
                event_id=uuid4().hex,
                timestamp=timestamp,
                session_id=session_id,
                turn_id=turn_id,
                channel="episodic",
                kind="user_message",
                text=user_text.strip(),
                summary=None,
                tags=["user", "turn"],
                importance=0.4,
                confidence=1.0,
                source="user",
                metadata={},
            ),
            MemoryEvent(
                event_id=uuid4().hex,
                timestamp=timestamp,
                session_id=session_id,
                turn_id=turn_id,
                channel="episodic",
                kind="assistant_message",
                text=final_answer.strip(),
                summary=None,
                tags=["assistant", "turn"],
                importance=0.4,
                confidence=1.0,
                source="nova",
                metadata={},
            ),
        ]

        combined_text = f"{user_text}\n{final_answer}".lower()
        if any(token in combined_text for token in ("identity", "name", "who are you", "continuity", "self")):
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="autobiographical",
                    kind="identity_note",
                    text=final_answer.strip(),
                    summary=None,
                    tags=["identity", "self-model"],
                    importance=0.9,
                    confidence=0.9,
                    source="nova",
                    metadata={},
                )
            )

        return [event for event in events if event.text]


class BasicMemoryRouter:
    """Route writes and retrieval across enabled memory channels."""

    def __init__(
        self,
        *,
        episodic: JsonlEpisodicMemoryStore | None = None,
        graph: SqliteGraphMemoryStore | None = None,
        autobiographical: JsonlAutobiographicalMemoryStore | None = None,
    ):
        self.stores: dict[str, Any] = {}
        if episodic is not None:
            self.stores["episodic"] = episodic
        if graph is not None:
            self.stores["graph"] = graph
        if autobiographical is not None:
            self.stores["autobiographical"] = autobiographical

    def add_events(self, events: list[MemoryEvent]) -> None:
        for event in events:
            store = self.stores.get(event.channel)
            if store is None:
                continue
            store.add(event)

    def retrieve(
        self,
        *,
        query: str,
        top_k_by_channel: dict[str, int],
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for channel, top_k in top_k_by_channel.items():
            if top_k <= 0:
                continue
            store = self.stores.get(channel)
            if store is None:
                continue
            hits.extend(store.search(query, top_k=top_k))

        hits.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_hits(hits)

    def stats(self) -> dict[str, Any]:
        return {
            channel: store.stats()
            for channel, store in self.stores.items()
        }

    def _dedupe_hits(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        deduped: list[RetrievalHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            key = (hit.channel, hit.text.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped
