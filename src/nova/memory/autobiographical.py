"""Autobiographical and self-state memory store."""

from __future__ import annotations

import json
from pathlib import Path

from nova.types import MemoryEvent, RetrievalHit


class JsonlAutobiographicalMemoryStore:
    """Privileged autobiographical memory for identity-relevant records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add(self, event: MemoryEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        tokens = self._tokens(query)
        if not tokens:
            return []

        hits: list[RetrievalHit] = []
        for payload in self._load_events():
            text = str(payload.get("text", "") or "")
            text_tokens = set(self._tokens(text))
            if not text_tokens:
                continue
            overlap = len(tokens.intersection(text_tokens))
            importance = float(payload.get("importance", 0.0) or 0.0)
            score = overlap + importance
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    channel="autobiographical",
                    text=text,
                    score=float(score),
                    kind=payload.get("kind"),
                    source_ref=payload.get("event_id"),
                    tags=list(payload.get("tags", []) or []),
                    metadata=dict(payload.get("metadata", {}) or {}),
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, top_k)]

    def stats(self) -> dict[str, int | str]:
        return {
            "channel": "autobiographical",
            "entries": len(self._load_events()),
            "path": str(self.path),
        }

    def list_events(self) -> list[MemoryEvent]:
        return [self._event_from_payload(payload) for payload in self._load_events()]

    def _load_events(self) -> list[dict]:
        events: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    events.append(payload)
        return events

    def _event_from_payload(self, payload: dict) -> MemoryEvent:
        return MemoryEvent(
            schema_version=str(payload.get("schema_version", "")) or "1.0",
            event_id=str(payload.get("event_id", "") or ""),
            timestamp=str(payload.get("timestamp", "") or ""),
            session_id=str(payload.get("session_id", "") or ""),
            turn_id=str(payload.get("turn_id", "") or ""),
            channel=str(payload.get("channel", "autobiographical") or "autobiographical"),
            kind=str(payload.get("kind", "") or ""),
            text=str(payload.get("text", "") or ""),
            summary=payload.get("summary"),
            tags=list(payload.get("tags", []) or []),
            importance=float(payload.get("importance", 0.0) or 0.0),
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            continuity_weight=float(payload.get("continuity_weight", 0.0) or 0.0),
            retention=str(payload.get("retention", "active") or "active"),
            supersedes=list(payload.get("supersedes", []) or []),
            source=str(payload.get("source", "") or ""),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def _tokens(self, text: str) -> list[str]:
        return [token for token in text.replace("\n", " ").lower().split() if token.strip()]
