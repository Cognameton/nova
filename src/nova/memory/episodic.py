"""Episodic memory store."""

from __future__ import annotations

import json
from pathlib import Path

from nova.types import MemoryEvent, RetrievalHit


class JsonlEpisodicMemoryStore:
    """Append-only episodic memory with lightweight lexical retrieval."""

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
            if overlap <= 0:
                continue
            score = overlap / max(1, len(tokens))
            hits.append(
                RetrievalHit(
                    channel="episodic",
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
            "channel": "episodic",
            "entries": len(self._load_events()),
            "path": str(self.path),
        }

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

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in text.replace("\n", " ").split()
            if token.strip()
        }
