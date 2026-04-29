"""Autobiographical and self-state memory store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nova.memory.governance import (
    merge_provenance,
    normalize_governed_event,
    payload_conflicts,
    payload_governance_scope,
    payload_support_count,
)
from nova.types import MemoryEvent, RetrievalHit


class JsonlAutobiographicalMemoryStore:
    """Privileged autobiographical memory for identity-relevant records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add(self, event: MemoryEvent) -> None:
        event = normalize_governed_event(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def merge_reflection_candidate(self, event: MemoryEvent) -> MemoryEvent:
        event = normalize_governed_event(event)
        payloads = self._load_events()
        scope = str((event.metadata or {}).get("governance_scope", "") or "")
        if not scope:
            self.add(event)
            return event

        conflict_ids: list[str] = []
        for index, payload in enumerate(payloads):
            metadata = dict(payload.get("metadata", {}) or {})
            if (
                str(payload.get("source", "") or "") == "reflection"
                and str(payload.get("retention", "active") or "active") == "active"
                and payload_governance_scope(payload) == scope
            ):
                if payload_conflicts(payload, event):
                    payloads[index] = self._archive_superseded_payload(payload, event.event_id, event.timestamp)
                    conflict_ids.append(str(payload.get("event_id", "") or ""))
                    continue
                merged = self._merge_payload(payload, event)
                payloads[index] = merged
                self._save_payloads(payloads)
                return self._event_from_payload(merged)

        if conflict_ids:
            event.metadata["supersedes_conflicts"] = conflict_ids
            self._save_payloads(payloads)
        self.add(event)
        return event

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        tokens = self._tokens(query)
        if not tokens:
            return []

        hits: list[RetrievalHit] = []
        for payload in self._load_events():
            retention = str(payload.get("retention", "active") or "active")
            if retention == "pruned":
                continue
            text = str(payload.get("text", "") or "")
            text_tokens = set(self._tokens(text))
            if not text_tokens:
                continue
            overlap = len(tokens.intersection(text_tokens))
            importance = float(payload.get("importance", 0.0) or 0.0)
            confidence = float(payload.get("confidence", 1.0) or 1.0)
            continuity_weight = float(payload.get("continuity_weight", 0.0) or 0.0)
            support_count = payload_support_count(payload)
            score = overlap + importance + (0.2 * confidence) + continuity_weight + min(1.0, support_count * 0.15)
            if score <= 0:
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if str(metadata.get("governance_status", "active") or "active") == "active":
                score += 0.2
            if retention == "demoted":
                score *= 0.7
            elif retention == "archived":
                score *= 0.45
            hits.append(
                RetrievalHit(
                    channel="autobiographical",
                    text=text,
                    score=float(score),
                    kind=payload.get("kind"),
                    source_ref=payload.get("event_id"),
                    tags=list(payload.get("tags", []) or []),
                    metadata={
                        **metadata,
                        "retention": retention,
                        "importance": importance,
                        "continuity_weight": continuity_weight,
                        "confidence": confidence,
                        "support_count": support_count,
                    },
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

    def apply_maintenance_decisions(self, decisions: list[object]) -> int:
        decision_map = {
            getattr(decision, "event_id", ""): decision
            for decision in decisions
            if getattr(decision, "event_id", "")
        }
        if not decision_map:
            return 0

        payloads = self._load_events()
        updated = 0
        for payload in payloads:
            event_id = str(payload.get("event_id", "") or "")
            decision = decision_map.get(event_id)
            if decision is None:
                continue
            payload["retention"] = str(getattr(decision, "target_retention", "active") or "active")
            metadata = dict(payload.get("metadata", {}) or {})
            metadata["maintenance"] = {
                "action": getattr(decision, "action", ""),
                "reason": getattr(decision, "reason", ""),
                "applied_at": datetime.now(timezone.utc).isoformat(),
            }
            payload["metadata"] = metadata
            updated += 1

        if updated:
            self._save_payloads(payloads)
        return updated

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

    def _save_payloads(self, payloads: list[dict]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _merge_payload(self, existing: dict, event: MemoryEvent) -> dict:
        existing_metadata = dict(existing.get("metadata", {}) or {})
        event_metadata = dict(event.metadata or {})
        merged_provenance = merge_provenance(existing_metadata, event_metadata)
        merged_source_ids = sorted(
            {
                *list(merged_provenance.get("source_event_ids", []) or []),
                *list(existing.get("supersedes", []) or []),
                *list(event.supersedes or []),
            }
        )
        merged_tags = sorted(set(list(existing.get("tags", []) or []) + list(event.tags or [])))
        revision_count = int(existing_metadata.get("revision_count", 0) or 0) + 1

        existing.update(
            {
                "timestamp": event.timestamp,
                "text": event.text,
                "summary": event.summary,
                "tags": merged_tags,
                "importance": max(float(existing.get("importance", 0.0) or 0.0), event.importance),
                "confidence": max(float(existing.get("confidence", 1.0) or 1.0), event.confidence),
                "continuity_weight": max(
                    float(existing.get("continuity_weight", 0.0) or 0.0),
                    event.continuity_weight,
                ),
                "retention": "active",
                "supersedes": merged_source_ids,
                "source": "reflection",
                "metadata": {
                    **existing_metadata,
                    **event_metadata,
                    "source_event_ids": merged_source_ids,
                    "source_channels": merged_provenance.get("source_channels", []),
                    "support_count": merged_provenance.get("support_count", 0),
                    "provenance": merged_provenance,
                    "governance_status": "active",
                    "revision_count": revision_count,
                    "revised_at": event.timestamp,
                },
            }
        )
        return existing

    def _archive_superseded_payload(self, payload: dict, successor_id: str, timestamp: str) -> dict:
        metadata = dict(payload.get("metadata", {}) or {})
        metadata["governance_status"] = "superseded"
        metadata["superseded_by"] = successor_id
        metadata["superseded_at"] = timestamp
        payload["retention"] = "archived"
        payload["metadata"] = metadata
        return payload

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

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in text.replace("\n", " ").lower().split()
            if token.strip()
        }
