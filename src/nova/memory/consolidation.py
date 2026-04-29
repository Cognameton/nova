"""Consolidation helpers for promoting episodic memory into semantic memory."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from nova.memory.governance import normalize_governed_event
from nova.types import MemoryEvent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SemanticConsolidator:
    """Build conservative semantic summary candidates from episodic memory."""

    THEME_LABELS = {
        "user-preferences": "User preferences",
        "user-values": "User values",
        "nova-identity": "Nova identity",
        "nova-values": "Nova values",
        "relationship-context": "Relationship context",
        "continuity-focus": "Continuity focus",
    }

    def build_candidates(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        eligible = [event for event in events if self._eligible(event)]
        clusters: dict[str, list[MemoryEvent]] = defaultdict(list)
        for event in eligible:
            clusters[self._theme_key(event)].append(event)

        candidates: list[MemoryEvent] = []
        for theme, cluster in clusters.items():
            candidate = self._build_candidate(theme, cluster)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _eligible(self, event: MemoryEvent) -> bool:
        if event.channel != "episodic":
            return False
        if not event.text.strip():
            return False
        if "turn" not in event.tags:
            return False
        if "preference" in event.tags or "value" in event.tags:
            return True
        if event.source == "nova" and "identity" in event.tags:
            return True
        if event.source == "nova" and "relationship" in event.tags:
            return False
        if event.importance >= 0.7 or event.continuity_weight >= 0.7:
            return True
        return False

    def _theme_key(self, event: MemoryEvent) -> str:
        if "preference" in event.tags:
            return "user-preferences" if event.source == "user" else "nova-values"
        if "value" in event.tags:
            return "user-values" if event.source == "user" else "nova-values"
        if "identity" in event.tags:
            return "nova-identity" if event.source == "nova" else "relationship-context"
        if "relationship" in event.tags:
            return "relationship-context"
        if event.continuity_weight >= 0.8:
            return "continuity-focus"
        return f"{event.source or 'memory'}-theme"

    def _build_candidate(self, theme: str, cluster: list[MemoryEvent]) -> MemoryEvent | None:
        unique_texts: list[str] = []
        seen = set()
        for event in cluster:
            normalized = event.text.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_texts.append(event.text.strip())

        if len(unique_texts) < 2 and max((event.importance for event in cluster), default=0.0) < 0.85:
            return None

        avg_importance = sum(event.importance for event in cluster) / max(1, len(cluster))
        avg_confidence = sum(event.confidence for event in cluster) / max(1, len(cluster))
        avg_continuity = sum(event.continuity_weight for event in cluster) / max(1, len(cluster))
        source_event_ids = [event.event_id for event in cluster if event.event_id]

        headline = self.THEME_LABELS.get(theme, "Semantic summary")
        evidence_preview = "; ".join(unique_texts[:3])
        summary_text = f"{headline}: {evidence_preview}"

        return normalize_governed_event(
            MemoryEvent(
                event_id=uuid4().hex,
                timestamp=utc_now_iso(),
                session_id=cluster[-1].session_id if cluster else "",
                turn_id=cluster[-1].turn_id if cluster else "",
                channel="semantic",
                kind="theme_summary",
                text=summary_text,
                summary=summary_text,
                tags=sorted({"semantic", "summary", theme, *[tag for event in cluster for tag in event.tags if tag != "turn"]}),
                importance=min(1.0, avg_importance + 0.15),
                confidence=min(1.0, max(0.6, avg_confidence)),
                continuity_weight=min(1.0, avg_continuity + 0.1),
                retention="active",
                supersedes=source_event_ids,
                source="reflection",
                metadata={
                    "theme": theme,
                    "source_event_ids": source_event_ids,
                    "event_count": len(cluster),
                    "source_channels": sorted({event.channel for event in cluster if event.channel}),
                    "evidence_preview": unique_texts[:3],
                    "generated_by": "semantic_consolidator",
                },
            )
        )
