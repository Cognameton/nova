"""Reflection helpers for promoting memory into higher-order continuity records."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from nova.memory.consolidation import SemanticConsolidator
from nova.types import MemoryEvent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReflectionEngine:
    """Produce governed semantic and autobiographical promotion candidates."""

    SELF_REFERENCE_CUES = (" i ", " my ", " me ", " myself ", "nova")
    GENERIC_EXPLANATION_CUES = (
        "you can",
        "you should",
        "here are",
        "for example",
        "this means",
        "the answer is",
    )
    SHIFT_CUES = (
        "used to",
        "no longer",
        "have changed",
        "have shifted",
        "shifted toward",
        "now orient",
        "now focus",
        "have become",
    )
    TENSION_CUES = (
        "but i also",
        "while also",
        "still balancing",
        "not yet settled",
        "unresolved",
        "tradeoff",
        "tension",
    )

    def __init__(self, semantic_consolidator: SemanticConsolidator | None = None):
        self.semantic_consolidator = semantic_consolidator or SemanticConsolidator()

    def build_semantic_candidates(self, episodic_events: list[MemoryEvent]) -> list[MemoryEvent]:
        return self.semantic_consolidator.build_candidates(episodic_events)

    def build_autobiographical_candidates(
        self,
        *,
        episodic_events: list[MemoryEvent],
        semantic_events: list[MemoryEvent] | None = None,
    ) -> list[MemoryEvent]:
        semantic_events = semantic_events or []
        eligible = [
            event
            for event in episodic_events
            if self._eligible_autobiographical_source(event)
        ]
        eligible.extend(
            event
            for event in semantic_events
            if self._eligible_semantic_source(event)
        )

        clusters: dict[str, list[MemoryEvent]] = defaultdict(list)
        for event in eligible:
            clusters[self._autobiographical_theme(event)].append(event)

        candidates: list[MemoryEvent] = []
        for theme, cluster in clusters.items():
            candidate = self._build_autobiographical_candidate(theme, cluster)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _eligible_autobiographical_source(self, event: MemoryEvent) -> bool:
        if event.channel != "episodic":
            return False
        if event.source != "nova":
            return False
        if not self._looks_self_interpreting(event.text):
            return False
        if self._looks_generic_explanatory(event.text):
            return False
        if event.continuity_weight >= 0.85 or event.importance >= 0.8:
            return True
        return any(tag in event.tags for tag in ("identity", "relationship", "value"))

    def _eligible_semantic_source(self, event: MemoryEvent) -> bool:
        if event.channel != "semantic":
            return False
        theme = str((event.metadata or {}).get("theme", "") or "")
        if theme == "relationship-context":
            return False
        if "nova-identity" in event.tags or "nova-values" in event.tags or "continuity-focus" in event.tags:
            return True
        return theme in {"nova-identity", "nova-values", "continuity-focus"} and event.continuity_weight >= 0.75

    def _autobiographical_theme(self, event: MemoryEvent) -> str:
        if "identity" in event.tags or "nova-identity" in event.tags:
            return "identity-continuity"
        if "relationship" in event.tags or "relationship-context" in event.tags:
            return "relationship-continuity"
        if "value" in event.tags or "nova-values" in event.tags:
            return "value-continuity"
        if event.continuity_weight >= 0.85:
            return "continuity-focus"
        return "self-model"

    def _build_autobiographical_candidate(
        self,
        theme: str,
        cluster: list[MemoryEvent],
    ) -> MemoryEvent | None:
        unique_texts: list[str] = []
        seen = set()
        for event in cluster:
            normalized = event.text.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_texts.append(event.text.strip())
        preview_texts = self._preview_texts(cluster, unique_texts)

        distinct_turns = {event.turn_id for event in cluster if event.turn_id}
        semantic_support = any(event.channel == "semantic" for event in cluster)
        if (
            len(unique_texts) < 2
            and len(distinct_turns) < 2
            and not semantic_support
            and max((event.continuity_weight for event in cluster), default=0.0) < 0.9
        ):
            return None

        avg_importance = sum(event.importance for event in cluster) / max(1, len(cluster))
        avg_confidence = sum(event.confidence for event in cluster) / max(1, len(cluster))
        avg_continuity = sum(event.continuity_weight for event in cluster) / max(1, len(cluster))
        source_event_ids = [event.event_id for event in cluster if event.event_id]
        note_type = self._autobiographical_note_type(
            cluster=cluster,
            unique_texts=unique_texts,
            semantic_support=semantic_support,
        )

        prefix = self._autobiographical_prefix(theme=theme, note_type=note_type)
        summary_text = f"{prefix}: {'; '.join(preview_texts[:3])}"

        return MemoryEvent(
            event_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=cluster[-1].session_id if cluster else "",
            turn_id=cluster[-1].turn_id if cluster else "",
            channel="autobiographical",
            kind=self._autobiographical_kind(note_type),
            text=summary_text,
            summary=summary_text,
            tags=sorted(
                {
                    "autobiographical",
                    "reflection",
                    theme,
                    note_type,
                    *[tag for event in cluster for tag in event.tags if tag != "turn"],
                }
            ),
            importance=min(1.0, avg_importance + 0.2),
            confidence=min(1.0, max(0.7, avg_confidence)),
            continuity_weight=min(1.0, avg_continuity + 0.15),
            retention="active",
            supersedes=source_event_ids,
            source="reflection",
            metadata={
                "theme": theme,
                "note_type": note_type,
                "source_event_ids": source_event_ids,
                "event_count": len(cluster),
                "distinct_turn_count": len(distinct_turns),
                "source_channels": sorted({event.channel for event in cluster if event.channel}),
                "semantic_support": semantic_support,
                "generated_by": "reflection_engine",
            },
        )

    def _preview_texts(self, cluster: list[MemoryEvent], unique_texts: list[str]) -> list[str]:
        episodic_texts: list[str] = []
        seen = set()
        for event in cluster:
            if event.channel != "episodic":
                continue
            normalized = event.text.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            episodic_texts.append(event.text.strip())
        return episodic_texts or unique_texts

    def _looks_self_interpreting(self, text: str) -> bool:
        lowered = f" {text.lower()} "
        return any(cue in lowered for cue in self.SELF_REFERENCE_CUES)

    def _looks_generic_explanatory(self, text: str) -> bool:
        lowered = " ".join(text.lower().split())
        if "\n- " in text or text.count("\n") >= 2:
            return True
        if not self._looks_self_interpreting(text):
            return any(cue in lowered for cue in self.GENERIC_EXPLANATION_CUES)
        return lowered.startswith("you ") or any(cue in lowered for cue in self.GENERIC_EXPLANATION_CUES)

    def _autobiographical_note_type(
        self,
        *,
        cluster: list[MemoryEvent],
        unique_texts: list[str],
        semantic_support: bool,
    ) -> str:
        lowered = " ".join(text.lower() for text in unique_texts)
        if any(cue in lowered for cue in self.TENSION_CUES):
            return "unresolved-tension"
        if any(cue in lowered for cue in self.SHIFT_CUES):
            return "continuity-shift"
        if semantic_support or len(unique_texts) >= 3:
            return "developmental-milestone"
        return "continuity-note"

    def _autobiographical_prefix(self, *, theme: str, note_type: str) -> str:
        base = {
            "identity-continuity": "Identity",
            "relationship-continuity": "Relationship",
            "value-continuity": "Value",
            "continuity-focus": "Continuity focus",
            "self-model": "Self-model",
        }.get(theme, "Autobiographical reflection")
        suffix = {
            "developmental-milestone": "milestone",
            "continuity-shift": "shift",
            "unresolved-tension": "tension",
            "continuity-note": "continuity",
        }.get(note_type, "continuity")
        return f"{base} {suffix}"

    def _autobiographical_kind(self, note_type: str) -> str:
        return {
            "developmental-milestone": "developmental_milestone",
            "continuity-shift": "continuity_shift",
            "unresolved-tension": "continuity_tension",
            "continuity-note": "reflection_note",
        }.get(note_type, "reflection_note")
