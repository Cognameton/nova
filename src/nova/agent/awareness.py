"""Awareness classification for Nova self-orientation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from nova.persona.state import PersonaState, SelfState
from nova.types import MemoryEvent, SCHEMA_VERSION


@dataclass(slots=True)
class AwarenessResult:
    schema_version: str = SCHEMA_VERSION
    known_facts: list[str] = field(default_factory=list)
    inferred_beliefs: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    confidence_by_section: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AwarenessClassifier:
    """Classify what Nova knows, infers, and does not know."""

    def build(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        graph_events: Iterable[MemoryEvent] = (),
        semantic_events: Iterable[MemoryEvent] = (),
        autobiographical_events: Iterable[MemoryEvent] = (),
    ) -> AwarenessResult:
        known_facts = self._known_facts(persona, self_state, graph_events, autobiographical_events)
        inferred_beliefs = self._inferred_beliefs(semantic_events, autobiographical_events)
        unknowns = self._unknowns(self_state)
        confidence_by_section = {
            "known_facts": self._confidence(len(known_facts), floor=0.55, ceiling=0.98),
            "inferred_beliefs": self._confidence(len(inferred_beliefs), floor=0.45, ceiling=0.9),
            "unknowns": 0.95 if unknowns else 0.85,
        }
        return AwarenessResult(
            known_facts=known_facts,
            inferred_beliefs=inferred_beliefs,
            unknowns=unknowns,
            confidence_by_section=confidence_by_section,
        )

    def _known_facts(
        self,
        persona: PersonaState,
        self_state: SelfState,
        graph_events: Iterable[MemoryEvent],
        autobiographical_events: Iterable[MemoryEvent],
    ) -> list[str]:
        facts: list[str] = [
            f"My name is {persona.name}.",
            self_state.identity_summary.strip(),
        ]

        for event in graph_events:
            if event.retention == "pruned":
                continue
            if not self._is_active(event):
                continue
            if event.confidence < 0.7:
                continue
            facts.append(self._normalize_fact_text(event))

        for event in autobiographical_events:
            if event.retention != "active":
                continue
            if event.confidence < 0.85:
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                facts.append(text)

        return self._dedupe_preserve_order(facts)

    def _inferred_beliefs(
        self,
        semantic_events: Iterable[MemoryEvent],
        autobiographical_events: Iterable[MemoryEvent],
    ) -> list[str]:
        beliefs: list[str] = []

        for event in semantic_events:
            if event.retention == "pruned":
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                beliefs.append(text)

        for event in autobiographical_events:
            if event.retention == "pruned":
                continue
            if event.source != "reflection":
                continue
            if event.confidence >= 0.85:
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                beliefs.append(text)

        return self._dedupe_preserve_order(beliefs)

    def _unknowns(self, self_state: SelfState) -> list[str]:
        unknowns = [item.strip() for item in self_state.active_questions if item.strip()]
        unknowns.extend(item.strip() for item in self_state.open_tensions if item.strip())
        return self._dedupe_preserve_order(unknowns)

    def _normalize_fact_text(self, event: MemoryEvent) -> str:
        metadata = dict(event.metadata or {})
        subject_name = str(metadata.get("subject_name", "") or metadata.get("subject_key", "") or "").strip()
        relation = str(metadata.get("relation", "") or "").strip()
        object_name = str(metadata.get("object_name", "") or metadata.get("object_key", "") or "").strip()
        if subject_name and relation and object_name:
            return f"{subject_name} {relation} {object_name}."
        return (event.summary or event.text or "").strip()

    def _is_active(self, event: MemoryEvent) -> bool:
        return bool(dict(event.metadata or {}).get("active", event.retention == "active"))

    def _confidence(self, count: int, *, floor: float, ceiling: float) -> float:
        return min(ceiling, floor + (0.08 * max(0, count)))

    def _dedupe_preserve_order(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            normalized = " ".join(value.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered
