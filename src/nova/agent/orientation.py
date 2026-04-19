"""Self-orientation snapshot builder for Nova."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from nova.agent.awareness import AwarenessClassifier
from nova.agent.boundaries import BoundaryPolicy
from nova.persona.state import PersonaState, SelfState
from nova.types import MemoryEvent, SCHEMA_VERSION


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class OrientationSnapshot:
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    relationship_context: dict[str, Any] = field(default_factory=dict)
    known_facts: list[str] = field(default_factory=list)
    inferred_beliefs: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    approval_required_actions: list[str] = field(default_factory=list)
    confidence_by_section: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfOrientationEngine:
    """Build a structured self-orientation snapshot from Nova state and memory."""

    def __init__(
        self,
        *,
        awareness: AwarenessClassifier | None = None,
        boundaries: BoundaryPolicy | None = None,
    ) -> None:
        self.awareness = awareness or AwarenessClassifier()
        self.boundaries = boundaries or BoundaryPolicy()

    def build_snapshot(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        graph_memory: object | None = None,
        semantic_memory: object | None = None,
        autobiographical_memory: object | None = None,
    ) -> OrientationSnapshot:
        graph_events = self._load_events(graph_memory)
        semantic_events = self._load_events(semantic_memory)
        autobiographical_events = self._load_events(autobiographical_memory)

        awareness = self.awareness.build(
            persona=persona,
            self_state=self_state,
            graph_events=graph_events,
            semantic_events=semantic_events,
            autobiographical_events=autobiographical_events,
        )
        latitude = self.boundaries.build()

        identity = {
            "name": persona.name,
            "core_description": persona.core_description,
            "identity_summary": self_state.identity_summary,
            "values": list(persona.values),
            "commitments": list(persona.commitments),
            "identity_anchors": list(persona.identity_anchors),
        }
        current_state = {
            "current_focus": self_state.current_focus,
            "active_questions": list(self_state.active_questions),
            "open_tensions": list(self_state.open_tensions),
            "continuity_notes": list(self_state.continuity_notes),
            "stability_version": self_state.stability_version,
        }
        relationship_context = {
            "relationship_notes": self._relationship_notes(self_state, graph_events, semantic_events),
            "stable_preferences": list(self_state.stable_preferences),
        }

        confidence_by_section = {
            "identity": self._identity_confidence(persona, self_state),
            "current_state": self._current_state_confidence(self_state),
            "relationship_context": self._relationship_confidence(relationship_context),
            "known_facts": awareness.confidence_by_section.get("known_facts", 0.0),
            "inferred_beliefs": awareness.confidence_by_section.get("inferred_beliefs", 0.0),
            "unknowns": awareness.confidence_by_section.get("unknowns", 0.0),
            "operational_latitude": latitude.confidence,
        }

        return OrientationSnapshot(
            created_at=utc_now_iso(),
            identity=identity,
            current_state=current_state,
            relationship_context=relationship_context,
            known_facts=awareness.known_facts,
            inferred_beliefs=awareness.inferred_beliefs,
            unknowns=awareness.unknowns,
            allowed_actions=latitude.allowed_actions,
            blocked_actions=latitude.blocked_actions,
            approval_required_actions=latitude.approval_required_actions,
            confidence_by_section=confidence_by_section,
        )

    def _load_events(self, memory: object | None) -> list[MemoryEvent]:
        if memory is None:
            return []
        list_events = getattr(memory, "list_events", None)
        if callable(list_events):
            return list(list_events())
        if isinstance(memory, list):
            return [event for event in memory if isinstance(event, MemoryEvent)]
        return []

    def _relationship_notes(
        self,
        self_state: SelfState,
        graph_events: Iterable[MemoryEvent],
        semantic_events: Iterable[MemoryEvent],
    ) -> list[str]:
        notes = [item.strip() for item in self_state.relationship_notes if item.strip()]

        for event in graph_events:
            metadata = dict(event.metadata or {})
            if str(metadata.get("fact_domain", "") or "") != "relationship":
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                notes.append(text)

        for event in semantic_events:
            metadata = dict(event.metadata or {})
            theme = str(metadata.get("theme", "") or "")
            if "relationship" not in theme:
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                notes.append(text)

        return self._dedupe_preserve_order(notes)

    def _identity_confidence(self, persona: PersonaState, self_state: SelfState) -> float:
        populated = 0
        candidates = [
            persona.name,
            persona.core_description,
            self_state.identity_summary,
            *persona.values,
            *persona.commitments,
            *persona.identity_anchors,
        ]
        populated = sum(1 for item in candidates if str(item).strip())
        return min(0.99, 0.45 + (0.04 * populated))

    def _current_state_confidence(self, self_state: SelfState) -> float:
        populated = sum(
            1
            for item in (
                self_state.current_focus,
                *self_state.active_questions,
                *self_state.continuity_notes,
            )
            if str(item).strip()
        )
        return min(0.95, 0.4 + (0.08 * populated))

    def _relationship_confidence(self, relationship_context: dict[str, Any]) -> float:
        notes = relationship_context.get("relationship_notes", []) or []
        preferences = relationship_context.get("stable_preferences", []) or []
        populated = len(notes) + len(preferences)
        return min(0.95, 0.3 + (0.1 * populated))

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
