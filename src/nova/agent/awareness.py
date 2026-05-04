"""Session-scoped awareness state for Nova Phase 10."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.persona.state import PersonaState, SelfState
from nova.agent.awareness_history import JsonlAwarenessHistoryStore
from nova.types import MemoryEvent
from nova.types import AwarenessHistoryEntry, AwarenessState, SCHEMA_VERSION


AWARENESS_MONITORING_MODES = {
    "bounded",
    "reflective",
    "attentive",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AwarenessResult:
    known_facts: list[str] = field(default_factory=list)
    inferred_beliefs: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    confidence_by_section: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AwarenessClassifier:
    """Compatibility layer for orientation-aware fact/belief/unknown summarization."""

    def build(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        graph_events: list[MemoryEvent],
        semantic_events: list[MemoryEvent],
        autobiographical_events: list[MemoryEvent],
    ) -> AwarenessResult:
        known_facts = self._known_facts(
            persona=persona,
            self_state=self_state,
            graph_events=graph_events,
            semantic_events=semantic_events,
            autobiographical_events=autobiographical_events,
        )
        inferred_beliefs = self._inferred_beliefs(
            self_state=self_state,
            autobiographical_events=autobiographical_events,
            semantic_events=semantic_events,
        )
        unknowns = self._unknowns(self_state=self_state)
        return AwarenessResult(
            known_facts=known_facts,
            inferred_beliefs=inferred_beliefs,
            unknowns=unknowns,
            confidence_by_section={
                "known_facts": self._confidence(len(known_facts), floor=0.55, ceiling=0.98),
                "inferred_beliefs": self._confidence(len(inferred_beliefs), floor=0.45, ceiling=0.9),
                "unknowns": 0.95 if unknowns else 0.85,
            },
        )

    def _known_facts(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        graph_events: list[MemoryEvent],
        semantic_events: list[MemoryEvent],
        autobiographical_events: list[MemoryEvent],
    ) -> list[str]:
        items: list[str] = [
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
            relation_text = self._graph_relation_text(event)
            if relation_text:
                items.append(relation_text)
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                items.append(text)
        for event in autobiographical_events:
            if event.retention != "active":
                continue
            if event.confidence < 0.85:
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                items.append(text)
        return self._dedupe_preserve_order(items)

    def _inferred_beliefs(
        self,
        *,
        self_state: SelfState,
        autobiographical_events: list[MemoryEvent],
        semantic_events: list[MemoryEvent],
    ) -> list[str]:
        items: list[str] = []
        for event in semantic_events:
            if event.retention == "pruned":
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                items.append(text)
        for event in autobiographical_events:
            if event.retention == "pruned":
                continue
            if event.source != "reflection":
                continue
            if event.confidence >= 0.85:
                continue
            text = (event.summary or event.text or "").strip()
            if text:
                items.append(text)
        for note in self_state.continuity_notes:
            if note.strip():
                items.append(note.strip())
        for note in self_state.open_tensions:
            if note.strip():
                items.append(note.strip())
        return self._dedupe_preserve_order(items)

    def _unknowns(self, *, self_state: SelfState) -> list[str]:
        items = [question.strip() for question in self_state.active_questions if question.strip()]
        items.extend(item.strip() for item in self_state.open_tensions if item.strip())
        return self._dedupe_preserve_order(items)

    def _graph_relation_text(self, event: MemoryEvent) -> str:
        metadata = dict(event.metadata or {})
        subject = str(metadata.get("subject_name", "") or "").strip()
        relation = str(metadata.get("relation", "") or "").strip()
        obj = str(metadata.get("object_name", "") or "").strip()
        if subject and relation and obj:
            return f"{subject} {relation} {obj}."
        return ""

    def _is_active(self, event: MemoryEvent) -> bool:
        return bool(dict(event.metadata or {}).get("active", event.retention == "active"))

    def _confidence(self, count: int, *, floor: float, ceiling: float) -> float:
        return min(ceiling, floor + (0.08 * max(0, count)))

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            normalized = " ".join(value.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered


class JsonAwarenessStateStore:
    """JSON-backed session awareness store."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        history_store: JsonlAwarenessHistoryStore | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_store = history_store or JsonlAwarenessHistoryStore(
            self.base_dir / "awareness_history.jsonl"
        )
        self._recent_history_entries: list[AwarenessHistoryEntry] = []

    def load(self, *, session_id: str) -> AwarenessState:
        path = self.get_awareness_path(session_id=session_id)
        if not path.exists():
            previous = self._latest_other_state(session_id=session_id)
            if previous is not None:
                awareness = self._seed_from_previous(
                    session_id=session_id,
                    previous=previous,
                )
                self.save(
                    awareness,
                    revision_class="cross_session_seed",
                    source_session_id=previous.session_id,
                )
            else:
                awareness = default_awareness_state(session_id=session_id)
                self.save(awareness)
            return awareness

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            awareness = default_awareness_state(session_id=session_id)
            self.save(awareness)
            return awareness
        if not isinstance(payload, dict):
            awareness = default_awareness_state(session_id=session_id)
            self.save(awareness)
            return awareness
        return awareness_state_from_payload(payload=payload, session_id=session_id)

    def save(
        self,
        awareness: AwarenessState,
        *,
        revision_class: str | None = None,
        source_session_id: str | None = None,
    ) -> None:
        path = self.get_awareness_path(session_id=awareness.session_id)
        previous = None
        if path.exists():
            previous = self._load_existing(path=path, session_id=awareness.session_id)
        awareness.monitoring_mode = normalize_monitoring_mode(awareness.monitoring_mode)
        awareness.updated_at = utc_now_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(awareness.to_dict(), handle, indent=2, ensure_ascii=False)
        self._record_history(
            awareness=awareness,
            revision_class=revision_class or "session_update",
            source_session_id=source_session_id or (previous.session_id if previous is not None else None),
            provenance={
                "carryover": bool(revision_class == "cross_session_seed"),
            },
        )

    def get_awareness_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.awareness.json"

    def list_history_entries(
        self,
        *,
        session_id: str | None = None,
        revision_class: str | None = None,
    ) -> list[AwarenessHistoryEntry]:
        return self.history_store.list_entries(
            session_id=session_id,
            revision_class=revision_class,
        )

    def consume_recent_history_entries(self) -> list[AwarenessHistoryEntry]:
        entries = list(self._recent_history_entries)
        self._recent_history_entries = []
        return entries

    def _load_existing(self, *, path: Path, session_id: str) -> AwarenessState | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return awareness_state_from_payload(payload=payload, session_id=session_id)

    def _latest_other_state(self, *, session_id: str) -> AwarenessState | None:
        candidates = sorted(
            (
                path
                for path in self.base_dir.glob("*.awareness.json")
                if path.name != f"{session_id}.awareness.json"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            prior_session_id = path.stem.removesuffix(".awareness")
            awareness = self._load_existing(path=path, session_id=prior_session_id)
            if awareness is not None:
                return awareness
        return None

    def _seed_from_previous(
        self,
        *,
        session_id: str,
        previous: AwarenessState,
    ) -> AwarenessState:
        return AwarenessState(
            session_id=session_id,
            monitoring_mode="bounded",
            self_signals=list(previous.self_signals[:5]),
            world_signals=[],
            active_pressures=[],
            candidate_goal_signals=list(previous.candidate_goal_signals[:5]),
            dominant_attention="rebuild monitoring from persisted self and initiative state",
            evidence_refs=[],
            updated_at=utc_now_iso(),
        )

    def _record_history(
        self,
        *,
        awareness: AwarenessState,
        revision_class: str,
        source_session_id: str | None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        entry = AwarenessHistoryEntry(
            entry_id=uuid4().hex,
            timestamp=awareness.updated_at or utc_now_iso(),
            session_id=awareness.session_id,
            source_session_id=source_session_id,
            revision_class=revision_class,
            monitoring_mode=awareness.monitoring_mode,
            self_signals=list(awareness.self_signals),
            world_signals=list(awareness.world_signals),
            active_pressures=list(awareness.active_pressures),
            candidate_goal_signals=list(awareness.candidate_goal_signals),
            dominant_attention=awareness.dominant_attention,
            evidence_refs=list(awareness.evidence_refs),
            provenance=dict(provenance or {}),
        )
        self.history_store.append(entry)
        self._recent_history_entries.append(entry)


def default_awareness_state(*, session_id: str) -> AwarenessState:
    return AwarenessState(
        session_id=session_id,
        monitoring_mode="bounded",
        self_signals=["maintain coherent continuity and bounded initiative awareness"],
        world_signals=[],
        active_pressures=[],
        candidate_goal_signals=[],
        dominant_attention="current interaction and persisted initiative state",
        evidence_refs=[],
        updated_at=utc_now_iso(),
    )


def awareness_state_from_payload(*, payload: dict[str, Any], session_id: str) -> AwarenessState:
    defaults = default_awareness_state(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(AwarenessState)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["session_id"] = session_id
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["monitoring_mode"] = normalize_monitoring_mode(str(merged.get("monitoring_mode", "bounded")))
    merged["self_signals"] = _string_list(merged.get("self_signals"))
    merged["world_signals"] = _string_list(merged.get("world_signals"))
    merged["active_pressures"] = _string_list(merged.get("active_pressures"))
    merged["candidate_goal_signals"] = _string_list(merged.get("candidate_goal_signals"))
    merged["dominant_attention"] = str(merged.get("dominant_attention", "") or "")
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["updated_at"] = str(merged.get("updated_at", ""))
    return AwarenessState(**merged)


def normalize_monitoring_mode(mode: str) -> str:
    normalized = (mode or "bounded").strip().lower()
    if normalized not in AWARENESS_MONITORING_MODES:
        return "bounded"
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
