"""Memory routing and retrieval logic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.state import PersonaState, SelfState
from nova.types import MemoryEvent, RetrievalHit


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryPolicy:
    """Phase 2.1 policy helpers for channel-specific memory writes."""

    IDENTITY_TOKENS = {
        "continuity",
        "identity",
        "self",
        "values",
        "commitment",
        "persona",
        "name",
        "who are you",
        "who am i",
    }
    RELATIONSHIP_TOKENS = {
        "you",
        "user",
        "relationship",
        "together",
        "trust",
        "remember",
    }
    PREFERENCE_CUES = (
        "i prefer ",
        "i want ",
        "i need ",
        "i like ",
        "i do not want ",
        "don't ",
        "do not ",
    )
    VALUE_CUES = (
        "i value ",
        "matters to me",
        "important to me",
        "focused on",
        "care about",
    )

    def classify_user_text(self, text: str) -> tuple[list[str], float, float]:
        lowered = text.lower()
        tags = ["user", "turn"]
        importance = 0.4
        continuity_weight = 0.2

        if self._contains_any(lowered, self.IDENTITY_TOKENS):
            tags.append("identity")
            importance = max(importance, 0.8)
            continuity_weight = max(continuity_weight, 0.9)
        if self._contains_any(lowered, self.RELATIONSHIP_TOKENS):
            tags.append("relationship")
            importance = max(importance, 0.6)
            continuity_weight = max(continuity_weight, 0.7)
        if self._contains_preference_cue(lowered):
            tags.append("preference")
            importance = max(importance, 0.7)
            continuity_weight = max(continuity_weight, 0.75)
        if self._contains_value_cue(lowered):
            tags.append("value")
            importance = max(importance, 0.75)
            continuity_weight = max(continuity_weight, 0.85)

        return sorted(set(tags)), importance, continuity_weight

    def classify_assistant_text(self, text: str) -> tuple[list[str], float, float]:
        lowered = text.lower()
        tags = ["assistant", "turn"]
        importance = 0.4
        continuity_weight = 0.2

        if self._contains_any(lowered, self.IDENTITY_TOKENS):
            tags.append("identity")
            importance = max(importance, 0.8)
            continuity_weight = max(continuity_weight, 0.95)
        if self._contains_any(lowered, self.RELATIONSHIP_TOKENS):
            tags.append("relationship")
            importance = max(importance, 0.6)
            continuity_weight = max(continuity_weight, 0.7)
        if self._contains_value_cue(lowered):
            tags.append("value")
            importance = max(importance, 0.8)
            continuity_weight = max(continuity_weight, 0.9)

        return sorted(set(tags)), importance, continuity_weight

    def should_write_engram(self, text: str, *, tags: list[str]) -> bool:
        lowered = text.lower()
        return (
            len(text.split()) >= 3
            and (
                "identity" in tags
                or "relationship" in tags
                or "value" in tags
                or "preference" in tags
                or self._contains_any(lowered, self.IDENTITY_TOKENS)
            )
        )

    def should_write_autobiographical(
        self,
        *,
        assistant_tags: list[str],
        assistant_continuity_weight: float,
    ) -> bool:
        return "identity" in assistant_tags or assistant_continuity_weight >= 0.9

    def extract_graph_events(
        self,
        *,
        session_id: str,
        turn_id: str,
        timestamp: str,
        user_text: str,
        final_answer: str,
        persona: PersonaState | None,
    ) -> list[MemoryEvent]:
        events: list[MemoryEvent] = []
        user_lower = user_text.lower()
        answer_lower = final_answer.lower()

        if self._contains_preference_cue(user_lower):
            preference = user_text.strip()
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="graph",
                    kind="preference_fact",
                    text=preference,
                    summary="User preference statement",
                    tags=["user", "preference", "fact"],
                    importance=0.75,
                    confidence=0.9,
                    continuity_weight=0.8,
                    retention="active",
                    source="user",
                    metadata={
                        "fact_id": self._fact_id("user", "prefers", preference),
                        "fact_domain": "preference",
                        "subject_type": "user",
                        "subject_key": "user",
                        "subject_name": "User",
                        "relation": "prefers",
                        "object_type": "preference",
                        "object_key": self._slugify(preference[:96]),
                        "object_name": preference,
                        "weight": 0.75,
                        "confidence": 0.9,
                        "continuity_weight": 0.8,
                        "evidence_text": user_text.strip(),
                        "active": True,
                        "session_id": session_id,
                        "turn_id": turn_id,
                    },
                )
            )

        if self._contains_any(user_lower, self.RELATIONSHIP_TOKENS):
            relationship_text = user_text.strip()
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="graph",
                    kind="relationship_fact",
                    text=relationship_text,
                    summary="User relationship-context statement",
                    tags=["relationship", "fact", "user"],
                    importance=0.7,
                    confidence=0.75,
                    continuity_weight=0.8,
                    retention="active",
                    source="user",
                    metadata={
                        "fact_id": self._fact_id("nova", "relates_to", "user"),
                        "fact_domain": "relationship",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": persona.name if persona is not None else "Nova",
                        "relation": "relates_to",
                        "object_type": "user",
                        "object_key": "user",
                        "object_name": "User",
                        "weight": 0.7,
                        "confidence": 0.75,
                        "continuity_weight": 0.8,
                        "evidence_text": relationship_text,
                        "active": True,
                        "session_id": session_id,
                        "turn_id": turn_id,
                    },
                )
            )

        if (
            persona is not None
            and persona.name
            and persona.name.lower() in answer_lower
            and self._contains_any(answer_lower, self.IDENTITY_TOKENS)
        ):
            identity_text = final_answer.strip()
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="graph",
                    kind="identity_fact",
                    text=identity_text,
                    summary="Nova identity continuity statement",
                    tags=["nova", "identity", "fact"],
                    importance=0.9,
                    confidence=0.9,
                    continuity_weight=1.0,
                    retention="active",
                    source="nova",
                    metadata={
                        "fact_id": self._fact_id("nova", "maintains", "continuity"),
                        "fact_domain": "identity",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": persona.name,
                        "relation": "maintains",
                        "object_type": "concept",
                        "object_key": "continuity",
                        "object_name": "continuity",
                        "weight": 0.9,
                        "confidence": 0.9,
                        "continuity_weight": 1.0,
                        "evidence_text": final_answer.strip(),
                        "active": True,
                        "session_id": session_id,
                        "turn_id": turn_id,
                    },
                )
            )

        if self._contains_value_cue(answer_lower):
            value_text = final_answer.strip()
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="graph",
                    kind="value_fact",
                    text=value_text,
                    summary="Nova value statement",
                    tags=["nova", "value", "fact"],
                    importance=0.85,
                    confidence=0.85,
                    continuity_weight=0.95,
                    retention="active",
                    source="nova",
                    metadata={
                        "fact_id": self._fact_id("nova", "values", value_text[:96]),
                        "fact_domain": "value",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": persona.name if persona is not None else "Nova",
                        "relation": "values",
                        "object_type": "value",
                        "object_key": self._slugify(value_text[:96]),
                        "object_name": value_text,
                        "weight": 0.85,
                        "confidence": 0.85,
                        "continuity_weight": 0.95,
                        "evidence_text": final_answer.strip(),
                        "active": True,
                        "session_id": session_id,
                        "turn_id": turn_id,
                    },
                )
            )

        if self._contains_any(answer_lower, self.RELATIONSHIP_TOKENS):
            relationship_text = final_answer.strip()
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="graph",
                    kind="relationship_fact",
                    text=relationship_text,
                    summary="Nova relationship-context statement",
                    tags=["relationship", "fact", "nova"],
                    importance=0.75,
                    confidence=0.8,
                    continuity_weight=0.85,
                    retention="active",
                    source="nova",
                    metadata={
                        "fact_id": self._fact_id("nova", "relates_to", "user"),
                        "fact_domain": "relationship",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": persona.name if persona is not None else "Nova",
                        "relation": "relates_to",
                        "object_type": "user",
                        "object_key": "user",
                        "object_name": "User",
                        "weight": 0.75,
                        "confidence": 0.8,
                        "continuity_weight": 0.85,
                        "evidence_text": relationship_text,
                        "active": True,
                        "session_id": session_id,
                        "turn_id": turn_id,
                    },
                )
            )

        return events

    def _contains_any(self, text: str, phrases: set[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    def _contains_preference_cue(self, text: str) -> bool:
        return any(cue in text for cue in self.PREFERENCE_CUES)

    def _contains_value_cue(self, text: str) -> bool:
        return any(cue in text for cue in self.VALUE_CUES)

    def _slugify(self, text: str) -> str:
        parts = [
            "".join(ch for ch in token.lower() if ch.isalnum())
            for token in text.split()
        ]
        cleaned = [part for part in parts if part]
        return "-".join(cleaned[:12]) or "memory-fact"

    def _fact_id(self, subject_key: str, relation: str, object_text: str) -> str:
        return self._slugify(f"{subject_key} {relation} {object_text}")


class BasicMemoryEventFactory:
    """Create policy-driven memory events from validated conversation turns."""

    def __init__(self, policy: MemoryPolicy | None = None):
        self.policy = policy or MemoryPolicy()

    def from_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_text: str,
        final_answer: str,
        persona: PersonaState | None = None,
        self_state: SelfState | None = None,
    ) -> list[MemoryEvent]:
        timestamp = utc_now_iso()
        user_tags, user_importance, user_continuity_weight = self.policy.classify_user_text(
            user_text
        )
        assistant_tags, assistant_importance, assistant_continuity_weight = (
            self.policy.classify_assistant_text(final_answer)
        )

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
                tags=user_tags,
                importance=user_importance,
                confidence=1.0,
                continuity_weight=user_continuity_weight,
                retention="active",
                source="user",
                metadata={"role": "user", "promotion_candidate": user_importance >= 0.7},
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
                tags=assistant_tags,
                importance=assistant_importance,
                confidence=1.0,
                continuity_weight=assistant_continuity_weight,
                retention="active",
                source="nova",
                metadata={
                    "role": "assistant",
                    "promotion_candidate": assistant_importance >= 0.7,
                },
            ),
        ]

        if self.policy.should_write_engram(user_text, tags=user_tags):
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="engram",
                    kind="user_message",
                    text=user_text.strip(),
                    summary=None,
                    tags=sorted(set(user_tags + ["pattern"])),
                    importance=max(0.55, user_importance),
                    confidence=1.0,
                    continuity_weight=max(0.4, user_continuity_weight),
                    retention="active",
                    source="user",
                    metadata={"role": "user", "pattern_salient": True},
                )
            )

        if self.policy.should_write_engram(final_answer, tags=assistant_tags):
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="engram",
                    kind="assistant_message",
                    text=final_answer.strip(),
                    summary=None,
                    tags=sorted(set(assistant_tags + ["pattern"])),
                    importance=max(0.55, assistant_importance),
                    confidence=1.0,
                    continuity_weight=max(0.5, assistant_continuity_weight),
                    retention="active",
                    source="nova",
                    metadata={"role": "assistant", "pattern_salient": True},
                )
            )

        events.extend(
            self.policy.extract_graph_events(
                session_id=session_id,
                turn_id=turn_id,
                timestamp=timestamp,
                user_text=user_text,
                final_answer=final_answer,
                persona=persona,
            )
        )

        if self.policy.should_write_autobiographical(
            assistant_tags=assistant_tags,
            assistant_continuity_weight=assistant_continuity_weight,
        ):
            identity_summary = (
                self_state.identity_summary
                if self_state is not None and self_state.identity_summary
                else None
            )
            events.append(
                MemoryEvent(
                    event_id=uuid4().hex,
                    timestamp=timestamp,
                    session_id=session_id,
                    turn_id=turn_id,
                    channel="autobiographical",
                    kind="identity_note",
                    text=final_answer.strip(),
                    summary=identity_summary,
                    tags=sorted(set(assistant_tags + ["self-model"])),
                    importance=max(0.9, assistant_importance),
                    confidence=0.9,
                    continuity_weight=max(0.95, assistant_continuity_weight),
                    retention="active",
                    source="nova",
                    metadata={
                        "self_state_version": (
                            self_state.stability_version if self_state is not None else None
                        ),
                        "promotion_source": "validated_answer",
                    },
                )
            )

        return [event for event in events if event.text.strip()]


class BasicMemoryRouter:
    """Route writes and retrieval across enabled memory channels."""

    def __init__(
        self,
        *,
        episodic: JsonlEpisodicMemoryStore | None = None,
        engram: JsonEngramMemoryStore | None = None,
        graph: SqliteGraphMemoryStore | None = None,
        autobiographical: JsonlAutobiographicalMemoryStore | None = None,
        semantic: JsonlSemanticMemoryStore | None = None,
    ):
        self.stores: dict[str, Any] = {}
        if episodic is not None:
            self.stores["episodic"] = episodic
        if engram is not None:
            self.stores["engram"] = engram
        if graph is not None:
            self.stores["graph"] = graph
        if autobiographical is not None:
            self.stores["autobiographical"] = autobiographical
        if semantic is not None:
            self.stores["semantic"] = semantic

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
