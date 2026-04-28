"""Retrieval policy for layered Nova memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from nova.persona.state import SelfState
from nova.types import RetrievalHit


@dataclass(slots=True)
class RetrievalPlan:
    """Per-turn retrieval allocation and ranking policy."""

    top_k_by_channel: dict[str, int]
    channel_order: list[str] = field(default_factory=list)


class IdentityFirstRetrievalPolicy:
    """Identity-first retrieval allocation and reranking for Nova."""

    BASE_BUDGETS = {
        "autobiographical": 3,
        "graph": 4,
        "semantic": 3,
        "episodic": 4,
        "engram": 3,
    }
    CHANNEL_PRIORITY = {
        "autobiographical": 1.35,
        "graph": 1.25,
        "semantic": 1.15,
        "episodic": 1.0,
        "engram": 0.9,
    }
    ORDER = ["autobiographical", "graph", "semantic", "episodic", "engram"]

    def plan(self, *, query: str, self_state: SelfState | None = None) -> RetrievalPlan:
        budgets = dict(self.BASE_BUDGETS)
        lowered = query.lower()

        if self._is_context_light_query(lowered):
            return RetrievalPlan(
                top_k_by_channel={channel: 0 for channel in budgets},
                channel_order=[],
            )

        if self._is_identity_query(lowered):
            budgets["autobiographical"] += 1
            budgets["graph"] += 1
        if self._is_preference_query(lowered):
            budgets["graph"] += 1
            budgets["semantic"] += 1
        if self._is_reflective_query(lowered):
            budgets["semantic"] += 1
            budgets["autobiographical"] += 1

        if self_state is not None and self_state.current_focus:
            focus = self_state.current_focus.lower()
            if "continuity" in focus:
                budgets["autobiographical"] += 1
                budgets["semantic"] += 1

        return RetrievalPlan(top_k_by_channel=budgets, channel_order=list(self.ORDER))

    def rerank_hits(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        ranked = sorted(
            hits,
            key=self._weighted_score,
            reverse=True,
        )
        return ranked

    def _weighted_score(self, hit: RetrievalHit) -> float:
        base = float(hit.score)
        channel_weight = self.CHANNEL_PRIORITY.get(hit.channel, 1.0)
        continuity_weight = float(hit.metadata.get("continuity_weight", 0.0) or 0.0)
        importance = float(hit.metadata.get("importance", 0.0) or 0.0)
        active_boost = 0.15 if hit.metadata.get("active", False) else 0.0
        return (base * channel_weight) + (0.25 * continuity_weight) + (0.1 * importance) + active_boost

    def _is_identity_query(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "who are you",
                "your identity",
                "who am i",
                "continuity",
                "self",
                "persona",
                "name",
            )
        )

    def _is_preference_query(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "prefer",
                "preference",
                "like",
                "want",
                "value",
            )
        )

    def _is_reflective_query(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "remember",
                "reflect",
                "history",
                "relationship",
                "what have we",
                "what do you know",
            )
        )

    def _is_context_light_query(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "exactly two sentences",
                "into two sentences",
                "five short bullets",
                "5 short bullets",
                "what did i just ask you to do",
            )
        )
