"""Memory maintenance policy and planning helpers for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nova.memory.reflection import ReflectionEngine
from nova.types import MemoryEvent


@dataclass(slots=True)
class MaintenanceDecision:
    """One maintenance recommendation for a memory item."""

    event_id: str
    channel: str
    action: str
    reason: str
    priority: float
    target_retention: str
    consolidation_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryMaintenancePlanner:
    """Score memory items for keep/demote/archive/prune/consolidate actions."""

    def plan_for_events(self, events: list[MemoryEvent]) -> list[MaintenanceDecision]:
        decisions = [self.assess_event(event) for event in events]
        decisions.sort(key=lambda item: item.priority, reverse=True)
        return decisions

    def assess_event(self, event: MemoryEvent) -> MaintenanceDecision:
        age_days = self._age_days(event.timestamp)
        hit_count = int((event.metadata or {}).get("hit_count", 0) or 0)
        redundancy = float((event.metadata or {}).get("redundancy", 0.0) or 0.0)
        promoted = bool((event.metadata or {}).get("promoted", False))
        active = bool((event.metadata or {}).get("active", True))

        if event.channel == "autobiographical":
            return self._autobiographical_decision(event, age_days=age_days, redundancy=redundancy)
        if event.channel == "graph":
            return self._graph_decision(event, age_days=age_days, active=active)
        if event.channel == "engram":
            return self._engram_decision(event, age_days=age_days, hit_count=hit_count)
        if event.channel == "episodic":
            return self._episodic_decision(
                event,
                age_days=age_days,
                redundancy=redundancy,
                promoted=promoted,
            )
        if event.channel == "semantic":
            return self._semantic_decision(event, age_days=age_days, redundancy=redundancy)

        return MaintenanceDecision(
            event_id=event.event_id,
            channel=event.channel,
            action="keep",
            reason="unknown-channel-default",
            priority=0.1,
            target_retention=event.retention,
            metadata={"age_days": age_days},
        )

    def _episodic_decision(
        self,
        event: MemoryEvent,
        *,
        age_days: float,
        redundancy: float,
        promoted: bool,
    ) -> MaintenanceDecision:
        if event.continuity_weight >= 0.8 or event.importance >= 0.8:
            return self._decision(event, "keep", "continuity-bearing-episode", 0.95, "active", age_days)
        if promoted and age_days >= 7:
            return self._decision(
                event,
                "demote",
                "promoted-to-higher-layer",
                0.75,
                "demoted",
                age_days,
                consolidation_key=self._cluster_key(event),
            )
        if redundancy >= 0.8 and age_days >= 14:
            return self._decision(
                event,
                "archive",
                "redundant-episodic-cluster",
                0.65,
                "archived",
                age_days,
                consolidation_key=self._cluster_key(event),
            )
        if event.importance <= 0.2 and event.continuity_weight <= 0.2 and age_days >= 30:
            return self._decision(event, "prune", "low-signal-stale-episode", 0.8, "pruned", age_days)
        return self._decision(event, "keep", "episodic-default", 0.2, event.retention, age_days)

    def _engram_decision(
        self,
        event: MemoryEvent,
        *,
        age_days: float,
        hit_count: int,
    ) -> MaintenanceDecision:
        if event.continuity_weight >= 0.8 or "identity" in event.tags:
            return self._decision(event, "keep", "identity-pattern", 0.9, "active", age_days)
        if hit_count >= 3:
            return self._decision(event, "keep", "reused-pattern", 0.6, "active", age_days)
        if age_days >= 21 and hit_count == 0 and event.importance <= 0.4:
            return self._decision(event, "prune", "stale-low-value-pattern", 0.85, "pruned", age_days)
        if age_days >= 10 and hit_count <= 1:
            return self._decision(event, "demote", "weak-pattern-signal", 0.55, "demoted", age_days)
        return self._decision(event, "keep", "engram-default", 0.2, event.retention, age_days)

    def _graph_decision(
        self,
        event: MemoryEvent,
        *,
        age_days: float,
        active: bool,
    ) -> MaintenanceDecision:
        if active and (event.continuity_weight >= 0.8 or event.confidence >= 0.85):
            return self._decision(event, "keep", "active-stable-fact", 0.9, "active", age_days)
        if not active and event.confidence >= 0.5:
            return self._decision(event, "archive", "historical-fact", 0.7, "archived", age_days)
        if not active and event.confidence < 0.5:
            return self._decision(event, "prune", "weak-historical-fact", 0.8, "pruned", age_days)
        if active and age_days >= 30 and event.confidence < 0.6:
            return self._decision(event, "demote", "aging-low-confidence-fact", 0.65, "demoted", age_days)
        return self._decision(event, "keep", "graph-default", 0.25, event.retention, age_days)

    def _semantic_decision(
        self,
        event: MemoryEvent,
        *,
        age_days: float,
        redundancy: float,
    ) -> MaintenanceDecision:
        if redundancy >= 0.7:
            return self._decision(
                event,
                "consolidate",
                "overlapping-semantic-summary",
                0.85,
                event.retention,
                age_days,
                consolidation_key=self._cluster_key(event),
            )
        if age_days >= 60 and event.confidence < 0.6:
            return self._decision(event, "archive", "stale-semantic-summary", 0.6, "archived", age_days)
        return self._decision(event, "keep", "semantic-default", 0.3, event.retention, age_days)

    def _autobiographical_decision(
        self,
        event: MemoryEvent,
        *,
        age_days: float,
        redundancy: float,
    ) -> MaintenanceDecision:
        if redundancy >= 0.75:
            return self._decision(
                event,
                "consolidate",
                "overlapping-continuity-note",
                0.9,
                "active",
                age_days,
                consolidation_key=self._cluster_key(event),
            )
        if event.continuity_weight >= 0.9 or event.importance >= 0.85:
            return self._decision(event, "keep", "core-continuity-memory", 1.0, "active", age_days)
        if age_days >= 90 and event.importance < 0.5:
            return self._decision(event, "archive", "low-signal-autobiographical-note", 0.5, "archived", age_days)
        return self._decision(event, "keep", "autobiographical-default", 0.35, event.retention, age_days)

    def _decision(
        self,
        event: MemoryEvent,
        action: str,
        reason: str,
        priority: float,
        target_retention: str,
        age_days: float,
        *,
        consolidation_key: str | None = None,
    ) -> MaintenanceDecision:
        return MaintenanceDecision(
            event_id=event.event_id,
            channel=event.channel,
            action=action,
            reason=reason,
            priority=priority,
            target_retention=target_retention,
            consolidation_key=consolidation_key,
            metadata={
                "age_days": age_days,
                "importance": event.importance,
                "confidence": event.confidence,
                "continuity_weight": event.continuity_weight,
            },
        )

    def _cluster_key(self, event: MemoryEvent) -> str:
        metadata = event.metadata or {}
        if event.channel == "graph":
            return str(metadata.get("fact_id") or event.kind or event.event_id)
        if event.channel == "semantic":
            return str(metadata.get("theme") or event.kind or event.event_id)
        if event.channel == "autobiographical":
            return str(metadata.get("self_state_version") or event.kind or event.event_id)
        return str(event.kind or event.event_id)

    def _age_days(self, timestamp: str) -> float:
        if not timestamp:
            return 0.0
        try:
            normalized = timestamp.replace("Z", "+00:00")
            then = datetime.fromisoformat(normalized)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)
        except ValueError:
            return 0.0


class MemoryMaintenanceRunner:
    """Non-destructive maintenance runner over the currently persisted stores."""

    def __init__(
        self,
        *,
        episodic: Any | None = None,
        engram: Any | None = None,
        graph: Any | None = None,
        autobiographical: Any | None = None,
        semantic: Any | None = None,
        planner: MemoryMaintenancePlanner | None = None,
        reflection_engine: ReflectionEngine | None = None,
    ):
        self.stores = {
            "episodic": episodic,
            "engram": engram,
            "graph": graph,
            "autobiographical": autobiographical,
            "semantic": semantic,
        }
        self.planner = planner or MemoryMaintenancePlanner()
        self.reflection_engine = reflection_engine or ReflectionEngine()

    def collect_events(self) -> list[MemoryEvent]:
        events: list[MemoryEvent] = []
        for store in self.stores.values():
            if store is None or not hasattr(store, "list_events"):
                continue
            events.extend(store.list_events())
        return events

    def build_plan(self) -> list[MaintenanceDecision]:
        return self.planner.plan_for_events(self.collect_events())

    def summarize_plan(self) -> dict[str, Any]:
        decisions = self.build_plan()
        summary: dict[str, Any] = {
            "total_events": len(decisions),
            "by_action": {},
            "by_channel": {},
        }
        for decision in decisions:
            summary["by_action"][decision.action] = summary["by_action"].get(decision.action, 0) + 1
            summary["by_channel"][decision.channel] = summary["by_channel"].get(decision.channel, 0) + 1
        return summary

    def build_semantic_candidates(self) -> list[MemoryEvent]:
        episodic_store = self.stores.get("episodic")
        if episodic_store is None or not hasattr(episodic_store, "list_events"):
            return []
        return self.reflection_engine.build_semantic_candidates(episodic_store.list_events())

    def write_semantic_candidates(self) -> list[MemoryEvent]:
        semantic_store = self.stores.get("semantic")
        candidates = self.build_semantic_candidates()
        if semantic_store is None or not hasattr(semantic_store, "add"):
            return candidates
        written: list[MemoryEvent] = []
        for candidate in candidates:
            if hasattr(semantic_store, "merge_reflection_candidate"):
                written.append(semantic_store.merge_reflection_candidate(candidate))
            else:
                semantic_store.add(candidate)
                written.append(candidate)
        return written

    def build_autobiographical_candidates(self) -> list[MemoryEvent]:
        episodic_store = self.stores.get("episodic")
        if episodic_store is None or not hasattr(episodic_store, "list_events"):
            return []
        semantic_store = self.stores.get("semantic")
        semantic_events = (
            semantic_store.list_events()
            if semantic_store is not None and hasattr(semantic_store, "list_events")
            else []
        )
        return self.reflection_engine.build_autobiographical_candidates(
            episodic_events=episodic_store.list_events(),
            semantic_events=semantic_events,
        )

    def write_autobiographical_candidates(self) -> list[MemoryEvent]:
        autobiographical_store = self.stores.get("autobiographical")
        candidates = self.build_autobiographical_candidates()
        if autobiographical_store is None or not hasattr(autobiographical_store, "add"):
            return candidates
        written: list[MemoryEvent] = []
        for candidate in candidates:
            if hasattr(autobiographical_store, "merge_reflection_candidate"):
                written.append(autobiographical_store.merge_reflection_candidate(candidate))
            else:
                autobiographical_store.add(candidate)
                written.append(candidate)
        return written

    def apply_plan(
        self,
        decisions: list[MaintenanceDecision] | None = None,
        *,
        actions: set[str] | None = None,
    ) -> dict[str, int]:
        selected_actions = actions or {"demote", "archive", "prune"}
        decisions = decisions or self.build_plan()
        by_channel: dict[str, list[MaintenanceDecision]] = {}
        for decision in decisions:
            if decision.action not in selected_actions:
                continue
            by_channel.setdefault(decision.channel, []).append(decision)

        results: dict[str, int] = {}
        for channel, channel_decisions in by_channel.items():
            store = self.stores.get(channel)
            if store is None or not hasattr(store, "apply_maintenance_decisions"):
                results[channel] = 0
                continue
            results[channel] = int(store.apply_maintenance_decisions(channel_decisions))
        return results
