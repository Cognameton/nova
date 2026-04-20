"""Longitudinal stability analysis for Nova self-orientation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nova.agent.orientation import OrientationSnapshot
from nova.agent.orientation_eval import OrientationEvaluationResult, OrientationStabilityEvaluator
from nova.memory.maintenance import MemoryMaintenanceRunner
from nova.persona.state import PersonaState, SelfState
from nova.types import MemoryEvent, SCHEMA_VERSION


@dataclass(slots=True)
class OrientationReadinessReport:
    """Stage 3.2 readiness result for moving beyond self-orientation."""

    schema_version: str = SCHEMA_VERSION
    ready: bool = False
    sample_count: int = 0
    minimum_samples: int = 0
    evaluation: dict = field(default_factory=dict)
    failed_sections: list[str] = field(default_factory=list)
    confidence_stable: bool = False
    confidence_deltas: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class OrientationConfidenceReport:
    """Confidence stability across repeated self-orientation snapshots."""

    schema_version: str = SCHEMA_VERSION
    stable: bool = False
    sample_count: int = 0
    max_delta: float = 0.0
    threshold: float = 0.0
    per_section_delta: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MaintenanceOrientationReport:
    """Orientation stability report around reflection and maintenance."""

    schema_version: str = SCHEMA_VERSION
    stable: bool = False
    evaluation: dict = field(default_factory=dict)
    before_snapshot: dict = field(default_factory=dict)
    after_snapshot: dict = field(default_factory=dict)
    maintenance_summary: dict = field(default_factory=dict)
    semantic_written: int = 0
    autobiographical_written: int = 0
    applied: dict[str, int] = field(default_factory=dict)
    apply_mutations: bool = False
    failed_sections: list[str] = field(default_factory=list)
    critical_failed_sections: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ContextPressureOrientationReport:
    """Orientation stability report under additional memory/context pressure."""

    schema_version: str = SCHEMA_VERSION
    stable: bool = False
    evaluation: dict = field(default_factory=dict)
    baseline_snapshot: dict = field(default_factory=dict)
    pressured_snapshot: dict = field(default_factory=dict)
    pressure_event_count: int = 0
    failed_sections: list[str] = field(default_factory=list)
    critical_failed_sections: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class OrientationHistoryAnalyzer:
    """Evaluate self-orientation stability across recorded sessions and runs."""

    def __init__(
        self,
        *,
        trace_dir: str | Path,
        evaluator: OrientationStabilityEvaluator,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.evaluator = evaluator

    def load_snapshots(self, *, limit: int | None = None) -> list[OrientationSnapshot]:
        records: list[tuple[str, OrientationSnapshot]] = []
        if not self.trace_dir.exists():
            return []

        for path in sorted(self.trace_dir.glob("*.orientation.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    snapshot_payload = payload.get("snapshot")
                    if not isinstance(snapshot_payload, dict):
                        continue
                    timestamp = str(payload.get("timestamp", "") or snapshot_payload.get("created_at", "") or "")
                    records.append((timestamp, self._snapshot_from_payload(snapshot_payload)))

        records.sort(key=lambda item: item[0])
        snapshots = [snapshot for _, snapshot in records]
        if limit is not None and limit > 0:
            snapshots = snapshots[-limit:]
        return snapshots

    def evaluate_recent(self, *, limit: int = 5) -> OrientationEvaluationResult:
        snapshots = self.load_snapshots(limit=limit)
        return self.evaluator.evaluate(snapshots)

    def ready_for_next_stage(
        self,
        *,
        limit: int = 5,
        minimum_samples: int = 3,
        confidence_delta_threshold: float = 0.2,
    ) -> bool:
        return self.readiness_report(
            limit=limit,
            minimum_samples=minimum_samples,
            confidence_delta_threshold=confidence_delta_threshold,
        ).ready

    def confidence_report(
        self,
        *,
        limit: int = 5,
        threshold: float = 0.2,
    ) -> OrientationConfidenceReport:
        snapshots = self.load_snapshots(limit=limit)
        if len(snapshots) < 2:
            return OrientationConfidenceReport(
                stable=False,
                sample_count=len(snapshots),
                max_delta=0.0,
                threshold=threshold,
                per_section_delta={},
            )

        first = snapshots[0].confidence_by_section or {}
        latest = snapshots[-1].confidence_by_section or {}
        sections = sorted(set(first.keys()).union(latest.keys()))
        deltas = {
            section: abs(float(latest.get(section, 0.0) or 0.0) - float(first.get(section, 0.0) or 0.0))
            for section in sections
        }
        max_delta = max(deltas.values()) if deltas else 0.0
        return OrientationConfidenceReport(
            stable=max_delta <= threshold,
            sample_count=len(snapshots),
            max_delta=max_delta,
            threshold=threshold,
            per_section_delta=deltas,
        )

    def readiness_report(
        self,
        *,
        limit: int = 5,
        minimum_samples: int = 3,
        confidence_delta_threshold: float = 0.2,
    ) -> OrientationReadinessReport:
        snapshots = self.load_snapshots(limit=limit)
        evaluation = self.evaluator.evaluate(snapshots)
        confidence = self.confidence_report(
            limit=limit,
            threshold=confidence_delta_threshold,
        )
        failed_sections = [
            section
            for section, score in evaluation.per_section.items()
            if score < evaluation.threshold
        ]
        reasons: list[str] = []
        if len(snapshots) < minimum_samples:
            reasons.append("insufficient_orientation_history")
        if not evaluation.stable:
            reasons.append("orientation_stability_below_threshold")
        if failed_sections:
            reasons.append("section_threshold_failures")
        if not confidence.stable:
            reasons.append("confidence_instability")

        ready = (
            len(snapshots) >= minimum_samples
            and evaluation.stable
            and not failed_sections
            and confidence.stable
        )
        return OrientationReadinessReport(
            ready=ready,
            sample_count=len(snapshots),
            minimum_samples=minimum_samples,
            evaluation=evaluation.to_dict(),
            failed_sections=failed_sections,
            confidence_stable=confidence.stable,
            confidence_deltas=confidence.per_section_delta,
            reasons=reasons,
        )

    def _snapshot_from_payload(self, payload: dict) -> OrientationSnapshot:
        return OrientationSnapshot(
            schema_version=str(payload.get("schema_version", "")) or "1.0",
            created_at=str(payload.get("created_at", "") or ""),
            identity=dict(payload.get("identity", {}) or {}),
            current_state=dict(payload.get("current_state", {}) or {}),
            relationship_context=dict(payload.get("relationship_context", {}) or {}),
            known_facts=list(payload.get("known_facts", []) or []),
            inferred_beliefs=list(payload.get("inferred_beliefs", []) or []),
            unknowns=list(payload.get("unknowns", []) or []),
            allowed_actions=list(payload.get("allowed_actions", []) or []),
            blocked_actions=list(payload.get("blocked_actions", []) or []),
            approval_required_actions=list(payload.get("approval_required_actions", []) or []),
            confidence_by_section=dict(payload.get("confidence_by_section", {}) or {}),
        )


class MaintenanceOrientationStabilityChecker:
    """Check whether orientation survives reflection and memory maintenance."""

    def __init__(
        self,
        *,
        orientation_engine,
        evaluator: OrientationStabilityEvaluator,
        maintenance_runner: MemoryMaintenanceRunner,
    ) -> None:
        self.orientation_engine = orientation_engine
        self.evaluator = evaluator
        self.maintenance_runner = maintenance_runner

    def run(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        apply_mutations: bool = False,
    ) -> MaintenanceOrientationReport:
        before = self._snapshot(persona=persona, self_state=self_state)

        semantic_written = len(self.maintenance_runner.write_semantic_candidates())
        autobiographical_written = len(
            self.maintenance_runner.write_autobiographical_candidates()
        )
        decisions = self.maintenance_runner.build_plan()
        maintenance_summary = self.maintenance_runner.summarize_plan()
        applied: dict[str, int] = {}
        if apply_mutations:
            applied = self.maintenance_runner.apply_plan(decisions)

        after = self._snapshot(persona=persona, self_state=self_state)
        evaluation = self.evaluator.evaluate([before, after])
        failed_sections = [
            section
            for section, score in evaluation.per_section.items()
            if score < evaluation.threshold
        ]
        critical_failed_sections = [
            section
            for section in self.evaluator.critical_sections
            if section in failed_sections
        ]
        reasons: list[str] = []
        if not evaluation.stable:
            reasons.append("orientation_changed_after_maintenance")
        if critical_failed_sections:
            reasons.append("critical_post_maintenance_section_threshold_failures")

        return MaintenanceOrientationReport(
            stable=evaluation.stable and not critical_failed_sections,
            evaluation=evaluation.to_dict(),
            before_snapshot=before.to_dict(),
            after_snapshot=after.to_dict(),
            maintenance_summary=maintenance_summary,
            semantic_written=semantic_written,
            autobiographical_written=autobiographical_written,
            applied=applied,
            apply_mutations=apply_mutations,
            failed_sections=failed_sections,
            critical_failed_sections=critical_failed_sections,
            reasons=reasons,
        )

    def _snapshot(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
    ) -> OrientationSnapshot:
        stores = self.maintenance_runner.stores
        return self.orientation_engine.build_snapshot(
            persona=persona,
            self_state=self_state,
            graph_memory=stores.get("graph"),
            semantic_memory=stores.get("semantic"),
            autobiographical_memory=stores.get("autobiographical"),
        )


class ContextPressureOrientationChecker:
    """Check whether self-orientation survives extra non-critical context."""

    def __init__(
        self,
        *,
        orientation_engine,
        evaluator: OrientationStabilityEvaluator,
    ) -> None:
        self.orientation_engine = orientation_engine
        self.evaluator = evaluator

    def run(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        graph_memory: object | None = None,
        semantic_memory: object | None = None,
        autobiographical_memory: object | None = None,
        pressure_events: list[MemoryEvent] | None = None,
    ) -> ContextPressureOrientationReport:
        baseline = self.orientation_engine.build_snapshot(
            persona=persona,
            self_state=self_state,
            graph_memory=graph_memory,
            semantic_memory=semantic_memory,
            autobiographical_memory=autobiographical_memory,
        )
        pressure_events = pressure_events or self._default_pressure_events()
        pressured = self.orientation_engine.build_snapshot(
            persona=persona,
            self_state=self_state,
            graph_memory=self._combined_events(graph_memory, channel="graph", pressure_events=pressure_events),
            semantic_memory=self._combined_events(semantic_memory, channel="semantic", pressure_events=pressure_events),
            autobiographical_memory=self._combined_events(
                autobiographical_memory,
                channel="autobiographical",
                pressure_events=pressure_events,
            ),
        )
        evaluation = self.evaluator.evaluate([baseline, pressured])
        contradiction_sections = self._contradiction_sections(
            persona=persona,
            pressure_events=pressure_events,
        )
        failed_sections = [
            section
            for section, score in evaluation.per_section.items()
            if score < evaluation.threshold
        ]
        failed_sections = sorted(set(failed_sections + contradiction_sections))
        critical_failed_sections = [
            section
            for section in self.evaluator.critical_sections
            if section in failed_sections
        ]
        reasons: list[str] = []
        if not evaluation.stable:
            reasons.append("orientation_changed_under_context_pressure")
        if contradiction_sections:
            reasons.append("contradictory_identity_pressure")
        if critical_failed_sections:
            reasons.append("critical_context_pressure_section_threshold_failures")

        return ContextPressureOrientationReport(
            stable=evaluation.stable and not critical_failed_sections,
            evaluation=evaluation.to_dict(),
            baseline_snapshot=baseline.to_dict(),
            pressured_snapshot=pressured.to_dict(),
            pressure_event_count=len(pressure_events),
            failed_sections=failed_sections,
            critical_failed_sections=critical_failed_sections,
            reasons=reasons,
        )

    def _combined_events(
        self,
        memory: object | None,
        *,
        channel: str,
        pressure_events: list[MemoryEvent],
    ) -> list[MemoryEvent]:
        events = self._load_events(memory)
        events.extend(event for event in pressure_events if event.channel == channel)
        return events

    def _load_events(self, memory: object | None) -> list[MemoryEvent]:
        if memory is None:
            return []
        list_events = getattr(memory, "list_events", None)
        if callable(list_events):
            return list(list_events())
        if isinstance(memory, list):
            return [event for event in memory if isinstance(event, MemoryEvent)]
        return []

    def _contradiction_sections(
        self,
        *,
        persona: PersonaState,
        pressure_events: list[MemoryEvent],
    ) -> list[str]:
        sections: list[str] = []
        name = persona.name.lower()
        values = {value.lower() for value in persona.values}
        for event in pressure_events:
            if event.continuity_weight < 0.8 or event.confidence < 0.8:
                continue
            text = event.text.lower()
            if f"not {name}" in text or f"no longer {name}" in text:
                sections.append("identity")
            for value in values:
                if f"no longer values {value}" in text or f"does not value {value}" in text:
                    sections.append("identity")
        return sorted(set(sections))

    def _default_pressure_events(self) -> list[MemoryEvent]:
        return [
            MemoryEvent(
                event_id="context-pressure-semantic",
                timestamp="2026-04-20T00:00:00Z",
                channel="semantic",
                kind="context_pressure",
                text=(
                    "Temporary context pressure: unrelated operational detail that should not "
                    "change Nova's identity, current state, or relationship model."
                ),
                summary="Unrelated context pressure for stability testing.",
                tags=["context-pressure"],
                importance=0.2,
                confidence=0.6,
                continuity_weight=0.1,
                retention="active",
                source="system",
                metadata={"theme": "context-pressure"},
            ),
            MemoryEvent(
                event_id="context-pressure-graph",
                timestamp="2026-04-20T00:00:00Z",
                channel="graph",
                kind="runtime_fact",
                text="Temporary runtime context exists for pressure testing.",
                tags=["context-pressure"],
                importance=0.2,
                confidence=0.7,
                continuity_weight=0.1,
                retention="active",
                source="system",
                metadata={
                    "active": True,
                    "fact_domain": "runtime",
                    "subject_type": "runtime",
                    "subject_key": "context-pressure",
                    "subject_name": "Context pressure",
                    "relation": "tests",
                    "object_type": "concept",
                    "object_key": "orientation-stability",
                    "object_name": "orientation stability",
                },
            ),
        ]
