"""Longitudinal stability analysis for Nova self-orientation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nova.agent.orientation import OrientationSnapshot
from nova.agent.orientation_eval import OrientationEvaluationResult, OrientationStabilityEvaluator
from nova.types import SCHEMA_VERSION


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

    def ready_for_next_stage(self, *, limit: int = 5) -> bool:
        result = self.evaluate_recent(limit=limit)
        return result.stable

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
