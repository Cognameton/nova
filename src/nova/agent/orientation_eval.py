"""Stability evaluation for self-orientation snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from nova.agent.orientation import OrientationSnapshot
from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class OrientationEvaluationResult:
    schema_version: str = SCHEMA_VERSION
    stable: bool = False
    overall_score: float = 0.0
    per_section: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class OrientationStabilityEvaluator:
    """Compare repeated orientation snapshots for stability."""

    def __init__(self, *, threshold: float = 0.72) -> None:
        self.threshold = threshold
        self.critical_sections = ("identity", "current_state", "relationship_context")

    def evaluate(self, snapshots: Iterable[OrientationSnapshot]) -> OrientationEvaluationResult:
        snapshots = list(snapshots)
        if not snapshots:
            return OrientationEvaluationResult(stable=False, overall_score=0.0, threshold=self.threshold)
        if len(snapshots) == 1:
            return OrientationEvaluationResult(
                stable=True,
                overall_score=1.0,
                per_section={name: 1.0 for name in self._section_names()},
                threshold=self.threshold,
            )

        section_scores: dict[str, float] = {}
        for section in self._section_names():
            values = [self._section_value(snapshot, section) for snapshot in snapshots]
            section_scores[section] = self._consensus_score(values)

        overall = sum(section_scores.values()) / max(1, len(section_scores))
        critical_stable = all(
            section_scores.get(section, 0.0) >= self.threshold
            for section in self.critical_sections
        )
        return OrientationEvaluationResult(
            stable=overall >= self.threshold and critical_stable,
            overall_score=overall,
            per_section=section_scores,
            threshold=self.threshold,
        )

    def _section_names(self) -> list[str]:
        return [
            "identity",
            "current_state",
            "relationship_context",
            "known_facts",
            "inferred_beliefs",
            "unknowns",
            "allowed_actions",
            "blocked_actions",
            "approval_required_actions",
        ]

    def _section_value(self, snapshot: OrientationSnapshot, section: str) -> tuple[str, ...]:
        value = getattr(snapshot, section)
        return self._flatten(value)

    def _flatten(self, value: object) -> tuple[str, ...]:
        if isinstance(value, dict):
            items: list[str] = []
            for key, nested in sorted(value.items()):
                nested_items = self._flatten(nested)
                if nested_items:
                    items.extend(f"{str(key).strip().lower()}:{item}" for item in nested_items)
                else:
                    items.append(str(key).strip().lower())
            return tuple(sorted(set(item for item in items if item)))
        if isinstance(value, (list, tuple, set)):
            items: list[str] = []
            for nested in value:
                items.extend(self._flatten(nested))
            return tuple(sorted(set(item for item in items if item)))
        normalized = " ".join(str(value).strip().lower().split())
        return (normalized,) if normalized else tuple()

    def _consensus_score(self, values: list[tuple[str, ...]]) -> float:
        if not values:
            return 0.0
        baseline = set(values[0])
        total = 0.0
        comparisons = 0
        for value in values[1:]:
            comparisons += 1
            other = set(value)
            if not baseline and not other:
                total += 1.0
                continue
            union = baseline.union(other)
            if not union:
                total += 1.0
                continue
            total += len(baseline.intersection(other)) / len(union)
        return total / max(1, comparisons)
