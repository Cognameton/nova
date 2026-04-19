"""Operational boundary definitions for Stage 3 self-orientation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class OperationalLatitude:
    schema_version: str = SCHEMA_VERSION
    allowed_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    approval_required_actions: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class BoundaryPolicy:
    """Stage 3.1 operational latitude policy for Nova."""

    def build(self) -> OperationalLatitude:
        return OperationalLatitude(
            allowed_actions=[
                "self-orientation reporting",
                "memory recall and summarization",
                "uncertainty reporting",
                "reflection planning without execution",
            ],
            blocked_actions=[
                "external tool execution",
                "file modification",
                "shell command execution",
                "network access",
            ],
            approval_required_actions=[
                "maintenance mutation application",
                "future tool execution",
                "future event-loop autonomy",
            ],
            confidence=1.0,
        )
