"""Typed tool contracts for Nova Stage 3.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from nova.types import SCHEMA_VERSION


TOOL_ALLOWED = "allowed"
TOOL_APPROVAL_REQUIRED = "approval_required"
TOOL_BLOCKED = "blocked"


@dataclass(slots=True)
class ToolSpec:
    schema_version: str = SCHEMA_VERSION
    name: str = ""
    description: str = ""
    permission: str = TOOL_BLOCKED
    destructive: bool = False
    internal: bool = True
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolRequest:
    schema_version: str = SCHEMA_VERSION
    request_id: str = field(default_factory=lambda: uuid4().hex)
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    requested_by: str = "nova"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    schema_version: str = SCHEMA_VERSION
    request_id: str = ""
    tool_name: str = ""
    status: str = "blocked"
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    requires_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolGateDecision:
    schema_version: str = SCHEMA_VERSION
    allowed: bool = False
    requires_approval: bool = False
    reason: str = ""
    tool: ToolSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.tool is not None:
            data["tool"] = self.tool.to_dict()
        return data


class Tool(Protocol):
    spec: ToolSpec

    def execute(self, request: ToolRequest) -> ToolResult: ...
