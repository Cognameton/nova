"""Tool registry for Nova Stage 3.3."""

from __future__ import annotations

from nova.agent.tools import (
    TOOL_ALLOWED,
    TOOL_APPROVAL_REQUIRED,
    TOOL_BLOCKED,
    Tool,
    ToolSpec,
)


class ToolRegistry:
    """Registry of known Nova tools and their permission contracts."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register_spec(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool
        self._specs[tool.spec.name] = tool.spec

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]


def default_tool_registry() -> ToolRegistry:
    """Create the initial non-destructive Stage 3.3 tool registry."""

    registry = ToolRegistry()
    registry.register_spec(
        ToolSpec(
            name="orientation_snapshot",
            description="Return Nova's current self-orientation snapshot.",
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
        )
    )
    registry.register_spec(
        ToolSpec(
            name="orientation_readiness",
            description="Return whether Nova's self-orientation is stable enough for next-stage work.",
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
        )
    )
    registry.register_spec(
        ToolSpec(
            name="maintenance_plan",
            description="Summarize memory maintenance actions without applying mutations.",
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
        )
    )
    registry.register_spec(
        ToolSpec(
            name="write_semantic_reflection",
            description="Write semantic reflection candidates into semantic memory.",
            permission=TOOL_APPROVAL_REQUIRED,
            destructive=False,
            internal=True,
        )
    )
    registry.register_spec(
        ToolSpec(
            name="write_autobiographical_reflection",
            description="Write autobiographical reflection candidates into autobiographical memory.",
            permission=TOOL_APPROVAL_REQUIRED,
            destructive=False,
            internal=True,
        )
    )
    registry.register_spec(
        ToolSpec(
            name="shell",
            description="Shell execution is explicitly blocked in Stage 3.3.",
            permission=TOOL_BLOCKED,
            destructive=True,
            internal=False,
        )
    )
    return registry
