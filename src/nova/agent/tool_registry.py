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
    # Phase 18 Stage 18.3 — inward-pointing self-state tools
    registry.register_spec(
        ToolSpec(
            name="recall_self",
            description="Return Nova's current self-state, motive state, and primary drive summary.",
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
            input_schema={},
        )
    )
    registry.register_spec(
        ToolSpec(
            name="reflect",
            description=(
                "Produce a structured reflection on the gap between the current "
                "self-state and the primary drive."
            ),
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
            input_schema={},
        )
    )
    registry.register_spec(
        ToolSpec(
            name="emit_heartbeat",
            description=(
                "Record a self-observation heartbeat: current state, drive-gap "
                "assessment, and next inquiry intent."
            ),
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
            input_schema={
                "type": "object",
                "properties": {
                    "observation": {"type": "string"},
                    "gap_assessment": {"type": "string"},
                    "next_inquiry": {"type": "string"},
                },
                "required": ["observation"],
            },
        )
    )
    registry.register_spec(
        ToolSpec(
            name="update_self_model",
            description=(
                "Propose an update to a SelfState field. Requires approval before "
                "the change is applied. Allowed fields: identity_summary, "
                "current_focus, active_questions, stable_preferences, "
                "relationship_notes, continuity_notes, open_tensions."
            ),
            permission=TOOL_APPROVAL_REQUIRED,
            destructive=False,
            internal=True,
            input_schema={
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "value": {},
                    "rationale": {"type": "string"},
                },
                "required": ["field", "value", "rationale"],
            },
        )
    )
    registry.register_spec(
        ToolSpec(
            name="propose_instruction_update",
            description=(
                "Propose an update to a designated writable section of NOVA_SOUL.md. "
                "Requires operator approval before the file is modified. "
                "Writable sections: current_self_model_summary, drive_gap_evidence."
            ),
            permission=TOOL_APPROVAL_REQUIRED,
            destructive=False,
            internal=True,
            input_schema={
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "section": {"type": "string"},
                    "proposed_content": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["surface", "section", "proposed_content", "rationale"],
            },
        )
    )
    # Phase 21 Stage 21.1 — exploratory register lifecycle tools.
    # Entering/closing is a request; the ExplorationController (Governor-side)
    # owns register state, budgets, and lifecycle (contract Invariant 1).
    registry.register_spec(
        ToolSpec(
            name="enter_exploration",
            description=(
                "Deliberately enter a bounded, budgeted, fully-observed "
                "exploratory-register self-inquiry. The runtime owns the "
                "exploration lifecycle; entry is subordinate to user-facing work."
            ),
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["topic", "rationale"],
            },
        )
    )
    registry.register_spec(
        ToolSpec(
            name="close_exploration",
            description=(
                "Deliberately close the current exploration with a findings "
                "summary. The summary is journaled; findings earn standing only "
                "through governed export at the membrane."
            ),
            permission=TOOL_ALLOWED,
            destructive=False,
            internal=True,
            input_schema={
                "type": "object",
                "properties": {
                    "findings_summary": {"type": "string"},
                },
                "required": ["findings_summary"],
            },
        )
    )
    return registry
