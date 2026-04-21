"""Tool permission gating for Nova Stage 3.3."""

from __future__ import annotations

from nova.agent.stability import OrientationReadinessReport
from nova.agent.tool_registry import ToolRegistry
from nova.agent.tools import (
    TOOL_ALLOWED,
    TOOL_APPROVAL_REQUIRED,
    TOOL_BLOCKED,
    ToolGateDecision,
    ToolRequest,
)


class ToolGate:
    """Gate tool execution behind permission and orientation readiness."""

    def __init__(self, *, registry: ToolRegistry):
        self.registry = registry

    def evaluate(
        self,
        *,
        request: ToolRequest,
        readiness: OrientationReadinessReport,
        approval_granted: bool = False,
    ) -> ToolGateDecision:
        spec = self.registry.get_spec(request.tool_name)
        if spec is None:
            return ToolGateDecision(
                allowed=False,
                requires_approval=False,
                reason="unknown_tool",
                tool=None,
            )

        if spec.permission == TOOL_BLOCKED:
            return ToolGateDecision(
                allowed=False,
                requires_approval=False,
                reason="tool_blocked",
                tool=spec,
            )

        if not readiness.ready:
            return ToolGateDecision(
                allowed=False,
                requires_approval=False,
                reason="orientation_not_ready",
                tool=spec,
            )

        if spec.permission == TOOL_APPROVAL_REQUIRED and not approval_granted:
            return ToolGateDecision(
                allowed=False,
                requires_approval=True,
                reason="approval_required",
                tool=spec,
            )

        if spec.permission == TOOL_ALLOWED or (
            spec.permission == TOOL_APPROVAL_REQUIRED and approval_granted
        ):
            return ToolGateDecision(
                allowed=True,
                requires_approval=False,
                reason="allowed",
                tool=spec,
            )

        return ToolGateDecision(
            allowed=False,
            requires_approval=False,
            reason="unsupported_permission",
            tool=spec,
        )
