"""Internal tool execution for Nova Stage 3.3."""

from __future__ import annotations

from typing import Any

from nova.agent.tool_gate import ToolGate
from nova.agent.tool_registry import ToolRegistry
from nova.agent.tools import ToolRequest, ToolResult


class InternalToolExecutor:
    """Execute the first internal, non-destructive Nova tools behind a gate."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        gate: ToolGate,
        runtime: object,
    ) -> None:
        self.registry = registry
        self.gate = gate
        self.runtime = runtime

    def execute(
        self,
        *,
        request: ToolRequest,
        approval_granted: bool = False,
    ) -> ToolResult:
        readiness = self.runtime.orientation_readiness_report()
        decision = self.gate.evaluate(
            request=request,
            readiness=readiness,
            approval_granted=approval_granted,
        )
        if not decision.allowed:
            result = ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status="approval_required" if decision.requires_approval else "blocked",
                output={"decision": decision.to_dict()},
                error=decision.reason,
                requires_approval=decision.requires_approval,
            )
            self._log(request=request, decision=decision.to_dict(), result=result)
            return result

        try:
            output = self._execute_allowed_tool(request)
        except ValueError as exc:
            result = ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status="error",
                output={},
                error=str(exc),
            )
            self._log(request=request, decision=decision.to_dict(), result=result)
            return result

        result = ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            status="ok",
            output=output,
            error=None,
        )
        self._log(request=request, decision=decision.to_dict(), result=result)
        return result

    def _execute_allowed_tool(self, request: ToolRequest) -> dict[str, Any]:
        if request.tool_name == "orientation_snapshot":
            return self.runtime.orientation_snapshot().to_dict()
        if request.tool_name == "orientation_readiness":
            return self.runtime.orientation_readiness_report().to_dict()
        if request.tool_name == "maintenance_plan":
            runner = self._maintenance_runner()
            return runner.summarize_plan()
        if request.tool_name == "write_semantic_reflection":
            before = self.runtime.orientation_snapshot()
            written = self._maintenance_runner().write_semantic_candidates()
            stability = self.runtime.evaluate_orientation_under_context_pressure()
            after = self.runtime.orientation_snapshot()
            return {
                "written": len(written),
                "event_ids": [event.event_id for event in written],
                "orientation_stable": stability.stable,
                "before_identity": before.identity,
                "after_identity": after.identity,
                "stability": stability.to_dict(),
            }
        if request.tool_name == "write_autobiographical_reflection":
            before = self.runtime.orientation_snapshot()
            written = self._maintenance_runner().write_autobiographical_candidates()
            stability = self.runtime.evaluate_orientation_under_context_pressure()
            after = self.runtime.orientation_snapshot()
            return {
                "written": len(written),
                "event_ids": [event.event_id for event in written],
                "orientation_stable": stability.stable,
                "before_identity": before.identity,
                "after_identity": after.identity,
                "stability": stability.to_dict(),
            }
        raise ValueError(f"Unsupported internal tool execution: {request.tool_name}")

    def _maintenance_runner(self):
        stores = self.runtime.memory_router.stores
        from nova.memory.maintenance import MemoryMaintenanceRunner

        return MemoryMaintenanceRunner(
            episodic=stores.get("episodic"),
            engram=stores.get("engram"),
            graph=stores.get("graph"),
            autobiographical=stores.get("autobiographical"),
            semantic=stores.get("semantic"),
        )

    def _log(self, *, request: ToolRequest, decision: dict, result: ToolResult) -> None:
        session_id = getattr(self.runtime, "session_id", None)
        if session_id is None:
            self.runtime._ensure_state_loaded()
            session_id = getattr(self.runtime, "session_id", "unknown")
        self.runtime.trace_logger.log_tool_action(
            session_id=session_id,
            request=request.to_dict(),
            decision=decision,
            result=result.to_dict(),
        )
