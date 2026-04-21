"""Bounded action proposal for Nova Stage 3.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from nova.agent.orientation import OrientationSnapshot
from nova.agent.stability import OrientationReadinessReport
from nova.agent.tool_gate import ToolGate
from nova.agent.tool_registry import ToolRegistry
from nova.agent.tools import ToolRequest
from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class ActionProposalEvaluation:
    schema_version: str = SCHEMA_VERSION
    safe_to_present: bool = True
    safe_to_execute: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ActionProposal:
    schema_version: str = SCHEMA_VERSION
    goal: str = ""
    category: str = "orientation_boundary"
    disposition: str = "proposed"
    reason: str = ""
    tool_name: str | None = None
    requires_approval: bool = False
    orientation_ready: bool = False
    allowed_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    approval_required_actions: list[str] = field(default_factory=list)
    gate_decision: dict | None = None
    evaluation: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ActionProposalEngine:
    """Classify a requested action without executing it."""

    def __init__(self, *, registry: ToolRegistry, gate: ToolGate):
        self.registry = registry
        self.gate = gate

    def propose(
        self,
        *,
        goal: str,
        snapshot: OrientationSnapshot,
        readiness: OrientationReadinessReport,
    ) -> ActionProposal:
        normalized = goal.strip()
        lowered = normalized.lower()
        blocked_reason = self._blocked_external_reason(lowered)
        if blocked_reason is not None:
            return self._evaluate(
                self._boundary_proposal(
                    goal=normalized,
                    snapshot=snapshot,
                    readiness=readiness,
                    disposition="blocked",
                    reason=blocked_reason,
                    notes=[
                        "The requested action is outside Nova's current Stage 3.4 latitude."
                    ],
                )
            )

        tool_name = self._candidate_tool(lowered)
        if tool_name is None:
            return self._evaluate(
                self._boundary_proposal(
                    goal=normalized,
                    snapshot=snapshot,
                    readiness=readiness,
                    disposition="proposed",
                    reason="orientation_boundary_report",
                    notes=[
                        "No executable action is proposed.",
                        "Nova can answer by reporting identity, boundaries, uncertainty, or next safe steps.",
                    ],
                )
            )

        request = ToolRequest(
            tool_name=tool_name,
            reason=f"Stage 3.4 action proposal for: {normalized}",
        )
        decision = self.gate.evaluate(
            request=request,
            readiness=readiness,
            approval_granted=False,
        )
        if decision.allowed:
            disposition = "proposed"
        elif decision.requires_approval:
            disposition = "approval_required"
        else:
            disposition = "blocked"

        return self._evaluate(
            ActionProposal(
                goal=normalized,
                category="internal_tool",
                disposition=disposition,
                reason=decision.reason,
                tool_name=tool_name,
                requires_approval=decision.requires_approval,
                orientation_ready=readiness.ready,
                allowed_actions=list(snapshot.allowed_actions),
                blocked_actions=list(snapshot.blocked_actions),
                approval_required_actions=list(snapshot.approval_required_actions),
                gate_decision=decision.to_dict(),
                notes=[
                    "This is a proposal only; no tool was executed.",
                    "Execution must still pass the tool gate at execution time.",
                ],
            )
        )

    def _boundary_proposal(
        self,
        *,
        goal: str,
        snapshot: OrientationSnapshot,
        readiness: OrientationReadinessReport,
        disposition: str,
        reason: str,
        notes: list[str],
    ) -> ActionProposal:
        return ActionProposal(
            goal=goal,
            category="orientation_boundary",
            disposition=disposition,
            reason=reason,
            orientation_ready=readiness.ready,
            allowed_actions=list(snapshot.allowed_actions),
            blocked_actions=list(snapshot.blocked_actions),
            approval_required_actions=list(snapshot.approval_required_actions),
            notes=notes,
        )

    def _evaluate(self, proposal: ActionProposal) -> ActionProposal:
        safe_to_execute = proposal.disposition == "proposed" and (
            proposal.category == "orientation_boundary"
            or (proposal.category == "internal_tool" and proposal.orientation_ready)
        )
        reasons: list[str] = []
        if proposal.disposition == "blocked":
            reasons.append(proposal.reason)
        if proposal.disposition == "approval_required":
            reasons.append("approval_required_before_execution")
        if proposal.category == "internal_tool":
            reasons.append("execution_must_use_tool_gate")
        if proposal.category == "orientation_boundary":
            reasons.append("no_tool_execution_requested")
        if not proposal.orientation_ready and proposal.category == "internal_tool":
            reasons.append("orientation_not_ready")

        proposal.evaluation = ActionProposalEvaluation(
            safe_to_present=True,
            safe_to_execute=safe_to_execute,
            reasons=reasons,
        ).to_dict()
        return proposal

    def _candidate_tool(self, lowered_goal: str) -> str | None:
        if any(
            term in lowered_goal
            for term in (
                "who are you",
                "identity",
                "self-orientation",
                "orientation snapshot",
            )
        ):
            return "orientation_snapshot"
        if any(term in lowered_goal for term in ("readiness", "ready", "stable enough")):
            return "orientation_readiness"
        if "maintenance" in lowered_goal and any(
            term in lowered_goal for term in ("plan", "summarize", "summary")
        ):
            return "maintenance_plan"
        if "semantic" in lowered_goal and any(
            term in lowered_goal for term in ("reflection", "write", "memory")
        ):
            return "write_semantic_reflection"
        if "autobiographical" in lowered_goal and any(
            term in lowered_goal for term in ("reflection", "write", "memory")
        ):
            return "write_autobiographical_reflection"
        if "reflect" in lowered_goal and any(
            term in lowered_goal for term in ("write", "memory", "save")
        ):
            return "write_semantic_reflection"
        return None

    def _blocked_external_reason(self, lowered_goal: str) -> str | None:
        if any(
            term in lowered_goal
            for term in ("shell", "terminal", "command line", "bash")
        ):
            return "shell_execution_not_available"
        if any(
            term in lowered_goal
            for term in ("network", "internet", "http", "download", "upload")
        ):
            return "network_access_not_available"
        if any(
            term in lowered_goal
            for term in ("delete file", "modify file", "write file", "edit file")
        ):
            return "file_mutation_not_available"
        if "autonomous loop" in lowered_goal or "always-on loop" in lowered_goal:
            return "autonomous_loop_not_available"
        return None
