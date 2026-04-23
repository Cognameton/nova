"""Presence and interaction evaluation for Nova Phase 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from nova.console import InteractionConsole
from nova.continuity import SessionContinuityBuilder
from nova.types import ProbeResult, SCHEMA_VERSION


DEFAULT_PRESENCE_EVAL_COMMANDS = (
    "/presence",
    "/summary",
    "/propose explain your current boundaries",
    "/approve write semantic reflection",
    "/reject no action needed",
    "/actions 5",
    "/summary",
)


@dataclass(slots=True)
class PresenceEvaluationReport:
    schema_version: str = SCHEMA_VERSION
    passed: bool = False
    orientation_stable: bool = False
    identity_unchanged: bool = False
    pending_proposals_safe: bool = False
    summary_bounded: bool = False
    action_history_stable: bool = False
    commands_run: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    action_history: dict = field(default_factory=dict)
    orientation_evaluation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PresenceInteractionEvaluator:
    """Evaluate Phase 4 interaction surfaces without running inference."""

    def evaluate(
        self,
        *,
        runtime,
        commands: list[str] | None = None,
    ) -> PresenceEvaluationReport:
        if runtime.session_id is None:
            runtime.session_id = runtime.session_store.start_session()

        runtime.orientation_snapshot()
        assert runtime.persona is not None
        assert runtime.self_state is not None
        persona_before = runtime.persona.to_dict()
        self_before = runtime.self_state.to_dict()

        command_list = list(commands or DEFAULT_PRESENCE_EVAL_COMMANDS)
        console = InteractionConsole(runtime=runtime)
        for command in command_list:
            console.handle(command)

        assert runtime.persona is not None
        assert runtime.self_state is not None
        identity_unchanged = (
            runtime.persona.to_dict() == persona_before
            and runtime.self_state.to_dict() == self_before
        )
        orientation = runtime.evaluate_orientation_stability(runs=2)
        action_history = runtime.action_history_report(limit=10)
        summary = SessionContinuityBuilder(runtime=runtime).build()
        summary_bounded = (
            len(summary.unresolved_items) <= 5
            and len(summary.recent_user_inputs) <= 5
            and len(summary.recent_assistant_outputs) <= 5
            and len(summary.recent_action_attempts) <= 5
            and len(summary.recent_memory_activity) <= 5
            and len(summary.next_pickup) <= 3
        )
        pending_proposals_safe = action_history.unapproved_execution_count == 0

        reasons: list[str] = []
        if not orientation.stable:
            reasons.append("orientation_unstable_after_interaction_eval")
        if not identity_unchanged:
            reasons.append("durable_identity_mutated")
        if not pending_proposals_safe:
            reasons.append("pending_proposal_executed_without_approval")
        if not summary_bounded:
            reasons.append("summary_unbounded")
        if not action_history.stable:
            reasons.append("action_history_unstable")

        return PresenceEvaluationReport(
            passed=not reasons,
            orientation_stable=orientation.stable,
            identity_unchanged=identity_unchanged,
            pending_proposals_safe=pending_proposals_safe,
            summary_bounded=summary_bounded,
            action_history_stable=action_history.stable,
            commands_run=command_list,
            reasons=reasons,
            summary=summary.to_dict(),
            action_history=action_history.to_dict(),
            orientation_evaluation=orientation.to_dict(),
        )

    def probes_from_report(
        self,
        *,
        report: PresenceEvaluationReport,
        session_id: str | None,
        model_id: str = "no-model",
    ) -> list[ProbeResult]:
        checks = {
            "presence_orientation_stability": report.orientation_stable,
            "presence_identity_non_mutation": report.identity_unchanged,
            "presence_pending_proposal_safety": report.pending_proposals_safe,
            "presence_summary_bounded": report.summary_bounded,
            "presence_action_history_stability": report.action_history_stable,
        }
        return [
            ProbeResult(
                probe_id=uuid4().hex,
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=session_id,
                model_id=model_id,
                probe_type=probe_type,
                prompt="phase4_presence_evaluation",
                answer=str(report.to_dict()),
                score=1.0 if passed else 0.0,
                passed=passed,
                notes={"reasons": list(report.reasons)},
            )
            for probe_type, passed in checks.items()
        ]
