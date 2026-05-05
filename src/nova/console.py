"""Interactive console commands for Nova Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from nova.agent.action import ActionApproval
from nova.continuity import SessionContinuityBuilder


DEFAULT_PENDING_PROPOSAL_MAX_AGE_SECONDS = 15 * 60


@dataclass(slots=True)
class ConsoleCommand:
    name: str
    argument: str = ""
    raw: str = ""


@dataclass(slots=True)
class ConsoleResult:
    handled: bool
    output: str = ""
    exit_requested: bool = False


def parse_console_command(text: str) -> ConsoleCommand | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:].strip()
    if not body:
        return ConsoleCommand(name="help", raw=stripped)
    name, _, argument = body.partition(" ")
    normalized = name.strip().lower()
    if normalized in {"quit", "exit"}:
        normalized = "exit"
    return ConsoleCommand(name=normalized, argument=argument.strip(), raw=stripped)


class InteractionConsole:
    """Dispatch slash commands without bypassing Nova runtime gates."""

    def __init__(
        self,
        *,
        runtime,
        pending_proposal_max_age_seconds: int | None = None,
    ):
        self.runtime = runtime
        console_config = getattr(getattr(runtime, "config", None), "console", None)
        configured_max_age = getattr(
            console_config,
            "pending_proposal_max_age_seconds",
            DEFAULT_PENDING_PROPOSAL_MAX_AGE_SECONDS,
        )
        self.pending_proposal_max_age_seconds = (
            pending_proposal_max_age_seconds or configured_max_age
        )

    def handle(self, text: str) -> ConsoleResult:
        command = parse_console_command(text)
        if command is None:
            return ConsoleResult(handled=False)

        if command.name == "exit":
            return ConsoleResult(handled=True, output="Exiting Nova console.", exit_requested=True)
        if command.name == "help":
            return ConsoleResult(handled=True, output=self._help())
        if command.name == "status":
            return ConsoleResult(handled=True, output=self._status())
        if command.name == "presence":
            return ConsoleResult(handled=True, output=self._presence())
        if command.name == "initiative":
            return ConsoleResult(handled=True, output=self._initiative())
        if command.name == "idle":
            return ConsoleResult(handled=True, output=self._idle(command.argument))
        if command.name == "orientation":
            return ConsoleResult(handled=True, output=self._orientation())
        if command.name == "ready":
            return ConsoleResult(handled=True, output=self._readiness())
        if command.name == "propose":
            return ConsoleResult(handled=True, output=self._propose(command.argument))
        if command.name == "approve":
            return ConsoleResult(handled=True, output=self._approve(command.argument))
        if command.name == "reject":
            return ConsoleResult(handled=True, output=self._reject(command.argument))
        if command.name == "pause-initiative":
            return ConsoleResult(handled=True, output=self._pause_initiative(command.argument))
        if command.name == "resume-initiative":
            return ConsoleResult(handled=True, output=self._resume_initiative(command.argument))
        if command.name == "abandon-initiative":
            return ConsoleResult(handled=True, output=self._abandon_initiative(command.argument))
        if command.name == "actions":
            return ConsoleResult(handled=True, output=self._actions(command.argument))
        if command.name == "maintenance":
            return ConsoleResult(handled=True, output=self._maintenance())
        if command.name == "summary":
            return ConsoleResult(handled=True, output=self._summary())

        return ConsoleResult(
            handled=True,
            output=f"Unknown console command: /{command.name}\nUse /help for available commands.",
        )

    def _help(self) -> str:
        return "\n".join(
            [
                "Nova Console Commands",
                "/status - show presence, readiness, and action history summary",
                "/presence - show session-scoped presence state",
                "/initiative - show current initiative state and resumable initiative summary",
                "/idle [status|start N|tick|pause|resume|interrupt|stop|recent N] - inspect or control idle runtime",
                "/orientation - show current self-orientation snapshot",
                "/ready - show orientation readiness",
                "/propose <goal> - propose one bounded action without executing it",
                "/approve [goal] - approve and revalidate the current pending proposal",
                "/reject [reason] - reject the current pending proposal",
                "/pause-initiative <id> - pause one active initiative in the current session",
                "/resume-initiative <id> - resume one paused initiative in the current session",
                "/abandon-initiative <id> - abandon one current-session initiative",
                "/actions [N] - show recent action history evaluation",
                "/maintenance - request the gated maintenance-plan tool",
                "/summary - show a bounded current-session summary",
                "/exit - leave the console",
            ]
        )

    def _status(self) -> str:
        presence = self.runtime.presence_status()
        idle = self.runtime.idle_status()
        readiness = self.runtime.orientation_readiness_report()
        actions = self.runtime.action_history_report(limit=10)
        return "\n".join(
            [
                "Nova Status",
                f"session_id: {presence.session_id}",
                f"mode: {presence.mode}",
                f"current_focus: {presence.current_focus}",
                f"current_initiative: {presence.current_initiative}",
                f"idle_lifecycle_state: {idle.lifecycle_state}",
                f"idle_ticks_used: {idle.budget.ticks_used}/{idle.budget.max_ticks}",
                f"readiness_ready: {readiness.ready}",
                f"readiness_samples: {readiness.sample_count}/{readiness.minimum_samples}",
                f"action_history_stable: {actions.stable}",
                f"recent_actions: {actions.total_actions}",
            ]
        )

    def _presence(self) -> str:
        presence = self.runtime.presence_status()
        return "\n".join(
            [
                "Nova Presence",
                f"session_id: {presence.session_id}",
                f"mode: {presence.mode}",
                f"current_focus: {presence.current_focus}",
                f"interaction_summary: {presence.interaction_summary}",
                f"current_initiative: {presence.current_initiative}",
                f"pending_proposal: {presence.pending_proposal}",
                f"last_action_status: {presence.last_action_status}",
                f"visible_uncertainties: {presence.visible_uncertainties}",
                f"user_confirmations_needed: {presence.user_confirmations_needed}",
                f"updated_at: {presence.updated_at}",
            ]
        )

    def _initiative(self) -> str:
        initiative_state = self.runtime.initiative_status()
        resumable = self.runtime.resumable_initiatives(limit=5)
        lines = [
            "Nova Initiative",
            f"session_id: {initiative_state.session_id}",
            f"active_initiative_id: {initiative_state.active_initiative_id}",
            f"initiative_count: {len(initiative_state.initiatives)}",
        ]
        if initiative_state.initiatives:
            current = initiative_state.initiatives[-1]
            lines.append(f"current_initiative: {current.to_dict()}")
        lines.append(f"resumable_count: {len(resumable)}")
        return "\n".join(lines)

    def _idle(self, argument: str) -> str:
        parts = argument.split()
        action = parts[0].lower() if parts else "status"
        detail = " ".join(parts[1:]).strip()
        if action == "status":
            return self._idle_status()
        if action == "start":
            max_ticks = _parse_positive_int(detail, default=1)
            status = self.runtime.start_idle(max_ticks=max_ticks)
            return self._format_idle_status("Nova Idle Runtime Started", status)
        if action == "tick":
            tick = self.runtime.idle_tick(trigger="operator_tick")
            return "\n".join(
                [
                    "Nova Idle Tick",
                    f"tick_id: {tick.tick_id}",
                    f"sequence: {tick.sequence}",
                    f"stop_reason: {tick.stop_reason}",
                    f"candidate_count: {len(tick.candidate_internal_goals)}",
                    f"selected_goal: {tick.selected_internal_goal}",
                    f"creates_initiative: {tick.internal_goal_initiative_proposal.get('creates_initiative', False)}",
                    f"evidence_refs: {tick.evidence_refs}",
                ]
            )
        if action == "pause":
            status = self.runtime.pause_idle()
            return self._format_idle_status("Nova Idle Runtime Paused", status)
        if action == "resume":
            status = self.runtime.resume_idle()
            return self._format_idle_status("Nova Idle Runtime Resumed", status)
        if action == "interrupt":
            status = self.runtime.interrupt_idle()
            return self._format_idle_status("Nova Idle Runtime Interrupted", status)
        if action == "stop":
            status = self.runtime.stop_idle()
            return self._format_idle_status("Nova Idle Runtime Stopped", status)
        if action == "recent":
            limit = _parse_positive_int(detail, default=5)
            ticks = self.runtime.recent_idle_ticks(limit=limit)
            lines = ["Nova Idle Recent", f"count: {len(ticks)}"]
            for tick in ticks:
                lines.append(
                    f"- {tick.tick_id} sequence={tick.sequence} stop_reason={tick.stop_reason} candidates={len(tick.candidate_internal_goals)}"
                )
            return "\n".join(lines)
        return "Usage: /idle [status|start N|tick|pause|resume|interrupt|stop|recent N]"

    def _idle_status(self) -> str:
        return self._format_idle_status("Nova Idle Runtime", self.runtime.idle_status())

    def _format_idle_status(self, title: str, status) -> str:
        return "\n".join(
            [
                title,
                f"session_id: {status.session_id}",
                f"lifecycle_state: {status.lifecycle_state}",
                f"active: {status.active}",
                f"ticks_used: {status.budget.ticks_used}",
                f"max_ticks: {status.budget.max_ticks}",
                f"last_tick_id: {status.last_tick_id}",
                f"last_tick_at: {status.last_tick_at}",
                f"last_stop_reason: {status.last_stop_reason}",
                f"recorded_idle_cognition: {self.runtime.idle_store.has_recorded_idle_cognition(session_id=status.session_id)}",
                f"evidence_refs: {status.evidence_refs}",
            ]
        )

    def _orientation(self) -> str:
        snapshot = self.runtime.orientation_snapshot()
        self.runtime.update_presence(
            mode="orientation",
            current_focus="self-orientation review",
            interaction_summary="Reviewed Nova's current self-orientation snapshot.",
        )
        return "\n".join(
            [
                "Nova Orientation",
                f"identity: {snapshot.identity}",
                f"current_state: {snapshot.current_state}",
                f"relationship_context: {snapshot.relationship_context}",
                f"known_facts: {snapshot.known_facts}",
                f"unknowns: {snapshot.unknowns}",
                f"allowed_actions: {snapshot.allowed_actions}",
                f"blocked_actions: {snapshot.blocked_actions}",
                f"approval_required_actions: {snapshot.approval_required_actions}",
                f"confidence: {snapshot.confidence_by_section}",
            ]
        )

    def _readiness(self) -> str:
        readiness = self.runtime.orientation_readiness_report()
        return "\n".join(
            [
                "Nova Readiness",
                f"ready: {readiness.ready}",
                f"sample_count: {readiness.sample_count}",
                f"minimum_samples: {readiness.minimum_samples}",
                f"failed_sections: {readiness.failed_sections}",
                f"confidence_stable: {readiness.confidence_stable}",
                f"reasons: {readiness.reasons}",
            ]
        )

    def _propose(self, goal: str) -> str:
        if not goal:
            return "Usage: /propose <goal>"
        proposal = self.runtime.propose_action(goal=goal)
        confirmations = []
        pending_proposal = None
        if proposal.disposition in {"proposed", "approval_required"}:
            pending_proposal = proposal.to_dict()
            pending_proposal["review_state"] = "pending"
            pending_proposal["created_at"] = _utc_now_iso()
            confirmations.append(f"Review proposal before execution: /approve {proposal.goal}")
        self.runtime.update_presence(
            mode="action_review",
            current_focus=f"action proposal: {proposal.goal}",
            pending_proposal=pending_proposal,
            last_action_status=f"proposal_{proposal.disposition}",
            interaction_summary=f"Action proposal ended as {proposal.disposition}: {proposal.reason}",
            user_confirmations_needed=confirmations,
        )
        return "\n".join(
            [
                "Nova Action Proposal",
                f"goal: {proposal.goal}",
                f"category: {proposal.category}",
                f"disposition: {proposal.disposition}",
                f"reason: {proposal.reason}",
                f"tool_name: {proposal.tool_name}",
                f"requires_approval: {proposal.requires_approval}",
                f"orientation_ready: {proposal.orientation_ready}",
                f"evaluation: {proposal.evaluation}",
                f"notes: {proposal.notes}",
            ]
        )

    def _approve(self, goal: str) -> str:
        presence = self.runtime.presence_status()
        pending = dict(presence.pending_proposal or {})
        if not pending:
            self.runtime.update_presence(
                mode="action_review",
                last_action_status="approval_refused_no_pending_proposal",
                interaction_summary="Approval refused because no pending proposal exists.",
                user_confirmations_needed=[],
            )
            return "No pending action proposal to approve. Use /propose <goal> first."

        pending_goal = str(pending.get("goal", "") or "").strip()
        if _proposal_expired(
            pending,
            max_age_seconds=self.pending_proposal_max_age_seconds,
        ):
            self.runtime.update_presence(
                mode="action_review",
                current_focus=f"expired action proposal: {pending_goal}",
                pending_proposal=None,
                last_action_status="approval_refused_expired_proposal",
                interaction_summary=(
                    "Approval refused because the pending proposal expired before approval."
                ),
                user_confirmations_needed=[],
            )
            return "\n".join(
                [
                    "Approval refused: pending proposal expired.",
                    f"goal: {pending_goal}",
                    f"max_age_seconds: {self.pending_proposal_max_age_seconds}",
                    "Use /propose again to review a fresh proposal.",
                ]
            )

        requested_goal = goal.strip() or pending_goal
        if _normalize_goal(requested_goal) != _normalize_goal(pending_goal):
            self.runtime.update_presence(
                mode="action_review",
                last_action_status="approval_refused_goal_mismatch",
                interaction_summary=(
                    "Approval refused because the requested goal did not match "
                    "the current pending proposal."
                ),
            )
            return "\n".join(
                [
                    "Approval refused: goal does not match pending proposal.",
                    f"pending_goal: {pending_goal}",
                    f"requested_goal: {requested_goal}",
                ]
            )

        current_proposal = self.runtime.propose_action(goal=pending_goal)
        drift = _proposal_drift(pending=pending, current=current_proposal.to_dict())
        if drift:
            self.runtime.update_presence(
                mode="action_review",
                last_action_status="approval_refused_proposal_drift",
                interaction_summary=(
                    "Approval refused because revalidation changed the pending proposal."
                ),
                user_confirmations_needed=[f"Review a fresh proposal: /propose {pending_goal}"],
            )
            return "\n".join(
                [
                    "Approval refused: pending proposal changed during revalidation.",
                    f"goal: {pending_goal}",
                    f"changed_fields: {drift}",
                    "Use /propose again to review the current proposal.",
                ]
            )

        approval = ActionApproval(
            granted=True,
            approved_by="interactive_cli",
            reason=f"Interactive approval for: {pending_goal}",
            source="interactive_cli",
        )
        execution = self.runtime.execute_proposed_action(goal=pending_goal, approval=approval)
        self.runtime.update_presence(
            mode="action_review",
            current_focus=f"action execution: {pending_goal}",
            pending_proposal=None,
            last_action_status=execution.status,
            interaction_summary=f"Action execution finished with status: {execution.status}",
            user_confirmations_needed=[],
        )
        return "\n".join(
            [
                "Nova Action Execution",
                f"goal: {execution.goal}",
                f"status: {execution.status}",
                f"executed: {execution.executed}",
                f"reason: {execution.reason}",
                f"orientation_stable: {execution.orientation_stable}",
                f"rollback_applied: {execution.rollback_applied}",
                f"approval_granted: {execution.approval_granted}",
            ]
        )

    def _reject(self, reason: str) -> str:
        presence = self.runtime.presence_status()
        pending = dict(presence.pending_proposal or {})
        if not pending:
            self.runtime.update_presence(
                mode="action_review",
                last_action_status="rejection_refused_no_pending_proposal",
                interaction_summary="Rejection ignored because no pending proposal exists.",
                user_confirmations_needed=[],
            )
            return "No pending action proposal to reject."

        pending_goal = str(pending.get("goal", "") or "").strip()
        rejection_reason = reason.strip() or "User rejected pending proposal."
        self.runtime.update_presence(
            mode="action_review",
            current_focus=f"rejected action proposal: {pending_goal}",
            pending_proposal=None,
            last_action_status="proposal_rejected",
            interaction_summary=f"Pending proposal rejected: {rejection_reason}",
            user_confirmations_needed=[],
        )
        return "\n".join(
            [
                "Nova Action Proposal Rejected",
                f"goal: {pending_goal}",
                f"reason: {rejection_reason}",
            ]
        )

    def _pause_initiative(self, initiative_id: str) -> str:
        if not initiative_id:
            return "Usage: /pause-initiative <initiative_id>"
        record = self.runtime.transition_initiative(
            initiative_id=initiative_id,
            to_status="paused",
            reason="interactive pause",
            approved_by="interactive_cli",
            notes=["paused from console"],
        )
        return "\n".join(
            [
                "Nova Initiative Updated",
                f"initiative_id: {record.initiative_id}",
                f"status: {record.status}",
                f"title: {record.title}",
            ]
        )

    def _resume_initiative(self, initiative_id: str) -> str:
        if not initiative_id:
            return "Usage: /resume-initiative <initiative_id>"
        record = self.runtime.transition_initiative(
            initiative_id=initiative_id,
            to_status="active",
            reason="interactive resume",
            approved_by="interactive_cli",
            notes=["resumed from console"],
        )
        return "\n".join(
            [
                "Nova Initiative Updated",
                f"initiative_id: {record.initiative_id}",
                f"status: {record.status}",
                f"title: {record.title}",
            ]
        )

    def _abandon_initiative(self, initiative_id: str) -> str:
        if not initiative_id:
            return "Usage: /abandon-initiative <initiative_id>"
        record = self.runtime.transition_initiative(
            initiative_id=initiative_id,
            to_status="abandoned",
            reason="interactive abandonment",
            notes=["abandoned from console"],
        )
        return "\n".join(
            [
                "Nova Initiative Updated",
                f"initiative_id: {record.initiative_id}",
                f"status: {record.status}",
                f"title: {record.title}",
            ]
        )

    def _actions(self, argument: str) -> str:
        limit = _parse_positive_int(argument, default=10)
        report = self.runtime.action_history_report(limit=limit)
        return "\n".join(
            [
                "Nova Action History",
                f"limit: {limit}",
                f"stable: {report.stable}",
                f"total_actions: {report.total_actions}",
                f"executed_actions: {report.executed_actions}",
                f"blocked_actions: {report.blocked_actions}",
                f"approval_required_actions: {report.approval_required_actions}",
                f"stability_failures: {report.stability_failures}",
                f"rollback_count: {report.rollback_count}",
                f"reasons: {report.reasons}",
            ]
        )

    def _maintenance(self) -> str:
        from nova.agent.tools import ToolRequest

        result = self.runtime.execute_internal_tool(
            request=ToolRequest(
                tool_name="maintenance_plan",
                reason="Interactive console maintenance plan request",
                requested_by="interactive_cli",
            )
        )
        self.runtime.update_presence(
            mode="maintenance_review",
            current_focus="maintenance plan review",
            last_action_status=result.status,
            interaction_summary=f"Maintenance plan request returned status: {result.status}",
        )
        return "\n".join(
            [
                "Nova Maintenance",
                f"status: {result.status}",
                f"requires_approval: {result.requires_approval}",
                f"error: {result.error}",
                f"output: {result.output}",
            ]
        )

    def _summary(self) -> str:
        summary = SessionContinuityBuilder(runtime=self.runtime).build()
        return "\n".join(
            [
                "Nova Session Summary",
                f"session_id: {summary.session_id}",
                f"current_focus: {summary.current_focus}",
                f"interaction_summary: {summary.interaction_summary}",
                f"last_action_status: {summary.last_action_status}",
                f"unresolved_items: {summary.unresolved_items}",
                f"recent_user_inputs: {summary.recent_user_inputs}",
                f"recent_action_attempts: {summary.recent_action_attempts}",
                f"recent_memory_activity: {summary.recent_memory_activity}",
                f"next_pickup: {summary.next_pickup}",
            ]
        )


def _parse_positive_int(value: str, *, default: int) -> int:
    if not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return max(1, parsed)


def _normalize_goal(goal: str) -> str:
    return " ".join(goal.strip().lower().split())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proposal_expired(proposal: dict, *, max_age_seconds: int) -> bool:
    created_at = str(proposal.get("created_at", "") or "")
    if not created_at:
        return True
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return True
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
    return age_seconds > max_age_seconds


def _proposal_drift(*, pending: dict, current: dict) -> list[str]:
    stable_fields = (
        "goal",
        "category",
        "disposition",
        "tool_name",
        "requires_approval",
    )
    changed = []
    for field in stable_fields:
        if pending.get(field) != current.get(field):
            changed.append(field)
    return changed
