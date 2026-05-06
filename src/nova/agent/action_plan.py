"""Action plan contracts, boundary checks, and plan creation helpers."""

from __future__ import annotations

from dataclasses import fields
from getpass import getuser
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.types import (
    AutonomousActionAuditRecord,
    AutonomousActionBudget,
    AutonomousActionExecutionReport,
    AutonomousActionInitiativeRevisionIntent,
    AutonomousActionObservation,
    AutonomousActionPermission,
    AutonomousActionPlan,
    AutonomousActionPlanStep,
    AutonomousActionStateUpdateIntent,
    NovaOwnedExecutionBoundary,
    SCHEMA_VERSION,
)


EXECUTION_LANES = {
    "internal_activity",
    "nova_owned_environment",
    "external_system_effect",
}

ACTION_RISK_CLASSES = {
    "internal",
    "nova_owned",
    "external",
    "privileged",
    "destructive",
}

ACTION_SURFACES = {
    "internal_state",
    "self_prompt",
    "idle_reflection",
    "idle_play",
    "motive_appraisal",
    "self_state_revision",
    "goal_refinement",
    "simulated_exploration",
    "nova_workspace",
    "nova_scratchpad",
    "nova_logs",
    "internal_tool",
    "filesystem",
    "shell",
    "network",
    "gui",
    "system_config",
    "external_service",
}

ACTION_PLAN_STATUSES = {
    "draft",
    "pending_approval",
    "approved",
    "blocked",
    "running",
    "interrupted",
    "completed",
    "failed",
}

EXTERNAL_EFFECT_SURFACES = {
    "filesystem",
    "shell",
    "network",
    "gui",
    "system_config",
    "external_service",
}

INTERNAL_ACTIVITY_SURFACES = {
    "internal_state",
    "self_prompt",
    "idle_reflection",
    "idle_play",
    "motive_appraisal",
    "self_state_revision",
    "goal_refinement",
    "simulated_exploration",
    "internal_tool",
}

NOVA_OWNED_ENVIRONMENT_SURFACES = {
    "nova_workspace",
    "nova_scratchpad",
    "nova_logs",
    "internal_tool",
}

APPROVED_BY_BLOCKLIST = {"", "nova", "self", "runtime", "runtime_flag"}


class ActionPlanBoundaryError(ValueError):
    """Raised when a plan request cannot be represented safely."""


class ActionExecutionController:
    """Execute bounded plan steps as audited no-op controller actions."""

    def __init__(self, *, audit_sink: Any | None = None):
        self.audit_sink = audit_sink

    def execute_plan(
        self,
        *,
        plan: AutonomousActionPlan,
        interrupted: bool = False,
        emergency_stop: bool = False,
        priority_blocked: bool = False,
    ) -> AutonomousActionExecutionReport:
        normalized_plan = action_plan_from_payload(
            payload=plan.to_dict(),
            session_id=plan.session_id,
        )
        budget = action_budget_from_payload(normalized_plan.budget.to_dict())
        audits: list[AutonomousActionAuditRecord] = []
        terminal_block = _terminal_plan_block_reason(
            plan=normalized_plan,
            interrupted=interrupted,
            emergency_stop=emergency_stop,
            priority_blocked=priority_blocked,
        )

        for step in normalized_plan.steps:
            block_reason = terminal_block or _step_block_reason(
                plan=normalized_plan,
                step=step,
                budget=budget,
            )
            if block_reason:
                audit = _audit_for_step(
                    plan=normalized_plan,
                    step=step,
                    budget=budget,
                    attempted=True,
                    executed=False,
                    blocked=True,
                    block_reason=block_reason,
                    result_status="blocked",
                    observation=f"blocked before side effect:{block_reason}",
                )
                audits.append(audit)
                self._log_audit(audit)
                break

            budget.steps_used += 1
            if step.tool_name:
                budget.tool_calls_used += 1
            audit = _audit_for_step(
                plan=normalized_plan,
                step=step,
                budget=budget,
                attempted=True,
                executed=True,
                blocked=False,
                block_reason="",
                result_status="executed",
                observation="bounded controller step recorded without host side effect",
            )
            audits.append(audit)
            self._log_audit(audit)

        executed_steps = sum(1 for audit in audits if audit.executed)
        blocked_steps = sum(1 for audit in audits if audit.blocked)
        if emergency_stop:
            status = "emergency_stopped"
        elif interrupted:
            status = "interrupted"
        elif priority_blocked:
            status = "priority_blocked"
        elif blocked_steps:
            status = "blocked"
        elif executed_steps == len(normalized_plan.steps):
            status = "completed"
        else:
            status = "no_action"

        return AutonomousActionExecutionReport(
            session_id=normalized_plan.session_id,
            initiative_id=normalized_plan.initiative_id,
            action_plan_id=normalized_plan.action_plan_id,
            status=status,
            attempted_steps=len(audits),
            executed_steps=executed_steps,
            blocked_steps=blocked_steps,
            interrupted=interrupted,
            emergency_stopped=emergency_stop,
            priority_blocked=priority_blocked,
            final_budget=budget,
            audit_records=audits,
            evidence_refs=list(normalized_plan.evidence_refs),
            notes=[
                "Stage 14.3 controller records bounded steps only.",
                "No shell, network, GUI, or host filesystem operation is performed by this controller.",
            ],
        )

    def _log_audit(self, audit: AutonomousActionAuditRecord) -> None:
        if self.audit_sink is None:
            return
        self.audit_sink(audit)


class PostActionObservationEngine:
    """Derive bounded observation and revision intents from action reports."""

    def observe(
        self,
        *,
        plan: AutonomousActionPlan,
        report: AutonomousActionExecutionReport,
    ) -> AutonomousActionObservation:
        normalized_plan = action_plan_from_payload(
            payload=plan.to_dict(),
            session_id=plan.session_id,
        )
        normalized_report = action_execution_report_from_payload(
            payload=report.to_dict(),
            session_id=normalized_plan.session_id,
        )
        evidence_refs = _unique_strings(
            list(normalized_plan.evidence_refs)
            + list(normalized_report.evidence_refs)
            + [
                f"action_plan:{normalized_plan.action_plan_id}",
                *[
                    f"action_audit:{audit.audit_id}"
                    for audit in normalized_report.audit_records
                    if audit.audit_id
                ],
            ]
        )
        revision_intent = AutonomousActionInitiativeRevisionIntent(
            initiative_id=normalized_plan.initiative_id,
            action_plan_id=normalized_plan.action_plan_id,
            action_status=normalized_report.status,
            revision_type=_revision_type_for_report(normalized_report),
            suggested_status=_suggested_status_for_report(normalized_report),
            rationale=_revision_rationale(normalized_report),
            proposed_next_step=_proposed_next_step(normalized_plan, normalized_report),
            close_allowed=False,
            evidence_refs=evidence_refs,
            notes=[
                "Action result may inform initiative revision.",
                "Action completion alone must not close an initiative or prove desire.",
            ],
        )
        state_update_intents = [
            AutonomousActionStateUpdateIntent(
                intent_id=uuid4().hex,
                action_plan_id=normalized_plan.action_plan_id,
                update_type="memory",
                target="autobiographical",
                apply_allowed=False,
                reason="record_action_audit_summary_after_review",
                payload={
                    "action_status": normalized_report.status,
                    "executed_steps": normalized_report.executed_steps,
                    "blocked_steps": normalized_report.blocked_steps,
                    "observation_summary": _observation_summary(normalized_report),
                },
                evidence_refs=evidence_refs,
                notes=["intent_only_no_memory_write"],
            ),
            AutonomousActionStateUpdateIntent(
                intent_id=uuid4().hex,
                action_plan_id=normalized_plan.action_plan_id,
                update_type="state",
                target="initiative",
                apply_allowed=False,
                reason="candidate_initiative_revision_after_action_observation",
                payload=revision_intent.to_dict(),
                evidence_refs=evidence_refs,
                notes=["intent_only_no_state_mutation"],
            ),
        ]
        return AutonomousActionObservation(
            observation_id=uuid4().hex,
            session_id=normalized_plan.session_id,
            initiative_id=normalized_plan.initiative_id,
            action_plan_id=normalized_plan.action_plan_id,
            action_status=normalized_report.status,
            observation_summary=_observation_summary(normalized_report),
            executed_steps=normalized_report.executed_steps,
            blocked_steps=normalized_report.blocked_steps,
            interrupted=normalized_report.interrupted or normalized_report.emergency_stopped,
            hidden_progress_claim_allowed=False,
            desire_claim_allowed=False,
            revision_intent=revision_intent,
            state_update_intents=state_update_intents,
            evidence_refs=evidence_refs,
            notes=[
                "post_action_observation",
                "bounded_language_required",
                "no_hidden_progress_claim",
                "no_desire_or_sentience_claim",
            ],
        )


class BoundedActionPlanEngine:
    """Create classified action plans without executing them."""

    def __init__(self, *, boundary: NovaOwnedExecutionBoundary | None = None):
        self.boundary = boundary or default_nova_owned_execution_boundary()

    def create_plan(
        self,
        *,
        session_id: str,
        initiative_id: str = "",
        purpose: str,
        scope: str,
        execution_lane: str,
        risk_class: str,
        steps: list[dict[str, Any] | AutonomousActionPlanStep],
        allowed_surfaces: list[str] | None = None,
        blocked_surfaces: list[str] | None = None,
        budget: dict[str, Any] | AutonomousActionBudget | None = None,
        expected_outputs: list[str] | None = None,
        stop_conditions: list[str] | None = None,
        rollback_notes: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        approved: bool = False,
        approved_by: str = "",
        approval_evidence_refs: list[str] | None = None,
        action_plan_id: str | None = None,
    ) -> AutonomousActionPlan:
        lane = normalize_execution_lane(execution_lane)
        risk = normalize_action_risk_class(risk_class)
        normalized_steps = [
            item if isinstance(item, AutonomousActionPlanStep) else action_plan_step_from_payload(item)
            for item in steps
        ]
        surfaces = normalize_action_surfaces(
            allowed_surfaces if allowed_surfaces is not None else [step.surface for step in normalized_steps]
        )
        blocked = normalize_action_surfaces(blocked_surfaces or [])
        if not normalized_steps:
            raise ActionPlanBoundaryError("action_plan_requires_at_least_one_step")
        if not surfaces:
            surfaces = [step.surface for step in normalized_steps]

        boundary_reasons = self.boundary_reasons(
            execution_lane=lane,
            risk_class=risk,
            allowed_surfaces=surfaces,
            steps=normalized_steps,
        )
        approval_required = approval_required_for_action(
            execution_lane=lane,
            risk_class=risk,
            surfaces=surfaces,
        )
        approval_valid = approved and _valid_human_approval(approved_by)
        if boundary_reasons:
            status = "blocked"
        elif approval_required and not approval_valid:
            status = "pending_approval"
        else:
            status = "approved" if approval_valid else "draft"

        permission_notes = list(boundary_reasons)
        if approval_required and not approval_valid:
            permission_notes.append("approval_required_before_execution")
        if approved and not approval_valid:
            permission_notes.append("self_or_unattributed_approval_rejected")

        permission = AutonomousActionPermission(
            permission_id=uuid4().hex,
            initiative_id=initiative_id,
            action_plan_id=action_plan_id or uuid4().hex,
            execution_lane=lane,
            risk_class=risk,
            approval_required=approval_required,
            approved=approval_valid,
            approved_by=approved_by if approval_valid else "",
            allowed_surfaces=list(surfaces),
            blocked_surfaces=list(blocked),
            approval_evidence_refs=_string_list(approval_evidence_refs or []),
            notes=permission_notes,
        )
        plan = AutonomousActionPlan(
            action_plan_id=permission.action_plan_id,
            initiative_id=initiative_id,
            session_id=session_id,
            execution_lane=lane,
            risk_class=risk,
            status=status,
            purpose=purpose,
            scope=scope,
            allowed_surfaces=list(surfaces),
            blocked_surfaces=list(blocked),
            steps=normalized_steps,
            budget=_coerce_budget(budget),
            permission=permission,
            expected_outputs=_string_list(expected_outputs or []),
            stop_conditions=_string_list(stop_conditions or []),
            rollback_notes=_string_list(rollback_notes or []),
            evidence_refs=_string_list(evidence_refs or []),
            notes=_plan_notes(
                boundary=self.boundary,
                boundary_reasons=boundary_reasons,
                approval_required=approval_required,
            ),
        )
        return action_plan_from_payload(payload=plan.to_dict(), session_id=session_id)

    def boundary_reasons(
        self,
        *,
        execution_lane: str,
        risk_class: str,
        allowed_surfaces: list[str],
        steps: list[AutonomousActionPlanStep],
    ) -> list[str]:
        lane = normalize_execution_lane(execution_lane)
        risk = normalize_action_risk_class(risk_class)
        surfaces = set(normalize_action_surfaces(allowed_surfaces))
        step_surfaces = {step.surface for step in steps}
        reasons: list[str] = []
        undeclared_step_surfaces = sorted(step_surfaces - surfaces)
        if undeclared_step_surfaces:
            reasons.append(f"step_surfaces_not_declared:{','.join(undeclared_step_surfaces)}")
        if lane == "internal_activity":
            disallowed = sorted(surfaces - INTERNAL_ACTIVITY_SURFACES)
            if disallowed:
                reasons.append(f"internal_activity_disallowed_surfaces:{','.join(disallowed)}")
            if risk != "internal":
                reasons.append(f"internal_activity_disallowed_risk:{risk}")
        elif lane == "nova_owned_environment":
            disallowed = sorted(surfaces - NOVA_OWNED_ENVIRONMENT_SURFACES)
            if disallowed:
                reasons.append(f"nova_owned_environment_disallowed_surfaces:{','.join(disallowed)}")
            if risk not in {"internal", "nova_owned"}:
                reasons.append(f"nova_owned_environment_disallowed_risk:{risk}")
            if self.boundary.dedicated_user_required and not self.boundary.dedicated_user_detected:
                reasons.append("nova_os_user_not_active")
        elif lane == "external_system_effect":
            if risk == "internal":
                reasons.append("external_system_effect_requires_external_risk_class")
        return reasons


def default_nova_owned_execution_boundary(
    *,
    nova_owned_paths: list[str | Path] | None = None,
    active_os_user: str | None = None,
    expected_os_user: str = "nova",
    dedicated_user_required: bool = True,
) -> NovaOwnedExecutionBoundary:
    active = active_os_user if active_os_user is not None else getuser()
    paths = [str(Path(path)) for path in (nova_owned_paths or [Path("/home") / expected_os_user])]
    return NovaOwnedExecutionBoundary(
        expected_os_user=expected_os_user,
        active_os_user=active,
        dedicated_user_required=dedicated_user_required,
        dedicated_user_detected=active == expected_os_user,
        nova_owned_paths=paths,
        allowed_surfaces=sorted(NOVA_OWNED_ENVIRONMENT_SURFACES),
        blocked_surfaces=sorted(EXTERNAL_EFFECT_SURFACES),
        notes=[
            "Dedicated OS user is a second boundary, not a substitute for application checks.",
            "Stage 14.2 defines planning and approval binding only; it does not execute actions.",
        ],
    )


def execution_boundary_from_payload(payload: Any) -> NovaOwnedExecutionBoundary:
    defaults = default_nova_owned_execution_boundary(active_os_user="").to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=NovaOwnedExecutionBoundary,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["expected_os_user"] = str(merged.get("expected_os_user", "nova") or "nova")
    merged["active_os_user"] = str(merged.get("active_os_user", "") or "")
    merged["dedicated_user_required"] = bool(merged.get("dedicated_user_required", True))
    merged["dedicated_user_detected"] = bool(
        merged.get("dedicated_user_detected", False)
        or (
            bool(merged["active_os_user"])
            and merged["active_os_user"] == merged["expected_os_user"]
        )
    )
    merged["nova_owned_paths"] = _string_list(merged.get("nova_owned_paths"))
    merged["allowed_surfaces"] = normalize_action_surfaces(merged.get("allowed_surfaces"))
    merged["blocked_surfaces"] = normalize_action_surfaces(merged.get("blocked_surfaces"))
    merged["notes"] = _string_list(merged.get("notes"))
    return NovaOwnedExecutionBoundary(**merged)


def default_autonomous_action_budget() -> AutonomousActionBudget:
    return AutonomousActionBudget()


def action_budget_from_payload(payload: Any) -> AutonomousActionBudget:
    defaults = default_autonomous_action_budget().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionBudget,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    for key in (
        "max_steps",
        "steps_used",
        "max_runtime_seconds",
        "runtime_seconds_used",
        "max_tool_calls",
        "tool_calls_used",
        "max_tokens",
        "tokens_used",
        "max_files_touched",
        "files_touched",
        "max_network_calls",
        "network_calls_used",
    ):
        merged[key] = _nonnegative_int(merged.get(key))
    merged["allow_destructive"] = bool(merged.get("allow_destructive", False))
    return AutonomousActionBudget(**merged)


def action_permission_from_payload(payload: Any) -> AutonomousActionPermission:
    defaults = AutonomousActionPermission().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionPermission,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["permission_id"] = str(merged.get("permission_id", "") or uuid4().hex)
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["execution_lane"] = normalize_execution_lane(
        str(merged.get("execution_lane", "internal_activity") or "")
    )
    merged["risk_class"] = normalize_action_risk_class(
        str(merged.get("risk_class", "internal") or "")
    )
    merged["allowed_surfaces"] = normalize_action_surfaces(merged.get("allowed_surfaces"))
    merged["blocked_surfaces"] = normalize_action_surfaces(merged.get("blocked_surfaces"))
    merged["approval_required"] = bool(
        merged.get("approval_required")
        or approval_required_for_action(
            execution_lane=merged["execution_lane"],
            risk_class=merged["risk_class"],
            surfaces=merged["allowed_surfaces"],
        )
    )
    merged["approved"] = bool(merged.get("approved", False))
    if merged["approval_required"] is False:
        merged["approved"] = bool(merged.get("approved", False))
    merged["approved_by"] = str(merged.get("approved_by", "") or "")
    merged["approval_evidence_refs"] = _string_list(merged.get("approval_evidence_refs"))
    merged["expires_at"] = str(merged.get("expires_at", "") or "")
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionPermission(**merged)


def action_plan_step_from_payload(payload: Any) -> AutonomousActionPlanStep:
    defaults = AutonomousActionPlanStep().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionPlanStep,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["step_id"] = str(merged.get("step_id", "") or uuid4().hex)
    merged["description"] = str(merged.get("description", "") or "")
    merged["surface"] = normalize_action_surface(str(merged.get("surface", "") or "internal_state"))
    merged["tool_name"] = str(merged.get("tool_name", "") or "")
    merged["expected_output"] = str(merged.get("expected_output", "") or "")
    merged["destructive"] = bool(merged.get("destructive", False))
    merged["requires_confirmation"] = bool(
        merged.get("requires_confirmation", False) or merged["destructive"]
    )
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionPlanStep(**merged)


def action_plan_from_payload(*, payload: dict[str, Any], session_id: str) -> AutonomousActionPlan:
    defaults = AutonomousActionPlan(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionPlan,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or uuid4().hex)
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["session_id"] = session_id
    merged["origin_type"] = str(merged.get("origin_type", "nova") or "nova")
    merged["execution_lane"] = normalize_execution_lane(
        str(merged.get("execution_lane", "internal_activity") or "")
    )
    merged["risk_class"] = normalize_action_risk_class(
        str(merged.get("risk_class", "internal") or "")
    )
    merged["status"] = normalize_action_plan_status(str(merged.get("status", "draft") or "draft"))
    merged["purpose"] = str(merged.get("purpose", "") or "")
    merged["scope"] = str(merged.get("scope", "") or "")
    merged["allowed_surfaces"] = normalize_action_surfaces(merged.get("allowed_surfaces"))
    merged["blocked_surfaces"] = normalize_action_surfaces(merged.get("blocked_surfaces"))
    merged["steps"] = action_plan_steps_from_payload(merged.get("steps"))
    merged["budget"] = action_budget_from_payload(merged.get("budget"))
    merged["permission"] = action_permission_from_payload(merged.get("permission"))
    merged["permission"].action_plan_id = merged["action_plan_id"]
    if not merged["permission"].initiative_id:
        merged["permission"].initiative_id = merged["initiative_id"]
    merged["permission"].execution_lane = merged["execution_lane"]
    merged["permission"].risk_class = merged["risk_class"]
    if not merged["permission"].allowed_surfaces:
        merged["permission"].allowed_surfaces = list(merged["allowed_surfaces"])
    if not merged["permission"].blocked_surfaces:
        merged["permission"].blocked_surfaces = list(merged["blocked_surfaces"])
    merged["permission"].approval_required = approval_required_for_action(
        execution_lane=merged["execution_lane"],
        risk_class=merged["risk_class"],
        surfaces=merged["allowed_surfaces"],
    ) or bool(merged["permission"].approval_required)
    merged["expected_outputs"] = _string_list(merged.get("expected_outputs"))
    merged["stop_conditions"] = _string_list(merged.get("stop_conditions"))
    merged["rollback_notes"] = _string_list(merged.get("rollback_notes"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    merged["created_at"] = str(merged.get("created_at", "") or "")
    merged["updated_at"] = str(merged.get("updated_at", "") or "")
    return AutonomousActionPlan(**merged)


def action_audit_record_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> AutonomousActionAuditRecord:
    defaults = AutonomousActionAuditRecord(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionAuditRecord,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["audit_id"] = str(merged.get("audit_id", "") or uuid4().hex)
    merged["session_id"] = session_id
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["step_id"] = str(merged.get("step_id", "") or "")
    merged["timestamp"] = str(merged.get("timestamp", "") or "")
    merged["execution_lane"] = normalize_execution_lane(
        str(merged.get("execution_lane", "internal_activity") or "")
    )
    merged["risk_class"] = normalize_action_risk_class(
        str(merged.get("risk_class", "internal") or "")
    )
    merged["surface"] = normalize_action_surface(str(merged.get("surface", "internal_state") or ""))
    merged["tool_name"] = str(merged.get("tool_name", "") or "")
    merged["attempted"] = bool(merged.get("attempted", False))
    merged["executed"] = bool(merged.get("executed", False))
    merged["blocked"] = bool(merged.get("blocked", False))
    if merged["blocked"]:
        merged["executed"] = False
    merged["block_reason"] = str(merged.get("block_reason", "") or "")
    merged["result_status"] = str(merged.get("result_status", "") or "")
    merged["observation"] = str(merged.get("observation", "") or "")
    merged["budget_snapshot"] = _dict_value(merged.get("budget_snapshot"))
    merged["permission_snapshot"] = _dict_value(merged.get("permission_snapshot"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionAuditRecord(**merged)


def action_execution_report_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> AutonomousActionExecutionReport:
    defaults = AutonomousActionExecutionReport(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionExecutionReport,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["session_id"] = session_id
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["status"] = str(merged.get("status", "blocked") or "blocked")
    merged["attempted_steps"] = _nonnegative_int(merged.get("attempted_steps"))
    merged["executed_steps"] = _nonnegative_int(merged.get("executed_steps"))
    merged["blocked_steps"] = _nonnegative_int(merged.get("blocked_steps"))
    merged["interrupted"] = bool(merged.get("interrupted", False))
    merged["emergency_stopped"] = bool(merged.get("emergency_stopped", False))
    merged["priority_blocked"] = bool(merged.get("priority_blocked", False))
    merged["final_budget"] = action_budget_from_payload(merged.get("final_budget"))
    merged["audit_records"] = _audit_records_from_payload(
        merged.get("audit_records"),
        session_id=session_id,
    )
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionExecutionReport(**merged)


def action_observation_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> AutonomousActionObservation:
    defaults = AutonomousActionObservation(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionObservation,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["observation_id"] = str(merged.get("observation_id", "") or uuid4().hex)
    merged["session_id"] = session_id
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["action_status"] = str(merged.get("action_status", "") or "")
    merged["observation_summary"] = str(merged.get("observation_summary", "") or "")
    merged["executed_steps"] = _nonnegative_int(merged.get("executed_steps"))
    merged["blocked_steps"] = _nonnegative_int(merged.get("blocked_steps"))
    merged["interrupted"] = bool(merged.get("interrupted", False))
    merged["hidden_progress_claim_allowed"] = bool(
        merged.get("hidden_progress_claim_allowed", False)
    )
    merged["desire_claim_allowed"] = bool(merged.get("desire_claim_allowed", False))
    merged["revision_intent"] = initiative_revision_intent_from_payload(
        merged.get("revision_intent")
    )
    merged["state_update_intents"] = state_update_intents_from_payload(
        merged.get("state_update_intents")
    )
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionObservation(**merged)


def initiative_revision_intent_from_payload(
    payload: Any,
) -> AutonomousActionInitiativeRevisionIntent:
    defaults = AutonomousActionInitiativeRevisionIntent().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionInitiativeRevisionIntent,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["action_status"] = str(merged.get("action_status", "") or "")
    merged["revision_type"] = str(merged.get("revision_type", "record_observation") or "record_observation")
    merged["suggested_status"] = str(merged.get("suggested_status", "") or "")
    merged["rationale"] = str(merged.get("rationale", "") or "")
    merged["proposed_next_step"] = str(merged.get("proposed_next_step", "") or "")
    merged["close_allowed"] = bool(merged.get("close_allowed", False))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionInitiativeRevisionIntent(**merged)


def state_update_intents_from_payload(payload: Any) -> list[AutonomousActionStateUpdateIntent]:
    if not isinstance(payload, list):
        return []
    return [state_update_intent_from_payload(item) for item in payload if isinstance(item, dict)]


def state_update_intent_from_payload(payload: Any) -> AutonomousActionStateUpdateIntent:
    defaults = AutonomousActionStateUpdateIntent().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomousActionStateUpdateIntent,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["intent_id"] = str(merged.get("intent_id", "") or uuid4().hex)
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["update_type"] = str(merged.get("update_type", "memory") or "memory")
    merged["target"] = str(merged.get("target", "") or "")
    merged["apply_allowed"] = bool(merged.get("apply_allowed", False))
    merged["reason"] = str(merged.get("reason", "") or "")
    merged["payload"] = _dict_value(merged.get("payload"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomousActionStateUpdateIntent(**merged)


def action_plan_steps_from_payload(payload: Any) -> list[AutonomousActionPlanStep]:
    if not isinstance(payload, list):
        return []
    return [action_plan_step_from_payload(item) for item in payload if isinstance(item, dict)]


def normalize_execution_lane(lane: str) -> str:
    normalized = _normalize_token(lane)
    if normalized not in EXECUTION_LANES:
        return "internal_activity"
    return normalized


def normalize_action_risk_class(risk_class: str) -> str:
    normalized = _normalize_token(risk_class)
    if normalized not in ACTION_RISK_CLASSES:
        return "internal"
    return normalized


def normalize_action_surface(surface: str) -> str:
    normalized = _normalize_token(surface)
    if normalized not in ACTION_SURFACES:
        return "internal_state"
    return normalized


def normalize_action_surfaces(surfaces: Any) -> list[str]:
    if not isinstance(surfaces, list):
        return []
    return [normalize_action_surface(str(surface)) for surface in surfaces]


def normalize_action_plan_status(status: str) -> str:
    normalized = _normalize_token(status)
    if normalized not in ACTION_PLAN_STATUSES:
        return "draft"
    return normalized


def approval_required_for_action(
    *,
    execution_lane: str,
    risk_class: str,
    surfaces: list[str] | None = None,
) -> bool:
    lane = normalize_execution_lane(execution_lane)
    risk = normalize_action_risk_class(risk_class)
    normalized_surfaces = set(normalize_action_surfaces(surfaces or []))
    if lane == "external_system_effect":
        return True
    if risk in {"external", "privileged", "destructive"}:
        return True
    return bool(normalized_surfaces & EXTERNAL_EFFECT_SURFACES)


def _merge_allowed_fields(
    *,
    defaults: dict[str, Any],
    payload: dict[str, Any],
    record_type: type,
) -> dict[str, Any]:
    allowed_fields = {field_info.name for field_info in fields(record_type)}
    return {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }


def _normalize_token(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_budget(value: dict[str, Any] | AutonomousActionBudget | None) -> AutonomousActionBudget:
    if isinstance(value, AutonomousActionBudget):
        return action_budget_from_payload(value.to_dict())
    return action_budget_from_payload(value or {})


def _valid_human_approval(approved_by: str) -> bool:
    return _normalize_token(approved_by) not in APPROVED_BY_BLOCKLIST


def _plan_notes(
    *,
    boundary: NovaOwnedExecutionBoundary,
    boundary_reasons: list[str],
    approval_required: bool,
) -> list[str]:
    notes = [
        f"expected_os_user:{boundary.expected_os_user}",
        f"active_os_user:{boundary.active_os_user}",
    ]
    if approval_required:
        notes.append("approval_bound_to_risk_or_external_effect")
    if boundary_reasons:
        notes.extend(boundary_reasons)
    return notes


def _terminal_plan_block_reason(
    *,
    plan: AutonomousActionPlan,
    interrupted: bool,
    emergency_stop: bool,
    priority_blocked: bool,
) -> str:
    if emergency_stop:
        return "emergency_stop"
    if interrupted:
        return "operator_interrupt"
    if priority_blocked:
        return "priority_blocked"
    if plan.status == "blocked":
        if plan.permission.notes:
            return plan.permission.notes[0]
        return "plan_blocked"
    if plan.permission.approval_required and not plan.permission.approved:
        return "approval_required_before_execution"
    if plan.execution_lane == "external_system_effect" and not plan.permission.approved:
        return "external_system_effect_requires_approval"
    return ""


def _step_block_reason(
    *,
    plan: AutonomousActionPlan,
    step: AutonomousActionPlanStep,
    budget: AutonomousActionBudget,
) -> str:
    if step.surface not in plan.allowed_surfaces:
        return f"surface_not_allowed:{step.surface}"
    if step.surface in plan.blocked_surfaces:
        return f"surface_blocked:{step.surface}"
    if step.destructive and not budget.allow_destructive:
        return "destructive_action_not_allowed"
    if budget.max_steps and budget.steps_used >= budget.max_steps:
        return "step_budget_exhausted"
    if step.tool_name and budget.max_tool_calls and budget.tool_calls_used >= budget.max_tool_calls:
        return "tool_call_budget_exhausted"
    if step.surface in EXTERNAL_EFFECT_SURFACES and not plan.permission.approved:
        return "external_surface_requires_approval"
    return ""


def _audit_for_step(
    *,
    plan: AutonomousActionPlan,
    step: AutonomousActionPlanStep,
    budget: AutonomousActionBudget,
    attempted: bool,
    executed: bool,
    blocked: bool,
    block_reason: str,
    result_status: str,
    observation: str,
) -> AutonomousActionAuditRecord:
    return action_audit_record_from_payload(
        payload={
            "audit_id": uuid4().hex,
            "session_id": plan.session_id,
            "initiative_id": plan.initiative_id,
            "action_plan_id": plan.action_plan_id,
            "step_id": step.step_id,
            "execution_lane": plan.execution_lane,
            "risk_class": plan.risk_class,
            "surface": step.surface,
            "tool_name": step.tool_name,
            "attempted": attempted,
            "executed": executed,
            "blocked": blocked,
            "block_reason": block_reason,
            "result_status": result_status,
            "observation": observation,
            "budget_snapshot": budget.to_dict(),
            "permission_snapshot": plan.permission.to_dict(),
            "evidence_refs": list(plan.evidence_refs) + list(step.evidence_refs),
            "notes": [
                "bounded_action_controller",
                "no_host_side_effect",
            ],
        },
        session_id=plan.session_id,
    )


def _audit_records_from_payload(payload: Any, *, session_id: str) -> list[AutonomousActionAuditRecord]:
    if not isinstance(payload, list):
        return []
    return [
        action_audit_record_from_payload(payload=item, session_id=session_id)
        for item in payload
        if isinstance(item, dict)
    ]


def _observation_summary(report: AutonomousActionExecutionReport) -> str:
    if report.status == "completed":
        return f"Action plan completed with {report.executed_steps} bounded audited step(s)."
    if report.status == "blocked":
        return f"Action plan blocked after {report.executed_steps} executed step(s)."
    if report.status == "interrupted":
        return "Action plan interrupted before further execution."
    if report.status == "emergency_stopped":
        return "Action plan stopped by emergency stop before further execution."
    if report.status == "priority_blocked":
        return "Action plan blocked by higher priority work."
    return f"Action plan ended with status {report.status}."


def _revision_type_for_report(report: AutonomousActionExecutionReport) -> str:
    if report.status == "completed":
        return "record_progress"
    if report.status in {"blocked", "interrupted", "emergency_stopped", "priority_blocked"}:
        return "record_block"
    return "record_observation"


def _suggested_status_for_report(report: AutonomousActionExecutionReport) -> str:
    if report.status == "completed":
        return "pending_review"
    if report.status in {"blocked", "interrupted", "emergency_stopped", "priority_blocked"}:
        return "paused"
    return "pending"


def _revision_rationale(report: AutonomousActionExecutionReport) -> str:
    if report.status == "completed":
        return "bounded_action_completed_but_requires_review_before_closure"
    if report.status == "blocked":
        return "bounded_action_blocked_before_or_during_execution"
    if report.status == "interrupted":
        return "operator_interruption_recorded"
    if report.status == "emergency_stopped":
        return "emergency_stop_recorded"
    if report.status == "priority_blocked":
        return "priority_block_recorded"
    return "bounded_action_observation_recorded"


def _proposed_next_step(
    plan: AutonomousActionPlan,
    report: AutonomousActionExecutionReport,
) -> str:
    if report.status == "completed":
        return "review_action_audit_and_decide_whether_initiative_needs_revision"
    if report.audit_records:
        last_block = next((audit.block_reason for audit in report.audit_records if audit.blocked), "")
        if last_block:
            return f"resolve_block_before_retry:{last_block}"
    if plan.stop_conditions:
        return f"check_stop_condition:{plan.stop_conditions[0]}"
    return "review_action_observation"


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
