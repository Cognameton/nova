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
    AutonomousActionPermission,
    AutonomousActionPlan,
    AutonomousActionPlanStep,
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
