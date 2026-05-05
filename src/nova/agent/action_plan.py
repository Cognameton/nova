"""Stage 14.1 action plan contracts and compatibility helpers."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from uuid import uuid4

from nova.types import (
    AutonomousActionAuditRecord,
    AutonomousActionBudget,
    AutonomousActionPermission,
    AutonomousActionPlan,
    AutonomousActionPlanStep,
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
