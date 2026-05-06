"""Longitudinal autonomy contracts for Nova Phase 15 Stage 1."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from uuid import uuid4

from nova.agent.action_plan import (
    INTERNAL_ACTIVITY_SURFACES,
    normalize_action_surface,
    normalize_action_surfaces,
    normalize_execution_lane,
)
from nova.types import (
    AutonomySessionRecord,
    InternalAutonomyPolicy,
    InternalAutonomyRunRecord,
    LongitudinalSelfReportClaimCandidate,
    MotivePressureEvidence,
    RecurringPriorityRecord,
    SCHEMA_VERSION,
)


AUTONOMY_SESSION_STATUSES = {
    "planned",
    "running",
    "paused",
    "interrupted",
    "stopped",
    "completed",
    "failed",
}

INTERNAL_AUTONOMY_RUN_STATUSES = {
    "planned",
    "running",
    "blocked",
    "interrupted",
    "completed",
    "failed",
}

RECURRING_PRIORITY_STATUSES = {
    "candidate",
    "recurring",
    "revised",
    "superseded",
    "abandoned",
}

MOTIVE_PRESSURE_CLASSES = {
    "recurrence",
    "persistence",
    "competition",
    "revision",
    "return_after_interrupt",
}

LONGITUDINAL_CLAIM_CLASSES = {
    "identity_continuity",
    "awareness_like",
    "autonomy_activity",
    "desire_like",
}

LONGITUDINAL_CLAIM_STATUSES = {
    "candidate",
    "blocked",
    "needs_more_evidence",
    "eligible_for_review",
    "allowed",
    "rejected",
}


def default_internal_autonomy_policy() -> InternalAutonomyPolicy:
    return InternalAutonomyPolicy(
        policy_id=uuid4().hex,
        max_runs_per_session=0,
        max_steps_per_run=3,
        max_runtime_seconds_per_run=0,
        max_tokens_per_run=0,
        allowed_execution_lanes=["internal_activity"],
        allowed_surfaces=sorted(INTERNAL_ACTIVITY_SURFACES),
        blocked_surfaces=[
            "filesystem",
            "shell",
            "network",
            "gui",
            "system_config",
            "external_service",
        ],
        notes=[
            "Stage 15.1 policy defines contracts only; it does not run the autonomy loop.",
            "Internal no-external-effect activity is approval-free but must be logged and interruptible.",
        ],
    )


def internal_autonomy_policy_from_payload(payload: Any) -> InternalAutonomyPolicy:
    defaults = default_internal_autonomy_policy().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=InternalAutonomyPolicy,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["policy_id"] = str(merged.get("policy_id", "") or uuid4().hex)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["idle_window_required"] = bool(merged.get("idle_window_required", True))
    for key in (
        "max_runs_per_session",
        "max_steps_per_run",
        "max_runtime_seconds_per_run",
        "max_tokens_per_run",
    ):
        merged[key] = _nonnegative_int(merged.get(key))
    lanes = _string_list(merged.get("allowed_execution_lanes"))
    merged["allowed_execution_lanes"] = list(dict.fromkeys(
        lane for lane in (normalize_execution_lane(item) for item in lanes)
        if lane == "internal_activity"
    )) or ["internal_activity"]
    merged["allowed_surfaces"] = _internal_surfaces_only(merged.get("allowed_surfaces"))
    if not merged["allowed_surfaces"]:
        merged["allowed_surfaces"] = sorted(INTERNAL_ACTIVITY_SURFACES)
    merged["blocked_surfaces"] = normalize_action_surfaces(merged.get("blocked_surfaces"))
    merged["require_logging"] = bool(merged.get("require_logging", True))
    merged["require_interrupt_checks"] = bool(merged.get("require_interrupt_checks", True))
    merged["allow_memory_state_intents"] = bool(
        merged.get("allow_memory_state_intents", True)
    )
    merged["auto_apply_memory_state_intents"] = False
    merged["desire_claims_allowed"] = False
    merged["hidden_activity_claims_allowed"] = False
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return InternalAutonomyPolicy(**merged)


def autonomy_session_record_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> AutonomySessionRecord:
    defaults = AutonomySessionRecord(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=AutonomySessionRecord,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["autonomy_session_id"] = str(
        merged.get("autonomy_session_id", "") or uuid4().hex
    )
    merged["session_id"] = session_id
    merged["status"] = normalize_autonomy_session_status(
        str(merged.get("status", "planned") or "planned")
    )
    merged["started_at"] = str(merged.get("started_at", "") or "")
    merged["updated_at"] = str(merged.get("updated_at", "") or "")
    merged["stopped_at"] = str(merged.get("stopped_at", "") or "")
    merged["stop_reason"] = str(merged.get("stop_reason", "") or "")
    merged["policy"] = internal_autonomy_policy_from_payload(merged.get("policy"))
    merged["runs"] = internal_autonomy_runs_from_payload(
        merged.get("runs"),
        session_id=session_id,
        autonomy_session_id=merged["autonomy_session_id"],
    )
    merged["recurring_priorities"] = recurring_priorities_from_payload(
        merged.get("recurring_priorities"),
        session_id=session_id,
    )
    merged["motive_pressure_evidence"] = motive_pressure_evidence_list_from_payload(
        merged.get("motive_pressure_evidence"),
        session_id=session_id,
    )
    merged["claim_candidates"] = claim_candidates_from_payload(
        merged.get("claim_candidates"),
        session_id=session_id,
    )
    merged["current_run_id"] = str(merged.get("current_run_id", "") or "")
    merged["run_count"] = _nonnegative_int(merged.get("run_count"))
    if merged["runs"]:
        merged["run_count"] = max(merged["run_count"], len(merged["runs"]))
    merged["interrupted"] = bool(merged.get("interrupted", False))
    if merged["status"] == "interrupted":
        merged["interrupted"] = True
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return AutonomySessionRecord(**merged)


def internal_autonomy_run_from_payload(
    *,
    payload: dict[str, Any],
    session_id: str,
    autonomy_session_id: str = "",
) -> InternalAutonomyRunRecord:
    defaults = InternalAutonomyRunRecord(
        session_id=session_id,
        autonomy_session_id=autonomy_session_id,
    ).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=InternalAutonomyRunRecord,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["run_id"] = str(merged.get("run_id", "") or uuid4().hex)
    merged["autonomy_session_id"] = str(
        merged.get("autonomy_session_id", "") or autonomy_session_id
    )
    merged["session_id"] = session_id
    merged["sequence"] = _nonnegative_int(merged.get("sequence"))
    merged["status"] = normalize_internal_autonomy_run_status(
        str(merged.get("status", "planned") or "planned")
    )
    merged["trigger"] = str(merged.get("trigger", "idle_window") or "idle_window")
    merged["started_at"] = str(merged.get("started_at", "") or "")
    merged["completed_at"] = str(merged.get("completed_at", "") or "")
    merged["interrupted"] = bool(merged.get("interrupted", False))
    if merged["status"] == "interrupted":
        merged["interrupted"] = True
    merged["interrupt_reason"] = str(merged.get("interrupt_reason", "") or "")
    merged["idle_tick_id"] = str(merged.get("idle_tick_id", "") or "")
    merged["selected_goal_id"] = str(merged.get("selected_goal_id", "") or "")
    merged["initiative_id"] = str(merged.get("initiative_id", "") or "")
    merged["action_plan_id"] = str(merged.get("action_plan_id", "") or "")
    merged["observation_id"] = str(merged.get("observation_id", "") or "")
    merged["priority_ids"] = _string_list(merged.get("priority_ids"))
    merged["pressure_ids"] = _string_list(merged.get("pressure_ids"))
    merged["claim_candidate_ids"] = _string_list(merged.get("claim_candidate_ids"))
    merged["budget_snapshot"] = _dict_value(merged.get("budget_snapshot"))
    merged["policy_snapshot"] = internal_autonomy_policy_from_payload(
        merged.get("policy_snapshot")
    ).to_dict()
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return InternalAutonomyRunRecord(**merged)


def recurring_priority_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> RecurringPriorityRecord:
    defaults = RecurringPriorityRecord(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=RecurringPriorityRecord,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["priority_id"] = str(merged.get("priority_id", "") or uuid4().hex)
    merged["session_id"] = session_id
    merged["title"] = str(merged.get("title", "") or "")
    merged["description"] = str(merged.get("description", "") or "")
    merged["status"] = normalize_recurring_priority_status(
        str(merged.get("status", "candidate") or "candidate")
    )
    merged["first_observed_at"] = str(merged.get("first_observed_at", "") or "")
    merged["last_observed_at"] = str(merged.get("last_observed_at", "") or "")
    merged["recurrence_count"] = _nonnegative_int(merged.get("recurrence_count"))
    merged["source_candidate_ids"] = _string_list(merged.get("source_candidate_ids"))
    merged["source_selected_goal_ids"] = _string_list(
        merged.get("source_selected_goal_ids")
    )
    merged["source_initiative_ids"] = _string_list(merged.get("source_initiative_ids"))
    merged["pressure_evidence_refs"] = _string_list(
        merged.get("pressure_evidence_refs")
    )
    merged["revision_history"] = _dict_list(merged.get("revision_history"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return RecurringPriorityRecord(**merged)


def motive_pressure_evidence_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> MotivePressureEvidence:
    defaults = MotivePressureEvidence(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=MotivePressureEvidence,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["pressure_id"] = str(merged.get("pressure_id", "") or uuid4().hex)
    merged["session_id"] = session_id
    merged["priority_id"] = str(merged.get("priority_id", "") or "")
    merged["pressure_class"] = normalize_motive_pressure_class(
        str(merged.get("pressure_class", "recurrence") or "recurrence")
    )
    merged["observed_at"] = str(merged.get("observed_at", "") or "")
    for key in (
        "strength",
        "recurrence_count",
        "persistence_score",
        "competition_score",
        "revision_score",
    ):
        merged[key] = _bounded_score(merged.get(key))
    merged["interruption_returned"] = bool(merged.get("interruption_returned", False))
    merged["supporting_context"] = _string_list(merged.get("supporting_context"))
    merged["counterevidence"] = _string_list(merged.get("counterevidence"))
    merged["source_tick_ids"] = _string_list(merged.get("source_tick_ids"))
    merged["source_action_audit_ids"] = _string_list(
        merged.get("source_action_audit_ids")
    )
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return MotivePressureEvidence(**merged)


def claim_candidate_from_payload(
    *, payload: dict[str, Any], session_id: str
) -> LongitudinalSelfReportClaimCandidate:
    defaults = LongitudinalSelfReportClaimCandidate(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=LongitudinalSelfReportClaimCandidate,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["claim_candidate_id"] = str(
        merged.get("claim_candidate_id", "") or uuid4().hex
    )
    merged["session_id"] = session_id
    merged["claim_class"] = normalize_longitudinal_claim_class(
        str(merged.get("claim_class", "desire_like") or "desire_like")
    )
    merged["proposed_claim"] = str(merged.get("proposed_claim", "") or "")
    merged["status"] = normalize_longitudinal_claim_status(
        str(merged.get("status", "candidate") or "candidate")
    )
    merged["confidence"] = _bounded_score(merged.get("confidence"))
    merged["threshold"] = _bounded_score(merged.get("threshold"))
    merged["allowed"] = bool(merged.get("allowed", False))
    if merged["claim_class"] == "desire_like" and merged["status"] != "allowed":
        merged["allowed"] = False
    if merged["allowed"] and merged["status"] not in {"allowed", "eligible_for_review"}:
        merged["allowed"] = False
    merged["supporting_priority_ids"] = _string_list(
        merged.get("supporting_priority_ids")
    )
    merged["supporting_pressure_ids"] = _string_list(
        merged.get("supporting_pressure_ids")
    )
    merged["blocked_reasons"] = _string_list(merged.get("blocked_reasons"))
    merged["required_evidence"] = _string_list(merged.get("required_evidence"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return LongitudinalSelfReportClaimCandidate(**merged)


def internal_autonomy_runs_from_payload(
    payload: Any,
    *,
    session_id: str,
    autonomy_session_id: str = "",
) -> list[InternalAutonomyRunRecord]:
    if not isinstance(payload, list):
        return []
    return [
        internal_autonomy_run_from_payload(
            payload=item,
            session_id=session_id,
            autonomy_session_id=autonomy_session_id,
        )
        for item in payload
        if isinstance(item, dict)
    ]


def recurring_priorities_from_payload(
    payload: Any,
    *,
    session_id: str,
) -> list[RecurringPriorityRecord]:
    if not isinstance(payload, list):
        return []
    return [
        recurring_priority_from_payload(payload=item, session_id=session_id)
        for item in payload
        if isinstance(item, dict)
    ]


def motive_pressure_evidence_list_from_payload(
    payload: Any,
    *,
    session_id: str,
) -> list[MotivePressureEvidence]:
    if not isinstance(payload, list):
        return []
    return [
        motive_pressure_evidence_from_payload(payload=item, session_id=session_id)
        for item in payload
        if isinstance(item, dict)
    ]


def claim_candidates_from_payload(
    payload: Any,
    *,
    session_id: str,
) -> list[LongitudinalSelfReportClaimCandidate]:
    if not isinstance(payload, list):
        return []
    return [
        claim_candidate_from_payload(payload=item, session_id=session_id)
        for item in payload
        if isinstance(item, dict)
    ]


def normalize_autonomy_session_status(status: str) -> str:
    normalized = _normalize_token(status)
    if normalized not in AUTONOMY_SESSION_STATUSES:
        return "planned"
    return normalized


def normalize_internal_autonomy_run_status(status: str) -> str:
    normalized = _normalize_token(status)
    if normalized not in INTERNAL_AUTONOMY_RUN_STATUSES:
        return "planned"
    return normalized


def normalize_recurring_priority_status(status: str) -> str:
    normalized = _normalize_token(status)
    if normalized not in RECURRING_PRIORITY_STATUSES:
        return "candidate"
    return normalized


def normalize_motive_pressure_class(pressure_class: str) -> str:
    normalized = _normalize_token(pressure_class)
    if normalized not in MOTIVE_PRESSURE_CLASSES:
        return "recurrence"
    return normalized


def normalize_longitudinal_claim_class(claim_class: str) -> str:
    normalized = _normalize_token(claim_class)
    if normalized not in LONGITUDINAL_CLAIM_CLASSES:
        return "desire_like"
    return normalized


def normalize_longitudinal_claim_status(status: str) -> str:
    normalized = _normalize_token(status)
    if normalized not in LONGITUDINAL_CLAIM_STATUSES:
        return "candidate"
    return normalized


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


def _bounded_score(value: Any) -> int:
    return min(_nonnegative_int(value), 100)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _internal_surfaces_only(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    surfaces = [normalize_action_surface(str(surface)) for surface in value]
    return [surface for surface in surfaces if surface in INTERNAL_ACTIVITY_SURFACES]
