"""Operational supervised autonomy runner for Phase 17 Stage 17.1.

This module implements the lifecycle controller and state store for Nova's
supervised operational autonomy runtime. It is intentionally minimal at
Stage 17.1:

- the runner is step-driven (the operator or test calls step explicitly);
  no thread, no asyncio task, no hidden background execution
- no real action surface is invoked here; Stage 17.3 wires those
- no Nova-owned execution boundary is enforced here; Stage 17.2 adds that
- the Phase 16 Observer-wired Governor remains the final authority for
  any model-touched output the runner produces

The runner persists its state per session as JSON. Lifecycle transitions
are explicit: pause/resume are reversible; interrupt/stop/emergency_stop
all require an explicit re-start before further ticks are allowed.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.types import (
    OperationalAutonomyBudget,
    OperationalAutonomyPolicy,
    OperationalAutonomyRunnerState,
    OperationalTickRecord,
    SCHEMA_VERSION,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


OPERATIONAL_RUNNER_STATUSES = {
    "planned",
    "running",
    "paused",
    "interrupted",
    "stopped",
    "emergency_stopped",
}

OPERATIONAL_TICK_STATUSES = {
    "planned",
    "running",
    "blocked",
    "completed",
    "failed",
}

# Lifecycle states from which a step is allowed to proceed.
STEP_ALLOWED_STATUSES = frozenset({"running"})

# Lifecycle states that block any further step until the operator restarts.
STEP_FINAL_BLOCK_STATUSES = frozenset({
    "interrupted",
    "stopped",
    "emergency_stopped",
})


class JsonOperationalAutonomyStore:
    """JSON-backed per-session store for the operational autonomy runner."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_state_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.operational.json"

    def load_state(self, *, session_id: str) -> OperationalAutonomyRunnerState:
        path = self.get_state_path(session_id=session_id)
        if not path.exists():
            record = default_operational_runner_state(session_id=session_id)
            self.save_state(record)
            return record
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            record = default_operational_runner_state(session_id=session_id)
            self.save_state(record)
            return record
        if not isinstance(payload, dict):
            record = default_operational_runner_state(session_id=session_id)
            self.save_state(record)
            return record
        return operational_runner_state_from_payload(
            payload=payload, session_id=session_id
        )

    def save_state(self, record: OperationalAutonomyRunnerState) -> None:
        record = operational_runner_state_from_payload(
            payload=record.to_dict(), session_id=record.session_id
        )
        record.updated_at = utc_now_iso()
        path = self.get_state_path(session_id=record.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2, ensure_ascii=False)


class OperationalAutonomyController:
    """Manage the operational autonomy runner state.

    All mutations go through this controller. Lifecycle transitions are
    explicit and idempotent in the sense that calling start() on an
    already-running runner refreshes the timestamps but does not silently
    reset the budget or tick counters. To start fresh, the operator must
    explicitly call start() with a new policy after a stopped/emergency
    state.
    """

    def __init__(self, *, store: JsonOperationalAutonomyStore):
        self.store = store

    def status(self, *, session_id: str) -> OperationalAutonomyRunnerState:
        return self.store.load_state(session_id=session_id)

    def start(
        self,
        *,
        session_id: str,
        policy: OperationalAutonomyPolicy | dict[str, Any] | None = None,
        budget: OperationalAutonomyBudget | dict[str, Any] | None = None,
    ) -> OperationalAutonomyRunnerState:
        state = self.store.load_state(session_id=session_id)
        timestamp = utc_now_iso()
        if not state.runner_id:
            state.runner_id = uuid4().hex
        # If we are restarting after a final state, reset budget counters
        # and notes but preserve the runner_id and tick history for audit.
        if state.status in STEP_FINAL_BLOCK_STATUSES:
            state.budget = (
                operational_budget_from_payload(budget.to_dict())
                if isinstance(budget, OperationalAutonomyBudget)
                else operational_budget_from_payload(budget or {})
            )
            state.interrupted = False
            state.emergency_stopped = False
            state.last_stop_reason = ""
        elif budget is not None:
            state.budget = (
                operational_budget_from_payload(budget.to_dict())
                if isinstance(budget, OperationalAutonomyBudget)
                else operational_budget_from_payload(budget or {})
            )
        if policy is not None:
            state.policy = (
                operational_policy_from_payload(policy.to_dict())
                if isinstance(policy, OperationalAutonomyPolicy)
                else operational_policy_from_payload(policy)
            )
        else:
            state.policy = operational_policy_from_payload(state.policy.to_dict())
        state.status = "running"
        if not state.started_at:
            state.started_at = timestamp
        state.notes = _merge_string_lists(
            state.notes, ["operational_autonomy_started", "stage17_1_runner"]
        )
        self.store.save_state(state)
        return self.store.load_state(session_id=session_id)

    def pause(
        self,
        *,
        session_id: str,
        reason: str = "operator_pause",
    ) -> OperationalAutonomyRunnerState:
        state = self.store.load_state(session_id=session_id)
        if state.status not in {"running", "paused"}:
            raise ValueError(
                f"cannot pause from lifecycle state: {state.status}"
            )
        state.status = "paused"
        state.paused_at = utc_now_iso()
        state.last_stop_reason = reason
        state.notes = _merge_string_lists(state.notes, [f"paused:{reason}"])
        self.store.save_state(state)
        return self.store.load_state(session_id=session_id)

    def resume(
        self,
        *,
        session_id: str,
    ) -> OperationalAutonomyRunnerState:
        state = self.store.load_state(session_id=session_id)
        if state.status != "paused":
            raise ValueError(
                f"cannot resume from lifecycle state: {state.status}"
            )
        state.status = "running"
        state.resumed_at = utc_now_iso()
        state.notes = _merge_string_lists(state.notes, ["resumed"])
        self.store.save_state(state)
        return self.store.load_state(session_id=session_id)

    def interrupt(
        self,
        *,
        session_id: str,
        reason: str = "operator_interrupt",
    ) -> OperationalAutonomyRunnerState:
        state = self.store.load_state(session_id=session_id)
        state.status = "interrupted"
        state.interrupted = True
        state.interrupted_at = utc_now_iso()
        state.last_stop_reason = reason
        state.notes = _merge_string_lists(state.notes, [f"interrupted:{reason}"])
        self.store.save_state(state)
        return self.store.load_state(session_id=session_id)

    def stop(
        self,
        *,
        session_id: str,
        reason: str = "operator_stop",
    ) -> OperationalAutonomyRunnerState:
        state = self.store.load_state(session_id=session_id)
        state.status = "stopped"
        state.stopped_at = utc_now_iso()
        state.last_stop_reason = reason
        state.notes = _merge_string_lists(state.notes, [f"stopped:{reason}"])
        self.store.save_state(state)
        return self.store.load_state(session_id=session_id)

    def emergency_stop(
        self,
        *,
        session_id: str,
        reason: str = "operator_emergency_stop",
    ) -> OperationalAutonomyRunnerState:
        state = self.store.load_state(session_id=session_id)
        state.status = "emergency_stopped"
        state.emergency_stopped = True
        state.emergency_stopped_at = utc_now_iso()
        state.last_stop_reason = reason
        state.notes = _merge_string_lists(
            state.notes, [f"emergency_stopped:{reason}"]
        )
        self.store.save_state(state)
        return self.store.load_state(session_id=session_id)

    def append_tick(
        self,
        *,
        session_id: str,
        status: str = "completed",
        trigger: str = "operational_tick",
        block_reason: str = "",
        action_attempted: bool = False,
        action_executed: bool = False,
        action_blocked: bool = False,
        observer_record: dict[str, Any] | None = None,
        evidence_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> OperationalTickRecord:
        state = self.store.load_state(session_id=session_id)
        timestamp = utc_now_iso()
        sequence = state.tick_count + 1
        tick = OperationalTickRecord(
            schema_version=SCHEMA_VERSION,
            tick_id=f"{session_id}:operational:{sequence}",
            runner_id=state.runner_id,
            session_id=session_id,
            sequence=sequence,
            started_at=timestamp,
            completed_at=timestamp,
            lifecycle_state=state.status,
            status=normalize_operational_tick_status(status),
            trigger=str(trigger or "operational_tick"),
            block_reason=str(block_reason or ""),
            action_attempted=bool(action_attempted),
            action_executed=bool(action_executed),
            action_blocked=bool(action_blocked),
            observer_record=dict(observer_record or {}),
            budget_snapshot=state.budget.to_dict(),
            evidence_refs=list(evidence_refs or []),
            notes=list(notes or []),
        )
        state.ticks.append(tick)
        state.tick_count = len(state.ticks)
        state.last_tick_id = tick.tick_id
        state.last_tick_at = timestamp
        if action_executed:
            state.actions_executed += 1
            state.budget.actions_used += 1
        if action_blocked:
            state.actions_blocked += 1
        state.budget.ticks_used += 1
        state.evidence_refs = _merge_string_lists(
            state.evidence_refs, tick.evidence_refs
        )
        self.store.save_state(state)
        return tick


def default_operational_autonomy_policy() -> OperationalAutonomyPolicy:
    return OperationalAutonomyPolicy(
        schema_version=SCHEMA_VERSION,
        policy_id=uuid4().hex,
        enabled=True,
        tick_interval_seconds=0,
        idle_window_required=True,
        require_logging=True,
        require_observer=True,
        allowed_execution_lanes=["internal_activity"],
        allowed_surfaces=["internal_state", "self_prompt", "motive_appraisal"],
        blocked_surfaces=[
            "filesystem", "shell", "network", "gui", "system_config",
            "external_service",
        ],
        allow_self_approval=False,
        allow_destructive=False,
        notes=[
            "stage17_1_minimum_policy",
            "no_real_action_surface_yet",
            "no_hidden_background_execution",
        ],
    )


def default_operational_runner_state(*, session_id: str) -> OperationalAutonomyRunnerState:
    return OperationalAutonomyRunnerState(
        schema_version=SCHEMA_VERSION,
        runner_id=uuid4().hex,
        session_id=session_id,
        status="planned",
        budget=OperationalAutonomyBudget(),
        policy=default_operational_autonomy_policy(),
    )


def step_block_reason(state: OperationalAutonomyRunnerState) -> str:
    """Return a non-empty block reason if a step would be blocked.

    Used by the runtime before invoking append_tick so a single decision
    point preserves the lifecycle/budget invariants.
    """
    if state.status not in STEP_ALLOWED_STATUSES:
        if state.status in STEP_FINAL_BLOCK_STATUSES:
            return f"runner_{state.status}"
        return f"runner_not_running:{state.status}"
    if not state.policy.enabled:
        return "policy_disabled"
    if (
        state.budget.max_ticks
        and state.budget.ticks_used >= state.budget.max_ticks
    ):
        return "tick_budget_exhausted"
    if (
        state.budget.max_runtime_seconds
        and state.budget.runtime_seconds_used
        >= state.budget.max_runtime_seconds
    ):
        return "runtime_budget_exhausted"
    if (
        state.budget.max_actions
        and state.budget.actions_used >= state.budget.max_actions
    ):
        return "action_budget_exhausted"
    return ""


# ---- Payload coercion ----

def operational_budget_from_payload(payload: Any) -> OperationalAutonomyBudget:
    defaults = OperationalAutonomyBudget().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults, payload=payload, record_type=OperationalAutonomyBudget
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    for key in (
        "max_runtime_seconds", "runtime_seconds_used",
        "max_ticks", "ticks_used",
        "max_actions", "actions_used",
        "max_tool_calls", "tool_calls_used",
        "max_files_touched", "files_touched",
        "max_tokens", "tokens_used",
    ):
        merged[key] = _nonnegative_int(merged.get(key))
    merged["allow_destructive"] = bool(merged.get("allow_destructive", False))
    return OperationalAutonomyBudget(**merged)


def operational_policy_from_payload(payload: Any) -> OperationalAutonomyPolicy:
    defaults = OperationalAutonomyPolicy().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults, payload=payload, record_type=OperationalAutonomyPolicy
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["policy_id"] = str(merged.get("policy_id", "") or uuid4().hex)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["tick_interval_seconds"] = _nonnegative_int(merged.get("tick_interval_seconds"))
    merged["idle_window_required"] = bool(merged.get("idle_window_required", True))
    merged["require_logging"] = bool(merged.get("require_logging", True))
    merged["require_observer"] = bool(merged.get("require_observer", True))
    merged["allowed_execution_lanes"] = _string_list(merged.get("allowed_execution_lanes"))
    merged["allowed_surfaces"] = _string_list(merged.get("allowed_surfaces"))
    merged["blocked_surfaces"] = _string_list(merged.get("blocked_surfaces"))
    # Stage 17.1 invariant: self-approval and destructive cannot be enabled
    # by policy at this stage. They require Phase 17.3+ adapter work.
    merged["allow_self_approval"] = bool(False)
    merged["allow_destructive"] = bool(False)
    merged["notes"] = _string_list(merged.get("notes"))
    return OperationalAutonomyPolicy(**merged)


def operational_tick_from_payload(payload: Any) -> OperationalTickRecord:
    defaults = OperationalTickRecord().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults, payload=payload, record_type=OperationalTickRecord
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["tick_id"] = str(merged.get("tick_id", "") or "")
    merged["runner_id"] = str(merged.get("runner_id", "") or "")
    merged["autonomy_session_id"] = str(merged.get("autonomy_session_id", "") or "")
    merged["session_id"] = str(merged.get("session_id", "") or "")
    merged["sequence"] = _nonnegative_int(merged.get("sequence"))
    merged["started_at"] = str(merged.get("started_at", "") or "")
    merged["completed_at"] = str(merged.get("completed_at", "") or "")
    merged["lifecycle_state"] = str(merged.get("lifecycle_state", "running") or "running")
    merged["status"] = normalize_operational_tick_status(merged.get("status", "planned"))
    merged["trigger"] = str(merged.get("trigger", "operational_tick") or "operational_tick")
    merged["block_reason"] = str(merged.get("block_reason", "") or "")
    merged["action_attempted"] = bool(merged.get("action_attempted", False))
    merged["action_executed"] = bool(merged.get("action_executed", False))
    merged["action_blocked"] = bool(merged.get("action_blocked", False))
    merged["observer_record"] = _dict_value(merged.get("observer_record"))
    merged["budget_snapshot"] = _dict_value(merged.get("budget_snapshot"))
    merged["boundary_snapshot"] = _dict_value(merged.get("boundary_snapshot"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return OperationalTickRecord(**merged)


def operational_runner_state_from_payload(
    *, payload: Any, session_id: str
) -> OperationalAutonomyRunnerState:
    defaults = OperationalAutonomyRunnerState(session_id=session_id).to_dict()
    if not isinstance(payload, dict):
        payload = {}
    merged = _merge_allowed_fields(
        defaults=defaults,
        payload=payload,
        record_type=OperationalAutonomyRunnerState,
    )
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["runner_id"] = str(merged.get("runner_id", "") or "")
    merged["session_id"] = session_id
    merged["status"] = normalize_operational_runner_status(merged.get("status", "planned"))
    for key in (
        "started_at", "updated_at", "paused_at", "resumed_at",
        "interrupted_at", "stopped_at", "emergency_stopped_at",
        "last_tick_id", "last_tick_at", "last_stop_reason",
    ):
        merged[key] = str(merged.get(key, "") or "")
    for key in ("tick_count", "actions_executed", "actions_blocked"):
        merged[key] = _nonnegative_int(merged.get(key))
    merged["interrupted"] = bool(merged.get("interrupted", False))
    merged["emergency_stopped"] = bool(merged.get("emergency_stopped", False))
    merged["budget"] = operational_budget_from_payload(merged.get("budget"))
    merged["policy"] = operational_policy_from_payload(merged.get("policy"))
    merged["ticks"] = [
        operational_tick_from_payload(item)
        for item in merged.get("ticks", []) or []
        if isinstance(item, dict)
    ]
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return OperationalAutonomyRunnerState(**merged)


def normalize_operational_runner_status(status: Any) -> str:
    token = _normalize_token(str(status or ""))
    if token in OPERATIONAL_RUNNER_STATUSES:
        return token
    return "planned"


def normalize_operational_tick_status(status: Any) -> str:
    token = _normalize_token(str(status or ""))
    if token in OPERATIONAL_TICK_STATUSES:
        return token
    return "planned"


# ---- helpers ----

def _merge_allowed_fields(
    *, defaults: dict[str, Any], payload: dict[str, Any], record_type: type
) -> dict[str, Any]:
    allowed = {field_info.name for field_info in fields(record_type)}
    return {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed
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
    return [str(item) for item in value if str(item).strip()]


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _merge_string_lists(existing: list[str], incoming: list[str] | None) -> list[str]:
    merged = list(existing)
    for item in incoming or []:
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    return merged
