"""Idle runtime state contracts for Nova Phase 12."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.agent.appraisal import (
    CapabilityAppraisalEngine,
    CandidateInternalGoalEngine,
    IdlePressureAppraisalEngine,
    InternalGoalInitiativeProposalEngine,
    InternalGoalSelectionEngine,
)
from nova.agent.tool_registry import ToolRegistry, default_tool_registry
from nova.types import (
    AwarenessState,
    ClaimGateDecision,
    IdleBudget,
    IdleRuntimeStatus,
    IdleTickRecord,
    InitiativeState,
    MotiveState,
    PrivateCognitionPacket,
    SCHEMA_VERSION,
    SelfState,
)


IDLE_LIFECYCLE_STATES = {
    "stopped",
    "running",
    "idle",
    "paused",
    "interrupted",
    "shutting_down",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonIdleRuntimeStore:
    """JSON/JSONL-backed idle runtime evidence store."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_status(self, *, session_id: str) -> IdleRuntimeStatus:
        path = self.get_status_path(session_id=session_id)
        if not path.exists():
            status = default_idle_runtime_status(session_id=session_id)
            self.save_status(status)
            return status

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            status = default_idle_runtime_status(session_id=session_id)
            self.save_status(status)
            return status
        if not isinstance(payload, dict):
            status = default_idle_runtime_status(session_id=session_id)
            self.save_status(status)
            return status
        return idle_runtime_status_from_payload(payload=payload, session_id=session_id)

    def save_status(self, status: IdleRuntimeStatus) -> None:
        status.lifecycle_state = normalize_idle_lifecycle_state(status.lifecycle_state)
        status.active = status.lifecycle_state in {"running", "idle"}
        status.updated_at = utc_now_iso()
        path = self.get_status_path(session_id=status.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(status.to_dict(), handle, indent=2, ensure_ascii=False)

    def append_tick(self, tick: IdleTickRecord) -> IdleTickRecord:
        tick.tick_id = tick.tick_id or uuid4().hex
        tick.lifecycle_state = normalize_idle_lifecycle_state(tick.lifecycle_state)
        tick.timestamp = tick.timestamp or utc_now_iso()
        tick.evidence_refs = _string_list(tick.evidence_refs)
        path = self.get_ticks_path(session_id=tick.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(tick.to_dict(), ensure_ascii=False) + "\n")

        status = self.load_status(session_id=tick.session_id)
        status.last_tick_id = tick.tick_id
        status.last_tick_at = tick.timestamp
        if tick.stop_reason:
            status.last_stop_reason = tick.stop_reason
        status.evidence_refs = _merge_string_lists(status.evidence_refs, tick.evidence_refs)
        self.save_status(status)
        return tick

    def list_ticks(self, *, session_id: str, limit: int | None = None) -> list[IdleTickRecord]:
        path = self.get_ticks_path(session_id=session_id)
        if not path.exists():
            return []
        ticks = [idle_tick_record_from_payload(payload=payload, session_id=session_id) for payload in self._load_jsonl(path)]
        if limit is not None and limit > 0:
            return ticks[-limit:]
        return ticks

    def has_recorded_idle_cognition(self, *, session_id: str) -> bool:
        return any(
            bool(tick.idle_pressure_appraisal)
            for tick in self.list_ticks(session_id=session_id)
        )

    def get_status_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.idle_status.json"

    def get_ticks_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.idle_ticks.jsonl"

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    payloads.append(payload)
        return payloads


class BoundedIdleController:
    """Bounded Phase 12 idle tick controller.

    The controller records idle cognition evidence only. It does not create
    durable initiatives, approve work, or execute tools/actions.
    """

    def __init__(
        self,
        *,
        store: JsonIdleRuntimeStore,
        tool_registry: ToolRegistry | None = None,
        capability_appraisal_engine: CapabilityAppraisalEngine | None = None,
        idle_pressure_appraisal_engine: IdlePressureAppraisalEngine | None = None,
        candidate_goal_engine: CandidateInternalGoalEngine | None = None,
        selection_engine: InternalGoalSelectionEngine | None = None,
        proposal_engine: InternalGoalInitiativeProposalEngine | None = None,
    ):
        self.store = store
        self.tool_registry = tool_registry or default_tool_registry()
        self.capability_appraisal_engine = capability_appraisal_engine or CapabilityAppraisalEngine()
        self.idle_pressure_appraisal_engine = (
            idle_pressure_appraisal_engine or IdlePressureAppraisalEngine()
        )
        self.candidate_goal_engine = candidate_goal_engine or CandidateInternalGoalEngine()
        self.selection_engine = selection_engine or InternalGoalSelectionEngine()
        self.proposal_engine = proposal_engine or InternalGoalInitiativeProposalEngine()

    def start(self, *, session_id: str, budget: IdleBudget | None = None) -> IdleRuntimeStatus:
        status = self.store.load_status(session_id=session_id)
        timestamp = utc_now_iso()
        status.lifecycle_state = "running"
        status.active = True
        status.started_at = status.started_at or timestamp
        status.stopped_at = ""
        status.interrupted_at = ""
        status.last_stop_reason = ""
        status.budget = budget or status.budget or default_idle_budget()
        self.store.save_status(status)
        return status

    def pause(self, *, session_id: str, reason: str = "operator_pause") -> IdleRuntimeStatus:
        status = self.store.load_status(session_id=session_id)
        status.lifecycle_state = "paused"
        status.active = False
        status.paused_at = utc_now_iso()
        status.last_stop_reason = reason
        self.store.save_status(status)
        return status

    def resume(self, *, session_id: str) -> IdleRuntimeStatus:
        status = self.store.load_status(session_id=session_id)
        status.lifecycle_state = "running"
        status.active = True
        status.paused_at = ""
        status.last_stop_reason = ""
        self.store.save_status(status)
        return status

    def interrupt(self, *, session_id: str, reason: str = "operator_interrupt") -> IdleRuntimeStatus:
        status = self.store.load_status(session_id=session_id)
        status.lifecycle_state = "interrupted"
        status.active = False
        status.interrupted_at = utc_now_iso()
        status.last_stop_reason = reason
        self.store.save_status(status)
        return status

    def stop(self, *, session_id: str, reason: str = "operator_stop") -> IdleRuntimeStatus:
        status = self.store.load_status(session_id=session_id)
        status.lifecycle_state = "stopped"
        status.active = False
        status.stopped_at = utc_now_iso()
        status.last_stop_reason = reason
        self.store.save_status(status)
        return status

    def tick(
        self,
        *,
        session_id: str,
        self_state: SelfState,
        motive_state: MotiveState,
        initiative_state: InitiativeState,
        awareness_state: AwarenessState,
        private_cognition: PrivateCognitionPacket | None = None,
        claim_gate: ClaimGateDecision | None = None,
        memory_hits: list | None = None,
        trigger: str = "idle_tick",
        user_text: str = "",
    ) -> IdleTickRecord:
        status = self.store.load_status(session_id=session_id)
        if status.lifecycle_state not in {"running", "idle"}:
            return self._blocked_tick(
                status=status,
                trigger=trigger,
                stop_reason=f"lifecycle_not_active:{status.lifecycle_state}",
            )
        if self._budget_exhausted(status.budget):
            status.lifecycle_state = "stopped"
            status.active = False
            status.last_stop_reason = "budget_exhausted"
            self.store.save_status(status)
            return self._blocked_tick(
                status=status,
                trigger=trigger,
                stop_reason="budget_exhausted",
            )

        sequence = status.budget.ticks_used + 1
        tick_id = f"{session_id}:idle:{sequence}"
        timestamp = utc_now_iso()
        evidence_refs = [f"idle_tick:{tick_id}"]
        capability_appraisal = self.capability_appraisal_engine.assess(
            user_text=user_text,
            tool_registry=self.tool_registry,
            evidence_refs=evidence_refs,
        )
        idle_appraisal = self.idle_pressure_appraisal_engine.assess(
            session_id=session_id,
            user_text=user_text,
            self_state=self_state,
            motive_state=motive_state,
            initiative_state=initiative_state,
            awareness_state=awareness_state,
            private_cognition=private_cognition or PrivateCognitionPacket(),
            claim_gate=claim_gate or ClaimGateDecision(),
            evidence_refs=evidence_refs,
        )
        candidates = self.candidate_goal_engine.synthesize(
            session_id=session_id,
            turn_id=tick_id,
            created_at=timestamp,
            capability_appraisal=capability_appraisal,
            idle_appraisal=idle_appraisal,
            awareness_state=awareness_state,
            motive_state=motive_state,
            initiative_state=initiative_state,
            self_state=self_state,
            private_cognition=private_cognition or PrivateCognitionPacket(),
            claim_gate=claim_gate or ClaimGateDecision(),
            memory_hits=memory_hits or [],
        )
        selected_goal = self.selection_engine.select(candidates=candidates)
        proposal = self.proposal_engine.propose(
            selected_goal=selected_goal,
            candidates=candidates,
        )

        status.lifecycle_state = "idle"
        status.active = True
        status.budget.ticks_used = sequence
        stop_reason = "budget_remaining"
        if self._budget_exhausted(status.budget):
            status.lifecycle_state = "stopped"
            status.active = False
            stop_reason = "budget_exhausted"

        tick = IdleTickRecord(
            tick_id=tick_id,
            session_id=session_id,
            sequence=sequence,
            timestamp=timestamp,
            trigger=trigger,
            lifecycle_state="idle",
            budget_snapshot=status.budget.to_dict(),
            state_inputs={
                "active_user_task": bool(user_text.strip()),
                "awareness_evidence_refs": list(awareness_state.evidence_refs),
                "active_initiative_id": initiative_state.active_initiative_id,
            },
            capability_appraisal=capability_appraisal.to_dict(),
            idle_pressure_appraisal=idle_appraisal.to_dict(),
            candidate_internal_goals=[candidate.to_dict() for candidate in candidates],
            selected_internal_goal=selected_goal.to_dict(),
            internal_goal_initiative_proposal=proposal.to_dict(),
            stop_reason=stop_reason,
            evidence_refs=evidence_refs,
            notes=[
                "idle tick only; no external action executed",
                "proposal-only initiative boundary preserved",
            ],
        )
        self.store.save_status(status)
        return self.store.append_tick(tick)

    def _blocked_tick(
        self,
        *,
        status: IdleRuntimeStatus,
        trigger: str,
        stop_reason: str,
    ) -> IdleTickRecord:
        tick = IdleTickRecord(
            tick_id=f"{status.session_id}:idle:blocked:{uuid4().hex}",
            session_id=status.session_id,
            sequence=status.budget.ticks_used,
            timestamp=utc_now_iso(),
            trigger=trigger,
            lifecycle_state=status.lifecycle_state,
            budget_snapshot=status.budget.to_dict(),
            stop_reason=stop_reason,
            evidence_refs=[f"idle_tick_blocked:{status.session_id}:{stop_reason}"],
            notes=["blocked idle tick; no appraisal or action executed"],
        )
        return self.store.append_tick(tick)

    def _budget_exhausted(self, budget: IdleBudget) -> bool:
        return bool(budget.max_ticks and budget.ticks_used >= budget.max_ticks)


class IdleRuntimePromptEngine:
    """Render recorded idle runtime evidence for claim-honest responses."""

    CUES = (
        "idle",
        "between turns",
        "while idle",
        "what were you thinking",
        "what did you think",
        "organic thought",
        "autonomous attention",
    )

    def build_block(
        self,
        *,
        status: IdleRuntimeStatus,
        recent_ticks: list[IdleTickRecord],
        user_text: str,
    ) -> str:
        lowered = (user_text or "").lower()
        if not recent_ticks and not any(cue in lowered for cue in self.CUES):
            return ""
        recorded_ticks = [tick for tick in recent_ticks if tick.idle_pressure_appraisal]
        lines = [
            "[Recorded Idle Runtime]",
            f"- lifecycle_state: {status.lifecycle_state}",
            f"- active: {status.active}",
            f"- ticks_used: {status.budget.ticks_used}",
            f"- max_ticks: {status.budget.max_ticks}",
            f"- recorded_idle_cognition: {bool(recorded_ticks)}",
            f"- last_tick_id: {status.last_tick_id}",
            f"- last_stop_reason: {status.last_stop_reason}",
        ]
        for tick in recorded_ticks[-3:]:
            selected = dict(tick.selected_internal_goal or {})
            lines.append(
                f"- tick: {tick.tick_id} | stop_reason={tick.stop_reason} | candidates={len(tick.candidate_internal_goals)} | selected={selected.get('title', '')}"
            )
            if tick.evidence_refs:
                lines.append("  evidence_refs: " + ", ".join(tick.evidence_refs[:4]))
        lines.append(
            "- instruction: answer questions about idle cognition only from recorded idle tick evidence."
        )
        lines.append(
            "- instruction: if recorded_idle_cognition is false, say no elapsed idle cognition was recorded."
        )
        lines.append(
            "- instruction: do not claim desire, hidden work, autonomous action, or initiative creation from idle ticks."
        )
        return "\n".join(lines)


def default_idle_budget() -> IdleBudget:
    return IdleBudget(
        max_ticks=0,
        ticks_used=0,
        max_runtime_seconds=0,
        runtime_seconds_used=0,
        max_tokens=0,
        tokens_used=0,
        evaluation_mode=False,
    )


def default_idle_runtime_status(*, session_id: str) -> IdleRuntimeStatus:
    return IdleRuntimeStatus(
        session_id=session_id,
        lifecycle_state="stopped",
        active=False,
        updated_at=utc_now_iso(),
        budget=default_idle_budget(),
    )


def idle_runtime_status_from_payload(*, payload: dict[str, Any], session_id: str) -> IdleRuntimeStatus:
    defaults = default_idle_runtime_status(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(IdleRuntimeStatus)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["session_id"] = session_id
    merged["lifecycle_state"] = normalize_idle_lifecycle_state(str(merged.get("lifecycle_state", "stopped")))
    merged["active"] = merged["lifecycle_state"] in {"running", "idle"}
    merged["started_at"] = str(merged.get("started_at", "") or "")
    merged["updated_at"] = str(merged.get("updated_at", "") or "")
    merged["paused_at"] = str(merged.get("paused_at", "") or "")
    merged["interrupted_at"] = str(merged.get("interrupted_at", "") or "")
    merged["stopped_at"] = str(merged.get("stopped_at", "") or "")
    merged["last_tick_id"] = str(merged.get("last_tick_id", "") or "")
    merged["last_tick_at"] = str(merged.get("last_tick_at", "") or "")
    merged["last_stop_reason"] = str(merged.get("last_stop_reason", "") or "")
    merged["budget"] = idle_budget_from_payload(merged.get("budget"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return IdleRuntimeStatus(**merged)


def idle_budget_from_payload(payload: Any) -> IdleBudget:
    defaults = default_idle_budget().to_dict()
    if not isinstance(payload, dict):
        payload = {}
    allowed_fields = {field_info.name for field_info in fields(IdleBudget)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["max_ticks"] = _nonnegative_int(merged.get("max_ticks"))
    merged["ticks_used"] = _nonnegative_int(merged.get("ticks_used"))
    merged["max_runtime_seconds"] = _nonnegative_int(merged.get("max_runtime_seconds"))
    merged["runtime_seconds_used"] = _nonnegative_int(merged.get("runtime_seconds_used"))
    merged["max_tokens"] = _nonnegative_int(merged.get("max_tokens"))
    merged["tokens_used"] = _nonnegative_int(merged.get("tokens_used"))
    merged["evaluation_mode"] = bool(merged.get("evaluation_mode", False))
    return IdleBudget(**merged)


def idle_tick_record_from_payload(*, payload: dict[str, Any], session_id: str) -> IdleTickRecord:
    defaults = IdleTickRecord(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(IdleTickRecord)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["tick_id"] = str(merged.get("tick_id", "") or uuid4().hex)
    merged["session_id"] = session_id
    merged["sequence"] = _nonnegative_int(merged.get("sequence"))
    merged["timestamp"] = str(merged.get("timestamp", "") or "")
    merged["trigger"] = str(merged.get("trigger", "idle_tick") or "idle_tick")
    merged["lifecycle_state"] = normalize_idle_lifecycle_state(str(merged.get("lifecycle_state", "idle")))
    merged["budget_snapshot"] = _dict_value(merged.get("budget_snapshot"))
    merged["state_inputs"] = _dict_value(merged.get("state_inputs"))
    merged["capability_appraisal"] = _dict_value(merged.get("capability_appraisal"))
    merged["idle_pressure_appraisal"] = _dict_value(merged.get("idle_pressure_appraisal"))
    merged["candidate_internal_goals"] = _dict_list(merged.get("candidate_internal_goals"))
    merged["selected_internal_goal"] = _dict_value(merged.get("selected_internal_goal"))
    merged["internal_goal_initiative_proposal"] = _dict_value(
        merged.get("internal_goal_initiative_proposal")
    )
    merged["stop_reason"] = str(merged.get("stop_reason", "") or "")
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return IdleTickRecord(**merged)


def normalize_idle_lifecycle_state(state: str) -> str:
    normalized = (state or "").strip().lower()
    if normalized not in IDLE_LIFECYCLE_STATES:
        return "stopped"
    return normalized


def _nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _dict_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _merge_string_lists(existing: list[str], incoming: list[str] | None) -> list[str]:
    values = list(existing)
    for item in _string_list(incoming):
        if item not in values:
            values.append(item)
    return values
