"""Session-scoped initiative state for Nova Phase 9."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.types import (
    InitiativeRecord,
    InitiativeState,
    InitiativeTransition,
    SCHEMA_VERSION,
)


INITIATIVE_STATUSES = {
    "pending",
    "approved",
    "active",
    "paused",
    "completed",
    "blocked",
    "abandoned",
}

INITIATIVE_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"approved", "blocked", "abandoned"},
    "approved": {"active", "blocked", "abandoned"},
    "active": {"paused", "completed", "blocked", "abandoned"},
    "paused": {"active", "blocked", "abandoned"},
    "blocked": set(),
    "completed": set(),
    "abandoned": set(),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InitiativeTransitionError(ValueError):
    """Raised when an initiative lifecycle transition is invalid."""


class JsonInitiativeStateStore:
    """JSON-backed session initiative store."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load(self, *, session_id: str) -> InitiativeState:
        path = self.get_initiative_path(session_id=session_id)
        if not path.exists():
            initiative = default_initiative_state(session_id=session_id)
            self.save(initiative)
            return initiative

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            initiative = default_initiative_state(session_id=session_id)
            self.save(initiative)
            return initiative
        if not isinstance(payload, dict):
            initiative = default_initiative_state(session_id=session_id)
            self.save(initiative)
            return initiative
        return initiative_state_from_payload(payload=payload, session_id=session_id)

    def save(self, initiative_state: InitiativeState) -> None:
        initiative_state.updated_at = utc_now_iso()
        path = self.get_initiative_path(session_id=initiative_state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(initiative_state.to_dict(), handle, indent=2, ensure_ascii=False)

    def get_initiative_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.initiative.json"

    def create_record(
        self,
        *,
        initiative_state: InitiativeState,
        title: str,
        goal: str,
        approval_required: bool = True,
        source: str = "runtime",
        evidence_refs: list[str] | None = None,
        related_motive_refs: list[str] | None = None,
        related_self_model_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InitiativeRecord:
        timestamp = utc_now_iso()
        record = InitiativeRecord(
            initiative_id=uuid4().hex,
            intent_id=uuid4().hex,
            session_id=initiative_state.session_id,
            origin_session_id=initiative_state.session_id,
            continued_from_session_id=None,
            continued_from_initiative_id=None,
            title=title.strip(),
            goal=goal.strip(),
            status="pending",
            approval_required=approval_required,
            source=source.strip() or "runtime",
            created_at=timestamp,
            updated_at=timestamp,
            last_transition_at=timestamp,
            evidence_refs=_string_list(evidence_refs),
            related_motive_refs=_string_list(related_motive_refs),
            related_self_model_refs=_string_list(related_self_model_refs),
            continuation_session_ids=[],
            notes=_string_list(notes),
            transitions=[
                InitiativeTransition(
                    from_status="",
                    to_status="pending",
                    reason="initiative_created",
                    timestamp=timestamp,
                    evidence_refs=_string_list(evidence_refs),
                )
            ],
        )
        initiative_state.initiatives.append(record)
        return record

    def resumable_records(self, *, limit: int | None = None) -> list[InitiativeRecord]:
        records: list[InitiativeRecord] = []
        for path in sorted(self.base_dir.glob("*.initiative.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            session_id = path.name.removesuffix(".initiative.json")
            initiative_state = initiative_state_from_payload(payload=payload, session_id=session_id)
            for record in initiative_state.initiatives:
                if normalize_initiative_status(record.status) in {"approved", "active", "paused"}:
                    records.append(record)
        records.sort(key=lambda item: item.updated_at or item.created_at)
        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    def continue_record(
        self,
        *,
        source_session_id: str,
        initiative_id: str,
        target_session_id: str,
        approved_by: str,
        reason: str,
        evidence_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InitiativeRecord:
        source_state = self.load(session_id=source_session_id)
        target_state = self.load(session_id=target_session_id)
        source_record = self._find_record(
            initiative_state=source_state,
            initiative_id=initiative_id,
        )
        source_status = normalize_initiative_status(source_record.status)
        if source_status not in {"approved", "active", "paused"}:
            raise InitiativeTransitionError(
                f"Initiative {initiative_id} in {source_session_id} is not resumable from status {source_status}."
            )
        if not approved_by.strip():
            raise InitiativeTransitionError("Cross-session continuation requires explicit approved_by.")

        timestamp = utc_now_iso()
        if target_session_id in source_record.continuation_session_ids:
            raise InitiativeTransitionError(
                f"Initiative {initiative_id} already continued into session {target_session_id}."
            )

        if source_status == "active":
            self.transition(
                initiative_state=source_state,
                initiative_id=initiative_id,
                to_status="paused",
                reason=f"continued_to_session:{target_session_id}",
                approved_by=approved_by,
                evidence_refs=evidence_refs,
                notes=_merge_string_lists(_string_list(notes), [reason]),
            )
            source_record = self._find_record(
                initiative_state=source_state,
                initiative_id=initiative_id,
            )

        source_record.continuation_session_ids = _merge_string_lists(
            source_record.continuation_session_ids,
            [target_session_id],
        )
        source_record.updated_at = timestamp
        source_record.notes = _merge_string_lists(
            source_record.notes,
            [f"continued_to_session:{target_session_id}"],
        )

        target_record = InitiativeRecord(
            initiative_id=uuid4().hex,
            intent_id=source_record.intent_id,
            session_id=target_session_id,
            origin_session_id=source_record.origin_session_id or source_session_id,
            continued_from_session_id=source_session_id,
            continued_from_initiative_id=source_record.initiative_id,
            title=source_record.title,
            goal=source_record.goal,
            status="approved",
            approval_required=source_record.approval_required,
            approved_by=approved_by.strip(),
            source=source_record.source,
            created_at=timestamp,
            updated_at=timestamp,
            last_transition_at=timestamp,
            evidence_refs=_merge_string_lists(source_record.evidence_refs, evidence_refs),
            related_motive_refs=list(source_record.related_motive_refs),
            related_self_model_refs=list(source_record.related_self_model_refs),
            continuation_session_ids=list(source_record.continuation_session_ids),
            notes=_merge_string_lists(
                source_record.notes,
                _merge_string_lists(_string_list(notes), [reason]),
            ),
            transitions=[
                InitiativeTransition(
                    from_status="",
                    to_status="approved",
                    reason=f"continued_from:{source_session_id}:{source_record.initiative_id}",
                    timestamp=timestamp,
                    approved_by=approved_by.strip(),
                    evidence_refs=_string_list(evidence_refs),
                    notes=_merge_string_lists(_string_list(notes), [reason]),
                )
            ],
        )
        target_state.initiatives.append(target_record)
        target_state.active_initiative_id = _derived_active_initiative_id(target_state.initiatives)

        self.save(source_state)
        self.save(target_state)
        return target_record

    def transition(
        self,
        *,
        initiative_state: InitiativeState,
        initiative_id: str,
        to_status: str,
        reason: str,
        approved_by: str = "",
        evidence_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InitiativeRecord:
        record = self._find_record(initiative_state=initiative_state, initiative_id=initiative_id)
        target_status = normalize_initiative_status(to_status)
        current_status = normalize_initiative_status(record.status)
        allowed = INITIATIVE_TRANSITIONS.get(current_status, set())
        if target_status == current_status:
            raise InitiativeTransitionError(
                f"Initiative {initiative_id} is already in status {current_status}."
            )
        if target_status not in allowed:
            raise InitiativeTransitionError(
                f"Cannot transition initiative {initiative_id} from {current_status} to {target_status}."
            )
        if target_status in {"approved", "active", "paused", "completed"} and not approved_by.strip():
            raise InitiativeTransitionError(
                f"Transition to {target_status} requires explicit approved_by."
            )
        if current_status == "pending" and target_status != "approved" and approved_by.strip():
            raise InitiativeTransitionError(
                "Only approved or active initiative may carry approval attribution."
            )

        timestamp = utc_now_iso()
        record.status = target_status
        if approved_by.strip():
            record.approved_by = approved_by.strip()
        record.updated_at = timestamp
        record.last_transition_at = timestamp
        if evidence_refs is not None:
            record.evidence_refs = _merge_string_lists(record.evidence_refs, evidence_refs)
        if notes is not None:
            record.notes = _merge_string_lists(record.notes, notes)
        record.transitions.append(
            InitiativeTransition(
                from_status=current_status,
                to_status=target_status,
                reason=reason.strip(),
                timestamp=timestamp,
                approved_by=approved_by.strip(),
                evidence_refs=_string_list(evidence_refs),
                notes=_string_list(notes),
            )
        )
        initiative_state.active_initiative_id = self._active_initiative_id(initiative_state)
        return record

    def _find_record(self, *, initiative_state: InitiativeState, initiative_id: str) -> InitiativeRecord:
        for record in initiative_state.initiatives:
            if record.initiative_id == initiative_id:
                return record
        raise InitiativeTransitionError(f"Unknown initiative id: {initiative_id}")

    def _active_initiative_id(self, initiative_state: InitiativeState) -> str | None:
        for record in reversed(initiative_state.initiatives):
            if normalize_initiative_status(record.status) == "active":
                return record.initiative_id
        return None


def default_initiative_state(*, session_id: str) -> InitiativeState:
    return InitiativeState(
        session_id=session_id,
        active_initiative_id=None,
        initiatives=[],
        updated_at=utc_now_iso(),
    )


def initiative_state_from_payload(*, payload: dict[str, Any], session_id: str) -> InitiativeState:
    defaults = default_initiative_state(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(InitiativeState)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["session_id"] = session_id
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["updated_at"] = str(merged.get("updated_at", ""))
    merged["active_initiative_id"] = _normalize_optional_string(merged.get("active_initiative_id"))
    merged["initiatives"] = _initiative_records(merged.get("initiatives"), session_id=session_id)

    known_ids = {record.initiative_id for record in merged["initiatives"] if record.initiative_id}
    if merged["active_initiative_id"] not in known_ids:
        merged["active_initiative_id"] = _derived_active_initiative_id(merged["initiatives"])
    return InitiativeState(**merged)


def initiative_record_from_payload(*, payload: dict[str, Any], session_id: str) -> InitiativeRecord:
    defaults = InitiativeRecord(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(InitiativeRecord)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["session_id"] = session_id
    merged["initiative_id"] = str(merged.get("initiative_id", "") or uuid4().hex)
    merged["intent_id"] = str(merged.get("intent_id", "") or uuid4().hex)
    merged["title"] = str(merged.get("title", "") or "")
    merged["goal"] = str(merged.get("goal", "") or "")
    merged["origin_session_id"] = str(merged.get("origin_session_id", "") or session_id)
    merged["continued_from_session_id"] = _normalize_optional_string(
        merged.get("continued_from_session_id")
    )
    merged["continued_from_initiative_id"] = _normalize_optional_string(
        merged.get("continued_from_initiative_id")
    )
    merged["status"] = normalize_initiative_status(str(merged.get("status", "pending")))
    merged["approval_required"] = bool(merged.get("approval_required", True))
    merged["approved_by"] = str(merged.get("approved_by", "") or "")
    merged["source"] = str(merged.get("source", "runtime") or "runtime")
    merged["created_at"] = str(merged.get("created_at", "") or "")
    merged["updated_at"] = str(merged.get("updated_at", "") or "")
    merged["last_transition_at"] = str(merged.get("last_transition_at", "") or "")
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["related_motive_refs"] = _string_list(merged.get("related_motive_refs"))
    merged["related_self_model_refs"] = _string_list(merged.get("related_self_model_refs"))
    merged["continuation_session_ids"] = _string_list(merged.get("continuation_session_ids"))
    merged["notes"] = _string_list(merged.get("notes"))
    merged["transitions"] = _initiative_transitions(merged.get("transitions"))
    return InitiativeRecord(**merged)


def initiative_transition_from_payload(payload: dict[str, Any]) -> InitiativeTransition:
    defaults = InitiativeTransition().to_dict()
    allowed_fields = {field_info.name for field_info in fields(InitiativeTransition)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["from_status"] = normalize_initiative_status(str(merged.get("from_status", "") or ""), allow_empty=True)
    merged["to_status"] = normalize_initiative_status(str(merged.get("to_status", "pending") or "pending"))
    merged["reason"] = str(merged.get("reason", "") or "")
    merged["timestamp"] = str(merged.get("timestamp", "") or "")
    merged["approved_by"] = str(merged.get("approved_by", "") or "")
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["notes"] = _string_list(merged.get("notes"))
    return InitiativeTransition(**merged)


def normalize_initiative_status(status: str, *, allow_empty: bool = False) -> str:
    normalized = (status or "").strip().lower()
    if allow_empty and not normalized:
        return ""
    if normalized not in INITIATIVE_STATUSES:
        return "pending"
    return normalized


def _initiative_records(value: Any, *, session_id: str) -> list[InitiativeRecord]:
    if not isinstance(value, list):
        return []
    records: list[InitiativeRecord] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        records.append(initiative_record_from_payload(payload=item, session_id=session_id))
    return records


def _initiative_transitions(value: Any) -> list[InitiativeTransition]:
    if not isinstance(value, list):
        return []
    transitions: list[InitiativeTransition] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        transitions.append(initiative_transition_from_payload(item))
    return transitions


def _derived_active_initiative_id(records: list[InitiativeRecord]) -> str | None:
    for record in reversed(records):
        if normalize_initiative_status(record.status) == "active":
            return record.initiative_id
    return None


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


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
