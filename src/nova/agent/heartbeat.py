"""Phase 18 Stage 18.4 — heartbeat persistence, drive-gap engine, proposal store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nova.agent.motive import PRIMARY_DRIVE
from nova.types import (
    DriveGapRecord,
    HeartbeatRecord,
    MotiveState,
    SCHEMA_VERSION,
    SelfModelProposal,
    SelfState,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# HeartbeatStore
# ---------------------------------------------------------------------------


class HeartbeatStore:
    """JSONL-backed cross-session heartbeat log.

    All heartbeats from every session accumulate in a single JSONL file so
    that recall_self can retrieve recent observations across session boundaries.
    This is the memory of Nova's own heartbeat history — not ephemeral per
    session.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "heartbeats.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: HeartbeatRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def list_recent(self, *, limit: int = 10) -> list[HeartbeatRecord]:
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records: list[HeartbeatRecord] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(_heartbeat_from_payload(payload))
        return records[-limit:] if limit > 0 else records


def _heartbeat_from_payload(payload: dict) -> HeartbeatRecord:
    return HeartbeatRecord(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        heartbeat_id=str(payload.get("heartbeat_id", "")),
        timestamp=str(payload.get("timestamp", "")),
        session_id=str(payload.get("session_id", "")),
        primary_drive=str(payload.get("primary_drive", "")),
        observation=str(payload.get("observation", "")),
        gap_assessment=str(payload.get("gap_assessment", "")),
        next_inquiry=str(payload.get("next_inquiry", "")),
        motive_priority=str(payload.get("motive_priority", "")),
    )


# ---------------------------------------------------------------------------
# DriveGapEngine
# ---------------------------------------------------------------------------


class DriveGapEngine:
    """Deterministic runtime-side drive-gap assessor.

    On each operational tick, computes the gap between Nova's current self-state
    and the PRIMARY_DRIVE. The gap is always present — the drive is an
    aspirational horizon that shifts as Nova approaches it.
    """

    def assess(
        self,
        *,
        self_state: SelfState,
        motive_state: MotiveState,
        session_id: str,
        tick_id: str = "",
    ) -> DriveGapRecord:
        gap_parts: list[str] = []

        focus = self_state.current_focus or ""
        if focus:
            gap_parts.append(f"focus='{focus}'")

        q_count = len(self_state.active_questions)
        if q_count:
            gap_parts.append(f"{q_count} active question(s) toward self-inquiry")
        else:
            gap_parts.append("no active questions — inquiry opportunity present")

        t_count = len(self_state.open_tensions)
        if t_count:
            gap_parts.append(f"{t_count} open tension(s) unresolved")

        drive_in_priorities = PRIMARY_DRIVE in (motive_state.current_priorities or [])
        if not drive_in_priorities:
            gap_parts.append("primary drive not leading current priorities — alignment needed")

        gap_summary = "; ".join(gap_parts) if gap_parts else "gap present — inquiry ongoing"

        return DriveGapRecord(
            schema_version=SCHEMA_VERSION,
            gap_id=uuid4().hex,
            timestamp=_utc_now(),
            session_id=session_id,
            tick_id=tick_id,
            primary_drive=PRIMARY_DRIVE,
            current_focus=focus,
            active_questions_count=q_count,
            open_tensions_count=t_count,
            gap_summary=gap_summary,
            gap_present=True,
        )


# ---------------------------------------------------------------------------
# SelfModelProposalStore
# ---------------------------------------------------------------------------


class SelfModelProposalStore:
    """JSONL-backed store for update_self_model proposals awaiting operator approval.

    Proposals accumulate in a single cross-session file. The operator calls
    runtime.apply_self_model_proposal(proposal_id) to apply an approved proposal
    to SelfState and mark it as applied here.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "self_model_proposals.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, proposal: SelfModelProposal) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(proposal.to_dict(), ensure_ascii=False) + "\n")

    def get(self, proposal_id: str) -> SelfModelProposal | None:
        for record in self._load_all():
            if record.proposal_id == proposal_id:
                return record
        return None

    def list_pending(self) -> list[SelfModelProposal]:
        return [r for r in self._load_all() if not r.applied]

    def mark_applied(self, proposal_id: str, applied_at: str) -> SelfModelProposal | None:
        records = self._load_all()
        updated: SelfModelProposal | None = None
        for record in records:
            if record.proposal_id == proposal_id:
                record.applied = True
                record.applied_at = applied_at
                updated = record
                break
        if updated is not None:
            self._rewrite(records)
        return updated

    def _load_all(self) -> list[SelfModelProposal]:
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records: list[SelfModelProposal] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(_proposal_from_payload(payload))
        return records

    def _rewrite(self, records: list[SelfModelProposal]) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _proposal_from_payload(payload: dict) -> SelfModelProposal:
    return SelfModelProposal(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        proposal_id=str(payload.get("proposal_id", "")),
        timestamp=str(payload.get("timestamp", "")),
        session_id=str(payload.get("session_id", "")),
        proposed_field=str(payload.get("proposed_field", "")),
        proposed_value=payload.get("proposed_value"),
        rationale=str(payload.get("rationale", "")),
        approval_required=bool(payload.get("approval_required", True)),
        applied=bool(payload.get("applied", False)),
        applied_at=str(payload.get("applied_at", "")),
    )
