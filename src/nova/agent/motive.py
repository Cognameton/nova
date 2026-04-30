"""Session-scoped motive state for Nova Phase 7."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.types import MotiveState, SCHEMA_VERSION


CLAIM_POSTURES = {
    "conservative",
    "evidence-backed",
    "uncertainty-marked",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MotiveStateView:
    """Compatibility shim for test/debug consumers expecting dataclass behavior."""

    state: MotiveState

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.state)


class JsonMotiveStateStore:
    """JSON-backed session motive store."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load(self, *, session_id: str) -> MotiveState:
        path = self.get_motive_path(session_id=session_id)
        if not path.exists():
            motive = default_motive_state(session_id=session_id)
            self.save(motive)
            return motive

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            motive = default_motive_state(session_id=session_id)
            self.save(motive)
            return motive
        if not isinstance(payload, dict):
            motive = default_motive_state(session_id=session_id)
            self.save(motive)
            return motive
        return motive_state_from_payload(payload=payload, session_id=session_id)

    def save(self, motive: MotiveState) -> None:
        motive.claim_posture = normalize_claim_posture(motive.claim_posture)
        motive.updated_at = utc_now_iso()
        path = self.get_motive_path(session_id=motive.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(motive.to_dict(), handle, indent=2, ensure_ascii=False)

    def get_motive_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.motive.json"


def default_motive_state(*, session_id: str) -> MotiveState:
    return MotiveState(
        session_id=session_id,
        current_priorities=["maintain coherent continuity with the user"],
        active_tensions=[],
        local_goals=["understand the current interaction well before claiming more than the evidence supports"],
        claim_posture="conservative",
        evidence_refs=[],
        updated_at=utc_now_iso(),
    )


def motive_state_from_payload(*, payload: dict[str, Any], session_id: str) -> MotiveState:
    defaults = default_motive_state(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(MotiveState)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["session_id"] = session_id
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["claim_posture"] = normalize_claim_posture(str(merged.get("claim_posture", "conservative")))
    merged["current_priorities"] = _string_list(merged.get("current_priorities"))
    merged["active_tensions"] = _string_list(merged.get("active_tensions"))
    merged["local_goals"] = _string_list(merged.get("local_goals"))
    merged["evidence_refs"] = _string_list(merged.get("evidence_refs"))
    merged["updated_at"] = str(merged.get("updated_at", ""))
    return MotiveState(**merged)


def normalize_claim_posture(claim_posture: str) -> str:
    normalized = (claim_posture or "conservative").strip().lower()
    if normalized not in CLAIM_POSTURES:
        return "conservative"
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
