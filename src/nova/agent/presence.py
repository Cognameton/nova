"""Session-scoped presence state for Nova Phase 4."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


PRESENCE_MODES = {
    "conversation",
    "orientation",
    "action_review",
    "maintenance_review",
    "diagnostics",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PresenceState:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    mode: str = "conversation"
    current_focus: str = "open conversation"
    interaction_summary: str = ""
    pending_proposal: dict[str, Any] | None = None
    last_action_status: str | None = None
    visible_uncertainties: list[str] = field(default_factory=list)
    user_confirmations_needed: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonPresenceStore:
    """JSON-backed session presence store."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load(self, *, session_id: str) -> PresenceState:
        path = self.get_presence_path(session_id=session_id)
        if not path.exists():
            presence = default_presence_state(session_id=session_id)
            self.save(presence)
            return presence

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            presence = default_presence_state(session_id=session_id)
            self.save(presence)
            return presence
        if not isinstance(payload, dict):
            presence = default_presence_state(session_id=session_id)
            self.save(presence)
            return presence
        return presence_state_from_payload(payload=payload, session_id=session_id)

    def save(self, presence: PresenceState) -> None:
        presence.mode = normalize_presence_mode(presence.mode)
        presence.updated_at = utc_now_iso()
        path = self.get_presence_path(session_id=presence.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(presence.to_dict(), handle, indent=2, ensure_ascii=False)

    def get_presence_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.presence.json"


def default_presence_state(*, session_id: str) -> PresenceState:
    return PresenceState(
        session_id=session_id,
        mode="conversation",
        current_focus="open conversation",
        interaction_summary="",
        pending_proposal=None,
        last_action_status=None,
        visible_uncertainties=[],
        user_confirmations_needed=[],
        updated_at=utc_now_iso(),
    )


def presence_state_from_payload(*, payload: dict[str, Any], session_id: str) -> PresenceState:
    """Load persisted presence state across minor schema changes."""
    defaults = default_presence_state(session_id=session_id).to_dict()
    allowed_fields = {field_info.name for field_info in fields(PresenceState)}
    merged = {
        key: payload.get(key, default_value)
        for key, default_value in defaults.items()
        if key in allowed_fields
    }
    merged["session_id"] = session_id
    merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
    merged["mode"] = normalize_presence_mode(str(merged.get("mode", "conversation")))
    merged["current_focus"] = str(merged.get("current_focus", "open conversation"))
    merged["interaction_summary"] = str(merged.get("interaction_summary", ""))
    merged["updated_at"] = str(merged.get("updated_at", ""))
    merged["visible_uncertainties"] = _string_list(merged.get("visible_uncertainties"))
    merged["user_confirmations_needed"] = _string_list(
        merged.get("user_confirmations_needed")
    )
    pending_proposal = merged.get("pending_proposal")
    if pending_proposal is not None and not isinstance(pending_proposal, dict):
        merged["pending_proposal"] = None
    if merged.get("last_action_status") is not None:
        merged["last_action_status"] = str(merged["last_action_status"])
    return PresenceState(**merged)


def normalize_presence_mode(mode: str) -> str:
    normalized = (mode or "conversation").strip().lower()
    if normalized not in PRESENCE_MODES:
        return "conversation"
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
