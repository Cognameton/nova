"""Session-scoped presence state for Nova Phase 4."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return PresenceState(**payload)

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


def normalize_presence_mode(mode: str) -> str:
    normalized = (mode or "conversation").strip().lower()
    if normalized not in PRESENCE_MODES:
        return "conversation"
    return normalized
