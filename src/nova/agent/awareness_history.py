"""Append-only awareness history for cross-session monitoring revision."""

from __future__ import annotations

import json
from pathlib import Path

from nova.types import AwarenessHistoryEntry


class JsonlAwarenessHistoryStore:
    """Append-only history for awareness-state revision and carryover events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, entry: AwarenessHistoryEntry) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def list_entries(
        self,
        *,
        session_id: str | None = None,
        revision_class: str | None = None,
    ) -> list[AwarenessHistoryEntry]:
        entries = [self._from_payload(payload) for payload in self._load_payloads()]
        if session_id is not None:
            entries = [entry for entry in entries if entry.session_id == session_id]
        if revision_class is not None:
            entries = [entry for entry in entries if entry.revision_class == revision_class]
        return entries

    def stats(self) -> dict[str, int | str]:
        return {
            "channel": "awareness_history",
            "entries": len(self._load_payloads()),
            "path": str(self.path),
        }

    def _load_payloads(self) -> list[dict]:
        payloads: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
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

    def _from_payload(self, payload: dict) -> AwarenessHistoryEntry:
        return AwarenessHistoryEntry(
            schema_version=str(payload.get("schema_version", "1.0") or "1.0"),
            entry_id=str(payload.get("entry_id", "") or ""),
            timestamp=str(payload.get("timestamp", "") or ""),
            session_id=str(payload.get("session_id", "") or ""),
            source_session_id=payload.get("source_session_id"),
            revision_class=str(payload.get("revision_class", "") or ""),
            monitoring_mode=str(payload.get("monitoring_mode", "bounded") or "bounded"),
            self_signals=list(payload.get("self_signals", []) or []),
            world_signals=list(payload.get("world_signals", []) or []),
            active_pressures=list(payload.get("active_pressures", []) or []),
            candidate_goal_signals=list(payload.get("candidate_goal_signals", []) or []),
            dominant_attention=str(payload.get("dominant_attention", "") or ""),
            evidence_refs=list(payload.get("evidence_refs", []) or []),
            provenance=dict(payload.get("provenance", {}) or {}),
        )
