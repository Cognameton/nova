"""Append-only identity history store for revision-aware self-model records."""

from __future__ import annotations

import json
from pathlib import Path

from nova.types import IdentityHistoryEntry


class JsonlIdentityHistoryStore:
    """Append-only history for governed self-model revision events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, entry: IdentityHistoryEntry) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def list_entries(
        self,
        *,
        self_model_status: str | None = None,
        governance_scope: str | None = None,
    ) -> list[IdentityHistoryEntry]:
        entries = [self._from_payload(payload) for payload in self._load_payloads()]
        latest_by_source: dict[str, IdentityHistoryEntry] = {}
        for entry in entries:
            if not entry.source_event_id:
                continue
            latest_by_source[entry.source_event_id] = entry
        if latest_by_source:
            entries = list(latest_by_source.values())
        if self_model_status is not None:
            entries = [entry for entry in entries if entry.self_model_status == self_model_status]
        if governance_scope is not None:
            entries = [entry for entry in entries if entry.governance_scope == governance_scope]
        return entries

    def stats(self) -> dict[str, int | str]:
        return {
            "channel": "identity_history",
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

    def _from_payload(self, payload: dict) -> IdentityHistoryEntry:
        return IdentityHistoryEntry(
            schema_version=str(payload.get("schema_version", "1.0") or "1.0"),
            entry_id=str(payload.get("entry_id", "") or ""),
            timestamp=str(payload.get("timestamp", "") or ""),
            session_id=str(payload.get("session_id", "") or ""),
            source_event_id=str(payload.get("source_event_id", "") or ""),
            governance_scope=str(payload.get("governance_scope", "") or ""),
            claim_axis=str(payload.get("claim_axis", "") or ""),
            self_model_status=str(payload.get("self_model_status", "") or ""),
            revision_class=str(payload.get("revision_class", "") or ""),
            text=str(payload.get("text", "") or ""),
            superseded_by=payload.get("superseded_by"),
            prior_event_ids=list(payload.get("prior_event_ids", []) or []),
            provenance=dict(payload.get("provenance", {}) or {}),
        )
