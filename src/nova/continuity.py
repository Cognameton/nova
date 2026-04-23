"""Session continuity summaries for Nova Phase 4."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class SessionContinuitySummary:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    current_focus: str = ""
    interaction_summary: str = ""
    pending_proposal: dict[str, Any] | None = None
    last_action_status: str | None = None
    unresolved_items: list[str] = field(default_factory=list)
    recent_user_inputs: list[str] = field(default_factory=list)
    recent_assistant_outputs: list[str] = field(default_factory=list)
    recent_action_attempts: list[dict[str, Any]] = field(default_factory=list)
    recent_memory_activity: list[dict[str, Any]] = field(default_factory=list)
    next_pickup: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionContinuityBuilder:
    """Build a read-only summary of current session continuity."""

    def __init__(self, *, runtime):
        self.runtime = runtime

    def build(
        self,
        *,
        recent_turn_limit: int = 5,
        recent_action_limit: int = 5,
        recent_memory_limit: int = 5,
    ) -> SessionContinuitySummary:
        presence = self.runtime.presence_status()
        recent_turns = self.runtime.session_store.recent_turns(
            session_id=presence.session_id,
            limit=recent_turn_limit,
        )
        actions = self._load_recent_actions(
            session_id=presence.session_id,
            limit=recent_action_limit,
        )
        memory_activity = self._load_recent_memory_activity(
            session_id=presence.session_id,
            limit=recent_memory_limit,
        )
        unresolved = self._unresolved_items(
            pending_proposal=presence.pending_proposal,
            visible_uncertainties=presence.visible_uncertainties,
            confirmations=presence.user_confirmations_needed,
        )
        return SessionContinuitySummary(
            session_id=presence.session_id,
            current_focus=presence.current_focus,
            interaction_summary=presence.interaction_summary,
            pending_proposal=presence.pending_proposal,
            last_action_status=presence.last_action_status,
            unresolved_items=unresolved,
            recent_user_inputs=[turn.user_text for turn in recent_turns],
            recent_assistant_outputs=[turn.final_answer for turn in recent_turns],
            recent_action_attempts=actions,
            recent_memory_activity=memory_activity,
            next_pickup=self._next_pickup(
                current_focus=presence.current_focus,
                unresolved_items=unresolved,
            ),
        )

    def _load_recent_actions(self, *, session_id: str, limit: int) -> list[dict[str, Any]]:
        proposal_path = Path(self.runtime.trace_logger.trace_dir) / f"{session_id}.proposals.jsonl"
        action_path = Path(self.runtime.trace_logger.trace_dir) / f"{session_id}.actions.jsonl"
        records: list[dict[str, Any]] = []
        for payload in _read_jsonl(proposal_path):
            proposal = payload.get("proposal")
            if not isinstance(proposal, dict):
                continue
            records.append(
                {
                    "timestamp": payload.get("timestamp", ""),
                    "kind": "proposal",
                    "goal": proposal.get("goal", ""),
                    "status": proposal.get("disposition", ""),
                    "executed": False,
                    "reason": proposal.get("reason", ""),
                    "requires_approval": bool(proposal.get("requires_approval", False)),
                }
            )
        for payload in _read_jsonl(action_path):
            execution = payload.get("execution")
            if not isinstance(execution, dict):
                continue
            records.append(
                {
                    "timestamp": payload.get("timestamp", ""),
                    "kind": "execution",
                    "goal": execution.get("goal", ""),
                    "status": execution.get("status", ""),
                    "executed": bool(execution.get("executed", False)),
                    "reason": execution.get("reason", ""),
                    "rollback_applied": bool(execution.get("rollback_applied", False)),
                }
            )
        records.sort(key=lambda item: str(item.get("timestamp", "") or ""))
        return records[-max(0, limit):]

    def _load_recent_memory_activity(self, *, session_id: str, limit: int) -> list[dict[str, Any]]:
        path = Path(self.runtime.trace_logger.trace_dir) / f"{session_id}.jsonl"
        records: list[dict[str, Any]] = []
        for payload in _read_jsonl(path):
            timestamp = payload.get("timestamp", "")
            turn_id = payload.get("turn_id", "")
            for event in payload.get("persisted_memory_events", []) or []:
                if not isinstance(event, dict):
                    continue
                records.append(
                    {
                        "timestamp": timestamp,
                        "turn_id": turn_id,
                        "event_id": event.get("event_id", ""),
                        "channel": event.get("channel", ""),
                        "kind": event.get("kind", ""),
                        "summary": event.get("summary"),
                        "text": event.get("text", ""),
                        "retention": event.get("retention", ""),
                    }
                )
        return records[-max(0, limit):]

    def _unresolved_items(
        self,
        *,
        pending_proposal: dict[str, Any] | None,
        visible_uncertainties: list[str],
        confirmations: list[str],
    ) -> list[str]:
        items = list(visible_uncertainties) + list(confirmations)
        if pending_proposal:
            goal = str(pending_proposal.get("goal", "") or "").strip()
            if goal:
                items.append(f"Pending proposal: {goal}")
        return items

    def _next_pickup(self, *, current_focus: str, unresolved_items: list[str]) -> list[str]:
        if unresolved_items:
            return unresolved_items[:3]
        if current_focus:
            return [f"Continue focus: {current_focus}"]
        return ["No unresolved session items recorded."]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
