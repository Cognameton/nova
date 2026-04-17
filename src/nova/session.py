"""Session persistence for Nova 2.0."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from nova.types import TurnRecord, ValidationResult, RetrievalHit


class JsonlSessionStore:
    """Append-only JSONL session storage."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def start_session(self, *, session_id: str | None = None) -> str:
        active_session_id = session_id or uuid.uuid4().hex
        session_path = self.get_session_path(session_id=active_session_id)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.touch(exist_ok=True)
        return active_session_id

    def append_turn(self, turn: TurnRecord) -> None:
        session_path = self.get_session_path(session_id=turn.session_id)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with session_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")

    def recent_turns(self, *, session_id: str, limit: int) -> list[TurnRecord]:
        session_path = self.get_session_path(session_id=session_id)
        if not session_path.exists():
            return []

        with session_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        turns: list[TurnRecord] = []
        for line in lines[-max(0, limit):]:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            turns.append(self._turn_from_dict(payload))
        return turns

    def get_session_path(self, *, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def _turn_from_dict(self, payload: dict) -> TurnRecord:
        validation_payload = payload.get("validation", {}) or {}
        memory_hits_payload = payload.get("memory_hits", []) or []
        return TurnRecord(
            schema_version=payload.get("schema_version", "1.0"),
            session_id=payload.get("session_id", ""),
            turn_id=payload.get("turn_id", ""),
            timestamp=payload.get("timestamp", ""),
            user_text=payload.get("user_text", ""),
            final_answer=payload.get("final_answer", ""),
            raw_answer=payload.get("raw_answer", ""),
            validation=ValidationResult(
                valid=validation_payload.get("valid", False),
                violations=list(validation_payload.get("violations", []) or []),
                sanitized_text=validation_payload.get("sanitized_text"),
                should_retry=validation_payload.get("should_retry", False),
            ),
            memory_hits=[
                RetrievalHit(
                    channel=item.get("channel", ""),
                    text=item.get("text", ""),
                    score=float(item.get("score", 0.0) or 0.0),
                    kind=item.get("kind"),
                    source_ref=item.get("source_ref"),
                    tags=list(item.get("tags", []) or []),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
                for item in memory_hits_payload
            ],
            prompt_token_estimate=int(payload.get("prompt_token_estimate", 0) or 0),
            completion_token_estimate=payload.get("completion_token_estimate"),
            latency_ms=payload.get("latency_ms"),
            model_id=payload.get("model_id", ""),
            retry_count=int(payload.get("retry_count", 0) or 0),
            notes=dict(payload.get("notes", {}) or {}),
        )
