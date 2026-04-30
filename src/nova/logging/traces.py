"""Trace logging for Nova 2.0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nova.types import ProbeResult, TraceRecord


class JsonlTraceLogger:
    """Append-only forensic trace logging."""

    def __init__(self, trace_dir: str | Path, probe_path: str | Path | None = None):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.probe_path = Path(probe_path) if probe_path is not None else self.trace_dir / "probes.jsonl"
        self.probe_path.parent.mkdir(parents=True, exist_ok=True)

    def log_trace(self, trace: TraceRecord) -> None:
        trace_path = self.trace_dir / f"{trace.session_id}.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def log_probe(self, probe: ProbeResult) -> None:
        with self.probe_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(probe.to_dict(), ensure_ascii=False) + "\n")

    def log_orientation(
        self,
        *,
        session_id: str,
        snapshot: dict,
        evaluation: dict | None = None,
    ) -> None:
        orientation_path = self.trace_dir / f"{session_id}.orientation.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "snapshot": snapshot,
        }
        if evaluation is not None:
            payload["evaluation"] = evaluation
        with orientation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_tool_action(
        self,
        *,
        session_id: str,
        request: dict,
        decision: dict,
        result: dict,
    ) -> None:
        tool_path = self.trace_dir / f"{session_id}.tools.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "request": request,
            "decision": decision,
            "result": result,
        }
        with tool_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_action_proposal(
        self,
        *,
        session_id: str,
        proposal: dict,
    ) -> None:
        proposal_path = self.trace_dir / f"{session_id}.proposals.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "proposal": proposal,
        }
        with proposal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_action_execution(
        self,
        *,
        session_id: str,
        execution: dict,
    ) -> None:
        action_path = self.trace_dir / f"{session_id}.actions.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "execution": execution,
        }
        with action_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_identity_history(
        self,
        *,
        session_id: str,
        entry: dict,
    ) -> None:
        history_path = self.trace_dir / f"{session_id}.identity-history.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "entry": entry,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
