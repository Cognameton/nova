"""Trace logging for Nova 2.0."""

from __future__ import annotations

import json
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
