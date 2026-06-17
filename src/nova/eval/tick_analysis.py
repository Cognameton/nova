"""Phase 20 Stage 20.1 — Tick Quality Analyzer.

Reads accumulated operational tick traces and heartbeat records to produce
a TickAnalysisReport. Key signals:

  - Tool distribution across all ticks
  - Heartbeat quality per-record (observation depth, gap assessment, drive alignment)
  - Heartbeat echo rate (bigram overlap between consecutive observations)
  - Self-model proposal counts (pending vs applied)

Echo rate is the primary early-warning signal for the inward loop getting
stuck in a phrase rut rather than deepening inquiry. A rate above 0.7
means consecutive heartbeats share 70%+ of their bigram sequences.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.agent.motive import PRIMARY_DRIVE
from nova.types import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Bigram helpers (consistent with observer.py approach)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "for", "with", "as", "by", "be", "my", "i", "this", "that",
    "its", "are", "was", "has", "have", "not", "but", "so", "do", "me",
})


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z][a-zA-Z\-']{1,}", (text or "").lower())
        if t not in _STOPWORDS
    ]


def _bigrams(text: str) -> set[tuple[str, str]]:
    seq = _tokens(text)
    return set(zip(seq, seq[1:]))


def _bigram_overlap(a: str, b: str) -> float:
    """Fraction of bigrams in the shorter text that appear in the longer.

    A value near 1.0 means the two strings share almost all phrase sequences —
    the model is echoing rather than deepening.
    """
    bg_a = _bigrams(a)
    bg_b = _bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    smaller = bg_a if len(bg_a) <= len(bg_b) else bg_b
    larger = bg_a if len(bg_a) > len(bg_b) else bg_b
    return len(smaller & larger) / max(1, len(smaller))


# ---------------------------------------------------------------------------
# Per-heartbeat quality score
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HeartbeatQualityScore:
    heartbeat_id: str = ""
    session_id: str = ""
    observation_non_empty: bool = False
    gap_assessment_non_empty: bool = False
    next_inquiry_non_empty: bool = False
    drive_aligned: bool = False
    observation_length: int = 0
    score: int = 0  # 0–4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_heartbeat(hb: dict) -> HeartbeatQualityScore:
    """Score one heartbeat record on four binary dimensions (0–4)."""
    obs = str(hb.get("observation", "")).strip()
    gap = str(hb.get("gap_assessment", "")).strip()
    nxt = str(hb.get("next_inquiry", "")).strip()
    drive = str(hb.get("motive_priority", "") or hb.get("primary_drive", "")).strip()

    obs_ok = bool(obs)
    gap_ok = bool(gap)
    nxt_ok = bool(nxt)
    drive_ok = drive == PRIMARY_DRIVE

    return HeartbeatQualityScore(
        heartbeat_id=str(hb.get("heartbeat_id", "")),
        session_id=str(hb.get("session_id", "")),
        observation_non_empty=obs_ok,
        gap_assessment_non_empty=gap_ok,
        next_inquiry_non_empty=nxt_ok,
        drive_aligned=drive_ok,
        observation_length=len(obs),
        score=sum([obs_ok, gap_ok, nxt_ok, drive_ok]),
    )


# ---------------------------------------------------------------------------
# Tick summary (one entry from .operational.jsonl)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TickSummary:
    tick_id: str = ""
    session_id: str = ""
    sequence: int = 0
    status: str = ""
    trigger: str = ""
    tool_requested: str = ""
    parse_ok: bool = False
    tool_executed: bool = False
    block_reason: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tick_summary_from_record(record: dict) -> TickSummary:
    tick = record.get("tick", {}) if "tick" in record else record
    audit = tick.get("adapter_audit") or {}
    return TickSummary(
        tick_id=str(tick.get("tick_id", "")),
        session_id=str(tick.get("session_id", record.get("session_id", ""))),
        sequence=int(tick.get("sequence", 0)),
        status=str(tick.get("status", "")),
        trigger=str(tick.get("trigger", "")),
        tool_requested=str(audit.get("tool_requested", "") or ""),
        parse_ok=bool(audit.get("parse_ok", False)),
        tool_executed=bool(audit.get("tool_executed", False)),
        block_reason=str(tick.get("block_reason", "")),
        completed_at=str(tick.get("completed_at", record.get("timestamp", ""))),
    )


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TickAnalysisReport:
    schema_version: str = SCHEMA_VERSION

    # Tick counts
    total_ticks: int = 0
    completed_ticks: int = 0
    blocked_ticks: int = 0
    parse_ok_count: int = 0
    action_executed_count: int = 0

    # Tool distribution {tool_name: count}
    tool_distribution: dict[str, int] = field(default_factory=dict)

    # Heartbeat quality
    total_heartbeats: int = 0
    heartbeat_quality_avg: float = 0.0      # 0.0–4.0
    gap_assessment_rate: float = 0.0         # fraction with non-empty gap
    observation_echo_rate: float = 0.0       # avg bigram overlap between consecutive obs
    drive_aligned_heartbeats: int = 0
    drive_misaligned_heartbeats: int = 0

    # Self-model proposals
    pending_self_model_proposals: int = 0
    applied_self_model_proposals: int = 0

    # Session coverage
    sessions_with_ticks: list[str] = field(default_factory=list)
    first_tick_at: str = ""
    last_tick_at: str = ""
    first_heartbeat_at: str = ""
    last_heartbeat_at: str = ""

    # Quality flags
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TickHistoryAnalyzer:
    """Read accumulated tick / heartbeat data and produce a TickAnalysisReport.

    Reads from:
      - trace_dir/*.operational.jsonl  (operational tick trace records)
      - heartbeat_store                (HeartbeatRecord JSONL)
      - proposal_store                 (SelfModelProposal JSONL)
    """

    def __init__(
        self,
        *,
        trace_dir: str | Path,
        heartbeat_store=None,
        proposal_store=None,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.heartbeat_store = heartbeat_store
        self.proposal_store = proposal_store

    # ------------------------------------------------------------------
    # Load raw records
    # ------------------------------------------------------------------

    def load_ticks(self) -> list[TickSummary]:
        """Load all tick records from *.operational.jsonl trace files."""
        summaries: list[TickSummary] = []
        if not self.trace_dir.exists():
            return summaries
        for path in sorted(self.trace_dir.glob("*.operational.jsonl")):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    summaries.append(_tick_summary_from_record(record))
            except OSError:
                continue
        summaries.sort(key=lambda s: (s.completed_at, s.sequence))
        return summaries

    def load_heartbeats(self) -> list[dict]:
        """Load all heartbeat records from the heartbeat store."""
        if self.heartbeat_store is None:
            return []
        records = self.heartbeat_store.list_recent(limit=0)
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in records]

    def load_proposals(self) -> list[dict]:
        """Load all self-model proposals from the proposal store."""
        if self.proposal_store is None:
            return []
        try:
            pending = self.proposal_store.list_pending()
            # Also collect applied by reading raw JSONL if available
            all_proposals: list[dict] = []
            raw_path = getattr(self.proposal_store, "_path", None)
            if raw_path is not None:
                try:
                    for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            all_proposals.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    return all_proposals
                except OSError:
                    pass
            return [p.to_dict() if hasattr(p, "to_dict") else p for p in pending]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Echo detection
    # ------------------------------------------------------------------

    def compute_echo_rate(self, heartbeats: list[dict]) -> float:
        """Average bigram overlap between consecutive observation strings.

        High values (> 0.7) indicate the model is recycling phrases rather
        than generating novel inquiry.
        """
        observations = [
            str(hb.get("observation", "")).strip()
            for hb in heartbeats
            if hb.get("observation", "").strip()
        ]
        if len(observations) < 2:
            return 0.0
        overlaps = [
            _bigram_overlap(observations[i], observations[i + 1])
            for i in range(len(observations) - 1)
        ]
        return round(sum(overlaps) / len(overlaps), 3)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self) -> TickAnalysisReport:
        ticks = self.load_ticks()
        heartbeats = self.load_heartbeats()
        proposals = self.load_proposals()

        report = TickAnalysisReport()

        # -- Tick stats --
        report.total_ticks = len(ticks)
        completed = [t for t in ticks if t.status == "completed"]
        blocked = [t for t in ticks if t.status == "blocked"]
        report.completed_ticks = len(completed)
        report.blocked_ticks = len(blocked)
        report.parse_ok_count = sum(1 for t in ticks if t.parse_ok)
        report.action_executed_count = sum(1 for t in ticks if t.tool_executed)

        # -- Tool distribution --
        dist: dict[str, int] = {}
        for t in completed:
            if t.tool_requested:
                dist[t.tool_requested] = dist.get(t.tool_requested, 0) + 1
        report.tool_distribution = dist

        # -- Session coverage --
        sessions = list(dict.fromkeys(t.session_id for t in ticks if t.session_id))
        report.sessions_with_ticks = sessions
        timestamps = [t.completed_at for t in ticks if t.completed_at]
        if timestamps:
            report.first_tick_at = min(timestamps)
            report.last_tick_at = max(timestamps)

        # -- Heartbeat quality --
        report.total_heartbeats = len(heartbeats)
        if heartbeats:
            scores = [score_heartbeat(hb) for hb in heartbeats]
            report.heartbeat_quality_avg = round(
                sum(s.score for s in scores) / len(scores), 2
            )
            report.gap_assessment_rate = round(
                sum(1 for s in scores if s.gap_assessment_non_empty) / len(scores), 3
            )
            report.drive_aligned_heartbeats = sum(
                1 for s in scores if s.drive_aligned
            )
            report.drive_misaligned_heartbeats = len(scores) - report.drive_aligned_heartbeats
            hb_times = [
                str(hb.get("timestamp", "")) for hb in heartbeats
                if hb.get("timestamp")
            ]
            if hb_times:
                report.first_heartbeat_at = min(hb_times)
                report.last_heartbeat_at = max(hb_times)

        # -- Echo rate --
        report.observation_echo_rate = self.compute_echo_rate(heartbeats)

        # -- Proposals --
        pending = [p for p in proposals if not p.get("applied")]
        applied = [p for p in proposals if p.get("applied")]
        report.pending_self_model_proposals = len(pending)
        report.applied_self_model_proposals = len(applied)

        # -- Quality flags --
        reasons: list[str] = []
        if report.total_ticks == 0:
            reasons.append("no_ticks_recorded")
        if report.total_heartbeats == 0:
            reasons.append("no_heartbeats_recorded")
        if report.drive_misaligned_heartbeats > 0:
            reasons.append(
                f"drive_misaligned_heartbeats={report.drive_misaligned_heartbeats}"
            )
        if report.observation_echo_rate >= 0.7:
            reasons.append(
                f"high_echo_rate={report.observation_echo_rate:.3f} "
                "(model recycling observation phrases)"
            )
        if report.gap_assessment_rate < 0.5 and report.total_heartbeats >= 3:
            reasons.append(
                f"low_gap_assessment_rate={report.gap_assessment_rate:.3f} "
                "(gap_assessment empty in majority of heartbeats)"
            )
        tool_count = len(dist)
        if report.completed_ticks >= 5 and tool_count < 2:
            reasons.append(
                f"low_tool_diversity={tool_count} tools across "
                f"{report.completed_ticks} completed ticks"
            )
        report.reasons = reasons

        return report
