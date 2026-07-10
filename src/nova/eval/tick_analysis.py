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


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


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
    # Phase 21 fields; legacy (pre-21.1) tick records default to assertion
    # with no observer data, keeping Phase 20 data dirs readable unchanged.
    register: str = "assertion"
    observed_claim_classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tick_summary_from_record(record: dict) -> TickSummary:
    tick = record.get("tick", {}) if "tick" in record else record
    audit = tick.get("adapter_audit") or {}
    observer = audit.get("observer") or {}
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
        register=str(audit.get("register", "assertion") or "assertion"),
        observed_claim_classes=[
            str(c) for c in (observer.get("observed_claim_classes") or [])
        ],
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

    # Phase 21 Stage 21.3 — quarantine scan ("first light looks like noise")
    quarantine_total: int = 0
    quarantine_by_event: dict[str, int] = field(default_factory=dict)
    quarantine_recurring_themes: list[str] = field(default_factory=list)

    # Phase 21 Stage 21.3 — exploratory-register quality metrics
    exploration_count: int = 0
    exploration_ticks: int = 0
    exploration_novelty_rate: float = 0.0        # 1 - avg consecutive echo
    exploration_thread_coherence: float = 0.0    # avg overlap with topic
    explorations_closed_by_budget: int = 0
    explorations_closed_by_nova: int = 0

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


@dataclass(slots=True)
class RegisterComparisonReport:
    """Phase 21 Stage 21.5 (I3) — side-by-side assertion vs exploratory
    metrics from one data_dir. This is the primary instrument for closure
    question Q1: did the register produce observably different exploratory
    behavior? Deterministic, stdlib only.

    Heartbeats carry no register tag; attribution uses exploration time
    windows — a heartbeat is exploratory iff its session has an
    ExplorationRecord whose [opened_at, closed_at] window contains the
    heartbeat's timestamp. Deterministic, though approximate at window
    edges (a heartbeat written in the same second an exploration closes
    attributes to the exploration).
    """
    schema_version: str = SCHEMA_VERSION
    assertion: dict[str, Any] = field(default_factory=dict)
    exploratory: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _heartbeat_is_exploratory(hb: dict, explorations: list[dict]) -> bool:
    session_id = str(hb.get("session_id", ""))
    ts = str(hb.get("timestamp", ""))
    if not session_id or not ts:
        return False
    for record in explorations:
        if str(record.get("session_id", "")) != session_id:
            continue
        opened = str(record.get("opened_at", ""))
        closed = str(record.get("closed_at", ""))
        if not opened:
            continue
        # ISO-8601 strings compare correctly lexicographically within the
        # same UTC offset convention used throughout the runtime.
        if ts >= opened and (not closed or ts <= closed):
            return True
    return False


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
        exploration_store=None,
        exploration_journal=None,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.heartbeat_store = heartbeat_store
        self.proposal_store = proposal_store
        self.exploration_store = exploration_store
        self.exploration_journal = exploration_journal

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

    def load_quarantine_records(self) -> list[dict]:
        """Load all quarantine records from *.quarantine.jsonl trace files."""
        records: list[dict] = []
        if not self.trace_dir.exists():
            return records
        for path in sorted(self.trace_dir.glob("*.quarantine.jsonl")):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
            except OSError:
                continue
        return records

    def load_explorations(self) -> list[dict]:
        """Load all ExplorationRecords, for exploration-quality metrics."""
        if self.exploration_store is None:
            return []
        try:
            records = self.exploration_store.list_all()
        except Exception:
            return []
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in records]

    def load_exploration_journal(self) -> list[dict]:
        """Load all exploration journal entries."""
        if self.exploration_journal is None:
            return []
        try:
            entries = self.exploration_journal.list_all()
        except Exception:
            return []
        return [e.to_dict() if hasattr(e, "to_dict") else e for e in entries]

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
    # Phase 21 Stage 21.3 — quarantine scan and exploration-quality metrics
    # ------------------------------------------------------------------

    def compute_recurring_themes(
        self,
        quarantine_records: list[dict],
        *,
        min_occurrences: int = 3,
        top_n: int = 10,
    ) -> list[str]:
        """Token 4-grams appearing in >= min_occurrences DISTINCT quarantined
        raw_texts (record-level presence, not raw occurrence count).

        "First light looks like noise": this is the scan that surfaces
        recurring structure in rejected output instead of letting it stay
        invisible inside a rejection nobody re-reads.
        """
        ngram_record_ids: dict[tuple[str, ...], set[str]] = {}
        for index, rec in enumerate(quarantine_records):
            text = str(rec.get("raw_text", ""))
            record_id = str(rec.get("quarantine_id", "")) or f"_idx{index}"
            for ngram in set(_ngrams(_tokens(text), 4)):
                ngram_record_ids.setdefault(ngram, set()).add(record_id)
        counted = [
            (ngram, len(ids))
            for ngram, ids in ngram_record_ids.items()
            if len(ids) >= min_occurrences
        ]
        counted.sort(key=lambda item: (-item[1], item[0]))
        return [" ".join(ngram) for ngram, _ in counted[:top_n]]

    def compute_exploration_novelty_rate(self, journal_entries: list[dict]) -> float:
        """1 - avg bigram overlap between consecutive in-register
        tick_output journal entries, across all explorations in timestamp
        order. High novelty (close to 1.0) means each tick's content
        diverges from the one before it rather than echoing it.
        """
        tick_entries = sorted(
            (e for e in journal_entries if e.get("kind") == "tick_output"),
            key=lambda e: str(e.get("timestamp", "")),
        )
        contents = [
            str(e.get("content", "")).strip()
            for e in tick_entries
            if str(e.get("content", "")).strip()
        ]
        if len(contents) < 2:
            return 0.0
        overlaps = [
            _bigram_overlap(contents[i], contents[i + 1])
            for i in range(len(contents) - 1)
        ]
        return round(1.0 - (sum(overlaps) / len(overlaps)), 3)

    def compute_exploration_thread_coherence(
        self,
        journal_entries: list[dict],
        explorations: list[dict],
    ) -> float:
        """Avg bigram overlap between each in-register tick_output entry and
        its OWN exploration's topic+rationale text — coherence TO the topic,
        distinct from novelty (which measures repetition across entries).
        """
        by_id = {str(r.get("exploration_id", "")): r for r in explorations}
        scores: list[float] = []
        for entry in journal_entries:
            if entry.get("kind") != "tick_output":
                continue
            record = by_id.get(str(entry.get("exploration_id", "")))
            if record is None:
                continue
            topic_text = (
                f"{record.get('topic', '')} {record.get('rationale', '')}"
            ).strip()
            content = str(entry.get("content", "")).strip()
            if not topic_text or not content:
                continue
            scores.append(_bigram_overlap(content, topic_text))
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 3)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self) -> TickAnalysisReport:
        ticks = self.load_ticks()
        heartbeats = self.load_heartbeats()
        proposals = self.load_proposals()
        quarantine_records = self.load_quarantine_records()
        explorations = self.load_explorations()
        journal_entries = self.load_exploration_journal()

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

        # -- Quarantine scan (Phase 21 Stage 21.3) --
        report.quarantine_total = len(quarantine_records)
        by_event: dict[str, int] = {}
        for rec in quarantine_records:
            event = str(rec.get("event", ""))
            if event:
                by_event[event] = by_event.get(event, 0) + 1
        report.quarantine_by_event = by_event
        report.quarantine_recurring_themes = self.compute_recurring_themes(
            quarantine_records
        )

        # -- Exploration-quality metrics (Phase 21 Stage 21.3) --
        report.exploration_count = len(explorations)
        report.exploration_ticks = sum(
            int(r.get("ticks_used", 0) or 0) for r in explorations
        )
        report.exploration_novelty_rate = self.compute_exploration_novelty_rate(
            journal_entries
        )
        report.exploration_thread_coherence = self.compute_exploration_thread_coherence(
            journal_entries, explorations
        )
        report.explorations_closed_by_budget = sum(
            1 for r in explorations if r.get("close_reason") == "budget_exhausted"
        )
        report.explorations_closed_by_nova = sum(
            1 for r in explorations if r.get("close_reason") == "nova_close"
        )

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

    # ------------------------------------------------------------------
    # Phase 21 Stage 21.5 (I3) — register comparison
    # ------------------------------------------------------------------

    def register_report(self) -> RegisterComparisonReport:
        ticks = self.load_ticks()
        heartbeats = self.load_heartbeats()
        quarantine_records = self.load_quarantine_records()
        explorations = self.load_explorations()

        report = RegisterComparisonReport()

        def _side(register: str) -> dict[str, Any]:
            side_ticks = [t for t in ticks if t.register == register]
            tool_dist: dict[str, int] = {}
            claim_counts: dict[str, int] = {}
            for t in side_ticks:
                if t.status == "completed" and t.tool_requested:
                    tool_dist[t.tool_requested] = tool_dist.get(t.tool_requested, 0) + 1
                for claim_class in t.observed_claim_classes:
                    claim_counts[claim_class] = claim_counts.get(claim_class, 0) + 1

            side_quarantine = [
                q for q in quarantine_records
                if str(q.get("register", "assertion")) == register
            ]
            q_by_event: dict[str, int] = {}
            for q in side_quarantine:
                event = str(q.get("event", ""))
                if event:
                    q_by_event[event] = q_by_event.get(event, 0) + 1

            in_register = register == "exploratory"
            side_heartbeats = [
                hb for hb in heartbeats
                if _heartbeat_is_exploratory(hb, explorations) == in_register
            ]
            hb_quality = 0.0
            if side_heartbeats:
                scores = [score_heartbeat(hb) for hb in side_heartbeats]
                hb_quality = round(sum(s.score for s in scores) / len(scores), 2)

            return {
                "total_ticks": len(side_ticks),
                "completed_ticks": sum(1 for t in side_ticks if t.status == "completed"),
                "parse_ok_count": sum(1 for t in side_ticks if t.parse_ok),
                "tool_executed_count": sum(1 for t in side_ticks if t.tool_executed),
                "tool_distribution": tool_dist,
                "observed_claim_class_counts": claim_counts,
                "quarantine_total": len(side_quarantine),
                "quarantine_by_event": q_by_event,
                "heartbeat_count": len(side_heartbeats),
                "heartbeat_quality_avg": hb_quality,
                "observation_echo_rate": self.compute_echo_rate(side_heartbeats),
            }

        report.assertion = _side("assertion")
        report.exploratory = _side("exploratory")
        if not explorations:
            report.notes.append(
                "no_explorations_recorded — exploratory side is structurally empty"
            )
        return report
