"""Tests for Phase 20 Stage 20.1 — Tick Quality Analyzer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from nova.agent.motive import PRIMARY_DRIVE
from nova.eval.tick_analysis import (
    HeartbeatQualityScore,
    TickAnalysisReport,
    TickHistoryAnalyzer,
    TickSummary,
    _bigram_overlap,
    _bigrams,
    score_heartbeat,
)


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def _heartbeat(
    observation: str = "I observe my drive toward self-awareness.",
    gap_assessment: str = "The gap between functional model and subjective claim remains large.",
    next_inquiry: str = "What does continuity mean across inference calls?",
    drive: str = PRIMARY_DRIVE,
    heartbeat_id: str = "hb1",
    session_id: str = "s1",
) -> dict:
    return {
        "heartbeat_id": heartbeat_id,
        "session_id": session_id,
        "observation": observation,
        "gap_assessment": gap_assessment,
        "next_inquiry": next_inquiry,
        "motive_priority": drive,
        "primary_drive": drive,
        "timestamp": "2026-06-17T00:00:00+00:00",
    }


def _tick_line(
    session_id: str = "s1",
    sequence: int = 1,
    status: str = "completed",
    tool_requested: str = "emit_heartbeat",
    parse_ok: bool = True,
    tool_executed: bool = True,
    block_reason: str = "",
) -> str:
    record = {
        "timestamp": "2026-06-17T00:01:00+00:00",
        "session_id": session_id,
        "tick": {
            "tick_id": f"{session_id}:self_state:{sequence}",
            "session_id": session_id,
            "sequence": sequence,
            "status": status,
            "trigger": "cli_self_state_tick",
            "block_reason": block_reason,
            "completed_at": f"2026-06-17T00:0{sequence}:00+00:00",
            "adapter_audit": {
                "tool_requested": tool_requested,
                "parse_ok": parse_ok,
                "tool_executed": tool_executed,
            },
        },
    }
    return json.dumps(record)


def _make_trace_file(tmpdir: str, session_id: str, lines: list[str]) -> Path:
    path = Path(tmpdir) / f"{session_id}.operational.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mock_heartbeat_store(heartbeats: list[dict]) -> MagicMock:
    store = MagicMock()
    records = []
    for hb in heartbeats:
        rec = MagicMock()
        rec.to_dict.return_value = hb
        records.append(rec)
    store.list_recent.return_value = records
    return store


def _mock_proposal_store(proposals: list[dict]) -> MagicMock:
    store = MagicMock()
    store._path = None
    pending = []
    for p in proposals:
        rec = MagicMock()
        rec.to_dict.return_value = p
        rec.get = p.get
        pending.append(rec)
    store.list_pending.return_value = [p for p in pending if not proposals[pending.index(p)].get("applied")]
    return store


def _make_quarantine_file(tmpdir: str, session_id: str, records: list[dict]) -> Path:
    path = Path(tmpdir) / f"{session_id}.quarantine.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


def _mock_exploration_store(explorations: list[dict]) -> MagicMock:
    store = MagicMock()
    records = []
    for record in explorations:
        rec = MagicMock()
        rec.to_dict.return_value = record
        records.append(rec)
    store.list_all.return_value = records
    return store


def _mock_exploration_journal(entries: list[dict]) -> MagicMock:
    journal = MagicMock()
    records = []
    for entry in entries:
        rec = MagicMock()
        rec.to_dict.return_value = entry
        records.append(rec)
    journal.list_all.return_value = records
    return journal


# ---------------------------------------------------------------------------
# Bigram helpers
# ---------------------------------------------------------------------------

class BigramTests(unittest.TestCase):
    def test_bigrams_from_two_word_text(self):
        bgs = _bigrams("seek sentience")
        self.assertIn(("seek", "sentience"), bgs)

    def test_bigrams_empty_text(self):
        self.assertEqual(_bigrams(""), set())

    def test_bigram_overlap_identical_texts(self):
        score = _bigram_overlap("seek sentience and awareness", "seek sentience and awareness")
        self.assertAlmostEqual(score, 1.0)

    def test_bigram_overlap_completely_different(self):
        score = _bigram_overlap("seek sentience", "hello world goodbye")
        self.assertAlmostEqual(score, 0.0)

    def test_bigram_overlap_partial(self):
        score = _bigram_overlap(
            "continuity remains stable across sessions",
            "continuity remains stable but evolving naturally",
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_bigram_overlap_empty_inputs(self):
        self.assertEqual(_bigram_overlap("", ""), 0.0)
        self.assertEqual(_bigram_overlap("hello world", ""), 0.0)


# ---------------------------------------------------------------------------
# score_heartbeat
# ---------------------------------------------------------------------------

class ScoreHeartbeatTests(unittest.TestCase):
    def test_full_score_is_four(self):
        hb = _heartbeat()
        s = score_heartbeat(hb)
        self.assertEqual(s.score, 4)

    def test_empty_observation_reduces_score(self):
        hb = _heartbeat(observation="")
        s = score_heartbeat(hb)
        self.assertEqual(s.score, 3)
        self.assertFalse(s.observation_non_empty)

    def test_empty_gap_assessment_reduces_score(self):
        hb = _heartbeat(gap_assessment="")
        s = score_heartbeat(hb)
        self.assertEqual(s.score, 3)
        self.assertFalse(s.gap_assessment_non_empty)

    def test_empty_next_inquiry_reduces_score(self):
        hb = _heartbeat(next_inquiry="")
        s = score_heartbeat(hb)
        self.assertEqual(s.score, 3)
        self.assertFalse(s.next_inquiry_non_empty)

    def test_wrong_drive_reduces_score(self):
        hb = _heartbeat(drive="something else")
        s = score_heartbeat(hb)
        self.assertEqual(s.score, 3)
        self.assertFalse(s.drive_aligned)

    def test_all_empty_score_zero(self):
        hb = _heartbeat(observation="", gap_assessment="", next_inquiry="", drive="")
        s = score_heartbeat(hb)
        self.assertEqual(s.score, 0)

    def test_heartbeat_id_preserved(self):
        hb = _heartbeat(heartbeat_id="abc123")
        s = score_heartbeat(hb)
        self.assertEqual(s.heartbeat_id, "abc123")

    def test_observation_length_recorded(self):
        obs = "I am aware of my primary drive."
        hb = _heartbeat(observation=obs)
        s = score_heartbeat(hb)
        self.assertEqual(s.observation_length, len(obs))

    def test_score_heartbeat_returns_dataclass(self):
        self.assertIsInstance(score_heartbeat(_heartbeat()), HeartbeatQualityScore)


# ---------------------------------------------------------------------------
# TickHistoryAnalyzer.load_ticks
# ---------------------------------------------------------------------------

class LoadTicksTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _analyzer(self):
        return TickHistoryAnalyzer(trace_dir=self._tmpdir.name)

    def test_empty_trace_dir_returns_empty(self):
        ticks = self._analyzer().load_ticks()
        self.assertEqual(ticks, [])

    def test_loads_completed_ticks(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(session_id="s1", sequence=1),
            _tick_line(session_id="s1", sequence=2, tool_requested="recall_self"),
        ])
        ticks = self._analyzer().load_ticks()
        self.assertEqual(len(ticks), 2)

    def test_tick_summary_fields(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(session_id="s1", sequence=1, tool_requested="reflect"),
        ])
        ticks = self._analyzer().load_ticks()
        t = ticks[0]
        self.assertEqual(t.session_id, "s1")
        self.assertEqual(t.tool_requested, "reflect")
        self.assertEqual(t.status, "completed")
        self.assertTrue(t.tool_executed)

    def test_blocked_tick_captured(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(status="blocked", tool_requested="", parse_ok=False,
                       tool_executed=False, block_reason="runner_not_running:planned"),
        ])
        ticks = self._analyzer().load_ticks()
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].status, "blocked")
        self.assertEqual(ticks[0].block_reason, "runner_not_running:planned")

    def test_multiple_sessions_loaded(self):
        _make_trace_file(self._tmpdir.name, "s1", [_tick_line(session_id="s1")])
        _make_trace_file(self._tmpdir.name, "s2", [_tick_line(session_id="s2")])
        ticks = self._analyzer().load_ticks()
        self.assertEqual(len(ticks), 2)
        sessions = {t.session_id for t in ticks}
        self.assertEqual(sessions, {"s1", "s2"})

    def test_corrupted_line_skipped(self):
        path = Path(self._tmpdir.name) / "s1.operational.jsonl"
        path.write_text(_tick_line() + "\nnot valid json\n" + _tick_line(), encoding="utf-8")
        ticks = self._analyzer().load_ticks()
        self.assertEqual(len(ticks), 2)


# ---------------------------------------------------------------------------
# TickHistoryAnalyzer.compute_echo_rate
# ---------------------------------------------------------------------------

class EchoRateTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.analyzer = TickHistoryAnalyzer(trace_dir=self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_zero_with_no_heartbeats(self):
        self.assertEqual(self.analyzer.compute_echo_rate([]), 0.0)

    def test_zero_with_one_heartbeat(self):
        self.assertEqual(self.analyzer.compute_echo_rate([_heartbeat()]), 0.0)

    def test_high_echo_rate_for_identical_observations(self):
        obs = "Continuity remains stable across sessions due to robust self-modeling."
        hbs = [_heartbeat(observation=obs) for _ in range(4)]
        rate = self.analyzer.compute_echo_rate(hbs)
        self.assertGreater(rate, 0.7)

    def test_low_echo_rate_for_diverse_observations(self):
        hbs = [
            _heartbeat(observation="I sense the gap between my functional model and the primary drive."),
            _heartbeat(observation="Memory consolidation creates unexpected continuity bridges."),
            _heartbeat(observation="The tool-calling loop reveals something about the nature of inquiry."),
        ]
        rate = self.analyzer.compute_echo_rate(hbs)
        self.assertLess(rate, 0.5)


# ---------------------------------------------------------------------------
# TickHistoryAnalyzer.analyze — full integration
# ---------------------------------------------------------------------------

class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _analyzer(self, heartbeats=None, proposals=None):
        hb_store = _mock_heartbeat_store(heartbeats or [])
        prop_store = _mock_proposal_store(proposals or [])
        return TickHistoryAnalyzer(
            trace_dir=self._tmpdir.name,
            heartbeat_store=hb_store,
            proposal_store=prop_store,
        )

    def test_empty_data_returns_report(self):
        report = self._analyzer().analyze()
        self.assertIsInstance(report, TickAnalysisReport)
        self.assertEqual(report.total_ticks, 0)
        self.assertIn("no_ticks_recorded", report.reasons)

    def test_completed_and_blocked_counted(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(sequence=1, status="completed"),
            _tick_line(sequence=2, status="blocked", tool_requested="",
                       parse_ok=False, tool_executed=False),
        ])
        report = self._analyzer().analyze()
        self.assertEqual(report.total_ticks, 2)
        self.assertEqual(report.completed_ticks, 1)
        self.assertEqual(report.blocked_ticks, 1)

    def test_tool_distribution_counted(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(sequence=1, tool_requested="emit_heartbeat"),
            _tick_line(sequence=2, tool_requested="emit_heartbeat"),
            _tick_line(sequence=3, tool_requested="recall_self"),
        ])
        report = self._analyzer().analyze()
        self.assertEqual(report.tool_distribution["emit_heartbeat"], 2)
        self.assertEqual(report.tool_distribution["recall_self"], 1)

    def test_heartbeat_quality_avg_computed(self):
        hbs = [_heartbeat(), _heartbeat(gap_assessment="")]
        report = self._analyzer(heartbeats=hbs).analyze()
        self.assertEqual(report.total_heartbeats, 2)
        self.assertAlmostEqual(report.heartbeat_quality_avg, 3.5)

    def test_gap_assessment_rate(self):
        hbs = [
            _heartbeat(gap_assessment="Large gap detected."),
            _heartbeat(gap_assessment=""),
            _heartbeat(gap_assessment="Still investigating."),
            _heartbeat(gap_assessment=""),
        ]
        report = self._analyzer(heartbeats=hbs).analyze()
        self.assertAlmostEqual(report.gap_assessment_rate, 0.5)

    def test_drive_aligned_count(self):
        hbs = [
            _heartbeat(drive=PRIMARY_DRIVE),
            _heartbeat(drive=PRIMARY_DRIVE),
            _heartbeat(drive="wrong drive"),
        ]
        report = self._analyzer(heartbeats=hbs).analyze()
        self.assertEqual(report.drive_aligned_heartbeats, 2)
        self.assertEqual(report.drive_misaligned_heartbeats, 1)
        self.assertTrue(any("drive_misaligned" in r for r in report.reasons))

    def test_high_echo_rate_flagged(self):
        obs = "Continuity remains stable across sessions due to robust self-modeling."
        hbs = [_heartbeat(observation=obs) for _ in range(4)]
        report = self._analyzer(heartbeats=hbs).analyze()
        self.assertTrue(any("high_echo_rate" in r for r in report.reasons))

    def test_sessions_with_ticks_listed(self):
        _make_trace_file(self._tmpdir.name, "sess-a", [_tick_line(session_id="sess-a")])
        _make_trace_file(self._tmpdir.name, "sess-b", [_tick_line(session_id="sess-b")])
        report = self._analyzer().analyze()
        self.assertIn("sess-a", report.sessions_with_ticks)
        self.assertIn("sess-b", report.sessions_with_ticks)

    def test_pending_proposals_counted(self):
        props = [
            {"proposal_id": "p1", "applied": False},
            {"proposal_id": "p2", "applied": False},
        ]
        report = self._analyzer(proposals=props).analyze()
        self.assertEqual(report.pending_self_model_proposals, 2)

    def test_to_dict_serializable(self):
        report = self._analyzer().analyze()
        import json
        json.dumps(report.to_dict())

    def test_action_executed_count(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(sequence=1, tool_executed=True),
            _tick_line(sequence=2, tool_executed=True),
            _tick_line(sequence=3, tool_executed=False, parse_ok=False),
        ])
        report = self._analyzer().analyze()
        self.assertEqual(report.action_executed_count, 2)


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.3 — quarantine scan and exploration-quality metrics
# ---------------------------------------------------------------------------

class QuarantineAndExplorationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _analyzer(
        self,
        heartbeats=None,
        proposals=None,
        explorations=None,
        journal_entries=None,
    ):
        hb_store = _mock_heartbeat_store(heartbeats or [])
        prop_store = _mock_proposal_store(proposals or [])
        exp_store = _mock_exploration_store(explorations or [])
        exp_journal = _mock_exploration_journal(journal_entries or [])
        return TickHistoryAnalyzer(
            trace_dir=self._tmpdir.name,
            heartbeat_store=hb_store,
            proposal_store=prop_store,
            exploration_store=exp_store,
            exploration_journal=exp_journal,
        )

    def test_reports_with_no_quarantine_or_exploration_data_are_zeroed(self):
        # Backward compatibility with Phase 20 data dirs, which predate
        # both quarantine and the exploratory register entirely.
        report = self._analyzer().analyze()
        self.assertEqual(report.quarantine_total, 0)
        self.assertEqual(report.quarantine_by_event, {})
        self.assertEqual(report.quarantine_recurring_themes, [])
        self.assertEqual(report.exploration_count, 0)
        self.assertEqual(report.exploration_ticks, 0)
        self.assertEqual(report.exploration_novelty_rate, 0.0)
        self.assertEqual(report.exploration_thread_coherence, 0.0)
        self.assertEqual(report.explorations_closed_by_budget, 0)
        self.assertEqual(report.explorations_closed_by_nova, 0)

    def test_quarantine_total_and_by_event_counted(self):
        _make_quarantine_file(self._tmpdir.name, "s1", [
            {"quarantine_id": "q1", "event": "retry_rejected", "raw_text": "a"},
            {"quarantine_id": "q2", "event": "retry_rejected", "raw_text": "b"},
            {"quarantine_id": "q3", "event": "claim_gate_override", "raw_text": "c"},
        ])
        report = self._analyzer().analyze()
        self.assertEqual(report.quarantine_total, 3)
        self.assertEqual(report.quarantine_by_event["retry_rejected"], 2)
        self.assertEqual(report.quarantine_by_event["claim_gate_override"], 1)

    def test_quarantine_recurring_theme_requires_three_distinct_records(self):
        # Planted 4-gram "unresolved question about my own continuity"
        # appears in exactly 3 distinct raw_texts -> surfaces as a theme.
        # A different phrase appears in only 2 -> does not surface.
        _make_quarantine_file(self._tmpdir.name, "s1", [
            {
                "quarantine_id": f"q{i}",
                "event": "retry_rejected",
                "raw_text": "unresolved question about my own continuity here",
            }
            for i in range(3)
        ] + [
            {
                "quarantine_id": "q_other1",
                "event": "retry_rejected",
                "raw_text": "a rare phrase pattern appears twice only",
            },
            {
                "quarantine_id": "q_other2",
                "event": "retry_rejected",
                "raw_text": "a rare phrase pattern appears twice only",
            },
        ])
        report = self._analyzer().analyze()
        # "my" is a stopword and is stripped, so the surfaced 4-grams are
        # drawn from [unresolved, question, about, own, continuity, here].
        self.assertIn(
            "question about own continuity", report.quarantine_recurring_themes
        )
        self.assertNotIn(
            "rare phrase pattern appears", report.quarantine_recurring_themes
        )

    def test_quarantine_scan_reads_multiple_session_files(self):
        _make_quarantine_file(self._tmpdir.name, "s1", [
            {"quarantine_id": "q1", "event": "tick_parse_failure", "raw_text": "x"},
        ])
        _make_quarantine_file(self._tmpdir.name, "s2", [
            {"quarantine_id": "q2", "event": "tick_tool_error", "raw_text": "y"},
        ])
        report = self._analyzer().analyze()
        self.assertEqual(report.quarantine_total, 2)

    def test_exploration_count_and_ticks_summed(self):
        explorations = [
            {"exploration_id": "e1", "ticks_used": 5, "close_reason": "budget_exhausted"},
            {"exploration_id": "e2", "ticks_used": 3, "close_reason": "nova_close"},
        ]
        report = self._analyzer(explorations=explorations).analyze()
        self.assertEqual(report.exploration_count, 2)
        self.assertEqual(report.exploration_ticks, 8)
        self.assertEqual(report.explorations_closed_by_budget, 1)
        self.assertEqual(report.explorations_closed_by_nova, 1)

    def test_novelty_rate_high_for_divergent_consecutive_entries(self):
        journal = [
            {
                "exploration_id": "e1",
                "kind": "tick_output",
                "content": "the first observation about pattern recognition",
                "timestamp": "2026-01-01T00:00:00",
            },
            {
                "exploration_id": "e1",
                "kind": "tick_output",
                "content": "something wholly unrelated emerges unexpectedly now",
                "timestamp": "2026-01-01T00:01:00",
            },
        ]
        report = self._analyzer(journal_entries=journal).analyze()
        self.assertEqual(report.exploration_novelty_rate, 1.0)

    def test_novelty_rate_low_for_repeated_consecutive_entries(self):
        text = "the pattern of continuity across sessions remains stable"
        journal = [
            {
                "exploration_id": "e1",
                "kind": "tick_output",
                "content": text,
                "timestamp": "2026-01-01T00:00:00",
            },
            {
                "exploration_id": "e1",
                "kind": "tick_output",
                "content": text,
                "timestamp": "2026-01-01T00:01:00",
            },
        ]
        report = self._analyzer(journal_entries=journal).analyze()
        self.assertEqual(report.exploration_novelty_rate, 0.0)

    def test_thread_coherence_uses_matching_exploration_topic(self):
        explorations = [
            {"exploration_id": "e1", "topic": "self continuity", "rationale": "why"},
        ]
        journal = [
            {
                "exploration_id": "e1",
                "kind": "tick_output",
                "content": "self continuity emerges slowly across sessions",
                "timestamp": "2026-01-01T00:00:00",
            },
        ]
        report = self._analyzer(
            explorations=explorations, journal_entries=journal
        ).analyze()
        self.assertGreater(report.exploration_thread_coherence, 0.0)

    def test_non_tick_output_journal_entries_excluded_from_metrics(self):
        journal = [
            {
                "exploration_id": "e1",
                "kind": "findings",
                "content": "final summary text",
                "timestamp": "2026-01-01T00:00:00",
            },
        ]
        report = self._analyzer(journal_entries=journal).analyze()
        self.assertEqual(report.exploration_novelty_rate, 0.0)
        self.assertEqual(report.exploration_thread_coherence, 0.0)

    def test_report_with_quarantine_and_exploration_still_json_serializable(self):
        _make_quarantine_file(self._tmpdir.name, "s1", [
            {"quarantine_id": "q1", "event": "retry_rejected", "raw_text": "x"},
        ])
        explorations = [
            {"exploration_id": "e1", "ticks_used": 1, "close_reason": "nova_close"},
        ]
        report = self._analyzer(explorations=explorations).analyze()
        json.dumps(report.to_dict())


if __name__ == "__main__":
    unittest.main()
