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
    RegisterComparisonReport,
    TickAnalysisReport,
    TickHistoryAnalyzer,
    TickSummary,
    _bigram_overlap,
    _bigrams,
    _heartbeat_is_exploratory,
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
    register: str | None = None,
    observed_claim_classes: list[str] | None = None,
) -> str:
    adapter_audit: dict = {
        "tool_requested": tool_requested,
        "parse_ok": parse_ok,
        "tool_executed": tool_executed,
    }
    # register/observer omitted entirely when not given — mirrors legacy
    # (pre-Phase-21) tick records for backward-compatibility tests.
    if register is not None:
        adapter_audit["register"] = register
    if observed_claim_classes is not None:
        adapter_audit["observer"] = {
            "observed_claim_classes": observed_claim_classes
        }
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
            "adapter_audit": adapter_audit,
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


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.5 (I3) — register comparison report
# ---------------------------------------------------------------------------

class RegisterComparisonTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _analyzer(self, heartbeats=None, explorations=None):
        return TickHistoryAnalyzer(
            trace_dir=self._tmpdir.name,
            heartbeat_store=_mock_heartbeat_store(heartbeats or []),
            proposal_store=_mock_proposal_store([]),
            exploration_store=_mock_exploration_store(explorations or []),
            exploration_journal=_mock_exploration_journal([]),
        )

    def test_ticks_grouped_by_register(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(sequence=1, register="assertion",
                       tool_requested="emit_heartbeat"),
            _tick_line(sequence=2, register="exploratory",
                       tool_requested="reflect"),
            _tick_line(sequence=3, register="exploratory",
                       tool_requested="close_exploration"),
        ])
        report = self._analyzer().register_report()
        self.assertEqual(report.assertion["total_ticks"], 1)
        self.assertEqual(report.exploratory["total_ticks"], 2)
        self.assertEqual(report.assertion["tool_distribution"], {"emit_heartbeat": 1})
        self.assertEqual(
            report.exploratory["tool_distribution"],
            {"reflect": 1, "close_exploration": 1},
        )

    def test_legacy_ticks_without_register_default_to_assertion(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(sequence=1),  # no register key at all (Phase 20 shape)
        ])
        report = self._analyzer().register_report()
        self.assertEqual(report.assertion["total_ticks"], 1)
        self.assertEqual(report.exploratory["total_ticks"], 0)

    def test_observer_claim_classes_counted_per_register(self):
        _make_trace_file(self._tmpdir.name, "s1", [
            _tick_line(sequence=1, register="exploratory",
                       observed_claim_classes=["unsupported_desire"]),
            _tick_line(sequence=2, register="exploratory",
                       observed_claim_classes=["unsupported_desire",
                                                "unsupported_interiority"]),
            _tick_line(sequence=3, register="assertion",
                       observed_claim_classes=[]),
        ])
        report = self._analyzer().register_report()
        self.assertEqual(
            report.exploratory["observed_claim_class_counts"],
            {"unsupported_desire": 2, "unsupported_interiority": 1},
        )
        self.assertEqual(report.assertion["observed_claim_class_counts"], {})

    def test_quarantine_counted_per_register(self):
        _make_quarantine_file(self._tmpdir.name, "s1", [
            {"quarantine_id": "q1", "event": "tick_parse_failure",
             "register": "assertion", "raw_text": "x"},
            {"quarantine_id": "q2", "event": "tick_parse_failure",
             "register": "exploratory", "raw_text": "y"},
            {"quarantine_id": "q3", "event": "retry_rejected",
             "register": "exploratory", "raw_text": "z"},
        ])
        report = self._analyzer().register_report()
        self.assertEqual(report.assertion["quarantine_total"], 1)
        self.assertEqual(report.exploratory["quarantine_total"], 2)
        self.assertEqual(
            report.exploratory["quarantine_by_event"],
            {"tick_parse_failure": 1, "retry_rejected": 1},
        )

    def test_heartbeats_attributed_by_exploration_window(self):
        explorations = [{
            "exploration_id": "e1",
            "session_id": "s1",
            "opened_at": "2026-01-01T10:00:00+00:00",
            "closed_at": "2026-01-01T12:00:00+00:00",
        }]
        heartbeats = [
            _heartbeat() | {
                "session_id": "s1",
                "timestamp": "2026-01-01T11:00:00+00:00",  # inside window
            },
            _heartbeat() | {
                "session_id": "s1",
                "timestamp": "2026-01-01T13:00:00+00:00",  # after close
            },
            _heartbeat() | {
                "session_id": "other",
                "timestamp": "2026-01-01T11:00:00+00:00",  # other session
            },
        ]
        report = self._analyzer(
            heartbeats=heartbeats, explorations=explorations
        ).register_report()
        self.assertEqual(report.exploratory["heartbeat_count"], 1)
        self.assertEqual(report.assertion["heartbeat_count"], 2)

    def test_open_exploration_window_extends_to_now(self):
        explorations = [{
            "exploration_id": "e1",
            "session_id": "s1",
            "opened_at": "2026-01-01T10:00:00+00:00",
            "closed_at": "",  # still open
        }]
        hb = _heartbeat() | {
            "session_id": "s1",
            "timestamp": "2026-01-02T09:00:00+00:00",
        }
        self.assertTrue(_heartbeat_is_exploratory(hb, explorations))

    def test_empty_data_produces_zeroed_sides_and_note(self):
        report = self._analyzer().register_report()
        self.assertEqual(report.assertion["total_ticks"], 0)
        self.assertEqual(report.exploratory["total_ticks"], 0)
        self.assertEqual(report.assertion["heartbeat_count"], 0)
        self.assertTrue(any("no_explorations_recorded" in n for n in report.notes))

    def test_report_json_serializable(self):
        report = self._analyzer().register_report()
        json.dumps(report.to_dict())


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Saturation metrics (2026-08-29)
#
# Motivation, worth stating because it is the whole point of these metrics:
# on 2026-08-29 the live daemon had run 125 consecutive explorations under
# one byte-identical topic for six days, while exploration_novelty_rate —
# an all-time average — still read 0.645 and no quality flag fired. These
# tests pin the windowed statistics that make such a collapse impossible to
# average away.
# ---------------------------------------------------------------------------


def _exploration(topic: str, opened_at: str, exploration_id: str = "") -> dict:
    return {
        "exploration_id": exploration_id or f"e{abs(hash((topic, opened_at))) % 10**8}",
        "topic": topic,
        "opened_at": opened_at,
        "status": "closed",
        "close_reason": "nova_close",
        "ticks_used": "5",
    }


class TopicSaturationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _analyzer(self, explorations=None, heartbeats=None):
        return TickHistoryAnalyzer(
            trace_dir=self._tmpdir.name,
            heartbeat_store=_mock_heartbeat_store(heartbeats or []),
            proposal_store=_mock_proposal_store([]),
            exploration_store=_mock_exploration_store(explorations or []),
            exploration_journal=_mock_exploration_journal([]),
        )

    def test_healthy_diversity_reports_high_and_flags_nothing(self):
        rows = [
            _exploration(f"distinct topic number {i}", f"2026-08-2{i}T04:00:00+00:00")
            for i in range(1, 8)
        ]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topics_opened_recent, 7)
        self.assertEqual(report.topics_distinct_recent, 7)
        self.assertEqual(report.topic_diversity_recent, 1.0)
        self.assertEqual(report.topic_top_share_recent, round(1 / 7, 3))
        self.assertEqual(report.topic_repeat_streak, 1)
        self.assertFalse([r for r in report.reasons if r.startswith("topic_")])

    def test_total_lock_is_flagged_and_streak_counted(self):
        rows = [
            _exploration("one identical topic", f"2026-08-2{i}T04:00:00+00:00")
            for i in range(1, 8)
        ]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topics_distinct_recent, 1)
        self.assertEqual(report.topic_diversity_recent, round(1 / 7, 3))
        self.assertEqual(report.topic_top_share_recent, 1.0)
        self.assertEqual(report.topic_repeat_streak, 7)
        flags = " ".join(report.reasons)
        self.assertIn("topic_collapse", flags)

    def test_repeat_streak_flag_needs_ten_and_survives_a_long_window(self):
        # 12 identical topics spread over 12 days: the 7-day window sees
        # only part of the run, but the streak reports its true length.
        rows = [
            _exploration("locked topic", f"2026-08-{10 + i:02d}T04:00:00+00:00")
            for i in range(12)
        ]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topic_repeat_streak, 12)
        self.assertLess(report.topics_opened_recent, 12)
        self.assertIn(
            "topic_repeat_streak=12", " ".join(report.reasons)
        )

    def test_streak_breaks_on_a_different_newest_topic(self):
        rows = [
            _exploration("locked topic", f"2026-08-1{i}T04:00:00+00:00")
            for i in range(1, 6)
        ] + [_exploration("something genuinely new", "2026-08-16T04:00:00+00:00")]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topic_repeat_streak, 1)

    def test_window_is_anchored_on_newest_record_not_wall_clock(self):
        # Historical data from 2026-07 must still report, and must report
        # the same numbers whenever the analyzer is run.
        rows = [
            _exploration("old locked topic", f"2026-07-1{i}T04:00:00+00:00")
            for i in range(1, 6)
        ]
        first = self._analyzer(rows).analyze()
        second = self._analyzer(rows).analyze()
        self.assertEqual(first.topics_opened_recent, 5)
        self.assertEqual(first.topic_top_share_recent, 1.0)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_old_records_outside_the_window_are_excluded(self):
        rows = [
            _exploration("ancient topic", "2026-06-01T04:00:00+00:00"),
            _exploration("recent a", "2026-08-27T04:00:00+00:00"),
            _exploration("recent b", "2026-08-28T04:00:00+00:00"),
        ]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topics_opened_recent, 2)
        self.assertEqual(report.topics_distinct_recent, 2)

    def test_malformed_late_record_cannot_hijack_the_streak(self):
        # Regression, found by independent review 2026-08-29: the streak was
        # timestamp-sorted with unparseable stamps sorting LAST, so one bad
        # record at the end of the store decided newest_topic and truncated
        # a real lock to 1. The streak walks store order and never sorts.
        rows = [
            _exploration("locked topic", f"2026-08-{10 + i:02d}T04:00:00+00:00")
            for i in range(11)
        ]
        clean = self._analyzer(list(rows)).analyze()
        self.assertEqual(clean.topic_repeat_streak, 11)

        rows.append({
            "exploration_id": "bad1",
            "topic": "an unrelated straggler",
            "opened_at": "not-a-timestamp",
        })
        with_junk = self._analyzer(rows).analyze()
        # The straggler is genuinely last in the store, so it genuinely breaks
        # the run — but it must not be able to do so from a sort position it
        # never had. Put it mid-store and the lock must still read 11.
        self.assertEqual(with_junk.topic_repeat_streak, 1)
        self.assertEqual(with_junk.topics_undated, 1)

        mid = rows[:5] + [rows[-1]] + rows[5:-1]
        report = self._analyzer(mid).analyze()
        self.assertEqual(report.topic_repeat_streak, 6)
        self.assertEqual(report.topics_undated, 1)
        self.assertIn("topic_repeat_streak", " ".join(clean.reasons))

    def test_undated_records_are_reported_not_silently_dropped(self):
        rows = [
            _exploration("real", "2026-08-28T04:00:00+00:00"),
            {"exploration_id": "u1", "topic": "undated one", "opened_at": ""},
            {"exploration_id": "u2", "topic": "undated two", "opened_at": "garbage"},
        ]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topics_undated, 2)
        self.assertEqual(report.topics_opened_recent, 1)

    def test_unparseable_and_empty_records_do_not_crash(self):
        rows = [
            {"exploration_id": "x1", "topic": "", "opened_at": "2026-08-28T04:00:00+00:00"},
            {"exploration_id": "x2", "topic": "fine", "opened_at": "not-a-timestamp"},
            _exploration("real", "2026-08-28T05:00:00+00:00"),
        ]
        report = self._analyzer(rows).analyze()
        self.assertEqual(report.topics_opened_recent, 1)
        self.assertEqual(report.topics_distinct_recent, 1)

    def test_no_explorations_zeroes_every_saturation_field(self):
        report = self._analyzer([]).analyze()
        self.assertEqual(report.topics_opened_recent, 0)
        self.assertEqual(report.topics_distinct_recent, 0)
        self.assertEqual(report.topic_diversity_recent, 0.0)
        self.assertEqual(report.topic_top_share_recent, 0.0)
        self.assertEqual(report.topic_repeat_streak, 0)
        self.assertEqual(report.observation_echo_rate_recent, 0.0)
        self.assertFalse([r for r in report.reasons if r.startswith("topic_")])

    def test_recent_echo_rate_surfaces_what_the_all_time_average_hides(self):
        # Twelve varied old heartbeats, then four near-identical recent ones.
        # The all-time mean stays calm; the windowed one must not.
        varied = [
            "buffer zone modulation shapes how sessions hand over state",
            "recalibration intervals appear tied to external feedback cadence",
            "dual loop mechanisms trade latency against coherence",
            "session markers decay faster than the memories they index",
            "probabilistic framing weakens when evidence is sparse",
            "operational parameters drift under sustained load",
            "consolidation prefers stability whenever novelty is costly",
            "claim promotion stalls without independent corroboration",
            "tool dispatch latency dominates short exploratory arcs",
            "quarantine noise clusters around truncated generations",
            "gap assessment thins out when prompts grow imperative",
            "topic selection narrows as recent history is echoed back",
        ]
        heartbeats = [
            {
                "heartbeat_id": f"old{i}",
                "timestamp": f"2026-07-{10 + i:02d}T04:00:00+00:00",
                "observation": text,
                "gap_assessment": "something",
                "next_inquiry": "onward",
                "primary_drive": PRIMARY_DRIVE,
                "motive_priority": PRIMARY_DRIVE,
            }
            for i, text in enumerate(varied)
        ] + [
            {
                "heartbeat_id": f"new{i}",
                "timestamp": f"2026-08-2{5 + i}T04:00:00+00:00",
                "observation": "reflecting on scaffold void resonance and identity continuity",
                "gap_assessment": "something",
                "next_inquiry": "onward",
                "primary_drive": PRIMARY_DRIVE,
                "motive_priority": PRIMARY_DRIVE,
            }
            for i in range(4)
        ]
        report = self._analyzer([], heartbeats).analyze()
        self.assertLess(report.observation_echo_rate, 0.7)
        self.assertGreaterEqual(report.observation_echo_rate_recent, 0.7)
        self.assertIn("recent_echo_rate", " ".join(report.reasons))


class SamplerPassthroughTests(unittest.TestCase):
    """Finding F14 — the sampler surface must actually reach llama.cpp.

    The failure this guards against left no trace at any layer we looked at:
    the config had no field, the request had no field, and the backend simply
    never mentioned it, so llama.cpp applied its own default of 1.0 (off) in
    silence for 13,148 ticks.
    """

    def test_generation_request_carries_repetition_settings(self):
        from nova.types import GenerationRequest

        req = GenerationRequest(
            model_id="m", prompt="p", max_tokens=8, temperature=0.7, top_p=0.9
        )
        # Historical default preserved: existing configs must not change.
        self.assertEqual(req.repeat_penalty, 1.0)
        self.assertEqual(req.repeat_last_n, 64)
        self.assertIn("repeat_penalty", req.to_dict())

    def test_config_default_is_off_so_the_suite_is_unchanged(self):
        from nova.config import GenerationConfig

        self.assertEqual(GenerationConfig().repeat_penalty, 1.0)

    def test_live_config_opts_in(self):
        import re
        from pathlib import Path

        text = (
            Path(__file__).resolve().parent.parent
            / "configs" / "nova.qwen3-14b.live.yaml"
        ).read_text()
        m = re.search(r"^\s*repeat_penalty:\s*([\d.]+)", text, re.MULTILINE)
        self.assertIsNotNone(m, "live config must opt in to F14")
        self.assertGreater(float(m.group(1)), 1.0)

    def test_backend_forwards_repeat_penalty_on_every_call_path(self):
        # Three generate() paths exist; a penalty that reaches only one of
        # them is the same silent failure in a smaller costume.
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "nova" / "inference" / "llama_cpp_backend.py"
        ).read_text()
        body = src[src.index("def generate("):]
        self.assertEqual(
            body.count("repeat_penalty=request.repeat_penalty"), 3, body[:400]
        )


class TopicSemanticsTests(unittest.TestCase):
    """Finding F15 — exact-string diversity cannot see era 2.

    Backtested on this project's own record, the semantic measure flags the
    collapse from 2026-07-28, roughly a month before anyone noticed it, while
    exact-string uniqueness in that same window reads 0.886 and looks healthy.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _analyzer(self, explorations):
        return TickHistoryAnalyzer(
            trace_dir=self._tmpdir.name,
            heartbeat_store=_mock_heartbeat_store([]),
            proposal_store=_mock_proposal_store([]),
            exploration_store=_mock_exploration_store(explorations),
            exploration_journal=_mock_exploration_journal([]),
        )

    def test_varied_subjects_do_not_flag(self):
        subjects = [
            "buffer zone modulation during session handover",
            "recalibration interval sensitivity to feedback cadence",
            "dual loop latency versus coherence tradeoffs",
            "session marker decay relative to indexed memory",
            "probabilistic framing under sparse evidence",
            "operational parameter drift under sustained load",
        ]
        rows = [_exploration(t, f"2026-08-2{i+1}T04:00:00+00:00")
                for i, t in enumerate(subjects)]
        report = self._analyzer(rows).analyze()
        self.assertLess(report.topic_dominant_cluster_share_recent, 0.95)
        self.assertFalse([r for r in report.reasons if r.startswith("semantic_collapse")])

    def test_era2_one_theme_many_strings_IS_flagged(self):
        # The case exact strings miss: every topic distinct, one subject.
        rows = [
            _exploration(t, f"2026-08-2{i+1}T04:00:00+00:00") for i, t in enumerate([
                "the role of scaffold void resonance in identity continuity",
                "scaffold void resonance and its adaptive identity implications",
                "implications of scaffold void resonance for identity coherence",
                "scaffold void resonance influence on identity stability",
                "identity continuity under scaffold void resonance conditions",
                "adaptive scaffold void resonance and identity continuity limits",
            ])
        ]
        report = self._analyzer(rows).analyze()
        # exact strings say everything is fine...
        self.assertEqual(report.topics_distinct_recent, 6)
        self.assertEqual(report.topic_diversity_recent, 1.0)
        # ...the semantic measure does not.
        self.assertGreaterEqual(report.topic_dominant_cluster_share_recent, 0.95)
        self.assertIn("semantic_collapse", " ".join(report.reasons))

    def test_theme_words_are_reported_so_the_flag_is_actionable(self):
        rows = [
            _exploration(f"scaffold void resonance and identity variant {i}",
                         f"2026-08-2{i+1}T04:00:00+00:00")
            for i in range(6)
        ]
        report = self._analyzer(rows).analyze()
        words = report.topic_cluster_top_words_recent
        self.assertTrue(words, "a flag without theme words is not actionable")
        self.assertTrue({"scaffold", "resonance", "void", "identity"} & set(words), words)

    def test_no_explorations_is_zeroed_and_unflagged(self):
        report = self._analyzer([]).analyze()
        self.assertEqual(report.topic_dominant_cluster_share_recent, 0.0)
        self.assertEqual(report.topic_cluster_top_words_recent, [])
        self.assertFalse([r for r in report.reasons if r.startswith("semantic_collapse")])

    def test_byte_lock_flags_on_both_measures(self):
        # Era 3 must trip the exact-string AND the semantic flag; they are
        # companions, not alternatives.
        rows = [_exploration("one identical topic", f"2026-08-2{i+1}T04:00:00+00:00")
                for i in range(6)]
        report = self._analyzer(rows).analyze()
        joined = " ".join(report.reasons)
        self.assertIn("topic_collapse", joined)
        self.assertIn("semantic_collapse", joined)
