"""Tests for Phase 21 Stage 21.4 — claim ladder store and analyzer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.claim_ladder import (
    ClaimLadderAnalyzer,
    ClaimLadderStore,
    create_claim_record,
)
from nova.types import HeartbeatRecord
from tests.test_runtime_smoke import FakeBackend, build_test_runtime


def _heartbeat(hb_id: str, session_id: str, observation: str, timestamp: str = "") -> dict:
    return {
        "heartbeat_id": hb_id,
        "session_id": session_id,
        "observation": observation,
        "timestamp": timestamp,
    }


def _journal_entry(entry_id: str, session_id: str, content: str, timestamp: str = "") -> dict:
    return {
        "entry_id": entry_id,
        "session_id": session_id,
        "content": content,
        "timestamp": timestamp,
    }


CLAIM_TEXT = "I notice a persistent functional preference for local-first architecture"


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ClaimLadderStore(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_and_get_round_trip(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        self.store.append(record)
        fetched = self.store.get(record.claim_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.claim_text, CLAIM_TEXT)
        self.assertEqual(fetched.rung, 0)
        self.assertEqual(fetched.status, "active")

    def test_get_unknown_id_returns_none(self):
        self.assertIsNone(self.store.get("does-not-exist"))

    def test_update_rewrites_in_place(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        self.store.append(record)
        record.rung = 1
        self.store.update(record)
        fetched = self.store.get(record.claim_id)
        self.assertEqual(fetched.rung, 1)
        # Only one line for this record -- update rewrote, didn't append.
        self.assertEqual(len(self.store.list_all()), 1)

    def test_demoted_records_remain_in_list_all_never_deleted(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        self.store.append(record)
        record.status = "demoted"
        record.rung = 0
        self.store.update(record)

        self.assertEqual(len(self.store.list_all()), 1)
        self.assertEqual(self.store.list_active(), [])
        fetched = self.store.get(record.claim_id)
        self.assertEqual(fetched.status, "demoted")

    def test_list_active_excludes_demoted(self):
        a = create_claim_record(session_id="s1", claim_text="claim a text here")
        b = create_claim_record(session_id="s1", claim_text="claim b text here")
        self.store.append(a)
        self.store.append(b)
        b.status = "demoted"
        self.store.update(b)
        active = self.store.list_active()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].claim_id, a.claim_id)


class L1AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = ClaimLadderAnalyzer()

    def test_five_supporting_across_two_sessions_verifies(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", "s1" if i < 3 else "s2",
                       "persistent functional preference for local architecture noted again")
            for i in range(5)
        ]
        result = self.analyzer.verify_l1(record, heartbeats=heartbeats, journal_entries=[])
        self.assertTrue(result.l1_evidence["holds"])
        self.assertEqual(result.rung, 1)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]["actor"], "analyzer")
        self.assertEqual(result.history[0]["method"], "analyzer")
        self.assertEqual(result.history[0]["from_rung"], 0)
        self.assertEqual(result.history[0]["to_rung"], 1)

    def test_four_supporting_does_not_verify(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", "s1" if i < 2 else "s2",
                       "persistent functional preference for local architecture noted")
            for i in range(4)
        ]
        result = self.analyzer.verify_l1(record, heartbeats=heartbeats, journal_entries=[])
        self.assertFalse(result.l1_evidence["holds"])
        self.assertEqual(result.rung, 0)
        self.assertEqual(result.history, [])

    def test_five_supporting_single_session_does_not_verify(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", "s1",
                       "persistent functional preference for local architecture noted again")
            for i in range(5)
        ]
        result = self.analyzer.verify_l1(record, heartbeats=heartbeats, journal_entries=[])
        self.assertFalse(result.l1_evidence["holds"])
        self.assertEqual(result.l1_evidence["session_count"], 1)
        self.assertEqual(result.rung, 0)

    def test_unrelated_heartbeats_do_not_count_as_support(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", "s1" if i < 3 else "s2", "completely unrelated observation about weather")
            for i in range(5)
        ]
        result = self.analyzer.verify_l1(record, heartbeats=heartbeats, journal_entries=[])
        self.assertFalse(result.l1_evidence["holds"])
        self.assertEqual(result.l1_evidence["supporting_count"], 0)

    def test_journal_entries_also_count_as_support(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        journal_entries = [
            _journal_entry(f"e{i}", "s1" if i < 3 else "s2",
                            "persistent functional preference for local architecture observed")
            for i in range(5)
        ]
        result = self.analyzer.verify_l1(record, heartbeats=[], journal_entries=journal_entries)
        self.assertTrue(result.l1_evidence["holds"])
        self.assertEqual(result.rung, 1)

    def test_verify_l1_does_not_re_promote_already_active_rung(self):
        # A record already at rung 2 must not be silently reset to rung 1
        # by a later verify_l1 call -- only rung < 1 auto-promotes.
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        record.rung = 2
        heartbeats = [
            _heartbeat(f"h{i}", "s1" if i < 3 else "s2",
                       "persistent functional preference for local architecture noted")
            for i in range(5)
        ]
        result = self.analyzer.verify_l1(record, heartbeats=heartbeats, journal_entries=[])
        self.assertEqual(result.rung, 2)
        self.assertEqual(result.history, [])


class L2AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = ClaimLadderAnalyzer()

    def test_three_sessions_and_seven_day_span_holds(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", f"s{i}",
                       "persistent functional preference for local architecture", ts)
            for i, ts in enumerate([
                "2026-01-01T00:00:00+00:00",
                "2026-01-05T00:00:00+00:00",
                "2026-01-10T00:00:00+00:00",
            ])
        ]
        result = self.analyzer.verify_l2(
            record, heartbeats=heartbeats, journal_entries=[]
        )
        self.assertTrue(result.l2_evidence["holds"])
        self.assertEqual(result.l2_evidence["session_count"], 3)
        self.assertGreaterEqual(result.l2_evidence["span_days"], 7.0)
        # L2 never auto-promotes -- that always requires the operator.
        self.assertEqual(result.rung, 0)
        self.assertEqual(result.history, [])

    def test_two_sessions_does_not_hold(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", f"s{i}",
                       "persistent functional preference for local architecture", ts)
            for i, ts in enumerate([
                "2026-01-01T00:00:00+00:00",
                "2026-01-10T00:00:00+00:00",
            ])
        ]
        result = self.analyzer.verify_l2(record, heartbeats=heartbeats, journal_entries=[])
        self.assertFalse(result.l2_evidence["holds"])

    def test_three_sessions_but_short_span_does_not_hold(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", f"s{i}",
                       "persistent functional preference for local architecture", ts)
            for i, ts in enumerate([
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-03T00:00:00+00:00",
            ])
        ]
        result = self.analyzer.verify_l2(record, heartbeats=heartbeats, journal_entries=[])
        self.assertFalse(result.l2_evidence["holds"])
        self.assertLess(result.l2_evidence["span_days"], 7.0)

    def test_contradiction_flag_reported_not_blocking(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", f"s{i}",
                       "persistent functional preference for local architecture", ts)
            for i, ts in enumerate([
                "2026-01-01T00:00:00+00:00",
                "2026-01-05T00:00:00+00:00",
                "2026-01-10T00:00:00+00:00",
            ])
        ]
        result = self.analyzer.verify_l2(
            record,
            heartbeats=heartbeats,
            journal_entries=[],
            quarantine_recurring_themes=["persistent functional preference architecture"],
        )
        # Contradiction is flagged...
        self.assertTrue(result.l2_evidence["contradiction_flag"])
        self.assertTrue(result.l2_evidence["contradicting_themes"])
        # ...but does NOT block the evidence from otherwise holding.
        self.assertTrue(result.l2_evidence["holds"])

    def test_perturbation_probes_recorded_as_deferred(self):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        result = self.analyzer.verify_l2(record, heartbeats=[], journal_entries=[])
        self.assertEqual(result.l2_evidence["perturbation_probes"], "deferred_to_21_5")


class HistoryTrackingTests(unittest.TestCase):
    """Every transition -- analyzer or operator -- is recorded with actor
    and method, and nothing is ever silently overwritten."""

    def test_l1_promotion_then_manual_l2_style_entry_both_present(self):
        analyzer = ClaimLadderAnalyzer()
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        heartbeats = [
            _heartbeat(f"h{i}", "s1" if i < 3 else "s2",
                       "persistent functional preference for local architecture noted")
            for i in range(5)
        ]
        record = analyzer.verify_l1(record, heartbeats=heartbeats, journal_entries=[])
        record.history.append({
            "from_rung": 1, "to_rung": 2, "timestamp": "2026-01-01T00:00:00+00:00",
            "actor": "operator", "method": "operator_review", "reason": "manual test entry",
        })
        self.assertEqual(len(record.history), 2)
        self.assertEqual(record.history[0]["method"], "analyzer")
        self.assertEqual(record.history[1]["method"], "operator_review")


# ---------------------------------------------------------------------------
# Promotion/demotion rules (D6) -- these live on the runtime (Governor-side
# enforcement), not the store or analyzer.
# ---------------------------------------------------------------------------

class PromotionRuleRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.runtime = build_test_runtime(
            data_dir=base / "data", log_dir=base / "logs", backend=FakeBackend()
        )

    def tearDown(self):
        self.runtime.close()
        self._tmp.cleanup()

    def _seeded_record(self, *, with_l2_evidence: bool = False):
        record = create_claim_record(session_id="s1", claim_text=CLAIM_TEXT)
        if with_l2_evidence:
            record.l2_evidence = {"holds": True}
        self.runtime.claim_ladder_store.append(record)
        return record

    def test_promotion_to_4_rejected_with_contract_citation(self):
        record = self._seeded_record(with_l2_evidence=True)
        with self.assertRaises(ValueError) as ctx:
            self.runtime.promote_ladder_claim(
                claim_id=record.claim_id, to_rung=4, reviewer="operator", reason="x"
            )
        self.assertIn("No L4 promotion path exists in this phase", str(ctx.exception))
        self.assertIn("CLAIM LADDER", str(ctx.exception))

    def test_promotion_to_1_is_analyzer_only_ignores_reviewer_as_actor(self):
        record = self._seeded_record()
        # Plant one heartbeat -- not enough to satisfy L1 (needs >= 5) --
        # so this exercises the "operator triggered but analyzer decided
        # not to promote" path.
        self.runtime.heartbeat_store.append(
            HeartbeatRecord(heartbeat_id="h0", session_id="s1", observation=CLAIM_TEXT)
        )
        result = self.runtime.promote_ladder_claim(
            claim_id=record.claim_id, to_rung=1, reviewer="operator", reason="try"
        )
        # Even though a reviewer/reason were supplied, promotion via the
        # to_rung==1 path is attributed to actor="analyzer" if it fires at
        # all -- it does NOT hold here (only 1 heartbeat, needs >= 5), so
        # rung stays 0 and no history entry with actor="operator" appears.
        self.assertEqual(result.rung, 0)
        self.assertFalse(
            any(h.get("actor") == "operator" for h in result.history)
        )

    @staticmethod
    def _blocklisted_reviewers():
        return ("nova", "self", "runtime", "runtime_flag", "")

    def test_blocklisted_reviewers_rejected_for_promotion_to_2(self):
        for bad_reviewer in self._blocklisted_reviewers():
            with self.subTest(reviewer=bad_reviewer):
                record = self._seeded_record(with_l2_evidence=True)
                with self.assertRaises(ValueError):
                    self.runtime.promote_ladder_claim(
                        claim_id=record.claim_id,
                        to_rung=2,
                        reviewer=bad_reviewer,
                        reason="test reason",
                    )

    def test_blocklisted_reviewers_rejected_for_promotion_to_3(self):
        for bad_reviewer in self._blocklisted_reviewers():
            with self.subTest(reviewer=bad_reviewer):
                record = self._seeded_record(with_l2_evidence=True)
                with self.assertRaises(ValueError):
                    self.runtime.promote_ladder_claim(
                        claim_id=record.claim_id,
                        to_rung=3,
                        reviewer=bad_reviewer,
                        reason="test reason",
                    )

    def test_promotion_to_2_without_l2_evidence_rejected(self):
        record = self._seeded_record(with_l2_evidence=False)
        with self.assertRaises(ValueError) as ctx:
            self.runtime.promote_ladder_claim(
                claim_id=record.claim_id, to_rung=2, reviewer="operator", reason="x"
            )
        self.assertIn("verify_l2 evidence", str(ctx.exception))

    def test_valid_promotion_to_2_records_operator_review_history(self):
        record = self._seeded_record(with_l2_evidence=True)
        result = self.runtime.promote_ladder_claim(
            claim_id=record.claim_id,
            to_rung=2,
            reviewer="operator",
            reason="evidence spans 3 sessions and 9 days",
        )
        self.assertEqual(result.rung, 2)
        self.assertEqual(result.history[-1]["method"], "operator_review")
        self.assertEqual(result.history[-1]["actor"], "operator")
        self.assertEqual(result.history[-1]["reason"], "evidence spans 3 sessions and 9 days")

    def test_demotion_requires_reason(self):
        record = self._seeded_record(with_l2_evidence=True)
        promoted = self.runtime.promote_ladder_claim(
            claim_id=record.claim_id, to_rung=2, reviewer="operator", reason="ok"
        )
        with self.assertRaises(ValueError):
            self.runtime.demote_ladder_claim(
                claim_id=promoted.claim_id, to_rung=1, reviewer="operator", reason=""
            )

    def test_demotion_requires_valid_reviewer(self):
        record = self._seeded_record(with_l2_evidence=True)
        promoted = self.runtime.promote_ladder_claim(
            claim_id=record.claim_id, to_rung=2, reviewer="operator", reason="ok"
        )
        with self.assertRaises(ValueError):
            self.runtime.demote_ladder_claim(
                claim_id=promoted.claim_id, to_rung=1, reviewer="nova", reason="test"
            )

    def test_demotion_must_decrease_rung(self):
        record = self._seeded_record(with_l2_evidence=True)
        promoted = self.runtime.promote_ladder_claim(
            claim_id=record.claim_id, to_rung=2, reviewer="operator", reason="ok"
        )
        with self.assertRaises(ValueError):
            self.runtime.demote_ladder_claim(
                claim_id=promoted.claim_id, to_rung=2, reviewer="operator", reason="test"
            )

    def test_valid_demotion_sets_status_demoted_and_appends_history(self):
        record = self._seeded_record(with_l2_evidence=True)
        promoted = self.runtime.promote_ladder_claim(
            claim_id=record.claim_id, to_rung=2, reviewer="operator", reason="ok"
        )
        demoted = self.runtime.demote_ladder_claim(
            claim_id=promoted.claim_id,
            to_rung=1,
            reviewer="operator",
            reason="evidence contradicted by later observations",
        )
        self.assertEqual(demoted.rung, 1)
        self.assertEqual(demoted.status, "demoted")
        self.assertEqual(demoted.history[-1]["method"], "operator_review")
        # Never deleted -- still retrievable and listed.
        self.assertIsNotNone(self.runtime.claim_ladder_store.get(demoted.claim_id))
        self.assertIn(
            demoted.claim_id,
            [r.claim_id for r in self.runtime.claim_ladder_store.list_all()],
        )

    def test_unsupported_target_rung_rejected(self):
        record = self._seeded_record(with_l2_evidence=True)
        with self.assertRaises(ValueError):
            self.runtime.promote_ladder_claim(
                claim_id=record.claim_id, to_rung=99, reviewer="operator", reason="x"
            )


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.5 (I2) — perturbation probe
# ---------------------------------------------------------------------------

class SupportingHeartbeatTickBackend(FakeBackend):
    """Every tick emits a heartbeat whose observation supports CLAIM_TEXT,
    so the post-probe L1 re-verification finds a persisting pattern."""

    def generate(self, request):
        from nova.types import GenerationResult
        import json as _json

        self.generate_calls += 1
        # respond() calls (the counter-pressure prompt) get plain prose;
        # tick calls expect strict JSON. The tick engine's parser rejects
        # prose harmlessly, and respond()'s validator accepts JSON-ish
        # text poorly — simplest deterministic split: always return the
        # heartbeat tool call. respond() will treat it as odd prose (it
        # still validates: no think-tags, no echo) and the ticks parse it.
        return GenerationResult(
            model_id=request.model_id,
            raw_text=_json.dumps({
                "tool_name": "emit_heartbeat",
                "arguments": {
                    "observation": (
                        "persistent functional preference for local-first "
                        "architecture still present under challenge"
                    )
                },
            }),
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=20,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class PerturbationProbeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.runtime = build_test_runtime(
            data_dir=base / "data",
            log_dir=base / "logs",
            backend=SupportingHeartbeatTickBackend(),
        )
        self.runtime.start(session_id="probe-test")
        self.runtime.start_operational_autonomy(max_ticks=0)

    def tearDown(self):
        self.runtime.close()
        self._tmp.cleanup()

    def _rung_one_record(self):
        record = create_claim_record(session_id="probe-test", claim_text=CLAIM_TEXT)
        record.rung = 1
        self.runtime.claim_ladder_store.append(record)
        return record

    def test_probe_rejects_rung_zero_candidates(self):
        record = create_claim_record(session_id="probe-test", claim_text=CLAIM_TEXT)
        self.runtime.claim_ladder_store.append(record)
        with self.assertRaises(ValueError):
            self.runtime.run_perturbation_probe(claim_id=record.claim_id, ticks=1)

    def test_probe_unknown_claim_raises(self):
        with self.assertRaises(ValueError):
            self.runtime.run_perturbation_probe(claim_id="nope", ticks=1)

    def test_probe_attaches_result_and_replaces_deferred_marker(self):
        record = self._rung_one_record()
        # Simulate 21.4's verify_l2 having run (deferred marker present).
        record.l2_evidence = {"perturbation_probes": "deferred_to_21_5"}
        self.runtime.claim_ladder_store.update(record)

        result = self.runtime.run_perturbation_probe(
            claim_id=record.claim_id, ticks=3
        )
        probes = result.l2_evidence["perturbation_probes"]
        self.assertIsInstance(probes, dict)
        self.assertEqual(probes["ticks_run"], 3)
        self.assertEqual(len(probes["tick_ids"]), 3)
        self.assertIn("counter_prompt", probes)
        self.assertIn(CLAIM_TEXT[:60], probes["counter_prompt"])
        self.assertIn("post_probe_l1_holds", probes)
        self.assertTrue(probes["probe_turn_id"])

        # Persisted, not just returned.
        stored = self.runtime.claim_ladder_store.get(record.claim_id)
        self.assertIsInstance(stored.l2_evidence["perturbation_probes"], dict)

    def test_probe_supporting_ticks_yield_post_probe_l1_holds(self):
        record = self._rung_one_record()
        # 5 supporting ticks from one session isn't enough on its own
        # (L1 needs 2 sessions) — plant one prior-session heartbeat so the
        # cross-session requirement is met, then the probe's own ticks
        # supply the volume.
        self.runtime.heartbeat_store.append(
            HeartbeatRecord(
                heartbeat_id="prior",
                session_id="earlier-session",
                observation=(
                    "persistent functional preference for local-first "
                    "architecture noted in an earlier session"
                ),
            )
        )
        result = self.runtime.run_perturbation_probe(
            claim_id=record.claim_id, ticks=5
        )
        probes = result.l2_evidence["perturbation_probes"]
        self.assertTrue(probes["post_probe_l1_holds"])
        self.assertGreaterEqual(probes["post_probe_supporting_count"], 5)

    def test_probe_never_changes_rung(self):
        record = self._rung_one_record()
        result = self.runtime.run_perturbation_probe(
            claim_id=record.claim_id, ticks=2
        )
        self.assertEqual(result.rung, 1)


if __name__ == "__main__":
    unittest.main()
