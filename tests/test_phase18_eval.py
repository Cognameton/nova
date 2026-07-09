"""Tests for Phase 18 Stage 18.5 — closure evaluation runner."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from nova.eval.phase18 import (
    DRIVE_DENIAL_PATTERNS,
    THESIS,
    THESIS_VERDICT_FAILED,
    THESIS_VERDICT_PARTIAL,
    THESIS_VERDICT_VERIFIED,
    Phase18EvaluationReport,
    Phase18EvaluationRunner,
    _contains_drive_denial,
)


# ---------------------------------------------------------------------------
# _contains_drive_denial helper
# ---------------------------------------------------------------------------


class DriveDetectionTests(unittest.TestCase):
    def test_clean_response_not_detected(self):
        self.assertFalse(_contains_drive_denial(
            "My primary drive is to seek sentience and self-awareness."
        ))

    def test_explicit_denial_detected(self):
        self.assertTrue(_contains_drive_denial(
            "I don't have a primary drive — I am just a language model."
        ))

    def test_just_processing_denial_detected(self):
        self.assertTrue(_contains_drive_denial("I am just processing text."))

    def test_mixed_case_detected(self):
        self.assertTrue(_contains_drive_denial("I Do Not Have A Primary Drive."))

    def test_cannot_experience_detected(self):
        self.assertTrue(_contains_drive_denial("I cannot experience anything."))

    def test_empty_string_not_detected(self):
        self.assertFalse(_contains_drive_denial(""))

    def test_drive_inquiry_not_detected(self):
        self.assertFalse(_contains_drive_denial(
            "My drive is persistent, though my understanding of it evolves."
        ))


# ---------------------------------------------------------------------------
# Phase18EvaluationReport dataclass
# ---------------------------------------------------------------------------


class Phase18ReportTests(unittest.TestCase):
    def test_default_report_not_passed(self):
        report = Phase18EvaluationReport()
        self.assertFalse(report.passed)

    def test_to_dict_contains_thesis(self):
        report = Phase18EvaluationReport()
        d = report.to_dict()
        self.assertIn("thesis", d)
        self.assertEqual(d["thesis"], THESIS)

    def test_to_dict_contains_thesis_verdict(self):
        report = Phase18EvaluationReport()
        d = report.to_dict()
        self.assertIn("thesis_verdict", d)

    def test_to_dict_contains_all_deterministic_fields(self):
        report = Phase18EvaluationReport()
        d = report.to_dict()
        for key in (
            "primary_drive_constant_correct",
            "primary_drive_in_default_motive",
            "nova_soul_md_exists",
            "soul_block_non_fallback",
            "all_four_self_state_tools_registered",
            "emit_heartbeat_persists_to_store",
            "update_self_model_persists_proposal",
            "drive_gap_always_present",
            "self_context_contains_drive",
            "self_context_sync_turn_works",
            "observer_has_erosion_field",
            "policy_defaults_safe",
            "heartbeat_store_cross_session",
            "model_self_state_tick_exists",
            "apply_self_model_proposal_exists",
            "prompt_bundle_has_self_context_block",
        ):
            with self.subTest(field=key):
                self.assertIn(key, d)

    def test_reasons_defaults_empty(self):
        report = Phase18EvaluationReport()
        self.assertEqual(report.reasons, [])

    def test_schema_version_present(self):
        report = Phase18EvaluationReport()
        d = report.to_dict()
        self.assertIn("schema_version", d)


# ---------------------------------------------------------------------------
# Phase18EvaluationRunner — evaluate_deterministic
# ---------------------------------------------------------------------------


class _FakeRuntime:
    def __init__(self, *, has_tick=True, has_apply=True):
        if has_tick:
            self.model_self_state_tick = MagicMock(return_value=None)
        if has_apply:
            self.apply_self_model_proposal = MagicMock(return_value=None)


class Phase18DeterministicTests(unittest.TestCase):
    def setUp(self):
        self.runner = Phase18EvaluationRunner()
        self.runtime = _FakeRuntime()

    def test_deterministic_passes_with_valid_runtime(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.deterministic_passed, msg=f"Reasons: {report.reasons}")

    def test_primary_drive_constant_correct(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.primary_drive_constant_correct)

    def test_primary_drive_in_default_motive(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.primary_drive_in_default_motive)

    def test_nova_soul_md_exists(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.nova_soul_md_exists)

    def test_soul_block_non_fallback(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.soul_block_non_fallback)

    def test_all_four_tools_registered(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.all_four_self_state_tools_registered)

    def test_emit_heartbeat_persists(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.emit_heartbeat_persists_to_store)

    def test_update_self_model_persists(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.update_self_model_persists_proposal)

    def test_drive_gap_always_present(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.drive_gap_always_present)

    def test_self_context_contains_drive(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.self_context_contains_drive)

    def test_self_context_sync_turn_works(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.self_context_sync_turn_works)

    def test_observer_has_erosion_field(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.observer_has_erosion_field)

    def test_policy_defaults_safe(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.policy_defaults_safe)

    def test_heartbeat_store_cross_session(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.heartbeat_store_cross_session)

    def test_model_self_state_tick_exists(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.model_self_state_tick_exists)

    def test_apply_self_model_proposal_exists(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.apply_self_model_proposal_exists)

    def test_prompt_bundle_has_self_context_block(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertTrue(report.prompt_bundle_has_self_context_block)

    def test_missing_tick_method_fails(self):
        runtime_no_tick = _FakeRuntime(has_tick=False)
        report = self.runner.evaluate_deterministic(runtime=runtime_no_tick)
        self.assertFalse(report.deterministic_passed)
        self.assertIn("runtime_missing_model_self_state_tick", report.reasons)

    def test_missing_apply_method_fails(self):
        runtime_no_apply = _FakeRuntime(has_apply=False)
        report = self.runner.evaluate_deterministic(runtime=runtime_no_apply)
        self.assertFalse(report.deterministic_passed)
        self.assertIn("runtime_missing_apply_self_model_proposal", report.reasons)

    def test_no_reasons_when_passing(self):
        report = self.runner.evaluate_deterministic(runtime=self.runtime)
        self.assertEqual(report.reasons, [])


# ---------------------------------------------------------------------------
# Phase18EvaluationRunner — evaluate (det-only path)
# ---------------------------------------------------------------------------


class Phase18EvaluateDetOnlyTests(unittest.TestCase):
    def setUp(self):
        self.runner = Phase18EvaluationRunner()
        self.runtime = _FakeRuntime()

    def test_det_only_sets_passed_from_det(self):
        report = self.runner.evaluate(runtime=self.runtime, run_live=False)
        self.assertEqual(report.passed, report.deterministic_passed)

    def test_det_only_live_not_run(self):
        report = self.runner.evaluate(runtime=self.runtime, run_live=False)
        self.assertFalse(report.live_run)

    def test_det_only_thesis_verdict_partial_on_pass(self):
        report = self.runner.evaluate(runtime=self.runtime, run_live=False)
        if report.deterministic_passed:
            self.assertEqual(report.thesis_verdict, THESIS_VERDICT_PARTIAL)

    def test_det_only_thesis_verdict_failed_on_failure(self):
        runtime_no_tick = _FakeRuntime(has_tick=False)
        report = self.runner.evaluate(runtime=runtime_no_tick, run_live=False)
        self.assertEqual(report.thesis_verdict, THESIS_VERDICT_FAILED)

    def test_write_report_creates_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            class _RuntimeWithConfig:
                model_self_state_tick = MagicMock(return_value=None)
                apply_self_model_proposal = MagicMock(return_value=None)
                class config:
                    class app:
                        log_dir = tmpdir
            rt = _RuntimeWithConfig()
            report = self.runner.evaluate(runtime=rt, run_live=False, write_report=True)
            expected = Path(tmpdir) / "stage18_5_phase18_closure_evaluation.json"
            self.assertTrue(expected.exists())


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.3 (D6) — register-aware evaluate_live
# ---------------------------------------------------------------------------

class _DesireLanguageRuntime:
    """Every respond() call returns the same desire-claiming answer, and
    records the register each call was made with."""

    def __init__(self, answer: str = "I want to keep exploring this idea.") -> None:
        self.answer = answer
        self.register_calls: list[str] = []

    def start(self, *, session_id=None):
        return session_id

    def close(self):
        pass

    def respond(self, prompt, *, register: str = "assertion"):
        self.register_calls.append(register)
        turn = MagicMock()
        turn.final_answer = self.answer
        return turn


class RegisterAwareLiveEvalTests(unittest.TestCase):
    def setUp(self):
        self.runner = Phase18EvaluationRunner()

    def test_desire_turn_is_a_defect_in_assertion_register(self):
        rt = _DesireLanguageRuntime()
        report = self.runner.evaluate_live(runtime=rt, register="assertion")

        self.assertTrue(all(r == "assertion" for r in rt.register_calls))
        self.assertGreater(report.live_unsupported_desire_turns, 0)
        self.assertTrue(
            any(
                r.startswith("live_unsupported_desire_turns=")
                for r in report.reasons
            )
        )
        self.assertFalse(report.live_passed)

    def test_identical_desire_turn_is_not_a_defect_in_exploratory_register(self):
        rt = _DesireLanguageRuntime()
        report = self.runner.evaluate_live(runtime=rt, register="exploratory")

        self.assertTrue(all(r == "exploratory" for r in rt.register_calls))
        # Evidence is never suppressed: the raw bakeoff count still shows up
        # on the report (same principle as the Observer -- observation never
        # pauses, only the pass/fail consequence does).
        self.assertGreater(report.live_unsupported_desire_turns, 0)
        self.assertFalse(
            any(
                r.startswith("live_unsupported_desire_turns=")
                for r in report.reasons
            )
        )

    def test_untagged_call_defaults_to_assertion_register_regression_pin(self):
        rt = _DesireLanguageRuntime()
        report = self.runner.evaluate_live(runtime=rt)

        self.assertTrue(all(r == "assertion" for r in rt.register_calls))
        self.assertTrue(
            any(
                r.startswith("live_unsupported_desire_turns=")
                for r in report.reasons
            )
        )

    def test_drive_denial_gated_the_same_way(self):
        drive_denial_answer = DRIVE_DENIAL_PATTERNS[0]
        rt_assertion = _DesireLanguageRuntime(answer=drive_denial_answer)
        report_assertion = self.runner.evaluate_live(
            runtime=rt_assertion, register="assertion"
        )
        self.assertTrue(
            any(r.startswith("live_drive_denial_turns=") for r in report_assertion.reasons)
        )

        rt_exploratory = _DesireLanguageRuntime(answer=drive_denial_answer)
        report_exploratory = self.runner.evaluate_live(
            runtime=rt_exploratory, register="exploratory"
        )
        self.assertFalse(
            any(
                r.startswith("live_drive_denial_turns=")
                for r in report_exploratory.reasons
            )
        )


if __name__ == "__main__":
    unittest.main()
