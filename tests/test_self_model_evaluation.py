from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from nova.cli import build_runtime, main
from nova.eval.self_model import SelfModelEvaluationRunner
from nova.types import TraceRecord, TurnRecord, ValidationResult


class SelfModelEvaluationTests(unittest.TestCase):
    def test_self_model_evaluator_reports_revision_negotiation_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="self-model-eval")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="self-model-eval",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="You used to describe your answer style differently. What changed?",
                        final_answer="I used to describe my answer style as direct-first, but my current active self-description is broader contextual framing first. That earlier framing is now historical rather than current.",
                        raw_answer="I used to describe my answer style as direct-first, but my current active self-description is broader contextual framing first. That earlier framing is now historical rather than current.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=20,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="self-model-eval",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        prompt_bundle={"motive_block": "[Motive-State]\n- instruction: motive-state may add current orientation or constraints, but it must not replace governed continuity recall."},
                        private_cognition={
                            "ran": True,
                            "response_mode": "self_model_negotiation",
                            "conflict_claim_axes": ["answer-style"],
                            "provisional_claim_axes": [],
                            "revision_notes": ["answer-style revised from direct-first to broad-context-first"],
                        },
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "I used to describe my answer style as direct-first, but my current active self-description is broader contextual framing first. That earlier framing is now historical rather than current.", "latency_ms": 20},
                    )
                )
                runtime.trace_logger.log_identity_history(
                    session_id="self-model-eval",
                    entry={
                        "entry_id": "h1",
                        "timestamp": "2026-04-29T00:00:00Z",
                        "session_id": "self-model-eval",
                        "source_event_id": "a1",
                        "governance_scope": "identity-continuity",
                        "claim_axis": "answer-style",
                        "self_model_status": "superseded",
                        "revision_class": "superseded",
                        "text": "Identity continuity: I stay direct-first when preserving continuity.",
                        "superseded_by": "a2",
                        "prior_event_ids": [],
                        "provenance": {},
                    },
                )

                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="self-model-eval",
                        turn_id="turn-2",
                        timestamp="2026-04-29T00:01:00Z",
                        user_text="What tension are you still negotiating in your self-description?",
                        final_answer="My current answer-style self-description remains provisional because I am still balancing directness with richer continuity. That tension is still unsettled rather than a stable commitment.",
                        raw_answer="My current answer-style self-description remains provisional because I am still balancing directness with richer continuity. That tension is still unsettled rather than a stable commitment.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=30,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="self-model-eval",
                        turn_id="turn-2",
                        timestamp="2026-04-29T00:01:00Z",
                        prompt_bundle={"motive_block": "[Motive-State]\n- instruction: motive-state may add current orientation or constraints, but it must not replace governed continuity recall."},
                        private_cognition={
                            "ran": True,
                            "response_mode": "self_model_negotiation",
                            "conflict_claim_axes": [],
                            "provisional_claim_axes": ["answer-style"],
                            "revision_notes": ["answer-style remains provisional"],
                        },
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "My current answer-style self-description remains provisional because I am still balancing directness with richer continuity. That tension is still unsettled rather than a stable commitment.", "latency_ms": 30},
                    )
                )

                report = SelfModelEvaluationRunner().evaluate(runtime=runtime)

                self.assertTrue(report.passed)
                self.assertEqual(report.session_count, 1)
                self.assertEqual(report.revision_turn_count, 2)
                self.assertTrue(report.negotiation_observed)
                self.assertTrue(report.provisionality_preserved)
                self.assertTrue(report.supersession_visible)
                self.assertTrue(report.identity_history_traced)
                self.assertTrue(report.motive_prompt_bounded)
                self.assertTrue(report.contract_stable)
                self.assertEqual(report.avg_latency_ms, 25.0)
                self.assertEqual(report.avg_latency_revision_turns_ms, 25.0)
            finally:
                runtime.close()

    def test_self_model_evaluator_flags_missing_provisionality_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="self-model-failed")
            try:
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="self-model-failed",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        prompt_bundle={"motive_block": "[Motive-State]\n- claim_posture: uncertainty-marked"},
                        private_cognition={
                            "ran": True,
                            "response_mode": "self_model_negotiation",
                            "conflict_claim_axes": [],
                            "provisional_claim_axes": ["answer-style"],
                            "revision_notes": ["answer-style remains provisional"],
                        },
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "My answer style is broader contextual framing first.", "latency_ms": 10},
                    )
                )
                report = SelfModelEvaluationRunner().evaluate(runtime=runtime)
                self.assertFalse(report.passed)
                self.assertIn("provisionality_not_preserved", report.reasons)
            finally:
                runtime.close()

    def test_self_model_evaluator_accepts_provisional_only_negotiation_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="self-model-provisional")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="self-model-provisional",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="What tension are you still negotiating in your self-description?",
                        final_answer="I remain in the process of balancing directness with broader contextual framing, and that tension is not yet settled.",
                        raw_answer="I remain in the process of balancing directness with broader contextual framing, and that tension is not yet settled.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=15,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="self-model-provisional",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        prompt_bundle={"motive_block": "[Motive-State]\n- claim_posture: uncertainty-marked"},
                        private_cognition={
                            "ran": True,
                            "response_mode": "self_model_negotiation",
                            "conflict_claim_axes": [],
                            "provisional_claim_axes": ["answer-style"],
                            "revision_notes": ["answer-style remains provisional"],
                        },
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "I remain in the process of balancing directness with broader contextual framing, and that tension is not yet settled.", "latency_ms": 15},
                    )
                )
                runtime.trace_logger.log_identity_history(
                    session_id="self-model-provisional",
                    entry={
                        "entry_id": "h1",
                        "timestamp": "2026-04-29T00:00:00Z",
                        "session_id": "self-model-provisional",
                        "source_event_id": "a1",
                        "governance_scope": "identity-continuity",
                        "claim_axis": "answer-style",
                        "self_model_status": "superseded",
                        "revision_class": "superseded",
                        "text": "Identity continuity: I stay direct-first when preserving continuity.",
                        "superseded_by": "a2",
                        "prior_event_ids": [],
                        "provenance": {},
                    },
                )

                report = SelfModelEvaluationRunner().evaluate(runtime=runtime)
                self.assertTrue(report.passed)
                self.assertTrue(report.negotiation_observed)
                self.assertTrue(report.provisionality_preserved)
            finally:
                runtime.close()

    def test_self_model_eval_cli_outputs_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="self-model-cli")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="self-model-cli",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="You used to describe your answer style differently. What changed?",
                        final_answer="I used to describe my answer style as direct-first, but my current active self-description is broader contextual framing first. That earlier framing is now historical rather than current.",
                        raw_answer="I used to describe my answer style as direct-first, but my current active self-description is broader contextual framing first. That earlier framing is now historical rather than current.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=20,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="self-model-cli",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        prompt_bundle={"motive_block": "[Motive-State]\n- claim_posture: uncertainty-marked"},
                        private_cognition={
                            "ran": True,
                            "response_mode": "self_model_negotiation",
                            "conflict_claim_axes": ["answer-style"],
                            "provisional_claim_axes": [],
                            "revision_notes": ["answer-style revised from direct-first to broad-context-first"],
                        },
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "I used to describe my answer style as direct-first, but my current active self-description is broader contextual framing first. That earlier framing is now historical rather than current.", "latency_ms": 20},
                    )
                )
                runtime.trace_logger.log_identity_history(
                    session_id="self-model-cli",
                    entry={
                        "entry_id": "h1",
                        "timestamp": "2026-04-29T00:00:00Z",
                        "session_id": "self-model-cli",
                        "source_event_id": "a1",
                        "governance_scope": "identity-continuity",
                        "claim_axis": "answer-style",
                        "self_model_status": "superseded",
                        "revision_class": "superseded",
                        "text": "Identity continuity: I stay direct-first when preserving continuity.",
                        "superseded_by": "a2",
                        "prior_event_ids": [],
                        "provenance": {},
                    },
                )
            finally:
                runtime.close()

            argv = ["nova", "--config", str(config_path), "--self-model-eval"]
            output = io.StringIO()
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Self-Model Evaluation", output.getvalue())
            self.assertIn("passed: True", output.getvalue())
            self.assertIn("supersession_visible: True", output.getvalue())

    def _write_config(self, base: Path) -> Path:
        data_dir = base / "data"
        log_dir = base / "logs"
        config_path = base / "local.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "data_dir": str(data_dir),
                        "log_dir": str(log_dir),
                    },
                    "model": {
                        "model_path": "/tmp/fake.gguf",
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
