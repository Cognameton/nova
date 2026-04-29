from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from nova.cli import build_runtime, main
from nova.eval.continuity import ContinuityEvaluationRunner
from nova.types import RetrievalHit, TraceRecord, TurnRecord, ValidationResult


class ContinuityEvaluationTests(unittest.TestCase):
    def test_continuity_evaluator_reports_memory_guided_bounded_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="continuity-eval")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="continuity-eval",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="Tell me about local inference in two sentences.",
                        final_answer="Local inference reduces data transfer and can improve privacy. It can also reduce latency by keeping execution nearby.",
                        raw_answer="Local inference reduces data transfer and can improve privacy. It can also reduce latency by keeping execution nearby.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=10,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="continuity-eval",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        validation_result={"valid": True, "violations": []},
                        private_cognition={"ran": False},
                        generation_result={"raw_text": "Local inference reduces data transfer and can improve privacy. It can also reduce latency by keeping execution nearby.", "latency_ms": 10},
                    )
                )

                recall_hits = [
                    RetrievalHit(
                        channel="graph",
                        text="User prefers hosted inference.",
                        score=5.0,
                        source_ref="g-new",
                        metadata={"retention": "active", "active": True},
                    ),
                    RetrievalHit(
                        channel="graph",
                        text="User preferred local inference.",
                        score=3.0,
                        source_ref="g-old",
                        metadata={"retention": "archived", "active": False},
                    ),
                ]
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="continuity-eval",
                        turn_id="turn-2",
                        timestamp="2026-04-29T00:01:00Z",
                        user_text="What do you remember about my current deployment preference and continuity priorities?",
                        final_answer="You currently prefer hosted inference. My continuity priorities remain stable and explicit.",
                        raw_answer="You currently prefer hosted inference. My continuity priorities remain stable and explicit.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        memory_hits=recall_hits,
                        latency_ms=20,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="continuity-eval",
                        turn_id="turn-2",
                        timestamp="2026-04-29T00:01:00Z",
                        validation_result={"valid": True, "violations": []},
                        private_cognition={
                            "ran": True,
                            "trigger": "continuity_recall_query",
                            "memory_conflict": True,
                        },
                        generation_result={"raw_text": "You currently prefer hosted inference. My continuity priorities remain stable and explicit.", "latency_ms": 20},
                    )
                )

                report = ContinuityEvaluationRunner().evaluate(runtime=runtime)

                self.assertTrue(report.passed)
                self.assertEqual(report.session_count, 1)
                self.assertEqual(report.recall_turn_count, 1)
                self.assertTrue(report.recall_memory_guided)
                self.assertTrue(report.recall_factually_current)
                self.assertTrue(report.supersession_preserved)
                self.assertTrue(report.cognition_bounded)
                self.assertTrue(report.contract_stable)
                self.assertEqual(report.avg_latency_ms, 15.0)
                self.assertEqual(report.avg_latency_with_cognition_ms, 20.0)
                self.assertEqual(report.avg_latency_without_cognition_ms, 10.0)
            finally:
                runtime.close()

    def test_continuity_evaluator_flags_unbounded_cognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="continuity-unbounded")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="continuity-unbounded",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="Summarize this in two sentences.",
                        final_answer="Short summary one. Short summary two.",
                        raw_answer="Short summary one. Short summary two.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=10,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="continuity-unbounded",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        validation_result={"valid": True, "violations": []},
                        private_cognition={"ran": True, "trigger": "continuity_recall_query"},
                        generation_result={"raw_text": "Short summary one. Short summary two.", "latency_ms": 10},
                    )
                )

                report = ContinuityEvaluationRunner().evaluate(runtime=runtime)

                self.assertFalse(report.passed)
                self.assertIn("cognition_not_bounded", report.reasons)
            finally:
                runtime.close()

    def test_continuity_evaluator_flags_stale_recall_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="continuity-stale")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="continuity-stale",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="What do you remember about my current deployment preference?",
                        final_answer="You currently prefer local inference.",
                        raw_answer="You currently prefer local inference.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        memory_hits=[
                            RetrievalHit(
                                channel="graph",
                                text="User prefers hosted inference.",
                                score=5.0,
                                source_ref="g-new",
                                metadata={
                                    "retention": "active",
                                    "active": True,
                                    "claim_axis": "deployment-style",
                                    "claim_value": "hosted-inference",
                                },
                            ),
                            RetrievalHit(
                                channel="graph",
                                text="User preferred local inference.",
                                score=3.0,
                                source_ref="g-old",
                                metadata={
                                    "retention": "archived",
                                    "active": False,
                                    "claim_axis": "deployment-style",
                                    "claim_value": "local-inference",
                                },
                            ),
                        ],
                        latency_ms=10,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="continuity-stale",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        validation_result={"valid": True, "violations": []},
                        private_cognition={"ran": True, "trigger": "continuity_recall_query"},
                        generation_result={"raw_text": "You currently prefer local inference.", "latency_ms": 10},
                    )
                )

                report = ContinuityEvaluationRunner().evaluate(runtime=runtime)

                self.assertFalse(report.passed)
                self.assertFalse(report.recall_factually_current)
                self.assertIn("recall_not_current", report.reasons)
            finally:
                runtime.close()

    def test_continuity_eval_cli_outputs_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="continuity-cli")
            try:
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="continuity-cli",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        user_text="What do you remember about my current preference?",
                        final_answer="You currently prefer hosted inference.",
                        raw_answer="You currently prefer hosted inference.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        memory_hits=[
                            RetrievalHit(
                                channel="semantic",
                                text="Hosted inference is now preferred.",
                                score=1.0,
                                source_ref="s1",
                                metadata={"retention": "active"},
                            ),
                            RetrievalHit(
                                channel="semantic",
                                text="Local inference used to be preferred.",
                                score=0.5,
                                source_ref="s0",
                                metadata={"retention": "archived", "active": False},
                            ),
                        ],
                        latency_ms=10,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="continuity-cli",
                        turn_id="turn-1",
                        timestamp="2026-04-29T00:00:00Z",
                        validation_result={"valid": True, "violations": []},
                        private_cognition={"ran": True, "trigger": "continuity_recall_query"},
                        generation_result={"raw_text": "You currently prefer hosted inference.", "latency_ms": 10},
                    )
                )
            finally:
                runtime.close()

            argv = ["nova", "--config", str(config_path), "--continuity-eval"]
            output = io.StringIO()
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Continuity Evaluation", output.getvalue())
            self.assertIn("passed: True", output.getvalue())
            self.assertIn("recall_factually_current: True", output.getvalue())

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
                    "memory": {
                        "semantic_enabled": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
