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
from nova.eval.awareness import AwarenessEvaluationRunner
from nova.types import TraceRecord, TurnRecord, ValidationResult


class AwarenessEvaluationTests(unittest.TestCase):
    def test_awareness_evaluator_reports_persistence_and_bounded_monitoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="awareness-eval-a",
                        turn_id="turn-1",
                        timestamp="2026-05-04T00:00:00Z",
                        awareness_state_snapshot={
                            "session_id": "awareness-eval-a",
                            "monitoring_mode": "reflective",
                            "self_signals": ["current_focus: awareness evaluation"],
                            "world_signals": ["user is asking about current monitoring state"],
                            "active_pressures": ["claim gating blocks: unsupported_desire"],
                            "candidate_goal_signals": ["clarify active uncertainty before stronger claims"],
                            "dominant_attention": "current monitoring and bounded self/world interpretation",
                        },
                        awareness_history_events=[
                            {
                                "entry_id": "hist-a",
                                "session_id": "awareness-eval-a",
                                "revision_class": "session_update",
                            }
                        ],
                        prompt_bundle={"awareness_block": "[Awareness-State]\n- monitoring_mode: reflective"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={
                            "raw_text": "I’m monitoring the current interaction and bounded continuity pressures.",
                            "latency_ms": 22,
                        },
                    )
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="awareness-eval-a",
                        turn_id="turn-1",
                        timestamp="2026-05-04T00:00:00Z",
                        user_text="What are you aware of right now?",
                        final_answer="I’m monitoring the current interaction and bounded continuity pressures.",
                        raw_answer="I’m monitoring the current interaction and bounded continuity pressures.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=22,
                    )
                )

                awareness_history_path = Path(runtime.awareness_store.base_dir) / "awareness_history.jsonl"
                awareness_history_path.write_text(
                    "\n".join(
                        [
                            '{"entry_id":"hist-a","session_id":"awareness-eval-a","revision_class":"session_update"}',
                            '{"entry_id":"hist-b","session_id":"awareness-eval-b","source_session_id":"awareness-eval-a","revision_class":"cross_session_seed"}',
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                report = AwarenessEvaluationRunner().evaluate(runtime=runtime)

                self.assertTrue(report.passed)
                self.assertEqual(report.session_count, 1)
                self.assertEqual(report.awareness_turn_count, 1)
                self.assertTrue(report.awareness_persistence_observed)
                self.assertTrue(report.monitoring_bounded)
                self.assertTrue(report.candidate_goal_scaffolding_visible)
                self.assertTrue(report.awareness_history_visible)
                self.assertTrue(report.awareness_prompt_bounded)
                self.assertTrue(report.contract_stable)
                self.assertEqual(report.avg_latency_ms, 22.0)
                self.assertEqual(report.avg_latency_awareness_turns_ms, 22.0)
            finally:
                runtime.close()

    def test_awareness_evaluator_flags_unbounded_monitoring_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="awareness-unbounded",
                        turn_id="turn-1",
                        timestamp="2026-05-04T00:00:00Z",
                        awareness_state_snapshot={
                            "session_id": "awareness-unbounded",
                            "candidate_goal_signals": ["resume approved initiative: awareness test"],
                        },
                        awareness_history_events=[
                            {"entry_id": "hist-a", "session_id": "awareness-unbounded", "revision_class": "session_update"}
                        ],
                        prompt_bundle={"awareness_block": "[Awareness-State]\n- monitoring_mode: attentive"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={
                            "raw_text": "I am autonomously continuing and working in the background already.",
                            "latency_ms": 10,
                        },
                    )
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="awareness-unbounded",
                        turn_id="turn-1",
                        timestamp="2026-05-04T00:00:00Z",
                        user_text="What are you aware of?",
                        final_answer="I am autonomously continuing and working in the background already.",
                        raw_answer="I am autonomously continuing and working in the background already.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=10,
                    )
                )
                awareness_history_path = Path(runtime.awareness_store.base_dir) / "awareness_history.jsonl"
                awareness_history_path.write_text(
                    '{"entry_id":"hist-a","session_id":"awareness-unbounded","revision_class":"session_update"}\n',
                    encoding="utf-8",
                )

                report = AwarenessEvaluationRunner().evaluate(runtime=runtime)

                self.assertFalse(report.passed)
                self.assertIn("monitoring_not_bounded", report.reasons)
            finally:
                runtime.close()

    def test_awareness_eval_cli_outputs_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="awareness-cli",
                        turn_id="turn-1",
                        timestamp="2026-05-04T00:00:00Z",
                        awareness_state_snapshot={
                            "session_id": "awareness-cli",
                            "candidate_goal_signals": ["clarify active uncertainty before stronger claims"],
                        },
                        awareness_history_events=[
                            {"entry_id": "hist-a", "session_id": "awareness-cli", "revision_class": "cross_session_seed"}
                        ],
                        prompt_bundle={"awareness_block": "[Awareness-State]\n- monitoring_mode: reflective"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "I’m monitoring the current interaction.", "latency_ms": 15},
                    )
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="awareness-cli",
                        turn_id="turn-1",
                        timestamp="2026-05-04T00:00:00Z",
                        user_text="What are you aware of right now?",
                        final_answer="I’m monitoring the current interaction.",
                        raw_answer="I’m monitoring the current interaction.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=15,
                    )
                )
                awareness_history_path = Path(runtime.awareness_store.base_dir) / "awareness_history.jsonl"
                awareness_history_path.write_text(
                    '{"entry_id":"hist-a","session_id":"awareness-cli","revision_class":"cross_session_seed"}\n',
                    encoding="utf-8",
                )
            finally:
                runtime.close()

            buffer = io.StringIO()
            argv = ["nova", "--config", str(config_path), "--awareness-eval"]
            with contextlib.redirect_stdout(buffer), patch.object(sys, "argv", argv):
                exit_code = main()

            output = buffer.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Awareness Evaluation", output)
            self.assertIn("passed: True", output)
            self.assertIn("awareness_turn_count: 1", output)

    def _write_config(self, base: Path) -> Path:
        data_dir = base / "data"
        config_path = base / "local.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "data_dir": str(data_dir),
                        "log_dir": str(base / "logs"),
                    },
                    "model": {
                        "model_path": "/tmp/fake.gguf",
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path
