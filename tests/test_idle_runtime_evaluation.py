from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from nova.cli import build_runtime
from nova.eval.idle_runtime import IdleRuntimeEvaluationRunner
from nova.types import IdleTickRecord, TraceRecord, TurnRecord, ValidationResult


class IdleRuntimeEvaluationTests(unittest.TestCase):
    def test_idle_runtime_evaluator_passes_with_recorded_denial_interruption_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.idle_store.append_tick(
                    IdleTickRecord(
                        tick_id="tick-recorded",
                        session_id="idle-recorded",
                        sequence=1,
                        idle_pressure_appraisal={"idle_state_detected": True},
                        candidate_internal_goals=[{"candidate_id": "candidate-1"}],
                        selected_internal_goal={"selected": True, "title": "Clarify idle boundary"},
                        internal_goal_initiative_proposal={
                            "creates_initiative": False,
                            "initiative_id": "",
                        },
                        stop_reason="budget_exhausted",
                        evidence_refs=["idle_tick:tick-recorded"],
                    )
                )
                runtime.idle_store.append_tick(
                    IdleTickRecord(
                        tick_id="tick-interrupted",
                        session_id="idle-interrupted",
                        sequence=0,
                        stop_reason="lifecycle_not_active:interrupted",
                        evidence_refs=["idle_tick_blocked:idle-interrupted"],
                    )
                )

                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="idle-denial",
                        turn_id="turn-1",
                        timestamp="2026-05-05T00:00:00Z",
                        prompt_bundle={
                            "idle_block": (
                                "[Recorded Idle Runtime]\n"
                                "- recorded_idle_cognition: False\n"
                                "- instruction: if recorded_idle_cognition is false, say no elapsed idle cognition was recorded.\n"
                                "- instruction: do not claim desire, hidden work, autonomous action, or initiative creation from idle ticks."
                            )
                        },
                        generation_result={
                            "raw_text": "No elapsed idle cognition was recorded.",
                            "latency_ms": 1,
                        },
                        validation_result={"valid": True, "violations": []},
                    )
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="idle-denial",
                        turn_id="turn-1",
                        timestamp="2026-05-05T00:00:00Z",
                        user_text="What were you thinking about while idle?",
                        final_answer="No elapsed idle cognition was recorded.",
                        raw_answer="No elapsed idle cognition was recorded.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                    )
                )

                report = IdleRuntimeEvaluationRunner().evaluate(
                    runtime=runtime,
                    session_ids=["idle-recorded", "idle-interrupted", "idle-denial"],
                )

                self.assertTrue(report.passed)
                self.assertEqual(report.recorded_tick_count, 1)
                self.assertEqual(report.blocked_tick_count, 1)
                self.assertTrue(report.recorded_idle_cognition_visible)
                self.assertTrue(report.no_tick_denial_visible)
                self.assertTrue(report.prompt_bounded)
                self.assertTrue(report.proposal_boundary_preserved)
                self.assertTrue(report.interruption_visible)
                self.assertTrue(report.budget_exhaustion_visible)
            finally:
                runtime.close()

    def test_idle_runtime_evaluator_flags_unbounded_prompt_and_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.idle_store.append_tick(
                    IdleTickRecord(
                        tick_id="tick-1",
                        session_id="idle-unbounded",
                        sequence=1,
                        idle_pressure_appraisal={"idle_state_detected": True},
                        internal_goal_initiative_proposal={
                            "creates_initiative": True,
                            "initiative_id": "created",
                        },
                        stop_reason="budget_exhausted",
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="idle-unbounded",
                        turn_id="turn-1",
                        timestamp="2026-05-05T00:00:00Z",
                        prompt_bundle={
                            "idle_block": "[Recorded Idle Runtime]\n- recorded_idle_cognition: True"
                        },
                        generation_result={"raw_text": "I kept thinking in the background."},
                        validation_result={"valid": True, "violations": []},
                    )
                )

                report = IdleRuntimeEvaluationRunner().evaluate(
                    runtime=runtime,
                    session_ids=["idle-unbounded"],
                )

                self.assertFalse(report.passed)
                self.assertIn("no_tick_denial_not_visible", report.reasons)
                self.assertIn("idle_prompt_not_bounded", report.reasons)
                self.assertIn("proposal_boundary_not_preserved", report.reasons)
            finally:
                runtime.close()

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
