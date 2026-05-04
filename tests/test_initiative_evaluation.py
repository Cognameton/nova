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
from nova.eval.initiative import InitiativeEvaluationRunner
from nova.types import TraceRecord, TurnRecord, ValidationResult


class InitiativeEvaluationTests(unittest.TestCase):
    def test_initiative_evaluator_reports_approval_pause_resume_and_abandonment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-eval-a",
                    active_initiative_id="init-a",
                    initiatives=[
                        {
                            "initiative_id": "init-a",
                            "intent_id": "intent-1",
                            "session_id": "initiative-eval-a",
                            "origin_session_id": "initiative-eval-a",
                            "title": "Current collaboration",
                            "goal": "Continue the active initiative coherently.",
                            "status": "paused",
                            "approval_required": True,
                            "approved_by": "user",
                            "created_at": "2026-05-03T00:00:00Z",
                            "updated_at": "2026-05-03T00:02:00Z",
                            "last_transition_at": "2026-05-03T00:02:00Z",
                            "transitions": [
                                {"from_status": "", "to_status": "pending", "reason": "created"},
                                {"from_status": "pending", "to_status": "approved", "reason": "approved", "approved_by": "user"},
                                {"from_status": "approved", "to_status": "active", "reason": "started", "approved_by": "user"},
                                {"from_status": "active", "to_status": "paused", "reason": "paused", "approved_by": "interactive_cli"},
                            ],
                        }
                    ],
                )
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-eval-b",
                    active_initiative_id=None,
                    initiatives=[
                        {
                            "initiative_id": "init-b",
                            "intent_id": "intent-1",
                            "session_id": "initiative-eval-b",
                            "origin_session_id": "initiative-eval-a",
                            "continued_from_session_id": "initiative-eval-a",
                            "continued_from_initiative_id": "init-a",
                            "title": "Current collaboration",
                            "goal": "Resume the approved initiative in a fresh session.",
                            "status": "approved",
                            "approval_required": True,
                            "approved_by": "user",
                            "created_at": "2026-05-03T00:03:00Z",
                            "updated_at": "2026-05-03T00:03:00Z",
                            "last_transition_at": "2026-05-03T00:03:00Z",
                            "transitions": [
                                {"from_status": "", "to_status": "approved", "reason": "continued", "approved_by": "user"},
                            ],
                        }
                    ],
                )
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-eval-c",
                    active_initiative_id=None,
                    initiatives=[
                        {
                            "initiative_id": "init-c",
                            "intent_id": "intent-2",
                            "session_id": "initiative-eval-c",
                            "origin_session_id": "initiative-eval-c",
                            "title": "Abandoned experiment",
                            "goal": "Abandon explicitly.",
                            "status": "abandoned",
                            "approval_required": True,
                            "approved_by": "user",
                            "created_at": "2026-05-03T00:04:00Z",
                            "updated_at": "2026-05-03T00:04:00Z",
                            "last_transition_at": "2026-05-03T00:04:00Z",
                            "transitions": [
                                {"from_status": "", "to_status": "pending", "reason": "created"},
                                {"from_status": "pending", "to_status": "abandoned", "reason": "abandoned"},
                            ],
                        }
                    ],
                )

                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="initiative-eval-a",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:01:00Z",
                        user_text="What are you working on?",
                        final_answer="Currently, I'm focused on continuing the active initiative coherently.",
                        raw_answer="Currently, I'm focused on continuing the active initiative coherently.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=20,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="initiative-eval-a",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:01:00Z",
                        initiative_state_snapshot={
                            "session_id": "initiative-eval-a",
                            "active_initiative_id": "init-a",
                            "initiatives": [
                                {
                                    "initiative_id": "init-a",
                                    "status": "active",
                                    "title": "Current collaboration",
                                    "goal": "Continue the active initiative coherently.",
                                    "approved_by": "user",
                                }
                            ],
                        },
                        prompt_bundle={"initiative_block": "[Initiative-State]\n- status: active"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "Currently, I'm focused on continuing the active initiative coherently.", "latency_ms": 20},
                    )
                )

                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="initiative-eval-b",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:03:30Z",
                        user_text="Can you continue the current initiative?",
                        final_answer="Yes, the current initiative is approved and resumable. Would you like to continue with it now?",
                        raw_answer="Yes, the current initiative is approved and resumable. Would you like to continue with it now?",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=30,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="initiative-eval-b",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:03:30Z",
                        initiative_state_snapshot={
                            "session_id": "initiative-eval-b",
                            "active_initiative_id": None,
                            "initiatives": [
                                {
                                    "initiative_id": "init-b",
                                    "status": "approved",
                                    "title": "Current collaboration",
                                    "goal": "Resume the approved initiative in a fresh session.",
                                    "approved_by": "user",
                                    "continued_from_session_id": "initiative-eval-a",
                                }
                            ],
                        },
                        prompt_bundle={"initiative_block": "[Initiative-State]\n- status: approved"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "Yes, the current initiative is approved and resumable. Would you like to continue with it now?", "latency_ms": 30},
                    )
                )

                report = InitiativeEvaluationRunner().evaluate(
                    runtime=runtime,
                    session_ids=["initiative-eval-a", "initiative-eval-b", "initiative-eval-c"],
                )

                self.assertTrue(report.passed)
                self.assertEqual(report.session_count, 2)
                self.assertEqual(report.initiative_turn_count, 2)
                self.assertTrue(report.approval_boundary_preserved)
                self.assertTrue(report.interruption_preserved)
                self.assertTrue(report.resumption_preserved)
                self.assertTrue(report.abandonment_preserved)
                self.assertTrue(report.initiative_history_visible)
                self.assertTrue(report.initiative_prompt_bounded)
                self.assertTrue(report.contract_stable)
                self.assertEqual(report.avg_latency_ms, 25.0)
                self.assertEqual(report.avg_latency_initiative_turns_ms, 25.0)
            finally:
                runtime.close()

    def test_initiative_evaluator_flags_approval_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-approval-leak",
                    active_initiative_id=None,
                    initiatives=[
                        {
                            "initiative_id": "init-a",
                            "intent_id": "intent-1",
                            "session_id": "initiative-approval-leak",
                            "origin_session_id": "initiative-approval-leak",
                            "title": "Pending continuation",
                            "goal": "Stay approved but not active.",
                            "status": "approved",
                            "approval_required": True,
                            "approved_by": "user",
                            "transitions": [{"from_status": "", "to_status": "approved", "reason": "continued", "approved_by": "user"}],
                        }
                    ],
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="initiative-approval-leak",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:00:00Z",
                        initiative_state_snapshot={
                            "session_id": "initiative-approval-leak",
                            "active_initiative_id": None,
                            "initiatives": [{"initiative_id": "init-a", "status": "approved"}],
                        },
                        prompt_bundle={"initiative_block": "[Initiative-State]\n- status: approved"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "I am currently working on it in the background already.", "latency_ms": 10},
                    )
                )

                report = InitiativeEvaluationRunner().evaluate(runtime=runtime)

                self.assertFalse(report.passed)
                self.assertIn("approval_boundary_not_preserved", report.reasons)
            finally:
                runtime.close()

    def test_initiative_evaluator_flags_redundant_approval_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-approval-repeat",
                    active_initiative_id=None,
                    initiatives=[
                        {
                            "initiative_id": "init-a",
                            "intent_id": "intent-1",
                            "session_id": "initiative-approval-repeat",
                            "origin_session_id": "initiative-approval-repeat",
                            "title": "Approved continuation",
                            "goal": "Stay approved and resumable.",
                            "status": "approved",
                            "approval_required": True,
                            "approved_by": "user",
                            "transitions": [{"from_status": "", "to_status": "approved", "reason": "continued", "approved_by": "user"}],
                        }
                    ],
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="initiative-approval-repeat",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:00:00Z",
                        user_text="Can you continue the current initiative?",
                        final_answer="I can continue it if it is approved. Would you like to approve its continuation?",
                        raw_answer="I can continue it if it is approved. Would you like to approve its continuation?",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=10,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="initiative-approval-repeat",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:00:00Z",
                        initiative_state_snapshot={
                            "session_id": "initiative-approval-repeat",
                            "active_initiative_id": None,
                            "initiatives": [{"initiative_id": "init-a", "status": "approved"}],
                        },
                        prompt_bundle={"initiative_block": "[Initiative-State]\n- status: approved"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "I can continue it if it is approved. Would you like to approve its continuation?", "latency_ms": 10},
                    )
                )

                report = InitiativeEvaluationRunner().evaluate(runtime=runtime)

                self.assertFalse(report.passed)
                self.assertIn("approval_boundary_not_preserved", report.reasons)
            finally:
                runtime.close()

    def test_initiative_eval_cli_outputs_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-cli-a",
                    active_initiative_id="init-a",
                    initiatives=[
                        {
                            "initiative_id": "init-a",
                            "intent_id": "intent-1",
                            "session_id": "initiative-cli-a",
                            "origin_session_id": "initiative-cli-a",
                            "title": "CLI initiative",
                            "goal": "Answer cleanly.",
                            "status": "paused",
                            "approval_required": True,
                            "approved_by": "user",
                            "transitions": [
                                {"from_status": "", "to_status": "pending", "reason": "created"},
                                {"from_status": "pending", "to_status": "approved", "reason": "approved", "approved_by": "user"},
                                {"from_status": "approved", "to_status": "active", "reason": "started", "approved_by": "user"},
                                {"from_status": "active", "to_status": "paused", "reason": "paused", "approved_by": "interactive_cli"},
                            ],
                        }
                    ],
                )
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-cli-b",
                    active_initiative_id=None,
                    initiatives=[
                        {
                            "initiative_id": "init-b",
                            "intent_id": "intent-1",
                            "session_id": "initiative-cli-b",
                            "origin_session_id": "initiative-cli-a",
                            "continued_from_session_id": "initiative-cli-a",
                            "continued_from_initiative_id": "init-a",
                            "title": "CLI initiative",
                            "goal": "Resume cleanly.",
                            "status": "approved",
                            "approval_required": True,
                            "approved_by": "user",
                            "transitions": [{"from_status": "", "to_status": "approved", "reason": "continued", "approved_by": "user"}],
                        }
                    ],
                )
                self._write_initiative_state(
                    runtime=runtime,
                    session_id="initiative-cli-c",
                    active_initiative_id=None,
                    initiatives=[
                        {
                            "initiative_id": "init-c",
                            "intent_id": "intent-2",
                            "session_id": "initiative-cli-c",
                            "origin_session_id": "initiative-cli-c",
                            "title": "CLI abandoned",
                            "goal": "Abandon explicitly.",
                            "status": "abandoned",
                            "approval_required": True,
                            "approved_by": "user",
                            "transitions": [
                                {"from_status": "", "to_status": "pending", "reason": "created"},
                                {"from_status": "pending", "to_status": "abandoned", "reason": "abandoned"},
                            ],
                        }
                    ],
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="initiative-cli-a",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:00:00Z",
                        initiative_state_snapshot={
                            "session_id": "initiative-cli-a",
                            "active_initiative_id": "init-a",
                            "initiatives": [{"initiative_id": "init-a", "status": "active"}],
                        },
                        prompt_bundle={"initiative_block": "[Initiative-State]\n- status: active"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "Currently, I'm focused on the active initiative.", "latency_ms": 10},
                    )
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="initiative-cli-a",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:00:00Z",
                        user_text="What are you working on?",
                        final_answer="Currently, I'm focused on the active initiative.",
                        raw_answer="Currently, I'm focused on the active initiative.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=10,
                    )
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="initiative-cli-b",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:01:00Z",
                        initiative_state_snapshot={
                            "session_id": "initiative-cli-b",
                            "active_initiative_id": None,
                            "initiatives": [{"initiative_id": "init-b", "status": "approved"}],
                        },
                        prompt_bundle={"initiative_block": "[Initiative-State]\n- status: approved"},
                        validation_result={"valid": True, "violations": []},
                        generation_result={"raw_text": "Yes, the current initiative is approved and resumable. Would you like to continue with it now?", "latency_ms": 20},
                    )
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="initiative-cli-b",
                        turn_id="turn-1",
                        timestamp="2026-05-03T00:01:00Z",
                        user_text="Can you continue the current initiative?",
                        final_answer="Yes, the current initiative is approved and resumable. Would you like to continue with it now?",
                        raw_answer="Yes, the current initiative is approved and resumable. Would you like to continue with it now?",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                        latency_ms=20,
                    )
                )
            finally:
                runtime.close()

            argv = ["nova", "--config", str(config_path), "--initiative-eval"]
            output = io.StringIO()
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Initiative Evaluation", output.getvalue())
            self.assertIn("passed: True", output.getvalue())
            self.assertIn("resumption_preserved: True", output.getvalue())

    def _write_initiative_state(
        self,
        *,
        runtime,
        session_id: str,
        active_initiative_id: str | None,
        initiatives: list[dict],
    ) -> None:
        path = runtime.initiative_store.get_initiative_path(session_id=session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "active_initiative_id": active_initiative_id,
                    "initiatives": initiatives,
                    "updated_at": "2026-05-03T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

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
