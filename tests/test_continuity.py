from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from nova.agent.action import ActionExecutionResult
from nova.cli import build_runtime
from nova.console import InteractionConsole
from nova.continuity import SessionContinuityBuilder
from nova.types import TraceRecord, TurnRecord, ValidationResult


class ContinuityTests(unittest.TestCase):
    def test_summary_combines_presence_turns_actions_and_memory_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="summary-session")
            semantic_path = data_dir / "memory" / "semantic.jsonl"
            autobiographical_path = data_dir / "memory" / "autobiographical.jsonl"
            semantic_before = semantic_path.read_text(encoding="utf-8")
            autobiographical_before = autobiographical_path.read_text(encoding="utf-8")
            try:
                runtime.update_presence(
                    current_focus="reviewing continuity",
                    interaction_summary="Reviewed the current session state.",
                    pending_proposal={"goal": "explain current boundaries"},
                    last_action_status="proposal_proposed",
                    visible_uncertainties=["whether enough turns exist"],
                    user_confirmations_needed=["Approve or reject pending proposal"],
                )
                runtime.session_store.append_turn(
                    TurnRecord(
                        session_id="summary-session",
                        turn_id="turn-1",
                        timestamp="2026-04-22T10:00:00+00:00",
                        user_text="What are we working on?",
                        final_answer="We are reviewing session continuity.",
                        validation=ValidationResult(valid=True),
                        model_id="fake-model",
                    )
                )
                runtime.trace_logger.log_action_proposal(
                    session_id="summary-session",
                    proposal={
                        "goal": "explain current boundaries",
                        "disposition": "approval_required",
                        "reason": "approval_required",
                        "requires_approval": True,
                    },
                )
                runtime.trace_logger.log_action_execution(
                    session_id="summary-session",
                    execution=ActionExecutionResult(
                        goal="explain current boundaries",
                        status="approval_required",
                        executed=False,
                        reason="approval_required_before_execution",
                    ).to_dict(),
                )
                runtime.trace_logger.log_trace(
                    TraceRecord(
                        session_id="summary-session",
                        turn_id="turn-1",
                        timestamp="2026-04-22T10:00:00+00:00",
                        persisted_memory_events=[
                            {
                                "event_id": "mem-1",
                                "channel": "episodic",
                                "kind": "user_message",
                                "text": "What are we working on?",
                                "retention": "active",
                            }
                        ],
                    )
                )

                summary = SessionContinuityBuilder(runtime=runtime).build()

                self.assertEqual(summary.session_id, "summary-session")
                self.assertEqual(summary.current_focus, "reviewing continuity")
                self.assertIn("What are we working on?", summary.recent_user_inputs)
                self.assertIn("Pending proposal: explain current boundaries", summary.unresolved_items)
                action_kinds = {action["kind"] for action in summary.recent_action_attempts}
                self.assertIn("proposal", action_kinds)
                self.assertIn("execution", action_kinds)
                self.assertEqual(summary.recent_memory_activity[-1]["event_id"], "mem-1")
                self.assertEqual(semantic_path.read_text(encoding="utf-8"), semantic_before)
                self.assertEqual(
                    autobiographical_path.read_text(encoding="utf-8"),
                    autobiographical_before,
                )
            finally:
                runtime.close()

    def test_console_summary_uses_continuity_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="console-summary")
            console = InteractionConsole(runtime=runtime)
            try:
                runtime.update_presence(current_focus="console summary check")

                result = console.handle("/summary")

                self.assertTrue(result.handled)
                self.assertIn("Nova Session Summary", result.output)
                self.assertIn("current_focus: console summary check", result.output)
                self.assertIn("next_pickup:", result.output)
            finally:
                runtime.close()

    def test_summary_bounds_unresolved_items_from_presence_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(
                session_id="bounded-unresolved"
            )
            try:
                runtime.update_presence(
                    visible_uncertainties=[
                        f"uncertainty {index}" for index in range(4)
                    ],
                    user_confirmations_needed=[
                        f"confirmation {index}" for index in range(4)
                    ],
                    pending_proposal={"goal": "review bounded summary"},
                )

                summary = SessionContinuityBuilder(runtime=runtime).build()

                self.assertLessEqual(len(summary.unresolved_items), 5)
                self.assertLessEqual(len(summary.next_pickup), 3)
                self.assertEqual(
                    summary.unresolved_items[0],
                    "Pending proposal: review bounded summary",
                )
            finally:
                runtime.close()

    def _write_config(self, base: Path) -> tuple[Path, Path, Path]:
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
        return config_path, data_dir, log_dir


if __name__ == "__main__":
    unittest.main()
