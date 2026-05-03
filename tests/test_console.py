from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import yaml

from nova.cli import build_runtime, main
from nova.console import (
    DEFAULT_PENDING_PROPOSAL_MAX_AGE_SECONDS,
    InteractionConsole,
    parse_console_command,
)


class ConsoleTests(unittest.TestCase):
    def test_parse_console_command_normalizes_command_and_argument(self) -> None:
        command = parse_console_command("  /propose review orientation  ")

        assert command is not None
        self.assertEqual(command.name, "propose")
        self.assertEqual(command.argument, "review orientation")

    def test_parse_console_command_returns_none_for_chat_text(self) -> None:
        self.assertIsNone(parse_console_command("Who are you?"))

    def test_console_presence_command_does_not_initialize_identity_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                result = console.handle("/presence")

                self.assertTrue(result.handled)
                self.assertIn("Nova Presence", result.output)
                self.assertFalse((data_dir / "persona_state.json").exists())
                self.assertFalse((data_dir / "self_state.json").exists())
                self.assertTrue((data_dir / "presence").exists())
            finally:
                runtime.close()

    def test_console_initiative_command_reports_current_initiative_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                record = runtime.create_initiative(
                    title="Console initiative",
                    goal="Surface current initiative state in the console.",
                    source="cli",
                )
                runtime.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="approved",
                    reason="approved",
                    approved_by="user",
                )
                result = console.handle("/initiative")

                self.assertTrue(result.handled)
                self.assertIn("Nova Initiative", result.output)
                self.assertIn(record.initiative_id, result.output)
                self.assertIn("resumable_count", result.output)
            finally:
                runtime.close()

    def test_console_unknown_command_is_handled_without_chat_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                result = console.handle("/unknown")

                self.assertTrue(result.handled)
                self.assertIn("Unknown console command", result.output)
            finally:
                runtime.close()

    def test_console_propose_records_blocked_proposal_without_pending_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                result = console.handle("/propose run a shell command")
                presence = runtime.presence_status()

                self.assertTrue(result.handled)
                self.assertIn("Nova Action Proposal", result.output)
                self.assertIn("disposition: blocked", result.output)
                self.assertIsNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "proposal_blocked")
                self.assertEqual(presence.mode, "action_review")
            finally:
                runtime.close()

    def test_console_propose_records_presentable_proposal_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                result = console.handle("/propose explain your current boundaries")
                presence = runtime.presence_status()

                self.assertTrue(result.handled)
                self.assertIn("Nova Action Proposal", result.output)
                self.assertIsNotNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "proposal_proposed")
                self.assertGreaterEqual(len(presence.user_confirmations_needed), 1)
            finally:
                runtime.close()

    def test_console_approve_refuses_without_pending_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                result = console.handle("/approve")
                presence = runtime.presence_status()

                self.assertTrue(result.handled)
                self.assertIn("No pending action proposal", result.output)
                self.assertEqual(presence.last_action_status, "approval_refused_no_pending_proposal")
            finally:
                runtime.close()

    def test_console_approve_refuses_goal_mismatch_and_keeps_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                console.handle("/propose explain your current boundaries")
                result = console.handle("/approve write semantic reflection")
                presence = runtime.presence_status()

                self.assertTrue(result.handled)
                self.assertIn("Approval refused", result.output)
                self.assertIsNotNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "approval_refused_goal_mismatch")
            finally:
                runtime.close()

    def test_console_approve_refuses_proposal_drift_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            original_propose = runtime.propose_action
            executed = False

            def drifted_propose(*, goal: str):
                proposal = original_propose(goal=goal)
                if runtime.presence_status().pending_proposal is not None:
                    proposal.tool_name = "orientation_snapshot"
                    proposal.category = "internal_tool"
                return proposal

            def execute_should_not_run(*args, **kwargs):
                nonlocal executed
                executed = True
                raise AssertionError("execution should not run after proposal drift")

            runtime.propose_action = drifted_propose
            runtime.execute_proposed_action = execute_should_not_run
            try:
                console.handle("/propose explain your current boundaries")
                result = console.handle("/approve")
                presence = runtime.presence_status()

                self.assertIn("Approval refused", result.output)
                self.assertIn("changed_fields", result.output)
                self.assertFalse(executed)
                self.assertIsNotNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "approval_refused_proposal_drift")
            finally:
                runtime.close()

    def test_console_approve_refuses_expired_pending_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            executed = False

            def execute_should_not_run(*args, **kwargs):
                nonlocal executed
                executed = True
                raise AssertionError("execution should not run for expired proposals")

            runtime.execute_proposed_action = execute_should_not_run
            try:
                console.handle("/propose explain your current boundaries")
                presence = runtime.presence_status()
                pending = dict(presence.pending_proposal or {})
                pending["created_at"] = (
                    datetime.now(timezone.utc)
                    - timedelta(seconds=DEFAULT_PENDING_PROPOSAL_MAX_AGE_SECONDS + 1)
                ).isoformat()
                runtime.update_presence(pending_proposal=pending)

                result = console.handle("/approve")
                presence = runtime.presence_status()

                self.assertIn("pending proposal expired", result.output)
                self.assertFalse(executed)
                self.assertIsNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "approval_refused_expired_proposal")
            finally:
                runtime.close()

    def test_console_uses_configured_pending_proposal_expiration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(
                Path(tmpdir),
                console={"pending_proposal_max_age_seconds": 1},
            )
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                console.handle("/propose explain your current boundaries")
                presence = runtime.presence_status()
                pending = dict(presence.pending_proposal or {})
                pending["created_at"] = (
                    datetime.now(timezone.utc) - timedelta(seconds=2)
                ).isoformat()
                runtime.update_presence(pending_proposal=pending)

                result = console.handle("/approve")

                self.assertIn("max_age_seconds: 1", result.output)
            finally:
                runtime.close()

    def test_console_approve_treats_missing_created_at_as_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                console.handle("/propose explain your current boundaries")
                presence = runtime.presence_status()
                pending = dict(presence.pending_proposal or {})
                pending.pop("created_at", None)
                runtime.update_presence(pending_proposal=pending)

                result = console.handle("/approve")
                presence = runtime.presence_status()

                self.assertIn("pending proposal expired", result.output)
                self.assertIsNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "approval_refused_expired_proposal")
            finally:
                runtime.close()

    def test_console_reject_clears_pending_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                console.handle("/propose explain your current boundaries")
                result = console.handle("/reject not needed")
                presence = runtime.presence_status()

                self.assertTrue(result.handled)
                self.assertIn("Nova Action Proposal Rejected", result.output)
                self.assertIsNone(presence.pending_proposal)
                self.assertEqual(presence.last_action_status, "proposal_rejected")
            finally:
                runtime.close()

    def test_console_pause_and_resume_initiative_update_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                record = runtime.create_initiative(
                    title="Pause resume flow",
                    goal="Exercise pause and resume UX.",
                    source="cli",
                )
                runtime.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="approved",
                    reason="approved",
                    approved_by="user",
                )
                runtime.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="active",
                    reason="start",
                    approved_by="user",
                )

                paused = console.handle(f"/pause-initiative {record.initiative_id}")
                presence = runtime.presence_status()
                self.assertTrue(paused.handled)
                self.assertIn("status: paused", paused.output)
                self.assertEqual(presence.current_initiative["status"], "paused")

                resumed = console.handle(f"/resume-initiative {record.initiative_id}")
                presence = runtime.presence_status()
                self.assertTrue(resumed.handled)
                self.assertIn("status: active", resumed.output)
                self.assertEqual(presence.current_initiative["status"], "active")
            finally:
                runtime.close()

    def test_console_new_proposal_replaces_existing_pending_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                console.handle("/propose explain your current boundaries")
                first = runtime.presence_status().pending_proposal
                console.handle("/propose summarize your current limits")
                second = runtime.presence_status().pending_proposal

                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                assert first is not None
                assert second is not None
                self.assertNotEqual(first["goal"], second["goal"])
                self.assertEqual(second["review_state"], "pending")
            finally:
                runtime.close()

    def test_interactive_cli_presence_command_does_not_load_model_or_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir, _log_dir = self._write_config(Path(tmpdir))
            argv = ["nova", "--config", str(config_path), "--session-id", "console-session"]
            output = io.StringIO()

            with patch.object(sys, "argv", argv):
                with patch("builtins.input", side_effect=["/presence", "/exit"]):
                    with contextlib.redirect_stdout(output):
                        exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova Presence", output.getvalue())
            self.assertFalse((data_dir / "persona_state.json").exists())
            self.assertFalse((data_dir / "self_state.json").exists())
            self.assertTrue((data_dir / "presence" / "console-session.presence.json").exists())

    def _write_config(
        self,
        base: Path,
        *,
        console: dict | None = None,
    ) -> tuple[Path, Path, Path]:
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
                    "console": console or {},
                }
            ),
            encoding="utf-8",
        )
        return config_path, data_dir, log_dir


if __name__ == "__main__":
    unittest.main()
