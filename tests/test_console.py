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
from nova.console import InteractionConsole, parse_console_command


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

    def test_console_approve_requires_goal_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            console = InteractionConsole(runtime=runtime)
            try:
                result = console.handle("/approve")

                self.assertTrue(result.handled)
                self.assertEqual(result.output, "Usage: /approve <goal>")
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
