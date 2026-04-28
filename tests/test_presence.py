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

from nova.agent.presence import JsonPresenceStore, PresenceState
from nova.cli import build_runtime, main


class PresenceTests(unittest.TestCase):
    def test_presence_store_creates_and_round_trips_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonPresenceStore(Path(tmpdir) / "presence")

            presence = store.load(session_id="session-a")
            self.assertEqual(presence.session_id, "session-a")
            self.assertEqual(presence.mode, "conversation")
            self.assertEqual(presence.current_focus, "open conversation")

            presence.mode = "diagnostics"
            presence.current_focus = "reviewing continuity"
            presence.visible_uncertainties = ["whether context is complete"]
            store.save(presence)

            loaded = store.load(session_id="session-a")
            self.assertEqual(loaded.mode, "diagnostics")
            self.assertEqual(loaded.current_focus, "reviewing continuity")
            self.assertEqual(loaded.visible_uncertainties, ["whether context is complete"])
            self.assertTrue(store.get_presence_path(session_id="session-a").exists())

    def test_presence_store_normalizes_unknown_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonPresenceStore(Path(tmpdir) / "presence")
            presence = PresenceState(session_id="session-a", mode="unsupported")

            store.save(presence)

            self.assertEqual(store.load(session_id="session-a").mode, "conversation")

    def test_presence_store_loads_across_minor_schema_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonPresenceStore(Path(tmpdir) / "presence")
            path = store.get_presence_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.9",
                        "session_id": "wrong-session",
                        "mode": "unsupported",
                        "current_focus": "schema check",
                        "pending_proposal": "invalid",
                        "visible_uncertainties": "invalid",
                        "user_confirmations_needed": [1, "confirm"],
                        "future_field": "ignored",
                    }
                ),
                encoding="utf-8",
            )

            presence = store.load(session_id="session-a")

            self.assertEqual(presence.session_id, "session-a")
            self.assertEqual(presence.mode, "conversation")
            self.assertEqual(presence.current_focus, "schema check")
            self.assertIsNone(presence.pending_proposal)
            self.assertEqual(presence.visible_uncertainties, [])
            self.assertEqual(presence.user_confirmations_needed, ["1", "confirm"])
            self.assertEqual(presence.interaction_summary, "")

    def test_presence_store_recovers_from_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonPresenceStore(Path(tmpdir) / "presence")
            path = store.get_presence_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not valid json", encoding="utf-8")

            presence = store.load(session_id="session-a")

            self.assertEqual(presence.session_id, "session-a")
            self.assertEqual(presence.mode, "conversation")
            self.assertEqual(presence.current_focus, "open conversation")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["session_id"], "session-a")

    def test_runtime_presence_updates_do_not_mutate_self_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                presence = runtime.presence_status()
                self.assertEqual(presence.mode, "conversation")
                runtime.orientation_snapshot()
                assert runtime.self_state is not None
                self_before = runtime.self_state.to_dict()

                updated = runtime.update_presence(
                    mode="diagnostics",
                    current_focus="presence test",
                    visible_uncertainties=["presence is session scoped"],
                    user_confirmations_needed=["continue Stage 4.1"],
                )

                self.assertEqual(updated.mode, "diagnostics")
                self.assertEqual(updated.current_focus, "presence test")
                self.assertEqual(runtime.self_state.to_dict(), self_before)
                self.assertTrue(
                    (data_dir / "presence" / f"{runtime.session_id}.presence.json").exists()
                )
            finally:
                runtime.close()

    def test_runtime_presence_can_clear_nullable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.update_presence(
                    pending_proposal={"goal": "review orientation"},
                    last_action_status="proposed",
                )

                cleared = runtime.update_presence(
                    pending_proposal=None,
                    last_action_status=None,
                )

                self.assertIsNone(cleared.pending_proposal)
                self.assertIsNone(cleared.last_action_status)
            finally:
                runtime.close()

    def test_presence_cli_does_not_require_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            argv = [
                "nova",
                "--config",
                str(config_path),
                "--session-id",
                "presence-cli",
                "--presence",
            ]
            output = io.StringIO()

            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Presence", output.getvalue())
            self.assertIn("session_id: presence-cli", output.getvalue())
            self.assertFalse((_data_dir / "persona_state.json").exists())
            self.assertFalse((_data_dir / "self_state.json").exists())
            self.assertTrue((_data_dir / "presence" / "presence-cli.presence.json").exists())

    def test_backend_check_cli_outputs_report_and_exit_code(self) -> None:
        argv = [
            "nova",
            "--backend-check",
            "--backend-check-prompt",
            "Say backend check OK.",
        ]
        output = io.StringIO()

        with patch.object(sys, "argv", argv):
            with patch("nova.cli.run_backend_check") as backend_check:
                backend_check.return_value = {
                    "session_id": "backend-check",
                    "backend": "fake",
                    "model_name": "fake-model",
                    "model_path": "/tmp/fake.gguf",
                    "prompt_token_estimate": 42,
                    "finish_reason": "stop",
                    "prompt_tokens": 42,
                    "completion_tokens": 8,
                    "latency_ms": 1,
                    "validation": {"valid": True, "violations": []},
                    "final_answer": "Backend check OK.",
                }
                with contextlib.redirect_stdout(output):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Nova 2.0 Backend Check", output.getvalue())
        self.assertIn("valid: True", output.getvalue())
        self.assertIn("Backend check OK.", output.getvalue())


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
