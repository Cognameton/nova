from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nova.cli import build_runtime
from nova.agent.tools import ToolRequest


class ToolExecutorTests(unittest.TestCase):
    def _config_path(self, base: Path) -> Path:
        config_path = base / "local.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "data_dir": str(base / "data"),
                        "log_dir": str(base / "logs"),
                    },
                    "model": {
                        "model_path": "/tmp/fake.gguf",
                    },
                    "memory": {
                        "semantic_enabled": True,
                    },
                    "eval": {
                        "orientation_min_runs": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_executor_runs_orientation_snapshot_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                result = runtime.execute_internal_tool(
                    request=ToolRequest(tool_name="orientation_snapshot")
                )
                session_id = runtime.session_id
            finally:
                runtime.close()

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.output["identity"]["name"], "Nova")
            tool_log = base / "logs" / "traces" / f"{session_id}.tools.jsonl"
            self.assertTrue(tool_log.exists())
            payloads = [
                json.loads(line)
                for line in tool_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(payloads[-1]["result"]["status"], "ok")

    def test_executor_blocks_approval_required_tool_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                result = runtime.execute_internal_tool(
                    request=ToolRequest(tool_name="write_semantic_reflection")
                )
            finally:
                runtime.close()

            self.assertEqual(result.status, "approval_required")
            self.assertTrue(result.requires_approval)
            self.assertEqual(result.error, "approval_required")

    def test_executor_blocks_shell_even_when_ready_and_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                result = runtime.execute_internal_tool(
                    request=ToolRequest(tool_name="shell"),
                    approval_granted=True,
                )
            finally:
                runtime.close()

            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.error, "tool_blocked")


if __name__ == "__main__":
    unittest.main()
