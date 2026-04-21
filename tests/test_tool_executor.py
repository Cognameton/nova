from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nova.cli import build_runtime
from nova.agent.tools import ToolRequest
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.types import MemoryEvent


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

    def test_executor_runs_approved_semantic_reflection_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            episodic = JsonlEpisodicMemoryStore(base / "data" / "memory" / "episodic.jsonl")
            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-20T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="user_message",
                    text="I prefer local inference for Nova.",
                    tags=["user", "turn", "preference"],
                    importance=0.75,
                    confidence=1.0,
                    continuity_weight=0.75,
                    source="user",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e2",
                    timestamp="2026-04-20T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="user_message",
                    text="I want Nova to stay local-first.",
                    tags=["user", "turn", "preference"],
                    importance=0.8,
                    confidence=1.0,
                    continuity_weight=0.8,
                    source="user",
                )
            )

            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                blocked = runtime.execute_internal_tool(
                    request=ToolRequest(tool_name="write_semantic_reflection")
                )
                approved = runtime.execute_internal_tool(
                    request=ToolRequest(tool_name="write_semantic_reflection"),
                    approval_granted=True,
                )
            finally:
                runtime.close()

            self.assertEqual(blocked.status, "approval_required")
            self.assertEqual(approved.status, "ok")
            self.assertGreaterEqual(approved.output["written"], 1)
            self.assertTrue(approved.output["orientation_stable"])

    def test_executor_runs_approved_autobiographical_reflection_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            episodic = JsonlEpisodicMemoryStore(base / "data" / "memory" / "episodic.jsonl")
            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-20T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="assistant_message",
                    text="My name is Nova. I remain focused on continuity.",
                    tags=["assistant", "turn", "identity"],
                    importance=0.85,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e2",
                    timestamp="2026-04-20T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="assistant_message",
                    text="I keep continuity at the center of my self-model.",
                    tags=["assistant", "turn", "identity", "value"],
                    importance=0.9,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )

            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                approved = runtime.execute_internal_tool(
                    request=ToolRequest(tool_name="write_autobiographical_reflection"),
                    approval_granted=True,
                )
            finally:
                runtime.close()

            self.assertEqual(approved.status, "ok")
            self.assertGreaterEqual(approved.output["written"], 1)
            self.assertTrue(approved.output["orientation_stable"])


if __name__ == "__main__":
    unittest.main()
