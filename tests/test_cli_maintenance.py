from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from nova.cli import run_maintenance_action


class CliMaintenanceTests(unittest.TestCase):
    def test_run_maintenance_plan_and_full_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
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

            episodic_path = data_dir / "memory" / "episodic.jsonl"
            episodic_path.parent.mkdir(parents=True, exist_ok=True)
            episodic_path.write_text(
                "\n".join(
                    [
                        '{"schema_version":"1.0","event_id":"e1","timestamp":"2026-04-18T00:00:00Z","session_id":"s1","turn_id":"t1","channel":"episodic","kind":"user_message","text":"I prefer local inference for Nova.","summary":null,"tags":["user","turn","preference"],"importance":0.75,"confidence":1.0,"continuity_weight":0.75,"retention":"active","supersedes":[],"source":"user","metadata":{}}',
                        '{"schema_version":"1.0","event_id":"e2","timestamp":"2026-04-18T00:01:00Z","session_id":"s1","turn_id":"t2","channel":"episodic","kind":"assistant_message","text":"My name is Nova. I remain focused on continuity.","summary":null,"tags":["assistant","turn","identity"],"importance":0.85,"confidence":1.0,"continuity_weight":0.95,"retention":"active","supersedes":[],"source":"nova","metadata":{}}',
                        '{"schema_version":"1.0","event_id":"e3","timestamp":"2026-04-18T00:02:00Z","session_id":"s1","turn_id":"t3","channel":"episodic","kind":"assistant_message","text":"I keep continuity at the center of my self-model.","summary":null,"tags":["assistant","turn","identity","value"],"importance":0.9,"confidence":1.0,"continuity_weight":0.95,"retention":"active","supersedes":[],"source":"nova","metadata":{}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            plan_result = run_maintenance_action(
                config_override=str(config_path),
                action="plan",
            )
            self.assertEqual(plan_result["action"], "plan")
            self.assertIn("summary", plan_result)

            full_result = run_maintenance_action(
                config_override=str(config_path),
                action="full",
            )
            self.assertEqual(full_result["action"], "full")
            self.assertGreaterEqual(full_result["semantic_written"], 1)
            self.assertGreaterEqual(full_result["autobiographical_written"], 1)
            self.assertGreaterEqual(full_result["identity_history_written"], 1)

    def test_write_autobiographical_reports_identity_history_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
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

            episodic_path = data_dir / "memory" / "episodic.jsonl"
            episodic_path.parent.mkdir(parents=True, exist_ok=True)
            episodic_path.write_text(
                "\n".join(
                    [
                        '{"schema_version":"1.0","event_id":"e1","timestamp":"2026-04-18T00:00:00Z","session_id":"s1","turn_id":"t1","channel":"episodic","kind":"assistant_message","text":"I remain oriented toward continuity in how I understand myself.","summary":null,"tags":["assistant","turn","identity"],"importance":0.82,"confidence":1.0,"continuity_weight":0.92,"retention":"active","supersedes":[],"source":"nova","metadata":{}}',
                        '{"schema_version":"1.0","event_id":"e2","timestamp":"2026-04-18T00:01:00Z","session_id":"s1","turn_id":"t2","channel":"episodic","kind":"assistant_message","text":"I keep continuity central when I decide how to answer.","summary":null,"tags":["assistant","turn","identity","value"],"importance":0.87,"confidence":1.0,"continuity_weight":0.93,"retention":"active","supersedes":[],"source":"nova","metadata":{}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_maintenance_action(
                config_override=str(config_path),
                action="write-autobiographical",
            )
            self.assertEqual(result["action"], "write-autobiographical")
            self.assertGreaterEqual(result["written"], 1)
            self.assertGreaterEqual(result["identity_history_written"], 1)


if __name__ == "__main__":
    unittest.main()
