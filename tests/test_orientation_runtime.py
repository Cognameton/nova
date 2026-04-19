from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nova.cli import build_runtime


class OrientationRuntimeTests(unittest.TestCase):
    def test_runtime_orientation_snapshot_and_stability_log_without_model_load(self) -> None:
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
                        "eval": {
                            "enable_probes": True,
                            "orientation_stability_threshold": 0.8,
                            "orientation_min_runs": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            runtime = build_runtime(config_override=str(config_path))
            session_id = None
            try:
                snapshot = runtime.orientation_snapshot()
                session_id = runtime.session_id
                self.assertEqual(snapshot.identity["name"], "Nova")
                self.assertIn("external tool execution", snapshot.blocked_actions)

                evaluation = runtime.evaluate_orientation_stability(runs=2)
                self.assertTrue(evaluation.stable)
                self.assertGreaterEqual(evaluation.overall_score, 0.9)
            finally:
                runtime.close()

            orientation_log = log_dir / "traces" / f"{session_id}.orientation.jsonl"
            self.assertTrue(orientation_log.exists())
            payloads = [
                json.loads(line)
                for line in orientation_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(payloads), 1)
            self.assertEqual(payloads[-1]["session_id"], session_id)
            self.assertIn("snapshot", payloads[-1])
            self.assertIn("evaluation", payloads[-1])

            probe_log = log_dir / "probes.jsonl"
            self.assertTrue(probe_log.exists())
            probe_payloads = [
                json.loads(line)
                for line in probe_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            probe_types = {payload["probe_type"] for payload in probe_payloads}
            self.assertIn("orientation_stability_threshold", probe_types)
