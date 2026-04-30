from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nova.agent.motive import JsonMotiveStateStore, MotiveState
from nova.cli import build_runtime


class MotiveTests(unittest.TestCase):
    def test_motive_store_creates_and_round_trips_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonMotiveStateStore(Path(tmpdir) / "motive")

            motive = store.load(session_id="session-a")
            self.assertEqual(motive.session_id, "session-a")
            self.assertEqual(motive.claim_posture, "conservative")
            self.assertTrue(motive.current_priorities)

            motive.current_priorities = ["preserve user-stated continuity preferences"]
            motive.local_goals = ["earn stronger self-claims carefully"]
            motive.evidence_refs = ["trace:session-a:turn-1"]
            motive.claim_posture = "evidence-backed"
            store.save(motive)

            loaded = store.load(session_id="session-a")
            self.assertEqual(loaded.current_priorities, ["preserve user-stated continuity preferences"])
            self.assertEqual(loaded.local_goals, ["earn stronger self-claims carefully"])
            self.assertEqual(loaded.evidence_refs, ["trace:session-a:turn-1"])
            self.assertEqual(loaded.claim_posture, "evidence-backed")
            self.assertTrue(store.get_motive_path(session_id="session-a").exists())

    def test_motive_store_normalizes_unknown_claim_posture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonMotiveStateStore(Path(tmpdir) / "motive")
            motive = MotiveState(session_id="session-a", claim_posture="unsupported")

            store.save(motive)

            self.assertEqual(store.load(session_id="session-a").claim_posture, "conservative")

    def test_motive_store_loads_across_minor_schema_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonMotiveStateStore(Path(tmpdir) / "motive")
            path = store.get_motive_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.9",
                        "session_id": "wrong-session",
                        "claim_posture": "unsupported",
                        "current_priorities": [1, "continuity"],
                        "active_tensions": "invalid",
                        "local_goals": [True],
                        "evidence_refs": "invalid",
                        "future_field": "ignored",
                    }
                ),
                encoding="utf-8",
            )

            motive = store.load(session_id="session-a")

            self.assertEqual(motive.session_id, "session-a")
            self.assertEqual(motive.claim_posture, "conservative")
            self.assertEqual(motive.current_priorities, ["1", "continuity"])
            self.assertEqual(motive.active_tensions, [])
            self.assertEqual(motive.local_goals, ["True"])
            self.assertEqual(motive.evidence_refs, [])

    def test_motive_store_recovers_from_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonMotiveStateStore(Path(tmpdir) / "motive")
            path = store.get_motive_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not valid json", encoding="utf-8")

            motive = store.load(session_id="session-a")

            self.assertEqual(motive.session_id, "session-a")
            self.assertEqual(motive.claim_posture, "conservative")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["session_id"], "session-a")

    def test_runtime_motive_updates_do_not_mutate_self_or_presence_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                motive = runtime.motive_status()
                self.assertEqual(motive.claim_posture, "conservative")
                runtime.orientation_snapshot()
                presence = runtime.presence_status()
                assert runtime.self_state is not None
                self_before = runtime.self_state.to_dict()
                presence_before = presence.to_dict()

                updated = runtime.update_motive(
                    current_priorities=["preserve continuity under self-inquiry"],
                    active_tensions=["do not overclaim beyond evidence"],
                    local_goals=["model current motive-state explicitly"],
                    claim_posture="uncertainty-marked",
                    evidence_refs=["memory:semantic:current-priority"],
                )

                self.assertEqual(updated.claim_posture, "uncertainty-marked")
                self.assertEqual(runtime.self_state.to_dict(), self_before)
                self.assertEqual(runtime.presence_status().to_dict(), presence_before)
                self.assertTrue(
                    (data_dir / "motive" / f"{runtime.session_id}.motive.json").exists()
                )
            finally:
                runtime.close()

    def _write_config(self, base: Path) -> tuple[Path, Path]:
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
        return config_path, data_dir
