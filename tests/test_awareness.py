from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nova.agent.awareness import JsonAwarenessStateStore
from nova.cli import build_runtime
from nova.types import AwarenessState


class AwarenessTests(unittest.TestCase):
    def test_awareness_store_creates_and_round_trips_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonAwarenessStateStore(Path(tmpdir) / "awareness")

            awareness = store.load(session_id="session-a")
            self.assertEqual(awareness.session_id, "session-a")
            self.assertEqual(awareness.monitoring_mode, "bounded")
            self.assertTrue(awareness.self_signals)

            awareness.monitoring_mode = "attentive"
            awareness.self_signals = ["track continuity and initiative pressure"]
            awareness.world_signals = ["user is asking about current task state"]
            awareness.active_pressures = ["paused initiative remains resumable"]
            awareness.candidate_goal_signals = ["resume bounded approved initiative"]
            awareness.dominant_attention = "initiative continuity under interruption"
            awareness.evidence_refs = ["trace:session-a:turn-1"]
            store.save(awareness)

            loaded = store.load(session_id="session-a")
            self.assertEqual(loaded.monitoring_mode, "attentive")
            self.assertEqual(loaded.self_signals, ["track continuity and initiative pressure"])
            self.assertEqual(loaded.world_signals, ["user is asking about current task state"])
            self.assertEqual(loaded.active_pressures, ["paused initiative remains resumable"])
            self.assertEqual(loaded.candidate_goal_signals, ["resume bounded approved initiative"])
            self.assertEqual(loaded.dominant_attention, "initiative continuity under interruption")
            self.assertEqual(loaded.evidence_refs, ["trace:session-a:turn-1"])
            self.assertTrue(store.get_awareness_path(session_id="session-a").exists())

    def test_awareness_store_normalizes_unknown_monitoring_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonAwarenessStateStore(Path(tmpdir) / "awareness")
            awareness = AwarenessState(session_id="session-a", monitoring_mode="unsupported")

            store.save(awareness)

            self.assertEqual(store.load(session_id="session-a").monitoring_mode, "bounded")

    def test_awareness_store_loads_across_minor_schema_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonAwarenessStateStore(Path(tmpdir) / "awareness")
            path = store.get_awareness_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.9",
                        "session_id": "wrong-session",
                        "monitoring_mode": "unsupported",
                        "self_signals": [1, "continuity"],
                        "world_signals": "invalid",
                        "active_pressures": [True],
                        "candidate_goal_signals": "invalid",
                        "dominant_attention": 7,
                        "evidence_refs": "invalid",
                        "future_field": "ignored",
                    }
                ),
                encoding="utf-8",
            )

            awareness = store.load(session_id="session-a")

            self.assertEqual(awareness.session_id, "session-a")
            self.assertEqual(awareness.monitoring_mode, "bounded")
            self.assertEqual(awareness.self_signals, ["1", "continuity"])
            self.assertEqual(awareness.world_signals, [])
            self.assertEqual(awareness.active_pressures, ["True"])
            self.assertEqual(awareness.candidate_goal_signals, [])
            self.assertEqual(awareness.dominant_attention, "7")
            self.assertEqual(awareness.evidence_refs, [])

    def test_awareness_store_recovers_from_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonAwarenessStateStore(Path(tmpdir) / "awareness")
            path = store.get_awareness_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not valid json", encoding="utf-8")

            awareness = store.load(session_id="session-a")

            self.assertEqual(awareness.session_id, "session-a")
            self.assertEqual(awareness.monitoring_mode, "bounded")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["session_id"], "session-a")

    def test_runtime_awareness_updates_do_not_mutate_self_motive_or_initiative_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.orientation_snapshot()
                motive_before = runtime.motive_status().to_dict()
                initiative_before = runtime.initiative_status().to_dict()
                assert runtime.self_state is not None
                self_before = runtime.self_state.to_dict()

                updated = runtime.update_awareness(
                    monitoring_mode="attentive",
                    self_signals=["track continuity and initiative coherence"],
                    world_signals=["user is requesting awareness-stage planning"],
                    active_pressures=["phase planning remains open"],
                    candidate_goal_signals=["define awareness-state schema"],
                    dominant_attention="awareness-state separation from motive and initiative",
                    evidence_refs=["trace:session-a:turn-1"],
                )

                self.assertEqual(updated.monitoring_mode, "attentive")
                self.assertEqual(runtime.self_state.to_dict(), self_before)
                self.assertEqual(runtime.motive_status().to_dict(), motive_before)
                self.assertEqual(runtime.initiative_status().to_dict(), initiative_before)
                self.assertTrue(
                    (data_dir / "awareness" / f"{runtime.session_id}.awareness.json").exists()
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
