from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nova.agent.initiative import (
    InitiativeTransitionError,
    JsonInitiativeStateStore,
)
from nova.cli import build_runtime


class InitiativeTests(unittest.TestCase):
    def test_initiative_store_creates_and_round_trips_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonInitiativeStateStore(Path(tmpdir) / "initiative")

            initiative_state = store.load(session_id="session-a")
            self.assertEqual(initiative_state.session_id, "session-a")
            self.assertEqual(initiative_state.initiatives, [])

            record = store.create_record(
                initiative_state=initiative_state,
                title="Phase 9 planning",
                goal="Carry the next planning objective across sessions.",
                approval_required=True,
                source="user-approved",
                evidence_refs=["trace:session-a:turn-1"],
                related_motive_refs=["motive:priority:planning"],
                notes=["created for Stage 9.1"],
            )
            store.transition(
                initiative_state=initiative_state,
                initiative_id=record.initiative_id,
                to_status="approved",
                reason="user approved continuation",
                approved_by="user",
                evidence_refs=["turn:approve"],
            )
            store.transition(
                initiative_state=initiative_state,
                initiative_id=record.initiative_id,
                to_status="active",
                reason="runtime resumed approved work",
                approved_by="user",
                evidence_refs=["turn:resume"],
            )
            store.save(initiative_state)

            loaded = store.load(session_id="session-a")
            self.assertEqual(len(loaded.initiatives), 1)
            self.assertEqual(loaded.active_initiative_id, record.initiative_id)
            loaded_record = loaded.initiatives[0]
            self.assertEqual(loaded_record.status, "active")
            self.assertEqual(loaded_record.related_motive_refs, ["motive:priority:planning"])
            self.assertEqual(len(loaded_record.transitions), 3)
            self.assertTrue(store.get_initiative_path(session_id="session-a").exists())

    def test_initiative_store_loads_across_minor_schema_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonInitiativeStateStore(Path(tmpdir) / "initiative")
            path = store.get_initiative_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.9",
                        "session_id": "wrong-session",
                        "active_initiative_id": "missing",
                        "initiatives": [
                            {
                                "initiative_id": "i-1",
                                "session_id": "wrong-session",
                                "title": 1,
                                "goal": True,
                                "status": "unsupported",
                                "approval_required": "truthy",
                                "approved_by": 7,
                                "evidence_refs": "invalid",
                                "related_motive_refs": [5],
                                "related_self_model_refs": "invalid",
                                "notes": [False],
                                "transitions": [
                                    {
                                        "from_status": "unsupported",
                                        "to_status": "unsupported",
                                        "reason": 2,
                                        "approved_by": 1,
                                        "evidence_refs": "invalid",
                                        "notes": [3],
                                    }
                                ],
                                "future_field": "ignored",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            loaded = store.load(session_id="session-a")

            self.assertEqual(loaded.session_id, "session-a")
            self.assertIsNone(loaded.active_initiative_id)
            self.assertEqual(len(loaded.initiatives), 1)
            record = loaded.initiatives[0]
            self.assertEqual(record.session_id, "session-a")
            self.assertEqual(record.title, "1")
            self.assertEqual(record.goal, "True")
            self.assertEqual(record.status, "pending")
            self.assertEqual(record.related_motive_refs, ["5"])
            self.assertEqual(record.related_self_model_refs, [])
            self.assertEqual(record.transitions[0].from_status, "pending")
            self.assertEqual(record.transitions[0].to_status, "pending")

    def test_initiative_store_recovers_from_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonInitiativeStateStore(Path(tmpdir) / "initiative")
            path = store.get_initiative_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not valid json", encoding="utf-8")

            initiative_state = store.load(session_id="session-a")

            self.assertEqual(initiative_state.session_id, "session-a")
            self.assertEqual(initiative_state.initiatives, [])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["session_id"], "session-a")

    def test_transition_requires_explicit_approval_for_approved_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonInitiativeStateStore(Path(tmpdir) / "initiative")
            initiative_state = store.load(session_id="session-a")
            record = store.create_record(
                initiative_state=initiative_state,
                title="Resume collaborative work",
                goal="Continue an approved task across sessions.",
            )

            with self.assertRaises(InitiativeTransitionError):
                store.transition(
                    initiative_state=initiative_state,
                    initiative_id=record.initiative_id,
                    to_status="approved",
                    reason="approval missing",
                )

    def test_runtime_initiative_updates_do_not_mutate_self_motive_or_presence_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.orientation_snapshot()
                motive_before = runtime.motive_status().to_dict()
                presence_before = runtime.presence_status().to_dict()
                assert runtime.self_state is not None
                self_before = runtime.self_state.to_dict()

                record = runtime.create_initiative(
                    title="Stage 9 implementation",
                    goal="Implement the initiative-state model cleanly.",
                    source="user-approved",
                    related_motive_refs=["motive:priority:stage9"],
                    evidence_refs=["trace:session:turn-1"],
                )
                runtime.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="approved",
                    reason="user approved the work",
                    approved_by="user",
                    evidence_refs=["turn:approve"],
                )

                initiative_state = runtime.initiative_status()
                self.assertEqual(len(initiative_state.initiatives), 1)
                self.assertEqual(initiative_state.initiatives[0].status, "approved")
                self.assertEqual(runtime.motive_status().to_dict(), motive_before)
                self.assertEqual(runtime.presence_status().to_dict(), presence_before)
                self.assertEqual(runtime.self_state.to_dict(), self_before)
                self.assertTrue(
                    (data_dir / "initiative" / f"{runtime.session_id}.initiative.json").exists()
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


if __name__ == "__main__":
    unittest.main()
