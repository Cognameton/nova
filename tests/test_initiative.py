from __future__ import annotations

import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from nova.agent.initiative import (
    InitiativeTransitionError,
    JsonInitiativeStateStore,
)
from nova.cli import build_runtime, main


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

    def test_initiative_store_can_continue_approved_work_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonInitiativeStateStore(Path(tmpdir) / "initiative")
            source_state = store.load(session_id="session-a")
            record = store.create_record(
                initiative_state=source_state,
                title="Long-running collaboration",
                goal="Carry approved work into a later session.",
                evidence_refs=["turn:create"],
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=record.initiative_id,
                to_status="approved",
                reason="user approved the task",
                approved_by="user",
                evidence_refs=["turn:approve"],
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=record.initiative_id,
                to_status="active",
                reason="runtime began work",
                approved_by="user",
                evidence_refs=["turn:start"],
            )
            store.save(source_state)

            continued = store.continue_record(
                source_session_id="session-a",
                initiative_id=record.initiative_id,
                target_session_id="session-b",
                approved_by="user",
                reason="resume work in a fresh session",
                evidence_refs=["turn:resume"],
                notes=["continued into session-b"],
            )

            reloaded_source = store.load(session_id="session-a")
            reloaded_target = store.load(session_id="session-b")
            self.assertEqual(reloaded_source.initiatives[0].status, "paused")
            self.assertIn("session-b", reloaded_source.initiatives[0].continuation_session_ids)
            self.assertEqual(len(reloaded_target.initiatives), 1)
            target_record = reloaded_target.initiatives[0]
            self.assertEqual(target_record.status, "approved")
            self.assertEqual(target_record.intent_id, record.intent_id)
            self.assertEqual(target_record.origin_session_id, "session-a")
            self.assertEqual(target_record.continued_from_session_id, "session-a")
            self.assertEqual(target_record.continued_from_initiative_id, record.initiative_id)
            self.assertEqual(continued.initiative_id, target_record.initiative_id)

    def test_resumable_initiatives_exclude_terminal_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonInitiativeStateStore(Path(tmpdir) / "initiative")
            source_state = store.load(session_id="session-a")
            active_record = store.create_record(
                initiative_state=source_state,
                title="Active work",
                goal="Resume later.",
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=active_record.initiative_id,
                to_status="approved",
                reason="approved",
                approved_by="user",
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=active_record.initiative_id,
                to_status="active",
                reason="active",
                approved_by="user",
            )
            terminal_record = store.create_record(
                initiative_state=source_state,
                title="Finished work",
                goal="Do not resume.",
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=terminal_record.initiative_id,
                to_status="approved",
                reason="approved",
                approved_by="user",
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=terminal_record.initiative_id,
                to_status="active",
                reason="active",
                approved_by="user",
            )
            store.transition(
                initiative_state=source_state,
                initiative_id=terminal_record.initiative_id,
                to_status="completed",
                reason="done",
                approved_by="user",
            )
            store.save(source_state)

            resumable = store.resumable_records()
            resumable_ids = {record.initiative_id for record in resumable}
            self.assertIn(active_record.initiative_id, resumable_ids)
            self.assertNotIn(terminal_record.initiative_id, resumable_ids)

    def test_runtime_initiative_updates_do_not_mutate_self_or_motive_state(self) -> None:
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
                self.assertEqual(runtime.self_state.to_dict(), self_before)
                presence_after = runtime.presence_status().to_dict()
                self.assertNotEqual(presence_after, presence_before)
                self.assertEqual(
                    presence_after["current_initiative"]["initiative_id"],
                    record.initiative_id,
                )
                self.assertEqual(
                    presence_after["current_initiative"]["status"],
                    "approved",
                )
                self.assertTrue(
                    (data_dir / "initiative" / f"{runtime.session_id}.initiative.json").exists()
                )
            finally:
                runtime.close()

    def test_runtime_can_continue_approved_initiative_into_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir = self._write_config(Path(tmpdir))
            runtime_a = build_runtime(config_override=str(config_path))
            runtime_b = build_runtime(config_override=str(config_path))
            try:
                runtime_a.session_id = runtime_a.session_store.start_session(session_id="session-a")
                runtime_a.initiative_state = runtime_a.initiative_store.load(session_id="session-a")
                record = runtime_a.create_initiative(
                    title="Cross-session implementation",
                    goal="Resume approved work in a later session.",
                    source="user-approved",
                    evidence_refs=["turn:create"],
                )
                runtime_a.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="approved",
                    reason="user approved the initiative",
                    approved_by="user",
                    evidence_refs=["turn:approve"],
                )
                runtime_a.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="active",
                    reason="runtime began work",
                    approved_by="user",
                    evidence_refs=["turn:start"],
                )

                runtime_b.session_id = runtime_b.session_store.start_session(session_id="session-b")
                runtime_b.initiative_state = runtime_b.initiative_store.load(session_id="session-b")

                continued = runtime_b.continue_initiative(
                    source_session_id="session-a",
                    initiative_id=record.initiative_id,
                    approved_by="user",
                    reason="resume work after interruption",
                    evidence_refs=["turn:resume"],
                )

                current = runtime_b.initiative_status()
                self.assertEqual(len(current.initiatives), 1)
                self.assertEqual(current.initiatives[0].initiative_id, continued.initiative_id)
                self.assertEqual(current.initiatives[0].continued_from_session_id, "session-a")
                self.assertEqual(current.initiatives[0].continued_from_initiative_id, record.initiative_id)
                self.assertTrue((data_dir / "initiative" / "session-a.initiative.json").exists())
                self.assertTrue((data_dir / "initiative" / "session-b.initiative.json").exists())
            finally:
                runtime_a.close()
                runtime_b.close()

    def test_initiative_cli_can_create_and_show_initiative_without_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir = self._write_config(Path(tmpdir))
            create_argv = [
                "nova",
                "--config",
                str(config_path),
                "--session-id",
                "initiative-cli",
                "--initiative-create",
                "Continue the current collaborative work.",
                "--initiative-title",
                "CLI initiative",
            ]
            create_output = io.StringIO()
            with patch.object(sys, "argv", create_argv):
                with contextlib.redirect_stdout(create_output):
                    create_exit = main()

            self.assertEqual(create_exit, 0)
            self.assertIn("Nova 2.0 Initiative Created", create_output.getvalue())
            self.assertTrue((data_dir / "initiative" / "initiative-cli.initiative.json").exists())

            show_argv = [
                "nova",
                "--config",
                str(config_path),
                "--session-id",
                "initiative-cli",
                "--initiative",
            ]
            show_output = io.StringIO()
            with patch.object(sys, "argv", show_argv):
                with contextlib.redirect_stdout(show_output):
                    show_exit = main()

            self.assertEqual(show_exit, 0)
            self.assertIn("Nova 2.0 Initiative", show_output.getvalue())
            self.assertIn("initiative_count: 1", show_output.getvalue())

    def test_initiative_cli_can_continue_existing_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, data_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.session_id = runtime.session_store.start_session(session_id="session-a")
                runtime.initiative_state = runtime.initiative_store.load(session_id="session-a")
                record = runtime.create_initiative(
                    title="CLI continuation",
                    goal="Continue approved work into a fresh session.",
                    source="cli",
                )
                runtime.transition_initiative(
                    initiative_id=record.initiative_id,
                    to_status="approved",
                    reason="approved",
                    approved_by="user",
                )
            finally:
                runtime.close()

            argv = [
                "nova",
                "--config",
                str(config_path),
                "--session-id",
                "session-b",
                "--continue-initiative",
                record.initiative_id,
                "--initiative-source-session",
                "session-a",
                "--initiative-approved-by",
                "user",
            ]
            output = io.StringIO()
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Initiative Continued", output.getvalue())
            self.assertTrue((data_dir / "initiative" / "session-b.initiative.json").exists())

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
