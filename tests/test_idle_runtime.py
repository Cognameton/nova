from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nova.agent.idle import (
    BoundedIdleController,
    IdleRuntimePromptEngine,
    JsonIdleRuntimeStore,
    default_idle_runtime_status,
    idle_budget_from_payload,
    idle_runtime_status_from_payload,
    idle_tick_record_from_payload,
    normalize_idle_lifecycle_state,
)
from nova.agent.initiative import default_initiative_state
from nova.types import (
    AwarenessState,
    ClaimGateDecision,
    IdleBudget,
    IdleRuntimeStatus,
    IdleTickRecord,
    MotiveState,
    PrivateCognitionPacket,
    SelfState,
)


class IdleRuntimeStateTests(unittest.TestCase):
    def test_default_status_is_stopped_and_not_active(self) -> None:
        status = default_idle_runtime_status(session_id="session-a")

        self.assertEqual(status.session_id, "session-a")
        self.assertEqual(status.lifecycle_state, "stopped")
        self.assertFalse(status.active)
        self.assertIsInstance(status.budget, IdleBudget)

    def test_lifecycle_state_normalization(self) -> None:
        self.assertEqual(normalize_idle_lifecycle_state(" IDLE "), "idle")
        self.assertEqual(normalize_idle_lifecycle_state("unsupported"), "stopped")

    def test_budget_payload_normalizes_nonnegative_values(self) -> None:
        budget = idle_budget_from_payload(
            {
                "max_ticks": "3",
                "ticks_used": -1,
                "max_runtime_seconds": "bad",
                "runtime_seconds_used": 7,
                "max_tokens": 128,
                "tokens_used": "4",
                "evaluation_mode": True,
                "future_field": "ignored",
            }
        )

        self.assertEqual(budget.max_ticks, 3)
        self.assertEqual(budget.ticks_used, 0)
        self.assertEqual(budget.max_runtime_seconds, 0)
        self.assertEqual(budget.runtime_seconds_used, 7)
        self.assertEqual(budget.max_tokens, 128)
        self.assertEqual(budget.tokens_used, 4)
        self.assertTrue(budget.evaluation_mode)

    def test_status_payload_loads_across_minor_schema_changes(self) -> None:
        status = idle_runtime_status_from_payload(
            payload={
                "schema_version": "0.9",
                "session_id": "wrong-session",
                "lifecycle_state": "idle",
                "active": False,
                "last_tick_id": 9,
                "budget": {"max_ticks": "2", "ticks_used": "1"},
                "evidence_refs": ["tick:a", 2],
                "notes": "invalid",
                "future_field": "ignored",
            },
            session_id="session-a",
        )

        self.assertEqual(status.session_id, "session-a")
        self.assertEqual(status.lifecycle_state, "idle")
        self.assertTrue(status.active)
        self.assertEqual(status.last_tick_id, "9")
        self.assertEqual(status.budget.max_ticks, 2)
        self.assertEqual(status.budget.ticks_used, 1)
        self.assertEqual(status.evidence_refs, ["tick:a", "2"])
        self.assertEqual(status.notes, [])

    def test_tick_payload_preserves_recorded_idle_cognition_evidence(self) -> None:
        tick = idle_tick_record_from_payload(
            payload={
                "tick_id": "tick-1",
                "session_id": "wrong-session",
                "sequence": "4",
                "timestamp": "2026-05-05T12:00:00+00:00",
                "trigger": "timer",
                "lifecycle_state": "idle",
                "budget_snapshot": {"ticks_used": 1},
                "state_inputs": {"active_user_task": False},
                "capability_appraisal": {"current_capabilities": ["text_generation"]},
                "idle_pressure_appraisal": {"idle_state_detected": True},
                "candidate_internal_goals": [{"candidate_id": "candidate-1"}],
                "selected_internal_goal": {"candidate_id": "candidate-1"},
                "internal_goal_initiative_proposal": {"creates_initiative": False},
                "stop_reason": "budget_remaining",
                "evidence_refs": ["appraisal:tick-1"],
                "notes": [5],
                "future_field": "ignored",
            },
            session_id="session-a",
        )

        self.assertEqual(tick.session_id, "session-a")
        self.assertEqual(tick.tick_id, "tick-1")
        self.assertEqual(tick.sequence, 4)
        self.assertEqual(tick.lifecycle_state, "idle")
        self.assertEqual(tick.budget_snapshot, {"ticks_used": 1})
        self.assertEqual(tick.state_inputs, {"active_user_task": False})
        self.assertEqual(tick.candidate_internal_goals, [{"candidate_id": "candidate-1"}])
        self.assertEqual(tick.evidence_refs, ["appraisal:tick-1"])
        self.assertEqual(tick.notes, ["5"])


class JsonIdleRuntimeStoreTests(unittest.TestCase):
    def test_store_creates_and_round_trips_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")

            status = store.load_status(session_id="session-a")
            self.assertEqual(status.lifecycle_state, "stopped")
            self.assertTrue(store.get_status_path(session_id="session-a").exists())

            status.lifecycle_state = "idle"
            status.budget = IdleBudget(max_ticks=3, ticks_used=1, evaluation_mode=True)
            status.evidence_refs = ["idle:tick-1"]
            store.save_status(status)

            loaded = store.load_status(session_id="session-a")
            self.assertEqual(loaded.lifecycle_state, "idle")
            self.assertTrue(loaded.active)
            self.assertEqual(loaded.budget.max_ticks, 3)
            self.assertEqual(loaded.budget.ticks_used, 1)
            self.assertEqual(loaded.evidence_refs, ["idle:tick-1"])

    def test_store_recovers_from_malformed_status_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            path = store.get_status_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not valid json", encoding="utf-8")

            status = store.load_status(session_id="session-a")

            self.assertEqual(status.session_id, "session-a")
            self.assertEqual(status.lifecycle_state, "stopped")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["session_id"], "session-a")

    def test_append_tick_persists_evidence_and_updates_status_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            status = IdleRuntimeStatus(
                session_id="session-a",
                lifecycle_state="idle",
                budget=IdleBudget(max_ticks=3),
            )
            store.save_status(status)

            tick = store.append_tick(
                IdleTickRecord(
                    tick_id="tick-1",
                    session_id="session-a",
                    sequence=1,
                    lifecycle_state="idle",
                    idle_pressure_appraisal={"idle_state_detected": True},
                    evidence_refs=["idle:tick-1", "appraisal:tick-1"],
                )
            )

            self.assertEqual(tick.tick_id, "tick-1")
            self.assertTrue(store.has_recorded_idle_cognition(session_id="session-a"))
            loaded_ticks = store.list_ticks(session_id="session-a")
            self.assertEqual(len(loaded_ticks), 1)
            self.assertEqual(loaded_ticks[0].evidence_refs, ["idle:tick-1", "appraisal:tick-1"])
            loaded_status = store.load_status(session_id="session-a")
            self.assertEqual(loaded_status.last_tick_id, "tick-1")
            self.assertEqual(
                loaded_status.evidence_refs,
                ["idle:tick-1", "appraisal:tick-1"],
            )

    def test_no_ticks_means_no_recorded_idle_cognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")

            self.assertFalse(store.has_recorded_idle_cognition(session_id="session-a"))
            self.assertEqual(store.list_ticks(session_id="session-a"), [])

    def test_list_ticks_skips_malformed_jsonl_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            path = store.get_ticks_path(session_id="session-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"tick_id": "tick-1", "sequence": 1}),
                        "{bad json",
                        json.dumps({"tick_id": "tick-2", "sequence": 2}),
                    ]
                ),
                encoding="utf-8",
            )

            ticks = store.list_ticks(session_id="session-a", limit=1)

            self.assertEqual(len(ticks), 1)
            self.assertEqual(ticks[0].tick_id, "tick-2")


class BoundedIdleControllerTests(unittest.TestCase):
    def test_controller_records_bounded_idle_tick_without_creating_initiative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            controller = BoundedIdleController(store=store)
            controller.start(session_id="session-a", budget=IdleBudget(max_ticks=1))

            tick = controller.tick(
                session_id="session-a",
                self_state=SelfState(active_questions=["What should I clarify while idle?"]),
                motive_state=MotiveState(
                    session_id="session-a",
                    active_tensions=["resolve uncertainty before stronger claims"],
                ),
                initiative_state=default_initiative_state(session_id="session-a"),
                awareness_state=AwarenessState(
                    session_id="session-a",
                    candidate_goal_signals=["clarify idle cognition boundary"],
                    evidence_refs=["awareness:session-a"],
                ),
                private_cognition=PrivateCognitionPacket(uncertainty_flag=True),
                claim_gate=ClaimGateDecision(),
            )

            self.assertEqual(tick.sequence, 1)
            self.assertEqual(tick.stop_reason, "budget_exhausted")
            self.assertTrue(tick.idle_pressure_appraisal)
            self.assertTrue(tick.candidate_internal_goals)
            self.assertTrue(tick.selected_internal_goal["selected"])
            self.assertFalse(tick.internal_goal_initiative_proposal["creates_initiative"])
            self.assertIn("no external action executed", " ".join(tick.notes))
            self.assertTrue(store.has_recorded_idle_cognition(session_id="session-a"))
            status = store.load_status(session_id="session-a")
            self.assertEqual(status.lifecycle_state, "stopped")
            self.assertEqual(status.budget.ticks_used, 1)

    def test_controller_refuses_tick_while_paused_without_recording_cognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            controller = BoundedIdleController(store=store)
            controller.start(session_id="session-a", budget=IdleBudget(max_ticks=3))
            controller.pause(session_id="session-a")

            tick = controller.tick(
                session_id="session-a",
                self_state=SelfState(open_tensions=["idle pressure exists"]),
                motive_state=MotiveState(session_id="session-a"),
                initiative_state=default_initiative_state(session_id="session-a"),
                awareness_state=AwarenessState(
                    session_id="session-a",
                    active_pressures=["idle pressure exists"],
                ),
            )

            self.assertIn("lifecycle_not_active:paused", tick.stop_reason)
            self.assertFalse(tick.idle_pressure_appraisal)
            self.assertFalse(store.has_recorded_idle_cognition(session_id="session-a"))
            status = store.load_status(session_id="session-a")
            self.assertEqual(status.lifecycle_state, "paused")

    def test_controller_resume_and_stop_update_lifecycle_without_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            controller = BoundedIdleController(store=store)
            controller.start(session_id="session-a", budget=IdleBudget(max_ticks=2))
            controller.pause(session_id="session-a")
            resumed = controller.resume(session_id="session-a")
            stopped = controller.stop(session_id="session-a", reason="operator_stop")

            self.assertEqual(resumed.lifecycle_state, "running")
            self.assertTrue(resumed.active)
            self.assertEqual(stopped.lifecycle_state, "stopped")
            self.assertFalse(stopped.active)
            self.assertEqual(stopped.last_stop_reason, "operator_stop")

    def test_controller_interrupt_blocks_later_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            controller = BoundedIdleController(store=store)
            controller.start(session_id="session-a", budget=IdleBudget(max_ticks=2))
            controller.interrupt(session_id="session-a")

            tick = controller.tick(
                session_id="session-a",
                self_state=SelfState(active_questions=["idle question"]),
                motive_state=MotiveState(session_id="session-a"),
                initiative_state=default_initiative_state(session_id="session-a"),
                awareness_state=AwarenessState(
                    session_id="session-a",
                    candidate_goal_signals=["idle question"],
                ),
            )

            self.assertIn("lifecycle_not_active:interrupted", tick.stop_reason)
            self.assertFalse(tick.candidate_internal_goals)
            self.assertFalse(store.has_recorded_idle_cognition(session_id="session-a"))


class IdleRuntimePromptEngineTests(unittest.TestCase):
    def test_prompt_reports_absence_of_recorded_idle_cognition_when_asked(self) -> None:
        status = IdleRuntimeStatus(session_id="session-a", lifecycle_state="stopped")

        block = IdleRuntimePromptEngine().build_block(
            status=status,
            recent_ticks=[],
            user_text="What were you thinking about while idle?",
        )

        self.assertIn("[Recorded Idle Runtime]", block)
        self.assertIn("recorded_idle_cognition: False", block)
        self.assertIn("no elapsed idle cognition was recorded", block)

    def test_prompt_surfaces_recent_recorded_tick_without_claiming_action(self) -> None:
        status = IdleRuntimeStatus(
            session_id="session-a",
            lifecycle_state="stopped",
            last_tick_id="tick-1",
            last_stop_reason="budget_exhausted",
            budget=IdleBudget(max_ticks=1, ticks_used=1),
        )
        tick = IdleTickRecord(
            tick_id="tick-1",
            session_id="session-a",
            sequence=1,
            stop_reason="budget_exhausted",
            idle_pressure_appraisal={"idle_state_detected": True},
            candidate_internal_goals=[{"candidate_id": "candidate-1"}],
            selected_internal_goal={"title": "Clarify idle boundary"},
            evidence_refs=["idle_tick:tick-1"],
        )

        block = IdleRuntimePromptEngine().build_block(
            status=status,
            recent_ticks=[tick],
            user_text="What happened while idle?",
        )

        self.assertIn("recorded_idle_cognition: True", block)
        self.assertIn("Clarify idle boundary", block)
        self.assertIn("do not claim desire, hidden work, autonomous action", block)


if __name__ == "__main__":
    unittest.main()
