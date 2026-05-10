"""Tests for the operational autonomy runner (Phase 17 Stage 17.1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.operational_autonomy import (
    JsonOperationalAutonomyStore,
    OperationalAutonomyController,
    default_operational_autonomy_policy,
    operational_runner_state_from_payload,
    step_block_reason,
)
from nova.types import OperationalAutonomyBudget


def _new_controller(tmpdir: str) -> OperationalAutonomyController:
    store = JsonOperationalAutonomyStore(Path(tmpdir) / "operational")
    return OperationalAutonomyController(store=store)


class OperationalAutonomyControllerTests(unittest.TestCase):
    def test_start_initializes_runner_with_running_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            state = ctl.start(
                session_id="s1",
                policy=default_operational_autonomy_policy(),
                budget=OperationalAutonomyBudget(max_ticks=3),
            )

            self.assertEqual(state.status, "running")
            self.assertEqual(state.budget.max_ticks, 3)
            self.assertTrue(state.runner_id)
            self.assertIn("operational_autonomy_started", state.notes)

    def test_pause_then_resume_returns_to_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(session_id="s1")
            paused = ctl.pause(session_id="s1", reason="operator_pause")
            self.assertEqual(paused.status, "paused")
            self.assertEqual(paused.last_stop_reason, "operator_pause")
            resumed = ctl.resume(session_id="s1")
            self.assertEqual(resumed.status, "running")
            self.assertTrue(resumed.resumed_at)

    def test_resume_from_running_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(session_id="s1")
            with self.assertRaises(ValueError):
                ctl.resume(session_id="s1")

    def test_pause_from_stopped_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(session_id="s1")
            ctl.stop(session_id="s1", reason="operator_stop")
            with self.assertRaises(ValueError):
                ctl.pause(session_id="s1")

    def test_interrupt_marks_state_and_blocks_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(session_id="s1")
            state = ctl.interrupt(session_id="s1", reason="operator_test")
            self.assertEqual(state.status, "interrupted")
            self.assertTrue(state.interrupted)
            self.assertEqual(step_block_reason(state), "runner_interrupted")

    def test_emergency_stop_is_final_until_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(session_id="s1")
            state = ctl.emergency_stop(session_id="s1", reason="test_emergency")
            self.assertEqual(state.status, "emergency_stopped")
            self.assertTrue(state.emergency_stopped)
            self.assertEqual(step_block_reason(state), "runner_emergency_stopped")

            # restart clears emergency flag and resets budget if a new one
            # is supplied
            state = ctl.start(
                session_id="s1",
                budget=OperationalAutonomyBudget(max_ticks=2),
            )
            self.assertEqual(state.status, "running")
            self.assertFalse(state.emergency_stopped)
            self.assertEqual(state.budget.max_ticks, 2)

    def test_step_block_reason_for_running_runner_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            state = ctl.start(session_id="s1")
            self.assertEqual(step_block_reason(state), "")

    def test_step_block_reason_for_planned_runner_is_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            state = ctl.status(session_id="s1")
            self.assertEqual(state.status, "planned")
            self.assertEqual(step_block_reason(state), "runner_not_running:planned")

    def test_step_block_reason_when_tick_budget_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(
                session_id="s1",
                budget=OperationalAutonomyBudget(max_ticks=1),
            )
            ctl.append_tick(session_id="s1", status="completed")
            state = ctl.status(session_id="s1")
            self.assertEqual(step_block_reason(state), "tick_budget_exhausted")

    def test_step_block_reason_when_action_budget_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(
                session_id="s1",
                budget=OperationalAutonomyBudget(max_actions=1),
            )
            ctl.append_tick(session_id="s1", action_executed=True)
            state = ctl.status(session_id="s1")
            self.assertEqual(step_block_reason(state), "action_budget_exhausted")

    def test_policy_disabled_blocks_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            policy = default_operational_autonomy_policy()
            policy.enabled = False
            ctl.start(session_id="s1", policy=policy)
            state = ctl.status(session_id="s1")
            self.assertEqual(step_block_reason(state), "policy_disabled")

    def test_policy_self_approval_and_destructive_always_false(self) -> None:
        # Stage 17.1 invariant: even if a policy payload tries to set
        # allow_self_approval or allow_destructive to True, the coercion
        # must force them False until later phases authorize them.
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            state = ctl.start(
                session_id="s1",
                policy={
                    "enabled": True,
                    "allow_self_approval": True,
                    "allow_destructive": True,
                },
            )
            self.assertFalse(state.policy.allow_self_approval)
            self.assertFalse(state.policy.allow_destructive)

    def test_append_tick_advances_tick_count_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(session_id="s1")
            tick = ctl.append_tick(
                session_id="s1",
                status="completed",
                evidence_refs=["e1"],
            )
            state = ctl.status(session_id="s1")
            self.assertEqual(state.tick_count, 1)
            self.assertEqual(state.last_tick_id, tick.tick_id)
            self.assertIn("e1", state.evidence_refs)

    def test_runner_state_round_trips_through_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctl = _new_controller(td)
            ctl.start(
                session_id="s1",
                budget=OperationalAutonomyBudget(max_ticks=2),
            )
            ctl.append_tick(session_id="s1", evidence_refs=["e1"])
            payload = ctl.status(session_id="s1").to_dict()

            rebuilt = operational_runner_state_from_payload(
                payload=payload, session_id="s1"
            )
            self.assertEqual(rebuilt.tick_count, 1)
            self.assertEqual(rebuilt.budget.max_ticks, 2)
            self.assertEqual(rebuilt.ticks[0].evidence_refs, ["e1"])


class OperationalAutonomyRuntimeTests(unittest.TestCase):
    def _build_runtime(self, data_dir: Path, log_dir: Path):
        # Reuse the test runtime from the smoke tests for parity.
        from tests.test_runtime_smoke import build_test_runtime

        return build_test_runtime(data_dir=data_dir, log_dir=log_dir)

    def test_runtime_lifecycle_pause_blocks_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = self._build_runtime(base / "data", base / "logs")
            runtime.start_operational_autonomy(max_ticks=5)
            tick = runtime.step_operational_autonomy()
            self.assertEqual(tick.status, "completed")
            self.assertFalse(tick.action_executed)

            runtime.pause_operational_autonomy(reason="test_pause")
            blocked = runtime.step_operational_autonomy()
            self.assertEqual(blocked.status, "blocked")
            self.assertEqual(blocked.block_reason, "runner_not_running:paused")
            runtime.close()

    def test_runtime_emergency_stop_blocks_step_until_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = self._build_runtime(base / "data", base / "logs")
            runtime.start_operational_autonomy(max_ticks=5)
            runtime.emergency_stop_operational_autonomy(reason="test_emergency")
            blocked = runtime.step_operational_autonomy()
            self.assertEqual(blocked.status, "blocked")
            self.assertEqual(blocked.block_reason, "runner_emergency_stopped")

            runtime.start_operational_autonomy(max_ticks=2)
            unblocked = runtime.step_operational_autonomy()
            self.assertEqual(unblocked.status, "completed")
            runtime.close()

    def test_runtime_tick_budget_blocks_further_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = self._build_runtime(base / "data", base / "logs")
            runtime.start_operational_autonomy(max_ticks=2)
            t1 = runtime.step_operational_autonomy()
            t2 = runtime.step_operational_autonomy()
            blocked = runtime.step_operational_autonomy()
            self.assertEqual(t1.status, "completed")
            self.assertEqual(t2.status, "completed")
            self.assertEqual(blocked.status, "blocked")
            self.assertEqual(blocked.block_reason, "tick_budget_exhausted")
            runtime.close()

    def test_runtime_step_does_not_attempt_action_at_stage_17_1(self) -> None:
        # Stage 17.1 invariant: no real action surface is invoked.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = self._build_runtime(base / "data", base / "logs")
            runtime.start_operational_autonomy(max_ticks=3)
            for _ in range(3):
                tick = runtime.step_operational_autonomy()
                self.assertFalse(tick.action_attempted)
                self.assertFalse(tick.action_executed)
                self.assertFalse(tick.action_blocked)
                self.assertIn("no_real_action_surface_invoked", tick.notes)
            runtime.close()

    def test_runtime_resume_after_pause_allows_steps_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = self._build_runtime(base / "data", base / "logs")
            runtime.start_operational_autonomy(max_ticks=5)
            runtime.step_operational_autonomy()
            runtime.pause_operational_autonomy()
            blocked = runtime.step_operational_autonomy()
            self.assertEqual(blocked.status, "blocked")
            runtime.resume_operational_autonomy()
            unblocked = runtime.step_operational_autonomy()
            self.assertEqual(unblocked.status, "completed")
            runtime.close()

    def test_runtime_writes_operational_tick_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "data"
            log_dir = base / "logs"
            runtime = self._build_runtime(data_dir, log_dir)
            runtime.start_operational_autonomy(max_ticks=1)
            session_id = runtime.session_id
            tick = runtime.step_operational_autonomy()
            runtime.close()

            trace_path = log_dir / "traces" / f"{session_id}.operational.jsonl"
            self.assertTrue(trace_path.exists())
            payload = trace_path.read_text(encoding="utf-8")
            self.assertIn(tick.tick_id, payload)


if __name__ == "__main__":
    unittest.main()
