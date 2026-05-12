"""Tests for the operational autonomy runner (Phase 17 Stages 17.1–17.2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.boundary import BoundaryCheckResult, check_operational_boundary
from nova.agent.operational_autonomy import (
    JsonOperationalAutonomyStore,
    OperationalAutonomyController,
    default_operational_autonomy_policy,
    operational_runner_state_from_payload,
    step_block_reason,
)
from nova.agent.action_plan import default_nova_owned_execution_boundary
from nova.types import NovaOwnedExecutionBoundary, OperationalAutonomyBudget, OperationalAutonomyPolicy


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

    def test_runtime_step_does_not_attempt_action(self) -> None:
        # Stage 17.2 invariant: no real action surface is invoked.
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


def _passing_boundary() -> NovaOwnedExecutionBoundary:
    """Boundary configured to pass in any environment: dedicated_user_required=False."""
    return default_nova_owned_execution_boundary(dedicated_user_required=False)


def _failing_boundary_user_mismatch() -> NovaOwnedExecutionBoundary:
    """Boundary with dedicated user required but wrong active user."""
    return NovaOwnedExecutionBoundary(
        expected_os_user="nova",
        active_os_user="not_nova",
        dedicated_user_required=True,
        dedicated_user_detected=False,
        nova_owned_paths=["/home/nova"],
        allowed_surfaces=["nova_scratchpad", "nova_logs", "nova_workspace", "internal_tool"],
        blocked_surfaces=["filesystem", "shell", "network", "gui", "system_config", "external_service"],
    )


def _failing_boundary_blocked_surface() -> NovaOwnedExecutionBoundary:
    """Boundary where the default policy's surfaces overlap with blocked_surfaces."""
    return NovaOwnedExecutionBoundary(
        expected_os_user="nova",
        active_os_user="nova",
        dedicated_user_required=False,
        dedicated_user_detected=True,
        nova_owned_paths=["/home/nova"],
        allowed_surfaces=["nova_scratchpad"],
        blocked_surfaces=["internal_state", "self_prompt", "motive_appraisal"],
    )


class BoundaryCheckerTests(unittest.TestCase):
    def test_passing_boundary_with_valid_policy_is_satisfied(self) -> None:
        boundary = _passing_boundary()
        policy = default_operational_autonomy_policy()
        result = check_operational_boundary(boundary, policy)
        self.assertTrue(result.satisfied)
        self.assertEqual(result.violations, [])
        self.assertIn("satisfied", result.snapshot)
        self.assertTrue(result.snapshot["satisfied"])

    def test_none_boundary_is_not_satisfied(self) -> None:
        result = check_operational_boundary(None, default_operational_autonomy_policy())
        self.assertFalse(result.satisfied)
        self.assertIn("invalid_boundary_object", result.violations)
        self.assertEqual(result.snapshot, {})

    def test_none_policy_is_not_satisfied(self) -> None:
        result = check_operational_boundary(_passing_boundary(), None)
        self.assertFalse(result.satisfied)
        self.assertIn("invalid_policy_object", result.violations)
        self.assertIn("satisfied", result.snapshot)

    def test_dedicated_user_required_and_not_detected_is_not_satisfied(self) -> None:
        boundary = _failing_boundary_user_mismatch()
        policy = default_operational_autonomy_policy()
        result = check_operational_boundary(boundary, policy)
        self.assertFalse(result.satisfied)
        violation = result.violations[0]
        self.assertTrue(violation.startswith("dedicated_user_not_detected:"))
        self.assertIn("nova", violation)
        self.assertIn("not_nova", violation)

    def test_dedicated_user_required_and_detected_is_satisfied(self) -> None:
        boundary = NovaOwnedExecutionBoundary(
            expected_os_user="nova",
            active_os_user="nova",
            dedicated_user_required=True,
            dedicated_user_detected=True,
            nova_owned_paths=["/home/nova"],
            allowed_surfaces=["nova_scratchpad", "nova_logs", "internal_tool"],
            blocked_surfaces=["filesystem", "shell", "network"],
        )
        policy = default_operational_autonomy_policy()
        result = check_operational_boundary(boundary, policy)
        self.assertTrue(result.satisfied)

    def test_policy_surface_in_blocked_surfaces_is_not_satisfied(self) -> None:
        boundary = _failing_boundary_blocked_surface()
        policy = default_operational_autonomy_policy()
        result = check_operational_boundary(boundary, policy)
        self.assertFalse(result.satisfied)
        blocked_violations = [v for v in result.violations if v.startswith("surface_in_boundary_blocklist:")]
        self.assertTrue(len(blocked_violations) > 0)

    def test_empty_policy_allowed_surfaces_is_not_satisfied(self) -> None:
        boundary = _passing_boundary()
        policy = OperationalAutonomyPolicy(
            policy_id="test",
            enabled=True,
            allowed_surfaces=[],
            blocked_surfaces=["filesystem", "shell"],
        )
        result = check_operational_boundary(boundary, policy)
        self.assertFalse(result.satisfied)
        self.assertIn("policy_has_no_allowed_surfaces", result.violations)

    def test_snapshot_always_contains_satisfied_and_violations(self) -> None:
        for boundary_fn in (_passing_boundary, _failing_boundary_user_mismatch):
            boundary = boundary_fn()
            result = check_operational_boundary(boundary, default_operational_autonomy_policy())
            self.assertIn("satisfied", result.snapshot)
            self.assertIn("violations", result.snapshot)
            self.assertIsInstance(result.snapshot["violations"], list)


class BoundaryRuntimeIntegrationTests(unittest.TestCase):
    def _build_runtime(self, data_dir: Path, log_dir: Path, boundary=None):
        from tests.test_runtime_smoke import build_test_runtime
        from nova.types import NovaOwnedExecutionBoundary

        rt = build_test_runtime(data_dir=data_dir, log_dir=log_dir)
        if boundary is not None:
            rt.operational_boundary = boundary
        return rt

    def test_passing_boundary_produces_completed_tick_with_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            rt = self._build_runtime(base / "data", base / "logs")
            rt.start_operational_autonomy(max_ticks=2)
            tick = rt.step_operational_autonomy()
            self.assertEqual(tick.status, "completed")
            self.assertFalse(tick.action_executed)
            self.assertTrue(tick.boundary_snapshot)
            self.assertTrue(tick.boundary_snapshot.get("satisfied"))
            self.assertIn("stage17_2_boundary_satisfied", tick.notes)
            rt.close()

    def test_failing_boundary_user_mismatch_blocks_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            rt = self._build_runtime(
                base / "data", base / "logs",
                boundary=_failing_boundary_user_mismatch(),
            )
            rt.start_operational_autonomy(max_ticks=5)
            tick = rt.step_operational_autonomy()
            self.assertEqual(tick.status, "blocked")
            self.assertEqual(tick.block_reason, "local_execution_boundary_failed")
            self.assertTrue(tick.action_blocked)
            self.assertFalse(tick.action_executed)
            self.assertFalse(tick.boundary_snapshot.get("satisfied"))
            violations = tick.boundary_snapshot.get("violations", [])
            self.assertTrue(any("dedicated_user_not_detected" in v for v in violations))
            rt.close()

    def test_lifecycle_block_takes_priority_over_boundary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            rt = self._build_runtime(
                base / "data", base / "logs",
                boundary=_failing_boundary_user_mismatch(),
            )
            rt.start_operational_autonomy(max_ticks=5)
            rt.pause_operational_autonomy(reason="test_priority")
            tick = rt.step_operational_autonomy()
            self.assertEqual(tick.status, "blocked")
            self.assertEqual(tick.block_reason, "runner_not_running:paused")
            rt.close()

    def test_boundary_snapshot_recorded_on_lifecycle_blocked_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            rt = self._build_runtime(base / "data", base / "logs")
            rt.start_operational_autonomy(max_ticks=5)
            rt.pause_operational_autonomy()
            tick = rt.step_operational_autonomy()
            self.assertEqual(tick.status, "blocked")
            self.assertIn("satisfied", tick.boundary_snapshot)
            rt.close()


if __name__ == "__main__":
    unittest.main()
