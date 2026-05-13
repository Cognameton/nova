"""Tests for action surface adapters (Phase 17 Stage 17.3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nova.agent.action_plan import (
    BoundedActionPlanEngine,
    NOVA_OWNED_ENVIRONMENT_SURFACES,
    default_nova_owned_execution_boundary,
)
from nova.agent.action_surface import (
    AdapterActionRequest,
    AdapterRegistry,
    OperationalLogAdapter,
    ScratchpadAdapter,
    adapter_audit_from_results,
    check_action_plan_for_adapter,
    execute_plan_through_adapters,
    _path_is_confined,
)
from nova.agent.operational_autonomy import (
    default_nova_owned_operational_policy,
    default_operational_autonomy_policy,
)
from nova.types import NovaOwnedExecutionBoundary


def _passing_boundary(nova_owned_paths: list[str]) -> NovaOwnedExecutionBoundary:
    return NovaOwnedExecutionBoundary(
        expected_os_user="nova",
        active_os_user="nova",
        dedicated_user_required=False,
        dedicated_user_detected=True,
        nova_owned_paths=nova_owned_paths,
        allowed_surfaces=sorted(NOVA_OWNED_ENVIRONMENT_SURFACES),
        blocked_surfaces=["filesystem", "shell", "network", "gui", "system_config", "external_service"],
    )


def _make_approved_scratchpad_plan(session_id: str, data_dir: Path):
    engine = BoundedActionPlanEngine(
        boundary=default_nova_owned_execution_boundary(
            nova_owned_paths=[str(data_dir)],
            dedicated_user_required=False,
        )
    )
    return engine.create_plan(
        session_id=session_id,
        purpose="write a test note",
        scope="scratchpad",
        execution_lane="nova_owned_environment",
        risk_class="nova_owned",
        steps=[{"description": "Test scratchpad write", "surface": "nova_scratchpad"}],
        approved=True,
        approved_by="operator",
    )


def _make_approved_log_plan(session_id: str, data_dir: Path):
    engine = BoundedActionPlanEngine(
        boundary=default_nova_owned_execution_boundary(
            nova_owned_paths=[str(data_dir)],
            dedicated_user_required=False,
        )
    )
    return engine.create_plan(
        session_id=session_id,
        purpose="write a test log entry",
        scope="nova_logs",
        execution_lane="nova_owned_environment",
        risk_class="nova_owned",
        steps=[{"description": "Test log write", "surface": "nova_logs"}],
        approved=True,
        approved_by="operator",
    )


class PathConfinementTests(unittest.TestCase):
    def test_path_under_owned_path_is_confined(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            self.assertTrue(_path_is_confined(base / "sub" / "file.json", [str(base)]))

    def test_path_outside_owned_path_is_not_confined(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(_path_is_confined(Path("/tmp/other/file.json"), [td]))

    def test_empty_nova_owned_paths_is_not_confined(self) -> None:
        self.assertFalse(_path_is_confined(Path("/tmp/file.json"), []))

    def test_path_confined_to_second_owned_path(self) -> None:
        with tempfile.TemporaryDirectory() as td1:
            with tempfile.TemporaryDirectory() as td2:
                self.assertTrue(
                    _path_is_confined(Path(td2) / "file.json", [td1, td2])
                )


class ScratchpadAdapterTests(unittest.TestCase):
    def test_write_creates_file_under_nova_owned_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = ScratchpadAdapter(base_dir=base / "scratchpad")
            boundary = _passing_boundary([str(base)])
            request = AdapterActionRequest(
                surface="nova_scratchpad",
                content={"description": "test note"},
                session_id="sess1",
                plan_id="plan1",
                step_id="step1",
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertTrue(result.executed)
            self.assertFalse(result.blocked)
            self.assertTrue(result.artifact_path)
            self.assertTrue(Path(result.artifact_path).exists())

    def test_write_blocked_when_path_outside_nova_owned(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = ScratchpadAdapter(base_dir=base / "scratchpad")
            # boundary owned path is /tmp/nonexistent, not base
            boundary = _passing_boundary(["/tmp/nonexistent_nova_owned_xyz"])
            request = AdapterActionRequest(
                surface="nova_scratchpad",
                content={"description": "test"},
                session_id="sess1",
                plan_id="plan1",
                step_id="step1",
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertFalse(result.executed)
            self.assertTrue(result.blocked)
            self.assertIn("path_outside_nova_owned", result.block_reason)
            self.assertFalse(Path(base / "scratchpad" / "sess1").exists())

    def test_entry_json_has_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = ScratchpadAdapter(base_dir=base / "scratchpad")
            boundary = _passing_boundary([str(base)])
            request = AdapterActionRequest(
                surface="nova_scratchpad",
                content={"description": "hello"},
                session_id="sess1",
                plan_id="planX",
                step_id="stepY",
                notes=["test_note"],
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertTrue(result.executed)
            payload = json.loads(Path(result.artifact_path).read_text())
            self.assertEqual(payload["session_id"], "sess1")
            self.assertEqual(payload["plan_id"], "planX")
            self.assertEqual(payload["step_id"], "stepY")
            self.assertEqual(payload["surface"], "nova_scratchpad")
            self.assertEqual(payload["content"], {"description": "hello"})
            self.assertIn("test_note", payload["notes"])
            self.assertIn("entry_id", payload)
            self.assertIn("written_at", payload)

    def test_each_write_creates_unique_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = ScratchpadAdapter(base_dir=base / "scratchpad")
            boundary = _passing_boundary([str(base)])
            request = AdapterActionRequest(
                surface="nova_scratchpad",
                content={"note": "x"},
                session_id="sess1",
                plan_id="p",
                step_id="s",
            )
            r1 = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            r2 = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertTrue(r1.executed)
            self.assertTrue(r2.executed)
            self.assertNotEqual(r1.artifact_path, r2.artifact_path)

    def test_evidence_refs_contain_entry_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = ScratchpadAdapter(base_dir=base / "scratchpad")
            boundary = _passing_boundary([str(base)])
            request = AdapterActionRequest(
                surface="nova_scratchpad",
                content={},
                session_id="sess1",
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertTrue(any(result.artifact_id in ref for ref in result.evidence_refs))


class OperationalLogAdapterTests(unittest.TestCase):
    def test_write_creates_file_under_nova_owned_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = OperationalLogAdapter(base_dir=base / "operational_logs")
            boundary = _passing_boundary([str(base)])
            request = AdapterActionRequest(
                surface="nova_logs",
                content={"event": "test log entry"},
                session_id="sess1",
                plan_id="plan1",
                step_id="step1",
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertTrue(result.executed)
            self.assertFalse(result.blocked)
            self.assertTrue(Path(result.artifact_path).exists())

    def test_write_blocked_when_path_outside_nova_owned(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = OperationalLogAdapter(base_dir=base / "operational_logs")
            boundary = _passing_boundary(["/tmp/nonexistent_nova_owned_xyz"])
            request = AdapterActionRequest(
                surface="nova_logs",
                content={"event": "test"},
                session_id="sess1",
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess1")
            self.assertFalse(result.executed)
            self.assertTrue(result.blocked)
            self.assertIn("path_outside_nova_owned", result.block_reason)

    def test_entry_json_has_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = OperationalLogAdapter(base_dir=base / "operational_logs")
            boundary = _passing_boundary([str(base)])
            request = AdapterActionRequest(
                surface="nova_logs",
                content={"event": "something happened"},
                session_id="sess2",
                plan_id="plan2",
                step_id="step2",
            )
            result = adapter.execute(request=request, boundary=boundary, session_id="sess2")
            self.assertTrue(result.executed)
            payload = json.loads(Path(result.artifact_path).read_text())
            self.assertEqual(payload["session_id"], "sess2")
            self.assertEqual(payload["surface"], "nova_logs")
            self.assertEqual(payload["content"], {"event": "something happened"})


class AdapterRegistryTests(unittest.TestCase):
    def test_registered_adapter_returned_by_surface_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            adapter = ScratchpadAdapter(base_dir=base / "scratchpad")
            registry = AdapterRegistry([adapter])
            self.assertIs(registry.get("nova_scratchpad"), adapter)

    def test_unknown_surface_returns_none(self) -> None:
        registry = AdapterRegistry()
        self.assertIsNone(registry.get("filesystem"))
        self.assertIsNone(registry.get("shell"))
        self.assertIsNone(registry.get("nova_scratchpad"))

    def test_surfaces_returns_sorted_registered_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            registry = AdapterRegistry([
                OperationalLogAdapter(base_dir=base / "logs"),
                ScratchpadAdapter(base_dir=base / "scratch"),
            ])
            self.assertEqual(registry.surfaces(), ["nova_logs", "nova_scratchpad"])

    def test_register_replaces_existing_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            a1 = ScratchpadAdapter(base_dir=base / "s1")
            a2 = ScratchpadAdapter(base_dir=base / "s2")
            registry = AdapterRegistry([a1])
            registry.register(a2)
            self.assertIs(registry.get("nova_scratchpad"), a2)


class CheckActionPlanForAdapterTests(unittest.TestCase):
    def _make_registry(self, tmpdir: Path) -> AdapterRegistry:
        return AdapterRegistry([
            ScratchpadAdapter(base_dir=tmpdir / "scratchpad"),
            OperationalLogAdapter(base_dir=tmpdir / "operational_logs"),
        ])

    def test_approved_nova_owned_scratchpad_plan_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            policy = default_nova_owned_operational_policy()
            boundary = _passing_boundary([str(base)])
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertEqual(result, "")

    def test_unapproved_draft_plan_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            engine = BoundedActionPlanEngine(
                boundary=default_nova_owned_execution_boundary(
                    nova_owned_paths=[str(base)], dedicated_user_required=False
                )
            )
            plan = engine.create_plan(
                session_id="sess1",
                purpose="test",
                scope="scratchpad",
                execution_lane="nova_owned_environment",
                risk_class="nova_owned",
                steps=[{"description": "write", "surface": "nova_scratchpad"}],
                approved=False,
            )
            policy = default_nova_owned_operational_policy()
            boundary = _passing_boundary([str(base)])
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertTrue(result.startswith("plan_not_approved:"))

    def test_internal_activity_lane_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            engine = BoundedActionPlanEngine(
                boundary=default_nova_owned_execution_boundary(
                    nova_owned_paths=[str(base)], dedicated_user_required=False
                )
            )
            plan = engine.create_plan(
                session_id="sess1",
                purpose="test",
                scope="internal",
                execution_lane="internal_activity",
                risk_class="internal",
                steps=[{"description": "think", "surface": "internal_state"}],
                approved=True,
                approved_by="operator",
            )
            policy = default_nova_owned_operational_policy()
            boundary = _passing_boundary([str(base)])
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertIn("plan_lane_not_nova_owned", result)

    def test_external_effect_lane_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # external_system_effect plans are always pending_approval by the engine
            # and can never pass check_action_plan_for_adapter
            engine = BoundedActionPlanEngine(
                boundary=default_nova_owned_execution_boundary(
                    nova_owned_paths=[str(base)], dedicated_user_required=False
                )
            )
            # Force an external plan by manually building a plan dict
            from nova.agent.action_plan import action_plan_from_payload
            plan = action_plan_from_payload(
                payload={
                    "action_plan_id": "test-plan",
                    "session_id": "sess1",
                    "execution_lane": "external_system_effect",
                    "risk_class": "external",
                    "status": "approved",
                    "allowed_surfaces": ["filesystem"],
                    "steps": [{"description": "write", "surface": "filesystem"}],
                    "permission": {"approved": True, "approved_by": "operator"},
                },
                session_id="sess1",
            )
            policy = default_nova_owned_operational_policy()
            boundary = _passing_boundary([str(base)])
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertIn("plan_lane_not_nova_owned", result)

    def test_non_nova_owned_surface_in_plan_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            from nova.agent.action_plan import action_plan_from_payload
            plan = action_plan_from_payload(
                payload={
                    "action_plan_id": "test-plan",
                    "session_id": "sess1",
                    "execution_lane": "nova_owned_environment",
                    "risk_class": "nova_owned",
                    "status": "approved",
                    "allowed_surfaces": ["filesystem"],
                    "steps": [{"description": "write", "surface": "filesystem"}],
                    "permission": {"approved": True, "approved_by": "operator"},
                },
                session_id="sess1",
            )
            policy = default_nova_owned_operational_policy()
            boundary = _passing_boundary([str(base)])
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertIn("surface_not_nova_owned_environment", result)

    def test_surface_not_in_policy_allowed_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            # Policy that explicitly does NOT include nova_scratchpad
            policy = default_operational_autonomy_policy()
            boundary = _passing_boundary([str(base)])
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertIn("surface_not_in_policy_allowed", result)

    def test_surface_not_in_boundary_allowed_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            policy = default_nova_owned_operational_policy()
            # Boundary that does not allow nova_scratchpad
            boundary = NovaOwnedExecutionBoundary(
                expected_os_user="nova",
                active_os_user="nova",
                dedicated_user_required=False,
                dedicated_user_detected=True,
                nova_owned_paths=[str(base)],
                allowed_surfaces=["nova_logs", "nova_workspace", "internal_tool"],
                blocked_surfaces=["filesystem", "shell"],
            )
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertIn("surface_not_in_boundary_allowed", result)

    def test_surface_in_boundary_blocked_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            policy = default_nova_owned_operational_policy()
            # Boundary that explicitly blocks nova_scratchpad
            boundary = NovaOwnedExecutionBoundary(
                expected_os_user="nova",
                active_os_user="nova",
                dedicated_user_required=False,
                dedicated_user_detected=True,
                nova_owned_paths=[str(base)],
                allowed_surfaces=sorted(NOVA_OWNED_ENVIRONMENT_SURFACES),
                blocked_surfaces=["nova_scratchpad", "filesystem", "shell"],
            )
            registry = self._make_registry(base)
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=registry
            )
            self.assertIn("surface_in_boundary_blocked", result)

    def test_no_adapter_for_surface_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            policy = default_nova_owned_operational_policy()
            boundary = _passing_boundary([str(base)])
            # Empty registry — no adapters registered
            result = check_action_plan_for_adapter(
                plan=plan, policy=policy, boundary=boundary, registry=AdapterRegistry()
            )
            self.assertIn("no_adapter_for_surface", result)


class ExecutePlanThroughAdaptersTests(unittest.TestCase):
    def test_approved_plan_executes_and_returns_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            registry = AdapterRegistry([ScratchpadAdapter(base_dir=base / "scratchpad")])
            boundary = _passing_boundary([str(base)])
            results = execute_plan_through_adapters(
                plan=plan,
                registry=registry,
                boundary=boundary,
                session_id="sess1",
                actions_budget_remaining=5,
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].executed)
            self.assertFalse(results[0].blocked)
            self.assertTrue(Path(results[0].artifact_path).exists())

    def test_zero_budget_blocks_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            registry = AdapterRegistry([ScratchpadAdapter(base_dir=base / "scratchpad")])
            boundary = _passing_boundary([str(base)])
            results = execute_plan_through_adapters(
                plan=plan,
                registry=registry,
                boundary=boundary,
                session_id="sess1",
                actions_budget_remaining=0,
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].blocked)
            self.assertEqual(results[0].block_reason, "action_budget_exhausted_mid_plan")

    def test_no_adapter_produces_blocked_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            plan = _make_approved_scratchpad_plan("sess1", base)
            # No scratchpad adapter in registry
            registry = AdapterRegistry([OperationalLogAdapter(base_dir=base / "logs")])
            boundary = _passing_boundary([str(base)])
            results = execute_plan_through_adapters(
                plan=plan,
                registry=registry,
                boundary=boundary,
                session_id="sess1",
                actions_budget_remaining=5,
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].blocked)
            self.assertIn("no_adapter_for_surface", results[0].block_reason)


class AdapterAuditFromResultsTests(unittest.TestCase):
    def test_audit_counts_executed_and_blocked(self) -> None:
        from nova.agent.action_surface import AdapterActionResult
        results = [
            AdapterActionResult("nova_scratchpad", "ScratchpadAdapter", True, False, ""),
            AdapterActionResult("nova_scratchpad", "ScratchpadAdapter", True, False, ""),
            AdapterActionResult("nova_scratchpad", "none", False, True, "budget_exhausted"),
        ]
        audit = adapter_audit_from_results(results)
        self.assertEqual(audit["steps_attempted"], 3)
        self.assertEqual(audit["steps_executed"], 2)
        self.assertEqual(audit["steps_blocked"], 1)
        self.assertEqual(len(audit["results"]), 3)

    def test_empty_results_produces_zero_counts(self) -> None:
        audit = adapter_audit_from_results([])
        self.assertEqual(audit["steps_attempted"], 0)
        self.assertEqual(audit["steps_executed"], 0)
        self.assertEqual(audit["steps_blocked"], 0)


if __name__ == "__main__":
    unittest.main()
