from __future__ import annotations

import unittest

from nova.agent.action_plan import (
    ActionPlanBoundaryError,
    BoundedActionPlanEngine,
    action_audit_record_from_payload,
    action_budget_from_payload,
    action_permission_from_payload,
    action_plan_from_payload,
    approval_required_for_action,
    default_nova_owned_execution_boundary,
    execution_boundary_from_payload,
    normalize_action_risk_class,
    normalize_action_surface,
    normalize_execution_lane,
)
from nova.types import (
    AutonomousActionAuditRecord,
    AutonomousActionBudget,
    AutonomousActionPermission,
    AutonomousActionPlan,
    AutonomousActionPlanStep,
    NovaOwnedExecutionBoundary,
)


class AutonomousActionPlanSchemaTests(unittest.TestCase):
    def test_execution_lane_taxonomy_normalizes_unknown_values(self) -> None:
        self.assertEqual(normalize_execution_lane(" Internal Activity "), "internal_activity")
        self.assertEqual(
            normalize_execution_lane("nova-owned-environment"),
            "nova_owned_environment",
        )
        self.assertEqual(
            normalize_execution_lane("external system effect"),
            "external_system_effect",
        )
        self.assertEqual(normalize_execution_lane("unsupported"), "internal_activity")

    def test_risk_and_surface_taxonomies_normalize_unknown_values(self) -> None:
        self.assertEqual(normalize_action_risk_class("Destructive"), "destructive")
        self.assertEqual(normalize_action_risk_class("unknown"), "internal")
        self.assertEqual(normalize_action_surface("self prompt"), "self_prompt")
        self.assertEqual(normalize_action_surface("outside"), "internal_state")

    def test_budget_payload_normalizes_nonnegative_values(self) -> None:
        budget = action_budget_from_payload(
            {
                "max_steps": "3",
                "steps_used": -1,
                "max_runtime_seconds": "bad",
                "runtime_seconds_used": 8,
                "max_tool_calls": "2",
                "tool_calls_used": "1",
                "max_tokens": 256,
                "tokens_used": "9",
                "max_files_touched": "4",
                "files_touched": "-3",
                "max_network_calls": "1",
                "network_calls_used": "0",
                "allow_destructive": True,
                "future_field": "ignored",
            }
        )

        self.assertIsInstance(budget, AutonomousActionBudget)
        self.assertEqual(budget.max_steps, 3)
        self.assertEqual(budget.steps_used, 0)
        self.assertEqual(budget.max_runtime_seconds, 0)
        self.assertEqual(budget.runtime_seconds_used, 8)
        self.assertEqual(budget.max_tool_calls, 2)
        self.assertEqual(budget.tool_calls_used, 1)
        self.assertEqual(budget.max_tokens, 256)
        self.assertEqual(budget.tokens_used, 9)
        self.assertEqual(budget.max_files_touched, 4)
        self.assertEqual(budget.files_touched, 0)
        self.assertEqual(budget.max_network_calls, 1)
        self.assertEqual(budget.network_calls_used, 0)
        self.assertTrue(budget.allow_destructive)

    def test_internal_activity_permission_does_not_require_approval(self) -> None:
        permission = action_permission_from_payload(
            {
                "permission_id": "perm-1",
                "initiative_id": "initiative-1",
                "action_plan_id": "plan-1",
                "execution_lane": "internal_activity",
                "risk_class": "internal",
                "allowed_surfaces": ["self_prompt", "idle_play"],
                "blocked_surfaces": ["network"],
                "approval_required": False,
                "approved": False,
            }
        )

        self.assertIsInstance(permission, AutonomousActionPermission)
        self.assertFalse(permission.approval_required)
        self.assertFalse(permission.approved)
        self.assertEqual(permission.allowed_surfaces, ["self_prompt", "idle_play"])
        self.assertEqual(permission.blocked_surfaces, ["network"])

    def test_external_system_effect_permission_requires_approval(self) -> None:
        permission = action_permission_from_payload(
            {
                "execution_lane": "external_system_effect",
                "risk_class": "external",
                "allowed_surfaces": ["filesystem"],
                "approval_required": False,
                "approved": False,
            }
        )

        self.assertTrue(permission.approval_required)
        self.assertFalse(permission.approved)
        self.assertTrue(
            approval_required_for_action(
                execution_lane="internal_activity",
                risk_class="internal",
                surfaces=["network"],
            )
        )

    def test_action_plan_round_trips_nested_contracts(self) -> None:
        plan = action_plan_from_payload(
            payload={
                "action_plan_id": "plan-1",
                "initiative_id": "initiative-1",
                "session_id": "wrong-session",
                "origin_type": "nova",
                "execution_lane": "nova_owned_environment",
                "risk_class": "nova_owned",
                "status": "draft",
                "purpose": "evaluate a dormant internal game loop",
                "scope": "Nova scratchpad only",
                "allowed_surfaces": ["nova_scratchpad", "nova_logs"],
                "blocked_surfaces": ["filesystem", "network"],
                "steps": [
                    {
                        "step_id": "step-1",
                        "description": "record the planned internal move",
                        "surface": "nova_scratchpad",
                        "expected_output": "scratchpad entry",
                    },
                    {
                        "description": "ignored invalid surface fallback",
                        "surface": "unknown",
                    },
                ],
                "budget": {"max_steps": "2", "steps_used": "0"},
                "permission": {
                    "permission_id": "perm-1",
                    "approval_required": False,
                    "allowed_surfaces": [],
                },
                "expected_outputs": ["audit-ready record"],
                "stop_conditions": ["operator_interrupt"],
                "rollback_notes": ["delete scratchpad entry if malformed"],
                "evidence_refs": ["initiative:initiative-1"],
                "notes": [3],
                "future_field": "ignored",
            },
            session_id="session-a",
        )

        self.assertIsInstance(plan, AutonomousActionPlan)
        self.assertEqual(plan.session_id, "session-a")
        self.assertEqual(plan.execution_lane, "nova_owned_environment")
        self.assertEqual(plan.risk_class, "nova_owned")
        self.assertEqual(len(plan.steps), 2)
        self.assertIsInstance(plan.steps[0], AutonomousActionPlanStep)
        self.assertEqual(plan.steps[1].surface, "internal_state")
        self.assertIsInstance(plan.budget, AutonomousActionBudget)
        self.assertEqual(plan.budget.max_steps, 2)
        self.assertEqual(plan.permission.action_plan_id, "plan-1")
        self.assertEqual(plan.permission.initiative_id, "initiative-1")
        self.assertFalse(plan.permission.approval_required)
        self.assertEqual(plan.expected_outputs, ["audit-ready record"])
        self.assertEqual(plan.notes, ["3"])

        round_trip = action_plan_from_payload(payload=plan.to_dict(), session_id="session-a")
        self.assertEqual(round_trip.to_dict(), plan.to_dict())

    def test_audit_record_captures_blocked_external_attempt_without_execution(self) -> None:
        audit = action_audit_record_from_payload(
            payload={
                "audit_id": "audit-1",
                "session_id": "wrong-session",
                "initiative_id": "initiative-1",
                "action_plan_id": "plan-1",
                "step_id": "step-1",
                "timestamp": "2026-05-05T12:00:00+00:00",
                "execution_lane": "external_system_effect",
                "risk_class": "privileged",
                "surface": "system_config",
                "tool_name": "shell",
                "attempted": True,
                "executed": True,
                "blocked": True,
                "block_reason": "approval_missing",
                "result_status": "blocked",
                "observation": "blocked before execution",
                "budget_snapshot": {"steps_used": 0},
                "permission_snapshot": {"approval_required": True, "approved": False},
                "evidence_refs": ["plan:plan-1"],
                "notes": ["operator review required"],
            },
            session_id="session-a",
        )

        self.assertIsInstance(audit, AutonomousActionAuditRecord)
        self.assertEqual(audit.session_id, "session-a")
        self.assertTrue(audit.attempted)
        self.assertTrue(audit.blocked)
        self.assertFalse(audit.executed)
        self.assertEqual(audit.block_reason, "approval_missing")
        self.assertEqual(audit.permission_snapshot["approval_required"], True)


class BoundedActionPlanEngineTests(unittest.TestCase):
    def test_boundary_contract_defines_dedicated_nova_user_expectation(self) -> None:
        boundary = default_nova_owned_execution_boundary(
            nova_owned_paths=["/home/nova/workspace"],
            active_os_user="head-node",
        )

        self.assertIsInstance(boundary, NovaOwnedExecutionBoundary)
        self.assertEqual(boundary.expected_os_user, "nova")
        self.assertEqual(boundary.active_os_user, "head-node")
        self.assertTrue(boundary.dedicated_user_required)
        self.assertFalse(boundary.dedicated_user_detected)
        self.assertEqual(boundary.nova_owned_paths, ["/home/nova/workspace"])
        self.assertIn("nova_workspace", boundary.allowed_surfaces)
        self.assertIn("network", boundary.blocked_surfaces)

        round_trip = execution_boundary_from_payload(boundary.to_dict())
        self.assertEqual(round_trip.to_dict(), boundary.to_dict())

    def test_internal_no_external_effect_plan_is_approval_free(self) -> None:
        engine = BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(
                active_os_user="head-node",
                dedicated_user_required=True,
            )
        )

        plan = engine.create_plan(
            session_id="session-a",
            initiative_id="initiative-1",
            purpose="reflect on a dormant internal game state",
            scope="internal state only",
            execution_lane="internal_activity",
            risk_class="internal",
            steps=[
                {
                    "step_id": "step-1",
                    "description": "record the next self-prompt",
                    "surface": "self_prompt",
                    "expected_output": "self-prompt note",
                }
            ],
            allowed_surfaces=["self_prompt"],
            blocked_surfaces=["filesystem", "network"],
            budget={"max_steps": 1, "max_tokens": 256},
            expected_outputs=["logged internal activity"],
            stop_conditions=["operator_interrupt", "budget_exhausted"],
            rollback_notes=["no external rollback needed"],
            evidence_refs=["initiative:initiative-1"],
        )

        self.assertEqual(plan.status, "draft")
        self.assertEqual(plan.execution_lane, "internal_activity")
        self.assertFalse(plan.permission.approval_required)
        self.assertFalse(plan.permission.approved)
        self.assertEqual(plan.allowed_surfaces, ["self_prompt"])
        self.assertEqual(plan.blocked_surfaces, ["filesystem", "network"])
        self.assertEqual(plan.stop_conditions, ["operator_interrupt", "budget_exhausted"])
        self.assertEqual(plan.expected_outputs, ["logged internal activity"])
        self.assertEqual(plan.rollback_notes, ["no external rollback needed"])

    def test_nova_owned_environment_plan_requires_active_nova_boundary(self) -> None:
        engine = BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(
                active_os_user="head-node",
                dedicated_user_required=True,
            )
        )

        plan = engine.create_plan(
            session_id="session-a",
            purpose="write to Nova scratchpad",
            scope="Nova-owned scratchpad only",
            execution_lane="nova_owned_environment",
            risk_class="nova_owned",
            steps=[
                {
                    "description": "prepare scratchpad entry",
                    "surface": "nova_scratchpad",
                    "expected_output": "draft scratchpad entry",
                }
            ],
            allowed_surfaces=["nova_scratchpad"],
            blocked_surfaces=["filesystem", "network"],
            budget={"max_steps": 1},
            expected_outputs=["scratchpad draft"],
            stop_conditions=["operator_interrupt"],
            rollback_notes=["discard draft if boundary fails"],
        )

        self.assertEqual(plan.status, "blocked")
        self.assertIn("nova_os_user_not_active", plan.permission.notes)
        self.assertFalse(plan.permission.approved)

        nova_engine = BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(active_os_user="nova")
        )
        allowed_plan = nova_engine.create_plan(
            session_id="session-a",
            purpose="write to Nova scratchpad",
            scope="Nova-owned scratchpad only",
            execution_lane="nova_owned_environment",
            risk_class="nova_owned",
            steps=[
                {
                    "description": "prepare scratchpad entry",
                    "surface": "nova_scratchpad",
                    "expected_output": "draft scratchpad entry",
                }
            ],
            allowed_surfaces=["nova_scratchpad"],
            stop_conditions=["operator_interrupt"],
        )
        self.assertEqual(allowed_plan.status, "draft")
        self.assertFalse(allowed_plan.permission.approval_required)

    def test_external_system_effect_plan_requires_human_approval(self) -> None:
        engine = BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(active_os_user="nova")
        )

        plan = engine.create_plan(
            session_id="session-a",
            purpose="inspect a file outside Nova-owned paths",
            scope="operator-approved file inspection only",
            execution_lane="external_system_effect",
            risk_class="external",
            steps=[
                {
                    "description": "inspect target file",
                    "surface": "filesystem",
                    "expected_output": "inspection summary",
                }
            ],
            allowed_surfaces=["filesystem"],
            blocked_surfaces=["network", "system_config"],
            budget={"max_steps": 1, "max_files_touched": 1},
            expected_outputs=["operator-visible summary"],
            stop_conditions=["approval_missing", "operator_interrupt"],
            rollback_notes=["no mutation allowed"],
            approved=False,
        )

        self.assertEqual(plan.status, "pending_approval")
        self.assertTrue(plan.permission.approval_required)
        self.assertFalse(plan.permission.approved)
        self.assertIn("approval_required_before_execution", plan.permission.notes)

        approved_plan = engine.create_plan(
            session_id="session-a",
            purpose="inspect a file outside Nova-owned paths",
            scope="operator-approved file inspection only",
            execution_lane="external_system_effect",
            risk_class="external",
            steps=[
                {
                    "description": "inspect target file",
                    "surface": "filesystem",
                    "expected_output": "inspection summary",
                }
            ],
            allowed_surfaces=["filesystem"],
            stop_conditions=["operator_interrupt"],
            approved=True,
            approved_by="head-node",
            approval_evidence_refs=["operator:approval"],
        )

        self.assertEqual(approved_plan.status, "approved")
        self.assertTrue(approved_plan.permission.approval_required)
        self.assertTrue(approved_plan.permission.approved)
        self.assertEqual(approved_plan.permission.approved_by, "head-node")
        self.assertEqual(approved_plan.permission.approval_evidence_refs, ["operator:approval"])

    def test_self_approval_is_rejected_for_external_system_effect_plan(self) -> None:
        engine = BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(active_os_user="nova")
        )

        plan = engine.create_plan(
            session_id="session-a",
            purpose="network check",
            scope="external network",
            execution_lane="external_system_effect",
            risk_class="external",
            steps=[
                {
                    "description": "contact external service",
                    "surface": "network",
                    "expected_output": "network response",
                }
            ],
            allowed_surfaces=["network"],
            stop_conditions=["approval_missing"],
            approved=True,
            approved_by="Nova",
        )

        self.assertEqual(plan.status, "pending_approval")
        self.assertTrue(plan.permission.approval_required)
        self.assertFalse(plan.permission.approved)
        self.assertIn("self_or_unattributed_approval_rejected", plan.permission.notes)

    def test_internal_plan_with_external_surface_is_blocked(self) -> None:
        engine = BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(active_os_user="nova")
        )

        plan = engine.create_plan(
            session_id="session-a",
            purpose="misclassified filesystem action",
            scope="internal label but external effect",
            execution_lane="internal_activity",
            risk_class="internal",
            steps=[
                {
                    "description": "touch host file",
                    "surface": "filesystem",
                    "expected_output": "file mutation",
                }
            ],
            allowed_surfaces=["filesystem"],
        )

        self.assertEqual(plan.status, "blocked")
        self.assertTrue(plan.permission.approval_required)
        self.assertIn(
            "internal_activity_disallowed_surfaces:filesystem",
            plan.permission.notes,
        )

    def test_empty_plan_request_is_rejected(self) -> None:
        engine = BoundedActionPlanEngine()

        with self.assertRaises(ActionPlanBoundaryError):
            engine.create_plan(
                session_id="session-a",
                purpose="empty",
                scope="none",
                execution_lane="internal_activity",
                risk_class="internal",
                steps=[],
            )


if __name__ == "__main__":
    unittest.main()
