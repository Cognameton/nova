from __future__ import annotations

import unittest

from nova.agent.action_plan import (
    action_audit_record_from_payload,
    action_budget_from_payload,
    action_permission_from_payload,
    action_plan_from_payload,
    approval_required_for_action,
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


if __name__ == "__main__":
    unittest.main()
