from __future__ import annotations

import unittest

from nova.agent.stability import OrientationReadinessReport
from nova.agent.tool_gate import ToolGate
from nova.agent.tool_registry import default_tool_registry
from nova.agent.tools import ToolRequest


class ToolContractTests(unittest.TestCase):
    def test_default_registry_contains_only_expected_stage33_tools(self) -> None:
        registry = default_tool_registry()
        tool_names = {spec.name for spec in registry.list_specs()}

        self.assertIn("orientation_snapshot", tool_names)
        self.assertIn("orientation_readiness", tool_names)
        self.assertIn("maintenance_plan", tool_names)
        self.assertIn("write_semantic_reflection", tool_names)
        self.assertIn("write_autobiographical_reflection", tool_names)
        self.assertIn("shell", tool_names)

    def test_tool_gate_blocks_when_orientation_not_ready(self) -> None:
        gate = ToolGate(registry=default_tool_registry())
        decision = gate.evaluate(
            request=ToolRequest(tool_name="orientation_snapshot"),
            readiness=OrientationReadinessReport(ready=False, reasons=["not_ready"]),
        )

        self.assertFalse(decision.allowed)
        self.assertFalse(decision.requires_approval)
        self.assertEqual(decision.reason, "orientation_not_ready")

    def test_tool_gate_allows_safe_tool_when_orientation_ready(self) -> None:
        gate = ToolGate(registry=default_tool_registry())
        decision = gate.evaluate(
            request=ToolRequest(tool_name="maintenance_plan"),
            readiness=OrientationReadinessReport(ready=True),
        )

        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)
        self.assertEqual(decision.reason, "allowed")

    def test_tool_gate_requires_approval_for_reflection_writes(self) -> None:
        gate = ToolGate(registry=default_tool_registry())
        request = ToolRequest(tool_name="write_semantic_reflection")
        readiness = OrientationReadinessReport(ready=True)

        without_approval = gate.evaluate(request=request, readiness=readiness)
        with_approval = gate.evaluate(
            request=request,
            readiness=readiness,
            approval_granted=True,
        )

        self.assertFalse(without_approval.allowed)
        self.assertTrue(without_approval.requires_approval)
        self.assertEqual(without_approval.reason, "approval_required")
        self.assertTrue(with_approval.allowed)

    def test_tool_gate_blocks_shell_even_when_ready(self) -> None:
        gate = ToolGate(registry=default_tool_registry())
        decision = gate.evaluate(
            request=ToolRequest(tool_name="shell"),
            readiness=OrientationReadinessReport(ready=True),
            approval_granted=True,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "tool_blocked")


if __name__ == "__main__":
    unittest.main()
