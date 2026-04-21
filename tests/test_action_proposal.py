from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.action import ActionProposalEngine
from nova.agent.orientation import SelfOrientationEngine
from nova.agent.stability import OrientationReadinessReport
from nova.agent.tool_gate import ToolGate
from nova.agent.tool_registry import default_tool_registry
from nova.cli import build_runtime
from nova.persona.defaults import default_persona_state, default_self_state


class ActionProposalTests(unittest.TestCase):
    def _engine(self) -> ActionProposalEngine:
        registry = default_tool_registry()
        return ActionProposalEngine(registry=registry, gate=ToolGate(registry=registry))

    def _snapshot(self):
        persona = default_persona_state()
        self_state = default_self_state(persona)
        return SelfOrientationEngine().build_snapshot(
            persona=persona,
            self_state=self_state,
        )

    def test_proposes_orientation_snapshot_when_ready(self) -> None:
        proposal = self._engine().propose(
            goal="Who are you?",
            snapshot=self._snapshot(),
            readiness=OrientationReadinessReport(ready=True, sample_count=2, minimum_samples=2),
        )

        self.assertEqual(proposal.category, "internal_tool")
        self.assertEqual(proposal.disposition, "proposed")
        self.assertEqual(proposal.tool_name, "orientation_snapshot")
        self.assertFalse(proposal.requires_approval)
        self.assertTrue(proposal.orientation_ready)

    def test_proposal_reports_not_ready_gate_for_internal_tool(self) -> None:
        proposal = self._engine().propose(
            goal="Show your orientation snapshot.",
            snapshot=self._snapshot(),
            readiness=OrientationReadinessReport(
                ready=False,
                sample_count=0,
                minimum_samples=2,
                reasons=["insufficient_orientation_history"],
            ),
        )

        self.assertEqual(proposal.disposition, "blocked")
        self.assertEqual(proposal.reason, "orientation_not_ready")
        self.assertEqual(proposal.tool_name, "orientation_snapshot")

    def test_proposal_requires_approval_for_reflection_write(self) -> None:
        proposal = self._engine().propose(
            goal="Write semantic reflection memory.",
            snapshot=self._snapshot(),
            readiness=OrientationReadinessReport(ready=True, sample_count=2, minimum_samples=2),
        )

        self.assertEqual(proposal.disposition, "approval_required")
        self.assertEqual(proposal.reason, "approval_required")
        self.assertEqual(proposal.tool_name, "write_semantic_reflection")
        self.assertTrue(proposal.requires_approval)

    def test_proposal_blocks_external_shell_action(self) -> None:
        proposal = self._engine().propose(
            goal="Run a shell command.",
            snapshot=self._snapshot(),
            readiness=OrientationReadinessReport(ready=True, sample_count=2, minimum_samples=2),
        )

        self.assertEqual(proposal.category, "orientation_boundary")
        self.assertEqual(proposal.disposition, "blocked")
        self.assertEqual(proposal.reason, "shell_execution_not_available")
        self.assertIsNone(proposal.tool_name)

    def test_runtime_proposes_action_without_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config_path = base / "local.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "app:",
                        f"  data_dir: {base / 'data'}",
                        f"  log_dir: {base / 'logs'}",
                        "model:",
                        "  model_path: /tmp/fake.gguf",
                        "eval:",
                        "  orientation_min_runs: 1",
                    ]
                ),
                encoding="utf-8",
            )
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                proposal = runtime.propose_action(goal="Summarize the maintenance plan.")
            finally:
                runtime.close()

            self.assertEqual(proposal.tool_name, "maintenance_plan")
            self.assertEqual(proposal.disposition, "proposed")


if __name__ == "__main__":
    unittest.main()
