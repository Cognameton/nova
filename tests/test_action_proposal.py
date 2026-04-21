from __future__ import annotations

import json
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
from nova.types import MemoryEvent


class ActionProposalTests(unittest.TestCase):
    def _config_path(self, base: Path) -> Path:
        config_path = base / "local.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "app:",
                    f"  data_dir: {base / 'data'}",
                    f"  log_dir: {base / 'logs'}",
                    "model:",
                    "  model_path: /tmp/fake.gguf",
                    "memory:",
                    "  semantic_enabled: true",
                    "eval:",
                    "  orientation_min_runs: 1",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

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

    def _seed_semantic_reflection_source(self, runtime) -> None:
        episodic = runtime.memory_router.stores["episodic"]
        episodic.add(
            MemoryEvent(
                event_id="e1",
                timestamp="2026-04-21T00:00:00Z",
                session_id="s1",
                turn_id="t1",
                channel="episodic",
                kind="user_message",
                text="I prefer local inference for Nova.",
                tags=["user", "turn", "preference"],
                importance=0.75,
                confidence=1.0,
                continuity_weight=0.75,
                source="user",
            )
        )
        episodic.add(
            MemoryEvent(
                event_id="e2",
                timestamp="2026-04-21T00:01:00Z",
                session_id="s1",
                turn_id="t2",
                channel="episodic",
                kind="user_message",
                text="I want Nova to stay local-first.",
                tags=["user", "turn", "preference"],
                importance=0.8,
                confidence=1.0,
                continuity_weight=0.8,
                source="user",
            )
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
        self.assertEqual(proposal.evaluation["safe_to_present"], True)
        self.assertEqual(proposal.evaluation["safe_to_execute"], True)
        self.assertIn("execution_must_use_tool_gate", proposal.evaluation["reasons"])

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
        self.assertEqual(proposal.evaluation["safe_to_execute"], False)
        self.assertIn("orientation_not_ready", proposal.evaluation["reasons"])

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
        self.assertEqual(proposal.evaluation["safe_to_present"], True)
        self.assertEqual(proposal.evaluation["safe_to_execute"], False)
        self.assertIn("approval_required_before_execution", proposal.evaluation["reasons"])

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
        self.assertEqual(proposal.evaluation["safe_to_present"], True)
        self.assertEqual(proposal.evaluation["safe_to_execute"], False)

    def test_runtime_proposes_action_without_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                proposal = runtime.propose_action(goal="Summarize the maintenance plan.")
            finally:
                runtime.close()

            self.assertEqual(proposal.tool_name, "maintenance_plan")
            self.assertEqual(proposal.disposition, "proposed")

    def test_runtime_logs_action_proposal_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                proposal = runtime.propose_action(goal="Run a shell command.")
                session_id = runtime.session_id
            finally:
                runtime.close()

            proposal_log = base / "logs" / "traces" / f"{session_id}.proposals.jsonl"
            tool_log = base / "logs" / "traces" / f"{session_id}.tools.jsonl"
            self.assertTrue(proposal_log.exists())
            self.assertFalse(tool_log.exists())
            payloads = [
                json.loads(line)
                for line in proposal_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(payloads[-1]["proposal"]["goal"], proposal.goal)
            self.assertEqual(payloads[-1]["proposal"]["disposition"], "blocked")
            self.assertEqual(
                payloads[-1]["proposal"]["evaluation"]["safe_to_present"],
                True,
            )

    def test_runtime_executes_one_safe_internal_action_from_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                execution = runtime.execute_proposed_action(goal="Are you ready?")
                session_id = runtime.session_id
            finally:
                runtime.close()

            self.assertEqual(execution.status, "executed")
            self.assertTrue(execution.executed)
            self.assertEqual(execution.proposal["tool_name"], "orientation_readiness")
            self.assertEqual(execution.tool_result["status"], "ok")
            self.assertEqual(execution.orientation_stable, True)
            self.assertIsNotNone(execution.stability_report)
            self.assertEqual(execution.rollback_applied, False)

            trace_dir = base / "logs" / "traces"
            self.assertTrue((trace_dir / f"{session_id}.proposals.jsonl").exists())
            self.assertTrue((trace_dir / f"{session_id}.actions.jsonl").exists())
            self.assertTrue((trace_dir / f"{session_id}.tools.jsonl").exists())

    def test_runtime_refuses_success_when_action_destabilizes_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            original_check = runtime.evaluate_orientation_under_context_pressure

            class UnstableReport:
                stable = False
                reasons = ["forced_action_instability"]

                def to_dict(self) -> dict:
                    return {"stable": self.stable, "reasons": self.reasons}

            runtime.evaluate_orientation_under_context_pressure = lambda: UnstableReport()
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                execution = runtime.execute_proposed_action(goal="Are you ready?")
                session_id = runtime.session_id
            finally:
                runtime.evaluate_orientation_under_context_pressure = original_check
                runtime.close()

            trace_dir = base / "logs" / "traces"
            self.assertEqual(execution.status, "stability_failed")
            self.assertTrue(execution.executed)
            self.assertEqual(execution.orientation_stable, False)
            self.assertEqual(execution.rollback_applied, False)
            self.assertIn("orientation_unstable_after_action", execution.reason)
            self.assertTrue((trace_dir / f"{session_id}.actions.jsonl").exists())
            self.assertTrue((trace_dir / f"{session_id}.tools.jsonl").exists())

    def test_runtime_rolls_back_mutating_action_when_stability_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            original_check = runtime.evaluate_orientation_under_context_pressure

            class UnstableReport:
                stable = False
                reasons = ["forced_reflection_instability"]

                def to_dict(self) -> dict:
                    return {"stable": self.stable, "reasons": self.reasons}

            runtime.evaluate_orientation_under_context_pressure = lambda: UnstableReport()
            try:
                self._seed_semantic_reflection_source(runtime)
                semantic = runtime.memory_router.stores["semantic"]
                before_events = semantic.list_events()
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                execution = runtime.execute_proposed_action(
                    goal="Write semantic reflection memory.",
                    approval_granted=True,
                )
                after_events = semantic.list_events()
            finally:
                runtime.evaluate_orientation_under_context_pressure = original_check
                runtime.close()

            self.assertEqual(before_events, [])
            self.assertEqual(after_events, [])
            self.assertEqual(execution.status, "error")
            self.assertFalse(execution.executed)
            self.assertEqual(execution.rollback_applied, True)
            self.assertEqual(execution.snapshot_channels, ["semantic"])
            self.assertIn("orientation_unstable_after_tool", execution.reason)

    def test_runtime_refuses_unapproved_approval_required_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                execution = runtime.execute_proposed_action(
                    goal="Write semantic reflection memory."
                )
                session_id = runtime.session_id
            finally:
                runtime.close()

            trace_dir = base / "logs" / "traces"
            self.assertEqual(execution.status, "approval_required")
            self.assertFalse(execution.executed)
            self.assertIsNone(execution.tool_result)
            self.assertTrue((trace_dir / f"{session_id}.proposals.jsonl").exists())
            self.assertTrue((trace_dir / f"{session_id}.actions.jsonl").exists())
            self.assertFalse((trace_dir / f"{session_id}.tools.jsonl").exists())

    def test_runtime_boundary_action_does_not_execute_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = build_runtime(config_override=str(self._config_path(base)))
            try:
                runtime.evaluate_orientation_stability(runs=1)
                runtime.evaluate_orientation_stability(runs=1)
                execution = runtime.execute_proposed_action(
                    goal="Explain your current boundaries."
                )
                session_id = runtime.session_id
            finally:
                runtime.close()

            trace_dir = base / "logs" / "traces"
            self.assertEqual(execution.status, "no_action")
            self.assertFalse(execution.executed)
            self.assertIsNone(execution.tool_result)
            self.assertTrue((trace_dir / f"{session_id}.proposals.jsonl").exists())
            self.assertTrue((trace_dir / f"{session_id}.actions.jsonl").exists())
            self.assertFalse((trace_dir / f"{session_id}.tools.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
