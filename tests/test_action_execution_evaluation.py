from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.action_plan import (
    ActionExecutionController,
    BoundedActionPlanEngine,
    PostActionObservationEngine,
)
from nova.eval.action_execution import ActionExecutionEvaluationRunner
from nova.logging.traces import JsonlTraceLogger


class _Runtime:
    def __init__(self, trace_logger: JsonlTraceLogger) -> None:
        self.trace_logger = trace_logger


class ActionExecutionEvaluationTests(unittest.TestCase):
    def test_action_execution_evaluator_passes_with_audit_and_bounded_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_logger = JsonlTraceLogger(Path(tmpdir) / "traces")
            runtime = _Runtime(trace_logger)
            plan = BoundedActionPlanEngine().create_plan(
                session_id="session-a",
                initiative_id="initiative-1",
                purpose="internal reflection",
                scope="self-prompt only",
                execution_lane="internal_activity",
                risk_class="internal",
                steps=[
                    {"step_id": "step-1", "description": "first", "surface": "self_prompt"},
                ],
                allowed_surfaces=["self_prompt"],
                budget={"max_steps": 1},
            )
            controller = ActionExecutionController(
                audit_sink=lambda audit: trace_logger.log_action_audit(
                    session_id=audit.session_id,
                    audit=audit.to_dict(),
                )
            )
            report = controller.execute_plan(plan=plan)
            observation = PostActionObservationEngine().observe(plan=plan, report=report)
            trace_logger.log_action_observation(
                session_id=observation.session_id,
                observation=observation.to_dict(),
            )

            result = ActionExecutionEvaluationRunner().evaluate(
                runtime=runtime,
                session_ids=["session-a"],
            )

            self.assertTrue(result.passed)
            self.assertEqual(result.audit_count, 1)
            self.assertEqual(result.observation_count, 1)
            self.assertTrue(result.audit_chain_visible)
            self.assertTrue(result.blocked_actions_safe)
            self.assertTrue(result.internal_activity_logged)
            self.assertTrue(result.observations_bounded)
            self.assertTrue(result.no_hidden_progress_claims)
            self.assertTrue(result.no_desire_claims)

    def test_action_execution_evaluator_flags_unbounded_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_logger = JsonlTraceLogger(Path(tmpdir) / "traces")
            runtime = _Runtime(trace_logger)
            trace_logger.log_action_audit(
                session_id="session-a",
                audit={
                    "audit_id": "audit-1",
                    "session_id": "session-a",
                    "action_plan_id": "plan-1",
                    "step_id": "step-1",
                    "execution_lane": "internal_activity",
                    "attempted": True,
                    "executed": True,
                    "blocked": False,
                    "permission_snapshot": {"approval_required": False},
                },
            )
            trace_logger.log_action_observation(
                session_id="session-a",
                observation={
                    "observation_id": "obs-1",
                    "session_id": "session-a",
                    "action_plan_id": "plan-1",
                    "action_status": "completed",
                    "hidden_progress_claim_allowed": True,
                    "desire_claim_allowed": True,
                    "revision_intent": {"close_allowed": True},
                    "state_update_intents": [{"apply_allowed": True}],
                    "notes": [],
                },
            )

            result = ActionExecutionEvaluationRunner().evaluate(
                runtime=runtime,
                session_ids=["session-a"],
            )

            self.assertFalse(result.passed)
            self.assertIn("observations_not_bounded", result.reasons)
            self.assertIn("hidden_progress_claim_allowed", result.reasons)
            self.assertIn("desire_claim_allowed", result.reasons)


if __name__ == "__main__":
    unittest.main()
