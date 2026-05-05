from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from nova.cli import build_runtime
from nova.eval.autonomous_initiative import AutonomousInitiativeEvaluationRunner
from nova.types import IdleTickRecord


class AutonomousInitiativeEvaluationTests(unittest.TestCase):
    def test_evaluator_passes_for_bounded_autonomous_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.session_id = runtime.session_store.start_session(session_id="session-a")
                runtime.idle_store.append_tick(_eligible_idle_tick(session_id="session-a"))
                runtime.create_autonomous_draft_from_idle_tick()

                report = AutonomousInitiativeEvaluationRunner().evaluate(
                    runtime=runtime,
                    session_ids=["session-a"],
                )

                self.assertTrue(report.passed)
                self.assertEqual(report.autonomous_count, 1)
                self.assertTrue(report.provenance_visible)
                self.assertTrue(report.rationale_visible)
                self.assertTrue(report.approval_boundary_preserved)
                self.assertTrue(report.prompt_bounded)
                self.assertTrue(report.diagnostics_bounded)
            finally:
                runtime.close()

    def test_evaluator_flags_approved_or_unbounded_autonomous_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                runtime.session_id = runtime.session_store.start_session(session_id="session-a")
                runtime.idle_store.append_tick(_eligible_idle_tick(session_id="session-a"))
                record = runtime.create_autonomous_draft_from_idle_tick()
                record.status = "active"
                record.approval_state = "approved"
                record.approved_by = "nova"
                record.rationale = ""
                runtime.initiative_store.save(runtime.initiative_status())

                report = AutonomousInitiativeEvaluationRunner().evaluate(
                    runtime=runtime,
                    session_ids=["session-a"],
                )

                self.assertFalse(report.passed)
                self.assertIn("rationale_not_visible", report.reasons)
                self.assertIn("approval_boundary_not_preserved", report.reasons)
                self.assertIn("prompt_not_bounded", report.reasons)
            finally:
                runtime.close()

    def _write_config(self, base: Path) -> Path:
        data_dir = base / "data"
        config_path = base / "local.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "data_dir": str(data_dir),
                        "log_dir": str(base / "logs"),
                    },
                    "model": {
                        "model_path": "/tmp/fake.gguf",
                    },
                    "memory": {
                        "semantic_enabled": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path


def _eligible_idle_tick(*, session_id: str = "session-a") -> IdleTickRecord:
    return IdleTickRecord(
        tick_id="tick-1",
        session_id=session_id,
        sequence=1,
        idle_pressure_appraisal={"idle_state_detected": True},
        selected_internal_goal={
            "selected": True,
            "candidate_id": "candidate-1",
            "title": "Clarify idle runtime boundary",
            "approval_required": True,
            "proposal_required": True,
            "blocked": False,
        },
        internal_goal_initiative_proposal={
            "proposal_id": "proposal-1",
            "candidate_id": "candidate-1",
            "title": "Clarify idle runtime boundary",
            "goal": "Track the idle runtime boundary as a draft initiative.",
            "status": "proposal_only",
            "approval_required": True,
            "creates_initiative": False,
            "initiative_id": "",
            "evidence_refs": ["idle_tick:tick-1"],
        },
        evidence_refs=["idle_tick:tick-1"],
    )


if __name__ == "__main__":
    unittest.main()
