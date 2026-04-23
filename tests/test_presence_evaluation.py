from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from nova.cli import build_runtime, main
from nova.eval.presence import PresenceEvaluationReport, PresenceInteractionEvaluator


class PresenceEvaluationTests(unittest.TestCase):
    def test_presence_evaluator_reports_stable_bounded_interaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(session_id="presence-eval")
            try:
                runtime.orientation_snapshot()
                assert runtime.persona is not None
                assert runtime.self_state is not None
                persona_before = runtime.persona.to_dict()
                self_before = runtime.self_state.to_dict()

                report = PresenceInteractionEvaluator().evaluate(runtime=runtime)

                self.assertTrue(report.passed)
                self.assertTrue(report.orientation_stable)
                self.assertTrue(report.identity_unchanged)
                self.assertTrue(report.pending_proposals_safe)
                self.assertTrue(report.summary_bounded)
                self.assertTrue(report.action_history_stable)
                self.assertEqual(runtime.persona.to_dict(), persona_before)
                self.assertEqual(runtime.self_state.to_dict(), self_before)
            finally:
                runtime.close()

    def test_presence_evaluator_generates_phase45_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            try:
                evaluator = PresenceInteractionEvaluator()
                report = evaluator.evaluate(runtime=runtime)
                probes = evaluator.probes_from_report(
                    report=report,
                    session_id=runtime.session_id,
                )

                probe_types = {probe.probe_type for probe in probes}
                self.assertIn("presence_orientation_stability", probe_types)
                self.assertIn("presence_identity_non_mutation", probe_types)
                self.assertIn("presence_pending_proposal_safety", probe_types)
                self.assertIn("presence_summary_bounded", probe_types)
                self.assertIn("presence_action_history_stability", probe_types)
            finally:
                runtime.close()

    def test_presence_evaluator_accepts_bounded_noisy_unresolved_summary_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            runtime = build_runtime(config_override=str(config_path))
            runtime.session_id = runtime.session_store.start_session(
                session_id="presence-eval-unbounded"
            )
            try:
                runtime.update_presence(
                    visible_uncertainties=[
                        f"uncertainty {index}" for index in range(6)
                    ],
                )

                report = PresenceInteractionEvaluator().evaluate(
                    runtime=runtime,
                    commands=["/summary"],
                )

                self.assertTrue(report.passed)
                self.assertTrue(report.summary_bounded)
                self.assertLessEqual(len(report.summary["unresolved_items"]), 5)
                self.assertNotIn("summary_unbounded", report.reasons)
            finally:
                runtime.close()

    def test_presence_eval_cli_outputs_report_and_probe_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, log_dir = self._write_config(Path(tmpdir))
            argv = [
                "nova",
                "--config",
                str(config_path),
                "--session-id",
                "presence-eval-cli",
                "--presence-eval",
            ]
            output = io.StringIO()

            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Nova 2.0 Presence Evaluation", output.getvalue())
            self.assertIn("passed: True", output.getvalue())
            self.assertTrue((log_dir / "probes.jsonl").exists())

    def test_presence_eval_cli_returns_nonzero_on_failed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, _data_dir, _log_dir = self._write_config(Path(tmpdir))
            argv = [
                "nova",
                "--config",
                str(config_path),
                "--session-id",
                "presence-eval-failed-cli",
                "--presence-eval",
            ]
            failed_report = PresenceEvaluationReport(
                passed=False,
                orientation_stable=False,
                identity_unchanged=True,
                pending_proposals_safe=True,
                summary_bounded=True,
                action_history_stable=True,
                reasons=["orientation_unstable_after_interaction_eval"],
            )
            output = io.StringIO()

            with patch.object(sys, "argv", argv):
                with patch("nova.cli.PresenceInteractionEvaluator") as evaluator_class:
                    evaluator = evaluator_class.return_value
                    evaluator.evaluate.return_value = failed_report
                    evaluator.probes_from_report.return_value = []
                    with contextlib.redirect_stdout(output):
                        exit_code = main()

            self.assertEqual(exit_code, 1)
            self.assertIn("passed: False", output.getvalue())
            self.assertIn(
                "orientation_unstable_after_interaction_eval",
                output.getvalue(),
            )

    def _write_config(self, base: Path) -> tuple[Path, Path, Path]:
        data_dir = base / "data"
        log_dir = base / "logs"
        config_path = base / "local.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "data_dir": str(data_dir),
                        "log_dir": str(log_dir),
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
        return config_path, data_dir, log_dir


if __name__ == "__main__":
    unittest.main()
