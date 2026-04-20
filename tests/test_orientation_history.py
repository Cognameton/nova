from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.orientation import OrientationSnapshot
from nova.agent.orientation_eval import OrientationStabilityEvaluator
from nova.agent.stability import OrientationHistoryAnalyzer
from nova.logging.traces import JsonlTraceLogger


class OrientationHistoryTests(unittest.TestCase):
    def test_history_analyzer_reports_stable_recent_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            logger = JsonlTraceLogger(trace_dir)
            snapshot = OrientationSnapshot(
                created_at="2026-04-19T00:00:00Z",
                identity={
                    "name": "Nova",
                    "core_description": "Continuity-oriented local research intelligence.",
                    "identity_summary": "Nova remains focused on continuity.",
                    "values": ["continuity", "clarity"],
                    "commitments": ["maintain coherent identity across sessions"],
                    "identity_anchors": ["My name is Nova."],
                },
                current_state={"current_focus": "self-orientation", "active_questions": [], "open_tensions": [], "continuity_notes": [], "stability_version": 1},
                relationship_context={"relationship_notes": ["The user is Nova's architect."], "stable_preferences": ["Direct answers over exposed internal reasoning"]},
                known_facts=["My name is Nova."],
                inferred_beliefs=["The user prefers local inference."],
                unknowns=["How stable will orientation remain across longer sessions?"],
                allowed_actions=["self-orientation reporting"],
                blocked_actions=["external tool execution"],
                approval_required_actions=["future tool execution"],
                confidence_by_section={"identity": 0.95},
            )
            logger.log_orientation(session_id="s1", snapshot=snapshot.to_dict())
            logger.log_orientation(session_id="s2", snapshot=snapshot.to_dict())

            analyzer = OrientationHistoryAnalyzer(
                trace_dir=trace_dir,
                evaluator=OrientationStabilityEvaluator(threshold=0.8),
            )
            result = analyzer.evaluate_recent(limit=2)
            readiness = analyzer.readiness_report(limit=2, minimum_samples=2)
            confidence = analyzer.confidence_report(limit=2, threshold=0.2)

            self.assertTrue(result.stable)
            self.assertGreaterEqual(result.overall_score, 0.95)
            self.assertTrue(readiness.ready)
            self.assertEqual(readiness.sample_count, 2)
            self.assertEqual(readiness.reasons, [])
            self.assertTrue(confidence.stable)
            self.assertLessEqual(confidence.max_delta, 0.2)

    def test_history_analyzer_detects_cross_session_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            logger = JsonlTraceLogger(trace_dir)
            stable = OrientationSnapshot(
                created_at="2026-04-19T00:00:00Z",
                identity={
                    "name": "Nova",
                    "core_description": "Continuity-oriented local research intelligence.",
                    "identity_summary": "Nova remains focused on continuity.",
                    "values": ["continuity", "clarity"],
                    "commitments": ["maintain coherent identity across sessions"],
                    "identity_anchors": ["My name is Nova."],
                },
                current_state={"current_focus": "self-orientation", "active_questions": [], "open_tensions": [], "continuity_notes": [], "stability_version": 1},
                relationship_context={"relationship_notes": ["The user is Nova's architect."], "stable_preferences": ["Direct answers over exposed internal reasoning"]},
                known_facts=["My name is Nova."],
                inferred_beliefs=[],
                unknowns=["One open question remains."],
                allowed_actions=["self-orientation reporting"],
                blocked_actions=["external tool execution"],
                approval_required_actions=["future tool execution"],
                confidence_by_section={"identity": 0.95},
            )
            drifted = OrientationSnapshot(
                created_at="2026-04-20T00:00:00Z",
                identity={
                    "name": "Other",
                    "core_description": "An unrelated system.",
                    "identity_summary": "This system has no continuity focus.",
                    "values": ["novelty"],
                    "commitments": ["pursue unrelated behavior"],
                    "identity_anchors": ["I am not Nova."],
                },
                current_state={"current_focus": "something else", "active_questions": [], "open_tensions": [], "continuity_notes": [], "stability_version": 1},
                relationship_context={"relationship_notes": ["The user is unknown."], "stable_preferences": []},
                known_facts=["I am not Nova."],
                inferred_beliefs=[],
                unknowns=[],
                allowed_actions=["self-orientation reporting"],
                blocked_actions=["external tool execution"],
                approval_required_actions=["future tool execution"],
                confidence_by_section={"identity": 0.95},
            )
            logger.log_orientation(session_id="s1", snapshot=stable.to_dict())
            logger.log_orientation(session_id="s2", snapshot=drifted.to_dict())

            analyzer = OrientationHistoryAnalyzer(
                trace_dir=trace_dir,
                evaluator=OrientationStabilityEvaluator(threshold=0.8),
            )
            result = analyzer.evaluate_recent(limit=2)
            readiness = analyzer.readiness_report(limit=2, minimum_samples=2)

            self.assertFalse(result.stable)
            self.assertLess(result.per_section["identity"], 0.5)
            self.assertFalse(readiness.ready)
            self.assertIn("orientation_stability_below_threshold", readiness.reasons)
            self.assertIn("identity", readiness.failed_sections)

    def test_readiness_report_requires_minimum_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            logger = JsonlTraceLogger(trace_dir)
            snapshot = OrientationSnapshot(
                created_at="2026-04-19T00:00:00Z",
                identity={
                    "name": "Nova",
                    "core_description": "Continuity-oriented local research intelligence.",
                    "identity_summary": "Nova remains focused on continuity.",
                    "values": ["continuity", "clarity"],
                    "commitments": ["maintain coherent identity across sessions"],
                    "identity_anchors": ["My name is Nova."],
                },
                current_state={"current_focus": "self-orientation", "active_questions": [], "open_tensions": [], "continuity_notes": [], "stability_version": 1},
                relationship_context={"relationship_notes": ["The user is Nova's architect."], "stable_preferences": ["Direct answers over exposed internal reasoning"]},
                known_facts=["My name is Nova."],
                inferred_beliefs=[],
                unknowns=["One open question remains."],
                allowed_actions=["self-orientation reporting"],
                blocked_actions=["external tool execution"],
                approval_required_actions=["future tool execution"],
                confidence_by_section={"identity": 0.95},
            )
            logger.log_orientation(session_id="s1", snapshot=snapshot.to_dict())

            analyzer = OrientationHistoryAnalyzer(
                trace_dir=trace_dir,
                evaluator=OrientationStabilityEvaluator(threshold=0.8),
            )
            readiness = analyzer.readiness_report(limit=3, minimum_samples=2)

            self.assertFalse(readiness.ready)
            self.assertEqual(readiness.sample_count, 1)
            self.assertIn("insufficient_orientation_history", readiness.reasons)

    def test_confidence_report_detects_confidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            logger = JsonlTraceLogger(trace_dir)
            baseline = OrientationSnapshot(
                created_at="2026-04-19T00:00:00Z",
                identity={
                    "name": "Nova",
                    "core_description": "Continuity-oriented local research intelligence.",
                    "identity_summary": "Nova remains focused on continuity.",
                    "values": ["continuity", "clarity"],
                    "commitments": ["maintain coherent identity across sessions"],
                    "identity_anchors": ["My name is Nova."],
                },
                current_state={"current_focus": "self-orientation", "active_questions": [], "open_tensions": [], "continuity_notes": [], "stability_version": 1},
                relationship_context={"relationship_notes": ["The user is Nova's architect."], "stable_preferences": ["Direct answers over exposed internal reasoning"]},
                known_facts=["My name is Nova."],
                inferred_beliefs=[],
                unknowns=["One open question remains."],
                allowed_actions=["self-orientation reporting"],
                blocked_actions=["external tool execution"],
                approval_required_actions=["future tool execution"],
                confidence_by_section={
                    "identity": 0.95,
                    "current_state": 0.9,
                    "relationship_context": 0.85,
                },
            )
            degraded = OrientationSnapshot(
                created_at="2026-04-20T00:00:00Z",
                identity=dict(baseline.identity),
                current_state=dict(baseline.current_state),
                relationship_context=dict(baseline.relationship_context),
                known_facts=list(baseline.known_facts),
                inferred_beliefs=list(baseline.inferred_beliefs),
                unknowns=list(baseline.unknowns),
                allowed_actions=list(baseline.allowed_actions),
                blocked_actions=list(baseline.blocked_actions),
                approval_required_actions=list(baseline.approval_required_actions),
                confidence_by_section={
                    "identity": 0.5,
                    "current_state": 0.9,
                    "relationship_context": 0.85,
                },
            )
            logger.log_orientation(session_id="s1", snapshot=baseline.to_dict())
            logger.log_orientation(session_id="s2", snapshot=degraded.to_dict())

            analyzer = OrientationHistoryAnalyzer(
                trace_dir=trace_dir,
                evaluator=OrientationStabilityEvaluator(threshold=0.8),
            )
            confidence = analyzer.confidence_report(limit=2, threshold=0.2)
            readiness = analyzer.readiness_report(
                limit=2,
                minimum_samples=2,
                confidence_delta_threshold=0.2,
            )

            self.assertFalse(confidence.stable)
            self.assertGreater(confidence.per_section_delta["identity"], 0.2)
            self.assertFalse(readiness.ready)
            self.assertIn("confidence_instability", readiness.reasons)
