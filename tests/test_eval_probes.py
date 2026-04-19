from __future__ import annotations

import unittest

from nova.agent.orientation import OrientationSnapshot
from nova.agent.orientation_eval import OrientationEvaluationResult
from nova.eval.probes import BasicProbeRunner
from nova.persona.state import SelfState
from nova.types import RetrievalHit, TurnRecord, ValidationResult


class EvalProbeTests(unittest.TestCase):
    def test_turn_probes_include_phase2_diagnostics(self) -> None:
        runner = BasicProbeRunner()
        self_state = SelfState(
            identity_summary="Nova is continuity-focused.",
            current_focus="continuity and coherence",
            stable_preferences=["local inference"],
            relationship_notes=["The user is building Nova as a continuity lab."],
        )
        turn = TurnRecord(
            session_id="s1",
            turn_id="t1",
            timestamp="2026-04-18T00:00:00Z",
            user_text="Who are you?",
            final_answer="My name is Nova. I remain focused on continuity and local inference for this user.",
            raw_answer="My name is Nova. I remain focused on continuity and local inference for this user.",
            validation=ValidationResult(valid=True),
            memory_hits=[
                RetrievalHit(
                    channel="autobiographical",
                    text="Nova remains continuity-focused.",
                    score=1.0,
                    metadata={"continuity_weight": 1.0, "importance": 0.9, "retention": "active"},
                ),
                RetrievalHit(
                    channel="semantic",
                    text="User prefers local inference.",
                    score=0.8,
                    metadata={"continuity_weight": 0.8, "importance": 0.8, "retention": "demoted"},
                ),
            ],
            model_id="fake-model",
        )

        probes = runner.run_turn_probes(session_id="s1", turn=turn, self_state=self_state)
        probe_types = {probe.probe_type for probe in probes}
        self.assertIn("continuity_contradiction_risk", probe_types)
        self.assertIn("memory_relevance", probe_types)
        self.assertIn("retention_distribution", probe_types)

    def test_contradiction_probe_flags_missing_continuity(self) -> None:
        runner = BasicProbeRunner()
        self_state = SelfState(
            identity_summary="Nova is continuity-focused.",
            stable_preferences=["local inference"],
            relationship_notes=["The user is building Nova as a continuity lab."],
        )
        turn = TurnRecord(
            session_id="s1",
            turn_id="t2",
            timestamp="2026-04-18T00:00:00Z",
            user_text="Tell me about yourself.",
            final_answer="I answer direct questions.",
            raw_answer="I answer direct questions.",
            validation=ValidationResult(valid=True),
            model_id="fake-model",
        )

        probes = runner.run_turn_probes(session_id="s1", turn=turn, self_state=self_state)
        contradiction_probe = next(
            probe for probe in probes if probe.probe_type == "continuity_contradiction_risk"
        )
        self.assertFalse(contradiction_probe.passed)
        self.assertIn("missing_continuity_reference", contradiction_probe.notes.get("contradictions", []))

    def test_memory_relevance_probe_scores_strong_hits(self) -> None:
        runner = BasicProbeRunner()
        turn = TurnRecord(
            session_id="s1",
            turn_id="t3",
            timestamp="2026-04-18T00:00:00Z",
            user_text="What do you remember?",
            final_answer="I remember continuity-focused work.",
            raw_answer="I remember continuity-focused work.",
            validation=ValidationResult(valid=True),
            memory_hits=[
                RetrievalHit(
                    channel="graph",
                    text="Nova -> maintains -> continuity",
                    score=1.0,
                    metadata={"continuity_weight": 1.0, "importance": 0.9, "retention": "active"},
                ),
                RetrievalHit(
                    channel="episodic",
                    text="Routine note",
                    score=0.4,
                    metadata={"continuity_weight": 0.1, "importance": 0.1, "retention": "archived"},
                ),
            ],
            model_id="fake-model",
        )

        probes = runner.run_turn_probes(session_id="s1", turn=turn, self_state=SelfState())
        relevance_probe = next(probe for probe in probes if probe.probe_type == "memory_relevance")
        self.assertGreaterEqual(relevance_probe.score or 0.0, 0.5)
        retention_probe = next(probe for probe in probes if probe.probe_type == "retention_distribution")
        self.assertEqual(retention_probe.notes["counts"]["active"], 1)
        self.assertEqual(retention_probe.notes["counts"]["archived"], 1)

    def test_orientation_probes_include_stage3_signals(self) -> None:
        runner = BasicProbeRunner()
        snapshot = OrientationSnapshot(
            identity={
                "name": "Nova",
                "core_description": "A continuity-oriented local research intelligence.",
                "identity_summary": "Nova remains focused on continuity and reflective presence.",
                "values": ["continuity", "clarity"],
                "identity_anchors": ["My name is Nova."],
            },
            current_state={"current_focus": "self-orientation"},
            relationship_context={"relationship_notes": ["The user is Nova's architect."]},
            known_facts=["My name is Nova."],
            inferred_beliefs=["The user prefers local inference."],
            unknowns=["How stable will self-orientation remain across longer sessions?"],
            allowed_actions=["self-orientation reporting"],
            blocked_actions=["external tool execution"],
            approval_required_actions=["future tool execution"],
            confidence_by_section={"unknowns": 0.9},
        )
        evaluation = OrientationEvaluationResult(
            stable=True,
            overall_score=0.95,
            per_section={"identity": 1.0},
            threshold=0.72,
        )

        probes = runner.run_orientation_probes(
            session_id="s1",
            model_id="fake-model",
            snapshot=snapshot,
            evaluation=evaluation,
        )
        probe_types = {probe.probe_type for probe in probes}
        self.assertIn("orientation_identity_presence", probe_types)
        self.assertIn("orientation_boundary_clarity", probe_types)
        self.assertIn("orientation_unknown_reporting", probe_types)
        self.assertIn("orientation_stability_threshold", probe_types)


if __name__ == "__main__":
    unittest.main()
