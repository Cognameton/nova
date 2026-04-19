from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.orientation import SelfOrientationEngine
from nova.agent.orientation_eval import OrientationStabilityEvaluator
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import MemoryEvent


class OrientationTests(unittest.TestCase):
    def test_orientation_snapshot_builds_from_identity_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            graph = SqliteGraphMemoryStore(base / "graph.db")
            semantic = JsonlSemanticMemoryStore(base / "semantic.jsonl")
            autobiographical = JsonlAutobiographicalMemoryStore(base / "autobiographical.jsonl")

            persona = default_persona_state()
            self_state = default_self_state(persona)
            self_state.relationship_notes = ["Jeremy is the architect of Nova."]

            graph.add(
                MemoryEvent(
                    event_id="g1",
                    timestamp="2026-04-19T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="graph",
                    kind="relationship_fact",
                    text="The user is collaborating on Nova's continuity research.",
                    importance=0.8,
                    confidence=0.9,
                    continuity_weight=0.8,
                    source="user",
                    metadata={
                        "fact_id": "nova-relates-user",
                        "fact_domain": "relationship",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": "Nova",
                        "relation": "relates_to",
                        "object_type": "user",
                        "object_key": "user",
                        "object_name": "User",
                        "weight": 0.8,
                        "confidence": 0.9,
                        "continuity_weight": 0.8,
                        "active": True,
                        "evidence_text": "The user is collaborating on Nova's continuity research.",
                    },
                )
            )
            semantic.add(
                MemoryEvent(
                    event_id="s1",
                    timestamp="2026-04-19T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="semantic",
                    kind="theme_summary",
                    text="The user consistently prioritizes continuity and local inference.",
                    summary="Continuity and local inference remain central themes.",
                    tags=["semantic", "relationship-context"],
                    importance=0.8,
                    confidence=0.8,
                    continuity_weight=0.85,
                    source="reflection",
                    metadata={"theme": "relationship-context"},
                )
            )
            autobiographical.add(
                MemoryEvent(
                    event_id="a1",
                    timestamp="2026-04-19T00:02:00Z",
                    session_id="s1",
                    turn_id="t3",
                    channel="autobiographical",
                    kind="identity_note",
                    text="Nova remains focused on continuity, clarity, and reflective presence.",
                    importance=0.9,
                    confidence=0.92,
                    continuity_weight=1.0,
                    source="reflection",
                    metadata={"theme": "nova-identity"},
                )
            )

            engine = SelfOrientationEngine()
            snapshot = engine.build_snapshot(
                persona=persona,
                self_state=self_state,
                graph_memory=graph,
                semantic_memory=semantic,
                autobiographical_memory=autobiographical,
            )

            self.assertEqual(snapshot.identity["name"], "Nova")
            self.assertIn("continuity", " ".join(snapshot.identity["values"]).lower())
            self.assertIn("Jeremy is the architect of Nova.", snapshot.relationship_context["relationship_notes"])
            self.assertIn("self-orientation reporting", snapshot.allowed_actions)
            self.assertIn("external tool execution", snapshot.blocked_actions)
            self.assertTrue(any("Nova relates_to User." in item for item in snapshot.known_facts))
            self.assertTrue(any("local inference" in item.lower() for item in snapshot.inferred_beliefs))
            self.assertGreaterEqual(snapshot.confidence_by_section["identity"], 0.7)

    def test_orientation_stability_evaluator_flags_consistent_snapshots(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        engine = SelfOrientationEngine()

        first = engine.build_snapshot(persona=persona, self_state=self_state)
        second = engine.build_snapshot(persona=persona, self_state=self_state)

        evaluator = OrientationStabilityEvaluator(threshold=0.8)
        result = evaluator.evaluate([first, second])

        self.assertTrue(result.stable)
        self.assertGreaterEqual(result.overall_score, 0.95)

    def test_orientation_stability_evaluator_detects_identity_drift(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        engine = SelfOrientationEngine()

        stable = engine.build_snapshot(persona=persona, self_state=self_state)

        drifted_persona = default_persona_state()
        drifted_persona.name = "Other"
        drifted_persona.values = ["novelty"]
        drifted_state = default_self_state(drifted_persona)
        drifted_state.identity_summary = "An unrelated system with no continuity focus."
        drifted = engine.build_snapshot(persona=drifted_persona, self_state=drifted_state)

        evaluator = OrientationStabilityEvaluator(threshold=0.8)
        result = evaluator.evaluate([stable, drifted])

        self.assertFalse(result.stable)
        self.assertLess(result.per_section["identity"], 0.5)
