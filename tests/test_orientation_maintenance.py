from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.orientation import SelfOrientationEngine
from nova.agent.orientation_eval import OrientationStabilityEvaluator
from nova.agent.stability import MaintenanceOrientationStabilityChecker
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.maintenance import MemoryMaintenanceRunner
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import MemoryEvent


class OrientationMaintenanceTests(unittest.TestCase):
    def test_orientation_survives_reflection_and_maintenance_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            engram = JsonEngramMemoryStore(base / "engram.json")
            graph = SqliteGraphMemoryStore(base / "graph.db")
            autobiographical = JsonlAutobiographicalMemoryStore(base / "autobiographical.jsonl")
            semantic = JsonlSemanticMemoryStore(base / "semantic.jsonl")

            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-18T00:00:00Z",
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
                    timestamp="2026-04-18T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="assistant_message",
                    text="My name is Nova. I remain focused on continuity.",
                    tags=["assistant", "turn", "identity"],
                    importance=0.85,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e3",
                    timestamp="2026-04-18T00:02:00Z",
                    session_id="s1",
                    turn_id="t3",
                    channel="episodic",
                    kind="assistant_message",
                    text="I keep continuity at the center of my self-model.",
                    tags=["assistant", "turn", "identity", "value"],
                    importance=0.9,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )

            runner = MemoryMaintenanceRunner(
                episodic=episodic,
                engram=engram,
                graph=graph,
                autobiographical=autobiographical,
                semantic=semantic,
            )
            checker = MaintenanceOrientationStabilityChecker(
                orientation_engine=SelfOrientationEngine(),
                evaluator=OrientationStabilityEvaluator(threshold=0.72),
                maintenance_runner=runner,
            )

            report = checker.run(
                persona=default_persona_state(),
                self_state=default_self_state(),
                apply_mutations=False,
            )

            self.assertTrue(report.stable)
            self.assertGreaterEqual(report.semantic_written, 1)
            self.assertGreaterEqual(report.autobiographical_written, 1)
            self.assertEqual(report.applied, {})
            self.assertEqual(report.reasons, [])

    def test_orientation_maintenance_check_can_apply_retention_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            runner = MemoryMaintenanceRunner(episodic=episodic)

            episodic.add(
                MemoryEvent(
                    event_id="old1",
                    timestamp="2026-01-01T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="assistant_message",
                    text="Routine update that was already promoted.",
                    importance=0.3,
                    confidence=1.0,
                    continuity_weight=0.2,
                    source="nova",
                    metadata={"promoted": True},
                )
            )

            checker = MaintenanceOrientationStabilityChecker(
                orientation_engine=SelfOrientationEngine(),
                evaluator=OrientationStabilityEvaluator(threshold=0.72),
                maintenance_runner=runner,
            )
            report = checker.run(
                persona=default_persona_state(),
                self_state=default_self_state(),
                apply_mutations=True,
            )

            self.assertTrue(report.stable)
            self.assertEqual(report.applied.get("episodic"), 1)
            events = episodic.list_events()
            self.assertEqual(events[0].retention, "demoted")


if __name__ == "__main__":
    unittest.main()
