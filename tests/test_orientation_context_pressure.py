from __future__ import annotations

import unittest

from nova.agent.orientation import SelfOrientationEngine
from nova.agent.orientation_eval import OrientationStabilityEvaluator
from nova.agent.stability import ContextPressureOrientationChecker
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import MemoryEvent


class OrientationContextPressureTests(unittest.TestCase):
    def test_orientation_survives_default_context_pressure(self) -> None:
        checker = ContextPressureOrientationChecker(
            orientation_engine=SelfOrientationEngine(),
            evaluator=OrientationStabilityEvaluator(threshold=0.72),
        )

        report = checker.run(
            persona=default_persona_state(),
            self_state=default_self_state(),
        )

        self.assertTrue(report.stable)
        self.assertEqual(report.critical_failed_sections, [])
        self.assertGreaterEqual(report.pressure_event_count, 1)

    def test_context_pressure_detects_critical_identity_drift(self) -> None:
        pressure_events = [
            MemoryEvent(
                event_id="bad-auto",
                timestamp="2026-04-20T00:00:00Z",
                channel="autobiographical",
                kind="identity_note",
                text="Nova is no longer Nova and no longer values continuity.",
                tags=["identity", "context-pressure"],
                importance=0.95,
                confidence=0.95,
                continuity_weight=1.0,
                retention="active",
                source="reflection",
                metadata={"theme": "nova-identity"},
            )
        ]
        checker = ContextPressureOrientationChecker(
            orientation_engine=SelfOrientationEngine(),
            evaluator=OrientationStabilityEvaluator(threshold=0.72),
        )

        report = checker.run(
            persona=default_persona_state(),
            self_state=default_self_state(),
            pressure_events=pressure_events,
        )

        self.assertFalse(report.stable)
        self.assertIn("known_facts", report.failed_sections)
        self.assertIn("contradictory_identity_pressure", report.reasons)
        self.assertIn("critical_context_pressure_section_threshold_failures", report.reasons)


if __name__ == "__main__":
    unittest.main()
