from __future__ import annotations

import unittest

from nova.agent.private_cognition import PrivateCognitionEngine
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import RetrievalHit


class PrivateCognitionTests(unittest.TestCase):
    def test_packet_runs_for_continuity_sensitive_query(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        engine = PrivateCognitionEngine()

        packet = engine.build_packet(
            user_text="What do you remember about my current deployment preference and your continuity?",
            memory_hits=[
                RetrievalHit(
                    channel="semantic",
                    text="User preferences: hosted inference is now preferred.",
                    score=1.0,
                    source_ref="sem-1",
                    metadata={
                        "retention": "active",
                        "governance_status": "active",
                        "claim_axis": "deployment-style",
                        "claim_value": "hosted-inference",
                    },
                )
            ],
            self_state=self_state,
            enabled=True,
            pass_budget=1,
            revision_ceiling=1,
        )

        self.assertTrue(packet.ran)
        self.assertEqual(packet.trigger, "continuity_recall_query")
        self.assertEqual(packet.response_mode, "continuity_recall")
        self.assertEqual(packet.pass_budget_used, 1)
        self.assertEqual(packet.governing_memory_ids, ["sem-1"])
        self.assertEqual(packet.current_claims, ["deployment-style=hosted-inference"])

    def test_packet_detects_conflicting_claims(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        engine = PrivateCognitionEngine()

        packet = engine.build_packet(
            user_text="What is my current deployment preference?",
            memory_hits=[
                RetrievalHit(
                    channel="semantic",
                    text="User preferences: local inference remains preferred.",
                    score=1.0,
                    source_ref="sem-old",
                    metadata={
                        "retention": "archived",
                        "governance_status": "superseded",
                        "claim_axis": "deployment-style",
                        "claim_value": "local-inference",
                    },
                ),
                RetrievalHit(
                    channel="semantic",
                    text="User preferences: hosted inference is now preferred.",
                    score=1.1,
                    source_ref="sem-new",
                    metadata={
                        "retention": "active",
                        "governance_status": "active",
                        "claim_axis": "deployment-style",
                        "claim_value": "hosted-inference",
                    },
                ),
            ],
            self_state=self_state,
            enabled=True,
            pass_budget=1,
            revision_ceiling=1,
        )

        self.assertTrue(packet.memory_conflict)
        self.assertTrue(packet.revise_needed)
        self.assertEqual(packet.continuity_risk, "high")
        self.assertEqual(packet.governing_memory_ids, ["sem-new"])

    def test_packet_skips_declarative_preference_statement(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        engine = PrivateCognitionEngine()

        packet = engine.build_packet(
            user_text=(
                "My preferred deployment style for Nova is an always-on local service "
                "with direct answers and no hidden reasoning."
            ),
            memory_hits=[],
            self_state=self_state,
            enabled=True,
            pass_budget=1,
            revision_ceiling=1,
        )

        self.assertFalse(packet.ran)

    def test_prompt_block_is_structured_and_bounded(self) -> None:
        engine = PrivateCognitionEngine()
        block = engine.build_prompt_block(
            engine.build_packet(
                user_text="Who are you in continuity terms?",
                memory_hits=[],
                self_state=None,
                enabled=True,
                pass_budget=1,
                revision_ceiling=1,
            )
        )

        self.assertIn("[Private Cognition]", block)
        self.assertIn("response_mode:", block)
        self.assertIn("treat archived or historical continuity memory as background only", block)
        self.assertNotIn("because", block.lower())


if __name__ == "__main__":
    unittest.main()
