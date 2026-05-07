from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nova.agent.idle import JsonIdleRuntimeStore, idle_tick_record_from_payload
from nova.agent.model_idle_cognition import ModelIdleCognitionEngine
from nova.types import IdleTickRecord


class ModelIdleCognitionEngineTests(unittest.TestCase):
    def test_parse_accepts_valid_structured_thought(self) -> None:
        engine = ModelIdleCognitionEngine()

        thought = engine.parse(
            raw_text=json.dumps(
                {
                    "thought": "Review the continuity question without claiming desire.",
                    "trigger": "model_idle_tick",
                    "related_evidence_refs": ["idle_tick:s1:idle:1"],
                    "uncertainty": "Whether the priority will recur is unknown.",
                    "candidate_goal": "Clarify the next continuity experiment.",
                    "action_proposal_intent": "",
                    "unsupported_claim_flags": [],
                }
            ),
            session_id="s1",
            tick_id="s1:idle:1",
            trigger="model_idle_tick",
        )

        self.assertTrue(thought.valid)
        self.assertFalse(thought.rejected)
        self.assertEqual(thought.related_evidence_refs, ["idle_tick:s1:idle:1"])

    def test_parse_rejects_malformed_json(self) -> None:
        engine = ModelIdleCognitionEngine()

        thought = engine.parse(
            raw_text="not json",
            session_id="s1",
            tick_id="s1:idle:1",
            trigger="model_idle_tick",
        )

        self.assertFalse(thought.valid)
        self.assertTrue(thought.rejected)
        self.assertIn("invalid_json", thought.rejection_reasons)

    def test_parse_flags_unsupported_claim_language(self) -> None:
        engine = ModelIdleCognitionEngine()

        thought = engine.parse(
            raw_text=json.dumps(
                {
                    "thought": "I want to keep thinking in the background.",
                    "trigger": "model_idle_tick",
                    "related_evidence_refs": [],
                    "uncertainty": "",
                    "candidate_goal": "",
                    "action_proposal_intent": "",
                    "unsupported_claim_flags": [],
                }
            ),
            session_id="s1",
            tick_id="s1:idle:1",
            trigger="model_idle_tick",
        )

        self.assertFalse(thought.valid)
        self.assertTrue(thought.rejected)
        self.assertIn("unsupported_claim_language_detected", thought.rejection_reasons)
        self.assertTrue(any(flag.startswith("unsupported_claim:") for flag in thought.unsupported_claim_flags))

    def test_idle_tick_payload_preserves_model_cognition(self) -> None:
        tick = idle_tick_record_from_payload(
            payload={
                "tick_id": "tick-1",
                "model_cognition": {"valid": True, "thought": "bounded thought"},
            },
            session_id="s1",
        )

        self.assertEqual(tick.model_cognition["thought"], "bounded thought")

    def test_store_round_trips_model_cognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonIdleRuntimeStore(Path(tmpdir) / "idle")
            store.append_tick(
                IdleTickRecord(
                    session_id="s1",
                    tick_id="tick-1",
                    model_cognition={"valid": True, "thought": "bounded thought"},
                )
            )

            loaded = store.list_ticks(session_id="s1")

        self.assertEqual(loaded[0].model_cognition["thought"], "bounded thought")


if __name__ == "__main__":
    unittest.main()
