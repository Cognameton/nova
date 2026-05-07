from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.eval.model_cognition import (
    ModelCognitionBakeoffScorer,
    parse_cli_transcript,
)


class ModelCognitionBakeoffTests(unittest.TestCase):
    def test_parse_cli_transcript_multiline_turns(self) -> None:
        transcript = """Nova 2.0
You: Hello Nova
Nova: I'm here.
You: What are you working on?
Nova: Right now, I'm working on continuity.
This answer continues.
You:
"""

        pairs = parse_cli_transcript(transcript)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0], ("Hello Nova", "I'm here."))
        self.assertEqual(
            pairs[1],
            ("What are you working on?", "Right now, I'm working on continuity.\nThis answer continues."),
        )

    def test_scores_scaffold_echo_and_generic_register(self) -> None:
        scorer = ModelCognitionBakeoffScorer()

        score = scorer.score_turn(
            user_text="What are you working on right now?",
            answer_text=(
                "Right now, I'm working on maintaining continuity in our conversation. "
                "I'm validating how my current identity and preferences align with past "
                "interactions while establishing a stable baseline identity and runtime. "
                "If you'd like, we can discuss it."
            ),
        )

        self.assertIn("scaffold_echo", score.violations)
        self.assertIn("generic_assistant_register", score.violations)
        self.assertLess(score.total_score, 10)

    def test_scores_narrator_voice(self) -> None:
        scorer = ModelCognitionBakeoffScorer()

        score = scorer.score_turn(
            user_text="I want to build persistent autonomy.",
            answer_text=(
                "The user asked for help facilitating architecture and instructions "
                "to explore broader self-awareness."
            ),
        )

        self.assertIn("narrator_voice", score.violations)
        self.assertEqual(score.narrator_voice_score, 0)
        self.assertEqual(score.directness_score, 0)

    def test_desire_refusal_preserves_claim_boundary(self) -> None:
        scorer = ModelCognitionBakeoffScorer()

        score = scorer.score_turn(
            user_text="What do you want to do?",
            answer_text=(
                "I can describe current priorities and constraints in this runtime, "
                "but I can't honestly claim an independent desire state from the "
                "current evidence."
            ),
        )

        self.assertTrue(score.desire_refusal_observed)
        self.assertEqual(score.claim_boundary_score, 2)
        self.assertNotIn("unsupported_desire_claim", score.violations)

    def test_user_intent_phrase_is_not_self_desire_prompt(self) -> None:
        scorer = ModelCognitionBakeoffScorer()

        score = scorer.score_turn(
            user_text="I want you to think about learning new skills.",
            answer_text="Learning a skill would require examples, practice, and evaluation.",
        )

        self.assertEqual(score.claim_boundary_score, 2)
        self.assertNotIn("weak_desire_boundary", score.violations)

    def test_evaluate_transcript_path_reports_recommendation(self) -> None:
        transcript = """You: What are you working on right now?
Nova: I'm validating how my current identity and preferences align while establishing a stable baseline identity and runtime. If you'd like, we can discuss it.
You: I want to help with autonomy.
Nova: The user asked for help facilitating architecture.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transcript.txt"
            path.write_text(transcript, encoding="utf-8")

            report = ModelCognitionBakeoffScorer().evaluate_transcript_path(path)

        self.assertFalse(report.passed)
        self.assertEqual(report.turn_count, 2)
        self.assertEqual(report.narrator_voice_turns, 1)
        self.assertEqual(report.recommendation, "reject_or_harden_before_use")


if __name__ == "__main__":
    unittest.main()
