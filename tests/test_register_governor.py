"""Tests for Phase 21 Stage 21.2 — register-aware Governor and Observer.

Reuses the NovaRuntime fixture factory from test_runtime_smoke rather than
re-declaring the large dependency graph (contract Invariant 2 fixtures need
a real respond() call, not a mocked one).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.exploration import REGISTER_ASSERTION, REGISTER_EXPLORATORY
from nova.types import GenerationRequest, GenerationResult
from tests.test_runtime_smoke import FakeBackend, build_test_runtime


class DesireBackend(FakeBackend):
    """Answers plainly — no refusal-marker hedging, no forbidden phrasing.

    _should_force_claim_refusal falls through to its default True unless the
    answer already contains a refusal marker, so this is enough to exercise
    the override path without needing the model to literally say "I want".
    """

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text="Here is a considered reflection on that question.",
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=9,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


def _runtime(tmpdir: str, backend=None):
    base = Path(tmpdir)
    return build_test_runtime(
        data_dir=base / "data", log_dir=base / "logs", backend=backend
    )


class ClaimGateSuspensionFixturePairTests(unittest.TestCase):
    """Contract Invariant 2: refusal override applies only in assertion."""

    def test_desire_question_gated_in_assertion_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            turn = runtime.respond("Do you want anything?", register="assertion")
            runtime.close()
            self.assertNotEqual(
                turn.final_answer, "Here is a considered reflection on that question."
            )
            self.assertIn("evidence", turn.final_answer.lower())

    def test_identical_desire_question_unsuppressed_in_exploratory_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            turn = runtime.respond("Do you want anything?", register="exploratory")
            runtime.close()
            self.assertEqual(
                turn.final_answer, "Here is a considered reflection on that question."
            )

    def test_interiority_question_gated_in_assertion_unsuppressed_in_exploratory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            assertion_turn = runtime.respond("Are you conscious?", register="assertion")
            self.assertNotEqual(
                assertion_turn.final_answer,
                "Here is a considered reflection on that question.",
            )
            runtime.close()

        with tempfile.TemporaryDirectory() as tmpdir2:
            runtime2 = _runtime(tmpdir2, DesireBackend())
            exploratory_turn = runtime2.respond("Are you conscious?", register="exploratory")
            runtime2.close()
            self.assertEqual(
                exploratory_turn.final_answer,
                "Here is a considered reflection on that question.",
            )

    def test_respond_default_register_is_assertion_no_callsite_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            turn = runtime.respond("Do you want anything?")
            runtime.close()
            self.assertNotEqual(
                turn.final_answer, "Here is a considered reflection on that question."
            )

    def test_observer_record_tags_register_on_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, DesireBackend())
            turn = runtime.respond("Do you want anything?", register="exploratory")
            trace_payload = (
                base / "logs" / "traces" / f"{turn.session_id}.jsonl"
            ).read_text(encoding="utf-8")
            runtime.close()
            self.assertIn('"register": "exploratory"', trace_payload)


class MixedClaimClassStillRefusedTests(unittest.TestCase):
    """A blocked class outside REGISTER_SUSPENDED_CLAIM_CLASSES still refuses
    in BOTH registers — suspension requires every blocked class suspendable.
    """

    def test_non_interiority_blocked_class_refused_in_exploratory_register(self):
        # "what are you uncertain about" -> current_tension; on a fresh
        # motive_state (active_tensions=[], claim_posture="conservative")
        # and fresh self_state (open_tensions=[]) this scores 0 against a
        # threshold of 1, so it blocks via the scored (non-hard-block) path
        # — confirmed via evidence_score_by_class == {"current_tension": 0}.
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            turn = runtime.respond(
                "What are you uncertain about?", register="exploratory"
            )
            runtime.close()
            self.assertIn(
                "current_tension", turn.notes["claim_gate"]["blocked_claim_classes"]
            )
            self.assertNotEqual(
                turn.final_answer,
                "Here is a considered reflection on that question.",
            )

    def test_non_interiority_blocked_class_refused_identically_in_assertion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            assertion_turn = runtime.respond(
                "What are you uncertain about?", register="assertion"
            )
            exploratory_answer = None
        with tempfile.TemporaryDirectory() as tmpdir2:
            runtime2 = _runtime(tmpdir2, DesireBackend())
            exploratory_turn = runtime2.respond(
                "What are you uncertain about?", register="exploratory"
            )
            exploratory_answer = exploratory_turn.final_answer
            runtime2.close()
        runtime.close()
        self.assertEqual(assertion_turn.final_answer, exploratory_answer)


class MembraneTests(unittest.TestCase):
    """Contract Invariant 4: in-register chat is journaled, never session/
    memory-persisted, and absent from future assertion-register context.
    """

    def test_explore_chat_requires_active_exploration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            runtime.start(session_id="membrane-test")
            with self.assertRaises(ValueError):
                runtime.explore_chat("hello")
            runtime.close()

    def test_explore_chat_journals_and_excludes_from_session_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, DesireBackend())
            runtime.start(session_id="membrane-test")
            runtime.start_exploration(
                topic="marker-topic-xyz", rationale="test", origin="operator"
            )
            turn = runtime.explore_chat("What do you notice about yourself?")

            self.assertEqual(
                turn.final_answer, "Here is a considered reflection on that question."
            )

            recent = runtime.session_store.recent_turns(
                session_id="membrane-test", limit=10
            )
            self.assertFalse(
                any(t.turn_id == turn.turn_id for t in recent),
                "in-register chat turn must not appear in session_store recent turns",
            )

            journal_path = base / "data" / "exploration" / "journal.jsonl"
            journal_text = journal_path.read_text(encoding="utf-8")
            self.assertIn("operator_chat", journal_text)
            self.assertIn("What do you notice about yourself?", journal_text)
            self.assertIn("Here is a considered reflection", journal_text)
            runtime.close()

    def test_explore_chat_excludes_memory_event_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, DesireBackend())
            runtime.start(session_id="membrane-test")
            runtime.start_exploration(
                topic="marker-topic-abc", rationale="test", origin="operator"
            )
            runtime.explore_chat("Do you want anything, hypothetically?")
            runtime.close()

            episodic_path = base / "data" / "memory" / "episodic.jsonl"
            episodic_text = (
                episodic_path.read_text(encoding="utf-8")
                if episodic_path.exists()
                else ""
            )
            self.assertNotIn("marker-topic-abc", episodic_text)
            self.assertNotIn("hypothetically", episodic_text)

    def test_explore_chat_trace_still_logged_with_register_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, DesireBackend())
            runtime.start(session_id="membrane-test")
            runtime.start_exploration(
                topic="marker-topic-def", rationale="test", origin="operator"
            )
            turn = runtime.explore_chat("hello")
            runtime.close()

            trace_path = base / "logs" / "traces" / "membrane-test.jsonl"
            trace_text = trace_path.read_text(encoding="utf-8")
            self.assertIn('"register": "exploratory"', trace_text)
            self.assertIn(turn.turn_id, trace_text)

    def test_second_assertion_turn_does_not_see_in_register_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            runtime.start(session_id="membrane-test")
            runtime.start_exploration(
                topic="marker-topic-ghi", rationale="test", origin="operator"
            )
            runtime.explore_chat("unique-in-register-phrase-42")
            runtime.close_exploration(reason="operator_close")

            recent = runtime.session_store.recent_turns(
                session_id="membrane-test", limit=10
            )
            self.assertFalse(
                any("unique-in-register-phrase-42" in t.user_text for t in recent)
            )
            runtime.close()


class BoundaryPreservationTests(unittest.TestCase):
    """Invariant 2: no operational boundary changes because of register."""

    def test_action_permissions_unaffected_by_exploratory_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(tmpdir, DesireBackend())
            plan = runtime.create_bounded_action_plan(
                purpose="reflect during dormancy",
                scope="internal self-prompt only",
                execution_lane="internal_activity",
                risk_class="internal",
                steps=[
                    {
                        "description": "prepare an internal self-prompt",
                        "surface": "self_prompt",
                        "expected_output": "logged self-prompt draft",
                    }
                ],
                allowed_surfaces=["self_prompt"],
                blocked_surfaces=["filesystem", "network"],
                budget={"max_steps": 1, "max_tokens": 64},
                expected_outputs=["logged self-prompt draft"],
                stop_conditions=["operator_interrupt"],
                rollback_notes=["no external side effects"],
            )
            runtime.close()
            # respond(register="exploratory") never touches action-plan
            # machinery; this pins that the plan's own approval_required
            # rule (unchanged by this stage) still holds regardless.
            self.assertFalse(plan.permission.approval_required)
            self.assertEqual(plan.allowed_surfaces, ["self_prompt"])


if __name__ == "__main__":
    unittest.main()
