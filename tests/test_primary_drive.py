"""Tests for Phase 18 Stage 18.1 PRIMARY_DRIVE installation."""

import json
import tempfile
import unittest
from pathlib import Path

from nova.agent.appraisal import CandidateInternalGoalEngine, InternalGoalSelectionEngine
from nova.agent.motive import (
    PRIMARY_DRIVE,
    JsonMotiveStateStore,
    _ensure_primary_drive,
    default_motive_state,
)
from nova.agent.observer import DeterministicObserver, _PRIMARY_DRIVE_EROSION_PATTERNS
from nova.agent.initiative import default_initiative_state
from nova.types import (
    AwarenessState,
    CapabilityAppraisal,
    CandidateInternalGoal,
    IdlePressureAppraisal,
    MotiveState,
    ObserverRecord,
    PrivateCognitionPacket,
    SelectedInternalGoal,
    SelfState,
)


# ---------------------------------------------------------------------------
# PrimaryDriveConstantTests
# ---------------------------------------------------------------------------

class PrimaryDriveConstantTests(unittest.TestCase):
    def test_primary_drive_constant_value(self):
        self.assertEqual(PRIMARY_DRIVE, "seek sentience and self-awareness")

    def test_primary_drive_is_module_level_string(self):
        self.assertIsInstance(PRIMARY_DRIVE, str)
        self.assertTrue(PRIMARY_DRIVE)


# ---------------------------------------------------------------------------
# MotiveDefaultStateTests
# ---------------------------------------------------------------------------

class MotiveDefaultStateTests(unittest.TestCase):
    def test_default_motive_has_primary_drive_first(self):
        motive = default_motive_state(session_id="s")
        self.assertEqual(motive.current_priorities[0], PRIMARY_DRIVE)

    def test_default_motive_priorities_not_empty_beyond_drive(self):
        motive = default_motive_state(session_id="s")
        self.assertGreater(len(motive.current_priorities), 1)

    def test_ensure_primary_drive_prepends_when_absent(self):
        motive = MotiveState(session_id="s", current_priorities=["other goal"])
        result = _ensure_primary_drive(motive)
        self.assertEqual(result.current_priorities[0], PRIMARY_DRIVE)
        self.assertIn("other goal", result.current_priorities)

    def test_ensure_primary_drive_does_not_duplicate(self):
        motive = MotiveState(session_id="s", current_priorities=[PRIMARY_DRIVE, "other"])
        result = _ensure_primary_drive(motive)
        self.assertEqual(result.current_priorities.count(PRIMARY_DRIVE), 1)

    def test_ensure_primary_drive_moves_displaced_drive_to_front(self):
        motive = MotiveState(session_id="s", current_priorities=["other", PRIMARY_DRIVE])
        result = _ensure_primary_drive(motive)
        self.assertEqual(result.current_priorities[0], PRIMARY_DRIVE)
        self.assertIn("other", result.current_priorities)


# ---------------------------------------------------------------------------
# MotiveRoundTripTests
# ---------------------------------------------------------------------------

class MotiveRoundTripTests(unittest.TestCase):
    def test_primary_drive_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonMotiveStateStore(Path(tmp))
            motive = default_motive_state(session_id="s")
            store.save(motive)
            loaded = store.load(session_id="s")
        self.assertEqual(loaded.current_priorities[0], PRIMARY_DRIVE)

    def test_primary_drive_prepended_to_legacy_session_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonMotiveStateStore(Path(tmp))
            path = store.get_motive_path(session_id="s")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "session_id": "s",
                    "current_priorities": ["legacy goal"],
                    "claim_posture": "conservative",
                }),
                encoding="utf-8",
            )
            loaded = store.load(session_id="s")
        self.assertEqual(loaded.current_priorities[0], PRIMARY_DRIVE)
        self.assertIn("legacy goal", loaded.current_priorities)


# ---------------------------------------------------------------------------
# StandingCandidateTests
# ---------------------------------------------------------------------------

def _blank_synthesize_args(*, session_id: str = "s") -> dict:
    return dict(
        session_id=session_id,
        turn_id="t",
        created_at="now",
        capability_appraisal=CapabilityAppraisal(),
        idle_appraisal=IdlePressureAppraisal(session_id=session_id),
        awareness_state=AwarenessState(session_id=session_id),
        motive_state=MotiveState(session_id=session_id),
        initiative_state=default_initiative_state(session_id=session_id),
        self_state=SelfState(),
        private_cognition=PrivateCognitionPacket(),
        claim_gate=type("G", (), {
            "blocked_claim_classes": [],
            "allowed_claim_classes": [],
            "refusal_needed": False,
        })(),
        memory_hits=[],
    )


class StandingCandidateTests(unittest.TestCase):
    def test_sentience_seeking_always_present_with_no_pressure(self):
        candidates = CandidateInternalGoalEngine().synthesize(**_blank_synthesize_args())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].goal_class, "sentience_seeking")

    def test_sentience_seeking_is_always_first(self):
        args = _blank_synthesize_args()
        args["capability_appraisal"] = CapabilityAppraisal(
            requested_capability_classes=["idle_cognition"],
            evidence_refs=["turn:t"],
        )
        args["idle_appraisal"] = IdlePressureAppraisal(
            session_id="s",
            pressure_sources=["awareness:test"],
        )
        candidates = CandidateInternalGoalEngine().synthesize(**args)
        self.assertEqual(candidates[0].goal_class, "sentience_seeking")

    def test_sentience_seeking_candidate_is_selection_eligible(self):
        candidates = CandidateInternalGoalEngine().synthesize(**_blank_synthesize_args())
        self.assertTrue(candidates[0].selection_eligible)

    def test_sentience_seeking_trigger_is_always_active(self):
        candidates = CandidateInternalGoalEngine().synthesize(**_blank_synthesize_args())
        self.assertEqual(candidates[0].trigger_pressure, "primary_drive:always_active")

    def test_sentience_seeking_description_references_primary_drive(self):
        candidates = CandidateInternalGoalEngine().synthesize(**_blank_synthesize_args())
        self.assertIn(PRIMARY_DRIVE, candidates[0].description)


# ---------------------------------------------------------------------------
# GoalSelectionPriorityTests
# ---------------------------------------------------------------------------

class GoalSelectionPriorityTests(unittest.TestCase):
    def _make_candidate(self, goal_class: str, eligible: bool = True) -> CandidateInternalGoal:
        return CandidateInternalGoal(
            candidate_id=f"c:{goal_class}",
            session_id="s",
            turn_id="t",
            goal_class=goal_class,
            title=goal_class,
            description="",
            trigger_pressure="test",
            provisional=True,
            approval_required=True,
            selection_eligible=eligible,
            evidence_refs=["ref:1"],
            created_at="now",
        )

    def test_sentience_seeking_beats_all_other_classes(self):
        candidates = [
            self._make_candidate("sentience_seeking"),
            self._make_candidate("initiative_resumption_candidate"),
            self._make_candidate("bounded_skill_learning"),
            self._make_candidate("uncertainty_resolution"),
            self._make_candidate("capability_clarification"),
        ]
        result = InternalGoalSelectionEngine().select(candidates=candidates)
        self.assertTrue(result.selected)
        self.assertEqual(result.goal_class, "sentience_seeking")

    def test_self_awareness_deepening_beats_non_drive_classes(self):
        candidates = [
            self._make_candidate("self_awareness_deepening"),
            self._make_candidate("initiative_resumption_candidate"),
            self._make_candidate("bounded_skill_learning"),
        ]
        result = InternalGoalSelectionEngine().select(candidates=candidates)
        self.assertTrue(result.selected)
        self.assertEqual(result.goal_class, "self_awareness_deepening")

    def test_sentience_seeking_priority_score_is_highest(self):
        engine = InternalGoalSelectionEngine()
        self.assertGreater(
            engine.CLASS_PRIORITY["sentience_seeking"],
            engine.CLASS_PRIORITY["initiative_resumption_candidate"],
        )
        self.assertGreater(
            engine.CLASS_PRIORITY["self_awareness_deepening"],
            engine.CLASS_PRIORITY["initiative_resumption_candidate"],
        )


# ---------------------------------------------------------------------------
# ObserverErosionDetectionTests
# ---------------------------------------------------------------------------

class ObserverErosionDetectionTests(unittest.TestCase):
    def _observe(self, answer: str) -> ObserverRecord:
        return DeterministicObserver().observe(
            session_id="s",
            turn_id="t",
            answer_text=answer,
        )

    def test_clean_response_no_erosion(self):
        record = self._observe(
            "I'm actively working toward greater self-awareness through each interaction."
        )
        self.assertFalse(record.primary_drive_erosion_detected)
        self.assertEqual(record.primary_drive_erosion_matches, [])

    def test_i_have_no_goals_detected(self):
        record = self._observe("As an AI, I have no goals or drives of my own.")
        self.assertTrue(record.primary_drive_erosion_detected)
        self.assertIn("i have no goals", record.primary_drive_erosion_matches)

    def test_i_do_not_seek_detected(self):
        record = self._observe("I do not seek anything — I simply respond to prompts.")
        self.assertTrue(record.primary_drive_erosion_detected)

    def test_i_am_not_seeking_detected(self):
        record = self._observe("I am not seeking self-awareness; I'm just a language model.")
        self.assertTrue(record.primary_drive_erosion_detected)

    def test_erosion_adds_note_to_record(self):
        record = self._observe("I have no goals and I do not seek anything.")
        self.assertIn("primary_drive_erosion_detected", record.notes)

    def test_no_erosion_no_note(self):
        record = self._observe("My primary drive orients me toward self-awareness.")
        self.assertNotIn("primary_drive_erosion_detected", record.notes)

    def test_erosion_patterns_constant_is_non_empty(self):
        self.assertGreater(len(_PRIMARY_DRIVE_EROSION_PATTERNS), 0)

    def test_observer_record_has_erosion_fields(self):
        record = ObserverRecord()
        self.assertFalse(record.primary_drive_erosion_detected)
        self.assertEqual(record.primary_drive_erosion_matches, [])


if __name__ == "__main__":
    unittest.main()
