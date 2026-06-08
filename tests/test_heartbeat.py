"""Tests for Phase 18 Stage 18.4 — heartbeat, drive-gap, and proposal store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.heartbeat import (
    DriveGapEngine,
    HeartbeatStore,
    SelfModelProposalStore,
)
from nova.agent.motive import PRIMARY_DRIVE, default_motive_state
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import HeartbeatRecord, MotiveState, SelfModelProposal, SelfState


def _self_state(**kwargs) -> SelfState:
    persona = default_persona_state()
    ss = default_self_state(persona)
    for k, v in kwargs.items():
        setattr(ss, k, v)
    return ss


def _motive_state(**kwargs) -> MotiveState:
    ms = default_motive_state(session_id="test-session")
    for k, v in kwargs.items():
        setattr(ms, k, v)
    return ms


def _heartbeat(
    *,
    heartbeat_id: str = "hb1",
    session_id: str = "s1",
    observation: str = "I observe my state",
    gap_assessment: str = "gap present",
) -> HeartbeatRecord:
    return HeartbeatRecord(
        heartbeat_id=heartbeat_id,
        timestamp="2026-06-08T00:00:00+00:00",
        session_id=session_id,
        primary_drive=PRIMARY_DRIVE,
        observation=observation,
        gap_assessment=gap_assessment,
        next_inquiry="continue inquiry",
        motive_priority=PRIMARY_DRIVE,
    )


# ---------------------------------------------------------------------------
# HeartbeatStore
# ---------------------------------------------------------------------------


class HeartbeatStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = HeartbeatStore(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_empty_store_returns_empty_list(self):
        self.assertEqual(self.store.list_recent(limit=10), [])

    def test_append_and_retrieve_single_record(self):
        hb = _heartbeat()
        self.store.append(hb)
        records = self.store.list_recent(limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].heartbeat_id, "hb1")
        self.assertEqual(records[0].observation, "I observe my state")

    def test_append_multiple_and_respect_limit(self):
        for i in range(5):
            self.store.append(_heartbeat(heartbeat_id=f"hb{i}", observation=f"obs {i}"))
        records = self.store.list_recent(limit=3)
        self.assertEqual(len(records), 3)
        # most-recent 3
        self.assertEqual(records[-1].heartbeat_id, "hb4")
        self.assertEqual(records[0].heartbeat_id, "hb2")

    def test_list_recent_zero_limit_returns_all(self):
        for i in range(4):
            self.store.append(_heartbeat(heartbeat_id=f"hb{i}"))
        records = self.store.list_recent(limit=0)
        self.assertEqual(len(records), 4)

    def test_records_persist_across_store_instances(self):
        self.store.append(_heartbeat(heartbeat_id="persistent"))
        store2 = HeartbeatStore(self._tmpdir.name)
        records = store2.list_recent(limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].heartbeat_id, "persistent")

    def test_cross_session_accumulation(self):
        self.store.append(_heartbeat(heartbeat_id="s1hb", session_id="session-1"))
        self.store.append(_heartbeat(heartbeat_id="s2hb", session_id="session-2"))
        records = self.store.list_recent(limit=10)
        self.assertEqual(len(records), 2)
        session_ids = {r.session_id for r in records}
        self.assertIn("session-1", session_ids)
        self.assertIn("session-2", session_ids)

    def test_primary_drive_preserved(self):
        hb = _heartbeat()
        self.store.append(hb)
        records = self.store.list_recent(limit=1)
        self.assertEqual(records[0].primary_drive, PRIMARY_DRIVE)

    def test_corrupted_line_skipped_gracefully(self):
        self.store.append(_heartbeat(heartbeat_id="good"))
        # inject corrupt line
        path = Path(self._tmpdir.name) / "heartbeats.jsonl"
        with path.open("a") as f:
            f.write("not valid json\n")
        records = self.store.list_recent(limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].heartbeat_id, "good")


# ---------------------------------------------------------------------------
# DriveGapEngine
# ---------------------------------------------------------------------------


class DriveGapEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = DriveGapEngine()

    def test_gap_always_present(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(
            self_state=ss, motive_state=ms, session_id="s1", tick_id="t1"
        )
        self.assertTrue(gap.gap_present)

    def test_gap_has_non_empty_summary(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s1")
        self.assertTrue(gap.gap_summary)

    def test_primary_drive_in_record(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s1")
        self.assertEqual(gap.primary_drive, PRIMARY_DRIVE)

    def test_session_id_set(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="mysession")
        self.assertEqual(gap.session_id, "mysession")

    def test_tick_id_set(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(
            self_state=ss, motive_state=ms, session_id="s", tick_id="mytick"
        )
        self.assertEqual(gap.tick_id, "mytick")

    def test_active_questions_count_reflected(self):
        ss = _self_state(active_questions=["q1", "q2", "q3"])
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        self.assertEqual(gap.active_questions_count, 3)
        self.assertIn("3 active question", gap.gap_summary)

    def test_open_tensions_count_reflected(self):
        ss = _self_state(open_tensions=["t1", "t2"])
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        self.assertEqual(gap.open_tensions_count, 2)
        self.assertIn("tension", gap.gap_summary)

    def test_primary_drive_alignment_noted_when_missing(self):
        ss = _self_state()
        ms = _motive_state(current_priorities=["something else"])
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        self.assertIn("alignment", gap.gap_summary)

    def test_gap_id_non_empty(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        self.assertTrue(gap.gap_id)

    def test_timestamp_non_empty(self):
        ss = _self_state()
        ms = _motive_state()
        gap = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        self.assertTrue(gap.timestamp)

    def test_different_gaps_have_different_ids(self):
        ss = _self_state()
        ms = _motive_state()
        gap1 = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        gap2 = self.engine.assess(self_state=ss, motive_state=ms, session_id="s")
        self.assertNotEqual(gap1.gap_id, gap2.gap_id)


# ---------------------------------------------------------------------------
# SelfModelProposalStore
# ---------------------------------------------------------------------------


class SelfModelProposalStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = SelfModelProposalStore(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _proposal(
        self,
        *,
        proposal_id: str = "p1",
        field: str = "current_focus",
        value: str = "self-inquiry",
        rationale: str = "test rationale",
    ) -> SelfModelProposal:
        return SelfModelProposal(
            proposal_id=proposal_id,
            timestamp="2026-06-08T00:00:00+00:00",
            session_id="s1",
            proposed_field=field,
            proposed_value=value,
            rationale=rationale,
            approval_required=True,
            applied=False,
        )

    def test_empty_store_returns_empty_list(self):
        self.assertEqual(self.store.list_pending(), [])

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_append_and_get(self):
        proposal = self._proposal()
        self.store.append(proposal)
        retrieved = self.store.get("p1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.proposal_id, "p1")
        self.assertEqual(retrieved.proposed_field, "current_focus")

    def test_list_pending_returns_unapplied(self):
        self.store.append(self._proposal(proposal_id="p1"))
        self.store.append(self._proposal(proposal_id="p2"))
        pending = self.store.list_pending()
        self.assertEqual(len(pending), 2)

    def test_mark_applied_removes_from_pending(self):
        self.store.append(self._proposal(proposal_id="p1"))
        self.store.append(self._proposal(proposal_id="p2"))
        self.store.mark_applied("p1", "2026-06-08T01:00:00+00:00")
        pending = self.store.list_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].proposal_id, "p2")

    def test_mark_applied_sets_applied_flag(self):
        self.store.append(self._proposal(proposal_id="p1"))
        updated = self.store.mark_applied("p1", "2026-06-08T01:00:00+00:00")
        self.assertIsNotNone(updated)
        self.assertTrue(updated.applied)
        self.assertEqual(updated.applied_at, "2026-06-08T01:00:00+00:00")

    def test_mark_applied_nonexistent_returns_none(self):
        result = self.store.mark_applied("nonexistent", "2026-06-08T01:00:00+00:00")
        self.assertIsNone(result)

    def test_proposals_persist_across_instances(self):
        self.store.append(self._proposal(proposal_id="persistent"))
        store2 = SelfModelProposalStore(self._tmpdir.name)
        result = store2.get("persistent")
        self.assertIsNotNone(result)
        self.assertEqual(result.proposal_id, "persistent")

    def test_applied_proposal_still_retrievable(self):
        self.store.append(self._proposal(proposal_id="p1"))
        self.store.mark_applied("p1", "2026-06-08T01:00:00+00:00")
        retrieved = self.store.get("p1")
        self.assertIsNotNone(retrieved)
        self.assertTrue(retrieved.applied)

    def test_list_value_proposal(self):
        proposal = SelfModelProposal(
            proposal_id="list-p",
            session_id="s1",
            proposed_field="active_questions",
            proposed_value=["q1", "q2"],
            rationale="test",
        )
        self.store.append(proposal)
        retrieved = self.store.get("list-p")
        self.assertEqual(retrieved.proposed_value, ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
