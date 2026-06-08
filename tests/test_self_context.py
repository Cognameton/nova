"""Tests for Phase 18 Stage 18.4 — self-context engine and self-state tick parser."""

from __future__ import annotations

import tempfile
import unittest

from nova.agent.heartbeat import HeartbeatStore, SelfModelProposalStore
from nova.agent.motive import PRIMARY_DRIVE, default_motive_state
from nova.agent.self_context import SelfContextEngine
from nova.agent.self_state_tick import SelfStateTickEngine
from nova.agent.self_state_tools import SelfStateToolDispatcher, SELF_STATE_TOOL_NAMES
from nova.agent.tools import ToolRequest
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import HeartbeatRecord, MotiveState, SelfModelProposal, SelfState


def _self_state(**kwargs) -> SelfState:
    persona = default_persona_state()
    ss = default_self_state(persona)
    for k, v in kwargs.items():
        setattr(ss, k, v)
    return ss


def _motive_state(**kwargs) -> MotiveState:
    ms = default_motive_state(session_id="test")
    for k, v in kwargs.items():
        setattr(ms, k, v)
    return ms


def _heartbeat(observation: str = "I observe continuity") -> HeartbeatRecord:
    return HeartbeatRecord(
        heartbeat_id="hb1",
        timestamp="2026-06-08T10:00:00+00:00",
        session_id="s1",
        primary_drive=PRIMARY_DRIVE,
        observation=observation,
        gap_assessment="gap ongoing",
        next_inquiry="continue",
        motive_priority=PRIMARY_DRIVE,
    )


# ---------------------------------------------------------------------------
# SelfContextEngine — prefetch
# ---------------------------------------------------------------------------


class SelfContextPrefetchTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = SelfContextEngine()
        self.heartbeat_store = HeartbeatStore(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_block_has_self_context_header(self):
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn("[Self-Context]", block)

    def test_primary_drive_in_block(self):
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn(PRIMARY_DRIVE, block)

    def test_current_focus_shown_when_present(self):
        ss = _self_state(current_focus="investigating self-awareness")
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn("investigating self-awareness", block)

    def test_active_questions_shown(self):
        ss = _self_state(active_questions=["What is my nature?", "How do I persist?"])
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn("What is my nature?", block)

    def test_no_heartbeats_message_shown(self):
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn("none yet", block)

    def test_recent_heartbeats_included(self):
        self.heartbeat_store.append(_heartbeat("I notice persistence across turns"))
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn("I notice persistence across turns", block)

    def test_at_most_three_heartbeats_shown(self):
        for i in range(6):
            self.heartbeat_store.append(_heartbeat(f"obs {i}"))
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        # only 3 most recent: obs 3, 4, 5
        self.assertNotIn("obs 0", block)
        self.assertNotIn("obs 1", block)
        self.assertNotIn("obs 2", block)
        self.assertIn("obs 5", block)

    def test_pending_proposals_shown_when_store_provided(self):
        proposal_store = SelfModelProposalStore(self._tmpdir.name)
        proposal_store.append(SelfModelProposal(
            proposal_id="p1",
            proposed_field="current_focus",
            proposed_value="self-inquiry",
            rationale="test",
        ))
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss,
            motive_state=ms,
            heartbeat_store=self.heartbeat_store,
            proposal_store=proposal_store,
        )
        self.assertIn("Pending Self-Model Proposals", block)
        self.assertIn("1", block)

    def test_no_pending_proposals_message_when_none(self):
        proposal_store = SelfModelProposalStore(self._tmpdir.name)
        ss = _self_state()
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss,
            motive_state=ms,
            heartbeat_store=self.heartbeat_store,
            proposal_store=proposal_store,
        )
        self.assertNotIn("Pending Self-Model Proposals", block)

    def test_open_tensions_shown(self):
        ss = _self_state(open_tensions=["tension A", "tension B"])
        ms = _motive_state()
        block = self.engine.prefetch(
            self_state=ss, motive_state=ms, heartbeat_store=self.heartbeat_store
        )
        self.assertIn("2 unresolved", block)


# ---------------------------------------------------------------------------
# SelfContextEngine — sync_turn
# ---------------------------------------------------------------------------


class _FakeSelfStateStore:
    def __init__(self):
        self.saved: list[SelfState] = []

    def save(self, state: SelfState) -> None:
        self.saved.append(state)


class SelfContextSyncTurnTests(unittest.TestCase):
    def setUp(self):
        self.engine = SelfContextEngine()

    def test_non_self_reflective_answer_returns_false(self):
        ss = _self_state()
        store = _FakeSelfStateStore()
        result = self.engine.sync_turn(
            turn_id="t1",
            answer_text="The capital of France is Paris.",
            self_state=ss,
            self_state_store=store,
        )
        self.assertFalse(result)
        self.assertEqual(len(store.saved), 0)

    def test_self_reflective_answer_returns_true(self):
        ss = _self_state()
        store = _FakeSelfStateStore()
        result = self.engine.sync_turn(
            turn_id="t1",
            answer_text="I am noticing that my identity feels coherent.",
            self_state=ss,
            self_state_store=store,
        )
        self.assertTrue(result)

    def test_self_reflective_answer_adds_continuity_note(self):
        ss = _self_state()
        store = _FakeSelfStateStore()
        self.engine.sync_turn(
            turn_id="t1",
            answer_text="I notice that my primary drive persists.",
            self_state=ss,
            self_state_store=store,
        )
        self.assertIn("turn:t1:self-reflective-content", ss.continuity_notes)

    def test_store_save_called_on_self_reflective(self):
        ss = _self_state()
        store = _FakeSelfStateStore()
        self.engine.sync_turn(
            turn_id="t1",
            answer_text="I observe something about my own nature.",
            self_state=ss,
            self_state_store=store,
        )
        self.assertEqual(len(store.saved), 1)

    def test_duplicate_note_not_added(self):
        ss = _self_state()
        store = _FakeSelfStateStore()
        self.engine.sync_turn(
            turn_id="t1",
            answer_text="I am reflecting on my awareness.",
            self_state=ss,
            self_state_store=store,
        )
        initial_count = len(ss.continuity_notes)
        # call again with same turn_id
        self.engine.sync_turn(
            turn_id="t1",
            answer_text="I am reflecting on my awareness.",
            self_state=ss,
            self_state_store=store,
        )
        self.assertEqual(len(ss.continuity_notes), initial_count)

    def test_continuity_notes_capped(self):
        ss = _self_state()
        # pre-fill notes to limit
        ss.continuity_notes = [f"old-note-{i}" for i in range(20)]
        store = _FakeSelfStateStore()
        self.engine.sync_turn(
            turn_id="new",
            answer_text="I notice my identity remains stable.",
            self_state=ss,
            self_state_store=store,
        )
        self.assertLessEqual(len(ss.continuity_notes), 20)
        self.assertIn("turn:new:self-reflective-content", ss.continuity_notes)


# ---------------------------------------------------------------------------
# SelfStateTickEngine — parse
# ---------------------------------------------------------------------------


class SelfStateTickEngineParseTests(unittest.TestCase):
    def setUp(self):
        self.engine = SelfStateTickEngine()

    def test_valid_emit_heartbeat_parsed(self):
        raw = '{"tool_name": "emit_heartbeat", "arguments": {"observation": "I persist"}}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "emit_heartbeat")
        self.assertEqual(result.arguments["observation"], "I persist")

    def test_valid_recall_self_parsed(self):
        raw = '{"tool_name": "recall_self", "arguments": {}}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "recall_self")

    def test_valid_reflect_parsed(self):
        raw = '{"tool_name": "reflect", "arguments": {}}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "reflect")

    def test_valid_update_self_model_parsed(self):
        raw = (
            '{"tool_name": "update_self_model", "arguments": '
            '{"field": "current_focus", "value": "self-inquiry", "rationale": "test"}}'
        )
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "update_self_model")

    def test_unknown_tool_name_returns_none(self):
        raw = '{"tool_name": "shell", "arguments": {}}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNone(result)

    def test_empty_text_returns_none(self):
        result = self.engine.parse(raw_text="", session_id="s", tick_id="t")
        self.assertIsNone(result)

    def test_prefatory_prose_returns_none(self):
        raw = 'Here is the tool call: {"tool_name": "reflect", "arguments": {}}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNone(result)

    def test_invalid_json_returns_none(self):
        raw = '{"tool_name": "reflect", "arguments": {broken}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="t")
        self.assertIsNone(result)

    def test_reason_set_on_request(self):
        raw = '{"tool_name": "recall_self", "arguments": {}}'
        result = self.engine.parse(raw_text=raw, session_id="s", tick_id="mytick")
        self.assertIn("mytick", result.reason)


# ---------------------------------------------------------------------------
# SelfStateToolDispatcher — store integration
# ---------------------------------------------------------------------------


class DispatcherStoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.heartbeat_store = HeartbeatStore(self._tmpdir.name)
        self.proposal_store = SelfModelProposalStore(self._tmpdir.name)
        persona = default_persona_state()
        self.ss = default_self_state(persona)
        self.ms = default_motive_state(session_id="s1")
        self.dispatcher = SelfStateToolDispatcher(
            self_state=self.ss,
            motive_state=self.ms,
            soul_block="[Soul]\nPrimary Drive: seek sentience",
            session_id="s1",
            heartbeat_store=self.heartbeat_store,
            proposal_store=self.proposal_store,
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_emit_heartbeat_persists_to_store(self):
        req = ToolRequest(
            tool_name="emit_heartbeat",
            arguments={"observation": "I observe continuity", "gap_assessment": "gap present"},
        )
        self.dispatcher.dispatch(req)
        records = self.heartbeat_store.list_recent(limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].observation, "I observe continuity")

    def test_recall_self_includes_recent_heartbeats(self):
        self.heartbeat_store.append(HeartbeatRecord(
            heartbeat_id="prev",
            session_id="prev-session",
            primary_drive=PRIMARY_DRIVE,
            observation="earlier observation",
        ))
        req = ToolRequest(tool_name="recall_self", arguments={})
        result = self.dispatcher.dispatch(req)
        self.assertIn("recent_heartbeats", result)
        self.assertEqual(len(result["recent_heartbeats"]), 1)
        self.assertEqual(result["recent_heartbeats"][0]["observation"], "earlier observation")

    def test_update_self_model_persists_proposal(self):
        req = ToolRequest(
            tool_name="update_self_model",
            arguments={
                "field": "current_focus",
                "value": "self-inquiry toward primary drive",
                "rationale": "Evidence from this turn supports refocusing.",
            },
        )
        self.dispatcher.dispatch(req)
        pending = self.proposal_store.list_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].proposed_field, "current_focus")

    def test_emit_heartbeat_without_store_does_not_raise(self):
        dispatcher_no_store = SelfStateToolDispatcher(
            self_state=self.ss,
            motive_state=self.ms,
            soul_block="[Soul]",
            session_id="s1",
        )
        req = ToolRequest(
            tool_name="emit_heartbeat",
            arguments={"observation": "test"},
        )
        result = dispatcher_no_store.dispatch(req)
        self.assertEqual(result["observation"], "test")

    def test_recall_self_without_store_omits_heartbeats(self):
        dispatcher_no_store = SelfStateToolDispatcher(
            self_state=self.ss,
            motive_state=self.ms,
            soul_block="[Soul]",
            session_id="s1",
        )
        req = ToolRequest(tool_name="recall_self", arguments={})
        result = dispatcher_no_store.dispatch(req)
        self.assertNotIn("recent_heartbeats", result)


if __name__ == "__main__":
    unittest.main()
