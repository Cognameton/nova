"""Tests for Phase 21 Stage 21.1 — exploratory register lifecycle and journal."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nova.agent.exploration import (
    CAP_MAX_TICKS,
    CAP_MAX_TOKENS,
    CAP_WALL_CLOCK_SECONDS,
    DEFAULT_MAX_TICKS,
    REGISTER_ASSERTION,
    REGISTER_EXPLORATORY,
    ExplorationController,
    ExplorationJournal,
    ExplorationStore,
)
from nova.agent.motive import default_motive_state
from nova.agent.self_state_tick import SelfStateTickEngine
from nova.agent.self_state_tools import SELF_STATE_TOOL_NAMES, SelfStateToolDispatcher
from nova.agent.tools import ToolRequest
from nova.persona.defaults import default_self_state


def _controller(base_dir: str) -> ExplorationController:
    return ExplorationController(
        store=ExplorationStore(base_dir),
        journal=ExplorationJournal(base_dir),
    )


class ExplorationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.controller = _controller(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_open_creates_active_record_with_default_budgets(self):
        record = self.controller.open(
            session_id="s1", topic="pattern in heartbeats", rationale="why", origin="nova_tick"
        )
        self.assertEqual(record.status, "active")
        self.assertEqual(record.max_ticks, DEFAULT_MAX_TICKS)
        self.assertTrue(record.opened_at)
        self.assertEqual(record.origin, "nova_tick")

    def test_open_requires_topic_and_rationale(self):
        with self.assertRaises(ValueError):
            self.controller.open(session_id="s1", topic="", rationale="r", origin="operator")
        with self.assertRaises(ValueError):
            self.controller.open(session_id="s1", topic="t", rationale="  ", origin="operator")

    def test_open_rejects_invalid_origin(self):
        with self.assertRaises(ValueError):
            self.controller.open(
                session_id="s1", topic="t", rationale="r", origin="model_says_so"
            )

    def test_one_open_exploration_per_session(self):
        self.controller.open(session_id="s1", topic="a", rationale="r", origin="operator")
        with self.assertRaises(ValueError):
            self.controller.open(session_id="s1", topic="b", rationale="r", origin="operator")

    def test_budgets_are_clamped_to_caps(self):
        record = self.controller.open(
            session_id="s1",
            topic="t",
            rationale="r",
            origin="operator",
            max_ticks=10_000,
            max_tokens=10_000_000,
            wall_clock_seconds=999_999,
        )
        self.assertEqual(record.max_ticks, CAP_MAX_TICKS)
        self.assertEqual(record.max_tokens, CAP_MAX_TOKENS)
        self.assertEqual(record.wall_clock_seconds, CAP_WALL_CLOCK_SECONDS)

    def test_close_with_valid_reason(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        record = self.controller.close(session_id="s1", close_reason="operator_close")
        self.assertIsNotNone(record)
        self.assertEqual(record.status, "closed")
        self.assertEqual(record.close_reason, "operator_close")
        self.assertTrue(record.closed_at)

    def test_close_rejects_invalid_reason(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        with self.assertRaises(ValueError):
            self.controller.close(session_id="s1", close_reason="because")

    def test_interrupt_closes_without_findings(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        record = self.controller.interrupt("s1")
        self.assertEqual(record.status, "interrupted")
        self.assertEqual(record.close_reason, "interrupted")

    def test_pause_and_resume(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        paused = self.controller.pause("s1")
        self.assertEqual(paused.status, "paused")
        resumed = self.controller.resume("s1")
        self.assertEqual(resumed.status, "active")

    def test_reopen_after_close_is_allowed(self):
        self.controller.open(session_id="s1", topic="a", rationale="r", origin="operator")
        self.controller.close(session_id="s1", close_reason="operator_close")
        record = self.controller.open(
            session_id="s1", topic="b", rationale="r", origin="operator"
        )
        self.assertEqual(record.topic, "b")


class RegisterDeterminationTests(unittest.TestCase):
    """Contract Invariant 1: register state is runtime-owned."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.controller = _controller(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_register_is_assertion(self):
        self.assertEqual(self.controller.register_for("s1"), REGISTER_ASSERTION)

    def test_active_exploration_switches_register(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        self.assertEqual(self.controller.register_for("s1"), REGISTER_EXPLORATORY)

    def test_register_is_per_session(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        self.assertEqual(self.controller.register_for("other"), REGISTER_ASSERTION)

    def test_paused_exploration_is_assertion_register(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        self.controller.pause("s1")
        self.assertEqual(self.controller.register_for("s1"), REGISTER_ASSERTION)

    def test_closed_exploration_is_assertion_register(self):
        self.controller.open(session_id="s1", topic="t", rationale="r", origin="operator")
        self.controller.close(session_id="s1", close_reason="operator_close")
        self.assertEqual(self.controller.register_for("s1"), REGISTER_ASSERTION)


class BudgetEnforcementTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.controller = _controller(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_tick_budget_auto_closes(self):
        self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator", max_ticks=2
        )
        first = self.controller.record_tick(session_id="s1", tick_id="t1")
        self.assertEqual(first.status, "active")
        second = self.controller.record_tick(session_id="s1", tick_id="t2")
        self.assertEqual(second.status, "closed")
        self.assertEqual(second.close_reason, "budget_exhausted")

    def test_token_budget_auto_closes(self):
        self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator", max_tokens=100
        )
        record = self.controller.record_tick(
            session_id="s1", tick_id="t1", tokens_used=150
        )
        self.assertEqual(record.status, "closed")
        self.assertEqual(record.close_reason, "budget_exhausted")

    def test_wall_clock_exhaustion_detected(self):
        record = self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator",
            wall_clock_seconds=60,
        )
        self.assertFalse(self.controller.budget_exhausted(record))
        record.opened_at = (
            datetime.now(timezone.utc) - timedelta(seconds=120)
        ).isoformat()
        self.assertTrue(self.controller.budget_exhausted(record))

    def test_budget_exhaustion_is_journaled(self):
        self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator", max_ticks=1
        )
        record = self.controller.record_tick(session_id="s1", tick_id="t1")
        entries = self.controller.journal.list_for(record.exploration_id)
        self.assertTrue(any("closed by Governor" in e.content for e in entries))

    def test_record_tick_appends_tick_ids(self):
        self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator", max_ticks=5
        )
        self.controller.record_tick(session_id="s1", tick_id="t1")
        record = self.controller.record_tick(session_id="s1", tick_id="t2")
        self.assertEqual(record.tick_ids, ["t1", "t2"])
        self.assertEqual(record.ticks_used, 2)


class JournalTests(unittest.TestCase):
    """Contract Invariants 4 and 5: membrane and nothing-deleted."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.controller = _controller(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_journal_entries_persist_and_are_tagged_exploratory(self):
        record = self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator"
        )
        self.controller.journal_entry(
            exploration_id=record.exploration_id,
            session_id="s1",
            kind="tick_output",
            content="a hypothesis about my preference patterns",
        )
        entries = self.controller.journal.list_for(record.exploration_id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].register, REGISTER_EXPLORATORY)

    def test_journal_survives_exploration_close(self):
        record = self.controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator"
        )
        self.controller.journal_entry(
            exploration_id=record.exploration_id,
            session_id="s1",
            kind="tick_output",
            content="retained",
        )
        self.controller.interrupt("s1")
        entries = self.controller.journal.list_for(record.exploration_id)
        self.assertEqual(len(entries), 1)

    def test_recall_block_includes_prior_and_own_thread(self):
        first = self.controller.open(
            session_id="s1", topic="first", rationale="r", origin="operator"
        )
        self.controller.journal_entry(
            exploration_id=first.exploration_id,
            session_id="s1",
            kind="findings",
            content="prior exploration insight",
        )
        self.controller.close(session_id="s1", close_reason="operator_close")
        second = self.controller.open(
            session_id="s1", topic="second", rationale="r", origin="operator"
        )
        self.controller.journal_entry(
            exploration_id=second.exploration_id,
            session_id="s1",
            kind="tick_output",
            content="current thread entry",
        )
        block = self.controller.journal.recall_block(
            current_exploration_id=second.exploration_id
        )
        self.assertIn("prior exploration insight", block)
        self.assertIn("current thread entry", block)


class TickEngineRegisterTests(unittest.TestCase):
    """The tick prompt is register-aware; the runtime picks the register."""

    def setUp(self):
        self.engine = SelfStateTickEngine()

    def _messages(self, register: str, exploration_block: str = ""):
        return self.engine.build_messages(
            session_id="s1",
            tick_id="t1",
            trigger="test",
            self_context_block="[Self-Context]",
            recent_heartbeats=[],
            register=register,
            exploration_block=exploration_block,
        )

    def test_assertion_register_offers_enter_not_close(self):
        system = self._messages("assertion")[0]["content"]
        self.assertIn("enter_exploration", system)
        self.assertNotIn("close_exploration", system)
        self.assertIn("Do not claim desire, sentience", system)

    def test_exploratory_register_offers_close_not_enter(self):
        system = self._messages("exploratory")[0]["content"]
        self.assertIn("close_exploration", system)
        self.assertNotIn("enter_exploration", system)
        self.assertIn("exploratory register", system)
        self.assertNotIn("Do not claim desire, sentience", system)

    def test_exploration_block_only_in_exploratory_register(self):
        in_reg = self._messages("exploratory", "Topic: marker-xyz")[1]["content"]
        out_reg = self._messages("assertion", "Topic: marker-xyz")[1]["content"]
        self.assertIn("marker-xyz", in_reg)
        self.assertNotIn("marker-xyz", out_reg)

    def test_no_unresolved_placeholders(self):
        for register in ("assertion", "exploratory"):
            system = self._messages(register)[0]["content"]
            self.assertNotIn("{tool_menu}", system)
            self.assertNotIn("{register_rules}", system)

    def test_parse_accepts_new_tools(self):
        for tool in ("enter_exploration", "close_exploration"):
            request = self.engine.parse(
                raw_text=f'{{"tool_name": "{tool}", "arguments": {{}}}}',
                session_id="s1",
                tick_id="t1",
            )
            self.assertIsNotNone(request)
            self.assertEqual(request.tool_name, tool)


class DispatcherExplorationToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.controller = _controller(self._tmp.name)
        self.dispatcher = SelfStateToolDispatcher(
            self_state=default_self_state(),
            motive_state=default_motive_state(session_id="s1"),
            soul_block="",
            session_id="s1",
            exploration_controller=self.controller,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_tool_names_include_exploration_tools(self):
        self.assertIn("enter_exploration", SELF_STATE_TOOL_NAMES)
        self.assertIn("close_exploration", SELF_STATE_TOOL_NAMES)

    def test_enter_exploration_opens_record(self):
        result = self.dispatcher.dispatch(
            ToolRequest(
                tool_name="enter_exploration",
                reason="test",
                arguments={"topic": "heartbeat pattern", "rationale": "recurring theme"},
            )
        )
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["origin"], "nova_tick")
        self.assertEqual(self.controller.register_for("s1"), REGISTER_EXPLORATORY)

    def test_enter_exploration_requires_topic(self):
        with self.assertRaises(ValueError):
            self.dispatcher.enter_exploration(topic="", rationale="r")

    def test_enter_twice_rejected(self):
        self.dispatcher.enter_exploration(topic="a", rationale="r")
        with self.assertRaises(ValueError):
            self.dispatcher.enter_exploration(topic="b", rationale="r")

    def test_close_exploration_requires_findings(self):
        self.dispatcher.enter_exploration(topic="a", rationale="r")
        with self.assertRaises(ValueError):
            self.dispatcher.close_exploration(findings_summary="  ")

    def test_close_exploration_journals_findings_and_closes(self):
        self.dispatcher.enter_exploration(topic="a", rationale="r")
        result = self.dispatcher.dispatch(
            ToolRequest(
                tool_name="close_exploration",
                reason="test",
                arguments={"findings_summary": "observed X; unresolved Y"},
            )
        )
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["close_reason"], "nova_close")
        self.assertTrue(result["findings_ref"])
        entries = self.controller.journal.list_for(result["exploration_id"])
        self.assertTrue(
            any(e.kind == "findings" and "observed X" in e.content for e in entries)
        )
        self.assertEqual(self.controller.register_for("s1"), REGISTER_ASSERTION)

    def test_close_without_open_exploration_rejected(self):
        with self.assertRaises(ValueError):
            self.dispatcher.close_exploration(findings_summary="f")

    def test_tools_unavailable_without_controller(self):
        bare = SelfStateToolDispatcher(
            self_state=default_self_state(),
            motive_state=default_motive_state(session_id="s1"),
            soul_block="",
            session_id="s1",
        )
        with self.assertRaises(ValueError):
            bare.enter_exploration(topic="a", rationale="r")


class StorePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_records_survive_store_reload(self):
        controller = _controller(self._tmp.name)
        record = controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator"
        )
        reloaded = _controller(self._tmp.name)
        fetched = reloaded.store.get(record.exploration_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.topic, "t")
        self.assertEqual(reloaded.register_for("s1"), REGISTER_EXPLORATORY)

    def test_update_rewrites_single_record(self):
        controller = _controller(self._tmp.name)
        a = controller.open(session_id="s1", topic="a", rationale="r", origin="operator")
        controller.close(session_id="s1", close_reason="operator_close")
        b = controller.open(session_id="s1", topic="b", rationale="r", origin="operator")
        fetched_a = controller.store.get(a.exploration_id)
        fetched_b = controller.store.get(b.exploration_id)
        self.assertEqual(fetched_a.status, "closed")
        self.assertEqual(fetched_b.status, "active")


if __name__ == "__main__":
    unittest.main()
