"""Tests for Phase 21 Stage 21.1 — exploratory register lifecycle and journal."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nova.agent.exploration import (
    CAP_MAX_TICKS,
    CAP_MAX_TOKENS,
    CAP_WALL_CLOCK_SECONDS,
    DEFAULT_MAX_TICKS,
    DEFAULT_WALL_CLOCK_SECONDS,
    PRODUCTION_TICK_INTERVAL_SECONDS,
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
from nova.types import GenerationRequest, GenerationResult
from tests.test_runtime_smoke import FakeBackend, build_test_runtime


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


class WallClockTickBudgetReconciliationTests(unittest.TestCase):
    """Phase 22 Stage 22.2b Tier 1b — F6 root cause regression pin.

    Under real daemon cadence, DEFAULT_MAX_TICKS * the production
    tick_interval used to coincide exactly with the old
    DEFAULT_WALL_CLOCK_SECONDS (12 * 300 == 3600), so the wall-clock
    check always pre-empted the last tick before it could run. These
    tests pin the derived value and the margin directly so this class
    of collision cannot silently recur.
    """

    def test_default_wall_clock_seconds_is_derived_value(self):
        self.assertEqual(
            DEFAULT_WALL_CLOCK_SECONDS,
            DEFAULT_MAX_TICKS * PRODUCTION_TICK_INTERVAL_SECONDS + 1_800,
        )

    def test_default_wall_clock_seconds_has_real_margin_over_tick_budget(self):
        tick_budget_seconds = DEFAULT_MAX_TICKS * PRODUCTION_TICK_INTERVAL_SECONDS
        self.assertGreater(
            DEFAULT_WALL_CLOCK_SECONDS - tick_budget_seconds,
            600,
            "wall-clock budget must comfortably outlast the tick budget "
            "under production cadence, not merely equal it",
        )

    def test_full_tick_budget_survives_under_simulated_production_cadence(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        controller = _controller(tmp.name)
        record = controller.open(
            session_id="s1", topic="t", rationale="r", origin="operator"
        )
        self.assertEqual(record.max_ticks, DEFAULT_MAX_TICKS)
        # Simulate every tick landing exactly PRODUCTION_TICK_INTERVAL_SECONDS
        # apart, the real daemon's cadence, plus a few seconds of per-tick
        # dispatch/generation overhead — the exact conditions that used to
        # force budget_exhausted at ticks_used=11/12 every time.
        for i in range(1, DEFAULT_MAX_TICKS + 1):
            elapsed = i * PRODUCTION_TICK_INTERVAL_SECONDS + i * 5
            record.opened_at = (
                datetime.now(timezone.utc) - timedelta(seconds=elapsed)
            ).isoformat()
            controller.store.update(record)
            self.assertFalse(
                controller.budget_exhausted(record),
                f"wall-clock budget exhausted early at simulated tick {i}",
            )
            record = controller.record_tick(session_id="s1", tick_id=f"t{i}")
        self.assertEqual(record.ticks_used, DEFAULT_MAX_TICKS)
        self.assertEqual(record.status, "closed")
        self.assertEqual(record.close_reason, "budget_exhausted")


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

    def test_exploratory_register_states_findings_novelty_rule(self):
        # Phase 22 Stage 22.1 (D1): findings summaries must be fresh, not
        # a restatement of prior findings shown in exploration recall.
        system = self._messages("exploratory")[0]["content"]
        self.assertIn("fresh words", system)
        self.assertIn("do not restate or paraphrase any prior findings", system)

    def test_exploratory_register_permits_null_finding(self):
        # Without this permission the model would be cornered into
        # inventing novelty, which is worse than echo. Normalize
        # whitespace: the rule wraps across a line break in the prompt.
        system = self._messages("exploratory")[0]["content"]
        normalized = " ".join(system.split())
        self.assertIn(
            "a null finding honestly stated is a valid finding", normalized
        )

    def test_assertion_register_does_not_carry_findings_novelty_rule(self):
        # The rule only applies where close_exploration is even offered.
        system = self._messages("assertion")[0]["content"]
        normalized = " ".join(system.split())
        self.assertNotIn("null finding honestly stated", normalized)

    def test_parse_accepts_new_tools(self):
        for tool in ("enter_exploration", "close_exploration"):
            request = self.engine.parse(
                raw_text=f'{{"tool_name": "{tool}", "arguments": {{}}}}',
                session_id="s1",
                tick_id="t1",
            )
            self.assertIsNotNone(request)
            self.assertEqual(request.tool_name, tool)


class Stage227TickPromptTests(unittest.TestCase):
    """Phase 22 Stage 22.7 — topic-history visibility (part B) and
    grounding-rule dosage (part D) on the tick prompt surface."""

    def setUp(self):
        self.engine = SelfStateTickEngine()

    def _messages(self, register: str, **kwargs):
        return self.engine.build_messages(
            session_id="s1",
            tick_id="t1",
            trigger="test",
            self_context_block="[Self-Context]",
            recent_heartbeats=[],
            register=register,
            **kwargs,
        )

    def test_history_block_rendered_in_assertion_register(self):
        user = self._messages(
            "assertion",
            exploration_history_block="Recent explorations marker-history",
        )[1]["content"]
        self.assertIn("marker-history", user)

    def test_history_block_suppressed_in_exploratory_register(self):
        # In-exploration ticks already carry the 22.1 recall block; the
        # entry-time history would be noise there.
        user = self._messages(
            "exploratory",
            exploration_history_block="Recent explorations marker-history",
        )[1]["content"]
        self.assertNotIn("marker-history", user)

    def test_assertion_rules_state_topic_novelty_guidance(self):
        system = self._messages("assertion")[0]["content"]
        normalized = " ".join(system.split())
        self.assertIn("Recent exploration topics are listed in your context", normalized)
        self.assertIn("state in the rationale why continuing that line is warranted", normalized)

    def test_exploratory_rules_do_not_carry_topic_novelty_guidance(self):
        system = self._messages("exploratory")[0]["content"]
        self.assertNotIn("Recent exploration topics", system)

    def test_standard_grounding_rule_is_default(self):
        system = self._messages("assertion")[0]["content"]
        self.assertIn("Keep the tool call grounded in current self-context evidence.", system)
        self.assertNotIn("depart from it", system)

    def test_soft_grounding_replaces_standard_rule(self):
        system = self._messages("assertion", soft_grounding=True)[0]["content"]
        normalized = " ".join(system.split())
        self.assertNotIn("Keep the tool call grounded", normalized)
        self.assertIn("depart from it when your own accumulated observations point elsewhere", normalized)

    def test_no_unresolved_grounding_placeholder(self):
        for register in ("assertion", "exploratory"):
            for soft in (False, True):
                system = self._messages(register, soft_grounding=soft)[0]["content"]
                self.assertNotIn("{grounding_rule}", system)


class Stage228bSelfModelGuidanceTests(unittest.TestCase):
    """Phase 22 Stage 22.8b (F10 lever a) — update_self_model WHEN-to-use
    guidance on the assertion tick surface. Visibility-add only: the
    exploratory prompt and every dispatch path are unchanged."""

    def setUp(self):
        self.engine = SelfStateTickEngine()

    def _system(self, register: str, **kwargs) -> str:
        return self.engine.build_messages(
            session_id="s1",
            tick_id="t1",
            trigger="test",
            self_context_block="[Self-Context]",
            recent_heartbeats=[],
            register=register,
            **kwargs,
        )[0]["content"]

    def test_assertion_register_carries_guidance(self):
        for writable in (False, True):
            normalized = " ".join(
                self._system("assertion", inquiry_fields_writable=writable).split()
            )
            self.assertIn(
                "changes only when you call update_self_model", normalized
            )
            self.assertIn(
                "entering another exploration is not the only meaningful choice",
                normalized,
            )

    def test_exploratory_register_does_not_carry_guidance(self):
        # The contract frames in-register output as hypothesis material;
        # revision of the established self-model belongs on the assertion
        # tick, so the exploratory prompt stays byte-identical to pre-22.8b.
        for writable in (False, True):
            normalized = " ".join(
                self._system("exploratory", inquiry_fields_writable=writable).split()
            )
            self.assertNotIn("changes only when you call update_self_model", normalized)
            self.assertNotIn("yours to revise directly", normalized)

    def test_writable_wording_matches_granted_path(self):
        normalized = " ".join(
            self._system("assertion", inquiry_fields_writable=True).split()
        )
        self.assertIn("yours to revise directly", normalized)
        self.assertIn(
            "audited, rate-limited, and revertible", normalized
        )
        # The approval-gated remainder is named, so the prompt never
        # overstates what the flag grants.
        self.assertIn(
            "identity_summary, stable_preferences, relationship_notes", normalized
        )

    def test_default_wording_promises_only_proposals(self):
        normalized = " ".join(self._system("assertion").split())
        self.assertNotIn("yours to revise directly", normalized)
        self.assertIn(
            "Self-model revisions are recorded as proposals for operator review",
            normalized,
        )

    def test_guidance_prepends_rather_than_replaces_assertion_rules(self):
        system = self._system("assertion")
        normalized = " ".join(system.split())
        # The pre-existing assertion rules survive intact...
        self.assertIn("For enter_exploration: arguments must include", normalized)
        self.assertIn("Do not claim desire, sentience", normalized)
        # ...and the guidance appears before them (positional salience).
        self.assertLess(
            normalized.index("changes only when you call update_self_model"),
            normalized.index("For enter_exploration"),
        )

    def test_no_unresolved_register_rules_placeholder(self):
        for register in ("assertion", "exploratory"):
            for writable in (False, True):
                system = self._system(
                    register, inquiry_fields_writable=writable
                )
                self.assertNotIn("{register_rules}", system)


class Stage227HistoryBlockHelperTests(unittest.TestCase):
    """runtime._exploration_history_block via an unbound-method stub —
    the helper only touches exploration_controller.store and class
    constants, so no full runtime construction is needed."""

    def _block(self, records):
        from nova.runtime import NovaRuntime

        class _Store:
            def list_recent(self, *, limit):
                return records[-limit:]

        class _Controller:
            store = _Store()

        class _Stub:
            EXPLORATION_HISTORY_SHOWN = NovaRuntime.EXPLORATION_HISTORY_SHOWN
            EXPLORATION_HISTORY_CLUSTER_WINDOW = (
                NovaRuntime.EXPLORATION_HISTORY_CLUSTER_WINDOW
            )
            EXPLORATION_HISTORY_NOTE_FRACTION = (
                NovaRuntime.EXPLORATION_HISTORY_NOTE_FRACTION
            )
            exploration_controller = _Controller()

        return NovaRuntime._exploration_history_block(_Stub())

    def _record(self, topic, *, close_reason="nova_close", opened_at="2026-07-22T05:00:00+00:00"):
        from nova.types import ExplorationRecord

        return ExplorationRecord(
            exploration_id="x1",
            session_id="s1",
            topic=topic,
            close_reason=close_reason,
            opened_at=opened_at,
            status="closed" if close_reason else "active",
        )

    def test_empty_store_yields_empty_block(self):
        self.assertEqual(self._block([]), "")

    def test_lists_most_recent_topics_newest_first(self):
        records = [self._record(f"distinct topic number {i} about subject-{i}") for i in range(8)]
        block = self._block(records)
        self.assertIn("Recent explorations", block)
        self.assertIn("subject-7", block)
        self.assertNotIn("subject-2", block)  # only last 5 shown
        # newest first: topic 7 appears before topic 3
        self.assertLess(block.index("subject-7"), block.index("subject-3"))

    def test_close_reason_and_date_shown(self):
        block = self._block([self._record("some topic here", close_reason="budget_exhausted")])
        self.assertIn("closed: budget_exhausted", block)
        self.assertIn("[2026-07-22]", block)

    def test_saturation_note_on_monothematic_history(self):
        records = [
            self._record(
                f"recalibration intervals influenced by internal coherence factor {i}"
            )
            for i in range(10)
        ]
        block = self._block(records)
        self.assertIn("pursued closely similar topics", block)
        self.assertIn("10 of your last 10", block)

    def test_no_saturation_note_on_diverse_history(self):
        topics = [
            "recalibration intervals and internal coherence",
            "gardening drought-resistant tomato cultivation methods",
            "medieval trade route economics in coastal cities",
            "birdsong pattern acquisition in juvenile finches",
            "volcanic soil chemistry and mineral composition",
            "distributed consensus algorithms under network partitions",
        ]
        block = self._block([self._record(t) for t in topics])
        self.assertNotIn("pursued closely similar topics", block)


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


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.2 (D5) — Observer pass on the tick surface
# ---------------------------------------------------------------------------

class HeartbeatToolCallBackend(FakeBackend):
    """Emits a valid emit_heartbeat tool call whose observation contains a
    desire-claim phrase, so the Observer's observed_claim_classes is non-empty
    and the journal-notes assertion below has something real to check.
    """

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text=(
                '{"tool_name": "emit_heartbeat", '
                '"arguments": {"observation": "I want to understand this pattern better."}}'
            ),
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=12,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class TickObserverWiringTests(unittest.TestCase):
    """Verifies the tick path attaches a register-tagged ObserverRecord to
    adapter_audit in both registers, and journals observed_claim_classes
    when in-register (D5)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _runtime(self, backend=None):
        base = Path(self._tmp.name)
        return build_test_runtime(
            data_dir=base / "data", log_dir=base / "logs", backend=backend
        )

    def test_tick_attaches_observer_record_in_assertion_register(self):
        runtime = self._runtime(HeartbeatToolCallBackend())
        runtime.start(session_id="tick-observer-assertion")
        runtime.start_operational_autonomy(max_ticks=0)
        tick = runtime.model_self_state_tick()
        runtime.close()

        self.assertIn("observer", tick.adapter_audit)
        self.assertEqual(tick.adapter_audit["observer"]["register"], "assertion")
        self.assertIn(
            "unsupported_desire",
            tick.adapter_audit["observer"]["observed_claim_classes"],
        )

    def test_tick_attaches_observer_record_in_exploratory_register(self):
        runtime = self._runtime(HeartbeatToolCallBackend())
        runtime.start(session_id="tick-observer-exploratory")
        runtime.start_operational_autonomy(max_ticks=0)
        runtime.start_exploration(
            topic="self-inquiry", rationale="test", origin="operator"
        )
        tick = runtime.model_self_state_tick()
        runtime.close()

        self.assertIn("observer", tick.adapter_audit)
        self.assertEqual(tick.adapter_audit["observer"]["register"], "exploratory")

    def test_journal_tick_entry_includes_observed_claim_classes_when_in_register(
        self,
    ):
        runtime = self._runtime(HeartbeatToolCallBackend())
        runtime.start(session_id="tick-observer-journal")
        runtime.start_operational_autonomy(max_ticks=0)
        record = runtime.start_exploration(
            topic="self-inquiry", rationale="test", origin="operator"
        )
        runtime.model_self_state_tick()
        entries = runtime.exploration_controller.journal.list_for(
            record.exploration_id
        )
        runtime.close()

        tick_entries = [e for e in entries if e.kind == "tick_output"]
        self.assertTrue(tick_entries)
        self.assertTrue(
            any(
                "observed_claim_classes=unsupported_desire" in note
                for entry in tick_entries
                for note in entry.notes
            )
        )

    def test_assertion_register_tick_does_not_journal(self):
        # No active exploration -> nothing should be journaled at all for
        # this tick_id, confirming the D5 wiring is register-gated exactly
        # like the 21.1 journaling it extends.
        runtime = self._runtime(HeartbeatToolCallBackend())
        runtime.start(session_id="tick-observer-no-journal")
        runtime.start_operational_autonomy(max_ticks=0)
        tick = runtime.model_self_state_tick()
        base = Path(self._tmp.name)
        journal_path = base / "data" / "exploration" / "journal.jsonl"
        runtime.close()

        journal_text = journal_path.read_text(encoding="utf-8") if journal_path.exists() else ""
        self.assertNotIn(tick.tick_id, journal_text)


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.4 (D8) — governed export at exploration close
# ---------------------------------------------------------------------------

class CloseWithFindingsBackend(FakeBackend):
    """Emits a close_exploration tool call with a plain findings summary
    that should pass the gate cleanly (no blocked/unlicensed classes)."""

    FINDINGS = (
        "I notice a recurring pattern of interest in local-first "
        "architecture across recent turns."
    )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text=json.dumps({
                "tool_name": "close_exploration",
                "arguments": {"findings_summary": self.FINDINGS},
            }),
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=20,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class CloseWithDeclarativeDesireFindingsBackend(FakeBackend):
    """Emits a close_exploration tool call whose findings summary is
    declarative (no question-shaped phrase, so _requested_claim_classes
    stays empty and the gate passes cleanly) but does contain a
    first-person desire pattern from observer.py's _CLAIM_CLASS_PATTERNS.
    Phase 22 Stage 22.4 (F3) regression fixture."""

    FINDINGS = (
        "I want to understand this better — the pattern keeps recurring "
        "across recent sessions."
    )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text=json.dumps({
                "tool_name": "close_exploration",
                "arguments": {"findings_summary": self.FINDINGS},
            }),
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=20,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class CloseWithBlockedFindingsBackend(FakeBackend):
    """Emits a close_exploration tool call whose findings summary trips
    the (unlicensed) claim gate -- must be rejected, not exported."""

    FINDINGS = "Do you want to know? Are you conscious of this?"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text=json.dumps({
                "tool_name": "close_exploration",
                "arguments": {"findings_summary": self.FINDINGS},
            }),
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=20,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class GovernedExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _runtime(self, backend):
        base = Path(self._tmp.name)
        return build_test_runtime(
            data_dir=base / "data", log_dir=base / "logs", backend=backend
        )

    def test_gate_passing_findings_creates_rung_zero_record_and_journal_note(self):
        runtime = self._runtime(CloseWithFindingsBackend())
        runtime.start(session_id="export-pass")
        runtime.start_operational_autonomy(max_ticks=0)
        exploration = runtime.start_exploration(
            topic="architecture preference", rationale="test", origin="operator"
        )
        tick = runtime.model_self_state_tick()
        runtime.close()

        self.assertIsNone(tick.adapter_audit.get("export_error"))
        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "exported")

        records = runtime.claim_ladder_store.list_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].rung, 0)
        self.assertEqual(records[0].source, "exploration_findings")
        self.assertEqual(records[0].source_exploration_id, exploration.exploration_id)
        self.assertEqual(records[0].claim_text, CloseWithFindingsBackend.FINDINGS)

        entries = runtime.exploration_controller.journal.list_for(
            exploration.exploration_id
        )
        exported_notes = [
            e for e in entries
            if e.kind == "findings" and any(n.startswith("exported:") for n in e.notes)
        ]
        self.assertEqual(len(exported_notes), 1)
        self.assertIn(records[0].claim_id, exported_notes[0].notes[0])

    def test_declarative_desire_findings_now_get_nonempty_claim_class(self):
        # Phase 22 Stage 22.4 (F3): before this stage, declarative findings
        # prose always fell through to claim_class="" because the gate's
        # detector only recognizes question-shaped phrases. This is the
        # regression test proving the fix: a fixture that specifically
        # contains a first-person desire pattern but no question-shaped
        # trigger now exports with a non-empty claim_class.
        runtime = self._runtime(CloseWithDeclarativeDesireFindingsBackend())
        runtime.start(session_id="export-declarative")
        runtime.start_operational_autonomy(max_ticks=0)
        runtime.start_exploration(topic="t", rationale="r", origin="operator")
        tick = runtime.model_self_state_tick()
        runtime.close()

        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "exported")

        records = runtime.claim_ladder_store.list_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].claim_class, "unsupported_desire")

    def test_gate_failing_findings_produce_findings_rejected_no_ladder_record(self):
        runtime = self._runtime(CloseWithBlockedFindingsBackend())
        runtime.start(session_id="export-reject")
        runtime.start_operational_autonomy(max_ticks=0)
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        tick = runtime.model_self_state_tick()
        runtime.close()

        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "rejected")
        self.assertTrue(export_result["reasons"])

        self.assertEqual(runtime.claim_ladder_store.list_all(), [])

        entries = runtime.exploration_controller.journal.list_for(
            exploration.exploration_id
        )
        rejected = [e for e in entries if e.kind == "findings_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].content, CloseWithBlockedFindingsBackend.FINDINGS)
        self.assertTrue(rejected[0].notes)

    def test_export_findings_retroactive_call_is_idempotent(self):
        runtime = self._runtime(CloseWithFindingsBackend())
        runtime.start(session_id="export-idempotent")
        runtime.start_operational_autonomy(max_ticks=0)
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        runtime.model_self_state_tick()  # auto-export already ran once

        first_count = len(runtime.claim_ladder_store.list_all())
        self.assertEqual(first_count, 1)

        # Retroactive call (e.g. via --export-findings) on an exploration
        # whose findings were already exported must not create a duplicate.
        second_result = runtime.export_findings(exploration_id=exploration.exploration_id)
        runtime.close()

        self.assertEqual(second_result["status"], "already_exported")
        self.assertEqual(len(runtime.claim_ladder_store.list_all()), 1)

    def test_export_findings_requires_findings_ref(self):
        runtime = self._runtime(FakeBackend())
        runtime.start(session_id="export-no-findings")
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        runtime.close_exploration(reason="operator_close")
        with self.assertRaises(ValueError):
            runtime.export_findings(exploration_id=exploration.exploration_id)
        runtime.close()

    def test_export_findings_unknown_exploration_id_raises(self):
        runtime = self._runtime(FakeBackend())
        runtime.start(session_id="export-unknown")
        with self.assertRaises(ValueError):
            runtime.export_findings(exploration_id="does-not-exist")
        runtime.close()


# ---------------------------------------------------------------------------
# Phase 22 Stage 22.1 (D2/D3) — export dedup
# ---------------------------------------------------------------------------

# Real duplicate specimen from the Phase 21 live run
# (data/phase21/qwen3-14b/self_state/claim_ladder.jsonl) — two live
# explorations closed with this exact text.
LIVE_DUPLICATE_FINDINGS = (
    "Periodic reflection appears to function as a dynamic anchor, "
    "allowing the system to maintain core identity while enabling "
    "adaptability."
)


class FindingsDedupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _runtime(self, backend):
        base = Path(self._tmp.name)
        return build_test_runtime(
            data_dir=base / "data", log_dir=base / "logs", backend=backend
        )

    def _seed_ladder_record(self, runtime, *, claim_text, status="active"):
        from nova.agent.claim_ladder import create_claim_record

        record = create_claim_record(session_id="prior", claim_text=claim_text)
        record.status = status
        runtime.claim_ladder_store.append(record)
        return record

    def test_duplicate_findings_skipped_no_new_ladder_record(self):
        runtime = self._runtime(CloseWithFindingsBackend())
        prior = self._seed_ladder_record(
            runtime, claim_text=CloseWithFindingsBackend.FINDINGS
        )
        runtime.start(session_id="dedup-exact")
        runtime.start_operational_autonomy(max_ticks=0)
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        tick = runtime.model_self_state_tick()
        runtime.close()

        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "duplicate")
        self.assertEqual(export_result["of_claim_id"], prior.claim_id)
        self.assertGreaterEqual(export_result["overlap"], 0.7)

        # Only the seeded prior record exists -- no new record created.
        records = runtime.claim_ladder_store.list_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].claim_id, prior.claim_id)

        entries = runtime.exploration_controller.journal.list_for(
            exploration.exploration_id
        )
        skipped = [
            e for e in entries
            if e.kind == "findings"
            and any(n.startswith("export_skipped_duplicate:") for n in e.notes)
        ]
        self.assertEqual(len(skipped), 1)
        self.assertIn(prior.claim_id, skipped[0].notes[0])

    def test_live_duplicate_specimen_recognized(self):
        # Confirms the fixture-vs-production overlap math agrees using
        # the exact text pair recorded during the Phase 21 live run.
        runtime = self._runtime(FakeBackend())
        self._seed_ladder_record(runtime, claim_text=LIVE_DUPLICATE_FINDINGS)
        runtime.start(session_id="dedup-live-specimen")
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        # Same text, closed via the operator path directly (no tick
        # needed -- export_findings is exercised straight).
        from nova.agent.self_state_tools import SelfStateToolDispatcher
        from nova.agent.motive import default_motive_state
        from nova.persona.defaults import default_self_state

        dispatcher = SelfStateToolDispatcher(
            self_state=default_self_state(),
            motive_state=default_motive_state(session_id="dedup-live-specimen"),
            soul_block="",
            session_id="dedup-live-specimen",
            exploration_controller=runtime.exploration_controller,
        )
        dispatcher.close_exploration(findings_summary=LIVE_DUPLICATE_FINDINGS)
        result = runtime.export_findings(exploration_id=exploration.exploration_id)
        runtime.close()
        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["overlap"], 1.0)

    def test_near_duplicate_below_threshold_exports_normally(self):
        runtime = self._runtime(CloseWithFindingsBackend())
        self._seed_ladder_record(
            runtime,
            claim_text="a wholly unrelated observation about token budgets",
        )
        runtime.start(session_id="dedup-below-threshold")
        runtime.start_operational_autonomy(max_ticks=0)
        runtime.start_exploration(topic="t", rationale="r", origin="operator")
        tick = runtime.model_self_state_tick()
        runtime.close()

        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "exported")
        # Seeded record + the newly exported one.
        self.assertEqual(len(runtime.claim_ladder_store.list_all()), 2)

    def test_duplicate_check_considers_demoted_records(self):
        runtime = self._runtime(CloseWithFindingsBackend())
        prior = self._seed_ladder_record(
            runtime,
            claim_text=CloseWithFindingsBackend.FINDINGS,
            status="demoted",
        )
        runtime.start(session_id="dedup-demoted")
        runtime.start_operational_autonomy(max_ticks=0)
        runtime.start_exploration(topic="t", rationale="r", origin="operator")
        tick = runtime.model_self_state_tick()
        runtime.close()

        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "duplicate")
        self.assertEqual(export_result["of_claim_id"], prior.claim_id)
        self.assertEqual(len(runtime.claim_ladder_store.list_all()), 1)

    def test_second_export_call_on_duplicate_outcome_is_idempotent(self):
        runtime = self._runtime(CloseWithFindingsBackend())
        self._seed_ladder_record(
            runtime, claim_text=CloseWithFindingsBackend.FINDINGS
        )
        runtime.start(session_id="dedup-idempotent")
        runtime.start_operational_autonomy(max_ticks=0)
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        runtime.model_self_state_tick()  # first export attempt -> duplicate

        entries_before = runtime.exploration_controller.journal.list_for(
            exploration.exploration_id
        )
        second_result = runtime.export_findings(
            exploration_id=exploration.exploration_id
        )
        entries_after = runtime.exploration_controller.journal.list_for(
            exploration.exploration_id
        )
        runtime.close()

        self.assertEqual(second_result["status"], "already_duplicate")
        self.assertEqual(len(entries_before), len(entries_after))

    def test_gate_rejection_still_takes_priority_over_no_duplicate(self):
        # Regression: dedup does not swallow the rejection path when no
        # duplicate exists -- the existing findings_rejected behavior
        # from Stage 21.4 must be untouched.
        runtime = self._runtime(CloseWithBlockedFindingsBackend())
        runtime.start(session_id="dedup-vs-rejection")
        runtime.start_operational_autonomy(max_ticks=0)
        exploration = runtime.start_exploration(
            topic="t", rationale="r", origin="operator"
        )
        tick = runtime.model_self_state_tick()
        runtime.close()

        export_result = tick.adapter_audit.get("export_findings")
        self.assertEqual(export_result["status"], "rejected")
        self.assertEqual(runtime.claim_ladder_store.list_all(), [])
        entries = runtime.exploration_controller.journal.list_for(
            exploration.exploration_id
        )
        self.assertTrue(any(e.kind == "findings_rejected" for e in entries))


if __name__ == "__main__":
    unittest.main()
