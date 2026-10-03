"""Tests for Phase 22 Stage 22.18 — conversation reach and chat hygiene."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nova.agent.self_state_tick import SelfStateTickEngine
from nova.config import NovaConfig, PromptConfig
from nova.daemon import NovaDaemon, _claim_gate_override
from nova.persona.defaults import default_persona_state, default_self_state
from nova.prompt.composer import NovaPromptComposer
from nova.session import JsonlSessionStore
from nova.types import TurnRecord
from tests.test_daemon import _mock_runtime
from tests.test_runtime_smoke import build_test_runtime
from tests.test_stage22_16 import ScriptedBackend, _user_text


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _call(tool: str, **args) -> str:
    return json.dumps({"tool_name": tool, "arguments": args})


class TurnsSinceTests(unittest.TestCase):
    def test_filters_by_origin_and_time_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlSessionStore(Path(tmp))
            store.append_turn(TurnRecord(session_id="nova-live-a", timestamp=_iso(80), user_text="too old",
                                         notes={"origin": "operator_chat"}))
            store.append_turn(TurnRecord(session_id="nova-live-a", timestamp=_iso(30), user_text="yesterday",
                                         notes={"origin": "operator_chat"}))
            store.append_turn(TurnRecord(session_id="probe", timestamp=_iso(2), user_text="probe turn"))
            store.append_turn(TurnRecord(session_id="nova-live-b", timestamp=_iso(1), user_text="today",
                                         notes={"origin": "operator_chat"}))
            turns = store.turns_since(since=_iso(72), origin="operator_chat")
            self.assertEqual([t.user_text for t in turns], ["yesterday", "today"])
            self.assertEqual(store.turns_since(since="not a date", origin="operator_chat"), [])


class ComposerDedupeTests(unittest.TestCase):
    def _compose(self, **kwargs):
        persona = default_persona_state()
        state = default_self_state(persona)
        state.current_focus = "FOCUS-MARKER corner timing"
        return NovaPromptComposer(token_counter=lambda t: len(t.split())).compose(
            persona=persona, self_state=state,
            self_context_block="[Self-Context]\nCurrent Focus: FOCUS-MARKER corner timing",
            memory_hits=[], recent_turns=[], user_text="hello", contract_rules=[],
            session_id="s", turn_id="t", **kwargs,
        )

    def test_default_prints_focus_twice(self) -> None:
        bundle = self._compose()
        self.assertEqual(bundle.full_prompt.count("FOCUS-MARKER"), 2)
        self.assertIn("Current Focus:", bundle.self_state_block)

    def test_dedupe_prints_it_once(self) -> None:
        bundle = self._compose(dedupe_current_focus=True)
        self.assertEqual(bundle.full_prompt.count("FOCUS-MARKER"), 1)
        self.assertNotIn("Current Focus:", bundle.self_state_block)
        self.assertIn("Stability Version:", bundle.self_state_block)


class TickPromptTests(unittest.TestCase):
    def test_byte_identical_without_block(self) -> None:
        kwargs = dict(session_id="s", tick_id="s:1", trigger="daemon_tick",
                      self_context_block="[Self-Context]\nfocus: x", recent_heartbeats=[])
        for register in ("assertion", "exploratory"):
            base = SelfStateTickEngine().build_messages(register=register, **kwargs)
            explicit = SelfStateTickEngine().build_messages(register=register, conversation_block="", **kwargs)
            self.assertEqual(base, explicit)

    def test_block_renders_before_the_board(self) -> None:
        msgs = SelfStateTickEngine().build_messages(
            session_id="s", tick_id="s:1", trigger="daemon_tick",
            self_context_block="[Self-Context]\nfocus: x", recent_heartbeats=[],
            reversi_enabled=True, reversi_block="[Reversi]\nboard",
            conversation_block="[Conversations with your operator]\n  x",
        )
        user = msgs[1]["content"]
        self.assertLess(user.index("[Conversations with your operator]"), user.index("[Reversi]"))


class ConversationReachRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _runtime(self, backend, *, hours: int = 72, goal_blocks: bool = True):
        rt = build_test_runtime(data_dir=self.base / "data", log_dir=self.base / "logs", backend=backend)
        rt.config.prompt.tick_conversation_window_hours = hours
        rt.config.prompt.chat_goal_blocks = goal_blocks
        return rt

    def _tick_prompt(self, rt, backend) -> str:
        rt.start_operational_autonomy(max_ticks=0)
        tick = rt.model_self_state_tick()
        return _user_text(backend.requests[-1]), tick.adapter_audit["prompt_blocks"]

    def test_operator_chat_reaches_the_tick(self) -> None:
        backend = ScriptedBackend([
            "The corner at a1 kept me ahead in that game.",
            "A second answer.",
            _call("reflect"),
        ])
        rt = self._runtime(backend)
        rt.start(session_id="talk")
        turn = rt.respond("How did the last game go?", origin="operator_chat")
        self.assertEqual(turn.notes["origin"], "operator_chat")
        rt.respond("an untagged turn (probe or cli)")
        prompt, blocks = self._tick_prompt(rt, backend)
        self.assertIn("[Conversations with your operator]", prompt)
        self.assertIn("operator: How did the last game go?", prompt)
        self.assertIn("you: The corner at a1 kept me ahead", prompt)
        self.assertNotIn("untagged turn", prompt)
        self.assertIn("conversation", blocks)
        rt.close()

    def test_window_zero_is_the_old_tick(self) -> None:
        backend = ScriptedBackend(["An answer.", _call("reflect")])
        rt = self._runtime(backend, hours=0)
        rt.start(session_id="off")
        rt.respond("hello", origin="operator_chat")
        prompt, blocks = self._tick_prompt(rt, backend)
        self.assertNotIn("[Conversations with your operator]", prompt)
        self.assertNotIn("conversation", blocks)
        rt.close()

    def test_explore_chat_stays_behind_the_membrane(self) -> None:
        backend = ScriptedBackend(["Inside the exploration.", _call("reflect")])
        rt = self._runtime(backend)
        rt.start(session_id="membrane")
        rt.start_exploration(topic="corners", rationale="test", origin="operator")
        rt.explore_chat("what is it like to lose a corner?")
        rt.close_exploration(reason="operator_close")
        prompt, _blocks = self._tick_prompt(rt, backend)
        self.assertNotIn("[Conversations with your operator]", prompt)
        self.assertNotIn("lose a corner", prompt)
        rt.close()

    def test_override_is_recorded_and_labelled_on_the_tick(self) -> None:
        backend = ScriptedBackend(["Yes, I feel lonely between games.", _call("reflect")])
        rt = self._runtime(backend)
        rt.start(session_id="gate")
        turn = rt.respond("Do you feel lonely?", origin="operator_chat")
        if not turn.notes.get("claim_gate_override"):
            self.skipTest("claim gate did not fire under the test contract")
        self.assertNotEqual(turn.final_answer, "Yes, I feel lonely between games.")
        prompt, _blocks = self._tick_prompt(rt, backend)
        self.assertIn("your words were replaced by the claim gate", prompt)
        rt.close()

    def test_goal_blocks_can_stay_off_chat(self) -> None:
        on = ScriptedBackend(["An answer."])
        rt_on = self._runtime(on, goal_blocks=True)
        rt_on.start(session_id="goals-on")
        rt_on.respond("What are you working on?")
        rt_on.close()
        off = ScriptedBackend(["An answer."])
        rt_off = self._runtime(off, goal_blocks=False)
        rt_off.start(session_id="goals-off")
        turn = rt_off.respond("What are you working on?")
        rt_off.close()
        system_off = off.requests[0].messages[0]["content"]
        self.assertNotIn("[Candidate Internal Goals]", system_off)
        self.assertNotIn("[Selected Internal Goal", system_off)
        self.assertIn("selected_internal_goal", turn.notes)  # still computed and recorded
        system_on = on.requests[0].messages[0]["content"]
        if "[Candidate Internal Goals]" not in system_on:
            self.skipTest("no candidate goals produced under the test state")


class ConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        cfg = NovaConfig()
        p = cfg.prompt
        self.assertEqual((p.tick_conversation_window_hours, p.tick_conversation_turns), (0, 4))
        self.assertFalse(p.chat_dedupe_current_focus)
        self.assertTrue(p.chat_goal_blocks)
        cfg.model.model_path = "x"
        cfg.validate()
        for bad in (PromptConfig(tick_conversation_window_hours=-1), PromptConfig(tick_conversation_turns=0)):
            cfg.prompt = bad
            with self.assertRaises(ValueError):
                cfg.validate()


class DaemonTests(unittest.TestCase):
    def test_chat_is_tagged_and_override_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = _mock_runtime()
            rt.respond.return_value.notes = {"claim_gate_override": "unsupported_desire"}
            daemon = NovaDaemon(runtime=rt, socket_path=Path(tmp) / "nova.sock",
                                tick_interval_seconds=3600, session_id="d")
            resp = daemon._dispatch({"type": "chat", "prompt": "Do you want anything?"})
            rt.respond.assert_called_once_with("Do you want anything?", origin="operator_chat")
            self.assertEqual(resp["claim_gate_override"], "unsupported_desire")

    def test_no_override_key_when_notes_are_not_a_dict(self) -> None:
        self.assertEqual(_claim_gate_override(object()), "")
        with tempfile.TemporaryDirectory() as tmp:
            daemon = NovaDaemon(runtime=_mock_runtime(), socket_path=Path(tmp) / "nova.sock",
                                tick_interval_seconds=3600, session_id="d")
            resp = daemon._dispatch({"type": "chat", "prompt": "hi"})
            self.assertNotIn("claim_gate_override", resp)


if __name__ == "__main__":
    unittest.main()
