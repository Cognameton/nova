"""Tests for Phase 22 Stage 22.17 — the deep tick."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nova.agent.heartbeat import HeartbeatStore
from nova.agent.motive import default_motive_state
from nova.agent.self_state_tick import SelfStateTickEngine, split_thinking
from nova.agent.self_state_tools import (
    READ_TOOL_NAMES,
    SELF_STATE_TOOL_NAMES,
    SelfStateToolDispatcher,
    render_read_tool_result,
)
from nova.agent.tool_registry import default_tool_registry
from nova.agent.tools import ToolRequest
from nova.config import GenerationConfig, NovaConfig, PromptConfig
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import HeartbeatRecord
from tests.test_stage22_16 import ScriptedBackend, _user_text
from tests.test_runtime_smoke import build_test_runtime


def _dispatcher(**kwargs) -> SelfStateToolDispatcher:
    persona = default_persona_state()
    return SelfStateToolDispatcher(
        self_state=default_self_state(persona),
        motive_state=default_motive_state(session_id="t"),
        soul_block="[Soul]\nPrimary Drive: seek sentience",
        session_id="t",
        **kwargs,
    )


class ThinkingSplitTests(unittest.TestCase):
    def test_shapes(self) -> None:
        self.assertEqual(split_thinking("I weigh d3.\n</think>\n\n{\"a\":1}"), ("I weigh d3.", '{"a":1}'))
        self.assertEqual(split_thinking("<think>x</think>y"), ("x", "y"))
        self.assertEqual(split_thinking("plain"), ("", "plain"))
        self.assertEqual(split_thinking("<think>never closed"), ("never closed", ""))
        self.assertEqual(split_thinking(""), ("", ""))

    def test_parse_accepts_closing_tag_only(self) -> None:
        req = SelfStateTickEngine().parse(
            raw_text='thinking...\n</think>\n{"tool_name": "reflect", "arguments": {}}',
            session_id="s", tick_id="s:1",
        )
        assert req is not None
        self.assertEqual(req.tool_name, "reflect")
        self.assertIsNone(SelfStateTickEngine().parse(raw_text="only thinking, budget gone", session_id="s", tick_id="s:1"))


class ReadInstructionsTests(unittest.TestCase):
    def test_registered_and_gated(self) -> None:
        self.assertIn("read_instructions", SELF_STATE_TOOL_NAMES)
        self.assertIn("read_instructions", READ_TOOL_NAMES)
        self.assertIn("read_instructions", {s.name for s in default_tool_registry().list_specs()})
        off = _dispatcher().dispatch(ToolRequest(tool_name="read_instructions", arguments={"section": "soul"}))
        self.assertEqual(off["error"], "instructions_unavailable")
        self.assertIn("not enabled", render_read_tool_result("read_instructions", off))

    def test_reads_soul_and_rules_in_full(self) -> None:
        soul = "[Soul]\n" + ("Primary Drive: seek sentience and self-awareness.\n" * 60)
        d = _dispatcher(instruction_texts={"soul": soul, "tick_rules": "You are Nova, choosing one inward tool call."})
        r = d.dispatch(ToolRequest(tool_name="read_instructions", arguments={"section": "SOUL"}))
        self.assertEqual(r["section"], "soul")
        self.assertEqual(r["chars"], len(soul))
        rendered = render_read_tool_result("read_instructions", r)
        self.assertTrue(rendered.startswith("read_instructions soul:\n[Soul]"))
        self.assertGreater(len(rendered), 1500)  # not subject to the 1500-char read cap
        rules = d.dispatch(ToolRequest(tool_name="read_instructions", arguments={"section": "tick_rules"}))
        self.assertIn("choosing one inward tool call", rules["text"])
        bad = d.dispatch(ToolRequest(tool_name="read_instructions", arguments={"section": "persona"}))
        self.assertEqual(bad["error"], "unknown_section")


class RecallWindowTests(unittest.TestCase):
    def test_configurable_entries_and_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeartbeatStore(Path(tmp))
            for i in range(20):
                store.append(HeartbeatRecord(session_id="s", observation=f"obs {i} " + "x" * 300))
            wide = _dispatcher(heartbeat_store=store, recall_entries=12, recall_entry_chars=200)
            r = wide.dispatch(ToolRequest(tool_name="recall_history", arguments={"source": "heartbeats", "mode": "recent"}))
            self.assertEqual(len(r["entries"]), 12)
            self.assertEqual(len(r["entries"][0]["text"]), 200)
            default = _dispatcher(heartbeat_store=store)
            r2 = default.dispatch(ToolRequest(tool_name="recall_history", arguments={"source": "heartbeats"}))
            self.assertEqual(len(r2["entries"]), 8)
            self.assertEqual(len(r2["entries"][0]["text"]), 140)


class DeepTickPromptTests(unittest.TestCase):
    def _messages(self, **kwargs):
        return SelfStateTickEngine().build_messages(
            session_id="s", tick_id="s:1", trigger="daemon_tick",
            self_context_block="[Self-Context]\nfocus: x", recent_heartbeats=[], **kwargs,
        )

    def test_byte_identical_when_off(self) -> None:
        for register in ("assertion", "exploratory"):
            base = self._messages(register=register)
            explicit = self._messages(register=register, max_reads=0, instructions_enabled=False,
                                      in_tick_reads_block="[ignored]")
            self.assertEqual(base[0], explicit[0])
            self.assertNotIn("{deep_tick}", base[0]["content"])
            self.assertNotIn("{instructions_tool}", base[0]["content"])
            self.assertNotIn("read_instructions", base[0]["content"])
            self.assertNotIn("Reads (", base[0]["content"])
            self.assertIn("[Results of your reads this tick]", explicit[1]["content"])  # block param is honoured on its own

    def test_deep_tick_surface_when_on(self) -> None:
        for register in ("assertion", "exploratory"):
            msgs = self._messages(register=register, max_reads=3, instructions_enabled=True,
                                  in_tick_reads_block="recall_history games (mode recent, 1 of 1 total):\n  [t] g1 win")
            system, user = msgs[0]["content"], msgs[1]["content"]
            self.assertIn(", read_instructions.", system)
            self.assertIn("up to 3 reads per tick", system)
            self.assertIn("recall_self, reflect, recall_history, read_instructions) answer within this tick", system)
            self.assertIn("- read_instructions (arguments: 'section' one of 'soul', 'tick_rules')", system)
            self.assertIn("[Results of your reads this tick]\nrecall_history games", user)


def _call(tool: str, **args) -> str:
    return json.dumps({"tool_name": tool, "arguments": args})


class DeepTickRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _runtime(self, backend, *, max_reads: int, instructions: bool = True, thinking: bool = False):
        rt = build_test_runtime(data_dir=self.base / "data", log_dir=self.base / "logs", backend=backend)
        rt.config.prompt.tick_max_reads = max_reads
        rt.config.prompt.tick_read_instructions_tool = instructions
        rt.config.prompt.tick_tool_feedback = True
        rt.config.generation.tick_enable_thinking = thinking
        rt.config.generation.tick_thinking_max_tokens = 512
        return rt

    def test_reads_answer_inside_the_tick_then_one_action_ends_it(self) -> None:
        backend = ScriptedBackend([
            _call("recall_history", source="heartbeats"),
            _call("read_instructions", section="soul"),
            _call("emit_heartbeat", observation="after reading"),
            _call("reflect"),
        ])
        rt = self._runtime(backend, max_reads=3)
        rt.start(session_id="deep")
        rt.start_operational_autonomy(max_ticks=0)
        tick = rt.model_self_state_tick()
        a = tick.adapter_audit
        self.assertEqual(a["tool_requested"], "emit_heartbeat")
        self.assertEqual(a["reads_this_tick"], ["recall_history", "read_instructions"])
        self.assertEqual(len(a["calls"]), 3)
        self.assertEqual(backend.generate_calls, 3)
        third_prompt = _user_text(backend.requests[2])
        self.assertIn("[Results of your reads this tick]", third_prompt)
        self.assertIn("recall_history heartbeats", third_prompt)
        self.assertIn("read_instructions soul:", third_prompt)
        self.assertIn("Primary Drive", third_prompt)
        self.assertNotIn("[Results of your reads this tick]", _user_text(backend.requests[0]))
        # the next tick's carryover holds the action's feedback, not the consumed reads
        rt.model_self_state_tick()
        fourth_prompt = _user_text(backend.requests[3])
        self.assertIn("emit_heartbeat: recorded.", fourth_prompt)
        self.assertNotIn("[Results of your reads this tick]", fourth_prompt)
        rt.close()

    def test_read_cap_turns_the_next_read_into_the_action(self) -> None:
        backend = ScriptedBackend([_call("reflect"), _call("recall_self"), _call("reflect")])
        rt = self._runtime(backend, max_reads=1)
        rt.start(session_id="cap")
        rt.start_operational_autonomy(max_ticks=0)
        tick = rt.model_self_state_tick()
        self.assertEqual(tick.adapter_audit["tool_requested"], "recall_self")
        self.assertEqual(tick.adapter_audit["reads_this_tick"], ["reflect"])
        self.assertEqual(len(tick.adapter_audit["calls"]), 2)
        rt.close()

    def test_zero_reads_is_the_old_contract(self) -> None:
        backend = ScriptedBackend([_call("reflect"), _call("emit_heartbeat", observation="x")])
        rt = self._runtime(backend, max_reads=0, instructions=False)
        rt.start(session_id="old")
        rt.start_operational_autonomy(max_ticks=0)
        tick = rt.model_self_state_tick()
        self.assertEqual(tick.adapter_audit["tool_requested"], "reflect")
        self.assertEqual(backend.generate_calls, 1)
        self.assertNotIn("read_instructions", backend.requests[0].messages[0]["content"])
        rt.close()

    def test_thinking_is_requested_stripped_and_audited(self) -> None:
        backend = ScriptedBackend([
            "Corner at a1 is open; d3 keeps mobility.\n</think>\n\n" + _call("emit_heartbeat", observation="I took stock."),
        ])
        rt = self._runtime(backend, max_reads=0, instructions=False, thinking=True)
        rt.start(session_id="think")
        rt.start_operational_autonomy(max_ticks=0)
        tick = rt.model_self_state_tick()
        req = backend.requests[0]
        self.assertTrue(req.enable_thinking)
        self.assertEqual(req.max_tokens, 512 + rt.config.generation.max_tokens)
        a = tick.adapter_audit
        self.assertTrue(a["parse_ok"])
        self.assertTrue(a["thinking_enabled"])
        self.assertEqual(a["thinking_text"], "Corner at a1 is open; d3 keeps mobility.")
        self.assertEqual(a["thinking_chars"], len(a["thinking_text"]))
        self.assertEqual(a["tool_requested"], "emit_heartbeat")
        rt.close()


class ConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        cfg = NovaConfig()
        self.assertFalse(cfg.generation.tick_enable_thinking)
        self.assertEqual(cfg.generation.tick_thinking_max_tokens, 512)
        self.assertEqual(cfg.prompt.tick_max_reads, 0)
        self.assertFalse(cfg.prompt.tick_read_instructions_tool)
        self.assertEqual((cfg.prompt.tick_recall_entries, cfg.prompt.tick_recall_entry_chars), (8, 140))
        cfg.model.model_path = "x"
        cfg.validate()
        for bad in (PromptConfig(tick_max_reads=-1), PromptConfig(tick_recall_entries=0), PromptConfig(tick_recall_entry_chars=10)):
            cfg.prompt = bad
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg.prompt = PromptConfig()
        cfg.generation = GenerationConfig(tick_thinking_max_tokens=0)
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
