"""Tests for Phase 22 Stage 22.16 — feedback closure and record-keeping."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nova.agent.exploration import (
    ExplorationController,
    ExplorationJournal,
    ExplorationStore,
    VALID_CLOSE_REASONS,
)
from nova.agent.motive import default_motive_state
from nova.agent.self_state_tools import (
    PARSE_FAILURE_FEEDBACK,
    SelfStateToolDispatcher,
    render_tool_feedback,
)
from nova.agent.heartbeat import SelfModelProposalStore
from nova.agent.tools import ToolRequest
from nova.config import NovaConfig, PromptConfig
from nova.daemon import NovaDaemon
from nova.persona.defaults import default_persona_state, default_self_state
from nova.persona.store import JsonSelfStateStore
from nova.types import GenerationRequest, GenerationResult
from tests.test_runtime_smoke import FakeBackend, build_test_runtime


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class FeedbackRendererTests(unittest.TestCase):
    def test_every_outcome_has_a_sentence(self) -> None:
        now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(render_tool_feedback("emit_heartbeat", {}), "emit_heartbeat: recorded.")
        self.assertIn(
            "current_focus revised and applied",
            render_tool_feedback("update_self_model", {"proposed_field": "current_focus", "outcome": "applied"}),
        )
        limited = render_tool_feedback(
            "update_self_model",
            {"proposed_field": "current_focus", "outcome": "rate_limited", "rate_limited_remaining_seconds": 1500},
            now=now,
        )
        self.assertIn("NOT revised", limited)
        self.assertIn("opens again in 25 min (at 12:25 UTC)", limited)
        self.assertIn(
            "queued for review",
            render_tool_feedback("update_self_model", {"proposed_field": "identity_summary", "outcome": "queued_for_operator"}),
        )
        self.assertIn("changes nothing until an operator applies it", render_tool_feedback("propose_instruction_update", {}))
        self.assertIn('opened "Why c4 loses the corner" (budget 12 ticks)',
                      render_tool_feedback("enter_exploration", {"topic": "Why c4 loses the corner", "max_ticks": 12}))
        self.assertIn("exported to your claim ladder",
                      render_tool_feedback("close_exploration", {}, export={"status": "exported"}))
        self.assertIn("NOT exported: too close to an existing record (0.74 overlap)",
                      render_tool_feedback("close_exploration", {}, export={"status": "duplicate", "overlap": 0.7412}))
        self.assertIn("NOT exported: unsupported_claim:unsupported_desire",
                      render_tool_feedback("close_exploration", {}, export={"status": "rejected", "reasons": ["unsupported_claim:unsupported_desire"]}))
        self.assertEqual(render_tool_feedback("enter_exploration", None, error="exploration already open"),
                         "enter_exploration failed: exploration already open")
        # reads and play still go through the read renderer
        self.assertTrue(render_tool_feedback("reflect", {"current_focus": "x"}).startswith("reflect:"))


# ---------------------------------------------------------------------------
# update_self_model outcome fields
# ---------------------------------------------------------------------------


class _RecordingStateStore:
    def __init__(self) -> None:
        self.saved = 0

    def save(self, state) -> None:  # noqa: ANN001
        self.saved += 1


class UpdateSelfModelOutcomeTests(unittest.TestCase):
    def test_outcome_and_remaining_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            persona = default_persona_state()
            dispatcher = SelfStateToolDispatcher(
                self_state=default_self_state(persona),
                motive_state=default_motive_state(session_id="s"),
                soul_block="[Soul]",
                session_id="s",
                proposal_store=SelfModelProposalStore(Path(tmp)),
                self_state_store=_RecordingStateStore(),
                self_model_writes_enabled=True,
                revision_min_seconds=3600,
            )
            first = dispatcher.dispatch(ToolRequest(tool_name="update_self_model",
                                                    arguments={"field": "current_focus", "value": "A", "rationale": "r"}))
            self.assertEqual(first["outcome"], "applied")
            second = dispatcher.dispatch(ToolRequest(tool_name="update_self_model",
                                                     arguments={"field": "current_focus", "value": "B", "rationale": "r"}))
            self.assertEqual(second["outcome"], "rate_limited")
            self.assertGreater(second["rate_limited_remaining_seconds"], 3500)
            gated = dispatcher.dispatch(ToolRequest(tool_name="update_self_model",
                                                    arguments={"field": "identity_summary", "value": "C", "rationale": "r"}))
            self.assertEqual(gated["outcome"], "queued_for_operator")


# ---------------------------------------------------------------------------
# Runtime: carryover of every outcome, persistence, audit, stranded closes
# ---------------------------------------------------------------------------


class ScriptedBackend(FakeBackend):
    """Returns the scripted raw texts in order, then repeats the last one."""

    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = list(outputs)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        self.requests.append(request)
        text = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return GenerationResult(model_id=request.model_id, raw_text=text, finish_reason="stop",
                                prompt_tokens=10, completion_tokens=12, latency_ms=7, metadata={})


def _user_text(request: GenerationRequest) -> str:
    return "\n".join(m.get("content", "") for m in (request.messages or []) if m.get("role") == "user")


def _focus(value: str) -> str:
    return json.dumps({"tool_name": "update_self_model",
                       "arguments": {"field": "current_focus", "value": value, "rationale": "r"}})


class TickFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _runtime(self, backend, *, feedback: bool, entries: int = 2, log_prompt: bool = False):
        rt = build_test_runtime(data_dir=self.base / "data", log_dir=self.base / "logs", backend=backend)
        rt.config.prompt.tick_tool_feedback = feedback
        rt.config.prompt.tick_carryover_entries = entries
        rt.config.prompt.tick_log_prompt_text = log_prompt
        rt.config.self_model.nova_writable_inquiry_fields = True
        rt.config.self_model.revision_min_seconds = 3600
        return rt

    def test_applied_then_rate_limited_feedback_reaches_her(self) -> None:
        backend = ScriptedBackend([_focus("A"), _focus("B"), "not json", _focus("C")])
        rt = self._runtime(backend, feedback=True, entries=3)
        rt.start(session_id="fb")
        rt.start_operational_autonomy(max_ticks=0)
        t1 = rt.model_self_state_tick()
        self.assertEqual(t1.adapter_audit["tool_result"]["outcome"], "applied")
        t2 = rt.model_self_state_tick()
        self.assertEqual(t2.adapter_audit["tool_result"]["outcome"], "rate_limited")
        prompt_2 = _user_text(backend.requests[1])
        self.assertIn("update_self_model: current_focus revised and applied.", prompt_2)
        t3 = rt.model_self_state_tick()  # "not json" -> parse failure
        self.assertFalse(t3.adapter_audit["parse_ok"])
        prompt_3 = _user_text(backend.requests[2])
        self.assertIn("current_focus NOT revised", prompt_3)
        self.assertIn("opens again in", prompt_3)
        rt.model_self_state_tick()
        prompt_4 = _user_text(backend.requests[3])
        self.assertIn(PARSE_FAILURE_FEEDBACK, prompt_4)
        self.assertIn("It began: not json", prompt_4)
        self.assertIn("[Results of your recent tool calls]", prompt_4)
        rt.close()

    def test_feedback_off_keeps_22_15_behaviour(self) -> None:
        backend = ScriptedBackend([_focus("A"), _focus("B")])
        rt = self._runtime(backend, feedback=False)
        rt.start(session_id="off")
        rt.start_operational_autonomy(max_ticks=0)
        rt.model_self_state_tick()
        rt.model_self_state_tick()
        self.assertNotIn("[Results of your recent tool calls]", _user_text(backend.requests[1]))
        rt.close()

    def test_carryover_persists_across_restart_and_respects_cap(self) -> None:
        backend = ScriptedBackend([_focus("A"), _focus("B"), _focus("C"), _focus("D")])
        rt = self._runtime(backend, feedback=True, entries=2)
        rt.start(session_id="p1")
        rt.start_operational_autonomy(max_ticks=0)
        for _ in range(4):
            rt.model_self_state_tick()
        self.assertEqual(len(rt._tick_read_results), 2)
        path = self.base / "data" / "tick_carryover.json"
        self.assertTrue(path.exists())
        rt.close()
        rt2 = self._runtime(ScriptedBackend([_focus("E")]), feedback=True, entries=2)
        rt2.start(session_id="p2")
        self.assertEqual(len(rt2._tick_read_results), 2)
        self.assertEqual(rt2._tick_read_results, rt._tick_read_results)
        rt2.close()

    def test_audit_carries_generation_facts_and_prompt(self) -> None:
        backend = ScriptedBackend([_focus("A")])
        rt = self._runtime(backend, feedback=True, log_prompt=True)
        rt.start(session_id="audit")
        rt.start_operational_autonomy(max_ticks=0)
        tick = rt.model_self_state_tick()
        a = tick.adapter_audit
        self.assertEqual(a["finish_reason"], "stop")
        self.assertEqual(a["completion_tokens"], 12)
        self.assertEqual(a["latency_ms"], 7)
        self.assertEqual(len(a["prompt_sha256"]), 64)
        self.assertIn("self_context", a["prompt_blocks"])
        self.assertEqual([m["role"] for m in a["prompt_text"]], ["system", "user"])
        rt.close()
        rt_off = self._runtime(ScriptedBackend([_focus("A")]), feedback=True, log_prompt=False)
        rt_off.start(session_id="audit2")
        rt_off.start_operational_autonomy(max_ticks=0)
        self.assertNotIn("prompt_text", rt_off.model_self_state_tick().adapter_audit)
        rt_off.close()

    def test_stranded_exploration_is_closed_on_next_session_start(self) -> None:
        backend = ScriptedBackend([json.dumps({"tool_name": "emit_heartbeat", "arguments": {"observation": "x"}})])
        rt = self._runtime(backend, feedback=True)
        rt.start(session_id="day1")
        rt.start_operational_autonomy(max_ticks=0)
        rt.start_exploration(topic="left open overnight", rationale="test", origin="operator")
        self.assertIsNotNone(rt.exploration_controller.active_exploration("day1"))
        rt.close()
        rt2 = self._runtime(ScriptedBackend([_focus("A")]), feedback=True)
        rt2.start(session_id="day2")
        records = rt2.exploration_controller.store.list_all()
        self.assertEqual([r.status for r in records], ["closed"])
        self.assertEqual(records[0].close_reason, "session_end")
        self.assertIn("session_end", VALID_CLOSE_REASONS)
        rt2.start_operational_autonomy(max_ticks=0)
        rt2.model_self_state_tick()
        self.assertNotIn("stranded", rt2._exploration_history_block())
        rt2.close()


class ConfigAndDaemonTests(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        cfg = NovaConfig()
        self.assertFalse(cfg.prompt.tick_tool_feedback)
        self.assertEqual(cfg.prompt.tick_carryover_entries, 2)
        self.assertFalse(cfg.prompt.tick_log_prompt_text)
        cfg.model.model_path = "x"
        cfg.validate()
        cfg.prompt = PromptConfig(tick_carryover_entries=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_tick_failure_is_logged_and_counted(self) -> None:
        class Boom:
            def resume_exploration(self):  # noqa: ANN201
                return None

            def model_self_state_tick(self, trigger):  # noqa: ANN001, ANN201
                raise RuntimeError("backend exploded")

        with tempfile.TemporaryDirectory() as tmp:
            daemon = NovaDaemon(runtime=Boom(), socket_path=Path(tmp) / "nova.sock", tick_interval_seconds=0.01, session_id="t")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                thread = threading.Thread(target=daemon._tick_loop, daemon=True)
                thread.start()
                deadline = time.time() + 2
                while daemon._tick_errors == 0 and time.time() < deadline:
                    time.sleep(0.01)
                daemon._stop_event.set()
                thread.join(timeout=2)
            self.assertGreaterEqual(daemon._tick_errors, 1)
            self.assertIn("tick failed", err.getvalue())
            self.assertIn("backend exploded", err.getvalue())


if __name__ == "__main__":
    unittest.main()
