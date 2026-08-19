"""Phase 22 Stage 22.10 — tick-surface self-history access.

Part A: read-tool results carry over into later tick prompts.
Part B: recall_history, a bounded deterministic read over Nova's own record.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from nova.agent.heartbeat import HeartbeatStore
from nova.agent.motive import default_motive_state
from nova.agent.self_state_tick import SelfStateTickEngine
from nova.agent.self_state_tools import (
    RECALL_HISTORY_COUNT,
    RECALL_HISTORY_ENTRY_CHARS,
    READ_TOOL_NAMES,
    RENDER_READ_RESULT_MAX_CHARS,
    SelfStateToolDispatcher,
    render_read_tool_result,
)
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import HeartbeatRecord


def _dispatcher(tmp: Path, **kwargs) -> SelfStateToolDispatcher:
    persona = default_persona_state()
    return SelfStateToolDispatcher(
        self_state=default_self_state(persona),
        motive_state=default_motive_state(session_id="s1"),
        soul_block="",
        session_id="s1",
        **kwargs,
    )


def _heartbeat_store(tmp: Path, count: int) -> HeartbeatStore:
    store = HeartbeatStore(tmp)
    for i in range(count):
        store.append(
            HeartbeatRecord(
                heartbeat_id=f"hb{i}",
                timestamp=f"2026-07-{(i % 28) + 1:02d}T00:00:{i % 60:02d}+00:00",
                session_id="s1",
                observation=f"observation number {i} with distinct content {i}",
            )
        )
    return store


class RecallHistoryDispatchTests(unittest.TestCase):
    def test_heartbeats_recent_returns_last_window_with_true_total(self):
        with TemporaryDirectory() as tmp:
            store = _heartbeat_store(Path(tmp), 20)
            d = _dispatcher(Path(tmp), heartbeat_store=store)
            result = d.recall_history(source="heartbeats")
            self.assertEqual(result["total"], 20)
            self.assertEqual(len(result["entries"]), RECALL_HISTORY_COUNT)
            self.assertIn("observation number 19", result["entries"][-1]["text"])

    def test_earliest_mode_returns_first_window(self):
        with TemporaryDirectory() as tmp:
            store = _heartbeat_store(Path(tmp), 20)
            d = _dispatcher(Path(tmp), heartbeat_store=store)
            result = d.recall_history(source="heartbeats", mode="earliest")
            self.assertIn("observation number 0", result["entries"][0]["text"])
            self.assertEqual(len(result["entries"]), RECALL_HISTORY_COUNT)

    def test_sample_mode_is_deterministic_and_spans(self):
        with TemporaryDirectory() as tmp:
            store = _heartbeat_store(Path(tmp), 30)
            d = _dispatcher(Path(tmp), heartbeat_store=store)
            first = d.recall_history(source="heartbeats", mode="sample")
            second = d.recall_history(source="heartbeats", mode="sample")
            self.assertEqual(first, second)
            self.assertIn("observation number 0", first["entries"][0]["text"])
            self.assertIn("observation number 29", first["entries"][-1]["text"])

    def test_around_mode_windows_on_date(self):
        with TemporaryDirectory() as tmp:
            store = _heartbeat_store(Path(tmp), 28)  # days 01..28
            d = _dispatcher(Path(tmp), heartbeat_store=store)
            result = d.recall_history(source="heartbeats", around="2026-07-15")
            self.assertEqual(result["mode"], "around")
            self.assertEqual(result["around"], "2026-07-15")
            texts = " ".join(e["text"] for e in result["entries"])
            self.assertIn("observation number 14", texts)  # day 15

    def test_entry_text_is_capped(self):
        with TemporaryDirectory() as tmp:
            store = HeartbeatStore(tmp)
            store.append(
                HeartbeatRecord(
                    heartbeat_id="hb-long",
                    timestamp="2026-07-01T00:00:00+00:00",
                    session_id="s1",
                    observation="x" * 500,
                )
            )
            d = _dispatcher(Path(tmp), heartbeat_store=store)
            result = d.recall_history(source="heartbeats")
            self.assertEqual(
                len(result["entries"][0]["text"]), RECALL_HISTORY_ENTRY_CHARS
            )

    def test_unknown_source_and_mode_return_error_dicts(self):
        with TemporaryDirectory() as tmp:
            d = _dispatcher(Path(tmp))
            self.assertEqual(
                d.recall_history(source="journal")["error"], "unknown_source"
            )
            self.assertEqual(
                d.recall_history(source="heartbeats", mode="random")["error"],
                "unknown_mode",
            )

    def test_findings_without_store_degrades_with_note(self):
        with TemporaryDirectory() as tmp:
            d = _dispatcher(Path(tmp))
            result = d.recall_history(source="findings")
            self.assertEqual(result["total"], 0)
            self.assertIn("not available", result["note"])

    def test_findings_render_rung_and_text(self):
        ladder = SimpleNamespace(
            list_active=lambda: [
                SimpleNamespace(
                    created_at="2026-07-18T00:00:00+00:00",
                    rung=1,
                    claim_text="a recorded pattern about recalibration",
                )
            ]
        )
        with TemporaryDirectory() as tmp:
            d = _dispatcher(Path(tmp), claim_ladder_store=ladder)
            result = d.recall_history(source="findings")
            self.assertEqual(result["total"], 1)
            self.assertIn("(rung 1)", result["entries"][0]["text"])

    def test_explorations_expose_metadata_only(self):
        # Membrane: topics/dates/outcomes only — never findings text.
        store = SimpleNamespace(
            list_all=lambda: [
                SimpleNamespace(
                    opened_at="2026-08-01T00:00:00+00:00",
                    topic="a past topic",
                    close_reason="nova_close",
                    status="closed",
                )
            ]
        )
        controller = SimpleNamespace(store=store)
        with TemporaryDirectory() as tmp:
            d = _dispatcher(Path(tmp), exploration_controller=controller)
            result = d.recall_history(source="explorations")
            self.assertIn("a past topic", result["entries"][0]["text"])
            self.assertIn("closed: nova_close", result["entries"][0]["text"])

    def test_dispatch_routes_recall_history(self):
        from nova.agent.tools import ToolRequest

        with TemporaryDirectory() as tmp:
            store = _heartbeat_store(Path(tmp), 3)
            d = _dispatcher(Path(tmp), heartbeat_store=store)
            result = d.dispatch(
                ToolRequest(
                    tool_name="recall_history",
                    reason="test",
                    arguments={"source": "heartbeats"},
                )
            )
            self.assertEqual(result["source"], "heartbeats")


class RenderReadResultTests(unittest.TestCase):
    def test_recall_history_render_lists_entries_and_total(self):
        text = render_read_tool_result(
            "recall_history",
            {
                "source": "heartbeats",
                "mode": "sample",
                "total": 500,
                "entries": [
                    {"timestamp": "2026-07-11T21:59:57", "text": "the first one"}
                ],
            },
        )
        self.assertIn("recall_history heartbeats", text)
        self.assertIn("1 of 500 total", text)
        self.assertIn("[2026-07-11T21:59:57] the first one", text)

    def test_recall_self_and_reflect_render_compactly(self):
        recall = render_read_tool_result(
            "recall_self",
            {
                "self_state": {
                    "current_focus": "a focus",
                    "active_questions": ["q one?"],
                    "open_tensions": [],
                    "continuity_notes": ["n1"],
                },
                "recent_heartbeats": [
                    {"timestamp": "2026-08-19T00:00:00", "observation": "obs"}
                ],
            },
        )
        self.assertIn("focus: a focus", recall)
        self.assertIn("question: q one?", recall)
        reflect = render_read_tool_result(
            "reflect",
            {
                "current_focus": "a focus",
                "active_questions": ["q one?"],
                "open_tensions": ["t1"],
                "continuity_notes": [],
                "claim_posture": "evidence-first",
            },
        )
        self.assertIn("tension: t1", reflect)
        self.assertIn("claim_posture: evidence-first", reflect)

    def test_render_is_capped_and_unknown_tool_is_empty(self):
        big = render_read_tool_result(
            "recall_history",
            {
                "source": "heartbeats",
                "mode": "recent",
                "total": 8,
                "entries": [
                    {"timestamp": "t" * 19, "text": "y" * 140} for _ in range(8)
                ],
            },
        )
        self.assertLessEqual(len(big), RENDER_READ_RESULT_MAX_CHARS)
        self.assertEqual(render_read_tool_result("emit_heartbeat", {}), "")


class TickPromptCarryoverTests(unittest.TestCase):
    def setUp(self):
        self.engine = SelfStateTickEngine()

    def _messages(self, register="assertion", **kwargs):
        return self.engine.build_messages(
            session_id="s1",
            tick_id="t1",
            trigger="test",
            self_context_block="[Self-Context]",
            recent_heartbeats=[],
            register=register,
            **kwargs,
        )

    def test_block_rendered_with_provenance_when_present(self):
        user = self._messages(tool_results_block="recall_history heartbeats ...")[
            1
        ]["content"]
        self.assertIn("[Results of your recent tool calls]", user)
        self.assertIn("(you asked for these on earlier ticks)", user)
        self.assertIn("recall_history heartbeats", user)

    def test_block_absent_when_empty(self):
        user = self._messages()[1]["content"]
        self.assertNotIn("Results of your recent tool calls", user)

    def test_block_rendered_in_both_registers(self):
        for register in ("assertion", "exploratory"):
            user = self._messages(
                register=register, tool_results_block="marker-carryover"
            )[1]["content"]
            self.assertIn("marker-carryover", user)

    def test_menus_offer_recall_history_in_both_registers(self):
        for register in ("assertion", "exploratory"):
            system = self._messages(register=register)[0]["content"]
            self.assertIn("recall_history", system)

    def test_read_tools_promise_next_tick_delivery(self):
        system = self._messages()[0]["content"]
        normalized = " ".join(system.split())
        self.assertEqual(
            normalized.count("The result appears in your context on your next tick"),
            3,
        )

    def test_parse_accepts_recall_history(self):
        request = self.engine.parse(
            raw_text='{"tool_name": "recall_history", "arguments": {"source": "heartbeats"}}',
            session_id="s1",
            tick_id="t1",
        )
        self.assertIsNotNone(request)
        self.assertEqual(request.tool_name, "recall_history")
        self.assertIn("recall_history", READ_TOOL_NAMES)


if __name__ == "__main__":
    unittest.main()
