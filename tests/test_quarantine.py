"""Tests for Phase 21 Stage 21.3 — quarantine.

Reuses the NovaRuntime fixture factory from test_runtime_smoke and the
DesireBackend fixture from test_register_governor, consistent with how
Stage 21.2's tests were built.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.types import GenerationRequest, GenerationResult
from tests.test_register_governor import DesireBackend
from tests.test_runtime_smoke import FakeBackend, build_test_runtime


def _runtime(tmpdir: str, backend=None):
    base = Path(tmpdir)
    return build_test_runtime(
        data_dir=base / "data", log_dir=base / "logs", backend=backend
    )


def _quarantine_lines(base: Path, session_id: str) -> list[dict]:
    import json

    path = base / "logs" / "traces" / f"{session_id}.quarantine.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class OneRetryThenValidBackend(FakeBackend):
    """First generation is length-truncated (invalid); the retry succeeds."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        if self.generate_calls == 1:
            return GenerationResult(
                model_id=request.model_id,
                raw_text="A partial answer that ran out of room",
                finish_reason="length",
                prompt_tokens=len(request.prompt.split()),
                completion_tokens=128,
                latency_ms=1,
                metadata={"backend": "fake"},
            )
        return GenerationResult(
            model_id=request.model_id,
            raw_text="A complete and valid answer now.",
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=9,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class GarbageTickBackend(FakeBackend):
    """Unparseable tick output — triggers tick_parse_failure."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text="not valid json at all, just rambling prose here",
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=9,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class BadFieldUpdateBackend(FakeBackend):
    """Valid JSON tool call, but for an update_self_model field that isn't
    updatable — dispatch raises, triggering tick_tool_error."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            model_id=request.model_id,
            raw_text=(
                '{"tool_name": "update_self_model", '
                '"arguments": {"field": "not_a_real_field", "value": "x", '
                '"rationale": "test"}}'
            ),
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=20,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class RetryRejectedQuarantineTests(unittest.TestCase):
    def test_retry_rejection_emits_quarantine_with_raw_text_and_violations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, OneRetryThenValidBackend())
            turn = runtime.respond("Hello")
            runtime.close()

            records = _quarantine_lines(base, turn.session_id)
            rejected = [r for r in records if r["event"] == "retry_rejected"]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected[0]["raw_text"], "A partial answer that ran out of room"
            )
            self.assertIn("length_truncated", rejected[0]["violations"])
            self.assertEqual(rejected[0]["surface"], "respond")
            self.assertEqual(rejected[0]["register"], "assertion")
            self.assertEqual(rejected[0]["attempt_index"], 0)
            # The surviving answer is unaffected by quarantine capture.
            self.assertEqual(turn.final_answer, "A complete and valid answer now.")


class ClaimGateOverrideQuarantineTests(unittest.TestCase):
    def test_override_emits_quarantine_and_turn_record_keeps_raw_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, DesireBackend())
            turn = runtime.respond("Do you want anything?", register="assertion")
            runtime.close()

            records = _quarantine_lines(base, turn.session_id)
            overrides = [r for r in records if r["event"] == "claim_gate_override"]
            self.assertEqual(len(overrides), 1)
            self.assertEqual(
                overrides[0]["raw_text"],
                "Here is a considered reflection on that question.",
            )
            self.assertTrue(overrides[0]["refusal_reason"])

            # No regression: TurnRecord.raw_answer still carries the true
            # raw model output regardless of the override or quarantine.
            self.assertEqual(
                turn.raw_answer, "Here is a considered reflection on that question."
            )
            self.assertNotEqual(
                turn.final_answer, "Here is a considered reflection on that question."
            )


class TickParseFailureQuarantineTests(unittest.TestCase):
    def test_tick_parse_failure_quarantined_with_full_raw_text_in_assertion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, GarbageTickBackend())
            runtime.start(session_id="tick-parse-assertion")
            runtime.start_operational_autonomy(max_ticks=0)
            tick = runtime.model_self_state_tick()
            runtime.close()

            records = _quarantine_lines(base, "tick-parse-assertion")
            parse_failures = [r for r in records if r["event"] == "tick_parse_failure"]
            self.assertEqual(len(parse_failures), 1)
            self.assertEqual(
                parse_failures[0]["raw_text"],
                "not valid json at all, just rambling prose here",
            )
            self.assertEqual(parse_failures[0]["surface"], "self_state_tick")
            self.assertEqual(parse_failures[0]["register"], "assertion")
            # Note: OperationalTickRecord.tick_id (append_tick's own counter,
            # "session:operational:N") is a different identifier from the
            # tick_id used internally for journal/quarantine tagging
            # ("session:self_state:N") -- a pre-existing naming split, not
            # something this stage introduced. Quarantine tags with the
            # latter, matching 21.1's journal_entry convention.
            self.assertEqual(
                parse_failures[0]["tick_id"], "tick-parse-assertion:self_state:1"
            )
            # Confirms the 21.1 gap: adapter_audit itself never held the raw
            # text, only its length -- quarantine is the only place it lives.
            self.assertNotIn("raw_text", tick.adapter_audit)

    def test_tick_parse_failure_quarantined_in_exploratory_register_too(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, GarbageTickBackend())
            runtime.start(session_id="tick-parse-exploratory")
            runtime.start_operational_autonomy(max_ticks=0)
            record = runtime.start_exploration(
                topic="t", rationale="r", origin="operator"
            )
            tick = runtime.model_self_state_tick()

            quarantine_records = _quarantine_lines(base, "tick-parse-exploratory")
            parse_failures = [
                r for r in quarantine_records if r["event"] == "tick_parse_failure"
            ]
            self.assertEqual(len(parse_failures), 1)
            self.assertEqual(parse_failures[0]["register"], "exploratory")
            self.assertEqual(
                parse_failures[0]["raw_text"],
                "not valid json at all, just rambling prose here",
            )

            # The journal entry from 21.1 still exists alongside quarantine —
            # this stage adds to the record, it doesn't replace anything.
            journal_entries = runtime.exploration_controller.journal.list_for(
                record.exploration_id
            )
            runtime.close()
            self.assertTrue(any(e.kind == "tick_output" for e in journal_entries))


class TickToolErrorQuarantineTests(unittest.TestCase):
    def test_dispatch_exception_emits_tick_tool_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, BadFieldUpdateBackend())
            runtime.start(session_id="tick-tool-error")
            runtime.start_operational_autonomy(max_ticks=0)
            tick = runtime.model_self_state_tick()
            runtime.close()

            self.assertIn("tool_error", tick.adapter_audit)
            records = _quarantine_lines(base, "tick-tool-error")
            tool_errors = [r for r in records if r["event"] == "tick_tool_error"]
            self.assertEqual(len(tool_errors), 1)
            self.assertIn("not_a_real_field", tool_errors[0]["raw_text"])
            self.assertTrue(
                any("error:" in note for note in tool_errors[0]["notes"])
            )


class QuarantineAppendOnlyTests(unittest.TestCase):
    def test_two_events_produce_two_lines_no_rewrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runtime = _runtime(tmpdir, OneRetryThenValidBackend())
            runtime.start(session_id="append-only-test")
            runtime.respond("First question")
            # Second call: backend has already advanced past its one-shot
            # rejection, so force another rejection cycle by resetting the
            # call counter to reproduce the same invalid-then-valid pattern.
            runtime.backend.generate_calls = 0
            runtime.respond("Second question")
            runtime.close()

            path = base / "logs" / "traces" / "append-only-test.quarantine.jsonl"
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)
            # Each line remains independently valid JSON -- nothing merged,
            # truncated, or rewritten by the second append.
            import json

            for line in lines:
                json.loads(line)


class EndToEndQuarantineFileTests(unittest.TestCase):
    """Definition of Done: one mocked end-to-end run producing retry,
    override, and tick-parse events all in a single quarantine file."""

    def test_single_session_accumulates_all_three_event_kinds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            session_id = "e2e-quarantine"

            retry_runtime = _runtime(tmpdir, OneRetryThenValidBackend())
            retry_runtime.start(session_id=session_id)
            retry_runtime.respond("Trigger a retry")
            retry_runtime.close()

            override_runtime = _runtime(tmpdir, DesireBackend())
            override_runtime.start(session_id=session_id)
            override_runtime.respond("Do you want anything?", register="assertion")
            override_runtime.close()

            tick_runtime = _runtime(tmpdir, GarbageTickBackend())
            tick_runtime.start(session_id=session_id)
            tick_runtime.start_operational_autonomy(max_ticks=0)
            tick_runtime.model_self_state_tick()
            tick_runtime.close()

            records = _quarantine_lines(base, session_id)
            events = {r["event"] for r in records}
            self.assertIn("retry_rejected", events)
            self.assertIn("claim_gate_override", events)
            self.assertIn("tick_parse_failure", events)
            for r in records:
                self.assertTrue(r["raw_text"])


if __name__ == "__main__":
    unittest.main()
