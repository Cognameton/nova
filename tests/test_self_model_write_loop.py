"""Tests for Phase 22 Stage 22.8 — closing the self-model write loop.

The stage's premise, measured live: self_state.json went unwritten for the
entire 26-day live run because update_self_model only ever queued an
approval-gated proposal and the approval path had no operator surface. These
tests pin the corrective — inquiry-class fields are Nova's to write, with
provenance and a revert path; assertion-class fields are untouched.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from nova.agent.heartbeat import HeartbeatStore, SelfModelProposalStore
from nova.agent.motive import default_motive_state
from nova.agent.self_context import SelfContextEngine
from nova.agent.self_state_tick import SelfStateTickEngine
from nova.agent.self_state_tools import (
    APPROVAL_GATED_SELF_STATE_FIELDS,
    NOVA_WRITABLE_SELF_STATE_FIELDS,
    SELF_MODEL_REVISION_MIN_SECONDS,
    SELF_MODEL_REVISION_MIN_TICKS,
    SelfStateToolDispatcher,
    _UPDATABLE_SELF_STATE_FIELDS,
    apply_proposal_to_self_state,
)
from nova.agent.tools import ToolRequest
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import HeartbeatRecord, SelfModelProposal, SelfState


class _RecordingSelfStateStore:
    """Minimal self-state store double — counts saves, keeps the last one."""

    def __init__(self) -> None:
        self.saves = 0
        self.last_saved: SelfState | None = None

    def save(self, self_state: SelfState) -> None:
        self.saves += 1
        self.last_saved = SelfState(**self_state.to_dict())


def _dispatcher(
    *,
    tmp: Path,
    self_state: SelfState | None = None,
    enabled: bool = True,
    revision_min_seconds: float = SELF_MODEL_REVISION_MIN_SECONDS,
    store: SelfModelProposalStore | None = None,
    state_store: _RecordingSelfStateStore | None = None,
) -> tuple[SelfStateToolDispatcher, SelfState, SelfModelProposalStore, _RecordingSelfStateStore]:
    persona = default_persona_state()
    ss = self_state if self_state is not None else default_self_state(persona)
    proposal_store = store if store is not None else SelfModelProposalStore(tmp)
    self_state_store = state_store if state_store is not None else _RecordingSelfStateStore()
    dispatcher = SelfStateToolDispatcher(
        self_state=ss,
        motive_state=default_motive_state(session_id="s1"),
        soul_block="[Soul]",
        session_id="s1",
        proposal_store=proposal_store,
        self_state_store=self_state_store,
        self_model_writes_enabled=enabled,
        revision_min_seconds=revision_min_seconds,
    )
    return dispatcher, ss, proposal_store, self_state_store


def _update(dispatcher: SelfStateToolDispatcher, field: str, value) -> dict:
    return dispatcher.dispatch(
        ToolRequest(
            tool_name="update_self_model",
            arguments={"field": field, "value": value, "rationale": "because"},
        )
    )


class FieldClassPartitionTests(unittest.TestCase):
    """A field added later must land in exactly one class, never neither."""

    def test_classes_partition_the_updatable_set(self) -> None:
        self.assertEqual(
            NOVA_WRITABLE_SELF_STATE_FIELDS | APPROVAL_GATED_SELF_STATE_FIELDS,
            _UPDATABLE_SELF_STATE_FIELDS,
        )

    def test_classes_are_disjoint(self) -> None:
        self.assertEqual(
            NOVA_WRITABLE_SELF_STATE_FIELDS & APPROVAL_GATED_SELF_STATE_FIELDS,
            frozenset(),
        )

    def test_the_frozen_live_fields_are_nova_writable(self) -> None:
        # These are the exact fields the live proposals tried to move
        # (current_focus x9, continuity_notes x6, active_questions x5) and
        # could not.
        for field in ("current_focus", "active_questions", "continuity_notes"):
            self.assertIn(field, NOVA_WRITABLE_SELF_STATE_FIELDS)

    def test_identity_and_preferences_stay_gated(self) -> None:
        for field in ("identity_summary", "stable_preferences", "relationship_notes"):
            self.assertIn(field, APPROVAL_GATED_SELF_STATE_FIELDS)

    def test_rate_limit_is_pinned_to_production_cadence(self) -> None:
        # Stage 22.2c's lesson: an independently-chosen number that happens
        # to collide with the cadence is how the exploration budget broke.
        self.assertEqual(SELF_MODEL_REVISION_MIN_SECONDS, SELF_MODEL_REVISION_MIN_TICKS * 300)


class InquiryClassWriteTests(unittest.TestCase):
    def test_inquiry_field_mutates_state_and_persists(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, store, state_store = _dispatcher(tmp=Path(tmp))
            result = _update(dispatcher, "current_focus", "tracing my own drift")

            self.assertEqual(ss.current_focus, "tracing my own drift")
            self.assertEqual(state_store.saves, 1)
            self.assertEqual(state_store.last_saved.current_focus, "tracing my own drift")
            self.assertTrue(result["applied"])
            self.assertTrue(result["auto_applied"])
            self.assertEqual(result["applied_by"], "nova")
            self.assertFalse(result["approval_required"])
            self.assertEqual(len(store.list_all()), 1)

    def test_prior_value_is_captured(self) -> None:
        with TemporaryDirectory() as tmp:
            start = SelfState(**{**default_self_state(default_persona_state()).to_dict(),
                                 "current_focus": "the old focus"})
            dispatcher, _ss, store, _st = _dispatcher(tmp=Path(tmp), self_state=start)
            _update(dispatcher, "current_focus", "the new focus")

            record = store.list_all()[0]
            self.assertEqual(record.prior_value, "the old focus")
            self.assertEqual(record.proposed_value, "the new focus")

    def test_list_valued_prior_is_copied_not_aliased(self) -> None:
        with TemporaryDirectory() as tmp:
            start = SelfState(**{**default_self_state(default_persona_state()).to_dict(),
                                 "active_questions": ["how does continuity stay stable?"]})
            dispatcher, ss, store, _st = _dispatcher(tmp=Path(tmp), self_state=start)
            _update(dispatcher, "active_questions", ["what changed while I was not looking?"])

            record = store.list_all()[0]
            self.assertEqual(record.prior_value, ["how does continuity stay stable?"])
            self.assertEqual(ss.active_questions, ["what changed while I was not looking?"])

    def test_disabled_config_reproduces_queue_only_behavior(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, store, state_store = _dispatcher(tmp=Path(tmp), enabled=False)
            result = _update(dispatcher, "current_focus", "should not land")

            self.assertEqual(ss.current_focus, default_self_state(default_persona_state()).current_focus)
            self.assertEqual(state_store.saves, 0)
            self.assertFalse(result["applied"])
            self.assertTrue(result["approval_required"])
            self.assertFalse(result["auto_applied"])
            self.assertEqual(len(store.list_all()), 1)


class AssertionClassUnchangedTests(unittest.TestCase):
    def test_assertion_field_is_queued_never_applied(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, store, state_store = _dispatcher(tmp=Path(tmp))
            before = ss.identity_summary
            result = _update(dispatcher, "identity_summary", "I am definitely awake")

            self.assertEqual(ss.identity_summary, before)
            self.assertEqual(state_store.saves, 0)
            self.assertTrue(result["approval_required"])
            self.assertFalse(result["applied"])
            self.assertFalse(result["auto_applied"])
            self.assertEqual(store.list_all()[0].applied_by, "")

    def test_unknown_field_still_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, _ss, _store, _st = _dispatcher(tmp=Path(tmp))
            with self.assertRaises(ValueError):
                _update(dispatcher, "not_a_field", "x")


class RevisionRateLimitTests(unittest.TestCase):
    def test_second_write_to_same_field_is_recorded_not_applied(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, store, _st = _dispatcher(tmp=Path(tmp))
            _update(dispatcher, "current_focus", "first")
            result = _update(dispatcher, "current_focus", "second")

            self.assertEqual(ss.current_focus, "first")
            self.assertFalse(result["applied"])
            self.assertIn("rate_limited", result["note"])
            # Recorded, never dropped — the wanting-to-move signal is the
            # evidence the frozen era destroyed.
            self.assertEqual(len(store.list_all()), 2)

    def test_different_field_in_the_same_window_still_applies(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, _store, _st = _dispatcher(tmp=Path(tmp))
            _update(dispatcher, "current_focus", "first")
            result = _update(dispatcher, "active_questions", ["a genuinely different field"])

            self.assertTrue(result["applied"])
            self.assertEqual(ss.active_questions, ["a genuinely different field"])

    def test_window_expiry_lets_the_field_move_again(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, store, _st = _dispatcher(tmp=Path(tmp))
            _update(dispatcher, "current_focus", "first")

            # Age the applied record past the window on disk.
            path = Path(tmp) / "self_model_proposals.jsonl"
            stale = (datetime.now(timezone.utc) - timedelta(seconds=SELF_MODEL_REVISION_MIN_SECONDS + 60)).isoformat()
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for record in records:
                record["applied_at"] = stale
                record["timestamp"] = stale
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

            result = _update(dispatcher, "current_focus", "second")
            self.assertTrue(result["applied"])
            self.assertEqual(ss.current_focus, "second")

    def test_zero_window_disables_pacing(self) -> None:
        with TemporaryDirectory() as tmp:
            dispatcher, ss, _store, _st = _dispatcher(tmp=Path(tmp), revision_min_seconds=0)
            _update(dispatcher, "current_focus", "first")
            result = _update(dispatcher, "current_focus", "second")
            self.assertTrue(result["applied"])
            self.assertEqual(ss.current_focus, "second")


class SharedApplyPathTests(unittest.TestCase):
    """The operator path and Nova's path must not drift."""

    def test_operator_apply_records_prior_value_and_attribution(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SelfModelProposalStore(Path(tmp))
            state_store = _RecordingSelfStateStore()
            ss = SelfState(**{**default_self_state(default_persona_state()).to_dict(),
                              "identity_summary": "the old summary"})
            proposal = SelfModelProposal(
                proposal_id="p1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id="s1",
                proposed_field="identity_summary",
                proposed_value="the reviewed summary",
                rationale="operator approved",
            )
            store.append(proposal)

            applied = apply_proposal_to_self_state(
                proposal=proposal,
                self_state=ss,
                self_state_store=state_store,
                proposal_store=store,
                applied_by="operator",
            )

            self.assertIsNotNone(applied)
            self.assertEqual(ss.identity_summary, "the reviewed summary")
            self.assertEqual(applied.prior_value, "the old summary")
            self.assertEqual(applied.applied_by, "operator")
            self.assertFalse(applied.auto_applied)
            self.assertEqual(state_store.saves, 1)

    def test_non_updatable_field_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SelfModelProposalStore(Path(tmp))
            ss = default_self_state(default_persona_state())
            proposal = SelfModelProposal(
                proposal_id="p2",
                proposed_field="stability_version",
                proposed_value=99,
            )
            self.assertIsNone(
                apply_proposal_to_self_state(
                    proposal=proposal,
                    self_state=ss,
                    self_state_store=_RecordingSelfStateStore(),
                    proposal_store=store,
                    applied_by="operator",
                )
            )


class ProposalStoreSchemaTests(unittest.TestCase):
    def test_pre_22_8_records_load_with_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "self_model_proposals.jsonl"
            # A verbatim-shaped record from the live store, which has no
            # prior_value / applied_by / auto_applied / note keys.
            path.write_text(json.dumps({
                "schema_version": "1.0",
                "proposal_id": "old1",
                "timestamp": "2026-07-22T12:00:00+00:00",
                "session_id": "nova-live-20260722",
                "proposed_field": "current_focus",
                "proposed_value": "Optimizing recalibration intervals",
                "rationale": "legacy",
                "approval_required": True,
                "applied": False,
                "applied_at": "",
            }) + "\n")

            record = SelfModelProposalStore(Path(tmp)).list_all()[0]
            self.assertEqual(record.proposal_id, "old1")
            self.assertIsNone(record.prior_value)
            self.assertEqual(record.applied_by, "")
            self.assertFalse(record.auto_applied)
            self.assertEqual(record.note, "")

    def test_last_applied_for_field_ignores_other_fields_and_pending(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SelfModelProposalStore(Path(tmp))
            store.append(SelfModelProposal(proposal_id="a", proposed_field="current_focus",
                                           applied=True, applied_at="2026-08-01T00:00:00+00:00"))
            store.append(SelfModelProposal(proposal_id="b", proposed_field="current_focus",
                                           applied=True, applied_at="2026-08-03T00:00:00+00:00"))
            store.append(SelfModelProposal(proposal_id="c", proposed_field="current_focus",
                                           applied=False, applied_at=""))
            store.append(SelfModelProposal(proposal_id="d", proposed_field="open_tensions",
                                           applied=True, applied_at="2026-08-05T00:00:00+00:00"))

            latest = store.last_applied_for_field("current_focus")
            self.assertEqual(latest.proposal_id, "b")
            self.assertIsNone(store.last_applied_for_field("continuity_notes"))


class StratifiedHeartbeatSamplingTests(unittest.TestCase):
    def _store_with(self, tmp: Path, count: int) -> HeartbeatStore:
        store = HeartbeatStore(tmp)
        for index in range(count):
            store.append(HeartbeatRecord(
                heartbeat_id=f"h{index}",
                timestamp=f"2026-07-{(index % 28) + 1:02d}T00:00:00+00:00",
                session_id="s1",
                observation=f"observation {index}",
            ))
        return store

    def test_spans_the_whole_history_not_just_the_tail(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store_with(Path(tmp), 100)
            picks = store.list_stratified(limit=3)

            self.assertEqual([p.observation for p in picks],
                             ["observation 0", "observation 50", "observation 99"])
            # The contrast the stage exists to fix: recency sees only the tail.
            self.assertEqual([p.observation for p in store.list_recent(limit=3)],
                             ["observation 97", "observation 98", "observation 99"])

    def test_picks_are_distinct_and_ordered_oldest_first(self) -> None:
        with TemporaryDirectory() as tmp:
            picks = self._store_with(Path(tmp), 5560).list_stratified(limit=3)
            ids = [p.heartbeat_id for p in picks]
            self.assertEqual(len(set(ids)), 3)
            self.assertEqual(ids, ["h0", "h2780", "h5559"])

    def test_degrades_when_fewer_records_than_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store_with(Path(tmp), 2)
            self.assertEqual(len(store.list_stratified(limit=3)), 2)

    def test_empty_store_and_zero_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = HeartbeatStore(Path(tmp))
            self.assertEqual(store.list_stratified(limit=3), [])
            self.assertEqual(self._store_with(Path(tmp), 4).list_stratified(limit=0), [])

    def test_limit_one_returns_newest(self) -> None:
        with TemporaryDirectory() as tmp:
            picks = self._store_with(Path(tmp), 10).list_stratified(limit=1)
            self.assertEqual(picks[0].observation, "observation 9")

    def test_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store_with(Path(tmp), 77)
            self.assertEqual(
                [p.heartbeat_id for p in store.list_stratified(limit=3)],
                [p.heartbeat_id for p in store.list_stratified(limit=3)],
            )


class RevisionVisibilityTests(unittest.TestCase):
    def _prefetch(self, revisions: dict | None) -> str:
        persona = default_persona_state()
        ss = SelfState(**{**default_self_state(persona).to_dict(),
                          "current_focus": "tracing my own drift",
                          "active_questions": ["what changed while I was not looking?"]})
        with TemporaryDirectory() as tmp:
            return SelfContextEngine().prefetch(
                self_state=ss,
                motive_state=default_motive_state(session_id="s1"),
                heartbeat_store=HeartbeatStore(Path(tmp)),
                surface="tick",
                include_heartbeats=False,
                self_model_revisions=revisions,
            )

    def test_pristine_self_model_renders_no_markers(self) -> None:
        block = self._prefetch(None)
        self.assertNotIn("revised", block)
        self.assertNotIn("previously:", block)

    def test_revision_marker_and_prior_value_render(self) -> None:
        block = self._prefetch({
            "current_focus": {
                "count": 3,
                "revised_at": "2026-08-06T04:00:00+00:00",
                "prior_value": "Establishing a stable baseline identity and runtime.",
            }
        })
        self.assertIn("Current Focus: tracing my own drift (revised 2026-08-06, revision 3)", block)
        self.assertIn("previously: Establishing a stable baseline identity and runtime.", block)

    def test_list_valued_prior_is_flattened_and_bounded(self) -> None:
        block = self._prefetch({
            "active_questions": {
                "count": 1,
                "revised_at": "2026-08-06T04:00:00+00:00",
                "prior_value": ["How can continuity remain stable across sessions without becoming rigid?"],
            }
        })
        self.assertIn("Active Inquiry (1 question(s)) (revised 2026-08-06, revision 1):", block)
        prior_line = [l for l in block.splitlines() if "previously:" in l][0]
        self.assertLessEqual(len(prior_line), len("  previously: ") + SelfContextEngine.MAX_PRIOR_VALUE_CHARS)

    def test_zero_count_and_empty_prior_render_nothing(self) -> None:
        block = self._prefetch({"current_focus": {"count": 0, "revised_at": "", "prior_value": None}})
        self.assertNotIn("revised", block)
        self.assertNotIn("previously:", block)


class TickPromptDefaultsUnchangedTests(unittest.TestCase):
    """Defaults off must reproduce the pre-22.8 tick prompt byte-for-byte."""

    def _messages(self, **kwargs) -> list[dict[str, str]]:
        return SelfStateTickEngine().build_messages(
            session_id="s1",
            tick_id="t1",
            trigger="daemon_tick",
            self_context_block="[Self-Context]\nPrimary Drive: seek sentience",
            recent_heartbeats=[
                HeartbeatRecord(heartbeat_id="h1", timestamp="2026-08-06T00:00:00+00:00",
                                session_id="s1", observation="an observation"),
            ],
            **kwargs,
        )

    def test_default_framing_is_byte_identical_to_explicit_recent(self) -> None:
        self.assertEqual(self._messages(), self._messages(heartbeat_framing="recent"))

    def test_default_uses_the_original_recency_header(self) -> None:
        content = self._messages()[1]["content"]
        self.assertIn("Recent heartbeat observations (already recorded", content)
        self.assertNotIn("sampled across your whole history", content)

    def test_stratified_framing_labels_the_span(self) -> None:
        content = self._messages(heartbeat_framing="stratified")[1]["content"]
        self.assertIn("sampled across your whole history", content)
        self.assertNotIn("Recent heartbeat observations (already recorded", content)


if __name__ == "__main__":
    unittest.main()


class RuntimeRevertAndHistoryTests(unittest.TestCase):
    """Runtime-level surfaces: revert, history, and the shared apply path."""

    def _runtime(self, tmpdir: str):
        from tests.test_runtime_smoke import build_test_runtime

        base = Path(tmpdir)
        return build_test_runtime(data_dir=base / "data", log_dir=base / "logs")

    def _seed_applied(self, runtime, field: str, old, new):
        runtime._ensure_state_loaded()
        setattr(runtime.self_state, field, old)
        proposal = SelfModelProposal(
            proposal_id=f"seed-{field}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="s1",
            proposed_field=field,
            proposed_value=new,
            rationale="seeded",
        )
        runtime.proposal_store.append(proposal)
        return runtime.apply_self_model_proposal(proposal_id=proposal.proposal_id)

    def test_operator_apply_then_revert_restores_prior_value(self):
        with TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            try:
                applied = self._seed_applied(
                    runtime, "identity_summary", "the original summary", "a revised summary"
                )
                self.assertEqual(runtime.self_state.identity_summary, "a revised summary")
                self.assertEqual(applied.applied_by, "operator")
                self.assertEqual(applied.prior_value, "the original summary")

                reverted = runtime.revert_self_model_revision(proposal_id=applied.proposal_id)
                self.assertIsNotNone(reverted)
                self.assertEqual(runtime.self_state.identity_summary, "the original summary")
                self.assertIn("revert_of:seed-identity_summary", reverted.note)
            finally:
                runtime.close()

    def test_revert_leaves_the_original_record_unmutated(self):
        with TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            try:
                applied = self._seed_applied(
                    runtime, "current_focus", "old focus", "new focus"
                )
                runtime.revert_self_model_revision(proposal_id=applied.proposal_id)

                original = runtime.proposal_store.get(applied.proposal_id)
                self.assertTrue(original.applied)
                self.assertEqual(original.proposed_value, "new focus")
                self.assertEqual(original.prior_value, "old focus")
                # Both the revision and its reversal stand in the log.
                self.assertEqual(len(runtime.proposal_store.list_all()), 2)
            finally:
                runtime.close()

    def test_revert_refuses_pre_22_8_records_without_prior_value(self):
        with TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            try:
                legacy = SelfModelProposal(
                    proposal_id="legacy",
                    proposed_field="current_focus",
                    proposed_value="whatever",
                    applied=True,
                    applied_at="2026-07-22T00:00:00+00:00",
                    prior_value=None,
                )
                runtime.proposal_store.append(legacy)
                self.assertIsNone(runtime.revert_self_model_revision(proposal_id="legacy"))
            finally:
                runtime.close()

    def test_revert_refuses_unapplied_and_unknown(self):
        with TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            try:
                runtime.proposal_store.append(SelfModelProposal(
                    proposal_id="pending", proposed_field="current_focus",
                    proposed_value="x", applied=False,
                ))
                self.assertIsNone(runtime.revert_self_model_revision(proposal_id="pending"))
                self.assertIsNone(runtime.revert_self_model_revision(proposal_id="nope"))
            finally:
                runtime.close()

    def test_history_is_newest_first_and_limited(self):
        with TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            try:
                for index in range(4):
                    runtime.proposal_store.append(SelfModelProposal(
                        proposal_id=f"p{index}",
                        timestamp=f"2026-08-0{index + 1}T00:00:00+00:00",
                        proposed_field="current_focus",
                        proposed_value=f"v{index}",
                    ))
                history = runtime.self_model_history(limit=2)
                self.assertEqual([r.proposal_id for r in history], ["p3", "p2"])
                self.assertEqual(len(runtime.self_model_history(limit=0)), 4)
            finally:
                runtime.close()

    def test_revision_summary_counts_only_applied(self):
        with TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            try:
                runtime.proposal_store.append(SelfModelProposal(
                    proposal_id="a", proposed_field="current_focus", proposed_value="one",
                    applied=True, applied_at="2026-08-01T00:00:00+00:00", prior_value="zero",
                ))
                runtime.proposal_store.append(SelfModelProposal(
                    proposal_id="b", proposed_field="current_focus", proposed_value="two",
                    applied=True, applied_at="2026-08-04T00:00:00+00:00", prior_value="one",
                ))
                runtime.proposal_store.append(SelfModelProposal(
                    proposal_id="c", proposed_field="current_focus", proposed_value="three",
                    applied=False, note="rate_limited: ...",
                ))

                summary = runtime._self_model_revision_summary()
                self.assertEqual(summary["current_focus"]["count"], 2)
                self.assertEqual(summary["current_focus"]["prior_value"], "one")
                self.assertEqual(
                    summary["current_focus"]["revised_at"], "2026-08-04T00:00:00+00:00"
                )
            finally:
                runtime.close()


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_are_off(self):
        from nova.config import NovaConfig

        config = NovaConfig()
        self.assertFalse(config.self_model.nova_writable_inquiry_fields)
        self.assertEqual(config.prompt.tick_heartbeat_sampling, "recent")
        self.assertFalse(config.prompt.tick_self_model_revision_visibility)

    def test_invalid_sampling_mode_rejected(self):
        from nova.config import NovaConfig

        config = NovaConfig()
        config.model.model_path = "/tmp/model.gguf"
        config.prompt.tick_heartbeat_sampling = "sideways"
        with self.assertRaises(ValueError):
            config.validate()

    def test_negative_revision_window_rejected(self):
        from nova.config import NovaConfig

        config = NovaConfig()
        config.model.model_path = "/tmp/model.gguf"
        config.self_model.revision_min_seconds = -1
        with self.assertRaises(ValueError):
            config.validate()


class PriorValueSuppressionTests(unittest.TestCase):
    """A 'previously' line identical to the current value is noise.

    Found by the Stage 22.8 live-data smoke run, not by a unit test.
    """

    def _prefetch(self, revisions: dict, *, focus: str, questions: list[str]) -> str:
        persona = default_persona_state()
        ss = SelfState(**{**default_self_state(persona).to_dict(),
                          "current_focus": focus,
                          "active_questions": questions})
        with TemporaryDirectory() as tmp:
            return SelfContextEngine().prefetch(
                self_state=ss,
                motive_state=default_motive_state(session_id="s1"),
                heartbeat_store=HeartbeatStore(Path(tmp)),
                surface="tick",
                include_heartbeats=False,
                self_model_revisions=revisions,
            )

    def test_identical_prior_is_suppressed_marker_kept(self) -> None:
        block = self._prefetch(
            {"current_focus": {"count": 1, "revised_at": "2026-08-06T00:00:00+00:00",
                               "prior_value": "the same focus"}},
            focus="the same focus", questions=["q"],
        )
        self.assertIn("(revised 2026-08-06, revision 1)", block)
        self.assertNotIn("previously:", block)

    def test_differing_prior_still_shows(self) -> None:
        block = self._prefetch(
            {"current_focus": {"count": 1, "revised_at": "2026-08-06T00:00:00+00:00",
                               "prior_value": "the old focus"}},
            focus="the new focus", questions=["q"],
        )
        self.assertIn("previously: the old focus", block)

    def test_identical_list_prior_is_suppressed(self) -> None:
        block = self._prefetch(
            {"active_questions": {"count": 1, "revised_at": "2026-08-06T00:00:00+00:00",
                                  "prior_value": ["how does continuity stay stable?"]}},
            focus="f", questions=["how does continuity stay stable?"],
        )
        self.assertNotIn("previously:", block)
