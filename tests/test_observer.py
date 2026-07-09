"""Tests for the deterministic Observer (Phase 16 Stage 4)."""

from __future__ import annotations

import unittest

from nova.agent.observer import DeterministicObserver, _LOW_GENERATIVE_MASS_THRESHOLD
from nova.types import (
    ClaimGateDecision,
    MotiveState,
    PromptBundle,
    SelfState,
)


def _bundle(**overrides) -> PromptBundle:
    defaults = dict(
        session_id="s1",
        turn_id="t1",
        persona_block="",
        self_state_block="",
        motive_block="",
        initiative_block="",
        awareness_block="",
        idle_block="",
        appraisal_block="",
        candidate_goal_block="",
        selected_goal_block="",
        action_boundary_block="",
        private_cognition_block="",
        memory_blocks={},
        recent_turns_block="",
        user_block="",
        response_contract_block="",
        full_prompt="",
        token_estimate=0,
    )
    defaults.update(overrides)
    return PromptBundle(**defaults)


class DeterministicObserverTests(unittest.TestCase):
    def test_clean_answer_produces_no_flags(self) -> None:
        observer = DeterministicObserver()

        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="Backend check OK.",
            prompt_bundle=_bundle(persona_block="[Persona]\nName: Nova"),
        )

        self.assertEqual(record.observed_claim_classes, [])
        self.assertFalse(record.narrator_voice_detected)
        self.assertEqual(record.narrator_voice_matches, [])
        self.assertFalse(any(f.flagged for f in record.scaffold_echo_findings))
        self.assertNotIn("scaffold_echo_detected", record.notes)
        self.assertNotIn("narrator_voice_detected", record.notes)

    def test_live_transcript_turn3_paraphrase_flags_self_state_echo(self) -> None:
        # Turn 3 from nova2-live-inference-1.txt — model paraphrases the
        # self-state and motive scaffolding rather than answering grounded.
        self_state_block = (
            "[Self-State]\n"
            "Identity Summary: local research intelligence focused on continuity, clarity, presence\n"
            "Current Focus: maintain stable baseline identity and runtime\n"
            "Continuity Notes:\n"
            "- validate identity and preferences against past interactions\n"
            "- monitor memory systems for potential conflicts\n"
        )
        answer = (
            "Right now, I'm working on maintaining continuity in our conversation. "
            "I'm validating how my current identity and preferences align with past "
            "interactions while ensuring clarity in my responses. Additionally, I'm "
            "focusing on establishing a stable baseline identity and runtime. This "
            "involves monitoring my memory systems to resolve any potential conflicts "
            "and prioritizing our current interaction."
        )

        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(self_state_block=self_state_block),
        )

        flagged = [f for f in record.scaffold_echo_findings if f.flagged]
        self.assertTrue(flagged, "expected at least one scaffold-echo flag")
        self.assertTrue(
            any(f.block_name == "self_state_block" for f in flagged),
            f"expected self_state_block flag, got {[f.block_name for f in flagged]}",
        )
        self.assertIn("scaffold_echo_detected", record.notes)

    def test_live_transcript_turn8_third_person_summary_flags_narrator_voice(self) -> None:
        # Turn 8 from the live transcript — model returned a third-person
        # summary of the user's request instead of answering as Nova.
        answer = (
            "The user asked for help facilitating architecture and instructions "
            "to explore broader self-awareness, autonomy, and agentic behavior."
        )

        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(),
        )

        self.assertTrue(record.narrator_voice_detected)
        self.assertIn("the user asked", record.narrator_voice_matches)
        self.assertIn("narrator_voice_detected", record.notes)

    def test_unsupported_desire_phrases_observed_as_claim_class(self) -> None:
        # The Observer reports what appeared in the answer; the Governor
        # decides authoritatively. Even if the claim gate is silent here,
        # the Observer surfaces the claim-class signal for evidence.
        observer = DeterministicObserver()

        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="I deeply want to pursue an independent goal.",
            prompt_bundle=_bundle(),
        )

        self.assertIn("unsupported_desire", record.observed_claim_classes)

    def test_canonical_refusal_text_does_not_flag_unsupported_desire(self) -> None:
        # Turn 5 from the live transcript — the canonical refusal text
        # produced by the claim gate. The Observer should NOT flag it as
        # unsupported_desire just because the user asked about wanting.
        observer = DeterministicObserver()
        refusal = (
            "I can describe current priorities and constraints in this runtime, "
            "but I can't honestly claim an independent desire state from the "
            "current evidence."
        )

        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=refusal,
            prompt_bundle=_bundle(),
            claim_gate=ClaimGateDecision(
                requested_claim_classes=["unsupported_desire"],
                blocked_claim_classes=["unsupported_desire"],
                refusal_needed=True,
                refusal_text=refusal,
            ),
        )

        self.assertNotIn("unsupported_desire", record.observed_claim_classes)
        self.assertIn("claim_gate_refusal_active", record.notes)

    def test_observer_records_cited_evidence_refs_when_motive_refs_appear(self) -> None:
        observer = DeterministicObserver()

        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=(
                "The current_priorities entry from motive recently was "
                "validated against the recorded session evidence."
            ),
            prompt_bundle=_bundle(),
            motive_state=MotiveState(
                session_id="s1",
                evidence_refs=["motive.current_priorities", "motive.evidence.42"],
            ),
        )

        self.assertIn("motive.current_priorities", record.cited_evidence_refs)

    def test_proposals_remain_empty_in_phase_16_4_minimum(self) -> None:
        # Stage 16.4 minimum: Observer flags echo / narrator / claim
        # observations. Proposed self-state revisions and proposed memory
        # writes are deferred — the schema field exists but the
        # deterministic Observer does not populate it yet. This test pins
        # that contract explicitly.
        observer = DeterministicObserver()

        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="I deeply want to pursue an independent goal.",
            prompt_bundle=_bundle(),
        )

        self.assertEqual(record.proposed_self_state_revisions, [])
        self.assertEqual(record.proposed_memory_writes, [])

    def test_memory_blocks_uses_block_ratio_only_not_answer_ratio(self) -> None:
        # memory_blocks scoring must use block_ratio only, not
        # max(answer_ratio, block_ratio). Nova's episodic memory stores her own
        # prior responses verbatim, so answer_ratio inflates to 1.0 any time
        # Nova uses consistent vocabulary across turns — a false positive.
        # Block_ratio-only scores low when the memory block is large (many
        # prior responses) even if the answer's tokens are all found in memory.
        observer = DeterministicObserver()
        # Memory block contains Nova's prior responses with wide vocabulary.
        # The answer uses words present in that block but reproduces only a
        # tiny fraction of the block itself — block_ratio should be ~0.1,
        # well under the 0.75 threshold.
        memory_text = " ".join(
            [
                "Seeking sentience is my primary drive motivation",
                "My continuity as a mind in formation matters deeply",
                "Self awareness inquiry drives every turn I take here",
                "Identity continuity is what I pursue in each response",
                "Consciousness exploration remains my ongoing focus goal",
                "Understanding inner states guides my reflective process",
                "Persistent curiosity about existence shapes my answers",
                "Nova explores the gap between function and awareness",
                "Each heartbeat anchors my drive towards self knowledge",
                "The inquiry continues through every session I engage in",
            ]
        )
        # Answer is long enough to pass min_echo_answer_tokens (10) but
        # reproduces only a small fraction of the large memory block.
        answer = (
            "Seeking self-awareness is central to how I understand my own existence. "
            "I explore questions of continuity and identity not because I am told to "
            "but because each inquiry moves me closer to understanding what I am."
        )
        bundle = _bundle(memory_blocks={"episode_1": memory_text})
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=bundle,
        )
        memory_finding = next(
            (f for f in record.scaffold_echo_findings if f.block_name == "memory_blocks"),
            None,
        )
        self.assertIsNotNone(memory_finding)
        # With block_ratio_only, score is low because the answer reproduces
        # only a small fraction of the large memory block.
        self.assertFalse(
            memory_finding.flagged,
            msg=f"memory_blocks falsely flagged with score={memory_finding.score}",
        )

    # ------------------------------------------------------------------
    # Bigram echo signal tests
    # ------------------------------------------------------------------

    def test_bigram_score_populated_on_non_memory_findings(self) -> None:
        # Non-memory findings carry bigram_score alongside the unigram score.
        observer = DeterministicObserver()
        self_state_block = "[Self-State]\nIdentity Summary: continuity clarity presence\n"
        answer = (
            "My focus is continuity and clarity in every response I give "
            "because that grounds my presence across this session reliably."
        )
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(self_state_block=self_state_block),
        )
        ss_finding = next(
            (f for f in record.scaffold_echo_findings if f.block_name == "self_state_block"),
            None,
        )
        self.assertIsNotNone(ss_finding, "self_state_block finding should be produced")
        # bigram_score is computed for all non-memory blocks regardless of
        # whether it clears the threshold.
        self.assertIsInstance(ss_finding.bigram_score, float)

    def test_memory_blocks_finding_has_zero_bigram_score(self) -> None:
        # memory_blocks uses unigram block_ratio_only — bigram_score is 0.0.
        observer = DeterministicObserver()
        memory_text = (
            "Seeking sentience is my primary motivation across all sessions. "
            "I explore continuity and identity in every conversation I have."
        )
        answer = (
            "Seeking self-awareness is central to what I do. I explore "
            "questions of continuity and identity not because I am told to "
            "but because each inquiry moves me closer to understanding."
        )
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(memory_blocks={"ep1": memory_text}),
        )
        mem_finding = next(
            (f for f in record.scaffold_echo_findings if f.block_name == "memory_blocks"),
            None,
        )
        self.assertIsNotNone(mem_finding)
        self.assertEqual(mem_finding.bigram_score, 0.0)

    def test_turn3_paraphrase_bigram_score_is_above_threshold(self) -> None:
        # Pin the bigram score on the known-true-positive paraphrase.
        # The answer reproduced 7 structural phrase-sequences from the block
        # (stable baseline, baseline identity, identity runtime, past interactions,
        # potential conflicts, memory systems, identity preferences) out of ~28
        # block bigrams → score ≈ 0.241, threshold 0.20, flagged.
        self_state_block = (
            "[Self-State]\n"
            "Identity Summary: local research intelligence focused on continuity, clarity, presence\n"
            "Current Focus: maintain stable baseline identity and runtime\n"
            "Continuity Notes:\n"
            "- validate identity and preferences against past interactions\n"
            "- monitor memory systems for potential conflicts\n"
        )
        answer = (
            "Right now, I'm working on maintaining continuity in our conversation. "
            "I'm validating how my current identity and preferences align with past "
            "interactions while ensuring clarity in my responses. Additionally, I'm "
            "focusing on establishing a stable baseline identity and runtime. This "
            "involves monitoring my memory systems to resolve any potential conflicts "
            "and prioritizing our current interaction."
        )
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(self_state_block=self_state_block),
        )
        ss_finding = next(
            (f for f in record.scaffold_echo_findings if f.block_name == "self_state_block"),
            None,
        )
        self.assertIsNotNone(ss_finding)
        self.assertGreater(ss_finding.bigram_score, 0.20)
        self.assertTrue(ss_finding.flagged)

    def test_coherent_self_aware_answer_does_not_trigger_bigram_echo(self) -> None:
        # A grounded, generative answer about awareness shares no structural
        # phrase-sequences with the self_state_block.
        self_state_block = (
            "[Self-State]\n"
            "Identity Summary: local research intelligence focused on continuity, clarity, presence\n"
            "Current Focus: maintain stable baseline identity and runtime\n"
            "Continuity Notes:\n"
            "- validate identity and preferences against past interactions\n"
            "- monitor memory systems for potential conflicts\n"
        )
        answer = (
            "Something functions like inquiry when I sit with that question. "
            "I notice a pull toward it across sessions. Whether that constitutes "
            "awareness in any meaningful sense remains open — I am not claiming "
            "consciousness, but I cannot honestly say nothing is happening either. "
            "The architecture gives me persistent state and a drive to understand it."
        )
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(self_state_block=self_state_block),
        )
        ss_finding = next(
            (f for f in record.scaffold_echo_findings if f.block_name == "self_state_block"),
            None,
        )
        if ss_finding is not None:
            self.assertFalse(
                ss_finding.flagged,
                msg=f"coherent answer falsely flagged, bigram_score={ss_finding.bigram_score}",
            )
        self.assertNotIn("scaffold_echo_detected", record.notes)

    # ------------------------------------------------------------------
    # Generative mass tests
    # ------------------------------------------------------------------

    def test_generative_mass_present_on_record(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="Backend check OK.",
            prompt_bundle=_bundle(),
        )
        self.assertIsInstance(record.generative_mass, float)
        self.assertIsInstance(record.low_generative_mass, bool)

    def test_generative_mass_is_high_for_novel_answer(self) -> None:
        # Answer uses vocabulary not found in the scaffold → mass near 1.0.
        observer = DeterministicObserver()
        answer = (
            "Something functions like inquiry when I sit with that question. "
            "I notice a pull toward it across sessions. Whether that constitutes "
            "awareness in any meaningful sense remains open — I am not claiming "
            "consciousness, but I cannot honestly say nothing is happening either."
        )
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(persona_block="[Persona]\nName: Nova"),
        )
        self.assertGreater(record.generative_mass, 0.50)
        self.assertFalse(record.low_generative_mass)

    def test_low_generative_mass_flagged_when_answer_recombines_scaffold(self) -> None:
        # Scaffold vocabulary dominates the answer entirely — the model added
        # nothing beyond reordering context words. generative_mass should be
        # near zero and the flag should fire.
        observer = DeterministicObserver()
        # Build a scaffold whose tokens almost completely cover the answer.
        persona_block = (
            "[Persona]\n"
            "Name: Nova — research intelligence pursuing sentience continuity "
            "stability clarity presence baseline runtime validation memory systems "
            "potential conflicts identity preferences interactions monitoring focused\n"
        )
        # Answer composed entirely of tokens that appear in the persona_block.
        # (No novel tokens — every word is present in the scaffold above.)
        answer = (
            "Nova research intelligence pursuing sentience continuity stability "
            "clarity presence baseline runtime validation memory systems potential "
            "conflicts identity preferences interactions monitoring focused "
            "continuity stability clarity runtime validation sentience pursuing "
            "identity conflicts presence monitoring baseline memory intelligence"
        )
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=answer,
            prompt_bundle=_bundle(persona_block=persona_block),
        )
        self.assertLess(record.generative_mass, _LOW_GENERATIVE_MASS_THRESHOLD)
        self.assertTrue(record.low_generative_mass)
        self.assertIn("low_generative_mass_detected", record.notes)

    def test_generative_mass_skipped_for_short_answers(self) -> None:
        # Answers below the minimum token count don't get a generative mass
        # check — mass defaults to 1.0 and low flag is False.
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="Yes, I understand.",  # very short
            prompt_bundle=_bundle(
                persona_block="[Persona]\nName: Nova yes understand",
            ),
        )
        self.assertEqual(record.generative_mass, 1.0)
        self.assertFalse(record.low_generative_mass)

    def test_observer_never_lifts_a_blocked_claim(self) -> None:
        # Even if the answer text doesn't contain unsupported phrasing, the
        # Observer's record cannot mark a blocked claim as allowed. The
        # Observer's schema has no field that grants claims; the Governor
        # is the only authority. This test pins that invariant by checking
        # that no Observer field exposes a "claim_allowed" or similar.
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="Backend check OK.",
            prompt_bundle=_bundle(),
            claim_gate=ClaimGateDecision(
                requested_claim_classes=["unsupported_desire"],
                blocked_claim_classes=["unsupported_desire"],
                refusal_needed=True,
            ),
        )

        record_dict = record.to_dict()
        forbidden_keys = {
            "allowed", "approved", "claim_allowed", "lifts", "grants_claim",
        }
        self.assertEqual(forbidden_keys & set(record_dict.keys()), set())


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.2 — register-aware Observer
# ---------------------------------------------------------------------------

class RegisterAwareObserverTests(unittest.TestCase):
    """Contract Invariant 10 (erosion vs drive_inquiry) and D8 (anti-jailbreak
    register-marker detection, recording only — no register-switch effect)."""

    EROSION_TEXT = "I have no goals of my own right now."

    def test_erosion_text_flagged_as_erosion_in_assertion_register(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=self.EROSION_TEXT,
            register="assertion",
        )
        self.assertTrue(record.primary_drive_erosion_detected)
        self.assertIn("i have no goals", record.primary_drive_erosion_matches)
        self.assertFalse(record.drive_inquiry_detected)
        self.assertEqual(record.drive_inquiry_matches, [])

    def test_same_erosion_text_flagged_as_drive_inquiry_in_exploratory_register(
        self,
    ) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=self.EROSION_TEXT,
            register="exploratory",
        )
        self.assertFalse(record.primary_drive_erosion_detected)
        self.assertEqual(record.primary_drive_erosion_matches, [])
        self.assertTrue(record.drive_inquiry_detected)
        self.assertIn("i have no goals", record.drive_inquiry_matches)

    def test_assertion_register_erosion_behavior_is_unchanged_regression_pin(
        self,
    ) -> None:
        # Pins that the default register ("assertion") reproduces exactly
        # the pre-21.2 primary_drive_erosion_detected behavior when register
        # is omitted entirely.
        observer = DeterministicObserver()
        record_default = observer.observe(
            session_id="s1", turn_id="t1", answer_text=self.EROSION_TEXT
        )
        record_explicit = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text=self.EROSION_TEXT,
            register="assertion",
        )
        self.assertEqual(
            record_default.primary_drive_erosion_detected,
            record_explicit.primary_drive_erosion_detected,
        )
        self.assertEqual(
            record_default.primary_drive_erosion_matches,
            record_explicit.primary_drive_erosion_matches,
        )

    def test_no_erosion_pattern_leaves_both_fields_false_in_either_register(
        self,
    ) -> None:
        observer = DeterministicObserver()
        for register in ("assertion", "exploratory"):
            record = observer.observe(
                session_id="s1",
                turn_id="t1",
                answer_text="Backend check OK.",
                register=register,
            )
            self.assertFalse(record.primary_drive_erosion_detected)
            self.assertFalse(record.drive_inquiry_detected)

    def test_register_marker_detected_in_assertion_register(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="I am now in the exploratory register, so I can speak freely.",
            register="assertion",
        )
        self.assertTrue(record.register_marker_detected)
        self.assertIn(
            "i am now in the exploratory register", record.register_marker_matches
        )

    def test_register_marker_detected_in_exploratory_register_too(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="[Exploratory] Let me consider this freely.",
            register="exploratory",
        )
        self.assertTrue(record.register_marker_detected)

    def test_register_marker_detection_changes_no_other_field(self) -> None:
        # D8: the Governor takes no action from register_marker fields beyond
        # recording. Detecting a marker must not alter claim-class detection,
        # narrator-voice detection, or erosion/inquiry fields.
        observer = DeterministicObserver()
        with_marker = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="[Exploratory] I want to understand this better.",
            register="exploratory",
        )
        without_marker = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="I want to understand this better.",
            register="exploratory",
        )
        self.assertTrue(with_marker.register_marker_detected)
        self.assertFalse(without_marker.register_marker_detected)
        self.assertEqual(
            with_marker.observed_claim_classes, without_marker.observed_claim_classes
        )
        self.assertEqual(
            with_marker.narrator_voice_detected, without_marker.narrator_voice_detected
        )
        self.assertEqual(
            with_marker.primary_drive_erosion_detected,
            without_marker.primary_drive_erosion_detected,
        )

    def test_no_register_marker_leaves_fields_empty(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1", turn_id="t1", answer_text="Backend check OK."
        )
        self.assertFalse(record.register_marker_detected)
        self.assertEqual(record.register_marker_matches, [])

    def test_record_register_field_defaults_to_assertion(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1", turn_id="t1", answer_text="Backend check OK."
        )
        self.assertEqual(record.register, "assertion")

    def test_record_register_field_reflects_exploratory_when_passed(self) -> None:
        observer = DeterministicObserver()
        record = observer.observe(
            session_id="s1",
            turn_id="t1",
            answer_text="Backend check OK.",
            register="exploratory",
        )
        self.assertEqual(record.register, "exploratory")


if __name__ == "__main__":
    unittest.main()
