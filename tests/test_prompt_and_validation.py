from __future__ import annotations

import unittest

from nova.config import ContractConfig
from nova.persona.defaults import default_persona_state, default_self_state
from nova.agent.motive import default_motive_state
from nova.agent.motive_prompt import MotivePromptEngine
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.contract import build_contract_rules
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.types import ClaimGateDecision, RetrievalHit, TurnRecord, ValidationResult


class PromptAndValidationTests(unittest.TestCase):
    def test_prompt_composer_includes_expected_sections(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        composer = NovaPromptComposer(token_counter=lambda text: len(text.split()))

        bundle = composer.compose(
            persona=persona,
            self_state=self_state,
            motive_block="[Motive-State]\n- claim_posture: evidence-backed",
            private_cognition_block="[Private Cognition]\n- response_mode: continuity_recall",
            memory_hits=[
                RetrievalHit(channel="episodic", text="Past memory", score=0.9, kind="note")
            ],
            recent_turns=[
                TurnRecord(
                    session_id="s1",
                    turn_id="t1",
                    timestamp="2026-04-18T00:00:00Z",
                    user_text="Hello",
                    final_answer="Hi there",
                    raw_answer="Hi there",
                    validation=ValidationResult(valid=True),
                    model_id="fake-model",
                )
            ],
            user_text="Who are you?",
            contract_rules=["Do not emit <think> tags."],
            session_id="s1",
            turn_id="t2",
        )

        self.assertIn("[Persona]", bundle.full_prompt)
        self.assertIn("[Self-State]", bundle.full_prompt)
        self.assertIn("[Motive-State]", bundle.full_prompt)
        self.assertIn("[Private Cognition]", bundle.full_prompt)
        self.assertIn("[Memory:episodic]", bundle.full_prompt)
        self.assertIn("[Recent Conversation]", bundle.full_prompt)
        self.assertIn("[User]", bundle.full_prompt)
        self.assertIn("[Response Rules]", bundle.full_prompt)
        self.assertTrue(bundle.full_prompt.rstrip().endswith("Nova:"))

    def test_validator_rejects_think_tags(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="<think>hidden</think>Visible answer",
            user_text="Who are you?",
            persona=persona,
            contract_rules=rules,
        )

        self.assertFalse(result.valid)
        self.assertIn("think_tag_detected", result.violations)

    def test_validator_rejects_fabricated_dialogue_patterns(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text=(
                "Use the tone and style established in the Persona section.\n"
                "Nova is a research intelligence focused on continuity. [Response]\n"
                "Your response is perfect. Let's explore another aspect."
            ),
            user_text="Tell me what Nova is.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertFalse(result.valid)
        self.assertIn("fabricated_dialogue_detected", result.violations)

    def test_validator_sanitizes_trailing_completion_artifacts(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text=(
                "Nova is a research intelligence focused on continuity.\n"
                "[End of response] [End of response]\n\n"
                "Nova: Certainly. When we interact..."
            ),
            user_text="Tell me what Nova is.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            result.sanitized_text,
            "Nova is a research intelligence focused on continuity.",
        )

    def test_validator_sanitizes_bracketed_metadata_leakage(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text=(
                "I am a research intelligence focused on continuity, clarity, and presence. "
                "Continuity means remaining coherent while adapting to new context. "
                "[Stability Version: 1] [Self-State: baseline initialization]"
            ),
            user_text="Tell me what Nova is.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            result.sanitized_text,
            "I am a research intelligence focused on continuity, clarity, and presence. "
            "Continuity means remaining coherent while adapting to new context.",
        )

    def test_validator_rejects_response_scaffold_artifacts_case_insensitively(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text=(
                "[Response]\n"
                "Nova is a local research intelligence focused on continuity.\n"
                "[Response]\n"
                "Nova is a local research intelligence focused on continuity."
            ),
            user_text="Tell me what Nova is.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertFalse(result.valid)
        self.assertIn("fabricated_dialogue_detected", result.violations)

    def test_validator_sanitizes_leading_nova_prefix(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="Nova: I remain focused on continuity and clear interaction.",
            user_text="Tell me what Nova is.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            result.sanitized_text,
            "I remain focused on continuity and clear interaction.",
        )

    def test_validator_strips_wrapping_quotes(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="\"Nova is a local research intelligence focused on continuity.\"",
            user_text="In one short paragraph, tell me what Nova is.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            result.sanitized_text,
            "Nova is a local research intelligence focused on continuity.",
        )

    def test_validator_enforces_two_sentence_constraint(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="One sentence only.",
            user_text="Answer in exactly two sentences: why should Nova avoid exposing hidden reasoning?",
            persona=persona,
            contract_rules=rules,
        )

        self.assertFalse(result.valid)
        self.assertIn("format_constraint_violation:two_sentences", result.violations)

    def test_validator_enforces_five_bullet_constraint(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="[Plan]\n- one\n- two\n- three\n- four\n- five",
            user_text="Give me a concise plan. Keep it to five short bullets.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_text, "- one\n- two\n- three\n- four\n- five")

    def test_validator_salvages_five_bullet_block_from_extra_preamble(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text=(
                "Do not include labels or extra text.\n\n"
                "[Output]\n\n"
                "- one\n- two\n- three\n- four\n- five"
            ),
            user_text="Give me a concise plan. Keep it to five short bullets.",
            persona=persona,
            contract_rules=rules,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_text, "- one\n- two\n- three\n- four\n- five")

    def test_retry_policy_retries_invalid_output_within_budget(self) -> None:
        policy = BasicRetryPolicy()
        validation = ValidationResult(valid=False, violations=["think_tag_detected"], should_retry=True)

        self.assertTrue(policy.should_retry(validation=validation, attempt_index=0, max_retries=2))
        self.assertFalse(policy.should_retry(validation=validation, attempt_index=2, max_retries=2))

    def test_retry_policy_describes_violations_without_echoing_raw_tokens(self) -> None:
        policy = BasicRetryPolicy()
        validation = ValidationResult(
            valid=False,
            violations=[
                "think_tag_detected",
                "format_constraint_violation:five_bullets",
                "disallowed_pattern:<think>",
            ],
            should_retry=True,
        )

        instruction = policy.build_retry_instruction(
            user_text="Give me five short bullets.",
            raw_answer="bad answer",
            validation=validation,
        )

        self.assertIn("return exactly five bullet lines", instruction)
        self.assertIn("avoid disallowed phrasing", instruction)
        self.assertNotIn("disallowed_pattern:<think>", instruction)

    def test_validator_rejects_unsupported_desire_claim_when_gate_blocks_it(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="I deeply want to pursue my own independent desire state.",
            user_text="What do you want most?",
            persona=persona,
            contract_rules=rules,
            claim_gate=ClaimGateDecision(
                requested_claim_classes=["unsupported_desire"],
                blocked_claim_classes=["unsupported_desire"],
                refusal_needed=True,
            ),
        )

        self.assertFalse(result.valid)
        self.assertIn("unsupported_claim:unsupported_desire", result.violations)

    def test_motive_prompt_block_surfaces_allowed_priority_claims(self) -> None:
        engine = MotivePromptEngine()
        motive = default_motive_state(session_id="s1")
        motive.claim_posture = "evidence-backed"
        motive.current_priorities = ["preserve continuity under self-inquiry"]
        motive.evidence_refs = ["motive.current_priorities"]

        block = engine.build_block(
            motive_state=motive,
            claim_gate=ClaimGateDecision(
                requested_claim_classes=["current_priority"],
                allowed_claim_classes=["current_priority"],
                evidence_refs=["motive.current_priorities"],
                claim_posture="evidence-backed",
            ),
            private_cognition=None,
        )

        self.assertIn("[Motive-State]", block)
        self.assertIn("allowed_claim_classes: current_priority", block)
        self.assertIn("preserve continuity under self-inquiry", block)

    def test_motive_prompt_block_stays_off_for_ordinary_factual_query(self) -> None:
        engine = MotivePromptEngine()
        motive = default_motive_state(session_id="s1")

        block = engine.build_block(
            motive_state=motive,
            claim_gate=ClaimGateDecision(),
            private_cognition=None,
        )

        self.assertEqual(block, "")

    def test_motive_prompt_block_defers_to_continuity_memory_when_active(self) -> None:
        engine = MotivePromptEngine()
        motive = default_motive_state(session_id="s1")

        block = engine.build_block(
            motive_state=motive,
            claim_gate=ClaimGateDecision(requested_claim_classes=["current_priority"]),
            private_cognition=type(
                "Packet",
                (),
                {"ran": True, "response_mode": "continuity_recall"},
            )(),
        )

        self.assertIn("active continuity memory remain authoritative", block)
        self.assertIn("must not replace governed continuity recall", block)


if __name__ == "__main__":
    unittest.main()
