from __future__ import annotations

import unittest

from nova.config import ContractConfig
from nova.persona.defaults import default_persona_state, default_self_state
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.contract import build_contract_rules
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.types import RetrievalHit, TurnRecord, ValidationResult


class PromptAndValidationTests(unittest.TestCase):
    def test_prompt_composer_includes_expected_sections(self) -> None:
        persona = default_persona_state()
        self_state = default_self_state(persona)
        composer = NovaPromptComposer(token_counter=lambda text: len(text.split()))

        bundle = composer.compose(
            persona=persona,
            self_state=self_state,
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
        self.assertIn("[Memory:episodic]", bundle.full_prompt)
        self.assertIn("[Recent Conversation]", bundle.full_prompt)
        self.assertIn("[User]", bundle.full_prompt)
        self.assertIn("[Response Rules]", bundle.full_prompt)

    def test_validator_rejects_think_tags(self) -> None:
        persona = default_persona_state()
        validator = NovaOutputValidator(ContractConfig())
        rules = build_contract_rules(persona, ContractConfig())

        result = validator.validate(
            raw_text="<think>hidden</think>Visible answer",
            persona=persona,
            contract_rules=rules,
        )

        self.assertFalse(result.valid)
        self.assertIn("think_tag_detected", result.violations)

    def test_retry_policy_retries_invalid_output_within_budget(self) -> None:
        policy = BasicRetryPolicy()
        validation = ValidationResult(valid=False, violations=["think_tag_detected"], should_retry=True)

        self.assertTrue(policy.should_retry(validation=validation, attempt_index=0, max_retries=2))
        self.assertFalse(policy.should_retry(validation=validation, attempt_index=2, max_retries=2))


if __name__ == "__main__":
    unittest.main()
