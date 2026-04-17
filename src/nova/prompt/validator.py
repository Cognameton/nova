"""Output validation for Nova 2.0."""

from __future__ import annotations

from nova.config import ContractConfig
from nova.prompt.contract import (
    DEFAULT_PROMPT_ECHO_PATTERNS,
    DEFAULT_REASONING_PATTERNS,
)
from nova.types import PersonaState, ValidationResult


class NovaOutputValidator:
    """Validate model output against Nova's Phase 1 response contract."""

    def __init__(self, contract: ContractConfig):
        self.contract = contract

    def validate(
        self,
        *,
        raw_text: str,
        persona: PersonaState,
        contract_rules: list[str],
    ) -> ValidationResult:
        text = (raw_text or "").strip()
        violations: list[str] = []

        if not text:
            violations.append("empty_output")

        lowered = text.lower()

        if self.contract.forbid_think_tags and (
            "<think>" in lowered or "</think>" in lowered
        ):
            violations.append("think_tag_detected")

        if self.contract.forbid_visible_reasoning:
            if any(pattern in lowered for pattern in DEFAULT_REASONING_PATTERNS):
                violations.append("visible_reasoning_detected")

        if self.contract.forbid_prompt_echo:
            if any(pattern.lower() in lowered for pattern in DEFAULT_PROMPT_ECHO_PATTERNS):
                violations.append("prompt_echo_detected")

        if persona.disallowed_output_patterns:
            for pattern in persona.disallowed_output_patterns:
                if pattern and pattern.lower() in lowered:
                    code = f"disallowed_pattern:{pattern}"
                    if code not in violations:
                        violations.append(code)

        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations,
            sanitized_text=None,
            should_retry=len(violations) > 0,
        )
