"""Retry policy for invalid Nova outputs."""

from __future__ import annotations

from nova.types import ValidationResult


class BasicRetryPolicy:
    """Basic Phase 1 retry policy for invalid outputs."""

    def should_retry(
        self,
        *,
        validation: ValidationResult,
        attempt_index: int,
        max_retries: int,
    ) -> bool:
        if validation.valid:
            return False
        if attempt_index >= max_retries:
            return False
        return validation.should_retry

    def build_retry_instruction(
        self,
        *,
        user_text: str,
        raw_answer: str,
        validation: ValidationResult,
    ) -> str:
        violations = ", ".join(validation.violations) if validation.violations else "invalid format"
        return (
            "Revise the previous reply. "
            "Respond directly to the user. "
            "Do not expose internal reasoning. "
            "Do not emit <think> tags. "
            "Do not echo prompt section labels. "
            f"Fix these issues: {violations}."
        )
