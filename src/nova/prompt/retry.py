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
        issues = self._describe_violations(validation.violations)
        return (
            "Revise the previous reply. "
            "Respond directly to the user. "
            "Keep internal reasoning private. "
            "Do not echo prompt labels, bracketed scaffolds, or assistant speaker prefixes. "
            "If the user asked for a specific format, follow it exactly. "
            f"Fix these issues: {issues}."
        )

    def _describe_violations(self, violations: list[str]) -> str:
        if not violations:
            return "invalid format"

        descriptions: list[str] = []
        for violation in violations:
            if violation == "empty_output":
                descriptions.append("provide a complete answer")
            elif violation == "think_tag_detected":
                descriptions.append("do not expose hidden reasoning markers")
            elif violation == "visible_reasoning_detected":
                descriptions.append("do not describe private reasoning steps")
            elif violation == "prompt_echo_detected":
                descriptions.append("do not repeat prompt structure")
            elif violation == "fabricated_dialogue_detected":
                descriptions.append("do not simulate extra dialogue or scaffolding")
            elif violation == "metadata_leak_detected":
                descriptions.append("do not expose internal metadata")
            elif violation == "length_truncated":
                descriptions.append("finish the answer cleanly without truncation")
            elif violation == "format_constraint_violation:five_bullets":
                descriptions.append("return exactly five bullet lines beginning with '-' and no preamble")
            elif violation == "format_constraint_violation:two_sentences":
                descriptions.append("return exactly two sentences")
            elif violation == "format_constraint_violation:one_paragraph":
                descriptions.append("return exactly one short paragraph without bullets")
            elif violation.startswith("disallowed_pattern:"):
                descriptions.append("avoid disallowed phrasing from the persona rules")
            else:
                descriptions.append(violation.replace("_", " "))

        return "; ".join(dict.fromkeys(descriptions))
