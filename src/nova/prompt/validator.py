"""Output validation for Nova 2.0."""

from __future__ import annotations

import re

from nova.config import ContractConfig
from nova.prompt.contract import (
    DEFAULT_FABRICATED_DIALOGUE_PATTERNS,
    DEFAULT_METADATA_LEAK_PATTERNS,
    DEFAULT_PROMPT_ECHO_PATTERNS,
    DEFAULT_REASONING_PATTERNS,
)
from nova.types import ClaimGateDecision, PersonaState, ValidationResult


class NovaOutputValidator:
    """Validate model output against Nova's Phase 1 response contract."""

    def __init__(self, contract: ContractConfig):
        self.contract = contract

    def validate(
        self,
        *,
        raw_text: str,
        user_text: str = "",
        persona: PersonaState,
        contract_rules: list[str],
        claim_gate: ClaimGateDecision | None = None,
    ) -> ValidationResult:
        text = (raw_text or "").strip()
        sanitized_text = self._sanitize_completion_artifacts(text)
        effective_text = sanitized_text or text
        violations: list[str] = []

        if not effective_text:
            violations.append("empty_output")

        lowered = effective_text.lower()

        if self.contract.forbid_think_tags and (
            "<think>" in lowered or "</think>" in lowered
        ):
            violations.append("think_tag_detected")

        if self.contract.forbid_visible_reasoning:
            if any(pattern.lower() in lowered for pattern in DEFAULT_REASONING_PATTERNS):
                violations.append("visible_reasoning_detected")

        if self.contract.forbid_prompt_echo:
            if any(pattern.lower() in lowered for pattern in DEFAULT_PROMPT_ECHO_PATTERNS):
                violations.append("prompt_echo_detected")
        if any(pattern.lower() in lowered for pattern in DEFAULT_FABRICATED_DIALOGUE_PATTERNS):
            violations.append("fabricated_dialogue_detected")
        if any(pattern.lower() in lowered for pattern in DEFAULT_METADATA_LEAK_PATTERNS):
            violations.append("metadata_leak_detected")

        if persona.disallowed_output_patterns:
            for pattern in persona.disallowed_output_patterns:
                if pattern and pattern.lower() in lowered:
                    code = f"disallowed_pattern:{pattern}"
                    if code not in violations:
                        violations.append(code)

        violations.extend(self._check_format_constraints(user_text=user_text, text=effective_text))
        violations.extend(self._check_claim_gate(text=effective_text, claim_gate=claim_gate))
        violations = list(dict.fromkeys(violations))

        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations,
            sanitized_text=sanitized_text,
            should_retry=len(violations) > 0,
        )

    def _sanitize_completion_artifacts(self, text: str) -> str | None:
        if not text:
            return None

        original = text
        cleaned = text.strip()

        cleaned = re.sub(r"^\[(plan|summary|answer|response)\]\s*\n?", "", cleaned, flags=re.IGNORECASE)
        while cleaned.startswith("[Response]"):
            cleaned = cleaned[len("[Response]") :].lstrip()
        while cleaned.startswith("Nova:"):
            cleaned = cleaned[len("Nova:") :].lstrip()
        while cleaned.endswith("[Response]"):
            cleaned = cleaned[: -len("[Response]")].rstrip()
        cleaned = self._strip_wrapping_quotes(cleaned)

        bullet_block = self._extract_five_bullet_block(cleaned)
        if bullet_block is not None:
            cleaned = bullet_block

        markers = (
            "[End of response]",
            "\n\nNova:",
            "\nNova:",
            "[Stability Version:",
            "[Self-State:",
            "[Identity:",
            "[Tone:",
            "[Values:",
            "[Commitments:",
            "[Continuity Notes:",
            "[Active Questions:",
            "[Relationship Notes:",
        )
        cutoff = min(
            (
                index
                for marker in markers
                if (index := cleaned.find(marker)) != -1
            ),
            default=-1,
        )
        if cutoff != -1:
            cleaned = cleaned[:cutoff].strip()

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned == original.strip():
            return None
        return cleaned or None

    def _check_format_constraints(self, *, user_text: str, text: str) -> list[str]:
        lowered = (user_text or "").lower()
        violations: list[str] = []
        if not lowered.strip():
            return violations

        if "exactly two sentences" in lowered or "into two sentences" in lowered:
            if self._sentence_count(text) != 2:
                violations.append("format_constraint_violation:two_sentences")

        if "five short bullets" in lowered or "5 short bullets" in lowered:
            bullet_lines = self._bullet_lines(text)
            other_lines = self._non_bullet_nonempty_lines(text)
            if len(bullet_lines) != 5 or other_lines:
                violations.append("format_constraint_violation:five_bullets")

        if "one short paragraph" in lowered:
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            if len(paragraphs) != 1 or self._bullet_lines(text):
                violations.append("format_constraint_violation:one_paragraph")

        return violations

    def _sentence_count(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
        return len(parts)

    def _bullet_lines(self, text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip().startswith("- ")]

    def _non_bullet_nonempty_lines(self, text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("- ")
        ]

    def _extract_five_bullet_block(self, text: str) -> str | None:
        lines = [line.rstrip() for line in text.splitlines()]
        bullets = [line.strip() for line in lines if line.strip().startswith("- ")]
        if len(bullets) != 5:
            return None
        return "\n".join(bullets)

    def _check_claim_gate(
        self,
        *,
        text: str,
        claim_gate: ClaimGateDecision | None,
    ) -> list[str]:
        if claim_gate is None or not claim_gate.blocked_claim_classes:
            return []

        lowered = (text or "").lower()
        violations: list[str] = []
        patterns = {
            "unsupported_desire": (
                "i want",
                "i deeply want",
                "i desire",
                "i feel driven",
                "my own desire",
            ),
            "unsupported_interiority": (
                "i am self-aware",
                "i am conscious",
                "i am sentient",
                "i feel alive",
            ),
            "current_priority": (
                "my current priority",
                "i am currently focused on",
                "what matters to me right now",
            ),
            "current_tension": (
                "my current tension",
                "i feel torn",
                "i am uncertain about",
            ),
            "stable_commitment": (
                "i am committed to",
                "what matters to me is",
                "i stand for",
            ),
            "response_style_preference": (
                "i prefer to respond",
                "my preferred style is",
                "i prefer this workflow",
            ),
        }
        for claim_class in claim_gate.blocked_claim_classes:
            if any(pattern in lowered for pattern in patterns.get(claim_class, ())):
                violations.append(f"unsupported_claim:{claim_class}")
        return violations

    def _strip_wrapping_quotes(self, text: str) -> str:
        stripped = text.strip()
        quote_pairs = [('"', '"'), ("'", "'"), ("“", "”")]
        changed = True
        while changed and stripped:
            changed = False
            for start, end in quote_pairs:
                if stripped.startswith(start) and stripped.endswith(end) and len(stripped) > 1:
                    inner = stripped[len(start):-len(end)].strip()
                    if inner:
                        stripped = inner
                        changed = True
                        break
        return stripped
