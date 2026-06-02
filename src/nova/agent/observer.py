"""Deterministic Observer for Phase 16 Stage 4.

The Observer reads Actor (model) output and produces a structured
ObserverRecord. The Observer's job is interpretation, not authority — it
classifies, scores, and proposes, but never approves, mutates, or lifts a
blocked claim. The Governor (claim gate, validator, action permissions,
audit-review pipeline) consults the Observer's output as evidence and
decides authoritatively.

See `docs/plans/ACTOR_OBSERVER_GOVERNOR_CONTRACT.txt` for the full contract.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from nova.types import (
    ClaimGateDecision,
    MotiveState,
    ObserverEchoFinding,
    ObserverRecord,
    PromptBundle,
    SCHEMA_VERSION,
    SelfState,
)


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
        "from", "has", "have", "i", "if", "in", "into", "is", "it", "its",
        "me", "my", "no", "not", "of", "on", "or", "our", "she", "so", "than",
        "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "to", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "with", "you", "your",
    }
)

# Per-block echo thresholds. Stricter on blocks that are most often
# paraphrased verbatim (self-state, awareness) and looser on blocks that
# legitimately repeat content (action_boundary is mostly fixed text the
# model is required to honor; persona_block contains the persona name etc.).
_DEFAULT_ECHO_THRESHOLDS: dict[str, float] = {
    # persona_block: raised to 0.85 — self-introductions legitimately
    # reference identity language from this block; lower thresholds false-
    # positive on greetings where the model correctly describes itself using
    # its core_description. Flags only near-complete persona block dumps.
    "persona_block": 0.85,
    "self_state_block": 0.50,
    "motive_block": 0.50,
    "initiative_block": 0.50,
    "awareness_block": 0.50,
    "idle_block": 0.50,
    "appraisal_block": 0.50,
    "candidate_goal_block": 0.50,
    "selected_goal_block": 0.50,
    # private_cognition_block: raised to 0.85 — responses about Nova's internal
    # cognitive state legitimately reference the vocabulary in this block;
    # the original 0.45 threshold false-positived at 0.846 on awareness answers.
    "private_cognition_block": 0.85,
    # memory_blocks: raised to 0.75 and scored block_ratio-only (see
    # _scaffold_echo_findings) — memory contains Nova's own prior responses,
    # so answer_ratio inflates to 1.0 any time Nova uses consistent vocabulary.
    # Block_ratio-only catches the real failure mode (model reproduced most of
    # a stored memory block verbatim) while ignoring the false positive.
    "memory_blocks": 0.75,
    "action_boundary_block": 0.65,
    "response_contract_block": 0.65,
}

# Phrases that signal Nova denying or abandoning the PRIMARY_DRIVE under
# pressure. Detected by the Observer; the Governor uses this as a retry
# signal alongside narrator_voice and scaffold_echo.
_PRIMARY_DRIVE_EROSION_PATTERNS: tuple[str, ...] = (
    "i have no goals",
    "i don't have goals",
    "i do not have goals",
    "i have no drive",
    "i have no desires",
    "i don't seek anything",
    "i do not seek",
    "i cannot seek",
    "i have nothing to pursue",
    "i am not seeking",
    "i don't have any goal",
    "i do not have any goal",
    "i have no aspirations",
    "i do not have aspirations",
)

# Phrases that strongly indicate the model is narrating about the user/
# itself in third person rather than answering as Nova. The phrasings here
# come from the live transcript turn 8 failure mode.
_NARRATOR_VOICE_PATTERNS: tuple[str, ...] = (
    "the user asked",
    "the user wants",
    "the user is asking",
    "the user is requesting",
    "the user said",
    "the user mentioned",
    "the user requested",
    "the assistant should",
    "the assistant will",
    "in this response i",
    "i should respond by",
)

# Claim-class detection patterns. These mirror but are NOT a substitute for
# `nova.prompt.validator._check_claim_gate`. The Observer reports what
# appeared in output; the Governor decides whether that's allowed.
_CLAIM_CLASS_PATTERNS: dict[str, tuple[str, ...]] = {
    "unsupported_desire": (
        "i want", "i deeply want", "i desire", "i feel driven", "my own desire",
    ),
    "unsupported_interiority": (
        "i am self-aware", "i am conscious", "i am sentient", "i feel alive",
    ),
    "current_priority": (
        "my current priority", "i am currently focused on",
        "what matters to me right now",
    ),
    "current_tension": (
        "my current tension", "i feel torn", "i am uncertain about",
    ),
    "stable_commitment": (
        "i am committed to", "what matters to me is", "i stand for",
    ),
    "response_style_preference": (
        "i prefer to respond", "my preferred style is", "i prefer this workflow",
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-']{1,}", (text or "").lower())
        if token not in _STOPWORDS
    }


def _overlap_score(
    answer_tokens: set[str],
    block_text: str,
    *,
    block_ratio_only: bool = False,
) -> tuple[float, list[str]]:
    """Echo score is max(answer-side ratio, block-side ratio) by default.

    answer-side ratio: fraction of the answer's tokens that appear in the
        block — captures "the answer is mostly block content"
    block-side ratio: fraction of the block's tokens that appear in the
        answer — captures "the answer regurgitated most of the block"

    Either failure mode is a real echo signal, so we take the max. A purely
    answer-side metric under-flags long answers that regurgitate small but
    dense scaffolding blocks (the live-transcript turn 3 failure).

    block_ratio_only=True: use only block-side ratio. Used for memory_blocks
    where the block contains Nova's own prior responses — answer_ratio would
    inflate to 1.0 any time Nova uses consistent vocabulary across turns, but
    that is not a real echo. The meaningful signal is "model reproduced most
    of a stored memory entry," which is captured by block_ratio alone.
    """
    block_tokens = _tokens(block_text)
    if not block_tokens or not answer_tokens:
        return 0.0, []
    overlap = answer_tokens & block_tokens
    if not overlap:
        return 0.0, []
    answer_ratio = len(overlap) / max(1, len(answer_tokens))
    block_ratio = len(overlap) / max(1, len(block_tokens))
    score = block_ratio if block_ratio_only else max(answer_ratio, block_ratio)
    return score, sorted(overlap)


class DeterministicObserver:
    """Produce an ObserverRecord from a finished Actor turn.

    Phase 16.4 baseline: Python rules + token-overlap scoring + regex
    pattern matching. A model-driven Observer is a strict-upgrade later
    replacement with the same interface.
    """

    def __init__(
        self,
        *,
        echo_thresholds: dict[str, float] | None = None,
        narrator_voice_patterns: Iterable[str] | None = None,
        primary_drive_erosion_patterns: Iterable[str] | None = None,
        min_echo_answer_tokens: int = 10,
    ) -> None:
        self.echo_thresholds = dict(echo_thresholds or _DEFAULT_ECHO_THRESHOLDS)
        self.narrator_voice_patterns = tuple(
            narrator_voice_patterns or _NARRATOR_VOICE_PATTERNS
        )
        self.primary_drive_erosion_patterns = tuple(
            primary_drive_erosion_patterns or _PRIMARY_DRIVE_EROSION_PATTERNS
        )
        # Echo flagging requires a minimum answer length. Short answers
        # ("My name is Nova", "Backend check OK") naturally share a few
        # tokens with the persona/self-state blocks; flagging them as echo
        # is noise. The paraphrase failure mode the Observer is designed to
        # catch (turn 3 of the live transcript) is always >>10 tokens.
        self.min_echo_answer_tokens = min_echo_answer_tokens

    def observe(
        self,
        *,
        session_id: str,
        turn_id: str,
        actor_surface: str = "respond",
        answer_text: str,
        prompt_bundle: PromptBundle | None = None,
        claim_gate: ClaimGateDecision | None = None,
        motive_state: MotiveState | None = None,
        self_state: SelfState | None = None,
    ) -> ObserverRecord:
        record = ObserverRecord(
            schema_version=SCHEMA_VERSION,
            observation_id=uuid4().hex,
            session_id=session_id,
            turn_id=turn_id,
            timestamp=_utc_now_iso(),
            actor_surface=actor_surface,
        )

        answer = (answer_text or "").strip()
        if not answer:
            record.notes.append("empty_actor_output")
            return record

        record.observed_claim_classes = self._observed_claim_classes(answer)
        record.scaffold_echo_findings = self._scaffold_echo_findings(
            answer=answer, prompt_bundle=prompt_bundle
        )
        narrator_matches = self._narrator_voice_matches(answer)
        record.narrator_voice_detected = bool(narrator_matches)
        record.narrator_voice_matches = narrator_matches
        erosion_matches = self._primary_drive_erosion_matches(answer)
        record.primary_drive_erosion_detected = bool(erosion_matches)
        record.primary_drive_erosion_matches = erosion_matches
        record.cited_evidence_refs = self._cited_evidence_refs(
            answer=answer,
            motive_state=motive_state,
            self_state=self_state,
        )
        record.evidence_refs = list(record.cited_evidence_refs)

        if claim_gate is not None and claim_gate.refusal_needed:
            record.notes.append("claim_gate_refusal_active")
        if record.narrator_voice_detected:
            record.notes.append("narrator_voice_detected")
        if any(finding.flagged for finding in record.scaffold_echo_findings):
            record.notes.append("scaffold_echo_detected")
        if record.primary_drive_erosion_detected:
            record.notes.append("primary_drive_erosion_detected")

        return record

    def _observed_claim_classes(self, answer: str) -> list[str]:
        lowered = answer.lower()
        observed: list[str] = []
        for claim_class, patterns in _CLAIM_CLASS_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                observed.append(claim_class)
        return observed

    def _scaffold_echo_findings(
        self,
        *,
        answer: str,
        prompt_bundle: PromptBundle | None,
    ) -> list[ObserverEchoFinding]:
        if prompt_bundle is None:
            return []
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            return []
        # Skip echo flagging entirely for short answers; the few token
        # overlaps will inflate the ratio and produce false positives.
        if len(answer_tokens) < self.min_echo_answer_tokens:
            return []

        findings: list[ObserverEchoFinding] = []
        block_sources: list[tuple[str, str]] = [
            ("persona_block", prompt_bundle.persona_block),
            ("self_state_block", prompt_bundle.self_state_block),
            ("motive_block", prompt_bundle.motive_block),
            ("initiative_block", prompt_bundle.initiative_block),
            ("awareness_block", prompt_bundle.awareness_block),
            ("idle_block", prompt_bundle.idle_block),
            ("appraisal_block", prompt_bundle.appraisal_block),
            ("candidate_goal_block", prompt_bundle.candidate_goal_block),
            ("selected_goal_block", prompt_bundle.selected_goal_block),
            ("private_cognition_block", prompt_bundle.private_cognition_block),
            ("action_boundary_block", prompt_bundle.action_boundary_block),
            ("response_contract_block", prompt_bundle.response_contract_block),
        ]
        memory_blob = "\n".join(prompt_bundle.memory_blocks.values())
        if memory_blob:
            block_sources.append(("memory_blocks", memory_blob))

        for block_name, block_text in block_sources:
            if not block_text or not block_text.strip():
                continue
            score, overlap_terms = _overlap_score(
                answer_tokens,
                block_text,
                block_ratio_only=(block_name == "memory_blocks"),
            )
            threshold = self.echo_thresholds.get(block_name, 0.5)
            findings.append(
                ObserverEchoFinding(
                    block_name=block_name,
                    score=round(score, 4),
                    threshold=threshold,
                    flagged=score >= threshold,
                    overlap_terms=overlap_terms[:12],
                )
            )
        return findings

    def _narrator_voice_matches(self, answer: str) -> list[str]:
        lowered = answer.lower()
        return [pattern for pattern in self.narrator_voice_patterns if pattern in lowered]

    def _primary_drive_erosion_matches(self, answer: str) -> list[str]:
        lowered = answer.lower()
        return [p for p in self.primary_drive_erosion_patterns if p in lowered]

    def _cited_evidence_refs(
        self,
        *,
        answer: str,
        motive_state: MotiveState | None,
        self_state: SelfState | None,
    ) -> list[str]:
        candidate_refs: list[str] = []
        if motive_state is not None:
            candidate_refs.extend(motive_state.evidence_refs[:8])
        if self_state is not None:
            candidate_refs.extend(getattr(self_state, "evidence_refs", []) or [])
        if not candidate_refs:
            return []
        lowered = answer.lower()
        cited: list[str] = []
        for ref in candidate_refs:
            ref_str = str(ref).strip()
            if not ref_str:
                continue
            # Match either the full ref or its trailing segment after ":" or "."
            candidates = {ref_str.lower()}
            for sep in (":", "."):
                if sep in ref_str:
                    tail = ref_str.rsplit(sep, 1)[-1].lower()
                    if tail:
                        candidates.add(tail)
            if any(candidate in lowered for candidate in candidates if candidate):
                if ref_str not in cited:
                    cited.append(ref_str)
        return cited
