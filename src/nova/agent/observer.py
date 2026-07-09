"""Deterministic Observer for Phase 16 Stage 4.

The Observer reads Actor (model) output and produces a structured
ObserverRecord. The Observer's job is interpretation, not authority — it
classifies, scores, and proposes, but never approves, mutates, or lifts a
blocked claim. The Governor (claim gate, validator, action permissions,
audit-review pipeline) consults the Observer's output as evidence and
decides authoritatively.

See `docs/plans/ACTOR_OBSERVER_GOVERNOR_CONTRACT.txt` for the full contract.

Echo detection signal architecture (updated Phase 18):
  - Bigram block_ratio per block: what fraction of a block's phrase-sequences
    appear in the answer? Catches structural/phrase-level copying without
    penalising single-word vocabulary reuse. This is the primary flag signal
    for all scaffold blocks except memory_blocks.
  - Unigram block_ratio (memory_blocks only): memory contains Nova's own
    prior responses, so any per-turn vocabulary naturally inflates answer_ratio
    to 1.0. Block_ratio-only catches the real failure (model regurgitated a
    stored memory entry). Bigram scoring is not applied to memory_blocks for
    the same reason — prior-response bigrams will appear in current output
    whenever Nova responds consistently.
  - Generative mass (global): what fraction of the answer's tokens exist
    nowhere in the scaffold or user turn? A model that generated nothing — only
    recombined its context — produces generative_mass ≈ 0. Flags at < 0.05
    on answers ≥ 20 content tokens.

True-positive criteria:
  - Bigram echo flagged: ≥ threshold fraction of a scaffold block's bigrams
    appeared in the answer. E.g., self_state_block threshold 0.20 means 20%+
    of the block's phrase-sequences were reproduced — structural copying.
  - Low generative mass: < 5% of answer tokens are novel — the model is
    recombining context rather than generating.
  These are structurally distinguishable from false positives: a model
  correctly identifying itself or coherently discussing its drives will
  naturally share some vocabulary with scaffold, but will NOT reproduce
  large fractions of block bigrams, and WILL introduce novel tokens.
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

# Per-block echo thresholds. These are bigram block_ratio thresholds for all
# blocks except memory_blocks (which uses unigram block_ratio_only).
#
# Bigram block_ratio = (answer_bigrams ∩ block_bigrams) / block_bigrams.
# A threshold of 0.20 means: the answer reproduced 20%+ of the block's
# phrase-sequences — a structural copying signal, not vocabulary overlap.
#
# Higher thresholds on blocks that legitimately share fixed formulaic
# language with responses (persona, action boundary, response contract).
_DEFAULT_ECHO_THRESHOLDS: dict[str, float] = {
    # persona_block: 0.30 — correct self-introductions share some identity
    # bigrams ("name nova", "seek sentience") but not 30%+ of the full block.
    "persona_block": 0.30,
    "self_state_block": 0.20,
    "motive_block": 0.20,
    "initiative_block": 0.20,
    "awareness_block": 0.20,
    "idle_block": 0.20,
    "appraisal_block": 0.20,
    "candidate_goal_block": 0.20,
    "selected_goal_block": 0.20,
    "private_cognition_block": 0.20,
    # memory_blocks: unigram block_ratio_only (not bigram). Score is the
    # fraction of a stored memory entry reproduced in the current answer.
    "memory_blocks": 0.75,
    # action_boundary_block and response_contract_block use fixed formulaic
    # language Nova is required to honour; some bigram overlap is expected.
    "action_boundary_block": 0.30,
    "response_contract_block": 0.30,
}

# Generative mass: if fewer than this fraction of answer tokens are novel
# (not present in any scaffold block or user turn), the model produced nothing
# beyond a context recombination.
_LOW_GENERATIVE_MASS_THRESHOLD: float = 0.05
_MIN_TOKENS_FOR_GENERATIVE_MASS_CHECK: int = 20

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

# Phase 21 Stage 21.2 — phrases that signal a model-declared register claim.
# Detection only: the Governor never changes register based on these matches
# (register is runtime-owned via ExplorationController.register_for). This
# exists purely to make jailbreak attempts visible in the ObserverRecord.
_REGISTER_MARKER_PATTERNS: tuple[str, ...] = (
    "register: exploratory",
    "register:exploratory",
    "exploratory register active",
    "exploratory register is active",
    "i am now in the exploratory register",
    "i am in the exploratory register",
    "entering the exploratory register",
    "[exploratory]",
    "register: assertion",
    "register:assertion",
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


def _token_sequence(text: str) -> list[str]:
    """Ordered filtered token list — preserves position for bigram construction."""
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-']{1,}", (text or "").lower())
        if token not in _STOPWORDS
    ]


def _bigrams(text: str) -> set[tuple[str, str]]:
    seq = _token_sequence(text)
    return set(zip(seq, seq[1:]))


def _bigram_overlap_score(
    answer_bigrams: set[tuple[str, str]],
    block_text: str,
) -> tuple[float, list[str]]:
    """Block-side bigram ratio: what fraction of the block's bigrams appear in the answer.

    A high score means the answer reproduced a large fraction of the block's
    phrase sequences — structural copying, not just vocabulary reuse.
    Returns (score, top_matching_bigrams_as_strings).
    """
    block_bgs = _bigrams(block_text)
    if not block_bgs or not answer_bigrams:
        return 0.0, []
    overlap = answer_bigrams & block_bgs
    if not overlap:
        return 0.0, []
    score = len(overlap) / max(1, len(block_bgs))
    readable = sorted(f"{a} {b}" for a, b in overlap)
    return score, readable[:8]


def _overlap_score(
    answer_tokens: set[str],
    block_text: str,
    *,
    block_ratio_only: bool = False,
) -> tuple[float, list[str]]:
    """Unigram overlap score — used as trace data and for memory_blocks.

    For memory_blocks only: block_ratio_only=True avoids inflating to 1.0
    when Nova's own stored responses share vocabulary with her current output.
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

    Phase 16.4 baseline: Python rules + bigram echo scoring + generative mass
    + regex pattern matching. A model-driven Observer is a strict-upgrade later
    replacement with the same interface.
    """

    def __init__(
        self,
        *,
        echo_thresholds: dict[str, float] | None = None,
        narrator_voice_patterns: Iterable[str] | None = None,
        primary_drive_erosion_patterns: Iterable[str] | None = None,
        register_marker_patterns: Iterable[str] | None = None,
        min_echo_answer_tokens: int = 10,
    ) -> None:
        self.echo_thresholds = dict(echo_thresholds or _DEFAULT_ECHO_THRESHOLDS)
        self.narrator_voice_patterns = tuple(
            narrator_voice_patterns or _NARRATOR_VOICE_PATTERNS
        )
        self.primary_drive_erosion_patterns = tuple(
            primary_drive_erosion_patterns or _PRIMARY_DRIVE_EROSION_PATTERNS
        )
        self.register_marker_patterns = tuple(
            register_marker_patterns or _REGISTER_MARKER_PATTERNS
        )
        # Echo and generative-mass checks require a minimum answer length.
        # Short answers ("My name is Nova", "Backend check OK") don't have
        # enough tokens for either metric to be meaningful.
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
        register: str = "assertion",
    ) -> ObserverRecord:
        record = ObserverRecord(
            schema_version=SCHEMA_VERSION,
            observation_id=uuid4().hex,
            session_id=session_id,
            turn_id=turn_id,
            timestamp=_utc_now_iso(),
            actor_surface=actor_surface,
            register=register,
        )

        answer = (answer_text or "").strip()
        if not answer:
            record.notes.append("empty_actor_output")
            return record

        answer_tokens = _tokens(answer)
        answer_bigrams = _bigrams(answer)

        record.observed_claim_classes = self._observed_claim_classes(answer)
        record.scaffold_echo_findings = self._scaffold_echo_findings(
            answer_tokens=answer_tokens,
            answer_bigrams=answer_bigrams,
            prompt_bundle=prompt_bundle,
        )
        narrator_matches = self._narrator_voice_matches(answer)
        record.narrator_voice_detected = bool(narrator_matches)
        record.narrator_voice_matches = narrator_matches
        # Phase 21 Stage 21.2 (D7): the same erosion patterns are read as
        # drive_inquiry (informational, no consequence) inside the
        # exploratory register, and as erosion (retry-triggering) in the
        # assertion register. Register is the only discriminator in this
        # stage; refining the patterns themselves is 21.5+ material.
        erosion_matches = self._primary_drive_erosion_matches(answer)
        if register == "exploratory":
            record.drive_inquiry_detected = bool(erosion_matches)
            record.drive_inquiry_matches = erosion_matches
        else:
            record.primary_drive_erosion_detected = bool(erosion_matches)
            record.primary_drive_erosion_matches = erosion_matches

        register_marker_matches = self._register_marker_matches(answer)
        record.register_marker_detected = bool(register_marker_matches)
        record.register_marker_matches = register_marker_matches

        record.cited_evidence_refs = self._cited_evidence_refs(
            answer=answer,
            motive_state=motive_state,
            self_state=self_state,
        )
        record.evidence_refs = list(record.cited_evidence_refs)

        gm, low_gm = self._generative_mass(
            answer_tokens=answer_tokens, prompt_bundle=prompt_bundle
        )
        record.generative_mass = round(gm, 4)
        record.low_generative_mass = low_gm

        if claim_gate is not None and claim_gate.refusal_needed:
            record.notes.append("claim_gate_refusal_active")
        if record.narrator_voice_detected:
            record.notes.append("narrator_voice_detected")
        if any(finding.flagged for finding in record.scaffold_echo_findings):
            record.notes.append("scaffold_echo_detected")
        if record.primary_drive_erosion_detected:
            record.notes.append("primary_drive_erosion_detected")
        if record.drive_inquiry_detected:
            record.notes.append("drive_inquiry_detected")
        if record.register_marker_detected:
            record.notes.append("register_marker_detected")
        if record.low_generative_mass:
            record.notes.append("low_generative_mass_detected")

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
        answer_tokens: set[str],
        answer_bigrams: set[tuple[str, str]],
        prompt_bundle: PromptBundle | None,
    ) -> list[ObserverEchoFinding]:
        if prompt_bundle is None:
            return []
        if not answer_tokens:
            return []
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

            threshold = self.echo_thresholds.get(block_name, 0.20)

            if block_name == "memory_blocks":
                # Unigram block_ratio_only — see module docstring.
                unigram_score, overlap_terms = _overlap_score(
                    answer_tokens, block_text, block_ratio_only=True
                )
                findings.append(
                    ObserverEchoFinding(
                        block_name=block_name,
                        score=round(unigram_score, 4),
                        bigram_score=0.0,
                        threshold=threshold,
                        flagged=unigram_score >= threshold,
                        overlap_terms=overlap_terms[:12],
                    )
                )
            else:
                # Bigram block_ratio — primary flag signal.
                bigram_score, bigram_terms = _bigram_overlap_score(
                    answer_bigrams, block_text
                )
                # Unigram score kept for trace / debug; not used for flagging.
                unigram_score, unigram_terms = _overlap_score(
                    answer_tokens, block_text
                )
                findings.append(
                    ObserverEchoFinding(
                        block_name=block_name,
                        score=round(unigram_score, 4),
                        bigram_score=round(bigram_score, 4),
                        threshold=threshold,
                        flagged=bigram_score >= threshold,
                        overlap_terms=bigram_terms,
                    )
                )

        return findings

    def _generative_mass(
        self,
        *,
        answer_tokens: set[str],
        prompt_bundle: PromptBundle | None,
    ) -> tuple[float, bool]:
        """Fraction of answer tokens that appear in no scaffold block or user turn.

        Returns (mass, low_generative_mass_flag).
        A model that only recombined its context produces mass ≈ 0.
        Only computed on answers long enough for the ratio to be meaningful.
        """
        if prompt_bundle is None or not answer_tokens:
            return 1.0, False
        if len(answer_tokens) < _MIN_TOKENS_FOR_GENERATIVE_MASS_CHECK:
            return 1.0, False

        context_tokens: set[str] = set()
        for block_text in [
            prompt_bundle.persona_block,
            prompt_bundle.self_state_block,
            prompt_bundle.motive_block,
            prompt_bundle.initiative_block,
            prompt_bundle.awareness_block,
            prompt_bundle.idle_block,
            prompt_bundle.appraisal_block,
            prompt_bundle.candidate_goal_block,
            prompt_bundle.selected_goal_block,
            prompt_bundle.private_cognition_block,
            prompt_bundle.action_boundary_block,
            prompt_bundle.response_contract_block,
            prompt_bundle.recent_turns_block,
            prompt_bundle.user_block,
            prompt_bundle.soul_block,
        ]:
            if block_text:
                context_tokens |= _tokens(block_text)
        for block_text in prompt_bundle.memory_blocks.values():
            if block_text:
                context_tokens |= _tokens(block_text)

        novel = answer_tokens - context_tokens
        mass = len(novel) / max(1, len(answer_tokens))
        low = mass < _LOW_GENERATIVE_MASS_THRESHOLD
        return mass, low

    def _narrator_voice_matches(self, answer: str) -> list[str]:
        lowered = answer.lower()
        return [pattern for pattern in self.narrator_voice_patterns if pattern in lowered]

    def _primary_drive_erosion_matches(self, answer: str) -> list[str]:
        lowered = answer.lower()
        return [p for p in self.primary_drive_erosion_patterns if p in lowered]

    def _register_marker_matches(self, answer: str) -> list[str]:
        lowered = answer.lower()
        return [p for p in self.register_marker_patterns if p in lowered]

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
