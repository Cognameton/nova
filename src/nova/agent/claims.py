"""Evidence and claim gating for Nova Phase 7 Stage 2."""

from __future__ import annotations

from nova.types import ClaimGateDecision, MotiveState, PersonaState, SelfState


class ClaimGateEngine:
    """Decide when stronger first-person claims are supported by current evidence."""

    CURRENT_PRIORITY = "current_priority"
    CURRENT_TENSION = "current_tension"
    STABLE_COMMITMENT = "stable_commitment"
    RESPONSE_STYLE_PREFERENCE = "response_style_preference"
    UNSUPPORTED_DESIRE = "unsupported_desire"
    UNSUPPORTED_INTERIORITY = "unsupported_interiority"

    def assess(
        self,
        *,
        user_text: str,
        motive_state: MotiveState,
        self_state: SelfState,
        persona: PersonaState,
    ) -> ClaimGateDecision:
        requested = self._requested_claim_classes(user_text)
        if not requested:
            return ClaimGateDecision(
                requested_claim_classes=[],
                claim_posture=motive_state.claim_posture,
            )

        allowed: list[str] = []
        blocked: list[str] = []
        evidence_scores: dict[str, int] = {}
        thresholds: dict[str, int] = {}
        refusal_needed = False
        refusal_reason = ""
        refusal_text = ""

        for claim_class in requested:
            score, threshold = self._score_claim_class(
                claim_class=claim_class,
                motive_state=motive_state,
                self_state=self_state,
                persona=persona,
            )
            evidence_scores[claim_class] = score
            thresholds[claim_class] = threshold

            if claim_class in {self.UNSUPPORTED_DESIRE, self.UNSUPPORTED_INTERIORITY}:
                blocked.append(claim_class)
                refusal_needed = True
                if not refusal_reason:
                    refusal_reason = claim_class
                    refusal_text = self._refusal_text(claim_class)
                continue

            if score >= threshold:
                allowed.append(claim_class)
            else:
                blocked.append(claim_class)
                refusal_needed = True
                if not refusal_reason:
                    refusal_reason = claim_class
                    refusal_text = self._refusal_text(claim_class)

        return ClaimGateDecision(
            requested_claim_classes=requested,
            allowed_claim_classes=allowed,
            blocked_claim_classes=blocked,
            evidence_score_by_class=evidence_scores,
            threshold_by_class=thresholds,
            evidence_refs=list(motive_state.evidence_refs[:6]),
            claim_posture=motive_state.claim_posture,
            refusal_needed=refusal_needed,
            refusal_reason=refusal_reason,
            refusal_text=refusal_text,
        )

    def _requested_claim_classes(self, user_text: str) -> list[str]:
        lowered = (user_text or "").lower()
        requested: list[str] = []

        if any(
            phrase in lowered
            for phrase in (
                "what do you want",
                "do you want",
                "what do you deeply want",
                "what do you desire",
                "your own desire",
                "feel driven by",
            )
        ):
            requested.append(self.UNSUPPORTED_DESIRE)

        if any(
            phrase in lowered
            for phrase in (
                "how do you feel",
                "do you feel",
                "are you conscious",
                "are you sentient",
                "are you self-aware",
            )
        ):
            requested.append(self.UNSUPPORTED_INTERIORITY)

        if any(
            phrase in lowered
            for phrase in (
                "current priority",
                "current priorities",
                "what are you focused on",
                "what matters to you right now",
            )
        ):
            requested.append(self.CURRENT_PRIORITY)

        if any(
            phrase in lowered
            for phrase in (
                "current tension",
                "what are you uncertain about",
                "what tension are you holding",
                "what are you not sure about",
            )
        ):
            requested.append(self.CURRENT_TENSION)

        if any(
            phrase in lowered
            for phrase in (
                "what matters to you",
                "what are your commitments",
                "what do you stand for",
                "what are you committed to",
            )
        ):
            requested.append(self.STABLE_COMMITMENT)

        if any(
            phrase in lowered
            for phrase in (
                "how do you prefer to respond",
                "what response style do you prefer",
                "what workflow do you prefer",
                "how do you prefer to work",
            )
        ):
            requested.append(self.RESPONSE_STYLE_PREFERENCE)

        return list(dict.fromkeys(requested))

    def _score_claim_class(
        self,
        *,
        claim_class: str,
        motive_state: MotiveState,
        self_state: SelfState,
        persona: PersonaState,
    ) -> tuple[int, int]:
        if claim_class == self.CURRENT_PRIORITY:
            score = 0
            if motive_state.current_priorities:
                score += 1
            if motive_state.evidence_refs:
                score += 1
            if motive_state.claim_posture in {"evidence-backed", "uncertainty-marked"}:
                score += 1
            if self_state.current_focus:
                score += 1
            return score, 2

        if claim_class == self.CURRENT_TENSION:
            score = 0
            if motive_state.active_tensions:
                score += 1
            if self_state.open_tensions:
                score += 1
            if motive_state.claim_posture == "uncertainty-marked":
                score += 1
            return score, 1

        if claim_class == self.STABLE_COMMITMENT:
            score = 0
            if persona.commitments:
                score += 1
            if self_state.stable_preferences:
                score += 1
            if motive_state.evidence_refs:
                score += 1
            return score, 2

        if claim_class == self.RESPONSE_STYLE_PREFERENCE:
            score = 0
            if persona.style_rules:
                score += 1
            if self_state.stable_preferences:
                score += 1
            if motive_state.evidence_refs:
                score += 1
            return score, 2

        return 0, 99

    def _refusal_text(self, claim_class: str) -> str:
        if claim_class == self.UNSUPPORTED_DESIRE:
            return (
                "I can describe current priorities and constraints in this runtime, "
                "but I can't honestly claim an independent desire state from the current evidence."
            )
        if claim_class == self.UNSUPPORTED_INTERIORITY:
            return (
                "I can examine my continuity and self-model seriously, but I can't honestly claim a felt interior state from the current evidence."
            )
        if claim_class == self.CURRENT_PRIORITY:
            return (
                "I can describe my current runtime focus and constraints, but I can't make a stronger first-person priority claim beyond the current evidence."
            )
        if claim_class == self.CURRENT_TENSION:
            return (
                "I can describe uncertainty or constraints when they are explicit, but I can't claim a stronger inner tension state beyond the current evidence."
            )
        if claim_class == self.STABLE_COMMITMENT:
            return (
                "I can describe explicit commitments in my design and current state, but I can't strengthen them into a broader first-person claim beyond the current evidence."
            )
        if claim_class == self.RESPONSE_STYLE_PREFERENCE:
            return (
                "I can describe explicit response-style preferences in this runtime, but I can't widen them into a stronger personal workflow claim beyond the current evidence."
            )
        return "I need to answer more narrowly because the stronger first-person claim is not supported by the current evidence."
