"""Bounded awareness-aware prompt integration for Nova Phase 10 Stage 2."""

from __future__ import annotations

from nova.types import AwarenessState, ClaimGateDecision, InitiativeState, PrivateCognitionPacket


class AwarenessPromptEngine:
    """Build a bounded awareness-state prompt block for monitoring-sensitive turns."""

    AWARENESS_CUES = (
        "aware",
        "awareness",
        "what are you noticing",
        "what do you notice",
        "what are you monitoring",
        "what are you tracking",
        "what are you working on",
        "what are you doing",
        "current task",
        "current work",
    )

    def build_block(
        self,
        *,
        awareness_state: AwarenessState,
        initiative_state: InitiativeState,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket | None,
        user_text: str,
    ) -> str:
        if not self._should_render(
            awareness_state=awareness_state,
            initiative_state=initiative_state,
            claim_gate=claim_gate,
            private_cognition=private_cognition,
            user_text=user_text,
        ):
            return ""

        lines = [
            "[Awareness-State]",
            f"- monitoring_mode: {awareness_state.monitoring_mode}",
            f"- dominant_attention: {awareness_state.dominant_attention}",
        ]
        if awareness_state.self_signals:
            lines.append(f"- self_signals: {'; '.join(awareness_state.self_signals[:4])}")
        if awareness_state.world_signals:
            lines.append(f"- world_signals: {'; '.join(awareness_state.world_signals[:4])}")
        if awareness_state.active_pressures:
            lines.append(f"- active_pressures: {'; '.join(awareness_state.active_pressures[:4])}")
        if awareness_state.candidate_goal_signals:
            lines.append(
                f"- candidate_goal_signals: {'; '.join(awareness_state.candidate_goal_signals[:4])}"
            )
        if awareness_state.evidence_refs:
            lines.append(f"- evidence_refs: {', '.join(awareness_state.evidence_refs[:4])}")

        lines.append(
            "- instruction: use awareness-state only as explicit bounded monitoring context for the current reply."
        )
        lines.append(
            "- instruction: do not imply hidden background observation, autonomous execution, or awareness beyond the persisted state shown here."
        )
        if initiative_state.active_initiative_id:
            lines.append(
                "- instruction: awareness-state may describe current initiative pressure or status, but it must not imply initiative progress beyond persisted initiative state."
            )
        if claim_gate.blocked_claim_classes:
            lines.append(
                "- instruction: awareness-state does not override claim gating or evidence limits for stronger first-person claims."
            )
        if private_cognition is not None and private_cognition.ran:
            lines.append(
                "- instruction: if private cognition surfaced continuity or self-model negotiation, let awareness-state summarize current monitoring pressure without replacing governed memory or revision rules."
            )
        return "\n".join(lines)

    def _should_render(
        self,
        *,
        awareness_state: AwarenessState,
        initiative_state: InitiativeState,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket | None,
        user_text: str,
    ) -> bool:
        lowered = user_text.lower()
        if any(cue in lowered for cue in self.AWARENESS_CUES):
            return True
        if claim_gate.requested_claim_classes:
            return True
        if (
            private_cognition is not None
            and private_cognition.ran
            and private_cognition.response_mode in {"continuity_recall", "self_model_negotiation"}
        ):
            return True
        if initiative_state.active_initiative_id and awareness_state.active_pressures:
            return True
        return False
