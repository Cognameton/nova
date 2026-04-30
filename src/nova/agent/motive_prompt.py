"""Bounded motive-aware prompt integration for Nova Phase 7 Stage 3."""

from __future__ import annotations

from nova.types import ClaimGateDecision, MotiveState, PrivateCognitionPacket


class MotivePromptEngine:
    """Build a bounded motive-state prompt block for relevant turns only."""

    def build_block(
        self,
        *,
        motive_state: MotiveState,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket | None,
    ) -> str:
        if not self._should_render(claim_gate=claim_gate, private_cognition=private_cognition):
            return ""

        lines = [
            "[Motive-State]",
            f"- claim_posture: {motive_state.claim_posture}",
        ]
        if motive_state.current_priorities:
            lines.append(f"- current_priorities: {'; '.join(motive_state.current_priorities[:3])}")
        if motive_state.active_tensions:
            lines.append(f"- active_tensions: {'; '.join(motive_state.active_tensions[:3])}")
        if motive_state.local_goals:
            lines.append(f"- local_goals: {'; '.join(motive_state.local_goals[:3])}")
        if claim_gate.allowed_claim_classes:
            lines.append(
                f"- allowed_claim_classes: {', '.join(claim_gate.allowed_claim_classes[:4])}"
            )
        if claim_gate.blocked_claim_classes:
            lines.append(
                f"- blocked_claim_classes: {', '.join(claim_gate.blocked_claim_classes[:4])}"
            )
        if claim_gate.refusal_text:
            lines.append(f"- refusal_text: {claim_gate.refusal_text}")
        if claim_gate.evidence_refs:
            lines.append(f"- evidence_refs: {', '.join(claim_gate.evidence_refs[:4])}")

        lines.append(
            "- instruction: use stronger first-person claims only for allowed_claim_classes."
        )
        if claim_gate.blocked_claim_classes:
            lines.append(
                "- instruction: if a blocked claim class was requested, answer narrowly from explicit runtime evidence and state the limit directly."
            )
            lines.append(
                "- instruction: prefer the refusal_text directly, or stay very close to it, instead of generic contract language."
            )
        if private_cognition is not None and private_cognition.ran:
            lines.append(
                "- instruction: if continuity memory is active, let active continuity memory remain authoritative for recalled preferences and history."
            )
            lines.append(
                "- instruction: motive-state may add current orientation or constraints, but it must not replace governed continuity recall."
            )
        return "\n".join(lines)

    def _should_render(
        self,
        *,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket | None,
    ) -> bool:
        if claim_gate.requested_claim_classes:
            return True
        if (
            private_cognition is not None
            and private_cognition.ran
            and private_cognition.response_mode in {"continuity_recall", "self_model_negotiation"}
        ):
            return True
        return False
