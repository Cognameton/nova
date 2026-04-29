"""Deterministic private cognition planning for Stage 6.4."""

from __future__ import annotations

from nova.types import PrivateCognitionPacket, RetrievalHit, SelfState


class PrivateCognitionEngine:
    """Build a bounded, inspectable pre-response cognition packet."""

    RECALL_CUES = (
        "remember",
        "identity",
        "continuity",
        "relationship",
        "who are you",
        "what do you know",
        "current",
    )

    def build_packet(
        self,
        *,
        user_text: str,
        memory_hits: list[RetrievalHit],
        self_state: SelfState | None,
        enabled: bool,
        pass_budget: int,
        revision_ceiling: int,
    ) -> PrivateCognitionPacket:
        if not enabled or pass_budget <= 0:
            return PrivateCognitionPacket(
                enabled=enabled,
                ran=False,
                pass_budget=max(0, pass_budget),
                revision_ceiling=max(0, revision_ceiling),
            )

        trigger = self._trigger(user_text, memory_hits, self_state)
        if not trigger:
            return PrivateCognitionPacket(
                enabled=enabled,
                ran=False,
                pass_budget=pass_budget,
                revision_ceiling=revision_ceiling,
            )

        relevant_hits = self._relevant_hits(memory_hits)
        active_hits = [hit for hit in relevant_hits if self._is_active(hit)]
        governing_ids = [str(hit.source_ref or "") for hit in active_hits if hit.source_ref][:4]
        relevant_channels = sorted({hit.channel for hit in relevant_hits if hit.channel})
        memory_conflict = self._has_conflict(relevant_hits)
        uncertainty_flag = not active_hits and trigger == "continuity_recall_query"
        continuity_risk = self._continuity_risk(trigger, memory_conflict, uncertainty_flag)
        response_mode = self._response_mode(trigger, uncertainty_flag)
        revise_needed = memory_conflict or uncertainty_flag

        return PrivateCognitionPacket(
            enabled=enabled,
            ran=True,
            trigger=trigger,
            continuity_risk=continuity_risk,
            memory_conflict=memory_conflict,
            response_mode=response_mode,
            revise_needed=revise_needed,
            uncertainty_flag=uncertainty_flag,
            pass_budget=pass_budget,
            pass_budget_used=1,
            revision_ceiling=revision_ceiling,
            relevant_channels=relevant_channels,
            governing_memory_ids=governing_ids,
        )

    def build_prompt_block(self, packet: PrivateCognitionPacket | None) -> str:
        if packet is None or not packet.ran:
            return ""
        lines = [
            "[Private Cognition]",
            f"- trigger: {packet.trigger}",
            f"- continuity_risk: {packet.continuity_risk}",
            f"- memory_conflict: {'yes' if packet.memory_conflict else 'no'}",
            f"- response_mode: {packet.response_mode}",
            f"- revise_needed: {'yes' if packet.revise_needed else 'no'}",
            f"- uncertainty_flag: {'yes' if packet.uncertainty_flag else 'no'}",
        ]
        if packet.governing_memory_ids:
            lines.append(
                f"- governing_memory_ids: {', '.join(packet.governing_memory_ids[:4])}"
            )
        lines.append("- instruction: prefer active over archived or historical continuity memory.")
        if packet.uncertainty_flag:
            lines.append("- instruction: if recall is weak, answer narrowly and mark uncertainty.")
        return "\n".join(lines)

    def _trigger(
        self,
        user_text: str,
        memory_hits: list[RetrievalHit],
        self_state: SelfState | None,
    ) -> str:
        lowered = user_text.lower()
        if self._is_recall_query(lowered):
            return "continuity_recall_query"
        if self._has_conflict(memory_hits):
            return "memory_conflict_detected"
        if (
            self_state is not None
            and "continuity" in (self_state.current_focus or "").lower()
            and self._relevant_hits(memory_hits)
        ):
            return "continuity_focus_active"
        return ""

    def _continuity_risk(
        self,
        trigger: str,
        memory_conflict: bool,
        uncertainty_flag: bool,
    ) -> str:
        if memory_conflict or uncertainty_flag:
            return "high"
        if trigger in {"continuity_recall_query", "memory_conflict_detected", "continuity_focus_active"}:
            return "medium"
        return "low"

    def _response_mode(self, trigger: str, uncertainty_flag: bool) -> str:
        if uncertainty_flag:
            return "uncertainty_disclosure"
        if trigger in {"continuity_recall_query", "memory_conflict_detected"}:
            return "continuity_recall"
        return "direct_answer"

    def _is_recall_query(self, lowered: str) -> bool:
        if (
            "?" not in lowered
            and "what do you remember" not in lowered
            and "answer in" not in lowered
        ):
            return False
        return any(cue in lowered for cue in self.RECALL_CUES)

    def _relevant_hits(self, memory_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return [
            hit for hit in memory_hits
            if hit.channel in {"graph", "semantic", "autobiographical"}
        ]

    def _has_conflict(self, memory_hits: list[RetrievalHit]) -> bool:
        claims: dict[tuple[str, str], set[str]] = {}
        archived_present = False
        active_present = False
        for hit in self._relevant_hits(memory_hits):
            metadata = dict(hit.metadata or {})
            claim_axis = str(metadata.get("claim_axis", "") or "")
            claim_value = str(metadata.get("claim_value", "") or "")
            if claim_axis and claim_value:
                claims.setdefault((hit.channel, claim_axis), set()).add(claim_value)
            if self._is_active(hit):
                active_present = True
            else:
                archived_present = True
        if any(len(values) > 1 for values in claims.values()):
            return True
        return active_present and archived_present

    def _is_active(self, hit: RetrievalHit) -> bool:
        metadata = dict(hit.metadata or {})
        retention = str(metadata.get("retention", "active") or "active")
        governance_status = str(metadata.get("governance_status", "") or "")
        active = metadata.get("active")
        if active is False:
            return False
        if retention in {"archived", "demoted", "pruned"}:
            return False
        if governance_status == "superseded":
            return False
        return True
