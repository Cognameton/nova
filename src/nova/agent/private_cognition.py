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
    NEGOTIATION_CUES = (
        "what changed",
        "have changed",
        "used to",
        "do you still",
        "contradict",
        "conflict",
        "revision",
        "tension",
        "still negotiating",
        "still balancing",
        "uncertain",
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
        current_claims = self._current_claims(active_hits)
        relevant_channels = sorted({hit.channel for hit in relevant_hits if hit.channel})
        memory_conflict = self._has_conflict(relevant_hits)
        conflict_claim_axes = self._conflict_claim_axes(relevant_hits)
        provisional_claim_axes = self._provisional_claim_axes(active_hits)
        revision_notes = self._revision_notes(relevant_hits, conflict_claim_axes, provisional_claim_axes)
        uncertainty_flag = not active_hits and trigger == "continuity_recall_query"
        continuity_risk = self._continuity_risk(
            trigger,
            memory_conflict,
            uncertainty_flag,
            negotiation_needed=bool(conflict_claim_axes or provisional_claim_axes),
        )
        response_mode = self._response_mode(
            trigger,
            uncertainty_flag,
            negotiation_needed=bool(conflict_claim_axes or provisional_claim_axes),
        )
        revise_needed = memory_conflict or uncertainty_flag or bool(conflict_claim_axes or provisional_claim_axes)

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
            current_claims=current_claims,
            conflict_claim_axes=conflict_claim_axes,
            provisional_claim_axes=provisional_claim_axes,
            revision_notes=revision_notes,
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
        if packet.current_claims:
            lines.append(f"- current_claims: {'; '.join(packet.current_claims[:3])}")
        if packet.conflict_claim_axes:
            lines.append(f"- conflict_claim_axes: {', '.join(packet.conflict_claim_axes[:3])}")
        if packet.provisional_claim_axes:
            lines.append(f"- provisional_claim_axes: {', '.join(packet.provisional_claim_axes[:3])}")
        if packet.revision_notes:
            lines.append(f"- revision_notes: {'; '.join(packet.revision_notes[:3])}")
        lines.append("- instruction: when asked for the current state, use current_claims and active memory as authoritative.")
        lines.append("- instruction: treat archived or historical continuity memory as background only, not as the present state.")
        if packet.uncertainty_flag:
            lines.append("- instruction: if recall is weak, answer narrowly and mark uncertainty.")
        if packet.response_mode == "self_model_negotiation":
            lines.append("- instruction: when prior self-description conflicts with current active memory, mark the older claim as historical and state the current claim explicitly.")
            lines.append("- instruction: when a self-model claim remains provisional, say that it is still unsettled and do not upgrade it into a stable commitment.")
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
        if self._is_negotiation_query(lowered):
            return "self_model_negotiation_query"
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
        negotiation_needed: bool,
    ) -> str:
        if memory_conflict or uncertainty_flag or negotiation_needed:
            return "high"
        if trigger in {"continuity_recall_query", "memory_conflict_detected", "continuity_focus_active"}:
            return "medium"
        return "low"

    def _response_mode(self, trigger: str, uncertainty_flag: bool, negotiation_needed: bool) -> str:
        if uncertainty_flag:
            return "uncertainty_disclosure"
        if negotiation_needed:
            return "self_model_negotiation"
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

    def _is_negotiation_query(self, lowered: str) -> bool:
        return any(cue in lowered for cue in self.NEGOTIATION_CUES)

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

    def _conflict_claim_axes(self, memory_hits: list[RetrievalHit]) -> list[str]:
        by_axis: dict[str, set[str]] = {}
        historical_axes: set[str] = set()
        active_axes: set[str] = set()
        for hit in self._relevant_hits(memory_hits):
            metadata = dict(hit.metadata or {})
            if hit.channel != "autobiographical":
                continue
            axis = str(metadata.get("claim_axis", "") or "").strip()
            value = str(metadata.get("claim_value", "") or "").strip()
            if not axis:
                continue
            if value:
                by_axis.setdefault(axis, set()).add(value)
            if self._is_active(hit):
                active_axes.add(axis)
            else:
                historical_axes.add(axis)
        conflict_axes = {
            axis for axis, values in by_axis.items() if len(values) > 1
        }
        conflict_axes.update(axis for axis in active_axes if axis in historical_axes)
        return sorted(conflict_axes)

    def _provisional_claim_axes(self, memory_hits: list[RetrievalHit]) -> list[str]:
        axes: set[str] = set()
        for hit in memory_hits:
            metadata = dict(hit.metadata or {})
            if hit.channel != "autobiographical":
                continue
            if not self._is_active(hit):
                continue
            if str(metadata.get("self_model_status", "") or "") != "provisional":
                continue
            axis = str(metadata.get("claim_axis", "") or "").strip()
            if axis:
                axes.add(axis)
        return sorted(axes)

    def _revision_notes(
        self,
        memory_hits: list[RetrievalHit],
        conflict_claim_axes: list[str],
        provisional_claim_axes: list[str],
    ) -> list[str]:
        notes: list[str] = []
        for axis in conflict_claim_axes:
            active_value = ""
            historical_value = ""
            for hit in self._relevant_hits(memory_hits):
                metadata = dict(hit.metadata or {})
                if hit.channel != "autobiographical":
                    continue
                if str(metadata.get("claim_axis", "") or "") != axis:
                    continue
                value = str(metadata.get("claim_value", "") or "").strip()
                if self._is_active(hit) and value and not active_value:
                    active_value = value
                if not self._is_active(hit) and value and not historical_value:
                    historical_value = value
            if active_value and historical_value:
                notes.append(f"{axis} revised from {historical_value} to {active_value}")
            else:
                notes.append(f"{axis} has conflicting self-description history")
        for axis in provisional_claim_axes:
            notes.append(f"{axis} remains provisional")
        return notes[:4]

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

    def _current_claims(self, memory_hits: list[RetrievalHit]) -> list[str]:
        claims: list[str] = []
        seen: set[str] = set()
        for hit in memory_hits:
            metadata = dict(hit.metadata or {})
            claim_axis = str(metadata.get("claim_axis", "") or "").strip()
            claim_value = str(metadata.get("claim_value", "") or "").strip()
            if not claim_axis or not claim_value:
                continue
            claim = f"{claim_axis}={claim_value}"
            if claim in seen:
                continue
            seen.add(claim)
            claims.append(claim)
        return claims
