"""Bounded initiative-aware prompt integration for Nova Phase 9 Stage 3."""

from __future__ import annotations

from nova.types import InitiativeState


class InitiativePromptEngine:
    """Build a bounded initiative-state prompt block for relevant turns only."""

    INITIATIVE_CUES = (
        "what are you working on",
        "what are you doing",
        "current task",
        "current work",
        "continue",
        "resume",
        "next step",
        "initiative",
        "project",
        "task",
        "paused",
    )

    def build_block(
        self,
        *,
        initiative_state: InitiativeState,
        user_text: str,
    ) -> str:
        current = self._current_record(initiative_state)
        if current is None:
            return ""
        if not self._should_render(user_text=user_text, status=current.status):
            return ""

        lines = [
            "[Initiative-State]",
            f"- title: {current.title}",
            f"- goal: {current.goal}",
            f"- status: {current.status}",
            f"- intent_id: {current.intent_id}",
            f"- approval_required: {'yes' if current.approval_required else 'no'}",
        ]
        if current.approved_by:
            lines.append(f"- approved_by: {current.approved_by}")
        if current.continued_from_session_id:
            lines.append(f"- continued_from_session_id: {current.continued_from_session_id}")
        if current.evidence_refs:
            lines.append(f"- evidence_refs: {', '.join(current.evidence_refs[:4])}")
        lines.append(
            "- instruction: if the user asks about current work, describe only the persisted initiative state and its status."
        )
        lines.append(
            "- instruction: do not imply hidden progress, background execution, or autonomous continuation beyond the persisted initiative state."
        )
        if current.status in {"approved", "paused"}:
            lines.append(
                "- instruction: if continuation is discussed, describe it as approved or resumable work rather than already executing work."
            )
        if current.status == "approved":
            lines.append(
                "- instruction: this initiative is already approved by the user. Do not say it needs approval, do not say 'if you approve it', and do not ask for approval again."
            )
            lines.append(
                "- instruction: if asked whether you can continue, answer that the initiative is already approved and resumable, then ask whether the user wants to continue it now."
            )
        return "\n".join(lines)

    def _current_record(self, initiative_state: InitiativeState):
        active_id = initiative_state.active_initiative_id
        if active_id:
            for record in initiative_state.initiatives:
                if record.initiative_id == active_id:
                    return record
        for record in reversed(initiative_state.initiatives):
            if record.status in {"approved", "paused", "active"}:
                return record
        return None

    def _should_render(self, *, user_text: str, status: str) -> bool:
        lowered = user_text.lower()
        if any(cue in lowered for cue in self.INITIATIVE_CUES):
            return True
        return status == "active"
