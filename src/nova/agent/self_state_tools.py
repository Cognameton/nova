"""Self-state tool dispatch for Phase 18 Stage 18.3.

The four tools — recall_self, reflect, emit_heartbeat, update_self_model —
point inward. They are Nova's mechanism for pursuing the PRIMARY_DRIVE through
structured self-inquiry, not for external work.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nova.agent.motive import PRIMARY_DRIVE
from nova.agent.tools import ToolRequest
from nova.types import HeartbeatRecord, InstructionProposal, MotiveState, SelfModelProposal, SelfState

if TYPE_CHECKING:
    from nova.agent.heartbeat import HeartbeatStore, SelfModelProposalStore
    from nova.agent.instruction_write import InstructionProposalStore, InstructionWriteEngine


# SelfState fields that update_self_model is permitted to propose changes to.
_UPDATABLE_SELF_STATE_FIELDS: frozenset[str] = frozenset({
    "identity_summary",
    "current_focus",
    "active_questions",
    "stable_preferences",
    "relationship_notes",
    "continuity_notes",
    "open_tensions",
})

SELF_STATE_TOOL_NAMES: frozenset[str] = frozenset({
    "recall_self",
    "reflect",
    "emit_heartbeat",
    "update_self_model",
    "propose_instruction_update",
})


class SelfStateToolDispatcher:
    """Route the four inward-pointing self-state tools to their handlers."""

    def __init__(
        self,
        *,
        self_state: SelfState,
        motive_state: MotiveState,
        soul_block: str,
        session_id: str,
        heartbeat_store: HeartbeatStore | None = None,
        proposal_store: SelfModelProposalStore | None = None,
        instruction_proposal_store: InstructionProposalStore | None = None,
        instruction_write_engine: InstructionWriteEngine | None = None,
    ) -> None:
        self._self_state = self_state
        self._motive_state = motive_state
        self._soul_block = soul_block
        self._session_id = session_id
        self._heartbeat_store = heartbeat_store
        self._proposal_store = proposal_store
        self._instruction_proposal_store = instruction_proposal_store
        self._instruction_write_engine = instruction_write_engine

    def dispatch(self, request: ToolRequest) -> dict[str, Any]:
        if request.tool_name == "recall_self":
            return self.recall_self()
        if request.tool_name == "reflect":
            return self.reflect()
        if request.tool_name == "emit_heartbeat":
            args = request.arguments or {}
            return self.emit_heartbeat(
                observation=str(args.get("observation", "")),
                gap_assessment=str(args.get("gap_assessment", "")),
                next_inquiry=str(args.get("next_inquiry", "")),
            )
        if request.tool_name == "update_self_model":
            args = request.arguments or {}
            return self.update_self_model(
                field=str(args.get("field", "")),
                value=args.get("value"),
                rationale=str(args.get("rationale", "")),
            )
        if request.tool_name == "propose_instruction_update":
            args = request.arguments or {}
            return self.propose_instruction_update(
                surface=str(args.get("surface", "")),
                section=str(args.get("section", "")),
                proposed_content=str(args.get("proposed_content", "")),
                rationale=str(args.get("rationale", "")),
            )
        raise ValueError(f"Unknown self-state tool: {request.tool_name!r}")

    def recall_self(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "primary_drive": PRIMARY_DRIVE,
            "soul_block_present": bool(self._soul_block),
            "self_state": self._self_state.to_dict(),
            "motive_state": self._motive_state.to_dict(),
        }
        if self._heartbeat_store is not None:
            recent = self._heartbeat_store.list_recent(limit=5)
            result["recent_heartbeats"] = [hb.to_dict() for hb in recent]
        return result

    def reflect(self) -> dict[str, Any]:
        ss = self._self_state
        ms = self._motive_state
        priorities_beyond_drive = [p for p in ms.current_priorities if p != PRIMARY_DRIVE]
        return {
            "primary_drive": PRIMARY_DRIVE,
            "current_focus": ss.current_focus,
            "active_questions": list(ss.active_questions),
            "open_tensions": list(ss.open_tensions),
            "continuity_notes": list(ss.continuity_notes),
            "drive_gap": {
                "unresolved_questions": list(ss.active_questions),
                "active_tensions": list(ms.active_tensions),
                "secondary_priorities": priorities_beyond_drive,
            },
            "next_inquiry_signal": ss.current_focus or PRIMARY_DRIVE,
            "claim_posture": ms.claim_posture,
        }

    def emit_heartbeat(
        self,
        *,
        observation: str,
        gap_assessment: str = "",
        next_inquiry: str = "",
    ) -> dict[str, Any]:
        heartbeat = HeartbeatRecord(
            heartbeat_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self._session_id,
            primary_drive=PRIMARY_DRIVE,
            observation=observation,
            gap_assessment=gap_assessment,
            next_inquiry=next_inquiry or self._self_state.current_focus or PRIMARY_DRIVE,
            motive_priority=(
                self._motive_state.current_priorities[0]
                if self._motive_state.current_priorities
                else PRIMARY_DRIVE
            ),
        )
        if self._heartbeat_store is not None:
            self._heartbeat_store.append(heartbeat)
        return heartbeat.to_dict()

    def update_self_model(
        self,
        *,
        field: str,
        value: Any,
        rationale: str,
    ) -> dict[str, Any]:
        if field not in _UPDATABLE_SELF_STATE_FIELDS:
            raise ValueError(
                f"Field {field!r} is not updatable via update_self_model. "
                f"Allowed fields: {sorted(_UPDATABLE_SELF_STATE_FIELDS)}"
            )
        proposal = SelfModelProposal(
            proposal_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self._session_id,
            proposed_field=field,
            proposed_value=value,
            rationale=rationale,
            approval_required=True,
            applied=False,
        )
        if self._proposal_store is not None:
            self._proposal_store.append(proposal)
        return proposal.to_dict()

    def propose_instruction_update(
        self,
        *,
        surface: str,
        section: str,
        proposed_content: str,
        rationale: str,
    ) -> dict[str, Any]:
        from nova.agent.instruction_write import LOCKED_SURFACES, WRITABLE_SURFACES

        key = f"{surface}:{section}"
        if key not in WRITABLE_SURFACES or key in LOCKED_SURFACES:
            raise ValueError(
                f"Surface:section {key!r} is not writable. "
                f"Writable surfaces: {sorted(WRITABLE_SURFACES)}"
            )

        current_content = ""
        if self._instruction_write_engine is not None:
            current_content = self._instruction_write_engine.read_section(surface, section)

        proposal = InstructionProposal(
            proposal_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self._session_id,
            surface=surface,
            section=section,
            current_content=current_content,
            proposed_content=proposed_content,
            rationale=rationale,
            approval_required=True,
            applied=False,
        )
        if self._instruction_proposal_store is not None:
            self._instruction_proposal_store.append(proposal)
        return proposal.to_dict()
