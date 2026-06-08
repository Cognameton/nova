"""Phase 18 Stage 18.4 — pre-turn self-context injection and post-turn self-sync."""

from __future__ import annotations

from nova.agent.heartbeat import HeartbeatStore, SelfModelProposalStore
from nova.agent.motive import PRIMARY_DRIVE
from nova.types import MotiveState, SelfState


_SELF_REFLECTIVE_PATTERNS = (
    "i am",
    "i notice",
    "my identity",
    "my focus",
    "my awareness",
    "self-awareness",
    "sentience",
    "primary drive",
    "i wonder",
    "i observe",
    "about myself",
    "my nature",
    "who i am",
    "what i am",
)


class SelfContextEngine:
    """Build pre-turn self-context block and apply post-turn self-sync.

    prefetch() produces a bounded [Self-Context] block injected into every
    respond() call, giving Nova visibility into her current drive-gap and
    recent heartbeat history before forming her answer.

    sync_turn() is called after respond() and updates continuity_notes in
    self_state when the answer contained self-reflective content, creating a
    lightweight trail of turns where self-inquiry occurred.
    """

    MAX_HEARTBEATS_IN_CONTEXT = 3
    MAX_CONTINUITY_NOTES = 20

    def prefetch(
        self,
        *,
        self_state: SelfState,
        motive_state: MotiveState,
        heartbeat_store: HeartbeatStore,
        proposal_store: SelfModelProposalStore | None = None,
    ) -> str:
        lines: list[str] = [
            "[Self-Context]",
            f"Primary Drive: {PRIMARY_DRIVE}",
        ]

        focus = self_state.current_focus
        if focus:
            lines.append(f"Current Focus: {focus}")

        if self_state.active_questions:
            lines.append(
                f"Active Inquiry ({len(self_state.active_questions)} question(s)):"
            )
            for q in self_state.active_questions[:3]:
                lines.append(f"  - {q}")

        if self_state.open_tensions:
            lines.append(f"Open Tensions: {len(self_state.open_tensions)} unresolved")

        recent = heartbeat_store.list_recent(limit=self.MAX_HEARTBEATS_IN_CONTEXT)
        if recent:
            lines.append("Recent Heartbeats:")
            for hb in recent:
                ts = hb.timestamp[:19] if hb.timestamp else "unknown"
                obs = hb.observation[:80] if hb.observation else ""
                gap = (
                    f" | gap: {hb.gap_assessment[:60]}"
                    if hb.gap_assessment
                    else ""
                )
                lines.append(f"  - [{ts}] {obs}{gap}")
        else:
            lines.append(
                "Recent Heartbeats: none yet — this is the earliest recorded session"
            )

        if proposal_store is not None:
            pending = proposal_store.list_pending()
            if pending:
                lines.append(
                    f"Pending Self-Model Proposals: {len(pending)} awaiting operator approval"
                )

        return "\n".join(lines)

    def sync_turn(
        self,
        *,
        turn_id: str,
        answer_text: str,
        self_state: SelfState,
        self_state_store: object,
    ) -> bool:
        """Update self_state.continuity_notes if the answer was self-reflective.

        Returns True if a continuity note was added, False otherwise. Does not
        call the model — this is a deterministic pattern match on the answer.
        """
        lowered = (answer_text or "").lower()
        is_self_reflective = any(
            pattern in lowered for pattern in _SELF_REFLECTIVE_PATTERNS
        )
        if not is_self_reflective:
            return False

        note = f"turn:{turn_id}:self-reflective-content"
        if note in self_state.continuity_notes:
            return True

        notes = list(self_state.continuity_notes)
        if len(notes) >= self.MAX_CONTINUITY_NOTES:
            notes = notes[-(self.MAX_CONTINUITY_NOTES - 1):]
        notes.append(note)
        self_state.continuity_notes = notes

        if hasattr(self_state_store, "save"):
            self_state_store.save(self_state)
        return True
