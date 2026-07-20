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
    MAX_LICENSED_EVIDENCE_LINES = 3

    # Phase 22 Stage 22.6: blind top-N licensed-evidence selection meant
    # every prompt showed the same content whenever multiple active rung>=1
    # records shared a theme (observed live: 3 "recalibration intervals"
    # variants dominating every single turn). Calibrated against the real
    # pairwise overlaps measured on that live data (0.208-0.25) — deliberately
    # mid-range so this has a real effect on today's data, not a no-op tuned
    # to pass silently. A starting point, not a formula; revisit as more
    # distinct themes accumulate.
    LICENSED_EVIDENCE_DIVERSITY_THRESHOLD = 0.22

    def _select_diverse_evidence(self, records: list, limit: int) -> list:
        """Greedy diversity selection over candidate claim-ladder records.

        Keeps a record only if its bigram overlap against EVERY
        already-selected record is below LICENSED_EVIDENCE_DIVERSITY_
        THRESHOLD. Returning fewer than `limit` records when the
        candidates are too thematically similar is intended, not a bug —
        showing 1-2 honestly-distinct lines beats 3 lines restating the
        same theme.
        """
        from nova.eval.tick_analysis import _bigram_overlap

        selected: list = []
        for record in records:
            if len(selected) >= limit:
                break
            if all(
                _bigram_overlap(record.claim_text, chosen.claim_text)
                < self.LICENSED_EVIDENCE_DIVERSITY_THRESHOLD
                for chosen in selected
            ):
                selected.append(record)
        return selected

    def prefetch(
        self,
        *,
        self_state: SelfState,
        motive_state: MotiveState,
        heartbeat_store: HeartbeatStore,
        proposal_store: SelfModelProposalStore | None = None,
        claim_ladder_store=None,
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

        # Phase 21 Stage 21.5 (I1): Nova should know what she has earned.
        # One bounded line per ACTIVE ladder record at rung >= 1 — never
        # rung-0 hypotheses (register-only, unverified) and never demoted
        # records. This informs; it does not license — the claim gate's
        # ladder consultation remains the only licensing mechanism.
        # Phase 22 Stage 22.6: selection is diversity-aware rather than a
        # blind top-N slice — see _select_diverse_evidence.
        if claim_ladder_store is not None:
            earned = [
                record
                for record in claim_ladder_store.list_active()
                if record.rung >= 1
            ]
            selected = self._select_diverse_evidence(
                earned, limit=self.MAX_LICENSED_EVIDENCE_LINES
            )
            for record in selected:
                lines.append(
                    f"Licensed evidence: {record.claim_text[:100]} (rung {record.rung})"
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
