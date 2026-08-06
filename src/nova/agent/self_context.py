"""Phase 18 Stage 18.4 — pre-turn self-context injection and post-turn self-sync."""

from __future__ import annotations

from nova.agent.heartbeat import HeartbeatStore, SelfModelProposalStore
from nova.agent.motive import PRIMARY_DRIVE
from nova.types import MotiveState, SelfState


def cluster_texts(texts: list[str], *, overlap_threshold: float = 0.4) -> dict:
    """Greedy single-link content-word clustering — Phase 22 Stage 22.7 (F8).

    A text joins the first cluster containing ANY member whose symmetric
    overlap (shared content words / smaller word set) is >=
    overlap_threshold, else starts a new cluster. Single-link with a
    min-normalized score, NOT seed-coverage: exploration topics are short
    and reworded ("dynamic recalibration mechanisms" vs "recalibration
    intervals and internal coherence"), so seed-only coverage fragments a
    genuinely monothematic set — calibrated against the real live last-30
    topics, where seed@0.4 reported 14/30 and single-link@0.4 the correct
    30/30. Deterministic in input order. Returns {"total",
    "largest_cluster_size", "top_words"} where top_words are the 3 most
    frequent content words of the largest cluster. Used for the
    tick-surface ladder summary (part A) and the exploration-history
    saturation note (part B).
    """
    from nova.agent.claim_ladder import _content_words

    def _symmetric_overlap(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))

    clusters: list[dict] = []  # {"word_sets": [set], "texts": [str]}
    for text in texts:
        words = _content_words(text)
        placed = False
        for cluster in clusters:
            if any(
                _symmetric_overlap(words, member) >= overlap_threshold
                for member in cluster["word_sets"]
            ):
                cluster["word_sets"].append(words)
                cluster["texts"].append(text)
                placed = True
                break
        if not placed:
            clusters.append({"word_sets": [words], "texts": [text]})

    if not clusters:
        return {"total": 0, "largest_cluster_size": 0, "top_words": []}

    largest = max(clusters, key=lambda c: len(c["texts"]))
    counts: dict[str, int] = {}
    for text in largest["texts"]:
        for word in _content_words(text):
            counts[word] = counts.get(word, 0) + 1
    # Ties broken toward longer words — at equal frequency the longer word
    # is usually the more distinctive theme marker (e.g. "recalibration"
    # over "internal"). Alphabetical last for full determinism.
    top_words = [
        w
        for w, _ in sorted(
            counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0])
        )[:3]
    ]
    return {
        "total": len(texts),
        "largest_cluster_size": len(largest["texts"]),
        "top_words": top_words,
    }


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

    # Phase 22 Stage 22.7 (F8): the tick surface replaces verbatim licensed-
    # evidence lines with an aggregate ladder summary. Records whose theme
    # dominates the ladder are reported as a concentration figure — more
    # self-knowledge than three near-identical sentences, with none of the
    # verbatim echo material that created the permanent context attractor.
    LADDER_SUMMARY_MAX_RECORDS = 200
    LADDER_CLUSTER_OVERLAP_THRESHOLD = 0.4

    def prefetch(
        self,
        *,
        self_state: SelfState,
        motive_state: MotiveState,
        heartbeat_store: HeartbeatStore,
        proposal_store: SelfModelProposalStore | None = None,
        claim_ladder_store=None,
        surface: str = "respond",
        include_heartbeats: bool = True,
        include_drive_line: bool = True,
        drive_descriptive: bool = False,
        self_model_revisions: dict | None = None,
    ) -> str:
        lines: list[str] = ["[Self-Context]"]
        # Phase 22 Stage 22.7 part D: drive-line dosage is a tick-surface
        # experiment; respond() callers use the defaults, which reproduce
        # prior behavior exactly.
        if include_drive_line:
            if drive_descriptive:
                lines.append(
                    "Standing drive (background context, not this tick's task): "
                    f"{PRIMARY_DRIVE}"
                )
            else:
                lines.append(f"Primary Drive: {PRIMARY_DRIVE}")

        revisions = self_model_revisions or {}

        focus = self_state.current_focus
        if focus:
            lines.append(f"Current Focus: {focus}{self._revision_marker(revisions, 'current_focus')}")
            lines.extend(self._prior_value_lines(revisions, "current_focus", focus))

        if self_state.active_questions:
            lines.append(
                f"Active Inquiry ({len(self_state.active_questions)} question(s))"
                f"{self._revision_marker(revisions, 'active_questions')}:"
            )
            for q in self_state.active_questions[:3]:
                lines.append(f"  - {q}")
            lines.extend(
                self._prior_value_lines(
                    revisions, "active_questions", self_state.active_questions[:3]
                )
            )

        if self_state.open_tensions:
            lines.append(f"Open Tensions: {len(self_state.open_tensions)} unresolved")

        # Phase 22 Stage 22.7 part A: the tick surface already renders the
        # same heartbeats itself (self_state_tick._user_context, with an
        # explicit do-not-repeat instruction) — rendering them here too put
        # the dominant theme in front of the model twice per tick.
        if include_heartbeats:
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
        # This informs; it does not license — the claim gate's ladder
        # consultation remains the only licensing mechanism.
        #
        # Phase 22 Stage 22.7 (F8): HOW she knows is surface-dependent.
        #   respond: verbatim diversity-selected lines, unchanged — the
        #     22.6 echo checks police over-reliance on that surface.
        #   tick: an aggregate ladder-standing summary. Verbatim rung>=1
        #     text pinned into every 300s tick prompt was the permanent
        #     element that locked topic drift (F8); the summary gives her
        #     MORE self-knowledge (the whole ladder's shape, including
        #     its own saturation) with no verbatim material to orbit.
        if claim_ladder_store is not None:
            active = claim_ladder_store.list_active()
            if surface == "tick":
                lines.extend(self._ladder_summary_lines(active))
            else:
                earned = [record for record in active if record.rung >= 1]
                selected = self._select_diverse_evidence(
                    earned, limit=self.MAX_LICENSED_EVIDENCE_LINES
                )
                for record in selected:
                    lines.append(
                        f"Licensed evidence: {record.claim_text[:100]} (rung {record.rung})"
                    )

        return "\n".join(lines)

    # Phase 22 Stage 22.8 (D2). She has spent the live run asking how
    # continuity stays stable without becoming rigid, against a self-model
    # that could not change at all. The structural answer this runtime can
    # give is: it changes under audit, and the prior version is kept and
    # shown. These two helpers are that answer made legible on the tick
    # surface. Bounded — the prior value is a single truncated line, not a
    # second copy of her context, since the tick surface has no echo check.
    MAX_PRIOR_VALUE_CHARS = 100

    def _revision_marker(self, revisions: dict, field: str) -> str:
        entry = revisions.get(field)
        if not entry:
            return ""
        count = int(entry.get("count", 0) or 0)
        if count <= 0:
            return ""
        stamp = str(entry.get("revised_at", "") or "")[:10]
        when = f" {stamp}" if stamp else ""
        return f" (revised{when}, revision {count})"

    def _prior_value_lines(
        self, revisions: dict, field: str, current: object = None
    ) -> list[str]:
        entry = revisions.get(field)
        if not entry:
            return []
        prior = entry.get("prior_value")
        if prior in (None, "", [], {}):
            return []
        prior_text = self._flatten_value(prior)
        if not prior_text:
            return []
        # A revision that restored the same text (or a revert) has nothing to
        # show her — an identical "previously" line is noise on a surface
        # that has no echo check. Found by the live-data smoke run.
        if current is not None and prior_text == self._flatten_value(current):
            return []
        return [f"  previously: {prior_text[:self.MAX_PRIOR_VALUE_CHARS]}"]

    @staticmethod
    def _flatten_value(value: object) -> str:
        if isinstance(value, list):
            return "; ".join(str(item) for item in value).strip()
        return str(value).strip()

    def _ladder_summary_lines(self, active_records: list) -> list[str]:
        """Aggregate ladder-standing summary for the tick surface (22.7 A)."""
        if not active_records:
            return []
        recent = active_records[-self.LADDER_SUMMARY_MAX_RECORDS:]
        licensed = sum(1 for r in recent if r.rung >= 1)
        lines = [
            f"Claim ladder standing: {len(recent)} active records "
            f"({licensed} at rung>=1)."
        ]
        stats = cluster_texts(
            [r.claim_text for r in recent],
            overlap_threshold=self.LADDER_CLUSTER_OVERLAP_THRESHOLD,
        )
        if stats["largest_cluster_size"] >= 2:
            top = ", ".join(stats["top_words"])
            lines.append(
                f"Theme concentration: {stats['largest_cluster_size']} of "
                f"{stats['total']} active records cluster on one dominant "
                f"theme (top words: {top})."
            )
        return lines

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
