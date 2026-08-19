"""Self-state tool dispatch for Phase 18 Stage 18.3.

The four tools — recall_self, reflect, emit_heartbeat, update_self_model —
point inward. They are Nova's mechanism for pursuing the PRIMARY_DRIVE through
structured self-inquiry, not for external work.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nova.agent.exploration import PRODUCTION_TICK_INTERVAL_SECONDS
from nova.agent.motive import PRIMARY_DRIVE
from nova.agent.tools import ToolRequest
from nova.types import HeartbeatRecord, InstructionProposal, MotiveState, SelfModelProposal, SelfState

if TYPE_CHECKING:
    from nova.agent.exploration import ExplorationController
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

# Phase 22 Stage 22.8 — the updatable set is split on the Exploratory
# Register Contract's own line: GATE ASSERTIONS, NEVER INQUIRY.
#
# Inquiry-class fields say what Nova is attending to and asking. They assert
# nothing about what she is, nothing licenses off them, and they pass through
# no claim gate. Before 22.8 they were the most tightly gated fields in the
# system — update_self_model only ever queued a proposal, the approval path
# had no operator surface, and self_state.json consequently went unwritten
# for the entire 26-day live run while active_questions[0] was injected
# verbatim into every tick prompt. That inversion is what this split
# corrects.
#
# Assertion-class fields make first-person claims about her nature
# (identity_summary), assert durable wants (stable_preferences — the same
# class respond()'s claim gate polices), or describe third parties
# (relationship_notes). Those keep operator approval exactly as before.
NOVA_WRITABLE_SELF_STATE_FIELDS: frozenset[str] = frozenset({
    "current_focus",
    "active_questions",
    "open_tensions",
    "continuity_notes",
})

APPROVAL_GATED_SELF_STATE_FIELDS: frozenset[str] = frozenset({
    "identity_summary",
    "stable_preferences",
    "relationship_notes",
})

# A field added to _UPDATABLE_SELF_STATE_FIELDS later must land in exactly
# one class. Failing at import beats defaulting silently into either one.
assert (
    NOVA_WRITABLE_SELF_STATE_FIELDS | APPROVAL_GATED_SELF_STATE_FIELDS
) == _UPDATABLE_SELF_STATE_FIELDS, (
    "self-model field classes must exactly partition _UPDATABLE_SELF_STATE_FIELDS"
)
assert not (
    NOVA_WRITABLE_SELF_STATE_FIELDS & APPROVAL_GATED_SELF_STATE_FIELDS
), "self-model field classes must be disjoint"

# Pacing on Nova's own writes, not a gate on them. A proposal arriving inside
# the window is still recorded — unapplied, noted "rate_limited" — never
# dropped and never treated as illegitimate. Pinned to the known production
# cadence the same way Stage 22.2c pinned the exploration wall clock, rather
# than being another independently-chosen number that happens to collide.
SELF_MODEL_REVISION_MIN_TICKS = 12
SELF_MODEL_REVISION_MIN_SECONDS = (
    SELF_MODEL_REVISION_MIN_TICKS * PRODUCTION_TICK_INTERVAL_SECONDS
)


def apply_proposal_to_self_state(
    *,
    proposal: SelfModelProposal,
    self_state: SelfState,
    self_state_store: Any,
    proposal_store: Any,
    applied_by: str,
) -> SelfModelProposal | None:
    """Capture prior value, mutate SelfState, persist, mark applied.

    Stage 22.8: the single application path, shared by Nova's auto-apply and
    the operator's apply_self_model_proposal. Centralised so the two cannot
    drift, and so the operator path gains prior_value (and therefore
    revertibility) without a second implementation.

    Returns the updated stored record, or None if the field is not updatable.
    """
    field = proposal.proposed_field
    if field not in _UPDATABLE_SELF_STATE_FIELDS:
        return None

    prior_value = getattr(self_state, field, None)
    if isinstance(prior_value, list):
        prior_value = list(prior_value)

    # Stage 22.9: coerce by TARGET field type. Proposals routinely carry str
    # values for list[str] fields (every live pre-22.8 record does), and a
    # bare str set onto active_questions renders per-character in prefetch.
    # Her exact words are preserved: str -> [str], list -> "; "-joined str.
    value = proposal.proposed_value
    if isinstance(value, list):
        if isinstance(prior_value, str):
            setattr(self_state, field, "; ".join(str(item) for item in value))
        else:
            setattr(self_state, field, list(value))
    elif isinstance(value, str):
        if isinstance(prior_value, list):
            setattr(self_state, field, [str(value)])
        else:
            setattr(self_state, field, str(value))
    elif value is not None:
        setattr(self_state, field, value)

    # Stage 22.9: updated_at is the canonical "has the self-model moved"
    # indicator (finding F9 hinged on it) — an applied revision must move it.
    applied_at = datetime.now(timezone.utc).isoformat()
    self_state.updated_at = applied_at

    if self_state_store is not None and hasattr(self_state_store, "save"):
        self_state_store.save(self_state)
    if proposal_store is None:
        return proposal
    return proposal_store.mark_applied(
        proposal.proposal_id,
        applied_at,
        prior_value=prior_value,
        applied_by=applied_by,
    )

SELF_STATE_TOOL_NAMES: frozenset[str] = frozenset({
    "recall_self",
    "reflect",
    "emit_heartbeat",
    "update_self_model",
    "propose_instruction_update",
    "enter_exploration",
    "close_exploration",
    "recall_history",
})

# Stage 22.10: bounded self-history reads. Fixed count and per-entry cap so
# a read can never flood the tick context; the result always carries the
# true total so a window reads as a window, not as everything.
RECALL_HISTORY_COUNT = 8
RECALL_HISTORY_ENTRY_CHARS = 140
RECALL_HISTORY_SOURCES = ("heartbeats", "explorations", "findings")
RECALL_HISTORY_MODES = ("recent", "earliest", "sample", "around")


def _select_history_entries(
    entries: list[tuple[str, str]], mode: str, around: str
) -> list[tuple[str, str]]:
    """Deterministic window selection (no RNG — same rationale as 22.8 D1)."""
    n = RECALL_HISTORY_COUNT
    if mode == "recent":
        return entries[-n:]
    if mode == "earliest":
        return entries[:n]
    if mode == "sample":
        if len(entries) <= n:
            return list(entries)
        step = (len(entries) - 1) / (n - 1)
        indices = sorted({round(i * step) for i in range(n)})
        return [entries[i] for i in indices]
    # mode == "around": window centered on the first entry at/after the date
    pivot = next(
        (i for i, entry in enumerate(entries) if (entry[0] or "") >= around),
        max(0, len(entries) - 1),
    )
    start = max(0, pivot - n // 2)
    return entries[start:start + n]


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
        exploration_controller: ExplorationController | None = None,
        self_state_store: Any = None,
        self_model_writes_enabled: bool = False,
        revision_min_seconds: float = SELF_MODEL_REVISION_MIN_SECONDS,
        claim_ladder_store: Any = None,
    ) -> None:
        self._self_state = self_state
        # Stage 22.8 — needed to persist Nova's own inquiry-class writes.
        # Defaults keep every pre-22.8 construction site (and the whole test
        # suite) on the queue-only behavior.
        self._self_state_store = self_state_store
        self._self_model_writes_enabled = self_model_writes_enabled
        self._revision_min_seconds = revision_min_seconds
        self._motive_state = motive_state
        self._soul_block = soul_block
        self._session_id = session_id
        self._heartbeat_store = heartbeat_store
        self._proposal_store = proposal_store
        self._instruction_proposal_store = instruction_proposal_store
        self._instruction_write_engine = instruction_write_engine
        self._exploration_controller = exploration_controller
        # Stage 22.10 — read-only; findings source degrades gracefully when
        # absent so every pre-22.10 construction site keeps working.
        self._claim_ladder_store = claim_ladder_store

    def dispatch(self, request: ToolRequest) -> dict[str, Any]:
        if request.tool_name == "recall_self":
            return self.recall_self()
        if request.tool_name == "reflect":
            return self.reflect()
        if request.tool_name == "recall_history":
            return self.recall_history(
                source=str(request.arguments.get("source", "") or ""),
                mode=str(request.arguments.get("mode", "") or ""),
                around=str(request.arguments.get("around", "") or ""),
            )
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
        if request.tool_name == "enter_exploration":
            args = request.arguments or {}
            return self.enter_exploration(
                topic=str(args.get("topic", "")),
                rationale=str(args.get("rationale", "")),
            )
        if request.tool_name == "close_exploration":
            args = request.arguments or {}
            return self.close_exploration(
                findings_summary=str(args.get("findings_summary", "")),
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

    def recall_history(
        self, *, source: str, mode: str = "", around: str = ""
    ) -> dict[str, Any]:
        """Stage 22.10 — bounded, deterministic read over Nova's own record.

        Sources are membrane-safe in both registers by construction:
        heartbeats are a single cross-register store already rendered on
        both surfaces; explorations expose topic/date/outcome metadata only
        (no findings text); findings are claim-ladder records, every one of
        which passed governed export's assertion-register gate at creation.
        Raw journal deep-read is deliberately absent (un-exported
        in-register material must not reach assertion ticks).
        """
        source = (source or "").strip().lower()
        around = (around or "").strip()
        mode = "around" if around else ((mode or "recent").strip().lower())
        if source not in RECALL_HISTORY_SOURCES:
            return {
                "error": "unknown_source",
                "note": f"source must be one of: {', '.join(RECALL_HISTORY_SOURCES)}",
            }
        if mode not in RECALL_HISTORY_MODES:
            return {
                "error": "unknown_mode",
                "note": "mode must be one of: recent, earliest, sample"
                " (or pass 'around' as YYYY-MM-DD)",
            }

        entries: list[tuple[str, str]] = []
        note = ""
        if source == "heartbeats":
            if self._heartbeat_store is not None:
                records = self._heartbeat_store.list_recent(limit=0)
                entries = [(r.timestamp or "", r.observation or "") for r in records]
        elif source == "explorations":
            if self._exploration_controller is not None:
                records = self._exploration_controller.store.list_all()
                entries = [
                    (
                        r.opened_at or "",
                        f"{r.topic} "
                        f"({'closed: ' + r.close_reason if r.close_reason else r.status})",
                    )
                    for r in records
                ]
        else:  # findings
            if self._claim_ladder_store is not None:
                records = self._claim_ladder_store.list_active()
                entries = [
                    (r.created_at or "", f"(rung {r.rung}) {r.claim_text}")
                    for r in records
                ]
            else:
                note = "findings are not available on this surface"

        selected = _select_history_entries(entries, mode, around)
        result: dict[str, Any] = {
            "source": source,
            "mode": mode,
            "total": len(entries),
            "entries": [
                {
                    "timestamp": (ts or "")[:19],
                    "text": text[:RECALL_HISTORY_ENTRY_CHARS],
                }
                for ts, text in selected
            ],
        }
        if around:
            result["around"] = around
        if note:
            result["note"] = note
        return result

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
        # Stage 22.8: inquiry-class fields are Nova's to write. Assertion-class
        # fields keep operator approval. Every call still produces a stored
        # proposal record either way, so the audit trail stays uniform and no
        # existing reader of this store changes shape.
        nova_writable = (
            self._self_model_writes_enabled
            and field in NOVA_WRITABLE_SELF_STATE_FIELDS
        )
        rate_limited_note = ""
        if nova_writable:
            rate_limited_note = self._revision_rate_limit_note(field)

        proposal = SelfModelProposal(
            proposal_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self._session_id,
            proposed_field=field,
            proposed_value=value,
            rationale=rationale,
            approval_required=not nova_writable,
            applied=False,
            auto_applied=bool(nova_writable and not rate_limited_note),
            note=rate_limited_note,
        )
        if self._proposal_store is not None:
            self._proposal_store.append(proposal)

        if nova_writable and not rate_limited_note:
            applied = apply_proposal_to_self_state(
                proposal=proposal,
                self_state=self._self_state,
                self_state_store=self._self_state_store,
                proposal_store=self._proposal_store,
                applied_by="nova",
            )
            if applied is not None:
                return applied.to_dict()
        return proposal.to_dict()

    def _revision_rate_limit_note(self, field: str) -> str:
        """Return a note if this field moved too recently, else "".

        Pacing only. The caller still records the proposal — a rate-limited
        revision is preserved as evidence that she wanted to move the field,
        which is exactly the signal the frozen-self-model era destroyed.
        """
        if self._proposal_store is None:
            return ""
        if not hasattr(self._proposal_store, "last_applied_for_field"):
            return ""
        previous = self._proposal_store.last_applied_for_field(field)
        if previous is None:
            return ""
        stamp = previous.applied_at or previous.timestamp
        if not stamp:
            return ""
        try:
            applied_at = datetime.fromisoformat(stamp)
        except ValueError:
            return ""
        if applied_at.tzinfo is None:
            applied_at = applied_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - applied_at).total_seconds()
        if elapsed >= self._revision_min_seconds:
            return ""
        return (
            f"rate_limited: {field} was revised "
            f"{int(elapsed)}s ago; minimum interval is "
            f"{int(self._revision_min_seconds)}s"
        )

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

    def enter_exploration(self, *, topic: str, rationale: str) -> dict[str, Any]:
        """Nova-originated deliberate entry into the exploratory register.

        Phase 21 Stage 21.1: the controller (Governor-side) validates, clamps
        budgets, enforces one-open-exploration-per-session, and owns register
        state. This tool call is a request; the ExplorationRecord is the fact.
        """
        if self._exploration_controller is None:
            raise ValueError("Exploration is not available in this runtime.")
        record = self._exploration_controller.open(
            session_id=self._session_id,
            topic=topic,
            rationale=rationale,
            origin="nova_tick",
        )
        self._exploration_controller.journal_entry(
            exploration_id=record.exploration_id,
            session_id=self._session_id,
            kind="tick_output",
            content=f"Exploration opened. Topic: {record.topic}. Rationale: {record.rationale}",
        )
        return record.to_dict()

    def close_exploration(self, *, findings_summary: str) -> dict[str, Any]:
        """Nova-originated deliberate close with a findings summary.

        The findings summary is journaled as kind="findings". Governed
        export of findings through the assertion-register gates
        (runtime.export_findings, Phase 21 Stage 21.4) runs automatically
        right after this dispatch succeeds, from model_self_state_tick —
        not from this method, since the claim-gate/validator machinery it
        needs lives on the runtime, not the dispatcher.
        """
        if self._exploration_controller is None:
            raise ValueError("Exploration is not available in this runtime.")
        record = self._exploration_controller.active_exploration(self._session_id)
        if record is None:
            raise ValueError("No active exploration to close for this session.")
        findings_summary = (findings_summary or "").strip()
        if not findings_summary:
            raise ValueError("close_exploration requires a findings_summary.")
        entry = self._exploration_controller.journal_entry(
            exploration_id=record.exploration_id,
            session_id=self._session_id,
            kind="findings",
            content=findings_summary,
        )
        closed = self._exploration_controller.close(
            session_id=self._session_id,
            close_reason="nova_close",
            findings_ref=entry.entry_id,
        )
        assert closed is not None
        return closed.to_dict()


# Stage 22.10 — the carryover loop. The tick surface is one-shot: before
# this stage, a read tool's result was journaled for the operator and never
# reached Nova. These renderers produce the compact block the runtime holds
# for her next ticks. Hard caps: a read can inform a tick, not become it.
READ_TOOL_NAMES: frozenset[str] = frozenset(
    {"recall_self", "reflect", "recall_history"}
)
# Sized so a full 8-entry recall_history result fits (8 × ~165-char lines
# plus header); truncation, when it fires, cuts at a line boundary so she
# never sees half an entry.
RENDER_READ_RESULT_MAX_CHARS = 1500


def render_read_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Render a read tool's result for injection into later tick prompts."""
    lines: list[str] = []
    if tool_name == "recall_history":
        header = (
            f"recall_history {result.get('source', '?')}"
            f" (mode {result.get('mode', '?')},"
            f" {len(result.get('entries', []) or [])} of"
            f" {result.get('total', 0)} total):"
        )
        if result.get("error"):
            header = f"recall_history failed: {result.get('note', result['error'])}"
        lines.append(header)
        if result.get("note") and not result.get("error"):
            lines.append(f"  note: {result['note']}")
        for entry in result.get("entries", []) or []:
            lines.append(f"  [{entry.get('timestamp', '?')}] {entry.get('text', '')}")
    elif tool_name == "recall_self":
        state = result.get("self_state") or {}
        lines.append("recall_self:")
        lines.append(f"  focus: {str(state.get('current_focus', ''))[:140]}")
        for q in (state.get("active_questions") or [])[:3]:
            lines.append(f"  question: {str(q)[:140]}")
        lines.append(
            f"  open_tensions: {len(state.get('open_tensions') or [])},"
            f" continuity_notes: {len(state.get('continuity_notes') or [])}"
        )
        for hb in (result.get("recent_heartbeats") or [])[-3:]:
            obs = str(hb.get("observation", ""))[:110]
            lines.append(f"  heartbeat [{str(hb.get('timestamp', ''))[:19]}] {obs}")
    elif tool_name == "reflect":
        lines.append("reflect:")
        lines.append(f"  focus: {str(result.get('current_focus', ''))[:140]}")
        for q in (result.get("active_questions") or [])[:3]:
            lines.append(f"  question: {str(q)[:140]}")
        for t in (result.get("open_tensions") or [])[:2]:
            lines.append(f"  tension: {str(t)[:120]}")
        for n in (result.get("continuity_notes") or [])[:2]:
            lines.append(f"  continuity: {str(n)[:120]}")
        lines.append(
            f"  claim_posture: {str(result.get('claim_posture', ''))[:80]}"
        )
    else:
        return ""
    rendered = "\n".join(lines)
    if len(rendered) <= RENDER_READ_RESULT_MAX_CHARS:
        return rendered
    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > RENDER_READ_RESULT_MAX_CHARS:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)
