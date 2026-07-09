"""Exploratory register lifecycle — Phase 21 Stage 21.1.

Implements the Governor-owned side of the Exploratory Register Contract
(docs/plans/EXPLORATORY_REGISTER_CONTRACT.txt): the ExplorationRecord store,
the exploration journal, and the ExplorationController that determines
register state, enforces budgets, and manages the lifecycle.

Contract Invariant 1: register state is runtime-owned. Only an active
ExplorationRecord — created, tracked, and closed by this controller — places
a surface in the exploratory register. Model output declaring register state
has no effect on anything in this module.

Contract Invariant 5: nothing is deleted. The journal and the exploration
store are append/rewrite-in-place JSONL; no code path removes records.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.types import ExplorationJournalEntry, ExplorationRecord, SCHEMA_VERSION


# Registers. Register state is a property of a surface, owned by the Governor.
REGISTER_ASSERTION = "assertion"
REGISTER_EXPLORATORY = "exploratory"

# Default budgets applied when an entry path omits them.
DEFAULT_MAX_TICKS = 12
DEFAULT_MAX_TOKENS = 24_000
DEFAULT_WALL_CLOCK_SECONDS = 3_600

# Hard caps. Budgets are clamped to these regardless of what any entry path
# (including a Nova-originated enter_exploration call) requests.
CAP_MAX_TICKS = 48
CAP_MAX_TOKENS = 96_000
CAP_WALL_CLOCK_SECONDS = 21_600

VALID_ORIGINS = frozenset({"nova_tick", "runtime_offer", "operator"})
VALID_STATUSES = frozenset({"active", "paused", "closed", "interrupted"})
VALID_CLOSE_REASONS = frozenset(
    {"nova_close", "budget_exhausted", "operator_close", "interrupted"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExplorationStore:
    """JSONL-backed store of ExplorationRecords."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "explorations.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExplorationRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _read_all(self) -> list[ExplorationRecord]:
        if not self._path.exists():
            return []
        records: list[ExplorationRecord] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            records.append(_exploration_from_dict(data))
        return records

    def get(self, exploration_id: str) -> ExplorationRecord | None:
        for record in self._read_all():
            if record.exploration_id == exploration_id:
                return record
        return None

    def list_recent(self, *, limit: int = 10) -> list[ExplorationRecord]:
        return self._read_all()[-limit:]

    def list_all(self) -> list[ExplorationRecord]:
        """All exploration records, for reporting (e.g. tick-analysis)."""
        return self._read_all()

    def open_for_session(self, session_id: str) -> ExplorationRecord | None:
        """Return the session's active or paused exploration, if any."""
        for record in reversed(self._read_all()):
            if record.session_id == session_id and record.status in (
                "active",
                "paused",
            ):
                return record
        return None

    def update(self, record: ExplorationRecord) -> None:
        """Rewrite the store with the updated record in place."""
        records = self._read_all()
        with self._path.open("w", encoding="utf-8") as fh:
            for existing in records:
                out = record if existing.exploration_id == record.exploration_id else existing
                fh.write(json.dumps(out.to_dict(), ensure_ascii=False) + "\n")


class ExplorationJournal:
    """JSONL-backed append-only exploration journal.

    Membrane rule (contract Invariant 4): journal content is readable only
    from inside the exploratory register. The runtime includes journal recall
    in in-register prompt composition and never in assertion-register
    composition. Nothing here is ever deleted (Invariant 5).
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "journal.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: ExplorationJournalEntry) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def _read_all(self) -> list[ExplorationJournalEntry]:
        if not self._path.exists():
            return []
        entries: list[ExplorationJournalEntry] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            entries.append(_journal_entry_from_dict(data))
        return entries

    def list_for(self, exploration_id: str) -> list[ExplorationJournalEntry]:
        return [e for e in self._read_all() if e.exploration_id == exploration_id]

    def list_recent(self, *, limit: int = 10) -> list[ExplorationJournalEntry]:
        """Recent entries across all explorations, for in-register recall."""
        return self._read_all()[-limit:]

    def list_all(self) -> list[ExplorationJournalEntry]:
        """All journal entries, for reporting (e.g. tick-analysis)."""
        return self._read_all()

    def recall_block(self, *, current_exploration_id: str, limit: int = 6) -> str:
        """Bounded journal recall for IN-REGISTER prompt composition only.

        Includes recent entries from prior explorations so inquiry has
        continuity across sessions, plus this exploration's own thread.
        """
        entries = self._read_all()
        prior = [e for e in entries if e.exploration_id != current_exploration_id]
        own = [e for e in entries if e.exploration_id == current_exploration_id]
        lines: list[str] = []
        if prior:
            lines.append("Prior exploration recall (exploratory register):")
            for e in prior[-limit:]:
                lines.append(f"  [{e.timestamp[:19]}] ({e.kind}) {e.content[:160]}")
        if own:
            lines.append("This exploration so far:")
            for e in own[-limit:]:
                lines.append(f"  [{e.timestamp[:19]}] ({e.kind}) {e.content[:160]}")
        return "\n".join(lines)


class ExplorationController:
    """Governor-side exploration lifecycle, budgets, and register determination."""

    def __init__(self, *, store: ExplorationStore, journal: ExplorationJournal) -> None:
        self.store = store
        self.journal = journal

    # ── Register determination (contract Invariant 1) ─────────────────────

    def register_for(self, session_id: str) -> str:
        """Return the register for a session's tick surfaces.

        Exploratory only while an ACTIVE exploration covers the session.
        Paused explorations leave the surface in the assertion register.
        """
        record = self.store.open_for_session(session_id)
        if record is not None and record.status == "active":
            return REGISTER_EXPLORATORY
        return REGISTER_ASSERTION

    def active_exploration(self, session_id: str) -> ExplorationRecord | None:
        record = self.store.open_for_session(session_id)
        if record is not None and record.status == "active":
            return record
        return None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def open(
        self,
        *,
        session_id: str,
        topic: str,
        rationale: str,
        origin: str,
        max_ticks: int = DEFAULT_MAX_TICKS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS,
    ) -> ExplorationRecord:
        topic = (topic or "").strip()
        rationale = (rationale or "").strip()
        if not topic:
            raise ValueError("Exploration topic is required.")
        if not rationale:
            raise ValueError("Exploration rationale is required.")
        if origin not in VALID_ORIGINS:
            raise ValueError(f"Invalid exploration origin: {origin!r}")
        if self.store.open_for_session(session_id) is not None:
            raise ValueError(
                "An exploration is already open for this session. "
                "Close it before entering another."
            )
        record = ExplorationRecord(
            exploration_id=uuid4().hex,
            session_id=session_id,
            topic=topic,
            rationale=rationale,
            origin=origin,
            max_ticks=max(1, min(int(max_ticks), CAP_MAX_TICKS)),
            max_tokens=max(1, min(int(max_tokens), CAP_MAX_TOKENS)),
            wall_clock_seconds=max(
                1, min(int(wall_clock_seconds), CAP_WALL_CLOCK_SECONDS)
            ),
            status="active",
            opened_at=_utc_now(),
        )
        self.store.append(record)
        return record

    def pause(self, session_id: str) -> ExplorationRecord | None:
        """Pause the session's active exploration (subordination rule)."""
        record = self.store.open_for_session(session_id)
        if record is None or record.status != "active":
            return record
        record.status = "paused"
        self.store.update(record)
        return record

    def resume(self, session_id: str) -> ExplorationRecord | None:
        """Resume the session's paused exploration when idle conditions hold."""
        record = self.store.open_for_session(session_id)
        if record is None or record.status != "paused":
            return record
        if self._budget_exhaustion_reason(record) is not None:
            return self._close(record, close_reason="budget_exhausted")
        record.status = "active"
        self.store.update(record)
        return record

    def close(
        self,
        *,
        session_id: str,
        close_reason: str,
        findings_ref: str = "",
    ) -> ExplorationRecord | None:
        if close_reason not in VALID_CLOSE_REASONS:
            raise ValueError(f"Invalid close reason: {close_reason!r}")
        record = self.store.open_for_session(session_id)
        if record is None:
            return None
        return self._close(record, close_reason=close_reason, findings_ref=findings_ref)

    def interrupt(self, session_id: str) -> ExplorationRecord | None:
        """Operator interrupt: closes without a findings pass, journal retained."""
        record = self.store.open_for_session(session_id)
        if record is None:
            return None
        record.status = "interrupted"
        record.close_reason = "interrupted"
        record.closed_at = _utc_now()
        self.store.update(record)
        return record

    def _close(
        self,
        record: ExplorationRecord,
        *,
        close_reason: str,
        findings_ref: str = "",
    ) -> ExplorationRecord:
        record.status = "closed"
        record.close_reason = close_reason
        record.closed_at = _utc_now()
        if findings_ref:
            record.findings_ref = findings_ref
        self.store.update(record)
        return record

    # ── Budgets (Governor-enforced) ────────────────────────────────────────

    def record_tick(
        self,
        *,
        session_id: str,
        tick_id: str,
        tokens_used: int = 0,
    ) -> ExplorationRecord | None:
        """Charge one in-register tick against the exploration's budgets.

        Auto-closes with close_reason=budget_exhausted when any budget is
        spent. Returns the updated record (possibly closed), or None when no
        exploration is open for the session.
        """
        record = self.store.open_for_session(session_id)
        if record is None or record.status != "active":
            return record
        record.ticks_used += 1
        record.tokens_used += max(0, int(tokens_used))
        record.tick_ids.append(tick_id)
        reason = self._budget_exhaustion_reason(record)
        if reason is not None:
            record.status = "closed"
            record.close_reason = "budget_exhausted"
            record.closed_at = _utc_now()
            self.store.update(record)
            self.journal_entry(
                exploration_id=record.exploration_id,
                session_id=session_id,
                tick_id=tick_id,
                kind="operator_note",
                content=f"Exploration closed by Governor: {reason}.",
            )
            return record
        self.store.update(record)
        return record

    def budget_exhausted(self, record: ExplorationRecord) -> bool:
        return self._budget_exhaustion_reason(record) is not None

    def _budget_exhaustion_reason(self, record: ExplorationRecord) -> str | None:
        if record.ticks_used >= record.max_ticks:
            return "tick budget exhausted"
        if record.tokens_used >= record.max_tokens:
            return "token budget exhausted"
        if record.opened_at:
            try:
                opened = datetime.fromisoformat(record.opened_at)
            except ValueError:
                return None
            elapsed = (datetime.now(timezone.utc) - opened).total_seconds()
            if elapsed >= record.wall_clock_seconds:
                return "wall clock budget exhausted"
        return None

    # ── Journal ────────────────────────────────────────────────────────────

    def journal_entry(
        self,
        *,
        exploration_id: str,
        session_id: str,
        kind: str,
        content: str,
        tick_id: str = "",
        notes: list[str] | None = None,
    ) -> ExplorationJournalEntry:
        entry = ExplorationJournalEntry(
            entry_id=uuid4().hex,
            exploration_id=exploration_id,
            session_id=session_id,
            tick_id=tick_id,
            timestamp=_utc_now(),
            register=REGISTER_EXPLORATORY,
            kind=kind,
            content=content,
            notes=list(notes or []),
        )
        self.journal.append(entry)
        return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exploration_from_dict(data: dict[str, Any]) -> ExplorationRecord:
    return ExplorationRecord(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        exploration_id=str(data.get("exploration_id", "")),
        session_id=str(data.get("session_id", "")),
        topic=str(data.get("topic", "")),
        rationale=str(data.get("rationale", "")),
        origin=str(data.get("origin", "")),
        max_ticks=int(data.get("max_ticks", 0) or 0),
        max_tokens=int(data.get("max_tokens", 0) or 0),
        wall_clock_seconds=int(data.get("wall_clock_seconds", 0) or 0),
        status=str(data.get("status", "active")),
        opened_at=str(data.get("opened_at", "")),
        closed_at=str(data.get("closed_at", "")),
        close_reason=str(data.get("close_reason", "")),
        ticks_used=int(data.get("ticks_used", 0) or 0),
        tokens_used=int(data.get("tokens_used", 0) or 0),
        tick_ids=[str(t) for t in (data.get("tick_ids") or [])],
        findings_ref=str(data.get("findings_ref", "")),
    )


def _journal_entry_from_dict(data: dict[str, Any]) -> ExplorationJournalEntry:
    return ExplorationJournalEntry(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        entry_id=str(data.get("entry_id", "")),
        exploration_id=str(data.get("exploration_id", "")),
        session_id=str(data.get("session_id", "")),
        tick_id=str(data.get("tick_id", "")),
        timestamp=str(data.get("timestamp", "")),
        register=str(data.get("register", REGISTER_EXPLORATORY)),
        kind=str(data.get("kind", "")),
        content=str(data.get("content", "")),
        notes=[str(n) for n in (data.get("notes") or [])],
    )
