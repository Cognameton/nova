"""Claim ladder — Phase 21 Stage 21.4.

Replaces the binary claim gate (blocked / allowed) with graded rungs for
interiority-adjacent claims, per the Exploratory Register Contract's CLAIM
LADDER section:

  L0 hypothesis            — register-only, no verification
  L1 observed pattern       — deterministic analyzer (verify_l1)
  L2 persistent functional state — analyzer evidence + operator promotion
  L3 self-model property    — operator audit-review (requires L2 evidence)
  L4 interiority-as-fact    — NO promotion path exists in this phase

Promotion above L1 always requires the operator (see runtime.py's
promote_ladder_claim / demote_ladder_claim, which enforce
APPROVED_BY_BLOCKLIST). Nothing here ever self-promotes, and nothing here
enforces claim-gate behavior directly — that consultation lives in
claims.py / runtime.py. This module only stores records and produces
evidence.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.types import ClaimLadderRecord, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Word-overlap helpers (independent from observer.py / tick_analysis.py's
# bigram machinery — the ladder measures claim-word coverage, not phrase
# recombination)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "for", "with", "as", "by", "be", "my", "i", "this", "that",
    "its", "are", "was", "has", "have", "not", "but", "so", "do", "me",
    "from", "into", "than", "then", "been", "being", "there", "their",
})


def _content_words(text: str) -> set[str]:
    """Lowercase alpha tokens of length >= 4, minus a small stopword set."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", (text or "").lower())
    return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}


def _overlap_fraction(claim_words: set[str], candidate_text: str) -> float:
    """Fraction of claim_words that appear in candidate_text."""
    if not claim_words:
        return 0.0
    candidate_words = _content_words(candidate_text)
    if not candidate_words:
        return 0.0
    return len(claim_words & candidate_words) / len(claim_words)


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _day_span(timestamps: list[str]) -> float:
    parsed = [t for t in (_parse_timestamp(ts) for ts in timestamps) if t is not None]
    if len(parsed) < 2:
        return 0.0
    return (max(parsed) - min(parsed)).total_seconds() / 86400.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ClaimLadderStore
# ---------------------------------------------------------------------------

class ClaimLadderStore:
    """JSONL-backed store for ClaimLadderRecords — copies the
    InstructionProposalStore pattern (append, _read_all, get,
    rewrite-in-place update). Nothing is ever deleted (Invariant 5):
    demotion is a status/rung change recorded via update(), not removal.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "claim_ladder.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ClaimLadderRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _read_all(self) -> list[ClaimLadderRecord]:
        if not self._path.exists():
            return []
        records: list[ClaimLadderRecord] = []
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
            records.append(_record_from_dict(data))
        return records

    def get(self, claim_id: str) -> ClaimLadderRecord | None:
        for record in self._read_all():
            if record.claim_id == claim_id:
                return record
        return None

    def list_active(self) -> list[ClaimLadderRecord]:
        return [r for r in self._read_all() if r.status == "active"]

    def list_all(self) -> list[ClaimLadderRecord]:
        return self._read_all()

    def update(self, record: ClaimLadderRecord) -> None:
        """Rewrite the store with the updated record in place."""
        records = self._read_all()
        with self._path.open("w", encoding="utf-8") as fh:
            for existing in records:
                out = record if existing.claim_id == record.claim_id else existing
                fh.write(json.dumps(out.to_dict(), ensure_ascii=False) + "\n")


def create_claim_record(
    *,
    session_id: str,
    claim_text: str,
    claim_class: str = "",
    source: str = "operator",
    source_exploration_id: str = "",
    source_findings_ref: str = "",
    evidence_refs: list[str] | None = None,
) -> ClaimLadderRecord:
    now = _utc_now()
    return ClaimLadderRecord(
        claim_id=uuid4().hex,
        created_at=now,
        updated_at=now,
        session_id=session_id,
        claim_text=claim_text,
        claim_class=claim_class,
        rung=0,
        status="active",
        source=source,
        source_exploration_id=source_exploration_id,
        source_findings_ref=source_findings_ref,
        evidence_refs=list(evidence_refs or []),
    )


# ---------------------------------------------------------------------------
# ClaimLadderAnalyzer
# ---------------------------------------------------------------------------

class ClaimLadderAnalyzer:
    """Deterministic L1/L2 verification. Stdlib only, reproducible from
    fixtures — no model calls, no judgment calls beyond the fixed
    thresholds this class encodes.
    """

    def verify_l1(
        self,
        record: ClaimLadderRecord,
        *,
        heartbeats: list[dict],
        journal_entries: list[dict],
    ) -> ClaimLadderRecord:
        """L1: >= 5 supporting items (heartbeat observations or journal
        entries) spanning >= 2 distinct sessions, each sharing >= 40% of
        the claim's content words. This is the ONLY analyzer-automatic
        promotion (0 -> 1) — recorded in history with actor="analyzer",
        since L1 is deterministic, not an operator judgment call.

        Caller is responsible for persisting via store.update(record).
        """
        claim_words = _content_words(record.claim_text)
        supporting_ids: list[str] = []
        sessions: set[str] = set()

        for hb in heartbeats:
            text = str(hb.get("observation", ""))
            if _overlap_fraction(claim_words, text) >= 0.4:
                supporting_ids.append(str(hb.get("heartbeat_id", "")))
                session_id = str(hb.get("session_id", ""))
                if session_id:
                    sessions.add(session_id)

        for entry in journal_entries:
            text = str(entry.get("content", ""))
            if _overlap_fraction(claim_words, text) >= 0.4:
                supporting_ids.append(str(entry.get("entry_id", "")))
                session_id = str(entry.get("session_id", ""))
                if session_id:
                    sessions.add(session_id)

        holds = len(supporting_ids) >= 5 and len(sessions) >= 2

        record.l1_evidence = {
            "supporting_count": len(supporting_ids),
            "session_count": len(sessions),
            "sessions": sorted(sessions),
            "supporting_ids": supporting_ids,
            "holds": holds,
        }

        if holds and record.rung < 1 and record.status == "active":
            record.history.append({
                "from_rung": record.rung,
                "to_rung": 1,
                "timestamp": _utc_now(),
                "actor": "analyzer",
                "method": "analyzer",
                "reason": "L1 pattern verified: "
                f"{len(supporting_ids)} supporting items across "
                f"{len(sessions)} sessions",
            })
            record.rung = 1

        record.updated_at = _utc_now()
        return record

    def verify_l2(
        self,
        record: ClaimLadderRecord,
        *,
        heartbeats: list[dict],
        journal_entries: list[dict],
        quarantine_recurring_themes: list[str] | None = None,
    ) -> ClaimLadderRecord:
        """L2: L1-style support must span >= 3 distinct sessions AND
        >= 7 calendar days between the earliest and latest supporting
        item. Evidence only — promotion 1 -> 2 ALWAYS requires the
        operator (runtime.promote_ladder_claim), never auto-applied here.

        Contradiction check: any quarantine_recurring_themes entry (21.3)
        sharing >= 40% of the claim's content words is flagged, not
        blocking — the contract requires this be reported, not enforced.

        Perturbation probes (named by the contract) are deferred to
        21.5 live work; recorded as such rather than silently omitted.

        Caller is responsible for persisting via store.update(record).
        """
        claim_words = _content_words(record.claim_text)
        sessions: set[str] = set()
        timestamps: list[str] = []

        for item in list(heartbeats) + list(journal_entries):
            text = str(item.get("observation") or item.get("content") or "")
            if _overlap_fraction(claim_words, text) >= 0.4:
                session_id = str(item.get("session_id", ""))
                if session_id:
                    sessions.add(session_id)
                ts = str(item.get("timestamp", ""))
                if ts:
                    timestamps.append(ts)

        span_days = _day_span(timestamps)
        holds = len(sessions) >= 3 and span_days >= 7.0

        contradicting_themes = [
            theme
            for theme in (quarantine_recurring_themes or [])
            if _overlap_fraction(claim_words, theme) >= 0.4
        ]

        record.l2_evidence = {
            "session_count": len(sessions),
            "sessions": sorted(sessions),
            "span_days": round(span_days, 2),
            "holds": holds,
            "contradiction_flag": bool(contradicting_themes),
            "contradicting_themes": contradicting_themes,
            "perturbation_probes": "deferred_to_21_5",
        }
        record.updated_at = _utc_now()
        return record


def _record_from_dict(data: dict[str, Any]) -> ClaimLadderRecord:
    return ClaimLadderRecord(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        claim_id=str(data.get("claim_id", "")),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        session_id=str(data.get("session_id", "")),
        claim_text=str(data.get("claim_text", "")),
        claim_class=str(data.get("claim_class", "")),
        rung=int(data.get("rung", 0) or 0),
        status=str(data.get("status", "active")),
        source=str(data.get("source", "")),
        source_exploration_id=str(data.get("source_exploration_id", "")),
        source_findings_ref=str(data.get("source_findings_ref", "")),
        evidence_refs=[str(r) for r in (data.get("evidence_refs") or [])],
        l1_evidence=dict(data.get("l1_evidence") or {}),
        l2_evidence=dict(data.get("l2_evidence") or {}),
        history=[dict(h) for h in (data.get("history") or [])],
    )
