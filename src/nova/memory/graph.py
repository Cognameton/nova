"""Graph memory store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from nova.types import GraphFact, MemoryEvent, RetrievalHit


class SqliteGraphMemoryStore:
    """Simple SQLite graph store for explicit facts and relations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add(self, event: MemoryEvent) -> None:
        metadata = event.metadata or {}
        fact = GraphFact(
            fact_id=metadata.get("fact_id", event.event_id),
            timestamp=event.timestamp,
            subject_type=metadata.get("subject_type", "entity"),
            subject_key=metadata.get("subject_key", ""),
            relation=metadata.get("relation", ""),
            object_type=metadata.get("object_type", "entity"),
            object_key=metadata.get("object_key", ""),
            subject_name=metadata.get("subject_name"),
            object_name=metadata.get("object_name"),
            weight=float(metadata.get("weight", event.importance or 1.0) or 1.0),
            confidence=float(metadata.get("confidence", event.confidence) or 1.0),
            continuity_weight=float(
                metadata.get("continuity_weight", event.continuity_weight) or 0.0
            ),
            active=bool(metadata.get("active", True)),
            superseded_by=metadata.get("superseded_by"),
            evidence_text=metadata.get("evidence_text", event.text),
            source=event.source,
            metadata=metadata,
        )
        if not fact.subject_key or not fact.relation or not fact.object_key:
            return

        with self._connect() as conn:
            self._supersede_conflicting_facts(conn, fact)
            fact = self._merge_existing_fact(conn, fact)
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_facts (
                    fact_id, timestamp, subject_type, subject_key, relation,
                    object_type, object_key, subject_name, object_name,
                    weight, confidence, continuity_weight, active, superseded_by,
                    evidence_text, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.timestamp,
                    fact.subject_type,
                    fact.subject_key,
                    fact.relation,
                    fact.object_type,
                    fact.object_key,
                    fact.subject_name,
                    fact.object_name,
                    fact.weight,
                    fact.confidence,
                    fact.continuity_weight,
                    1 if fact.active else 0,
                    fact.superseded_by,
                    fact.evidence_text,
                    fact.source,
                    json.dumps(fact.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        tokens = self._tokens(query)
        if not tokens:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT fact_id, subject_type, subject_key, relation, object_type, object_key,
                       subject_name, object_name, weight, confidence, continuity_weight, active,
                       superseded_by, evidence_text, source, metadata_json
                FROM graph_facts
                ORDER BY active DESC, timestamp DESC
                """
            ).fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            subject_name = row["subject_name"] or row["subject_key"]
            object_name = row["object_name"] or row["object_key"]
            text = f"{subject_name} [{row['subject_type']}] -> {row['relation']} -> {object_name} [{row['object_type']}]"
            evidence = str(row["evidence_text"] or "").strip()
            haystack = f"{text} {evidence}".lower()
            lexical_score = float(sum(1 for token in tokens if token in haystack))
            structural_score = (
                float(row["weight"] or 0.0)
                + float(row["confidence"] or 0.0)
                + float(row["continuity_weight"] or 0.0)
            )
            active_boost = 0.5 if int(row["active"] or 0) else 0.0
            score = lexical_score + structural_score + active_boost
            if score <= 0:
                continue
            status = "active" if int(row["active"] or 0) else "historical"
            metadata = json.loads(row["metadata_json"] or "{}")
            retention = str(metadata.get("retention", "active") or "active")
            if retention == "pruned":
                continue
            if retention == "demoted":
                score *= 0.7
            elif retention == "archived":
                score *= 0.45
            domain = str(metadata.get("fact_domain", "") or "")
            if evidence:
                text = f"{text} | Status: {status} | Evidence: {evidence}"
            else:
                text = f"{text} | Status: {status}"
            hits.append(
                RetrievalHit(
                    channel="graph",
                    text=text,
                    score=score,
                    kind=domain or "fact",
                    source_ref=row["fact_id"],
                    tags=[tag for tag in (domain, status, retention) if tag],
                    metadata=metadata,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, top_k)]

    def stats(self) -> dict[str, int | str]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS count,
                    SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS active_count
                FROM graph_facts
                """
            ).fetchone()
        return {
            "channel": "graph",
            "entries": int(row["count"] if row is not None else 0),
            "active_entries": int(row["active_count"] if row is not None and row["active_count"] is not None else 0),
            "path": str(self.path),
        }

    def list_events(self) -> list[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT fact_id, timestamp, subject_type, subject_key, relation, object_type, object_key,
                       subject_name, object_name, weight, confidence, continuity_weight, active,
                       superseded_by, evidence_text, source, metadata_json
                FROM graph_facts
                ORDER BY timestamp DESC
                """
            ).fetchall()

        events: list[MemoryEvent] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            events.append(
                MemoryEvent(
                    event_id=str(row["fact_id"] or ""),
                    timestamp=str(row["timestamp"] or ""),
                    session_id=str(metadata.get("session_id", "") or ""),
                    turn_id=str(metadata.get("turn_id", "") or ""),
                    channel="graph",
                    kind=str(metadata.get("fact_domain") or "fact"),
                    text=str(row["evidence_text"] or ""),
                    summary=None,
                    tags=[tag for tag in (metadata.get("fact_domain"), "active" if int(row["active"] or 0) else "historical") if tag],
                    importance=float(row["weight"] or 0.0),
                    confidence=float(row["confidence"] or 1.0),
                    continuity_weight=float(row["continuity_weight"] or 0.0),
                    retention=str(metadata.get("retention", "active" if int(row["active"] or 0) else "archived")),
                    supersedes=[row["superseded_by"]] if row["superseded_by"] else [],
                    source=str(row["source"] or ""),
                    metadata=metadata,
                )
            )
        return events

    def apply_maintenance_decisions(self, decisions: list[object]) -> int:
        decision_map = {
            getattr(decision, "event_id", ""): decision
            for decision in decisions
            if getattr(decision, "event_id", "")
        }
        if not decision_map:
            return 0

        updated = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact_id, metadata_json FROM graph_facts"
            ).fetchall()
            for row in rows:
                fact_id = str(row["fact_id"] or "")
                decision = decision_map.get(fact_id)
                if decision is None:
                    continue
                metadata = json.loads(row["metadata_json"] or "{}")
                target_retention = str(getattr(decision, "target_retention", "active") or "active")
                metadata["retention"] = target_retention
                metadata["maintenance"] = {
                    "action": getattr(decision, "action", ""),
                    "reason": getattr(decision, "reason", ""),
                    "applied_at": "maintenance",
                }
                active = 1 if target_retention == "active" else 0
                conn.execute(
                    """
                    UPDATE graph_facts
                    SET active = ?, metadata_json = ?
                    WHERE fact_id = ?
                    """,
                    (
                        active,
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        fact_id,
                    ),
                )
                updated += 1
        return updated

    def _merge_existing_fact(self, conn: sqlite3.Connection, fact: GraphFact) -> GraphFact:
        existing = conn.execute(
            """
            SELECT metadata_json, weight, confidence, continuity_weight, evidence_text
            FROM graph_facts
            WHERE fact_id = ?
            """,
            (fact.fact_id,),
        ).fetchone()
        if existing is None:
            return fact

        existing_metadata = json.loads(existing["metadata_json"] or "{}")
        revision_count = int(existing_metadata.get("revision_count", 0) or 0) + 1
        merged_evidence = self._merge_evidence(
            str(existing["evidence_text"] or ""),
            str(fact.evidence_text or ""),
        )
        fact.weight = max(float(existing["weight"] or 0.0), fact.weight)
        fact.confidence = max(float(existing["confidence"] or 0.0), fact.confidence)
        fact.continuity_weight = max(
            float(existing["continuity_weight"] or 0.0),
            fact.continuity_weight,
        )
        fact.evidence_text = merged_evidence
        fact.metadata = {
            **existing_metadata,
            **fact.metadata,
            "revision_count": revision_count,
            "revised_at": fact.timestamp,
        }
        return fact

    def _supersede_conflicting_facts(self, conn: sqlite3.Connection, fact: GraphFact) -> None:
        domain = str(fact.metadata.get("fact_domain", "") or "")
        if not domain:
            return
        rows = conn.execute(
            """
            SELECT fact_id, object_key, metadata_json
            FROM graph_facts
            WHERE subject_key = ? AND relation = ? AND active = 1
            """,
            (fact.subject_key, fact.relation),
        ).fetchall()
        for row in rows:
            prior_fact_id = str(row["fact_id"] or "")
            if prior_fact_id == fact.fact_id:
                continue
            prior_metadata = json.loads(row["metadata_json"] or "{}")
            prior_domain = str(prior_metadata.get("fact_domain", "") or "")
            if prior_domain != domain:
                continue
            prior_object_key = str(row["object_key"] or "")
            if prior_object_key == fact.object_key:
                continue
            prior_metadata["retention"] = "archived"
            prior_metadata["superseded_by"] = fact.fact_id
            prior_metadata["superseded_at"] = fact.timestamp
            conn.execute(
                """
                UPDATE graph_facts
                SET active = 0, superseded_by = ?, metadata_json = ?
                WHERE fact_id = ?
                """,
                (
                    fact.fact_id,
                    json.dumps(prior_metadata, ensure_ascii=False, sort_keys=True),
                    prior_fact_id,
                ),
            )

    def _merge_evidence(self, existing: str, incoming: str) -> str:
        existing = existing.strip()
        incoming = incoming.strip()
        if not existing:
            return incoming
        if not incoming or incoming in existing:
            return existing
        return f"{existing} || {incoming}"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_facts (
                    fact_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    subject_name TEXT,
                    object_name TEXT,
                    weight REAL NOT NULL,
                    confidence REAL NOT NULL,
                    continuity_weight REAL NOT NULL DEFAULT 0.0,
                    active INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT,
                    evidence_text TEXT,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(graph_facts)").fetchall()
            }
            if "continuity_weight" not in columns:
                conn.execute(
                    "ALTER TABLE graph_facts ADD COLUMN continuity_weight REAL NOT NULL DEFAULT 0.0"
                )
            if "active" not in columns:
                conn.execute(
                    "ALTER TABLE graph_facts ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
            if "superseded_by" not in columns:
                conn.execute(
                    "ALTER TABLE graph_facts ADD COLUMN superseded_by TEXT"
                )

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in text.replace("\n", " ").split()
            if token.strip()
        }
