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
            evidence_text=metadata.get("evidence_text", event.text),
            source=event.source,
            metadata=metadata,
        )
        if not fact.subject_key or not fact.relation or not fact.object_key:
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_facts (
                    fact_id, timestamp, subject_type, subject_key, relation,
                    object_type, object_key, subject_name, object_name,
                    weight, confidence, evidence_text, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                       subject_name, object_name, weight, confidence, evidence_text, source, metadata_json
                FROM graph_facts
                ORDER BY timestamp DESC
                """
            ).fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            subject_name = row["subject_name"] or row["subject_key"]
            object_name = row["object_name"] or row["object_key"]
            text = f"{subject_name} [{row['subject_type']}] -> {row['relation']} -> {object_name} [{row['object_type']}]"
            evidence = str(row["evidence_text"] or "").strip()
            haystack = f"{text} {evidence}".lower()
            score = float(sum(1 for token in tokens if token in haystack))
            if score <= 0:
                continue
            if evidence:
                text = f"{text} | Evidence: {evidence}"
            metadata = json.loads(row["metadata_json"] or "{}")
            hits.append(
                RetrievalHit(
                    channel="graph",
                    text=text,
                    score=score,
                    kind="fact",
                    source_ref=row["fact_id"],
                    tags=[],
                    metadata=metadata,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, top_k)]

    def stats(self) -> dict[str, int | str]:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM graph_facts").fetchone()
        return {
            "channel": "graph",
            "entries": int(row["count"] if row is not None else 0),
            "path": str(self.path),
        }

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
                    evidence_text TEXT,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in text.replace("\n", " ").split()
            if token.strip()
        }
