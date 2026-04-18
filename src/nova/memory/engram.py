"""Engram memory store."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nova.types import MemoryEvent, RetrievalHit


class JsonEngramMemoryStore:
    """Hashed n-gram associative memory for recurrent language patterns."""

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        ngram_min: int = 2,
        ngram_max: int = 4,
        max_postings: int = 256,
        auto_prune: bool = True,
        retention_days: int = 30,
        keep_min_uses: int = 3,
    ):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.ngram_min = max(1, int(ngram_min))
        self.ngram_max = max(self.ngram_min, int(ngram_max))
        self.max_postings = max(16, int(max_postings))
        self.auto_prune = bool(auto_prune)
        self.retention_days = max(1, int(retention_days))
        self.keep_min_uses = max(1, int(keep_min_uses))
        self.entries: list[dict[str, Any]] = []
        self.postings: dict[str, list[int]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            self._load()
            self.auto_prune_stale()

    def add(self, event: MemoryEvent) -> None:
        if not self.enabled:
            return
        clean = (event.text or "").strip()
        if not clean:
            return
        entry_id = len(self.entries)
        grams = sorted(self._entry_ngrams(clean))
        created_at = self._timestamp_or_now(event.timestamp)
        entry = {
            "id": entry_id,
            "event_id": event.event_id,
            "text": clean,
            "kind": event.kind,
            "source": event.source,
            "tags": list(event.tags),
            "meta": dict(event.metadata or {}),
            "ngrams": grams,
            "created_at": created_at,
            "last_accessed_at": created_at,
            "hit_count": 0,
            "importance": float(event.importance or 0.0),
            "confidence": float(event.confidence or 1.0),
        }
        self.entries.append(entry)
        for gram_hash in grams:
            bucket = self.postings.get(gram_hash)
            if bucket is None:
                self.postings[gram_hash] = [entry_id]
            elif not bucket or bucket[-1] != entry_id:
                bucket.append(entry_id)
                if len(bucket) > self.max_postings:
                    del bucket[:-self.max_postings]
        self._save()

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if not self.enabled or not self.entries:
            return []
        query_grams = self._entry_ngrams(query)
        if not query_grams:
            return []

        hit_counts: dict[int, float] = defaultdict(float)
        for gram_hash in query_grams:
            for entry_id in self.postings.get(gram_hash, [])[-self.max_postings :]:
                hit_counts[int(entry_id)] += 1.0

        ranked: list[RetrievalHit] = []
        query_norm = max(1.0, float(len(query_grams)))
        now = time.time()
        changed = False

        for entry_id, hits in hit_counts.items():
            if entry_id < 0 or entry_id >= len(self.entries):
                continue
            entry = self.entries[entry_id]
            denom = (query_norm * max(1.0, float(len(entry.get("ngrams", []))))) ** 0.5
            score = float(hits / denom) + (0.1 * float(entry.get("importance", 0.0) or 0.0))
            if score <= 0.0:
                continue
            ranked.append(
                RetrievalHit(
                    channel="engram",
                    text=str(entry.get("text", "") or ""),
                    score=score,
                    kind=entry.get("kind"),
                    source_ref=entry.get("event_id"),
                    tags=list(entry.get("tags", []) or []),
                    metadata={
                        **dict(entry.get("meta", {}) or {}),
                        "source": entry.get("source"),
                        "confidence": entry.get("confidence", 1.0),
                        "memory_channel": "engram",
                    },
                )
            )
            entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
            entry["last_accessed_at"] = now
            changed = True

        ranked.sort(key=lambda item: item.score, reverse=True)
        if changed:
            self._save()
        return ranked[: max(1, top_k)]

    def stats(self) -> dict[str, Any]:
        return {
            "channel": "engram",
            "enabled": self.enabled,
            "entries": len(self.entries),
            "ngrams": len(self.postings),
            "retention_days": self.retention_days,
            "keep_min_uses": self.keep_min_uses,
            "path": str(self.path),
        }

    def list_events(self) -> list[MemoryEvent]:
        if not self.enabled:
            return []
        events: list[MemoryEvent] = []
        for entry in self.entries:
            created_at = float(entry.get("created_at", 0.0) or 0.0)
            timestamp = datetime.fromtimestamp(created_at, tz=datetime.now().astimezone().tzinfo).astimezone().isoformat() if created_at else ""
            metadata = dict(entry.get("meta", {}) or {})
            metadata["hit_count"] = int(entry.get("hit_count", 0) or 0)
            events.append(
                MemoryEvent(
                    event_id=str(entry.get("event_id", "") or ""),
                    timestamp=timestamp,
                    session_id=str(metadata.get("session_id", "") or ""),
                    turn_id=str(metadata.get("turn_id", "") or ""),
                    channel="engram",
                    kind=str(entry.get("kind", "") or ""),
                    text=str(entry.get("text", "") or ""),
                    summary=None,
                    tags=list(entry.get("tags", []) or []),
                    importance=float(entry.get("importance", 0.0) or 0.0),
                    confidence=float(entry.get("confidence", 1.0) or 1.0),
                    continuity_weight=float(metadata.get("continuity_weight", 0.0) or 0.0),
                    retention=str(metadata.get("retention", "active") or "active"),
                    supersedes=list(metadata.get("supersedes", []) or []),
                    source=str(entry.get("source", "") or ""),
                    metadata=metadata,
                )
            )
        return events

    def set_auto_prune(self, enabled: bool) -> dict[str, Any]:
        self.auto_prune = bool(enabled)
        if self.auto_prune:
            self.auto_prune_stale()
        else:
            self._save()
        return self.stats()

    def purge_recent(self, seconds: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"purged": 0, **self.stats()}
        if seconds is None or int(seconds) <= 0:
            purged = len(self.entries)
            self.entries = []
            self.postings = {}
            self._save()
            return {"purged": purged, **self.stats()}
        cutoff = time.time() - int(seconds)
        purged = self._prune_entries(
            lambda entry: float(entry.get("created_at", 0.0) or 0.0) < cutoff
        )
        return {"purged": purged, **self.stats()}

    def auto_prune_stale(self) -> dict[str, Any]:
        if not self.enabled or not self.auto_prune:
            return {"purged": 0, **self.stats()}
        cutoff = time.time() - (self.retention_days * 86400)
        purged = self._prune_entries(
            lambda entry: (
                float(entry.get("created_at", 0.0) or 0.0) >= cutoff
                or int(entry.get("hit_count", 0) or 0) >= self.keep_min_uses
            )
        )
        return {"purged": purged, **self.stats()}

    def _prune_entries(self, keep_fn) -> int:
        kept = [entry for entry in self.entries if keep_fn(entry)]
        purged = len(self.entries) - len(kept)
        if purged <= 0:
            return 0
        self.entries = kept
        self._rebuild_postings()
        self._save()
        return purged

    def _load(self) -> None:
        if not self.path.exists():
            self.entries = []
            self.postings = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle) or {}
        except Exception:
            self.entries = []
            self.postings = {}
            return
        entries = payload.get("entries", [])
        postings = payload.get("postings", {})
        self.entries = [row for row in entries if isinstance(row, dict) and row.get("text")]
        now = time.time()
        for row in self.entries:
            if not isinstance(row.get("created_at"), (int, float)):
                row["created_at"] = now
            if not isinstance(row.get("last_accessed_at"), (int, float)):
                row["last_accessed_at"] = row["created_at"]
            if not isinstance(row.get("hit_count"), int):
                row["hit_count"] = int(row.get("hit_count", 0) or 0)
            if not isinstance(row.get("ngrams"), list):
                row["ngrams"] = sorted(self._entry_ngrams(str(row.get("text", "") or "")))
        self.postings = {
            str(key): [int(v) for v in values if isinstance(v, int) or str(v).isdigit()]
            for key, values in postings.items()
            if isinstance(values, list)
        }
        if self.entries and not self.postings:
            self._rebuild_postings()

    def _save(self) -> None:
        payload = {"entries": self.entries, "postings": self.postings}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def _rebuild_postings(self) -> None:
        self.postings = {}
        for entry_id, entry in enumerate(self.entries):
            entry["id"] = entry_id
            grams = entry.get("ngrams", [])
            if not isinstance(grams, list):
                grams = sorted(self._entry_ngrams(str(entry.get("text", "") or "")))
                entry["ngrams"] = grams
            for gram_hash in grams:
                bucket = self.postings.get(str(gram_hash))
                if bucket is None:
                    self.postings[str(gram_hash)] = [entry_id]
                elif not bucket or bucket[-1] != entry_id:
                    bucket.append(entry_id)
                    if len(bucket) > self.max_postings:
                        del bucket[:-self.max_postings]

    def _normalize_tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _hash_ngram(self, gram: tuple[str, ...]) -> str:
        raw = "\x1f".join(gram).encode("utf-8", errors="ignore")
        return hashlib.blake2b(raw, digest_size=8).hexdigest()

    def _entry_ngrams(self, text: str) -> set[str]:
        tokens = self._normalize_tokens(text)
        grams: set[str] = set()
        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(tokens) < n:
                continue
            for idx in range(0, len(tokens) - n + 1):
                grams.add(self._hash_ngram(tuple(tokens[idx : idx + n])))
        return grams

    def _timestamp_or_now(self, timestamp: str) -> float:
        if not timestamp:
            return time.time()
        try:
            return float(timestamp)
        except ValueError:
            pass
        try:
            iso_text = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(iso_text).timestamp()
        except Exception:
            return time.time()
