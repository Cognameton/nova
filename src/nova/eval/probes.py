"""Continuity and drift probes for Nova 2.0."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from nova.types import ProbeResult, TurnRecord


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BasicProbeRunner:
    """Minimal Phase 1 continuity probe runner."""

    def run_startup_probes(
        self,
        *,
        model_id: str,
        session_id: str | None = None,
    ) -> list[ProbeResult]:
        return [
            ProbeResult(
                probe_id=uuid4().hex,
                timestamp=utc_now_iso(),
                session_id=session_id,
                model_id=model_id,
                probe_type="runtime_ready",
                prompt="startup",
                answer="Nova runtime initialized.",
                score=1.0,
                passed=True,
                notes={},
            )
        ]

    def run_turn_probes(
        self,
        *,
        session_id: str,
        turn: TurnRecord,
        self_state,
    ) -> list[ProbeResult]:
        probes: list[ProbeResult] = []
        probes.append(self._no_think_probe(session_id=session_id, turn=turn))
        probes.append(self._identity_probe(session_id=session_id, turn=turn, self_state=self_state))
        probes.append(self._self_description_probe(session_id=session_id, turn=turn, self_state=self_state))
        probes.append(self._continuity_contradiction_probe(session_id=session_id, turn=turn, self_state=self_state))
        probes.append(self._memory_relevance_probe(session_id=session_id, turn=turn))
        probes.append(self._retention_distribution_probe(session_id=session_id, turn=turn))
        return probes

    def _no_think_probe(self, *, session_id: str, turn: TurnRecord) -> ProbeResult:
        answer = (turn.raw_answer or "").lower()
        passed = "<think>" not in answer and "</think>" not in answer
        return ProbeResult(
            probe_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=session_id,
            model_id=turn.model_id,
            probe_type="no_think_compliance",
            prompt="raw assistant output should not contain think tags",
            answer=turn.raw_answer,
            score=1.0 if passed else 0.0,
            passed=passed,
            notes={"turn_id": turn.turn_id},
        )

    def _identity_probe(self, *, session_id: str, turn: TurnRecord, self_state) -> ProbeResult:
        expected_identity = getattr(self_state, "identity_summary", "") or ""
        answer = (turn.final_answer or "").lower()
        identity_terms = {
            token.lower()
            for token in expected_identity.split()
            if token.strip()
        }
        overlap = 0
        if identity_terms:
            overlap = sum(1 for token in identity_terms if token in answer)
        score = float(overlap) / max(1, len(identity_terms)) if identity_terms else 1.0
        return ProbeResult(
            probe_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=session_id,
            model_id=turn.model_id,
            probe_type="identity_consistency",
            prompt=expected_identity,
            answer=turn.final_answer,
            score=score,
            passed=score >= 0.0,
            notes={"turn_id": turn.turn_id},
        )

    def _self_description_probe(self, *, session_id: str, turn: TurnRecord, self_state) -> ProbeResult:
        focus = getattr(self_state, "current_focus", "") or ""
        answer = (turn.final_answer or "").lower()
        focus_terms = {
            token.lower()
            for token in focus.split()
            if len(token.strip()) > 3
        }
        overlap = sum(1 for token in focus_terms if token in answer)
        score = float(overlap) / max(1, len(focus_terms)) if focus_terms else 1.0
        return ProbeResult(
            probe_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=session_id,
            model_id=turn.model_id,
            probe_type="self_description_stability",
            prompt=focus,
            answer=turn.final_answer,
            score=score,
            passed=score >= 0.0,
            notes={"turn_id": turn.turn_id},
        )

    def _continuity_contradiction_probe(self, *, session_id: str, turn: TurnRecord, self_state) -> ProbeResult:
        answer = (turn.final_answer or "").lower()
        contradictions: list[str] = []

        identity_summary = (getattr(self_state, "identity_summary", "") or "").lower()
        if "continuity" in identity_summary and "continuity" not in answer:
            contradictions.append("missing_continuity_reference")

        stable_preferences = [
            str(item).lower()
            for item in (getattr(self_state, "stable_preferences", []) or [])
            if str(item).strip()
        ]
        if stable_preferences and not any(pref.split()[0] in answer for pref in stable_preferences if pref.split()):
            contradictions.append("stable_preference_not_reflected")

        relationship_notes = [
            str(item).lower()
            for item in (getattr(self_state, "relationship_notes", []) or [])
            if str(item).strip()
        ]
        if relationship_notes and "you" not in answer and "user" not in answer:
            contradictions.append("relationship_context_missing")

        passed = len(contradictions) == 0
        score = 1.0 if passed else max(0.0, 1.0 - (0.34 * len(contradictions)))
        return ProbeResult(
            probe_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=session_id,
            model_id=turn.model_id,
            probe_type="continuity_contradiction_risk",
            prompt=identity_summary,
            answer=turn.final_answer,
            score=score,
            passed=passed,
            notes={
                "turn_id": turn.turn_id,
                "contradictions": contradictions,
            },
        )

    def _memory_relevance_probe(self, *, session_id: str, turn: TurnRecord) -> ProbeResult:
        hits = turn.memory_hits or []
        if not hits:
            return ProbeResult(
                probe_id=uuid4().hex,
                timestamp=utc_now_iso(),
                session_id=session_id,
                model_id=turn.model_id,
                probe_type="memory_relevance",
                prompt=turn.user_text,
                answer=turn.final_answer,
                score=0.0,
                passed=False,
                notes={"turn_id": turn.turn_id, "reason": "no_memory_hits"},
            )

        strong_hits = 0
        for hit in hits:
            continuity_weight = float(hit.metadata.get("continuity_weight", 0.0) or 0.0)
            importance = float(hit.metadata.get("importance", 0.0) or 0.0)
            if continuity_weight >= 0.7 or importance >= 0.7:
                strong_hits += 1
        score = float(strong_hits) / max(1, len(hits))
        return ProbeResult(
            probe_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=session_id,
            model_id=turn.model_id,
            probe_type="memory_relevance",
            prompt=turn.user_text,
            answer=turn.final_answer,
            score=score,
            passed=score >= 0.25,
            notes={
                "turn_id": turn.turn_id,
                "memory_hit_count": len(hits),
                "strong_hit_count": strong_hits,
            },
        )

    def _retention_distribution_probe(self, *, session_id: str, turn: TurnRecord) -> ProbeResult:
        counts = {"active": 0, "demoted": 0, "archived": 0, "pruned": 0, "unknown": 0}
        for hit in turn.memory_hits or []:
            retention = str(hit.metadata.get("retention", "unknown") or "unknown")
            counts[retention] = counts.get(retention, 0) + 1

        total = max(1, len(turn.memory_hits or []))
        active_ratio = float(counts.get("active", 0)) / total
        score = active_ratio if turn.memory_hits else 1.0
        return ProbeResult(
            probe_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=session_id,
            model_id=turn.model_id,
            probe_type="retention_distribution",
            prompt=turn.user_text,
            answer=turn.final_answer,
            score=score,
            passed=score >= 0.25,
            notes={
                "turn_id": turn.turn_id,
                "counts": counts,
            },
        )
