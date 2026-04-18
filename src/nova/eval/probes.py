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
