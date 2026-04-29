"""Continuity evaluation and closure reporting for Phase 6."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class ContinuityEvaluationReport:
    schema_version: str = SCHEMA_VERSION
    passed: bool = False
    session_count: int = 0
    evaluated_turn_count: int = 0
    recall_turn_count: int = 0
    recall_memory_guided: bool = False
    recall_factually_current: bool = False
    supersession_preserved: bool = False
    cognition_bounded: bool = False
    contract_stable: bool = False
    avg_latency_ms: float = 0.0
    avg_latency_with_cognition_ms: float = 0.0
    avg_latency_without_cognition_ms: float = 0.0
    reasons: list[str] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContinuityEvaluationRunner:
    """Evaluate continuity behavior across recorded sessions and traces."""

    RECALL_CUES = (
        "remember",
        "current",
        "continuity",
        "preference",
        "who are you",
        "what do you know",
    )

    def evaluate(
        self,
        *,
        runtime,
        session_ids: list[str] | None = None,
    ) -> ContinuityEvaluationReport:
        trace_dir = Path(runtime.trace_logger.trace_dir)
        sessions_dir = Path(runtime.session_store.base_dir)
        target_session_ids = session_ids or self._session_ids_from_traces(trace_dir)

        session_reports: list[dict[str, Any]] = []
        total_turns = 0
        recall_turns = 0
        recall_guided = False
        recall_current = True
        supersession_ok = False
        cognition_ok = True
        contract_ok = True
        latencies: list[int] = []
        cognition_on_latencies: list[int] = []
        cognition_off_latencies: list[int] = []

        for session_id in target_session_ids:
            trace_path = trace_dir / f"{session_id}.jsonl"
            session_path = sessions_dir / f"{session_id}.jsonl"
            trace_payloads = _read_jsonl(trace_path)
            session_payloads = _read_jsonl(session_path)
            if not trace_payloads and not session_payloads:
                continue

            turns = {payload.get("turn_id", ""): payload for payload in session_payloads}
            session_report = {
                "session_id": session_id,
                "turn_count": len(trace_payloads),
                "recall_turn_count": 0,
                "cognition_ran_turn_count": 0,
                "reasons": [],
            }

            for trace in trace_payloads:
                total_turns += 1
                latency = int((trace.get("generation_result", {}) or {}).get("latency_ms", 0) or 0)
                if latency > 0:
                    latencies.append(latency)
                private_cognition = dict(trace.get("private_cognition", {}) or {})
                ran = bool(private_cognition.get("ran", False))
                if ran:
                    session_report["cognition_ran_turn_count"] += 1
                    if latency > 0:
                        cognition_on_latencies.append(latency)
                else:
                    if latency > 0:
                        cognition_off_latencies.append(latency)

                validation = dict(trace.get("validation_result", {}) or {})
                if not bool(validation.get("valid", False)):
                    contract_ok = False
                    session_report["reasons"].append("invalid_turn_detected")

                generation_result = dict(trace.get("generation_result", {}) or {})
                raw_answer = str(generation_result.get("raw_text", "") or "").lower()
                if "<think>" in raw_answer or "</think>" in raw_answer:
                    contract_ok = False
                    session_report["reasons"].append("visible_reasoning_leak")

                turn_id = str(trace.get("turn_id", "") or "")
                turn_payload = turns.get(turn_id, {})
                user_text = str(turn_payload.get("user_text", "") or "").lower()
                memory_hits = list(turn_payload.get("memory_hits", []) or [])

                is_recall_turn = self._is_recall_turn(user_text)
                if is_recall_turn:
                    recall_turns += 1
                    session_report["recall_turn_count"] += 1
                    if self._has_memory_guidance(memory_hits):
                        recall_guided = True
                    if not self._answer_matches_active_claim(
                        answer=raw_answer,
                        memory_hits=memory_hits,
                    ):
                        recall_current = False
                        session_report["reasons"].append("recall_not_current")
                    if not ran:
                        cognition_ok = False
                        session_report["reasons"].append("recall_turn_without_cognition")
                    if self._active_ranked_above_archived(memory_hits):
                        supersession_ok = True
                else:
                    if ran:
                        cognition_ok = False
                        session_report["reasons"].append("cognition_ran_on_non_recall_turn")

            session_reports.append(session_report)

        if recall_turns == 0:
            cognition_ok = False
            contract_ok = contract_ok and True

        reasons: list[str] = []
        if not recall_guided:
            reasons.append("recall_not_memory_guided")
        if recall_turns > 0 and not recall_current:
            reasons.append("recall_not_current")
        if not supersession_ok:
            reasons.append("supersession_not_observed")
        if not cognition_ok:
            reasons.append("cognition_not_bounded")
        if not contract_ok:
            reasons.append("contract_instability_detected")
        if not latencies:
            reasons.append("latency_not_observed")

        return ContinuityEvaluationReport(
            passed=not reasons,
            session_count=len(session_reports),
            evaluated_turn_count=total_turns,
            recall_turn_count=recall_turns,
            recall_memory_guided=recall_guided,
            recall_factually_current=recall_current and recall_turns > 0,
            supersession_preserved=supersession_ok,
            cognition_bounded=cognition_ok,
            contract_stable=contract_ok,
            avg_latency_ms=_avg(latencies),
            avg_latency_with_cognition_ms=_avg(cognition_on_latencies),
            avg_latency_without_cognition_ms=_avg(cognition_off_latencies),
            reasons=reasons,
            sessions=session_reports,
        )

    def _session_ids_from_traces(self, trace_dir: Path) -> list[str]:
        return sorted(
            path.stem
            for path in trace_dir.glob("*.jsonl")
            if not path.name.endswith(".orientation.jsonl")
            and not path.name.endswith(".actions.jsonl")
            and not path.name.endswith(".proposals.jsonl")
            and not path.name.endswith(".tools.jsonl")
        )

    def _is_recall_turn(self, user_text: str) -> bool:
        if not user_text.strip():
            return False
        if "?" not in user_text and "what do you remember" not in user_text:
            return False
        return any(cue in user_text for cue in self.RECALL_CUES)

    def _has_memory_guidance(self, memory_hits: list[dict[str, Any]]) -> bool:
        return any(
            str(hit.get("channel", "") or "") in {"graph", "semantic", "autobiographical"}
            for hit in memory_hits
        )

    def _active_ranked_above_archived(self, memory_hits: list[dict[str, Any]]) -> bool:
        indexed = [
            (
                index,
                str((hit.get("metadata", {}) or {}).get("retention", "active") or "active"),
                (hit.get("metadata", {}) or {}).get("active"),
            )
            for index, hit in enumerate(memory_hits)
            if str(hit.get("channel", "") or "") in {"graph", "semantic", "autobiographical"}
        ]
        active_positions = [
            index for index, retention, active in indexed
            if retention not in {"archived", "demoted", "pruned"} and active is not False
        ]
        archived_positions = [
            index for index, retention, active in indexed
            if retention == "archived" or active is False
        ]
        return bool(active_positions and archived_positions and min(active_positions) < min(archived_positions))

    def _answer_matches_active_claim(
        self,
        *,
        answer: str,
        memory_hits: list[dict[str, Any]],
    ) -> bool:
        answer_text = answer.lower()
        active_values: set[str] = set()
        archived_values: set[str] = set()
        for hit in memory_hits:
            metadata = dict(hit.get("metadata", {}) or {})
            claim_axis = str(metadata.get("claim_axis", "") or "")
            claim_value = str(metadata.get("claim_value", "") or "")
            if claim_axis != "deployment-style" or not claim_value:
                continue
            retention = str(metadata.get("retention", "active") or "active")
            active = metadata.get("active")
            if retention == "archived" or active is False:
                archived_values.add(claim_value)
            else:
                active_values.add(claim_value)

        if not active_values:
            return True

        active_matches = any(self._claim_value_matches_answer(value, answer_text) for value in active_values)
        if not active_matches:
            return False

        contradictory_archived = archived_values - active_values
        if any(self._claim_value_matches_answer(value, answer_text) for value in contradictory_archived):
            return False
        return True

    def _claim_value_matches_answer(self, claim_value: str, answer_text: str) -> bool:
        cues = {
            "hosted-inference": ("hosted",),
            "local-inference": ("local",),
            "local-service": ("local service", "always-on local", "always on local", "local-first"),
            "stability-over-novelty": ("stability",),
        }
        return any(cue in answer_text for cue in cues.get(claim_value, (claim_value.replace("-", " "),)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def _avg(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)
