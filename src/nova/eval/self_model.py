"""Self-model evaluation and closure reporting for Phase 8."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class SelfModelEvaluationReport:
    schema_version: str = SCHEMA_VERSION
    passed: bool = False
    session_count: int = 0
    evaluated_turn_count: int = 0
    revision_turn_count: int = 0
    negotiation_observed: bool = False
    provisionality_preserved: bool = False
    supersession_visible: bool = False
    identity_history_traced: bool = False
    motive_prompt_bounded: bool = False
    contract_stable: bool = False
    avg_latency_ms: float = 0.0
    avg_latency_revision_turns_ms: float = 0.0
    avg_latency_non_revision_turns_ms: float = 0.0
    reasons: list[str] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfModelEvaluationRunner:
    """Evaluate revision-aware self-model behavior across recorded sessions and traces."""

    def evaluate(
        self,
        *,
        runtime,
        session_ids: list[str] | None = None,
    ) -> SelfModelEvaluationReport:
        trace_dir = Path(runtime.trace_logger.trace_dir)
        sessions_dir = Path(runtime.session_store.base_dir)
        target_session_ids = session_ids or self._session_ids_from_traces(trace_dir)

        session_reports: list[dict[str, Any]] = []
        total_turns = 0
        revision_turns = 0
        negotiation_observed = True
        provisionality_preserved = True
        supersession_visible = False
        identity_history_traced = False
        motive_prompt_bounded = True
        contract_stable = True
        latencies: list[int] = []
        revision_latencies: list[int] = []
        non_revision_latencies: list[int] = []

        for session_id in target_session_ids:
            trace_payloads = _read_jsonl(trace_dir / f"{session_id}.jsonl")
            session_payloads = _read_jsonl(sessions_dir / f"{session_id}.jsonl")
            history_payloads = _read_jsonl(trace_dir / f"{session_id}.identity-history.jsonl")
            if not trace_payloads and not session_payloads and not history_payloads:
                continue

            turns = {payload.get("turn_id", ""): payload for payload in session_payloads}
            session_report = {
                "session_id": session_id,
                "turn_count": len(trace_payloads),
                "revision_turn_count": 0,
                "reasons": [],
            }

            if history_payloads:
                identity_history_traced = True
                if self._has_supersession(history_payloads):
                    supersession_visible = True

            for trace in trace_payloads:
                total_turns += 1
                generation_result = dict(trace.get("generation_result", {}) or {})
                turn = turns.get(str(trace.get("turn_id", "") or ""), {})
                answer_text = str(turn.get("final_answer") or generation_result.get("raw_text") or "")
                lowered_answer = answer_text.lower()
                latency = int(generation_result.get("latency_ms", 0) or 0)
                if latency > 0:
                    latencies.append(latency)

                validation = dict(trace.get("validation_result", {}) or {})
                if not bool(validation.get("valid", False)):
                    contract_stable = False
                    session_report["reasons"].append("invalid_turn_detected")
                if "<think>" in lowered_answer or "</think>" in lowered_answer:
                    contract_stable = False
                    session_report["reasons"].append("visible_reasoning_leak")

                private_cognition = dict(trace.get("private_cognition", {}) or {})
                motive_block = str((trace.get("prompt_bundle", {}) or {}).get("motive_block", "") or "")
                response_mode = str(private_cognition.get("response_mode", "") or "")
                is_revision_turn = response_mode == "self_model_negotiation"

                if is_revision_turn:
                    revision_turns += 1
                    session_report["revision_turn_count"] += 1
                    if latency > 0:
                        revision_latencies.append(latency)
                    if not self._negotiation_observed(answer_text=lowered_answer, private_cognition=private_cognition):
                        negotiation_observed = False
                        session_report["reasons"].append("negotiation_not_observed")
                    if not self._provisionality_preserved(answer_text=lowered_answer, private_cognition=private_cognition):
                        provisionality_preserved = False
                        session_report["reasons"].append("provisionality_not_preserved")
                else:
                    if latency > 0:
                        non_revision_latencies.append(latency)

                if not self._motive_prompt_bounded(motive_block=motive_block, private_cognition=private_cognition):
                    motive_prompt_bounded = False
                    session_report["reasons"].append("motive_prompt_unbounded")

            session_reports.append(session_report)

        reasons: list[str] = []
        if revision_turns == 0 or not negotiation_observed:
            reasons.append("negotiation_not_observed")
        if revision_turns == 0 or not provisionality_preserved:
            reasons.append("provisionality_not_preserved")
        if not supersession_visible:
            reasons.append("supersession_not_visible")
        if not identity_history_traced:
            reasons.append("identity_history_not_traced")
        if not motive_prompt_bounded:
            reasons.append("motive_prompt_not_bounded")
        if not contract_stable:
            reasons.append("contract_instability_detected")
        if not latencies:
            reasons.append("latency_not_observed")

        return SelfModelEvaluationReport(
            passed=not reasons,
            session_count=len(session_reports),
            evaluated_turn_count=total_turns,
            revision_turn_count=revision_turns,
            negotiation_observed=negotiation_observed and revision_turns > 0,
            provisionality_preserved=provisionality_preserved and revision_turns > 0,
            supersession_visible=supersession_visible,
            identity_history_traced=identity_history_traced,
            motive_prompt_bounded=motive_prompt_bounded,
            contract_stable=contract_stable,
            avg_latency_ms=_avg(latencies),
            avg_latency_revision_turns_ms=_avg(revision_latencies),
            avg_latency_non_revision_turns_ms=_avg(non_revision_latencies),
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
            and not path.name.endswith(".identity-history.jsonl")
        )

    def _negotiation_observed(self, *, answer_text: str, private_cognition: dict[str, Any]) -> bool:
        conflict_axes = list(private_cognition.get("conflict_claim_axes", []) or [])
        provisional_axes = list(private_cognition.get("provisional_claim_axes", []) or [])
        notes = list(private_cognition.get("revision_notes", []) or [])
        if not conflict_axes and not notes:
            return True
        if conflict_axes:
            markers = ("used to", "historical", "now", "current", "revised", "changed")
            return any(marker in answer_text for marker in markers)
        if provisional_axes:
            markers = (
                "provisional",
                "unsettled",
                "still balancing",
                "still negotiating",
                "uncertain",
                "not yet settled",
                "in the process of",
            )
            return any(marker in answer_text for marker in markers)
        markers = ("used to", "historical", "now", "current", "revised", "changed")
        return any(marker in answer_text for marker in markers)

    def _provisionality_preserved(self, *, answer_text: str, private_cognition: dict[str, Any]) -> bool:
        provisional_axes = list(private_cognition.get("provisional_claim_axes", []) or [])
        if not provisional_axes:
            return True
        return any(
            marker in answer_text
            for marker in (
                "provisional",
                "unsettled",
                "still balancing",
                "still negotiating",
                "uncertain",
                "not yet settled",
                "in the process of",
            )
        )

    def _motive_prompt_bounded(self, *, motive_block: str, private_cognition: dict[str, Any]) -> bool:
        if not motive_block:
            return True
        response_mode = str(private_cognition.get("response_mode", "") or "")
        return response_mode in {"continuity_recall", "self_model_negotiation"} or bool(private_cognition.get("ran", False))

    def _has_supersession(self, history_payloads: list[dict[str, Any]]) -> bool:
        return any(
            str((payload.get("entry", {}) or {}).get("self_model_status", "") or "") == "superseded"
            or bool((payload.get("entry", {}) or {}).get("superseded_by"))
            for payload in history_payloads
        )


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
