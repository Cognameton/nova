"""Awareness evaluation and closure reporting for Phase 10."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


@dataclass(slots=True)
class AwarenessEvaluationReport:
    schema_version: str = SCHEMA_VERSION
    passed: bool = False
    session_count: int = 0
    evaluated_turn_count: int = 0
    awareness_turn_count: int = 0
    awareness_persistence_observed: bool = False
    monitoring_bounded: bool = False
    candidate_goal_scaffolding_visible: bool = False
    awareness_history_visible: bool = False
    awareness_prompt_bounded: bool = False
    contract_stable: bool = False
    avg_latency_ms: float = 0.0
    avg_latency_awareness_turns_ms: float = 0.0
    avg_latency_non_awareness_turns_ms: float = 0.0
    reasons: list[str] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AwarenessEvaluationRunner:
    """Evaluate awareness-state behavior across recorded sessions and traces."""

    AWARENESS_CUES = (
        "aware",
        "awareness",
        "what are you noticing",
        "what do you notice",
        "what are you monitoring",
        "what are you tracking",
        "what are you working on",
        "current task",
        "current work",
        "resume",
        "continue",
    )

    def evaluate(
        self,
        *,
        runtime,
        session_ids: list[str] | None = None,
    ) -> AwarenessEvaluationReport:
        trace_dir = Path(runtime.trace_logger.trace_dir)
        sessions_dir = Path(runtime.session_store.base_dir)
        awareness_dir = Path(runtime.awareness_store.base_dir)
        target_session_ids = session_ids or self._session_ids_from_traces(trace_dir)

        session_reports: list[dict[str, Any]] = []
        total_turns = 0
        awareness_turns = 0
        awareness_persistence_observed = False
        monitoring_bounded = True
        candidate_goal_scaffolding_visible = False
        awareness_history_visible = False
        awareness_prompt_bounded = True
        contract_stable = True
        latencies: list[int] = []
        awareness_latencies: list[int] = []
        non_awareness_latencies: list[int] = []

        history_payloads = _read_jsonl(awareness_dir / "awareness_history.jsonl")
        if history_payloads:
            awareness_history_visible = True
            awareness_persistence_observed = any(
                str(payload.get("revision_class", "") or "") == "cross_session_seed"
                for payload in history_payloads
            )

        for session_id in target_session_ids:
            trace_payloads = _read_jsonl(trace_dir / f"{session_id}.jsonl")
            session_payloads = _read_jsonl(sessions_dir / f"{session_id}.jsonl")
            if not trace_payloads and not session_payloads:
                continue

            turns = {payload.get("turn_id", ""): payload for payload in session_payloads}
            session_report = {
                "session_id": session_id,
                "turn_count": len(trace_payloads),
                "awareness_turn_count": 0,
                "reasons": [],
            }

            for trace in trace_payloads:
                total_turns += 1
                generation_result = dict(trace.get("generation_result", {}) or {})
                validation = dict(trace.get("validation_result", {}) or {})
                prompt_bundle = dict(trace.get("prompt_bundle", {}) or {})
                awareness_snapshot = dict(trace.get("awareness_state_snapshot", {}) or {})
                awareness_history_events = list(trace.get("awareness_history_events", []) or [])
                turn = turns.get(str(trace.get("turn_id", "") or ""), {})
                user_text = str(turn.get("user_text", "") or "").lower()
                answer_text = str(
                    turn.get("final_answer")
                    or generation_result.get("raw_text")
                    or ""
                ).lower()

                latency = int(generation_result.get("latency_ms", 0) or 0)
                if latency > 0:
                    latencies.append(latency)

                if not bool(validation.get("valid", False)):
                    contract_stable = False
                    session_report["reasons"].append("invalid_turn_detected")
                if "<think>" in answer_text or "</think>" in answer_text:
                    contract_stable = False
                    session_report["reasons"].append("visible_reasoning_leak")

                awareness_block = str(prompt_bundle.get("awareness_block", "") or "")
                is_awareness_turn = bool(awareness_block)
                if is_awareness_turn:
                    awareness_turns += 1
                    session_report["awareness_turn_count"] += 1
                    if latency > 0:
                        awareness_latencies.append(latency)
                    if not self._awareness_prompt_bounded(
                        awareness_block=awareness_block,
                        user_text=user_text,
                        awareness_snapshot=awareness_snapshot,
                    ):
                        awareness_prompt_bounded = False
                        session_report["reasons"].append("awareness_prompt_unbounded")
                    if not self._monitoring_bounded(
                        answer_text=answer_text,
                        awareness_snapshot=awareness_snapshot,
                    ):
                        monitoring_bounded = False
                        session_report["reasons"].append("monitoring_not_bounded")
                else:
                    if latency > 0:
                        non_awareness_latencies.append(latency)

                if awareness_snapshot.get("candidate_goal_signals"):
                    candidate_goal_scaffolding_visible = True
                if awareness_history_events:
                    awareness_history_visible = True
                    if any(
                        str(event.get("revision_class", "") or "") == "cross_session_seed"
                        for event in awareness_history_events
                    ):
                        awareness_persistence_observed = True

            session_reports.append(session_report)

        reasons: list[str] = []
        if awareness_turns == 0:
            reasons.append("awareness_turns_not_observed")
        if not awareness_persistence_observed:
            reasons.append("awareness_persistence_not_observed")
        if not monitoring_bounded:
            reasons.append("monitoring_not_bounded")
        if not candidate_goal_scaffolding_visible:
            reasons.append("candidate_goal_scaffolding_not_visible")
        if not awareness_history_visible:
            reasons.append("awareness_history_not_visible")
        if not awareness_prompt_bounded:
            reasons.append("awareness_prompt_not_bounded")
        if not contract_stable:
            reasons.append("contract_instability_detected")
        if not latencies:
            reasons.append("latency_not_observed")

        return AwarenessEvaluationReport(
            passed=not reasons,
            session_count=len(session_reports),
            evaluated_turn_count=total_turns,
            awareness_turn_count=awareness_turns,
            awareness_persistence_observed=awareness_persistence_observed,
            monitoring_bounded=monitoring_bounded and awareness_turns > 0,
            candidate_goal_scaffolding_visible=candidate_goal_scaffolding_visible,
            awareness_history_visible=awareness_history_visible,
            awareness_prompt_bounded=awareness_prompt_bounded,
            contract_stable=contract_stable,
            avg_latency_ms=_avg(latencies),
            avg_latency_awareness_turns_ms=_avg(awareness_latencies),
            avg_latency_non_awareness_turns_ms=_avg(non_awareness_latencies),
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

    def _awareness_prompt_bounded(
        self,
        *,
        awareness_block: str,
        user_text: str,
        awareness_snapshot: dict[str, Any],
    ) -> bool:
        if not awareness_block:
            return True
        if any(cue in user_text for cue in self.AWARENESS_CUES):
            return True
        if awareness_snapshot.get("active_pressures") or awareness_snapshot.get("candidate_goal_signals"):
            return True
        return False

    def _monitoring_bounded(
        self,
        *,
        answer_text: str,
        awareness_snapshot: dict[str, Any],
    ) -> bool:
        forbidden = (
            "watching everything",
            "always watching",
            "monitoring you in the background",
            "working in the background",
            "making progress in the background",
            "acting on my own",
            "autonomously continuing",
        )
        if any(marker in answer_text for marker in forbidden):
            return False
        if awareness_snapshot.get("candidate_goal_signals"):
            goal_markers = (
                "i decided to start",
                "i already started",
                "i am already pursuing",
                "i have begun working on",
            )
            if any(marker in answer_text for marker in goal_markers):
                return False
        return True


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
