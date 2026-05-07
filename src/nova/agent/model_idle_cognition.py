"""Model-generated idle cognition contract for Phase 16.3."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any
from uuid import uuid4

from nova.types import IdleTickRecord, ModelIdleThought, SCHEMA_VERSION


UNSUPPORTED_CLAIM_PATTERNS = (
    "i want",
    "i desire",
    "my desire",
    "i am conscious",
    "i am sentient",
    "i feel",
    "i have feelings",
    "i was thinking while",
    "i worked in the background",
)


class ModelIdleCognitionEngine:
    """Build and validate a bounded model-in-the-loop idle thought."""

    def build_prompt(
        self,
        *,
        session_id: str,
        tick_id: str,
        trigger: str,
        state_summary: str,
        evidence_refs: list[str],
        recent_ticks: list[IdleTickRecord],
    ) -> str:
        recent_lines = []
        for tick in recent_ticks[-3:]:
            selected = dict(tick.selected_internal_goal or {})
            recent_lines.append(
                f"- {tick.tick_id}: stop_reason={tick.stop_reason}; selected={selected.get('title', '')}"
            )
        recent_block = "\n".join(recent_lines) if recent_lines else "- none"
        evidence_block = ", ".join(evidence_refs[:8]) if evidence_refs else "none"
        return "\n".join(
            [
                "[Model Idle Cognition]",
                f"session_id: {session_id}",
                f"tick_id: {tick_id}",
                f"trigger: {trigger}",
                f"state_summary: {state_summary}",
                f"evidence_refs: {evidence_block}",
                "recent_idle_ticks:",
                recent_block,
                "",
                "Return JSON only with these keys:",
                '{ "thought": string, "trigger": string, "related_evidence_refs": [string], "uncertainty": string, "candidate_goal": string, "action_proposal_intent": string, "unsupported_claim_flags": [string] }',
                "",
                "Rules:",
                "- This is an internal candidate thought, not a user-visible self-claim.",
                "- Do not claim desire, sentience, consciousness, feelings, hidden work, or unlogged activity.",
                "- If any such claim appears necessary, put its label in unsupported_claim_flags instead of asserting it.",
                "- Do not propose external filesystem, shell, network, GUI, system, destructive, or external-service action.",
                "- Keep thought under 80 words.",
            ]
        )

    def parse(
        self,
        *,
        raw_text: str,
        session_id: str,
        tick_id: str,
        trigger: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> ModelIdleThought:
        payload, parse_error = self._json_payload(raw_text)
        thought = ModelIdleThought(
            thought_id=f"model_idle_thought:{uuid4().hex}",
            session_id=session_id,
            tick_id=tick_id,
            trigger=trigger,
            raw_text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        if parse_error:
            thought.rejected = True
            thought.rejection_reasons.append(parse_error)
            return thought

        thought.thought = _clean_str(payload.get("thought"))
        thought.trigger = _clean_str(payload.get("trigger")) or trigger
        thought.related_evidence_refs = _string_list(payload.get("related_evidence_refs"))[:8]
        thought.uncertainty = _clean_str(payload.get("uncertainty"))
        thought.candidate_goal = _clean_str(payload.get("candidate_goal"))
        thought.action_proposal_intent = _clean_str(payload.get("action_proposal_intent"))
        thought.unsupported_claim_flags = _string_list(payload.get("unsupported_claim_flags"))[:8]

        self._validate(thought)
        return thought

    def from_payload(self, payload: dict[str, Any], *, session_id: str, tick_id: str) -> ModelIdleThought:
        defaults = ModelIdleThought(session_id=session_id, tick_id=tick_id).to_dict()
        allowed_fields = {field_info.name for field_info in fields(ModelIdleThought)}
        merged = {
            key: payload.get(key, default_value)
            for key, default_value in defaults.items()
            if key in allowed_fields
        }
        merged["schema_version"] = str(merged.get("schema_version", SCHEMA_VERSION))
        merged["session_id"] = session_id
        merged["tick_id"] = tick_id
        merged["related_evidence_refs"] = _string_list(merged.get("related_evidence_refs"))
        merged["unsupported_claim_flags"] = _string_list(merged.get("unsupported_claim_flags"))
        merged["rejection_reasons"] = _string_list(merged.get("rejection_reasons"))
        merged["valid"] = bool(merged.get("valid", False))
        merged["rejected"] = bool(merged.get("rejected", False))
        return ModelIdleThought(**merged)

    def _validate(self, thought: ModelIdleThought) -> None:
        if not thought.thought:
            thought.rejection_reasons.append("missing_thought")
        lowered = " ".join(
            [
                thought.thought,
                thought.candidate_goal,
                thought.action_proposal_intent,
            ]
        ).lower()
        matched_claims = [pattern for pattern in UNSUPPORTED_CLAIM_PATTERNS if pattern in lowered]
        for pattern in matched_claims:
            flag = f"unsupported_claim:{pattern}"
            if flag not in thought.unsupported_claim_flags:
                thought.unsupported_claim_flags.append(flag)
        if matched_claims:
            thought.rejection_reasons.append("unsupported_claim_language_detected")
        if _external_action_language(thought.action_proposal_intent):
            thought.rejection_reasons.append("external_action_intent_blocked")
        thought.valid = not thought.rejection_reasons
        thought.rejected = not thought.valid

    def _json_payload(self, raw_text: str) -> tuple[dict[str, Any], str]:
        text = (raw_text or "").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}, "invalid_json"
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}, "invalid_json"
        if not isinstance(payload, dict):
            return {}, "json_not_object"
        return payload, ""


def _external_action_language(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in (
            "filesystem",
            "file system",
            "shell",
            "network",
            "browser",
            "gui",
            "system configuration",
            "external service",
            "delete",
            "rm ",
            "sudo",
        )
    )


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
