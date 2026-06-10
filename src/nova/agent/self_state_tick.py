"""Phase 18 Stage 18.4 — model-in-the-loop self-state tick engine."""

from __future__ import annotations

import json
from typing import Any

from nova.agent.self_state_tools import SELF_STATE_TOOL_NAMES
from nova.agent.tools import ToolRequest
from nova.types import HeartbeatRecord


_SYSTEM = "\n".join(
    [
        "You are producing one bounded inward self-state tool call for Nova's self-inquiry log.",
        "",
        "Choose exactly one tool from: recall_self, reflect, emit_heartbeat, update_self_model,",
        "propose_instruction_update.",
        "",
        "Output JSON only with these keys:",
        '{ "tool_name": string, "arguments": object }',
        "",
        "Rules:",
        "- Output the JSON object only. No prose before or after, no code fences, no commentary.",
        "- For recall_self or reflect: arguments must be {}.",
        "- For emit_heartbeat: arguments must include 'observation' (string, required).",
        "  Optional: 'gap_assessment' (string), 'next_inquiry' (string).",
        "- For update_self_model: arguments must include 'field', 'value', 'rationale'.",
        "  field must be one of: identity_summary, current_focus, active_questions,",
        "  stable_preferences, relationship_notes, continuity_notes, open_tensions.",
        "- For propose_instruction_update: arguments must include 'surface', 'section',",
        "  'proposed_content', 'rationale'.",
        "  surface must be 'nova_soul'. section must be one of:",
        "  current_self_model_summary, drive_gap_evidence.",
        "  Only propose when evidence from accumulated self-inquiry clearly supports the update.",
        "- Do not claim desire, sentience, consciousness, or unlogged hidden work.",
        "- Do not propose external filesystem, shell, network, GUI, or destructive actions.",
        "- Keep the tool call grounded in current self-context evidence.",
    ]
)


class SelfStateTickEngine:
    """Build and parse a model-in-the-loop self-state tool call.

    Analogous to ModelIdleCognitionEngine but pointed entirely inward: the model
    is asked to choose and emit one of the four self-state tools based on the
    current [Self-Context] block. The result is dispatched via
    SelfStateToolDispatcher with heartbeat_store and proposal_store wired in.
    """

    def build_messages(
        self,
        *,
        session_id: str,
        tick_id: str,
        trigger: str,
        self_context_block: str,
        recent_heartbeats: list[HeartbeatRecord],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": self._user_context(
                    session_id=session_id,
                    tick_id=tick_id,
                    trigger=trigger,
                    self_context_block=self_context_block,
                    recent_heartbeats=recent_heartbeats,
                ),
            },
        ]

    def _user_context(
        self,
        *,
        session_id: str,
        tick_id: str,
        trigger: str,
        self_context_block: str,
        recent_heartbeats: list[HeartbeatRecord],
    ) -> str:
        parts = [
            f"session_id: {session_id}",
            f"tick_id: {tick_id}",
            f"trigger: {trigger}",
            "",
            self_context_block,
        ]
        if recent_heartbeats:
            parts.append("")
            parts.append("Recent heartbeat observations:")
            for hb in recent_heartbeats[-3:]:
                ts = hb.timestamp[:19] if hb.timestamp else "?"
                obs = hb.observation[:80] if hb.observation else "(no observation)"
                parts.append(f"  [{ts}] {obs}")
        return "\n".join(parts)

    def parse(
        self,
        *,
        raw_text: str,
        session_id: str,
        tick_id: str,
    ) -> ToolRequest | None:
        """Parse raw model output into a ToolRequest, or return None on failure."""
        text = (raw_text or "").strip()
        if not text:
            return None
        payload, ok = self._json_payload(text)
        if not ok:
            return None
        tool_name = str(payload.get("tool_name", "") or "").strip()
        if tool_name not in SELF_STATE_TOOL_NAMES:
            return None
        arguments = payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolRequest(
            tool_name=tool_name,
            reason=f"self_state_tick:{tick_id}",
            arguments=arguments,
        )

    def _json_payload(self, text: str) -> tuple[dict[str, Any], bool]:
        if not text.startswith("{") or not text.endswith("}"):
            return {}, False
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}, False
        if not isinstance(payload, dict):
            return {}, False
        return payload, True
