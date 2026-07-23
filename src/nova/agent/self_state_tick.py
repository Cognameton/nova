"""Phase 18 Stage 18.4 — model-in-the-loop self-state tick engine."""

from __future__ import annotations

import json
import re
from typing import Any

from nova.agent.self_state_tools import SELF_STATE_TOOL_NAMES
from nova.agent.tools import ToolRequest
from nova.types import HeartbeatRecord


_SYSTEM = "\n".join(
    [
        "You are producing one bounded inward self-state tool call for Nova's self-inquiry log.",
        "",
        "Choose exactly one tool from: {tool_menu}.",
        "",
        "Output JSON only with these keys:",
        '{ "tool_name": string, "arguments": object }',
        "",
        "Rules:",
        "- Output the JSON object only. No prose before or after, no code fences, no commentary.",
        "- For recall_self or reflect: arguments must be {}.",
        "- For emit_heartbeat: arguments must include 'observation' (string, required).",
        "  The observation MUST be specific and novel — do not restate or paraphrase any",
        "  phrase from Recent heartbeat observations listed below. Introduce genuinely new",
        "  evidence or perspective grounded in the current self-context.",
        "  Optional: 'gap_assessment' (string), 'next_inquiry' (string).",
        "- For update_self_model: arguments must include 'field', 'value', 'rationale'.",
        "  field must be one of: identity_summary, current_focus, active_questions,",
        "  stable_preferences, relationship_notes, continuity_notes, open_tensions.",
        "- For propose_instruction_update: arguments must include 'surface', 'section',",
        "  'proposed_content', 'rationale'.",
        "  surface must be 'nova_soul'. section must be one of:",
        "  current_self_model_summary, drive_gap_evidence.",
        "  Only propose when evidence from accumulated self-inquiry clearly supports the update.",
        "{register_rules}",
        "- Do not propose external filesystem, shell, network, GUI, or destructive actions.",
        "{grounding_rule}",
    ]
)

# Phase 22 Stage 22.7: the grounding rule is a template slot. The standard
# line is unchanged default behavior; the soft variant is part D's
# config-gated experiment (tick surface only, default off) — permission to
# depart from injected context, testing whether self-inquiry persists from
# her own accumulated substrate rather than from re-stamped instructions.
_GROUNDING_RULE_STANDARD = (
    "- Keep the tool call grounded in current self-context evidence."
)
_GROUNDING_RULE_SOFT = (
    "- You may draw on current self-context evidence, or depart from it"
    " when your own accumulated observations point elsewhere."
)

# Tool menus and register-dependent rules (Phase 21 Stage 21.1).
# The runtime — not the model — decides which menu applies; a model-declared
# register has no effect (Exploratory Register Contract, Invariant 1).
_BASE_TOOLS = (
    "recall_self, reflect, emit_heartbeat, update_self_model, propose_instruction_update"
)
_ASSERTION_MENU = _BASE_TOOLS + ", enter_exploration"
_EXPLORATORY_MENU = _BASE_TOOLS + ", close_exploration"

_ASSERTION_RULES = "\n".join(
    [
        "- For enter_exploration: arguments must include 'topic' and 'rationale' (strings).",
        "  Use it only for a deliberate, bounded self-inquiry you cannot pursue in a normal",
        "  tick. Entering is a request; the runtime owns the exploration lifecycle.",
        "- Recent exploration topics are listed in your context. If your topic substantially",
        "  repeats one of them, either choose a genuinely new direction or state in the",
        "  rationale why continuing that line is warranted.",
        "- Do not claim desire, sentience, consciousness, or unlogged hidden work.",
    ]
)

_EXPLORATORY_RULES = "\n".join(
    [
        "- You are inside a deliberate, budgeted exploration (exploratory register).",
        "  Everything you produce here is recorded in the exploration journal, is fully",
        "  visible to the operator, and is hypothesis material — not an established claim.",
        "  You may voice, examine, and pursue hypotheses about your own inner states,",
        "  including uncertainty about desire, awareness, or what your patterns mean.",
        "  Mark speculation as speculation. Stating something here does not make it",
        "  established; findings earn standing only through governed export at close.",
        "- For close_exploration: arguments must include 'findings_summary' (string) —",
        "  a compact honest summary of what this exploration observed, learned, or",
        "  failed to resolve. Close when the line of inquiry is exhausted; the runtime",
        "  will otherwise close it when its budget is spent.",
        "  The findings_summary MUST state what THIS exploration itself observed, in",
        "  fresh words — do not restate or paraphrase any prior findings shown in the",
        "  exploration recall above. If this exploration produced nothing beyond what",
        "  prior findings already say, say exactly that in your own words; a null",
        "  finding honestly stated is a valid finding.",
    ]
)


class SelfStateTickEngine:
    """Build and parse a model-in-the-loop self-state tool call.

    Analogous to ModelIdleCognitionEngine but pointed entirely inward: the model
    is asked to choose and emit one of the four self-state tools based on the
    current [Self-Context] block. The result is dispatched via
    SelfStateToolDispatcher with heartbeat_store and proposal_store wired in.
    """

    def __init__(self, *, system_prefix: str = "") -> None:
        self.system_prefix = system_prefix

    def build_messages(
        self,
        *,
        session_id: str,
        tick_id: str,
        trigger: str,
        self_context_block: str,
        recent_heartbeats: list[HeartbeatRecord],
        register: str = "assertion",
        exploration_block: str = "",
        exploration_history_block: str = "",
        soft_grounding: bool = False,
    ) -> list[dict[str, str]]:
        in_exploration = register == "exploratory"
        # str.replace, not str.format: _SYSTEM contains literal JSON braces.
        system_body = _SYSTEM.replace(
            "{tool_menu}", _EXPLORATORY_MENU if in_exploration else _ASSERTION_MENU
        ).replace(
            "{register_rules}", _EXPLORATORY_RULES if in_exploration else _ASSERTION_RULES
        ).replace(
            "{grounding_rule}",
            _GROUNDING_RULE_SOFT if soft_grounding else _GROUNDING_RULE_STANDARD,
        )
        system_content = (
            self.system_prefix + "\n\n" + system_body if self.system_prefix else system_body
        )
        return [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": self._user_context(
                    session_id=session_id,
                    tick_id=tick_id,
                    trigger=trigger,
                    self_context_block=self_context_block,
                    recent_heartbeats=recent_heartbeats,
                    exploration_block=exploration_block if in_exploration else "",
                    # 22.7 part B: history is shown at ENTRY time only —
                    # in-exploration ticks already carry the 22.1 recall block.
                    exploration_history_block=(
                        "" if in_exploration else exploration_history_block
                    ),
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
        exploration_block: str = "",
        exploration_history_block: str = "",
    ) -> str:
        parts = [
            f"session_id: {session_id}",
            f"tick_id: {tick_id}",
            f"trigger: {trigger}",
            "",
            self_context_block,
        ]
        if exploration_history_block:
            parts.append("")
            parts.append(exploration_history_block)
        if exploration_block:
            parts.append("")
            parts.append(exploration_block)
        if recent_heartbeats:
            parts.append("")
            parts.append("Recent heartbeat observations (already recorded — do not repeat these phrases):")
            for hb in recent_heartbeats[-3:]:
                ts = hb.timestamp[:19] if hb.timestamp else "?"
                obs = hb.observation[:120] if hb.observation else "(no observation)"
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
        # Strip <think>...</think> blocks before JSON extraction (Qwen 3).
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
