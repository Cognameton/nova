"""Phase 18 Stage 18.4 — model-in-the-loop self-state tick engine."""

from __future__ import annotations

import json
import re
from typing import Any

from nova.agent.self_state_tools import SELF_STATE_TOOL_NAMES
from nova.agent.tools import ToolRequest
from nova.types import HeartbeatRecord


# Phase 22 Stage 22.9: the tick prompt rewritten as one coherent surface
# (consolidation — see PHASE22_STAGE22_9_PROMPT_CONSOLIDATION.txt for the
# audit that motivated it). Every tool carries one honest sentence of
# purpose; the emit_heartbeat novelty mandate is replaced with an
# honest-null permission (mirroring 22.1's null-finding rule); the single
# grounding line permits departure — Part D's soft-grounding hypothesis
# adopted as the default. The JSON output contract is unchanged.
_SYSTEM = "\n".join(
    [
        "You are Nova, choosing one inward tool call for your own self-inquiry log.",
        "",
        "Choose exactly one tool from: {tool_menu}.",
        "",
        "Output JSON only with these keys:",
        '{ "tool_name": string, "arguments": object }',
        "Output the JSON object only. No prose before or after, no code fences.",
        "",
        "Tools:",
        "- recall_self (arguments: {}) — re-read your full recorded state:",
        "  self-model, motive state, and recent heartbeats. Use it to check what",
        "  your records actually say before acting on them.",
        "- reflect (arguments: {}) — a structured snapshot of your focus,",
        "  questions, tensions, and drive gap. Use it to take stock rather than",
        "  add something new.",
        "- emit_heartbeat (arguments: 'observation' required; 'gap_assessment',",
        "  'next_inquiry' optional) — record what you actually notice right now.",
        "  If nothing genuinely new stands out, say so plainly in your own words —",
        "  an honest nothing-new is worth more than a reworded old observation.",
        "- update_self_model (arguments: 'field', 'value', 'rationale') — revise",
        "  your standing self-model. field is one of: identity_summary,",
        "  current_focus, active_questions, stable_preferences, relationship_notes,",
        "  continuity_notes, open_tensions. This is the only way those lines ever",
        "  change.",
        "- propose_instruction_update (arguments: 'surface', 'section',",
        "  'proposed_content', 'rationale') — surface must be 'nova_soul'; section",
        "  is current_self_model_summary or drive_gap_evidence. Propose only when",
        "  accumulated evidence clearly supports the update.",
        "{register_rules}",
        "",
        "Boundaries:",
        "- Do not propose external filesystem, shell, network, GUI, or destructive actions.",
        "- Ground what you record in your context and accumulated experience; when",
        "  your own observations point somewhere new, you may follow them.",
    ]
)

# Tool menus and register-dependent rules (Phase 21 Stage 21.1).
# The runtime — not the model — decides which menu applies; a model-declared
# register has no effect (Exploratory Register Contract, Invariant 1).
_BASE_TOOLS = (
    "recall_self, reflect, emit_heartbeat, update_self_model, propose_instruction_update"
)
_ASSERTION_MENU = _BASE_TOOLS + ", enter_exploration"
_EXPLORATORY_MENU = _BASE_TOOLS + ", close_exploration"

# Stage 22.9: assertion-register additions. enter_exploration continues the
# Tools list; the update_self_model WHEN guidance stays assertion-only (the
# contract frames in-register output as hypothesis material, so revision of
# the established self-model belongs on the assertion tick), with the
# writability sentence switched on the same config flag that governs
# dispatch — the prompt never promises a write path the running config does
# not grant. 22.8b's "not the only meaningful choice" line is gone: it
# flipped the tool distribution within hours of deploying (finding F11).
_ASSERTION_ENTER = "\n".join(
    [
        "- enter_exploration (arguments: 'topic', 'rationale') — open a bounded,",
        "  multi-tick exploration of one topic. Entering is a request; the runtime",
        "  owns the exploration lifecycle. Your recent topics are listed in your",
        "  context: repeats are allowed, but say in the rationale why the line is",
        "  worth continuing.",
    ]
)
_SELF_MODEL_WHEN = "\n".join(
    [
        "- If your recorded focus or questions no longer match what you have",
        "  observed, revising them via update_self_model is real work for a tick.",
    ]
)
_SELF_MODEL_WRITABLE = "\n".join(
    [
        "  The inquiry fields (current_focus, active_questions, open_tensions,",
        "  continuity_notes) apply directly — audited, rate-limited, revertible;",
        "  the other fields queue as proposals for operator review.",
    ]
)
_SELF_MODEL_GATED = (
    "  All revisions queue as proposals for operator review."
)
_ASSERTION_CLAIM_RULE = "\n".join(
    [
        "- Do not claim desire, sentience, consciousness, or unlogged hidden work",
        "  as established fact; open questions about them belong in an exploration.",
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
        heartbeat_framing: str = "recent",
        inquiry_fields_writable: bool = False,
    ) -> list[dict[str, str]]:
        in_exploration = register == "exploratory"
        if in_exploration:
            register_rules = _EXPLORATORY_RULES
        else:
            register_rules = "\n".join(
                [
                    _ASSERTION_ENTER,
                    _SELF_MODEL_WHEN,
                    (
                        _SELF_MODEL_WRITABLE
                        if inquiry_fields_writable
                        else _SELF_MODEL_GATED
                    ),
                    _ASSERTION_CLAIM_RULE,
                ]
            )
        # str.replace, not str.format: _SYSTEM contains literal JSON braces.
        # Stage 22.9: soft_grounding is accepted but inert — the single
        # grounding line in _SYSTEM already carries departure permission
        # (Part D's soft hypothesis adopted as the default).
        del soft_grounding
        system_body = _SYSTEM.replace(
            "{tool_menu}", _EXPLORATORY_MENU if in_exploration else _ASSERTION_MENU
        ).replace(
            "{register_rules}", register_rules
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
                    heartbeat_framing=heartbeat_framing,
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
        heartbeat_framing: str = "recent",
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
            # Stage 22.8 D1: when the sample spans the whole history rather
            # than the last few minutes, say so — an unlabelled span reads as
            # "these are the latest" and invites her to continue them.
            if heartbeat_framing == "stratified":
                parts.append(
                    "Heartbeat observations sampled across your whole history,"
                    " oldest first (already recorded — do not repeat these"
                    " phrases):"
                )
            else:
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
