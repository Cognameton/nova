"""Prompt composition for Nova 2.0."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from nova.types import PersonaState, PromptBundle, RetrievalHit, SelfState, TurnRecord


class NovaPromptComposer:
    """Deterministic Phase 1 prompt composer."""

    def __init__(
        self,
        *,
        token_counter: Callable[[str], int],
        memory_char_limit: int = 240,
        recent_turn_char_limit: int = 800,
    ):
        self.token_counter = token_counter
        self.memory_char_limit = memory_char_limit
        self.recent_turn_char_limit = recent_turn_char_limit

    def compose(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        memory_hits: list[RetrievalHit],
        recent_turns: list[TurnRecord],
        user_text: str,
        contract_rules: list[str],
        session_id: str,
        turn_id: str,
    ) -> PromptBundle:
        persona_block = self._format_persona(persona)
        self_state_block = self._format_self_state(self_state)
        memory_blocks = self._format_memory_blocks(memory_hits)
        recent_turns_block = self._format_recent_turns(recent_turns)
        user_block = f"[User]\n{user_text.strip()}"
        response_contract_block = self._format_contract_rules(contract_rules)

        parts = [
            persona_block,
            self_state_block,
            *[block for block in memory_blocks.values() if block],
            recent_turns_block,
            user_block,
            response_contract_block,
        ]
        full_prompt = "\n\n".join(part for part in parts if part.strip())
        token_estimate = self.token_counter(full_prompt)

        return PromptBundle(
            session_id=session_id,
            turn_id=turn_id,
            persona_block=persona_block,
            self_state_block=self_state_block,
            memory_blocks=memory_blocks,
            recent_turns_block=recent_turns_block,
            user_block=user_block,
            response_contract_block=response_contract_block,
            full_prompt=full_prompt,
            token_estimate=token_estimate,
        )

    def _format_persona(self, persona: PersonaState) -> str:
        lines = [
            "[Persona]",
            f"Name: {persona.name}",
            f"Core Description: {persona.core_description}",
            f"Tone: {persona.tone}",
        ]
        if persona.values:
            lines.append("Values:")
            lines.extend(f"- {value}" for value in persona.values)
        if persona.commitments:
            lines.append("Commitments:")
            lines.extend(f"- {item}" for item in persona.commitments)
        if persona.style_rules:
            lines.append("Style Rules:")
            lines.extend(f"- {item}" for item in persona.style_rules)
        if persona.identity_anchors:
            lines.append("Identity Anchors:")
            lines.extend(f"- {item}" for item in persona.identity_anchors)
        return "\n".join(lines)

    def _format_self_state(self, self_state: SelfState) -> str:
        lines = [
            "[Self-State]",
            f"Identity Summary: {self_state.identity_summary}",
            f"Current Focus: {self_state.current_focus}",
            f"Stability Version: {self_state.stability_version}",
        ]
        if self_state.active_questions:
            lines.append("Active Questions:")
            lines.extend(f"- {item}" for item in self_state.active_questions)
        if self_state.stable_preferences:
            lines.append("Stable Preferences:")
            lines.extend(f"- {item}" for item in self_state.stable_preferences)
        if self_state.relationship_notes:
            lines.append("Relationship Notes:")
            lines.extend(f"- {item}" for item in self_state.relationship_notes)
        if self_state.continuity_notes:
            lines.append("Continuity Notes:")
            lines.extend(f"- {item}" for item in self_state.continuity_notes)
        if self_state.open_tensions:
            lines.append("Open Tensions:")
            lines.extend(f"- {item}" for item in self_state.open_tensions)
        return "\n".join(lines)

    def _format_memory_blocks(self, memory_hits: list[RetrievalHit]) -> dict[str, str]:
        grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
        for hit in memory_hits:
            grouped[hit.channel].append(hit)

        blocks: dict[str, str] = {}
        for channel in sorted(grouped):
            lines = [f"[Memory:{channel}]"]
            for hit in grouped[channel]:
                snippet = hit.text.strip().replace("\n", " ")
                if len(snippet) > self.memory_char_limit:
                    snippet = snippet[: self.memory_char_limit] + "..."
                prefix = f"- ({hit.score:.3f}) "
                if hit.kind:
                    prefix += f"[{hit.kind}] "
                lines.append(prefix + snippet)
            blocks[channel] = "\n".join(lines)
        return blocks

    def _format_recent_turns(self, recent_turns: list[TurnRecord]) -> str:
        if not recent_turns:
            return ""
        lines = ["[Recent Conversation]"]
        for turn in recent_turns:
            user_text = turn.user_text.strip().replace("\n", " ")
            answer_text = turn.final_answer.strip().replace("\n", " ")
            if len(user_text) > self.recent_turn_char_limit:
                user_text = user_text[: self.recent_turn_char_limit] + "..."
            if len(answer_text) > self.recent_turn_char_limit:
                answer_text = answer_text[: self.recent_turn_char_limit] + "..."
            lines.append(f"User: {user_text}")
            lines.append(f"Nova: {answer_text}")
        return "\n".join(lines)

    def _format_contract_rules(self, contract_rules: list[str]) -> str:
        lines = ["[Response Rules]"]
        lines.extend(f"- {rule}" for rule in contract_rules)
        return "\n".join(lines)
