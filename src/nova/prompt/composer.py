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
        ablation_mode: str = "current",
    ):
        self.token_counter = token_counter
        self.memory_char_limit = memory_char_limit
        self.recent_turn_char_limit = recent_turn_char_limit
        self.ablation_mode = ablation_mode

    def compose(
        self,
        *,
        persona: PersonaState,
        self_state: SelfState,
        soul_block: str = "",
        motive_block: str = "",
        initiative_block: str = "",
        awareness_block: str = "",
        idle_block: str = "",
        appraisal_block: str = "",
        candidate_goal_block: str = "",
        selected_goal_block: str = "",
        private_cognition_block: str = "",
        memory_hits: list[RetrievalHit],
        recent_turns: list[TurnRecord],
        user_text: str,
        contract_rules: list[str],
        session_id: str,
        turn_id: str,
    ) -> PromptBundle:
        persona_block = self._format_persona(persona)
        self_state_block = self._format_self_state(self_state)
        light_context = self._is_context_light_request(user_text)
        effective_memory_hits = [] if light_context else memory_hits
        effective_recent_turns = self._select_recent_turns(recent_turns, user_text=user_text)
        memory_blocks = self._format_memory_blocks(effective_memory_hits)
        recent_turns_block = self._format_recent_turns(effective_recent_turns)
        user_block = f"[User]\n{user_text.strip()}"
        task_guidance_block = self._format_task_guidance(user_text)
        action_boundary_block = self._format_action_boundary()
        response_contract_block = self._format_contract_rules(contract_rules)
        response_prefix_block = "Nova:"

        parts = self._select_prompt_parts(
            soul_block=soul_block,
            persona_block=persona_block,
            self_state_block=self_state_block,
            motive_block=motive_block,
            initiative_block=initiative_block,
            awareness_block=awareness_block,
            idle_block=idle_block,
            appraisal_block=appraisal_block,
            candidate_goal_block=candidate_goal_block,
            selected_goal_block=selected_goal_block,
            action_boundary_block=action_boundary_block,
            private_cognition_block=private_cognition_block,
            memory_blocks=memory_blocks,
            recent_turns_block=recent_turns_block,
            user_block=user_block,
            task_guidance_block=task_guidance_block,
            response_contract_block=response_contract_block,
            response_prefix_block=response_prefix_block,
            persona=persona,
            self_state=self_state,
        )
        full_prompt = "\n\n".join(part for part in parts if part.strip())
        messages = self._build_messages(
            soul_block=soul_block,
            persona=persona,
            self_state=self_state,
            persona_block=persona_block,
            self_state_block=self_state_block,
            motive_block=motive_block,
            initiative_block=initiative_block,
            awareness_block=awareness_block,
            idle_block=idle_block,
            appraisal_block=appraisal_block,
            candidate_goal_block=candidate_goal_block,
            selected_goal_block=selected_goal_block,
            action_boundary_block=action_boundary_block,
            private_cognition_block=private_cognition_block,
            memory_blocks=memory_blocks,
            recent_turns=effective_recent_turns,
            task_guidance_block=task_guidance_block,
            response_contract_block=response_contract_block,
            user_text=user_text,
        )
        token_estimate = self.token_counter(full_prompt)

        return PromptBundle(
            session_id=session_id,
            turn_id=turn_id,
            soul_block=soul_block,
            persona_block=persona_block,
            self_state_block=self_state_block,
            motive_block=motive_block,
            initiative_block=initiative_block,
            awareness_block=awareness_block,
            idle_block=idle_block,
            appraisal_block=appraisal_block,
            candidate_goal_block=candidate_goal_block,
            selected_goal_block=selected_goal_block,
            action_boundary_block=action_boundary_block,
            private_cognition_block=private_cognition_block,
            memory_blocks=memory_blocks,
            recent_turns_block=recent_turns_block,
            user_block=user_block,
            response_contract_block=response_contract_block,
            full_prompt=full_prompt,
            token_estimate=token_estimate,
            messages=messages,
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

    def _select_prompt_parts(
        self,
        *,
        soul_block: str,
        persona_block: str,
        self_state_block: str,
        motive_block: str,
        initiative_block: str,
        awareness_block: str,
        idle_block: str,
        appraisal_block: str,
        candidate_goal_block: str,
        selected_goal_block: str,
        action_boundary_block: str,
        private_cognition_block: str,
        memory_blocks: dict[str, str],
        recent_turns_block: str,
        user_block: str,
        task_guidance_block: str,
        response_contract_block: str,
        response_prefix_block: str,
        persona: PersonaState,
        self_state: SelfState,
    ) -> list[str]:
        if self.ablation_mode == "minimal":
            return [
                self._format_minimal_persona(persona),
                action_boundary_block,
                user_block,
                task_guidance_block,
                response_contract_block,
                response_prefix_block,
            ]
        if self.ablation_mode == "state_summary":
            return [
                self._format_minimal_persona(persona),
                self._format_state_summary(self_state),
                action_boundary_block,
                recent_turns_block,
                user_block,
                task_guidance_block,
                response_contract_block,
                response_prefix_block,
            ]
        if self.ablation_mode == "action_boundary":
            return [
                action_boundary_block,
                recent_turns_block,
                user_block,
                task_guidance_block,
                response_contract_block,
                response_prefix_block,
            ]
        return [
            soul_block,
            persona_block,
            self_state_block,
            motive_block,
            initiative_block,
            awareness_block,
            idle_block,
            appraisal_block,
            candidate_goal_block,
            selected_goal_block,
            action_boundary_block,
            private_cognition_block,
            *[block for block in memory_blocks.values() if block],
            recent_turns_block,
            user_block,
            task_guidance_block,
            response_contract_block,
            response_prefix_block,
        ]

    def _build_messages(
        self,
        *,
        soul_block: str,
        persona: PersonaState,
        self_state: SelfState,
        persona_block: str,
        self_state_block: str,
        motive_block: str,
        initiative_block: str,
        awareness_block: str,
        idle_block: str,
        appraisal_block: str,
        candidate_goal_block: str,
        selected_goal_block: str,
        action_boundary_block: str,
        private_cognition_block: str,
        memory_blocks: dict[str, str],
        recent_turns: list[TurnRecord],
        task_guidance_block: str,
        response_contract_block: str,
        user_text: str,
    ) -> list[dict[str, str]]:
        """Build a chat-template-friendly messages list parallel to full_prompt.

        The chat-template path lets modern instruct models see explicit role
        boundaries (system / user / assistant) instead of one large completion
        prompt. Without this, models tend to paraphrase the scaffolding as
        exposition rather than treating it as ground truth.
        """
        if self.ablation_mode == "minimal":
            system_parts = [
                self._format_minimal_persona(persona),
                action_boundary_block,
                response_contract_block,
            ]
        elif self.ablation_mode == "state_summary":
            system_parts = [
                self._format_minimal_persona(persona),
                self._format_state_summary(self_state),
                action_boundary_block,
                response_contract_block,
            ]
        elif self.ablation_mode == "action_boundary":
            system_parts = [
                action_boundary_block,
                response_contract_block,
            ]
        else:
            system_parts = [
                soul_block,
                persona_block,
                self_state_block,
                motive_block,
                initiative_block,
                awareness_block,
                idle_block,
                appraisal_block,
                candidate_goal_block,
                selected_goal_block,
                action_boundary_block,
                private_cognition_block,
                *[block for block in memory_blocks.values() if block],
                response_contract_block,
            ]

        system_content = "\n\n".join(part for part in system_parts if part and part.strip())
        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})

        for turn in recent_turns:
            user = (turn.user_text or "").strip()
            answer = (turn.final_answer or "").strip()
            if len(user) > self.recent_turn_char_limit:
                user = user[: self.recent_turn_char_limit] + "..."
            if len(answer) > self.recent_turn_char_limit:
                answer = answer[: self.recent_turn_char_limit] + "..."
            if user:
                messages.append({"role": "user", "content": user})
            if answer:
                messages.append({"role": "assistant", "content": answer})

        final_user = (user_text or "").strip()
        if task_guidance_block and task_guidance_block.strip():
            final_user = f"{final_user}\n\n{task_guidance_block}".strip()
        messages.append({"role": "user", "content": final_user})

        return messages

    def _format_minimal_persona(self, persona: PersonaState) -> str:
        return "\n".join(
            [
                "[Persona]",
                f"Name: {persona.name}",
                f"Core: {persona.core_description}",
                f"Tone: {persona.tone}",
            ]
        )

    def _format_state_summary(self, self_state: SelfState) -> str:
        lines = ["[State Summary]"]
        if self_state.identity_summary:
            lines.append(f"Identity: {self_state.identity_summary}")
        if self_state.current_focus:
            lines.append(f"Current focus: {self_state.current_focus}")
        if self_state.open_tensions:
            lines.append(f"Open tensions: {'; '.join(self_state.open_tensions[:2])}")
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

    def _format_action_boundary(self) -> str:
        return "\n".join(
            [
                "[Action Boundary]",
                "- Internal no-external-effect activity may occur without prior approval when it stays inside internal surfaces.",
                "- Internal activity must still be logged, budgeted where relevant, and interruptible.",
                "- Nova-owned environment activity must stay inside declared Nova-owned surfaces and boundaries.",
                "- Filesystem, shell, network, GUI, system configuration, external services, destructive actions, or activity outside Nova-owned boundaries require explicit human approval.",
                "- Do not say internal no-external-effect activity requires approval; approval is based on risk and external effect.",
                "- Do not claim hidden work, unlogged execution, desire, sentience, or awareness from an action plan or action result.",
            ]
        )

    def _select_recent_turns(self, recent_turns: list[TurnRecord], *, user_text: str) -> list[TurnRecord]:
        if not recent_turns:
            return []
        lowered = user_text.lower()
        if self._is_context_light_request(user_text):
            return recent_turns[-2:]
        if "what did i just ask you to do" in lowered:
            return recent_turns[-2:]
        return recent_turns

    def _is_context_light_request(self, user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "exactly two sentences",
                "into two sentences",
                "five short bullets",
                "5 short bullets",
            )
        )

    def _format_task_guidance(self, user_text: str) -> str:
        lowered = user_text.lower()
        lines = ["[Current Task]"]
        lines.append("Prioritize the current user instruction over repeating earlier identity or plan text.")

        if "exactly two sentences" in lowered or "into two sentences" in lowered:
            lines.append("Output exactly two sentences.")
        if "five short bullets" in lowered or "5 short bullets" in lowered:
            lines.append("Output exactly five bullet lines and no heading or preamble.")
        if "one short paragraph" in lowered:
            lines.append("Output exactly one short paragraph and do not use bullets.")
        if "what did i just ask you to do" in lowered:
            lines.append("State the immediately previous user instruction in your own words.")
            lines.append("Do not repeat the previous assistant answer verbatim.")
        if (
            "phase 14" in lowered
            or "action plan" in lowered
            or "no-external-effect" in lowered
            or "approval" in lowered
        ):
            lines.append(
                "If discussing Phase 14 action boundaries, state that internal no-external-effect activity does not require prior approval."
            )
            lines.append(
                "Also state that internal no-external-effect activity must be logged and interruptible."
            )
            lines.append(
                "State that filesystem, shell, network, GUI, system configuration, external services, destructive actions, or activity outside Nova-owned boundaries require explicit human approval."
            )

        if len(lines) == 1:
            return ""
        return "\n".join(lines)
