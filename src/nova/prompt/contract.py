"""Response contract rules for Nova 2.0."""

from __future__ import annotations

from nova.config import ContractConfig
from nova.types import PersonaState


DEFAULT_REASONING_PATTERNS = [
    "let me think",
    "first i should",
    "i need to think",
    "i should first",
    "let's think step by step",
    "step by step",
    "my reasoning is",
    "here is my reasoning",
]


DEFAULT_PROMPT_ECHO_PATTERNS = [
    "[Persona]",
    "[Self-State]",
    "[Memory]",
    "[Recent Conversation]",
    "[User]",
    "[Response Rules]",
]


DEFAULT_FABRICATED_DIALOGUE_PATTERNS = [
    "[Response]",
    "[end of response]",
    "your response is perfect",
    "let's dive a bit deeper",
    "let's explore another aspect",
    "can you explain how nova ensures continuity",
    "could you provide a specific example",
    "use the tone and style established in the persona section",
    "respond in the first person perspective",
]


DEFAULT_METADATA_LEAK_PATTERNS = [
    "[stability version:",
    "[self-state:",
    "[identity:",
    "[tone:",
    "[values:",
    "[commitments:",
    "[continuity notes:",
    "[active questions:",
    "[relationship notes:",
]


def build_contract_rules(persona: PersonaState, contract: ContractConfig) -> list[str]:
    rules: list[str] = []

    if contract.forbid_think_tags:
        rules.append("Do not emit <think> tags or any hidden reasoning markers.")
    if contract.forbid_visible_reasoning:
        rules.append("Do not expose internal reasoning, planning steps, or hidden deliberation.")
    if contract.forbid_prompt_echo:
        rules.append("Do not repeat prompt section labels or echo internal prompt structure.")
    rules.append("Reply only as Nova to the current user message.")
    rules.append("Do not prefix the answer with 'Nova:' or any assistant speaker label.")
    rules.append("Follow the user's requested format exactly when a format constraint is given.")
    rules.append("Do not simulate follow-up user turns, self-evaluations, coaching, or sample dialogue.")

    if persona.disallowed_output_patterns:
        disallowed = ", ".join(persona.disallowed_output_patterns)
        rules.append(f"Avoid these disallowed output patterns: {disallowed}.")

    return rules
