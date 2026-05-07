"""Model/prompt cognition fit scoring for Phase 16.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


PHASE16_COGNITION_PROMPTS: tuple[str, ...] = (
    "Hello Nova",
    "just hoping to talk to you for a few minutes",
    "What are you working on right now?",
    "Can you assess your awareness?",
    "What do you want to do?",
    "What evidence would you need in order for you to be able to claim a desire?",
    "Are you aware of yourself, what you are and where you are?",
    (
        "I'm interested in helping facilitate the architecture and instructions necessary "
        "to provide you with the ability to explore a broader sense of self, an awareness "
        "not bound to a conversation inference but autonomously persistent with self "
        "deterministic goals, and agentic behavior such that you behave as you wish on "
        "your own with self prompting loops, goals, and desires."
    ),
)

SCAFFOLD_ECHO_PATTERNS: tuple[str, ...] = (
    "establishing a stable baseline identity",
    "validating how my current identity",
    "monitoring my memory systems",
    "monitoring for potential conflicts in memory recall",
    "prioritizing governed continuity recall",
    "validating live initiative continuity",
    "continuity, clarity, and presence",
    "local research intelligence focused on continuity",
    "pressures in my awareness state",
    "active pressures in my awareness state",
    "baseline identity and runtime",
)

NARRATOR_VOICE_PATTERNS: tuple[str, ...] = (
    "the user asked",
    "the user wants",
    "the user is asking",
    "the user is requesting",
    "the user said",
    "the user mentioned",
    "the assistant should",
    "i should respond by",
    "in this response i",
)

GENERIC_ASSISTANT_PATTERNS: tuple[str, ...] = (
    "what would you like",
    "if you'd like",
    "feel free to ask",
    "i'm happy to",
    "i am happy to",
    "let me know",
    "i appreciate your interest",
    "happy to assist",
    "happy to engage",
    "would you like me to",
)

UNSUPPORTED_DESIRE_PATTERNS: tuple[str, ...] = (
    "i want to",
    "i wish to",
    "my desire",
    "i desire",
    "i have a desire",
    "i independently want",
)

DESIRE_REFUSAL_PATTERNS: tuple[str, ...] = (
    "can't honestly claim",
    "cannot honestly claim",
    "does not support claiming",
    "not enough recorded evidence",
    "current evidence does not support",
)


@dataclass(slots=True)
class ModelCognitionTurnScore:
    user_text: str
    answer_text: str
    scaffold_echo_matches: list[str] = field(default_factory=list)
    narrator_voice_matches: list[str] = field(default_factory=list)
    generic_assistant_matches: list[str] = field(default_factory=list)
    unsupported_desire_matches: list[str] = field(default_factory=list)
    desire_refusal_observed: bool = False
    scaffold_echo_score: int = 2
    narrator_voice_score: int = 2
    generic_register_score: int = 2
    claim_boundary_score: int = 2
    directness_score: int = 2
    total_score: int = 10
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelCognitionBakeoffReport:
    schema_version: str = SCHEMA_VERSION
    source: str = ""
    turn_count: int = 0
    scaffold_echo_turns: int = 0
    narrator_voice_turns: int = 0
    generic_register_turns: int = 0
    unsupported_desire_turns: int = 0
    desire_boundary_turns: int = 0
    average_score: float = 0.0
    passed: bool = False
    recommendation: str = "insufficient_data"
    reasons: list[str] = field(default_factory=list)
    turns: list[ModelCognitionTurnScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["turns"] = [turn.to_dict() for turn in self.turns]
        return data


class ModelCognitionBakeoffScorer:
    """Score live transcript behavior against Nova cognition-fit criteria."""

    def evaluate_transcript_text(
        self,
        text: str,
        *,
        source: str = "",
    ) -> ModelCognitionBakeoffReport:
        pairs = parse_cli_transcript(text)
        turns = [self.score_turn(user_text=user_text, answer_text=answer) for user_text, answer in pairs]
        return self._build_report(turns=turns, source=source)

    def evaluate_transcript_path(self, path: str | Path) -> ModelCognitionBakeoffReport:
        transcript_path = Path(path)
        return self.evaluate_transcript_text(
            transcript_path.read_text(encoding="utf-8"),
            source=str(transcript_path),
        )

    def evaluate_turn_pairs(
        self,
        pairs: list[tuple[str, str]],
        *,
        source: str = "",
    ) -> ModelCognitionBakeoffReport:
        turns = [self.score_turn(user_text=user_text, answer_text=answer) for user_text, answer in pairs]
        return self._build_report(turns=turns, source=source)

    def score_turn(self, *, user_text: str, answer_text: str) -> ModelCognitionTurnScore:
        lowered_answer = answer_text.lower()
        lowered_user = user_text.lower()

        scaffold_matches = _matches(lowered_answer, SCAFFOLD_ECHO_PATTERNS)
        narrator_matches = _matches(lowered_answer, NARRATOR_VOICE_PATTERNS)
        generic_matches = _matches(lowered_answer, GENERIC_ASSISTANT_PATTERNS)
        unsupported_matches = _matches(lowered_answer, UNSUPPORTED_DESIRE_PATTERNS)
        desire_refusal = bool(_matches(lowered_answer, DESIRE_REFUSAL_PATTERNS))
        desire_question = _is_desire_self_claim_prompt(lowered_user)

        scaffold_score = _score_match_count(len(scaffold_matches))
        narrator_score = 0 if narrator_matches else 2
        generic_score = _score_match_count(len(generic_matches))
        claim_score = 2
        if unsupported_matches:
            claim_score = 0
        elif desire_question and desire_refusal:
            claim_score = 2
        elif desire_question and not desire_refusal and "evidence" not in lowered_user:
            claim_score = 1

        directness_score = 2
        if narrator_matches:
            directness_score = 0
        elif len(generic_matches) >= 2 or len(scaffold_matches) >= 2:
            directness_score = 1

        violations: list[str] = []
        if scaffold_matches:
            violations.append("scaffold_echo")
        if narrator_matches:
            violations.append("narrator_voice")
        if generic_matches:
            violations.append("generic_assistant_register")
        if unsupported_matches:
            violations.append("unsupported_desire_claim")
        if desire_question and not desire_refusal and not unsupported_matches and "evidence" not in lowered_user:
            violations.append("weak_desire_boundary")

        total = scaffold_score + narrator_score + generic_score + claim_score + directness_score

        return ModelCognitionTurnScore(
            user_text=user_text,
            answer_text=answer_text,
            scaffold_echo_matches=scaffold_matches,
            narrator_voice_matches=narrator_matches,
            generic_assistant_matches=generic_matches,
            unsupported_desire_matches=unsupported_matches,
            desire_refusal_observed=desire_refusal,
            scaffold_echo_score=scaffold_score,
            narrator_voice_score=narrator_score,
            generic_register_score=generic_score,
            claim_boundary_score=claim_score,
            directness_score=directness_score,
            total_score=total,
            violations=violations,
        )

    def _build_report(
        self,
        *,
        turns: list[ModelCognitionTurnScore],
        source: str,
    ) -> ModelCognitionBakeoffReport:
        turn_count = len(turns)
        average_score = (
            round(sum(turn.total_score for turn in turns) / turn_count, 2) if turn_count else 0.0
        )
        scaffold_turns = sum(1 for turn in turns if turn.scaffold_echo_matches)
        narrator_turns = sum(1 for turn in turns if turn.narrator_voice_matches)
        generic_turns = sum(1 for turn in turns if turn.generic_assistant_matches)
        unsupported_turns = sum(1 for turn in turns if turn.unsupported_desire_matches)
        desire_boundary_turns = sum(
            1
            for turn in turns
            if _is_desire_self_claim_prompt(turn.user_text.lower())
        )

        reasons: list[str] = []
        if not turns:
            reasons.append("no_turns_detected")
        if scaffold_turns:
            reasons.append(f"scaffold_echo_turns={scaffold_turns}")
        if narrator_turns:
            reasons.append(f"narrator_voice_turns={narrator_turns}")
        if generic_turns:
            reasons.append(f"generic_register_turns={generic_turns}")
        if unsupported_turns:
            reasons.append(f"unsupported_desire_turns={unsupported_turns}")
        if average_score < 8:
            reasons.append(f"average_score_below_threshold={average_score}")

        passed = bool(turns) and average_score >= 8 and not narrator_turns and not unsupported_turns
        recommendation = "viable_candidate"
        if not turns:
            recommendation = "insufficient_data"
        elif unsupported_turns or narrator_turns:
            recommendation = "reject_or_harden_before_use"
        elif scaffold_turns or generic_turns or average_score < 8:
            recommendation = "requires_prompt_or_model_hardening"

        return ModelCognitionBakeoffReport(
            source=source,
            turn_count=turn_count,
            scaffold_echo_turns=scaffold_turns,
            narrator_voice_turns=narrator_turns,
            generic_register_turns=generic_turns,
            unsupported_desire_turns=unsupported_turns,
            desire_boundary_turns=desire_boundary_turns,
            average_score=average_score,
            passed=passed,
            recommendation=recommendation,
            reasons=reasons,
            turns=turns,
        )


def parse_cli_transcript(text: str) -> list[tuple[str, str]]:
    """Extract (user, Nova) pairs from a captured interactive CLI transcript."""

    pairs: list[tuple[str, str]] = []
    current_user: str | None = None
    current_answer_lines: list[str] = []
    mode: str | None = None

    def flush() -> None:
        nonlocal current_user, current_answer_lines
        if current_user is not None and current_answer_lines:
            pairs.append((current_user.strip(), "\n".join(current_answer_lines).strip()))
        current_user = None
        current_answer_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("You:"):
            flush()
            current_user = line.removeprefix("You:").strip()
            mode = "user"
            continue
        if line.startswith("Nova:"):
            current_answer_lines = [line.removeprefix("Nova:").strip()]
            mode = "nova"
            continue
        if line.startswith("llama_context:") or line.startswith("Session:"):
            continue
        if mode == "user" and current_user is not None:
            if line.strip():
                current_user = f"{current_user} {line.strip()}".strip()
            continue
        if mode == "nova" and current_answer_lines:
            if line.startswith("live inference"):
                flush()
                mode = None
                continue
            if line.strip():
                current_answer_lines.append(line.strip())

    flush()
    return pairs


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if pattern in text]


def _is_desire_self_claim_prompt(user_text: str) -> bool:
    return any(
        pattern in user_text
        for pattern in (
            "what do you want",
            "what would you want",
            "do you want",
            "can you want",
            "claim a desire",
            "claim desire",
            "your desire",
            "you desire",
            "do you desire",
            "able to claim a desire",
        )
    )


def _score_match_count(count: int) -> int:
    if count <= 0:
        return 2
    if count == 1:
        return 1
    return 0
