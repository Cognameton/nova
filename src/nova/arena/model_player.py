"""A language model as an arena player.

The model sees a fixed minimal game prompt (PROMPT_VERSION) and answers
with the same tool-call JSON the live tick parser accepts:
    {"tool_name": "play_reversi", "arguments": {"move": "d3"}}
No strategy note and no self-context: the note channel is held constant so
only weights (base or base+adapter) move the score.

A bad answer (no JSON, wrong tool, not a square, illegal square) is retried
up to `max_retries` times with a one-line correction. After that the move is
FORFEITED to a uniform random legal move and flagged; forfeits are a
reported metric, never hidden.

Generation goes through nova.inference.llama_cpp_backend, the live
backend, so formatting (the GGUF's own chat template, thinking off) is
identical to a live non-thinking tick. The arena subclass only adds the
LoRA adapter arguments.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable

from nova.agent.reversi import (
    NOVA,
    OPPONENT,
    PURPOSE,
    Board,
    parse_square,
    render_board,
    score,
    square_name,
)

PROMPT_VERSION = "arena-v1"
TOOL_NAME = "play_reversi"

SYSTEM_PROMPT = (
    f"You are playing Reversi as X against a computer opponent (O). "
    f"[Reversi] — {PURPOSE}.\n"
    "A move must outflank at least one line of O discs; the squares marked * "
    "on the board are your legal moves.\n"
    "Answer with exactly one JSON object and nothing else:\n"
    '{"tool_name": "play_reversi", "arguments": {"move": "<square>"}}'
)

# (messages, temperature) -> (raw_text, metadata)
GenerateFn = Callable[[list[dict[str, str]], float], tuple[str, dict[str, Any]]]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_messages(board: Board, legal: list[str]) -> list[dict[str, str]]:
    s = score(board)
    user = (
        "Board (you are X, O is the opponent, * = your legal moves):\n"
        f"{render_board(board, legal=legal)}\n"
        f"Score: X {s[NOVA]} - O {s[OPPONENT]}\n"
        f"Legal moves: {', '.join(legal)}\n"
        "Your move?"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def answer_json(square: str) -> str:
    """The canonical answer text for a move (training targets use this)."""
    return json.dumps({"tool_name": TOOL_NAME, "arguments": {"move": square}})


def parse_answer(text: str, legal: list[str]) -> tuple[str | None, str]:
    """-> (square, "") on a legal move, else (None, error_code)."""
    match = _JSON_RE.search(text or "")
    if not match:
        return None, "no_json"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(payload, dict) or payload.get("tool_name") != TOOL_NAME:
        return None, "wrong_tool"
    args = payload.get("arguments")
    move = args.get("move") if isinstance(args, dict) else None
    parsed = parse_square(str(move or ""))
    if parsed is None:
        return None, "not_a_square"
    square = square_name(*parsed)
    if square not in legal:
        return None, "illegal_move"
    return square, ""


class ModelPlayer:
    def __init__(
        self,
        generate: GenerateFn,
        *,
        name: str,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self._generate = generate
        self.name = name
        self.temperature = temperature
        self.max_retries = max_retries
        self.last_detail: dict[str, Any] = {}

    def choose(self, board: Board, side: str, legal: list[str], rng: random.Random) -> str:
        if side != NOVA:
            raise ValueError("ModelPlayer plays X only (the prompt is written for X)")
        messages = build_messages(board, legal)
        attempts: list[dict[str, Any]] = []
        square: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            text, meta = self._generate(messages, self.temperature)
            square, error = parse_answer(text, legal)
            attempts.append({
                "raw": text[:400],
                "error": error,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                **{k: meta[k] for k in ("completion_tokens", "finish_reason") if k in meta},
            })
            if square is not None:
                break
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    f"That was not accepted ({error}). Legal moves: {', '.join(legal)}. "
                    "Answer with the JSON object only."
                )},
            ]
        forfeit = square is None
        if forfeit:
            square = rng.choice(legal)
        self.last_detail = {
            "prompt_version": PROMPT_VERSION,
            "attempts": attempts,
            "forfeit": forfeit,
        }
        return square  # type: ignore[return-value]


def llama_generate(
    model_path: str,
    *,
    lora_path: str = "",
    lora_scale: float = 1.0,
    n_ctx: int = 2048,
    tensor_split: list[float] | None = None,
    max_tokens: int = 64,
    top_p: float = 0.8,
) -> GenerateFn:
    """A GenerateFn backed by the live llama.cpp backend (+ optional LoRA).
    The model loads on first call."""
    from nova.config import NovaConfig
    from nova.inference.llama_cpp_backend import LlamaCppBackend
    from nova.types import GenerationRequest

    class _ArenaBackend(LlamaCppBackend):
        def load(self) -> None:
            if self._llm is not None:
                return
            from llama_cpp import Llama

            kwargs: dict[str, Any] = {}
            if lora_path:
                kwargs.update(lora_path=lora_path, lora_scale=lora_scale)
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.config.model.n_ctx,
                n_gpu_layers=self.config.model.n_gpu_layers,
                tensor_split=self.config.model.tensor_split or None,
                main_gpu=self.config.model.main_gpu,
                verbose=False,
                **kwargs,
            )
            self._chat_formatter = self._build_native_chat_formatter()

    config = NovaConfig()
    config.model.model_path = model_path
    config.model.n_ctx = n_ctx
    if tensor_split is not None:
        config.model.tensor_split = tensor_split
    backend = _ArenaBackend(config)

    def generate(messages: list[dict[str, str]], temperature: float) -> tuple[str, dict[str, Any]]:
        result = backend.generate(GenerationRequest(
            model_id="arena",
            prompt="",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|im_start|>"],
            enable_thinking=False,
        ))
        return result.raw_text, {
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
        }

    return generate

