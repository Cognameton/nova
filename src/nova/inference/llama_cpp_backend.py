"""llama.cpp backend implementation for Nova 2.0."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from nova.config import NovaConfig
from nova.types import GenerationRequest, GenerationResult


class LlamaCppBackend:
    """Phase 1 local inference backend using llama-cpp-python."""

    def __init__(self, config: NovaConfig):
        self.config = config
        self._llm: Llama | None = None

    @property
    def model_path(self) -> Path:
        return Path(self.config.model.model_path).expanduser()

    def load(self) -> None:
        if self._llm is not None:
            return

        model_path = self.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        tensor_split = self.config.model.tensor_split or None
        if tensor_split is not None and len(tensor_split) == 0:
            tensor_split = None

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.n_ctx,
            n_gpu_layers=self.config.model.n_gpu_layers,
            tensor_split=tensor_split,
            main_gpu=self.config.model.main_gpu,
            verbose=False,
        )

    def unload(self) -> None:
        self._llm = None

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "llama_cpp",
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "n_ctx": self.config.model.n_ctx,
            "n_gpu_layers": self.config.model.n_gpu_layers,
            "tensor_split": list(self.config.model.tensor_split),
            "main_gpu": self.config.model.main_gpu,
        }

    def tokenize(self, text: str) -> int:
        if self._llm is None:
            self.load()
        assert self._llm is not None
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=True))

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._llm is None:
            self.load()
        assert self._llm is not None

        started_at = time.perf_counter()
        response = self._llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        choices = response.get("choices", [])
        first_choice = choices[0] if choices else {}
        usage = response.get("usage", {})

        return GenerationResult(
            model_id=request.model_id,
            raw_text=str(first_choice.get("text", "") or "").strip(),
            finish_reason=first_choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            metadata={
                "backend": "llama_cpp",
                "response_id": response.get("id"),
                "object": response.get("object"),
            },
        )
