"""llama.cpp backend implementation for Nova 2.0."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from nova.config import NovaConfig
from nova.types import GenerationRequest, GenerationResult


class LlamaCppBackend:
    """Phase 1 local inference backend using llama-cpp-python."""

    def __init__(self, config: NovaConfig):
        self.config = config
        self._llm: Any | None = None
        self._chat_formatter: Any | None = None

    @property
    def model_path(self) -> Path:
        return Path(self.config.model.model_path).expanduser()

    def load(self) -> None:
        if self._llm is not None:
            return

        from llama_cpp import Llama

        model_path = self.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        tensor_split = self.config.model.tensor_split or None
        if tensor_split is not None and len(tensor_split) == 0:
            tensor_split = None

        chat_format = (self.config.model.chat_format or "").strip() or None

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.n_ctx,
            n_gpu_layers=self.config.model.n_gpu_layers,
            tensor_split=tensor_split,
            main_gpu=self.config.model.main_gpu,
            chat_format=chat_format,
            verbose=False,
        )
        self._chat_formatter = self._build_native_chat_formatter()

    def _build_native_chat_formatter(self) -> Any | None:
        """Build a formatter from the GGUF's own embedded chat template,
        when present, so template-level controls (e.g. Qwen 3's
        enable_thinking) are available instead of only the generic
        chat_format string formatter. Returns None when no embedded
        template exists or extraction fails for any reason; callers
        fall back to create_chat_completion in that case. Metadata
        shape is an external file/library boundary, not an internal
        invariant, so this degrades gracefully rather than raising.
        """
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        assert self._llm is not None
        template = self._llm.metadata.get("tokenizer.chat_template")
        if not template:
            return None
        try:
            eos_token = self._llm.detokenize(
                [self._llm.token_eos()], special=True
            ).decode("utf-8", "ignore")
            bos_token = self._llm.detokenize(
                [self._llm.token_bos()], special=True
            ).decode("utf-8", "ignore")
            return Jinja2ChatFormatter(
                template=template, eos_token=eos_token, bos_token=bos_token
            )
        except Exception:
            return None

    def unload(self) -> None:
        self._llm = None
        self._chat_formatter = None

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "llama_cpp",
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "n_ctx": self.config.model.n_ctx,
            "n_gpu_layers": self.config.model.n_gpu_layers,
            "tensor_split": list(self.config.model.tensor_split),
            "main_gpu": self.config.model.main_gpu,
            "chat_format": self.config.model.chat_format or "",
            "native_chat_template": self._chat_formatter is not None,
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
        if request.messages and self._chat_formatter is not None:
            formatted = self._chat_formatter(
                messages=request.messages, enable_thinking=request.enable_thinking
            )
            stop = list(
                dict.fromkeys(list(formatted.stop or []) + list(request.stop))
            )
            response = self._llm(
                formatted.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
            )
            mode = "native_template_completion"
        elif request.messages:
            response = self._llm.create_chat_completion(
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            mode = "chat_completion"
        else:
            response = self._llm(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            mode = "completion"
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        choices = response.get("choices", [])
        first_choice = choices[0] if choices else {}
        usage = response.get("usage", {})

        if mode == "chat_completion":
            message = first_choice.get("message", {}) or {}
            raw_text = str(message.get("content", "") or "").strip()
        else:
            raw_text = str(first_choice.get("text", "") or "").strip()

        return GenerationResult(
            model_id=request.model_id,
            raw_text=raw_text,
            finish_reason=first_choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            metadata={
                "backend": "llama_cpp",
                "mode": mode,
                "response_id": response.get("id"),
                "object": response.get("object"),
            },
        )
