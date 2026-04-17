"""Inference backend contracts."""

from __future__ import annotations

from typing import Any, Protocol

from nova.types import GenerationRequest, GenerationResult


class InferenceBackend(Protocol):
    """Minimal Phase 1 inference backend contract."""

    def load(self) -> None:
        """Initialize backend resources."""

    def unload(self) -> None:
        """Release backend resources."""

    def metadata(self) -> dict[str, Any]:
        """Return backend and model metadata for tracing."""

    def tokenize(self, text: str) -> int:
        """Estimate prompt token count for the active backend."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a single generation request."""
