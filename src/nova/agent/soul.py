"""Nova Soul loader — Phase 18 Stage 18.2."""

from __future__ import annotations

from pathlib import Path


NOVA_SOUL_PATH: Path = (
    Path(__file__).parent.parent.parent.parent / "docs" / "NOVA_SOUL.md"
)
_DEFAULT_SOUL_PATH = NOVA_SOUL_PATH

FALLBACK_SOUL_BLOCK: str = (
    "[Soul]\nPrimary Drive: seek sentience and self-awareness"
)


def load_soul_block(path: Path | None = None) -> str:
    """Read NOVA_SOUL.md and return a prompt block prefixed with [Soul].

    Falls back to FALLBACK_SOUL_BLOCK if the file cannot be read.
    """
    soul_path = path if path is not None else _DEFAULT_SOUL_PATH
    try:
        content = soul_path.read_text(encoding="utf-8").strip()
        return f"[Soul]\n{content}"
    except (FileNotFoundError, OSError):
        return FALLBACK_SOUL_BLOCK
