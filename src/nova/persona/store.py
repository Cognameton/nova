"""Persona persistence for Nova 2.0."""

from __future__ import annotations

import json
from pathlib import Path

from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import PersonaState, SelfState


class JsonPersonaStore:
    """JSON-backed store for Nova's stable persona state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> PersonaState:
        if not self.path.exists():
            persona = default_persona_state()
            self.save(persona)
            return persona

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return PersonaState(**payload)

    def save(self, persona: PersonaState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(persona.to_dict(), handle, indent=2, ensure_ascii=False)


class JsonSelfStateStore:
    """JSON-backed store for Nova's mutable self-state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, *, persona: PersonaState | None = None) -> SelfState:
        if not self.path.exists():
            self_state = default_self_state(persona)
            self.save(self_state)
            return self_state

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return SelfState(**payload)

    def save(self, self_state: SelfState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self_state.to_dict(), handle, indent=2, ensure_ascii=False)
