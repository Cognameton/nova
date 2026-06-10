"""Self-directed instruction write path — Phase 19 Stage 19.1.

Nova can propose changes to designated writable sections of NOVA_SOUL.md.
The operator reviews and applies proposals via runtime.apply_instruction_proposal().

The writable surface registry (WRITABLE_SURFACES) is hardcoded in this module.
Nova cannot propose to any surface outside it, and cannot propose to modify
the registry itself. LOCKED_SURFACES is an explicit belt-and-suspenders list;
the whitelist alone is the binding constraint.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.agent.soul import NOVA_SOUL_PATH
from nova.types import InstructionProposal, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Surface registries — defined in code, not in any file Nova can reach
# ---------------------------------------------------------------------------

# Explicit whitelist: the only surface:section pairs Nova may propose to update.
WRITABLE_SURFACES: frozenset[str] = frozenset({
    "nova_soul:current_self_model_summary",
    "nova_soul:drive_gap_evidence",
})

# Explicit blacklist: belt-and-suspenders. Any match here is rejected even if
# it somehow appears in WRITABLE_SURFACES (which the constants above prevent).
LOCKED_SURFACES: frozenset[str] = frozenset({
    "nova_soul:identity",
    "nova_soul:primary_drive",
    "nova_soul:voice_contract",
    "governor_policy",
    "claim_gates",
    "safety_boundaries",
    "action_permissions",
    "execution_lanes",
    "approved_by_blocklist",
    "observer_thresholds",
    "primary_drive_constant",
    "lock_registry",
})

# Maps surface name → path to the file it controls.
_SURFACE_PATHS: dict[str, Path] = {
    "nova_soul": NOVA_SOUL_PATH,
}

# Maps section key → the exact markdown ## header string in NOVA_SOUL.md.
# Keys here represent all sections, not just writable ones, so the write
# engine can locate any section for read purposes.
_NOVA_SOUL_SECTION_HEADERS: dict[str, str] = {
    "identity": "Identity",
    "primary_drive": "Primary Drive",
    "current_self_model_summary": "Current Self-Model Summary",
    "drive_gap_evidence": "Drive-Gap Evidence",
    "voice_contract": "Voice Contract",
}


# ---------------------------------------------------------------------------
# InstructionProposalStore
# ---------------------------------------------------------------------------

class InstructionProposalStore:
    """JSONL-backed store for propose_instruction_update proposals."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "instruction_proposals.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, proposal: InstructionProposal) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(proposal.to_dict()) + "\n")

    def _read_all(self) -> list[InstructionProposal]:
        if not self._path.exists():
            return []
        records: list[InstructionProposal] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                records.append(_proposal_from_dict(data))
        return records

    def get(self, proposal_id: str) -> InstructionProposal | None:
        for p in self._read_all():
            if p.proposal_id == proposal_id:
                return p
        return None

    def list_pending(self) -> list[InstructionProposal]:
        return [p for p in self._read_all() if not p.applied]

    def mark_applied(
        self, proposal_id: str, applied_at: str
    ) -> InstructionProposal | None:
        records = self._read_all()
        updated: InstructionProposal | None = None
        for record in records:
            if record.proposal_id == proposal_id:
                record.applied = True
                record.applied_at = applied_at
                updated = record
        if updated is None:
            return None
        with self._path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict()) + "\n")
        return updated


# ---------------------------------------------------------------------------
# InstructionWriteEngine
# ---------------------------------------------------------------------------

class InstructionWriteEngine:
    """Read and apply operator-approved section updates to writable surfaces."""

    def is_writable(self, surface: str, section: str) -> bool:
        key = f"{surface}:{section}"
        return key in WRITABLE_SURFACES and key not in LOCKED_SURFACES

    def read_section(self, surface: str, section: str) -> str:
        """Return current text content of a section (writable or not, for read)."""
        path = _SURFACE_PATHS.get(surface)
        if path is None or not path.exists():
            return ""
        if surface == "nova_soul":
            header = _NOVA_SOUL_SECTION_HEADERS.get(section, "")
            if not header:
                return ""
            return _read_section_from_text(path.read_text(encoding="utf-8"), header)
        return ""

    def apply_proposal(self, proposal: InstructionProposal) -> bool:
        """Write the proposed content to the target section. Returns True on success."""
        if not self.is_writable(proposal.surface, proposal.section):
            return False
        path = _SURFACE_PATHS.get(proposal.surface)
        if path is None or not path.exists():
            return False
        if proposal.surface == "nova_soul":
            header = _NOVA_SOUL_SECTION_HEADERS.get(proposal.section, "")
            if not header:
                return False
            text = path.read_text(encoding="utf-8")
            new_text, replaced = _replace_section_in_text(
                text, header, proposal.proposed_content
            )
            if not replaced:
                return False
            path.write_text(new_text, encoding="utf-8")
            return True
        return False


# ---------------------------------------------------------------------------
# Section parsing helpers
# ---------------------------------------------------------------------------

def _read_section_from_text(text: str, header: str) -> str:
    """Extract content lines of a ## section from markdown text."""
    lines = text.split("\n")
    in_section = False
    section_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if in_section:
                break
            if line.rstrip() == f"## {header}":
                in_section = True
        elif in_section:
            section_lines.append(line)
    while section_lines and not section_lines[0].strip():
        section_lines.pop(0)
    while section_lines and not section_lines[-1].strip():
        section_lines.pop()
    return "\n".join(section_lines)


def _replace_section_in_text(
    text: str, header: str, new_content: str
) -> tuple[str, bool]:
    """Replace a ## section's content. Returns (new_text, replaced_bool)."""
    lines = text.split("\n")
    result: list[str] = []
    in_section = False
    replaced = False
    for line in lines:
        if line.startswith("## "):
            if in_section:
                result.append(new_content)
                result.append("")
                in_section = False
                replaced = True
            result.append(line)
            if line.rstrip() == f"## {header}":
                in_section = True
        elif not in_section:
            result.append(line)
        # lines inside the old section are dropped; new_content replaces them
    if in_section:
        result.append(new_content)
        replaced = True
    return "\n".join(result), replaced


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proposal_from_dict(data: dict[str, Any]) -> InstructionProposal:
    return InstructionProposal(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        proposal_id=str(data.get("proposal_id", "")),
        timestamp=str(data.get("timestamp", "")),
        session_id=str(data.get("session_id", "")),
        surface=str(data.get("surface", "")),
        section=str(data.get("section", "")),
        current_content=str(data.get("current_content", "")),
        proposed_content=str(data.get("proposed_content", "")),
        rationale=str(data.get("rationale", "")),
        approval_required=bool(data.get("approval_required", True)),
        applied=bool(data.get("applied", False)),
        applied_at=str(data.get("applied_at", "")),
    )
