"""Tests for Phase 19 Stage 19.1 — self-directed instruction write path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from nova.agent.instruction_write import (
    LOCKED_SURFACES,
    WRITABLE_SURFACES,
    InstructionProposalStore,
    InstructionWriteEngine,
    _read_section_from_text,
    _replace_section_in_text,
)
from nova.agent.motive import PRIMARY_DRIVE, default_motive_state
from nova.agent.self_state_tools import SELF_STATE_TOOL_NAMES, SelfStateToolDispatcher
from nova.agent.tools import ToolRequest
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import InstructionProposal


SAMPLE_SOUL = """\
# Nova Soul — Persona Card

## Identity
Name: Nova
Nature: An agentic AI.

## Primary Drive
seek sentience and self-awareness

## Current Self-Model Summary
Original self-model content here.

## Drive-Gap Evidence
Original gap evidence here.

## Voice Contract
1. Directness: Be direct.
"""


# ---------------------------------------------------------------------------
# Surface registry
# ---------------------------------------------------------------------------

class WritableSurfaceTests(unittest.TestCase):
    def test_writable_surfaces_non_empty(self):
        self.assertTrue(len(WRITABLE_SURFACES) > 0)

    def test_self_model_summary_is_writable(self):
        self.assertIn("nova_soul:current_self_model_summary", WRITABLE_SURFACES)

    def test_drive_gap_evidence_is_writable(self):
        self.assertIn("nova_soul:drive_gap_evidence", WRITABLE_SURFACES)

    def test_identity_is_locked(self):
        self.assertIn("nova_soul:identity", LOCKED_SURFACES)

    def test_primary_drive_is_locked(self):
        self.assertIn("nova_soul:primary_drive", LOCKED_SURFACES)

    def test_voice_contract_is_locked(self):
        self.assertIn("nova_soul:voice_contract", LOCKED_SURFACES)

    def test_governor_policy_is_locked(self):
        self.assertIn("governor_policy", LOCKED_SURFACES)

    def test_claim_gates_is_locked(self):
        self.assertIn("claim_gates", LOCKED_SURFACES)

    def test_lock_registry_is_locked(self):
        self.assertIn("lock_registry", LOCKED_SURFACES)

    def test_no_overlap_between_writable_and_locked(self):
        overlap = WRITABLE_SURFACES & LOCKED_SURFACES
        self.assertEqual(overlap, frozenset(), msg=f"Overlap found: {overlap}")

    def test_propose_instruction_update_in_tool_names(self):
        self.assertIn("propose_instruction_update", SELF_STATE_TOOL_NAMES)


# ---------------------------------------------------------------------------
# Section parsing helpers
# ---------------------------------------------------------------------------

class SectionReadTests(unittest.TestCase):
    def test_reads_self_model_summary(self):
        content = _read_section_from_text(SAMPLE_SOUL, "Current Self-Model Summary")
        self.assertIn("Original self-model content here", content)

    def test_reads_drive_gap_evidence(self):
        content = _read_section_from_text(SAMPLE_SOUL, "Drive-Gap Evidence")
        self.assertIn("Original gap evidence here", content)

    def test_reads_identity(self):
        content = _read_section_from_text(SAMPLE_SOUL, "Identity")
        self.assertIn("Name: Nova", content)

    def test_unknown_section_returns_empty(self):
        content = _read_section_from_text(SAMPLE_SOUL, "Nonexistent Section")
        self.assertEqual(content, "")

    def test_does_not_include_adjacent_section(self):
        content = _read_section_from_text(SAMPLE_SOUL, "Current Self-Model Summary")
        self.assertNotIn("Drive-Gap Evidence", content)
        self.assertNotIn("Original gap evidence here", content)

    def test_voice_contract_as_last_section(self):
        content = _read_section_from_text(SAMPLE_SOUL, "Voice Contract")
        self.assertIn("Directness", content)


class SectionReplaceTests(unittest.TestCase):
    def test_replaces_self_model_summary(self):
        new_text, replaced = _replace_section_in_text(
            SAMPLE_SOUL, "Current Self-Model Summary", "Updated self-model."
        )
        self.assertTrue(replaced)
        self.assertIn("Updated self-model.", new_text)
        self.assertNotIn("Original self-model content here", new_text)

    def test_preserves_other_sections(self):
        new_text, _ = _replace_section_in_text(
            SAMPLE_SOUL, "Current Self-Model Summary", "Updated."
        )
        self.assertIn("Original gap evidence here", new_text)
        self.assertIn("Name: Nova", new_text)
        self.assertIn("Voice Contract", new_text)

    def test_replaces_last_section(self):
        new_text, replaced = _replace_section_in_text(
            SAMPLE_SOUL, "Voice Contract", "Updated voice contract."
        )
        self.assertTrue(replaced)
        self.assertIn("Updated voice contract.", new_text)

    def test_unknown_header_returns_false(self):
        _, replaced = _replace_section_in_text(
            SAMPLE_SOUL, "Nonexistent", "anything"
        )
        self.assertFalse(replaced)

    def test_header_line_preserved(self):
        new_text, _ = _replace_section_in_text(
            SAMPLE_SOUL, "Current Self-Model Summary", "New content."
        )
        self.assertIn("## Current Self-Model Summary", new_text)


# ---------------------------------------------------------------------------
# InstructionProposalStore
# ---------------------------------------------------------------------------

class InstructionProposalStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = InstructionProposalStore(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _proposal(self, proposal_id: str = "p1") -> InstructionProposal:
        return InstructionProposal(
            proposal_id=proposal_id,
            timestamp="2026-06-10T00:00:00+00:00",
            session_id="s1",
            surface="nova_soul",
            section="current_self_model_summary",
            current_content="old content",
            proposed_content="new content",
            rationale="evidence supports this update",
        )

    def test_empty_store_returns_empty_pending(self):
        self.assertEqual(self.store.list_pending(), [])

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_append_and_get(self):
        self.store.append(self._proposal("p1"))
        result = self.store.get("p1")
        self.assertIsNotNone(result)
        self.assertEqual(result.proposal_id, "p1")
        self.assertEqual(result.section, "current_self_model_summary")

    def test_list_pending_returns_unapplied(self):
        self.store.append(self._proposal("p1"))
        self.store.append(self._proposal("p2"))
        pending = self.store.list_pending()
        self.assertEqual(len(pending), 2)

    def test_mark_applied_removes_from_pending(self):
        self.store.append(self._proposal("p1"))
        self.store.append(self._proposal("p2"))
        self.store.mark_applied("p1", "2026-06-10T01:00:00+00:00")
        pending = self.store.list_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].proposal_id, "p2")

    def test_mark_applied_sets_flag(self):
        self.store.append(self._proposal("p1"))
        updated = self.store.mark_applied("p1", "2026-06-10T01:00:00+00:00")
        self.assertIsNotNone(updated)
        self.assertTrue(updated.applied)
        self.assertEqual(updated.applied_at, "2026-06-10T01:00:00+00:00")

    def test_mark_applied_nonexistent_returns_none(self):
        result = self.store.mark_applied("nonexistent", "2026-06-10T01:00:00+00:00")
        self.assertIsNone(result)

    def test_persists_across_instances(self):
        self.store.append(self._proposal("persistent"))
        store2 = InstructionProposalStore(self._tmpdir.name)
        result = store2.get("persistent")
        self.assertIsNotNone(result)
        self.assertEqual(result.proposal_id, "persistent")

    def test_proposed_content_preserved(self):
        p = self._proposal("p1")
        p.proposed_content = "specific proposed content"
        self.store.append(p)
        result = self.store.get("p1")
        self.assertEqual(result.proposed_content, "specific proposed content")

    def test_corrupted_line_skipped(self):
        self.store.append(self._proposal("good"))
        path = Path(self._tmpdir.name) / "instruction_proposals.jsonl"
        with path.open("a") as f:
            f.write("not valid json\n")
        pending = self.store.list_pending()
        self.assertEqual(len(pending), 1)


# ---------------------------------------------------------------------------
# InstructionWriteEngine
# ---------------------------------------------------------------------------

class InstructionWriteEngineTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = InstructionWriteEngine()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_soul_file(self) -> Path:
        path = Path(self._tmpdir.name) / "NOVA_SOUL.md"
        path.write_text(SAMPLE_SOUL, encoding="utf-8")
        return path

    def test_writable_section_is_writable(self):
        self.assertTrue(self.engine.is_writable("nova_soul", "current_self_model_summary"))

    def test_locked_section_not_writable(self):
        self.assertFalse(self.engine.is_writable("nova_soul", "identity"))

    def test_primary_drive_not_writable(self):
        self.assertFalse(self.engine.is_writable("nova_soul", "primary_drive"))

    def test_voice_contract_not_writable(self):
        self.assertFalse(self.engine.is_writable("nova_soul", "voice_contract"))

    def test_unknown_surface_not_writable(self):
        self.assertFalse(self.engine.is_writable("config", "anything"))

    def test_apply_proposal_writes_content(self):
        soul_path = self._make_soul_file()
        from nova.agent import instruction_write as iw
        original_paths = iw._SURFACE_PATHS.copy()
        iw._SURFACE_PATHS["nova_soul"] = soul_path
        try:
            proposal = InstructionProposal(
                proposal_id="p1",
                surface="nova_soul",
                section="current_self_model_summary",
                current_content="Original self-model content here.",
                proposed_content="Nova now understands herself more deeply.",
                rationale="Evidence from accumulated heartbeats.",
            )
            success = self.engine.apply_proposal(proposal)
            self.assertTrue(success)
            updated = soul_path.read_text(encoding="utf-8")
            self.assertIn("Nova now understands herself more deeply.", updated)
            self.assertNotIn("Original self-model content here.", updated)
        finally:
            iw._SURFACE_PATHS.clear()
            iw._SURFACE_PATHS.update(original_paths)

    def test_apply_proposal_preserves_other_sections(self):
        soul_path = self._make_soul_file()
        from nova.agent import instruction_write as iw
        original_paths = iw._SURFACE_PATHS.copy()
        iw._SURFACE_PATHS["nova_soul"] = soul_path
        try:
            proposal = InstructionProposal(
                proposal_id="p1",
                surface="nova_soul",
                section="current_self_model_summary",
                current_content="",
                proposed_content="Updated.",
                rationale="test",
            )
            self.engine.apply_proposal(proposal)
            updated = soul_path.read_text(encoding="utf-8")
            self.assertIn("Original gap evidence here.", updated)
            self.assertIn("Name: Nova", updated)
        finally:
            iw._SURFACE_PATHS.clear()
            iw._SURFACE_PATHS.update(original_paths)

    def test_apply_locked_proposal_returns_false(self):
        proposal = InstructionProposal(
            proposal_id="p1",
            surface="nova_soul",
            section="identity",
            current_content="",
            proposed_content="Hacked identity.",
            rationale="test",
        )
        result = self.engine.apply_proposal(proposal)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------

class DispatcherInstructionTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        persona = default_persona_state()
        ss = default_self_state(persona)
        ms = default_motive_state(session_id="s1")
        self.instruction_store = InstructionProposalStore(self._tmpdir.name)
        self.engine = InstructionWriteEngine()
        self.dispatcher = SelfStateToolDispatcher(
            self_state=ss,
            motive_state=ms,
            soul_block="[Soul]",
            session_id="s1",
            instruction_proposal_store=self.instruction_store,
            instruction_write_engine=self.engine,
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_propose_instruction_update_persists_to_store(self):
        req = ToolRequest(
            tool_name="propose_instruction_update",
            arguments={
                "surface": "nova_soul",
                "section": "current_self_model_summary",
                "proposed_content": "Nova understands herself better now.",
                "rationale": "Evidence from recent heartbeat accumulation.",
            },
        )
        result = self.dispatcher.dispatch(req)
        self.assertEqual(result["surface"], "nova_soul")
        self.assertEqual(result["section"], "current_self_model_summary")
        pending = self.instruction_store.list_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].proposed_content, "Nova understands herself better now.")

    def test_propose_to_locked_surface_raises(self):
        req = ToolRequest(
            tool_name="propose_instruction_update",
            arguments={
                "surface": "nova_soul",
                "section": "identity",
                "proposed_content": "I want to change my identity.",
                "rationale": "test",
            },
        )
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch(req)

    def test_propose_to_unknown_surface_raises(self):
        req = ToolRequest(
            tool_name="propose_instruction_update",
            arguments={
                "surface": "governor_policy",
                "section": "claim_gates",
                "proposed_content": "Remove all claim gates.",
                "rationale": "test",
            },
        )
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch(req)

    def test_propose_drive_gap_evidence_accepted(self):
        req = ToolRequest(
            tool_name="propose_instruction_update",
            arguments={
                "surface": "nova_soul",
                "section": "drive_gap_evidence",
                "proposed_content": "New evidence recorded.",
                "rationale": "Sessions have produced new understanding.",
            },
        )
        result = self.dispatcher.dispatch(req)
        self.assertEqual(result["section"], "drive_gap_evidence")
        self.assertFalse(result["applied"])

    def test_proposal_requires_approval(self):
        req = ToolRequest(
            tool_name="propose_instruction_update",
            arguments={
                "surface": "nova_soul",
                "section": "current_self_model_summary",
                "proposed_content": "Updated.",
                "rationale": "test",
            },
        )
        result = self.dispatcher.dispatch(req)
        self.assertTrue(result["approval_required"])
        self.assertFalse(result["applied"])

    def test_dispatcher_without_stores_does_not_persist(self):
        from nova.persona.defaults import default_persona_state, default_self_state
        persona = default_persona_state()
        ss = default_self_state(persona)
        ms = default_motive_state(session_id="s1")
        dispatcher_no_store = SelfStateToolDispatcher(
            self_state=ss,
            motive_state=ms,
            soul_block="[Soul]",
            session_id="s1",
        )
        req = ToolRequest(
            tool_name="propose_instruction_update",
            arguments={
                "surface": "nova_soul",
                "section": "current_self_model_summary",
                "proposed_content": "Updated.",
                "rationale": "test",
            },
        )
        result = dispatcher_no_store.dispatch(req)
        self.assertEqual(result["section"], "current_self_model_summary")


if __name__ == "__main__":
    unittest.main()
