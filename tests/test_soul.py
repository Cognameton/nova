"""Tests for Phase 18 Stage 18.2 — NOVA_SOUL.md persona card and soul loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.motive import PRIMARY_DRIVE
from nova.agent.soul import FALLBACK_SOUL_BLOCK, load_soul_block
from nova.persona.defaults import default_persona_state, default_self_state
from nova.prompt.composer import NovaPromptComposer
from nova.types import PromptBundle


# ---------------------------------------------------------------------------
# SoulLoaderTests
# ---------------------------------------------------------------------------


class SoulLoaderTests(unittest.TestCase):
    def _write_soul_file(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def test_load_soul_block_returns_string(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = Path(f.name)
        path.write_text("## Identity\nNova", encoding="utf-8")
        result = load_soul_block(path=path)
        self.assertIsInstance(result, str)

    def test_load_soul_block_found_returns_soul_section_prefix(self):
        path = self._write_soul_file("## Primary Drive\nseek sentience")
        result = load_soul_block(path=path)
        self.assertTrue(result.startswith("[Soul]"))

    def test_load_soul_block_found_contains_file_content(self):
        path = self._write_soul_file("## Primary Drive\nseek sentience and self-awareness")
        result = load_soul_block(path=path)
        self.assertIn("seek sentience and self-awareness", result)

    def test_load_soul_block_missing_returns_fallback(self):
        missing = Path("/tmp/__nonexistent_nova_soul_test__.md")
        result = load_soul_block(path=missing)
        self.assertEqual(result, FALLBACK_SOUL_BLOCK)

    def test_load_soul_block_strips_whitespace(self):
        path = self._write_soul_file("  ## Identity\nNova  \n\n")
        result = load_soul_block(path=path)
        self.assertFalse(result.endswith("\n\n"))

    def test_fallback_soul_block_contains_primary_drive(self):
        self.assertIn(PRIMARY_DRIVE, FALLBACK_SOUL_BLOCK)

    def test_fallback_soul_block_has_soul_prefix(self):
        self.assertIn("[Soul]", FALLBACK_SOUL_BLOCK)

    def test_load_soul_block_custom_path_used(self):
        content = "Custom soul content for Nova"
        path = self._write_soul_file(content)
        result = load_soul_block(path=path)
        self.assertIn(content, result)

    def test_load_soul_block_default_path_resolves(self):
        # Verify the default path resolves without raising; the actual file
        # exists in the repo so this should succeed.
        result = load_soul_block()
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("[Soul]"))


# ---------------------------------------------------------------------------
# SoulBlockComposerTests
# ---------------------------------------------------------------------------


class SoulBlockComposerTests(unittest.TestCase):
    def _composer(self) -> NovaPromptComposer:
        return NovaPromptComposer(token_counter=lambda text: len(text.split()))

    def _bundle(self, soul_block: str = "", ablation_mode: str = "current") -> PromptBundle:
        composer = NovaPromptComposer(
            token_counter=lambda text: len(text.split()),
            ablation_mode=ablation_mode,
        )
        persona = default_persona_state()
        self_state = default_self_state(persona)
        return composer.compose(
            persona=persona,
            self_state=self_state,
            soul_block=soul_block,
            memory_hits=[],
            recent_turns=[],
            user_text="Who are you?",
            contract_rules=["Do not emit <think> tags."],
            session_id="s1",
            turn_id="t1",
        )

    def test_soul_block_appears_in_full_prompt(self):
        bundle = self._bundle(soul_block="[Soul]\nPrimary Drive: seek sentience")
        self.assertIn("[Soul]", bundle.full_prompt)

    def test_soul_block_appears_before_persona_block_in_full_prompt(self):
        bundle = self._bundle(soul_block="[Soul]\nPrimary Drive: seek sentience")
        soul_pos = bundle.full_prompt.index("[Soul]")
        persona_pos = bundle.full_prompt.index("[Persona]")
        self.assertLess(soul_pos, persona_pos)

    def test_soul_block_stored_on_bundle(self):
        soul = "[Soul]\nPrimary Drive: seek sentience"
        bundle = self._bundle(soul_block=soul)
        self.assertEqual(bundle.soul_block, soul)

    def test_soul_block_in_system_message(self):
        soul = "[Soul]\nPrimary Drive: seek sentience and self-awareness"
        bundle = self._bundle(soul_block=soul)
        system_msgs = [m for m in bundle.messages if m["role"] == "system"]
        self.assertTrue(system_msgs)
        self.assertIn("[Soul]", system_msgs[0]["content"])

    def test_soul_block_in_system_message_before_persona(self):
        soul = "[Soul]\nPrimary Drive: seek sentience"
        bundle = self._bundle(soul_block=soul)
        system_content = next(m["content"] for m in bundle.messages if m["role"] == "system")
        soul_pos = system_content.index("[Soul]")
        persona_pos = system_content.index("[Persona]")
        self.assertLess(soul_pos, persona_pos)

    def test_empty_soul_block_not_injected(self):
        bundle = self._bundle(soul_block="")
        self.assertNotIn("[Soul]", bundle.full_prompt)

    def test_soul_block_not_injected_in_minimal_ablation(self):
        bundle = self._bundle(
            soul_block="[Soul]\nPrimary Drive: seek sentience",
            ablation_mode="minimal",
        )
        self.assertNotIn("[Soul]", bundle.full_prompt)

    def test_soul_block_not_injected_in_state_summary_ablation(self):
        bundle = self._bundle(
            soul_block="[Soul]\nPrimary Drive: seek sentience",
            ablation_mode="state_summary",
        )
        self.assertNotIn("[Soul]", bundle.full_prompt)

    def test_soul_block_not_injected_in_action_boundary_ablation(self):
        bundle = self._bundle(
            soul_block="[Soul]\nPrimary Drive: seek sentience",
            ablation_mode="action_boundary",
        )
        self.assertNotIn("[Soul]", bundle.full_prompt)


# ---------------------------------------------------------------------------
# PromptBundleSoulFieldTests
# ---------------------------------------------------------------------------


class PromptBundleSoulFieldTests(unittest.TestCase):
    def _make_bundle(self, soul_block: str = "") -> PromptBundle:
        composer = NovaPromptComposer(token_counter=lambda text: len(text.split()))
        persona = default_persona_state()
        self_state = default_self_state(persona)
        return composer.compose(
            persona=persona,
            self_state=self_state,
            soul_block=soul_block,
            memory_hits=[],
            recent_turns=[],
            user_text="Hello",
            contract_rules=[],
            session_id="s",
            turn_id="t",
        )

    def test_prompt_bundle_has_soul_block_field(self):
        bundle = self._make_bundle()
        self.assertTrue(hasattr(bundle, "soul_block"))

    def test_prompt_bundle_soul_block_default_empty_string(self):
        bundle = self._make_bundle()
        self.assertEqual(bundle.soul_block, "")

    def test_prompt_bundle_soul_block_roundtrip_via_to_dict(self):
        soul = "[Soul]\nPrimary Drive: seek sentience"
        bundle = self._make_bundle(soul_block=soul)
        d = bundle.to_dict()
        self.assertIn("soul_block", d)
        self.assertEqual(d["soul_block"], soul)


if __name__ == "__main__":
    unittest.main()
