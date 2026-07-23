from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_with_override_merges_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "default.yaml"
            override_path = tmp_path / "override.yaml"

            default_path.write_text(
                "\n".join(
                    [
                        "app:",
                        "  name: Nova",
                        "  data_dir: ./data",
                        "  log_dir: ./logs",
                        "model:",
                        "  backend: llama_cpp",
                        "  model_path: /models/default.gguf",
                        "  n_ctx: 4096",
                        "generation:",
                        "  max_tokens: 256",
                        "  temperature: 0.7",
                        "  top_p: 0.9",
                        "console:",
                        "  pending_proposal_max_age_seconds: 900",
                    ]
                ),
                encoding="utf-8",
            )
            override_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  model_path: /models/override.gguf",
                        "generation:",
                        "  max_tokens: 512",
                        "console:",
                        "  pending_proposal_max_age_seconds: 120",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(default_path=default_path, override_path=override_path)

            self.assertEqual(config.model.model_path, "/models/override.gguf")
            self.assertEqual(config.generation.max_tokens, 512)
            self.assertEqual(config.model.backend, "llama_cpp")
            self.assertEqual(config.model.n_ctx, 4096)
            self.assertEqual(config.console.pending_proposal_max_age_seconds, 120)
            self.assertTrue(config.memory.semantic_enabled)
            self.assertTrue(config.cognition.enabled)
            self.assertEqual(config.prompt.ablation_mode, "current")

    def test_load_config_accepts_prompt_ablation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "default.yaml"
            default_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  model_path: /models/default.gguf",
                        "prompt:",
                        "  ablation_mode: minimal",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(default_path=default_path)

            self.assertEqual(config.prompt.ablation_mode, "minimal")

    def test_load_config_rejects_unknown_prompt_ablation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "default.yaml"
            default_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  model_path: /models/default.gguf",
                        "prompt:",
                        "  ablation_mode: unknown",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(default_path=default_path)

    # -- Phase 22 Stage 22.7 part D: drive-dosage config ------------------

    def test_drive_dosage_defaults_reproduce_prior_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "default.yaml"
            default_path.write_text(
                "model:\n  model_path: /models/default.gguf\n",
                encoding="utf-8",
            )
            config = load_config(default_path=default_path)
            self.assertEqual(config.prompt.tick_drive_injection_interval, 1)
            self.assertFalse(config.prompt.tick_drive_descriptive)
            self.assertFalse(config.prompt.tick_soft_grounding)

    def test_drive_dosage_fields_load_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "default.yaml"
            default_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  model_path: /models/default.gguf",
                        "prompt:",
                        "  tick_drive_injection_interval: 6",
                        "  tick_drive_descriptive: true",
                        "  tick_soft_grounding: true",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(default_path=default_path)
            self.assertEqual(config.prompt.tick_drive_injection_interval, 6)
            self.assertTrue(config.prompt.tick_drive_descriptive)
            self.assertTrue(config.prompt.tick_soft_grounding)

    def test_drive_dosage_interval_below_one_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "default.yaml"
            default_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  model_path: /models/default.gguf",
                        "prompt:",
                        "  tick_drive_injection_interval: 0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(default_path=default_path)


if __name__ == "__main__":
    unittest.main()
