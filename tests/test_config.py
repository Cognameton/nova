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


if __name__ == "__main__":
    unittest.main()
