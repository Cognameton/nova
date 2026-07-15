"""Tests for LlamaCppBackend's native chat-template thinking suppression.

Phase 22 Stage 22.2b: llama_cpp.Llama is mocked throughout — no real
model load, no GPU/VRAM use. A minimal but real Jinja template
(structurally mirroring Qwen 3's enable_thinking conditional) is used
so the render assertions exercise actual Jinja2ChatFormatter behavior,
not a stub.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from nova.config import ModelConfig, NovaConfig
from nova.inference.llama_cpp_backend import LlamaCppBackend
from nova.types import GenerationRequest


MINI_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
)


def _config(model_path: str = "/fake/model.gguf") -> NovaConfig:
    config = NovaConfig()
    config.model = ModelConfig(model_path=model_path, chat_format="chatml")
    return config


def _mock_llm(chat_template: str | None = MINI_TEMPLATE) -> MagicMock:
    llm = MagicMock()
    metadata: dict[str, str] = {}
    if chat_template is not None:
        metadata["tokenizer.chat_template"] = chat_template
    llm.metadata = metadata
    llm.token_eos.return_value = 151645
    llm.token_bos.return_value = 151643

    def _detokenize(tokens, special=False):
        if tokens == [151645]:
            return b"<|im_end|>"
        if tokens == [151643]:
            return b"<|endoftext|>"
        return b""

    llm.detokenize.side_effect = _detokenize
    return llm


def _completion_response(text: str = "hello") -> dict:
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _chat_completion_response(text: str = "hello") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {"message": {"content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _load_backend(chat_template: str | None = MINI_TEMPLATE):
    backend = LlamaCppBackend(_config())
    mock_llm = _mock_llm(chat_template=chat_template)
    mock_llm.return_value = _completion_response("emit_heartbeat call")
    mock_llm.create_chat_completion.return_value = _chat_completion_response(
        "chat completion path"
    )
    with patch("llama_cpp.Llama", return_value=mock_llm), patch(
        "pathlib.Path.exists", return_value=True
    ):
        backend.load()
    return backend, mock_llm


class LoadBuildsFormatterTests(unittest.TestCase):
    def test_builds_native_formatter_when_template_present(self):
        backend, _ = _load_backend(chat_template=MINI_TEMPLATE)
        self.assertIsNotNone(backend._chat_formatter)

    def test_no_formatter_when_metadata_has_no_template(self):
        backend, _ = _load_backend(chat_template=None)
        self.assertIsNone(backend._chat_formatter)

    def test_no_formatter_and_no_crash_when_detokenize_raises(self):
        backend = LlamaCppBackend(_config())
        mock_llm = _mock_llm(chat_template=MINI_TEMPLATE)
        mock_llm.detokenize.side_effect = RuntimeError("boom")
        with patch("llama_cpp.Llama", return_value=mock_llm), patch(
            "pathlib.Path.exists", return_value=True
        ):
            backend.load()
        self.assertIsNone(backend._chat_formatter)

    def test_unload_resets_formatter_and_llm(self):
        backend, _ = _load_backend(chat_template=MINI_TEMPLATE)
        self.assertIsNotNone(backend._chat_formatter)
        backend.unload()
        self.assertIsNone(backend._llm)
        self.assertIsNone(backend._chat_formatter)


class MetadataReportsNativeTemplateTests(unittest.TestCase):
    def test_metadata_true_when_formatter_present(self):
        backend, _ = _load_backend(chat_template=MINI_TEMPLATE)
        self.assertTrue(backend.metadata()["native_chat_template"])

    def test_metadata_false_when_formatter_absent(self):
        backend, _ = _load_backend(chat_template=None)
        self.assertFalse(backend.metadata()["native_chat_template"])


class GenerateNativeTemplatePathTests(unittest.TestCase):
    def test_uses_raw_completion_with_pre_seeded_think_block_when_formatter_present(
        self,
    ):
        backend, mock_llm = _load_backend()
        request = GenerationRequest(
            model_id="test-model",
            prompt="",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>", "<|im_start|>"],
            messages=[
                {"role": "system", "content": "/no_think\n\nYou are Nova."},
                {"role": "user", "content": "Choose one tool."},
            ],
        )

        result = backend.generate(request)

        mock_llm.create_chat_completion.assert_not_called()
        self.assertEqual(mock_llm.call_count, 1)
        called_prompt = mock_llm.call_args.args[0]
        self.assertIsInstance(called_prompt, str)
        self.assertTrue(called_prompt.endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(result.raw_text, "emit_heartbeat call")
        self.assertEqual(result.metadata["mode"], "native_template_completion")

    def test_merges_formatter_stop_with_request_stop_deduplicated(self):
        backend, mock_llm = _load_backend()
        request = GenerationRequest(
            model_id="test-model",
            prompt="",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>", "<|im_start|>"],
            messages=[{"role": "user", "content": "hi"}],
        )

        backend.generate(request)

        called_stop = mock_llm.call_args.kwargs["stop"]
        self.assertEqual(called_stop, ["<|im_end|>", "<|im_start|>"])

    def test_falls_back_to_chat_completion_when_no_native_formatter(self):
        backend, mock_llm = _load_backend(chat_template=None)
        request = GenerationRequest(
            model_id="test-model",
            prompt="",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>"],
            messages=[{"role": "user", "content": "hi"}],
        )

        result = backend.generate(request)

        mock_llm.create_chat_completion.assert_called_once()
        self.assertEqual(mock_llm.call_count, 0)
        self.assertEqual(result.raw_text, "chat completion path")
        self.assertEqual(result.metadata["mode"], "chat_completion")

    def test_no_messages_uses_plain_completion_unaffected(self):
        backend, mock_llm = _load_backend()
        request = GenerationRequest(
            model_id="test-model",
            prompt="raw prompt text",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>"],
            messages=None,
        )

        result = backend.generate(request)

        mock_llm.create_chat_completion.assert_not_called()
        self.assertEqual(mock_llm.call_args.args[0], "raw prompt text")
        self.assertEqual(result.metadata["mode"], "completion")


if __name__ == "__main__":
    unittest.main()
