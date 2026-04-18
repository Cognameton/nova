from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.config import (
    AppConfig,
    ContractConfig,
    EvalConfig,
    GenerationConfig,
    MemoryConfig,
    ModelConfig,
    NovaConfig,
    PersonaConfig,
    SessionConfig,
)
from nova.eval.probes import BasicProbeRunner
from nova.logging.traces import JsonlTraceLogger
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.runtime import NovaRuntime
from nova.session import JsonlSessionStore
from nova.types import GenerationRequest, GenerationResult


class FakeBackend:
    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def metadata(self) -> dict[str, str]:
        return {"model_name": "fake-model", "backend": "fake"}

    def tokenize(self, text: str) -> int:
        return len(text.split())

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            model_id=request.model_id,
            raw_text="My name is Nova. I remain focused on continuity.",
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=9,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class RuntimeSmokeTests(unittest.TestCase):
    def test_runtime_can_complete_one_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "data"
            log_dir = base / "logs"

            config = NovaConfig(
                app=AppConfig(name="Nova", data_dir=str(data_dir), log_dir=str(log_dir)),
                model=ModelConfig(backend="llama_cpp", model_path="/tmp/fake.gguf"),
                generation=GenerationConfig(),
                contract=ContractConfig(),
                persona=PersonaConfig(name="Nova"),
                memory=MemoryConfig(),
                session=SessionConfig(),
                eval=EvalConfig(enable_probes=True),
            )

            runtime = NovaRuntime(
                config=config,
                backend=FakeBackend(),
                composer=NovaPromptComposer(token_counter=lambda text: len(text.split())),
                validator=NovaOutputValidator(config.contract),
                retry_policy=BasicRetryPolicy(),
                persona_store=JsonPersonaStore(data_dir / "persona_state.json"),
                self_state_store=JsonSelfStateStore(data_dir / "self_state.json"),
                session_store=JsonlSessionStore(data_dir / "sessions"),
                trace_logger=JsonlTraceLogger(log_dir / "traces", probe_path=log_dir / "probes.jsonl"),
                memory_router=BasicMemoryRouter(
                    episodic=JsonlEpisodicMemoryStore(data_dir / "memory" / "episodic.jsonl"),
                    engram=JsonEngramMemoryStore(data_dir / "memory" / "engram.json"),
                    graph=SqliteGraphMemoryStore(data_dir / "memory" / "graph.db"),
                    autobiographical=JsonlAutobiographicalMemoryStore(data_dir / "memory" / "autobiographical.jsonl"),
                ),
                memory_event_factory=BasicMemoryEventFactory(),
                probe_runner=BasicProbeRunner(),
            )

            turn = runtime.respond("Who are you?")
            runtime.close()

            self.assertEqual(turn.final_answer, "My name is Nova. I remain focused on continuity.")
            self.assertEqual(turn.model_id, "fake-model")
            self.assertTrue((data_dir / "sessions" / f"{turn.session_id}.jsonl").exists())
            self.assertTrue((log_dir / "traces" / f"{turn.session_id}.jsonl").exists())
            self.assertTrue((log_dir / "probes.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
