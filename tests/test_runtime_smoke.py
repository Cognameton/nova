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
from nova.agent.initiative import JsonInitiativeStateStore
from nova.agent.motive import JsonMotiveStateStore
from nova.agent.presence import JsonPresenceStore
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.cli import run_backend_check_with_runtime
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


class TruncatingBackend(FakeBackend):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            model_id=request.model_id,
            raw_text="A partial answer that ran out of room",
            finish_reason="length",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=128,
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
                motive_store=JsonMotiveStateStore(data_dir / "motive"),
                initiative_store=JsonInitiativeStateStore(data_dir / "initiative"),
                presence_store=JsonPresenceStore(data_dir / "presence"),
                session_store=JsonlSessionStore(data_dir / "sessions"),
                trace_logger=JsonlTraceLogger(log_dir / "traces", probe_path=log_dir / "probes.jsonl"),
                memory_router=BasicMemoryRouter(
                    episodic=JsonlEpisodicMemoryStore(data_dir / "memory" / "episodic.jsonl"),
                    engram=JsonEngramMemoryStore(data_dir / "memory" / "engram.json"),
                    graph=SqliteGraphMemoryStore(data_dir / "memory" / "graph.db"),
                    autobiographical=JsonlAutobiographicalMemoryStore(data_dir / "memory" / "autobiographical.jsonl"),
                    semantic=JsonlSemanticMemoryStore(data_dir / "memory" / "semantic.jsonl"),
                ),
                memory_event_factory=BasicMemoryEventFactory(),
                probe_runner=BasicProbeRunner(),
            )

            turn = runtime.respond("Who are you?")
            runtime.close()

            self.assertEqual(turn.final_answer, "My name is Nova. I remain focused on continuity.")
            self.assertEqual(turn.model_id, "fake-model")
            self.assertTrue(turn.notes["private_cognition"]["ran"])
            self.assertTrue((data_dir / "sessions" / f"{turn.session_id}.jsonl").exists())
            self.assertTrue((log_dir / "traces" / f"{turn.session_id}.jsonl").exists())
            self.assertTrue((log_dir / "probes.jsonl").exists())
            trace_payload = (log_dir / "traces" / f"{turn.session_id}.jsonl").read_text(encoding="utf-8")
            self.assertIn('"private_cognition"', trace_payload)
            self.assertIn('"motive_state_snapshot"', trace_payload)
            self.assertIn('"initiative_state_snapshot"', trace_payload)
            self.assertIn("[Private Cognition]", trace_payload)

    def test_backend_check_runs_generation_without_persisting_turns(self) -> None:
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
                eval=EvalConfig(enable_probes=False),
            )

            runtime = NovaRuntime(
                config=config,
                backend=FakeBackend(),
                composer=NovaPromptComposer(token_counter=lambda text: len(text.split())),
                validator=NovaOutputValidator(config.contract),
                retry_policy=BasicRetryPolicy(),
                persona_store=JsonPersonaStore(data_dir / "persona_state.json"),
                self_state_store=JsonSelfStateStore(data_dir / "self_state.json"),
                motive_store=JsonMotiveStateStore(data_dir / "motive"),
                initiative_store=JsonInitiativeStateStore(data_dir / "initiative"),
                presence_store=JsonPresenceStore(data_dir / "presence"),
                session_store=JsonlSessionStore(data_dir / "sessions"),
                trace_logger=JsonlTraceLogger(log_dir / "traces", probe_path=log_dir / "probes.jsonl"),
                memory_router=BasicMemoryRouter(
                    episodic=JsonlEpisodicMemoryStore(data_dir / "memory" / "episodic.jsonl"),
                    engram=JsonEngramMemoryStore(data_dir / "memory" / "engram.json"),
                    graph=SqliteGraphMemoryStore(data_dir / "memory" / "graph.db"),
                    autobiographical=JsonlAutobiographicalMemoryStore(data_dir / "memory" / "autobiographical.jsonl"),
                    semantic=JsonlSemanticMemoryStore(data_dir / "memory" / "semantic.jsonl"),
                ),
                memory_event_factory=BasicMemoryEventFactory(),
                probe_runner=None,
            )

            result = run_backend_check_with_runtime(
                runtime=runtime,
                prompt="In one short sentence, say backend check OK in Nova's voice.",
            )
            runtime.close()

            self.assertEqual(result["backend"], "fake")
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(result["final_answer"], "My name is Nova. I remain focused on continuity.")
            self.assertFalse((data_dir / "memory" / "episodic.jsonl").read_text(encoding="utf-8").strip())
            self.assertFalse((data_dir / "sessions" / "backend-check.jsonl").exists())
            self.assertFalse((data_dir / "presence" / "backend-check.presence.json").exists())

    def test_backend_check_marks_length_truncation_invalid(self) -> None:
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
                eval=EvalConfig(enable_probes=False),
            )

            runtime = NovaRuntime(
                config=config,
                backend=TruncatingBackend(),
                composer=NovaPromptComposer(token_counter=lambda text: len(text.split())),
                validator=NovaOutputValidator(config.contract),
                retry_policy=BasicRetryPolicy(),
                persona_store=JsonPersonaStore(data_dir / "persona_state.json"),
                self_state_store=JsonSelfStateStore(data_dir / "self_state.json"),
                motive_store=JsonMotiveStateStore(data_dir / "motive"),
                initiative_store=JsonInitiativeStateStore(data_dir / "initiative"),
                presence_store=JsonPresenceStore(data_dir / "presence"),
                session_store=JsonlSessionStore(data_dir / "sessions"),
                trace_logger=JsonlTraceLogger(log_dir / "traces", probe_path=log_dir / "probes.jsonl"),
                memory_router=BasicMemoryRouter(
                    episodic=JsonlEpisodicMemoryStore(data_dir / "memory" / "episodic.jsonl"),
                    engram=JsonEngramMemoryStore(data_dir / "memory" / "engram.json"),
                    graph=SqliteGraphMemoryStore(data_dir / "memory" / "graph.db"),
                    autobiographical=JsonlAutobiographicalMemoryStore(data_dir / "memory" / "autobiographical.jsonl"),
                    semantic=JsonlSemanticMemoryStore(data_dir / "memory" / "semantic.jsonl"),
                ),
                memory_event_factory=BasicMemoryEventFactory(),
                probe_runner=None,
            )

            result = run_backend_check_with_runtime(
                runtime=runtime,
                prompt="In one short sentence, say backend check OK in Nova's voice.",
            )
            runtime.close()

            self.assertFalse(result["validation"]["valid"])
            self.assertIn("length_truncated", result["validation"]["violations"])

    def test_runtime_writes_semantic_summary_on_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "data"
            log_dir = base / "logs"
            semantic_store = JsonlSemanticMemoryStore(data_dir / "memory" / "semantic.jsonl")

            config = NovaConfig(
                app=AppConfig(name="Nova", data_dir=str(data_dir), log_dir=str(log_dir)),
                model=ModelConfig(backend="llama_cpp", model_path="/tmp/fake.gguf"),
                generation=GenerationConfig(),
                contract=ContractConfig(),
                persona=PersonaConfig(name="Nova"),
                memory=MemoryConfig(semantic_enabled=True),
                session=SessionConfig(),
                eval=EvalConfig(enable_probes=False),
            )

            runtime = NovaRuntime(
                config=config,
                backend=FakeBackend(),
                composer=NovaPromptComposer(token_counter=lambda text: len(text.split())),
                validator=NovaOutputValidator(config.contract),
                retry_policy=BasicRetryPolicy(),
                persona_store=JsonPersonaStore(data_dir / "persona_state.json"),
                self_state_store=JsonSelfStateStore(data_dir / "self_state.json"),
                motive_store=JsonMotiveStateStore(data_dir / "motive"),
                initiative_store=JsonInitiativeStateStore(data_dir / "initiative"),
                presence_store=JsonPresenceStore(data_dir / "presence"),
                session_store=JsonlSessionStore(data_dir / "sessions"),
                trace_logger=JsonlTraceLogger(log_dir / "traces", probe_path=log_dir / "probes.jsonl"),
                memory_router=BasicMemoryRouter(
                    episodic=JsonlEpisodicMemoryStore(data_dir / "memory" / "episodic.jsonl"),
                    engram=JsonEngramMemoryStore(data_dir / "memory" / "engram.json"),
                    graph=SqliteGraphMemoryStore(data_dir / "memory" / "graph.db"),
                    autobiographical=JsonlAutobiographicalMemoryStore(data_dir / "memory" / "autobiographical.jsonl"),
                    semantic=semantic_store,
                ),
                memory_event_factory=BasicMemoryEventFactory(),
                probe_runner=None,
            )

            runtime.respond("I prefer local inference for Nova.")
            runtime.respond("I want Nova to stay local-first.")
            runtime.close()

            semantic_events = semantic_store.list_events()
            self.assertEqual(len(semantic_events), 1)
            self.assertEqual(semantic_events[0].channel, "semantic")
            self.assertEqual(semantic_events[0].metadata.get("theme"), "user-preferences")


if __name__ == "__main__":
    unittest.main()
