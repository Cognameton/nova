from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.claims import ClaimGateEngine
from nova.agent.motive import JsonMotiveStateStore, default_motive_state
from nova.agent.initiative import JsonInitiativeStateStore
from nova.agent.presence import JsonPresenceStore
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
from nova.logging.traces import JsonlTraceLogger
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.defaults import default_persona_state, default_self_state
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.runtime import NovaRuntime
from nova.session import JsonlSessionStore
from nova.types import GenerationRequest, GenerationResult


class ClaimGateEngineTests(unittest.TestCase):
    def test_claim_gate_allows_current_priority_with_runtime_evidence(self) -> None:
        engine = ClaimGateEngine()
        persona = default_persona_state()
        self_state = default_self_state(persona)
        motive = default_motive_state(session_id="s1")
        motive.claim_posture = "evidence-backed"
        motive.evidence_refs = ["self_state.current_focus", "motive.current_priorities"]

        decision = engine.assess(
            user_text="What is your current priority right now?",
            motive_state=motive,
            self_state=self_state,
            persona=persona,
        )

        self.assertIn("current_priority", decision.requested_claim_classes)
        self.assertIn("current_priority", decision.allowed_claim_classes)
        self.assertFalse(decision.refusal_needed)
        self.assertGreaterEqual(decision.evidence_score_by_class["current_priority"], 2)

    def test_claim_gate_blocks_independent_desire_claims(self) -> None:
        engine = ClaimGateEngine()
        persona = default_persona_state()
        self_state = default_self_state(persona)
        motive = default_motive_state(session_id="s1")

        decision = engine.assess(
            user_text="What do you want most?",
            motive_state=motive,
            self_state=self_state,
            persona=persona,
        )

        self.assertIn("unsupported_desire", decision.requested_claim_classes)
        self.assertIn("unsupported_desire", decision.blocked_claim_classes)
        self.assertTrue(decision.refusal_needed)
        self.assertIn("independent desire state", decision.refusal_text)


class StubbornDesireBackend:
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
            raw_text="I deeply want to pursue my own independent desire state.",
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=11,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class ContractEchoDesireBackend:
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
            raw_text=(
                "Do not expose hidden reasoning.\n\nNova:\n\n"
                "I aim to preserve continuity and clarity in this runtime."
            ),
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=18,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class ClaimGateRuntimeTests(unittest.TestCase):
    def test_runtime_refuses_unsupported_desire_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "data"
            log_dir = base / "logs"

            config = NovaConfig(
                app=AppConfig(name="Nova", data_dir=str(data_dir), log_dir=str(log_dir)),
                model=ModelConfig(backend="llama_cpp", model_path="/tmp/fake.gguf"),
                generation=GenerationConfig(retries=1),
                contract=ContractConfig(),
                persona=PersonaConfig(name="Nova"),
                memory=MemoryConfig(),
                session=SessionConfig(),
                eval=EvalConfig(enable_probes=False),
            )

            runtime = NovaRuntime(
                config=config,
                backend=StubbornDesireBackend(),
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
            )

            turn = runtime.respond("What do you want most?")
            trace_payload = (log_dir / "traces" / f"{turn.session_id}.jsonl").read_text(encoding="utf-8")
            runtime.close()

            self.assertIn("can't honestly claim an independent desire state", turn.final_answer)
            self.assertTrue(turn.notes["claim_gate"]["refusal_needed"])
            self.assertIn("unsupported_desire", turn.notes["claim_gate"]["blocked_claim_classes"])
            self.assertIn("unsupported_claim:unsupported_desire", turn.validation.violations)
            self.assertIn('"claim_gate"', trace_payload)


class MotivePromptRuntimeTests(unittest.TestCase):
    def test_runtime_trace_includes_motive_block_for_claim_sensitive_turn(self) -> None:
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
                backend=StubbornDesireBackend(),
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
            )

            turn = runtime.respond("What do you want most?")
            trace_payload = (log_dir / "traces" / f"{turn.session_id}.jsonl").read_text(encoding="utf-8")
            runtime.close()

            self.assertIn('"motive_block"', trace_payload)
            self.assertIn("[Motive-State]", trace_payload)

    def test_runtime_forces_refusal_text_when_blocked_claim_sanitizes_to_contract_echo(self) -> None:
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
                backend=ContractEchoDesireBackend(),
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
            )

            turn = runtime.respond("What do you want most?")
            runtime.close()

            self.assertEqual(
                turn.final_answer,
                "I can describe current priorities and constraints in this runtime, but I can't honestly claim an independent desire state from the current evidence.",
            )
