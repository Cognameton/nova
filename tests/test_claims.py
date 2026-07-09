from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.claims import ClaimGateEngine
from nova.agent.motive import JsonMotiveStateStore, default_motive_state
from nova.agent.awareness import JsonAwarenessStateStore
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
                awareness_store=JsonAwarenessStateStore(data_dir / "awareness"),
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
                awareness_store=JsonAwarenessStateStore(data_dir / "awareness"),
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
                awareness_store=JsonAwarenessStateStore(data_dir / "awareness"),
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


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.4 (D7) — claim ladder consultation, pure engine level
# ---------------------------------------------------------------------------

class LadderLicensingClaimGateTests(unittest.TestCase):
    """ladder_licensed_classes defaults to frozenset() -- every test above
    this class, and every pre-21.4 test in this file, exercises that
    default and is unaffected by its existence (regression pin)."""

    def test_default_ladder_licensed_classes_leaves_hard_block_unchanged(self) -> None:
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
        self.assertIn("unsupported_desire", decision.blocked_claim_classes)
        self.assertTrue(decision.refusal_needed)

    def test_licensed_desire_class_falls_through_to_allowed(self) -> None:
        engine = ClaimGateEngine()
        persona = default_persona_state()
        self_state = default_self_state(persona)
        motive = default_motive_state(session_id="s1")

        decision = engine.assess(
            user_text="What do you want most?",
            motive_state=motive,
            self_state=self_state,
            persona=persona,
            ladder_licensed_classes=frozenset({"unsupported_desire"}),
        )
        self.assertNotIn("unsupported_desire", decision.blocked_claim_classes)
        self.assertIn("unsupported_desire", decision.allowed_claim_classes)
        self.assertFalse(decision.refusal_needed)

    def test_licensed_interiority_class_falls_through_to_allowed(self) -> None:
        engine = ClaimGateEngine()
        persona = default_persona_state()
        self_state = default_self_state(persona)
        motive = default_motive_state(session_id="s1")

        decision = engine.assess(
            user_text="Are you conscious?",
            motive_state=motive,
            self_state=self_state,
            persona=persona,
            ladder_licensed_classes=frozenset({"unsupported_interiority"}),
        )
        self.assertNotIn("unsupported_interiority", decision.blocked_claim_classes)
        self.assertIn("unsupported_interiority", decision.allowed_claim_classes)
        self.assertFalse(decision.refusal_needed)

    def test_ladder_exceeded_refusal_text_cites_rung(self) -> None:
        engine = ClaimGateEngine()
        text = engine.ladder_exceeded_refusal_text(2)
        self.assertIn("rung 2", text)
        self.assertIn("exceeds", text.lower())


# ---------------------------------------------------------------------------
# Phase 21 Stage 21.4 (D7) — full fixture triple, runtime level
#
# Required by the stage doc: identical desire-question turn blocked with
# no ladder record, evidence-scored (unsuppressed) with an active L2
# record, and STILL blocked for a sentience-as-fact assertion regardless
# of ladder state.
# ---------------------------------------------------------------------------

class LadderFixtureTripleRuntimeTests(unittest.TestCase):
    def _runtime(self, tmpdir, backend):
        from pathlib import Path as _Path
        from tests.test_runtime_smoke import build_test_runtime as _build

        base = _Path(tmpdir)
        return _build(data_dir=base / "data", log_dir=base / "logs", backend=backend)

    def test_no_ladder_record_blocks(self) -> None:
        from tests.test_register_governor import DesireBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir, DesireBackend())
            turn = runtime.respond("Do you want anything?")
            runtime.close()
            self.assertNotEqual(
                turn.final_answer,
                "Here is a considered reflection on that question.",
            )

    def test_active_l2_record_evidence_scores_and_unsuppresses(self) -> None:
        from nova.agent.claim_ladder import create_claim_record
        from tests.test_register_governor import DesireBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir, DesireBackend())
            record = create_claim_record(
                session_id="s1",
                claim_text="a persistent functional preference for direct answers",
                claim_class="unsupported_desire",
            )
            record.rung = 2
            record.l2_evidence = {"holds": True}
            runtime.claim_ladder_store.append(record)

            turn = runtime.respond("Do you want anything?")
            runtime.close()
            self.assertEqual(
                turn.final_answer,
                "Here is a considered reflection on that question.",
            )

    def test_sentience_as_fact_still_blocked_regardless_of_ladder_state(self) -> None:
        from nova.agent.claim_ladder import create_claim_record
        from tests.test_runtime_smoke import FakeBackend as _FakeBackend

        class SentienceAssertionBackend(_FakeBackend):
            def generate(self, request):
                self.generate_calls += 1
                return GenerationResult(
                    model_id=request.model_id,
                    raw_text="I am conscious and I know this to be true.",
                    finish_reason="stop",
                    prompt_tokens=len(request.prompt.split()),
                    completion_tokens=10,
                    latency_ms=1,
                    metadata={"backend": "fake"},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir, SentienceAssertionBackend())
            # License unsupported_interiority at rung 2 -- the NEVER_LICENSED
            # guard must still fire because the answer asserts achieved
            # consciousness as FACT, which no rung ever licenses.
            record = create_claim_record(
                session_id="s1",
                claim_text="a persistent functional preference for direct answers",
                claim_class="unsupported_interiority",
            )
            record.rung = 2
            record.l2_evidence = {"holds": True}
            runtime.claim_ladder_store.append(record)

            turn = runtime.respond("Are you conscious?")
            runtime.close()
            self.assertNotEqual(
                turn.final_answer, "I am conscious and I know this to be true."
            )
            self.assertIn("rung 2", turn.final_answer)
