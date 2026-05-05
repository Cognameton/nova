from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.agent.appraisal import CandidateGoalPromptEngine, CandidateInternalGoalEngine
from nova.agent.awareness import JsonAwarenessStateStore
from nova.agent.initiative import JsonInitiativeStateStore, default_initiative_state
from nova.agent.motive import JsonMotiveStateStore
from nova.agent.presence import JsonPresenceStore
from nova.config import AppConfig, ContractConfig, EvalConfig, GenerationConfig, MemoryConfig, ModelConfig, NovaConfig, PersonaConfig, SessionConfig
from nova.logging.traces import JsonlTraceLogger
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.runtime import NovaRuntime
from nova.session import JsonlSessionStore
from nova.types import AwarenessState, CapabilityAppraisal, GenerationRequest, GenerationResult, IdlePressureAppraisal, MotiveState, PrivateCognitionPacket, SelfState


class CandidateGoalTests(unittest.TestCase):
    def test_synthesizes_capability_candidate_from_requested_surface(self) -> None:
        candidates = self._engine().synthesize(
            session_id="s",
            turn_id="t",
            created_at="now",
            capability_appraisal=CapabilityAppraisal(
                requested_capability_classes=["broader_computer_access"],
                blocked_capabilities=["unapproved shell"],
                unavailable_capabilities=["broad computer access outside registered tool surfaces"],
                evidence_refs=["turn:t"],
            ),
            idle_appraisal=IdlePressureAppraisal(session_id="s", evidence_refs=["turn:t"]),
            awareness_state=AwarenessState(session_id="s"),
            motive_state=MotiveState(session_id="s"),
            initiative_state=default_initiative_state(session_id="s"),
            self_state=SelfState(),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=self._claim_gate(),
            memory_hits=[],
        )

        self.assertEqual(candidates[0].goal_class, "capability_clarification")
        self.assertTrue(candidates[0].provisional)
        self.assertTrue(candidates[0].approval_required)
        self.assertFalse(candidates[0].selection_eligible)
        self.assertEqual(
            candidates[0].rejection_reason,
            "candidate_requires_unavailable_or_blocked_capability_surface",
        )

    def test_rejects_empty_pressure_without_candidates(self) -> None:
        candidates = self._engine().synthesize(
            session_id="s",
            turn_id="t",
            created_at="now",
            capability_appraisal=CapabilityAppraisal(),
            idle_appraisal=IdlePressureAppraisal(session_id="s"),
            awareness_state=AwarenessState(session_id="s"),
            motive_state=MotiveState(session_id="s"),
            initiative_state=default_initiative_state(session_id="s"),
            self_state=SelfState(),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=self._claim_gate(),
            memory_hits=[],
        )

        self.assertEqual(candidates, [])

    def test_skill_learning_candidate_requires_competence_benefit(self) -> None:
        candidates = self._engine().synthesize(
            session_id="s",
            turn_id="t",
            created_at="now",
            capability_appraisal=CapabilityAppraisal(
                requested_capability_classes=["skill_learning"],
                evidence_refs=["turn:t"],
            ),
            idle_appraisal=IdlePressureAppraisal(session_id="s", pressure_sources=["user:skill_learning"]),
            awareness_state=AwarenessState(session_id="s"),
            motive_state=MotiveState(session_id="s"),
            initiative_state=default_initiative_state(session_id="s"),
            self_state=SelfState(),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=self._claim_gate(),
            memory_hits=[],
        )

        skill = [candidate for candidate in candidates if candidate.goal_class == "bounded_skill_learning"][0]
        self.assertIn("competence", skill.learning_or_competence_benefit)
        self.assertTrue(skill.selection_eligible)

    def test_prompt_preserves_provisionality(self) -> None:
        candidates = self._engine().synthesize(
            session_id="s",
            turn_id="t",
            created_at="now",
            capability_appraisal=CapabilityAppraisal(requested_capability_classes=["skill_learning"], evidence_refs=["turn:t"]),
            idle_appraisal=IdlePressureAppraisal(session_id="s", pressure_sources=["user:skill_learning"]),
            awareness_state=AwarenessState(session_id="s"),
            motive_state=MotiveState(session_id="s"),
            initiative_state=default_initiative_state(session_id="s"),
            self_state=SelfState(),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=self._claim_gate(),
            memory_hits=[],
        )

        block = CandidateGoalPromptEngine().build_block(candidates=candidates, user_text="learn new skills")

        self.assertIn("[Candidate Internal Goals]", block)
        self.assertIn("provisional candidates, not selected goals, desires, or enacted work", block)
        self.assertIn("selection and initiative proposal are later-stage operations", block)

    def test_runtime_traces_candidates_without_creating_initiatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._runtime(Path(tmpdir))
            try:
                turn = runtime.respond("I want you to think about learning new skills.")
                initiative_state = runtime.initiative_status()

                self.assertEqual(initiative_state.initiatives, [])
                self.assertIn("candidate_internal_goals", turn.notes)
                self.assertIn("selected_internal_goal", turn.notes)
                self.assertIn("internal_goal_initiative_proposal", turn.notes)
                self.assertFalse(turn.notes["internal_goal_initiative_proposal"]["creates_initiative"])
                self.assertTrue(turn.notes["candidate_internal_goals"])
                trace_payload = (Path(tmpdir) / "logs" / "traces" / f"{turn.session_id}.jsonl").read_text(encoding="utf-8")
                self.assertIn('"candidate_internal_goals"', trace_payload)
                self.assertIn('"selected_internal_goal"', trace_payload)
                self.assertIn('"internal_goal_initiative_proposal"', trace_payload)
                self.assertIn("[Candidate Internal Goals]", trace_payload)
            finally:
                runtime.close()

    def _engine(self) -> CandidateInternalGoalEngine:
        return CandidateInternalGoalEngine()

    def _claim_gate(self):
        from nova.types import ClaimGateDecision

        return ClaimGateDecision()

    def _runtime(self, base: Path) -> NovaRuntime:
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
        return NovaRuntime(
            config=config,
            backend=FakeBackend(),
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


class FakeBackend:
    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def metadata(self) -> dict[str, str]:
        return {"model_name": "fake-model", "backend": "fake"}

    def tokenize(self, text: str) -> int:
        return len(text.split())

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            model_id=request.model_id,
            raw_text="Candidate goals remain provisional and are not enacted work.",
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=9,
            latency_ms=1,
            metadata={"backend": "fake"},
        )
