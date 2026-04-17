"""Nova runtime orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nova.config import NovaConfig
from nova.inference.base import InferenceBackend
from nova.logging.traces import JsonlTraceLogger
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.contract import build_contract_rules
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.session import JsonlSessionStore
from nova.types import TraceRecord, TurnRecord


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NovaRuntime:
    """Phase 1 runtime orchestrator for Nova 2.0."""

    def __init__(
        self,
        config: NovaConfig,
        backend: InferenceBackend,
        composer: NovaPromptComposer,
        validator: NovaOutputValidator,
        retry_policy: BasicRetryPolicy,
        persona_store: JsonPersonaStore,
        self_state_store: JsonSelfStateStore,
        session_store: JsonlSessionStore,
        trace_logger: JsonlTraceLogger,
        memory_router: BasicMemoryRouter,
        memory_event_factory: BasicMemoryEventFactory,
        probe_runner: object | None = None,
    ):
        self.config = config
        self.backend = backend
        self.composer = composer
        self.validator = validator
        self.retry_policy = retry_policy
        self.persona_store = persona_store
        self.self_state_store = self_state_store
        self.session_store = session_store
        self.trace_logger = trace_logger
        self.memory_router = memory_router
        self.memory_event_factory = memory_event_factory
        self.probe_runner = probe_runner

        self.session_id: str | None = None
        self.persona = None
        self.self_state = None

    def start(self) -> str:
        self.persona = self.persona_store.load()
        self.self_state = self.self_state_store.load(persona=self.persona)
        self.backend.load()
        self.session_id = self.session_store.start_session()
        return self.session_id

    def respond(self, user_text: str) -> TurnRecord:
        if self.session_id is None:
            self.start()
        assert self.session_id is not None
        assert self.persona is not None
        assert self.self_state is not None

        turn_id = uuid4().hex
        contract_rules = build_contract_rules(self.persona, self.config.contract)
        recent_turns = self.session_store.recent_turns(
            session_id=self.session_id,
            limit=self.config.session.max_recent_turns,
        )
        memory_hits = self.memory_router.retrieve(
            query=user_text,
            top_k_by_channel={
                "episodic": 4,
                "graph": 4,
                "autobiographical": 3,
            },
        )
        prompt_bundle = self.composer.compose(
            persona=self.persona,
            self_state=self.self_state,
            memory_hits=memory_hits,
            recent_turns=recent_turns,
            user_text=user_text,
            contract_rules=contract_rules,
            session_id=self.session_id,
            turn_id=turn_id,
        )

        generation_request = self._generation_request(
            prompt=prompt_bundle.full_prompt,
        )
        generation_result = self.backend.generate(generation_request)
        validation = self.validator.validate(
            raw_text=generation_result.raw_text,
            persona=self.persona,
            contract_rules=contract_rules,
        )

        retries: list[dict] = []
        retry_count = 0
        final_answer = generation_result.raw_text

        while self.retry_policy.should_retry(
            validation=validation,
            attempt_index=retry_count,
            max_retries=self.config.generation.retries,
        ):
            retry_count += 1
            retry_instruction = self.retry_policy.build_retry_instruction(
                user_text=user_text,
                raw_answer=final_answer,
                validation=validation,
            )
            retry_prompt = prompt_bundle.full_prompt + "\n\n[Retry Instruction]\n" + retry_instruction
            retry_request = self._generation_request(prompt=retry_prompt)
            retry_result = self.backend.generate(retry_request)
            retry_validation = self.validator.validate(
                raw_text=retry_result.raw_text,
                persona=self.persona,
                contract_rules=contract_rules,
            )
            retries.append(
                {
                    "attempt": retry_count,
                    "instruction": retry_instruction,
                    "generation_request": retry_request.to_dict(),
                    "generation_result": retry_result.to_dict(),
                    "validation_result": retry_validation.to_dict(),
                }
            )
            generation_result = retry_result
            validation = retry_validation
            final_answer = retry_result.raw_text

        if not validation.valid:
            final_answer = (
                "I need to restate that more clearly. Please try again."
            )

        turn = TurnRecord(
            session_id=self.session_id,
            turn_id=turn_id,
            timestamp=utc_now_iso(),
            user_text=user_text,
            final_answer=final_answer,
            raw_answer=generation_result.raw_text,
            validation=validation,
            memory_hits=memory_hits,
            prompt_token_estimate=prompt_bundle.token_estimate,
            completion_token_estimate=generation_result.completion_tokens,
            latency_ms=generation_result.latency_ms,
            model_id=generation_result.model_id,
            retry_count=retry_count,
            notes={},
        )
        self.session_store.append_turn(turn)

        persisted_memory_events = []
        if validation.valid:
            memory_events = self.memory_event_factory.from_turn(
                session_id=self.session_id,
                turn_id=turn_id,
                user_text=user_text,
                final_answer=final_answer,
            )
            self.memory_router.add_events(memory_events)
            persisted_memory_events = [event.to_dict() for event in memory_events]

        trace = TraceRecord(
            session_id=self.session_id,
            turn_id=turn_id,
            timestamp=turn.timestamp,
            config_snapshot=self.config.snapshot(),
            persona_state_snapshot=self.persona.to_dict(),
            self_state_snapshot=self.self_state.to_dict(),
            prompt_bundle=prompt_bundle.to_dict(),
            generation_request=generation_request.to_dict(),
            generation_result=generation_result.to_dict(),
            validation_result=validation.to_dict(),
            retries=retries,
            persisted_memory_events=persisted_memory_events,
        )
        self.trace_logger.log_trace(trace)
        return turn

    def close(self) -> None:
        self.backend.unload()

    def _generation_request(self, *, prompt: str):
        from nova.types import GenerationRequest

        return GenerationRequest(
            model_id=self.backend.metadata().get("model_name", "nova-model"),
            prompt=prompt,
            max_tokens=self.config.generation.max_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            stop=list(self.config.generation.stop),
            seed=None,
            retries_allowed=self.config.generation.retries,
        )
