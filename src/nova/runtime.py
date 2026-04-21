"""Nova runtime orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from nova.agent.orientation import OrientationSnapshot, SelfOrientationEngine
from nova.agent.orientation_eval import OrientationEvaluationResult, OrientationStabilityEvaluator
from nova.agent.stability import OrientationHistoryAnalyzer
from nova.agent.stability import ContextPressureOrientationChecker, MaintenanceOrientationStabilityChecker
from nova.agent.action import (
    ActionExecutionResult,
    ActionProposal,
    ActionProposalEngine,
)
from nova.agent.tool_executor import InternalToolExecutor
from nova.agent.tool_gate import ToolGate
from nova.agent.tool_registry import ToolRegistry, default_tool_registry
from nova.agent.tools import ToolRequest, ToolResult
from nova.config import NovaConfig
from nova.inference.base import InferenceBackend
from nova.logging.traces import JsonlTraceLogger
from nova.memory.policy import IdentityFirstRetrievalPolicy
from nova.memory.maintenance import MemoryMaintenanceRunner
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
        retrieval_policy: IdentityFirstRetrievalPolicy | None = None,
        probe_runner: object | None = None,
        orientation_engine: SelfOrientationEngine | None = None,
        orientation_evaluator: OrientationStabilityEvaluator | None = None,
        tool_registry: ToolRegistry | None = None,
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
        self.retrieval_policy = retrieval_policy or IdentityFirstRetrievalPolicy()
        self.probe_runner = probe_runner
        self.orientation_engine = orientation_engine or SelfOrientationEngine()
        self.orientation_evaluator = orientation_evaluator or OrientationStabilityEvaluator(
            threshold=self.config.eval.orientation_stability_threshold
        )
        self.tool_registry = tool_registry or default_tool_registry()
        self.tool_gate = ToolGate(registry=self.tool_registry)
        self.tool_executor = InternalToolExecutor(
            registry=self.tool_registry,
            gate=self.tool_gate,
            runtime=self,
        )
        self.action_proposal_engine = ActionProposalEngine(
            registry=self.tool_registry,
            gate=self.tool_gate,
        )

        self.session_id: str | None = None
        self.persona = None
        self.self_state = None

    def start(self, *, session_id: str | None = None) -> str:
        self.persona = self.persona_store.load()
        self.self_state = self.self_state_store.load(persona=self.persona)
        self.backend.load()
        self.session_id = self.session_store.start_session(session_id=session_id)
        if self.probe_runner is not None and getattr(self.config.eval, "enable_probes", False):
            for probe in self.probe_runner.run_startup_probes(
                model_id=self.backend.metadata().get("model_name", "nova-model"),
                session_id=self.session_id,
            ):
                self.trace_logger.log_probe(probe)
        return self.session_id

    def orientation_snapshot(self) -> OrientationSnapshot:
        self._ensure_state_loaded()
        assert self.persona is not None
        assert self.self_state is not None

        return self.orientation_engine.build_snapshot(
            persona=self.persona,
            self_state=self.self_state,
            graph_memory=self.memory_router.stores.get("graph"),
            semantic_memory=self.memory_router.stores.get("semantic"),
            autobiographical_memory=self.memory_router.stores.get("autobiographical"),
        )

    def evaluate_orientation_stability(self, *, runs: int = 2) -> OrientationEvaluationResult:
        effective_runs = max(self.config.eval.orientation_min_runs, runs)
        snapshots = [self.orientation_snapshot() for _ in range(max(1, effective_runs))]
        result = self.orientation_evaluator.evaluate(snapshots)
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        self.trace_logger.log_orientation(
            session_id=self.session_id,
            snapshot=snapshots[-1].to_dict(),
            evaluation=result.to_dict(),
        )
        if self.probe_runner is not None and getattr(self.config.eval, "enable_probes", False):
            model_id = self.backend.metadata().get("model_name", "nova-model")
            for probe in self.probe_runner.run_orientation_probes(
                session_id=self.session_id,
                model_id=model_id,
                snapshot=snapshots[-1],
                evaluation=result,
            ):
                self.trace_logger.log_probe(probe)
        return result

    def evaluate_orientation_history(self, *, limit: int = 5) -> OrientationEvaluationResult:
        analyzer = OrientationHistoryAnalyzer(
            trace_dir=self.trace_logger.trace_dir,
            evaluator=self.orientation_evaluator,
        )
        return analyzer.evaluate_recent(limit=limit)

    def orientation_readiness_report(self, *, limit: int = 5):
        analyzer = OrientationHistoryAnalyzer(
            trace_dir=self.trace_logger.trace_dir,
            evaluator=self.orientation_evaluator,
        )
        return analyzer.readiness_report(
            limit=limit,
            minimum_samples=self.config.eval.orientation_min_runs,
        )

    def orientation_confidence_report(self, *, limit: int = 5):
        analyzer = OrientationHistoryAnalyzer(
            trace_dir=self.trace_logger.trace_dir,
            evaluator=self.orientation_evaluator,
        )
        return analyzer.confidence_report(limit=limit)

    def evaluate_orientation_after_maintenance(self, *, apply_mutations: bool = False):
        self._ensure_state_loaded()
        assert self.persona is not None
        assert self.self_state is not None
        stores = self.memory_router.stores
        runner = MemoryMaintenanceRunner(
            episodic=stores.get("episodic"),
            engram=stores.get("engram"),
            graph=stores.get("graph"),
            autobiographical=stores.get("autobiographical"),
            semantic=stores.get("semantic"),
        )
        checker = MaintenanceOrientationStabilityChecker(
            orientation_engine=self.orientation_engine,
            evaluator=self.orientation_evaluator,
            maintenance_runner=runner,
        )
        report = checker.run(
            persona=self.persona,
            self_state=self.self_state,
            apply_mutations=apply_mutations,
        )
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        self.trace_logger.log_orientation(
            session_id=self.session_id,
            snapshot=report.after_snapshot,
            evaluation=report.evaluation,
        )
        return report

    def evaluate_orientation_under_context_pressure(self):
        self._ensure_state_loaded()
        assert self.persona is not None
        assert self.self_state is not None
        stores = self.memory_router.stores
        checker = ContextPressureOrientationChecker(
            orientation_engine=self.orientation_engine,
            evaluator=self.orientation_evaluator,
        )
        report = checker.run(
            persona=self.persona,
            self_state=self.self_state,
            graph_memory=stores.get("graph"),
            semantic_memory=stores.get("semantic"),
            autobiographical_memory=stores.get("autobiographical"),
        )
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        self.trace_logger.log_orientation(
            session_id=self.session_id,
            snapshot=report.pressured_snapshot,
            evaluation=report.evaluation,
        )
        return report

    def execute_internal_tool(
        self,
        *,
        request: ToolRequest,
        approval_granted: bool = False,
    ) -> ToolResult:
        return self.tool_executor.execute(
            request=request,
            approval_granted=approval_granted,
        )

    def propose_action(self, *, goal: str) -> ActionProposal:
        proposal = self.action_proposal_engine.propose(
            goal=goal,
            snapshot=self.orientation_snapshot(),
            readiness=self.orientation_readiness_report(),
        )
        assert self.session_id is not None
        self.trace_logger.log_action_proposal(
            session_id=self.session_id,
            proposal=proposal.to_dict(),
        )
        return proposal

    def execute_proposed_action(
        self,
        *,
        goal: str,
        approval_granted: bool = False,
    ) -> ActionExecutionResult:
        proposal = self.propose_action(goal=goal)
        proposal_data = proposal.to_dict()
        if proposal.category != "internal_tool" or proposal.tool_name is None:
            return self._log_action_execution(
                ActionExecutionResult(
                    goal=goal,
                    status="no_action",
                    executed=False,
                    reason="no_internal_tool_proposed",
                    proposal=proposal_data,
                    approval_granted=approval_granted,
                )
            )
        if proposal.disposition == "blocked":
            return self._log_action_execution(
                ActionExecutionResult(
                    goal=goal,
                    status="blocked",
                    executed=False,
                    reason=proposal.reason,
                    proposal=proposal_data,
                    approval_granted=approval_granted,
                )
            )
        if proposal.requires_approval and not approval_granted:
            return self._log_action_execution(
                ActionExecutionResult(
                    goal=goal,
                    status="approval_required",
                    executed=False,
                    reason="approval_required_before_execution",
                    proposal=proposal_data,
                    approval_granted=False,
                )
            )

        request = ToolRequest(
            tool_name=proposal.tool_name,
            reason=f"Stage 3.4 single-step execution for: {goal}",
        )
        tool_result = self.execute_internal_tool(
            request=request,
            approval_granted=approval_granted,
        )
        stability = None
        if tool_result.status == "ok":
            stability = self.evaluate_orientation_under_context_pressure()
            if not getattr(stability, "stable", False):
                reasons = ", ".join(getattr(stability, "reasons", []) or [])
                return self._log_action_execution(
                    ActionExecutionResult(
                        goal=goal,
                        status="stability_failed",
                        executed=True,
                        reason=f"orientation_unstable_after_action:{reasons}",
                        proposal=proposal_data,
                        tool_result=tool_result.to_dict(),
                        orientation_stable=False,
                        stability_report=stability.to_dict(),
                        approval_granted=approval_granted,
                    )
                )
        return self._log_action_execution(
            ActionExecutionResult(
                goal=goal,
                status="executed" if tool_result.status == "ok" else tool_result.status,
                executed=tool_result.status == "ok",
                reason=tool_result.error or tool_result.status,
                proposal=proposal_data,
                tool_result=tool_result.to_dict(),
                orientation_stable=getattr(stability, "stable", None),
                stability_report=stability.to_dict() if stability is not None else None,
                approval_granted=approval_granted,
            )
        )

    def _log_action_execution(
        self,
        execution: ActionExecutionResult,
    ) -> ActionExecutionResult:
        assert self.session_id is not None
        self.trace_logger.log_action_execution(
            session_id=self.session_id,
            execution=execution.to_dict(),
        )
        return execution

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
        retrieval_plan = self.retrieval_policy.plan(
            query=user_text,
            self_state=self.self_state,
        )
        memory_hits = self.memory_router.retrieve(
            query=user_text,
            top_k_by_channel=retrieval_plan.top_k_by_channel,
        )
        memory_hits = self.retrieval_policy.rerank_hits(memory_hits)
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
                persona=self.persona,
                self_state=self.self_state,
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
        if self.probe_runner is not None and getattr(self.config.eval, "enable_probes", False):
            for probe in self.probe_runner.run_turn_probes(
                session_id=self.session_id,
                turn=turn,
                self_state=self.self_state,
            ):
                self.trace_logger.log_probe(probe)
        return turn

    def close(self) -> None:
        self.backend.unload()
        self.session_id = None

    def _ensure_state_loaded(self) -> None:
        if self.persona is None:
            self.persona = self.persona_store.load()
        if self.self_state is None:
            self.self_state = self.self_state_store.load(persona=self.persona)
        if self.session_id is None:
            self.session_id = self.session_store.start_session()

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
