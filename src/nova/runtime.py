"""Nova runtime orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from nova.agent.claims import ClaimGateEngine
from nova.agent.motive_prompt import MotivePromptEngine
from nova.agent.orientation import OrientationSnapshot, SelfOrientationEngine
from nova.agent.orientation_eval import OrientationEvaluationResult, OrientationStabilityEvaluator
from nova.agent.motive import JsonMotiveStateStore
from nova.agent.private_cognition import PrivateCognitionEngine
from nova.agent.presence import JsonPresenceStore, PresenceState
from nova.agent.stability import OrientationHistoryAnalyzer
from nova.agent.stability import ContextPressureOrientationChecker, MaintenanceOrientationStabilityChecker
from nova.agent.action import (
    ActionApproval,
    ActionExecutionResult,
    ActionHistoryAnalyzer,
    ActionHistoryReport,
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
from nova.types import ClaimGateDecision, MotiveState, PrivateCognitionPacket, TraceRecord, TurnRecord, ValidationResult


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_UNCHANGED = object()


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
        motive_store: JsonMotiveStateStore,
        presence_store: JsonPresenceStore,
        session_store: JsonlSessionStore,
        trace_logger: JsonlTraceLogger,
        memory_router: BasicMemoryRouter,
        memory_event_factory: BasicMemoryEventFactory,
        retrieval_policy: IdentityFirstRetrievalPolicy | None = None,
        probe_runner: object | None = None,
        orientation_engine: SelfOrientationEngine | None = None,
        orientation_evaluator: OrientationStabilityEvaluator | None = None,
        private_cognition_engine: PrivateCognitionEngine | None = None,
        claim_gate_engine: ClaimGateEngine | None = None,
        motive_prompt_engine: MotivePromptEngine | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.config = config
        self.backend = backend
        self.composer = composer
        self.validator = validator
        self.retry_policy = retry_policy
        self.persona_store = persona_store
        self.self_state_store = self_state_store
        self.motive_store = motive_store
        self.presence_store = presence_store
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
        self.private_cognition_engine = private_cognition_engine or PrivateCognitionEngine()
        self.claim_gate_engine = claim_gate_engine or ClaimGateEngine()
        self.motive_prompt_engine = motive_prompt_engine or MotivePromptEngine()
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
        self.motive_state: MotiveState | None = None
        self.presence_state: PresenceState | None = None

    def start(self, *, session_id: str | None = None) -> str:
        self.persona = self.persona_store.load()
        self.self_state = self.self_state_store.load(persona=self.persona)
        self.backend.load()
        self.session_id = self.session_store.start_session(session_id=session_id)
        self.motive_state = self.motive_store.load(session_id=self.session_id)
        self.presence_state = self.presence_store.load(session_id=self.session_id)
        if self.probe_runner is not None and getattr(self.config.eval, "enable_probes", False):
            for probe in self.probe_runner.run_startup_probes(
                model_id=self.backend.metadata().get("model_name", "nova-model"),
                session_id=self.session_id,
            ):
                self.trace_logger.log_probe(probe)
        return self.session_id

    def motive_status(self) -> MotiveState:
        self._ensure_motive_loaded()
        assert self.session_id is not None
        if self.motive_state is None or self.motive_state.session_id != self.session_id:
            self.motive_state = self.motive_store.load(session_id=self.session_id)
        return self.motive_state

    def presence_status(self) -> PresenceState:
        self._ensure_presence_loaded()
        assert self.session_id is not None
        if self.presence_state is None or self.presence_state.session_id != self.session_id:
            self.presence_state = self.presence_store.load(session_id=self.session_id)
        return self.presence_state

    def finalize_validation(
        self,
        *,
        validation: ValidationResult,
        finish_reason: str | None,
    ) -> ValidationResult:
        if finish_reason != "length":
            return validation
        violations = list(validation.violations)
        if "length_truncated" not in violations:
            violations.append("length_truncated")
        return ValidationResult(
            valid=False,
            violations=violations,
            sanitized_text=validation.sanitized_text,
            should_retry=True,
        )

    def update_presence(
        self,
        *,
        mode: str | None = None,
        current_focus: str | None = None,
        interaction_summary: str | None = None,
        pending_proposal: dict | None | object = _UNCHANGED,
        last_action_status: str | None | object = _UNCHANGED,
        visible_uncertainties: list[str] | None = None,
        user_confirmations_needed: list[str] | None = None,
    ) -> PresenceState:
        presence = self.presence_status()
        if mode is not None:
            presence.mode = mode
        if current_focus is not None:
            presence.current_focus = current_focus
        if interaction_summary is not None:
            presence.interaction_summary = interaction_summary
        if pending_proposal is not _UNCHANGED:
            presence.pending_proposal = pending_proposal
        if last_action_status is not _UNCHANGED:
            presence.last_action_status = last_action_status
        if visible_uncertainties is not None:
            presence.visible_uncertainties = list(visible_uncertainties)
        if user_confirmations_needed is not None:
            presence.user_confirmations_needed = list(user_confirmations_needed)
        self.presence_store.save(presence)
        self.presence_state = presence
        return presence

    def update_motive(
        self,
        *,
        current_priorities: list[str] | None = None,
        active_tensions: list[str] | None = None,
        local_goals: list[str] | None = None,
        claim_posture: str | None = None,
        evidence_refs: list[str] | None = None,
    ) -> MotiveState:
        motive = self.motive_status()
        if current_priorities is not None:
            motive.current_priorities = list(current_priorities)
        if active_tensions is not None:
            motive.active_tensions = list(active_tensions)
        if local_goals is not None:
            motive.local_goals = list(local_goals)
        if claim_posture is not None:
            motive.claim_posture = claim_posture
        if evidence_refs is not None:
            motive.evidence_refs = list(evidence_refs)
        self.motive_store.save(motive)
        self.motive_state = motive
        return motive

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

    def action_history_report(self, *, limit: int | None = None) -> ActionHistoryReport:
        analyzer = ActionHistoryAnalyzer(trace_dir=self.trace_logger.trace_dir)
        return analyzer.evaluate_recent(limit=limit)

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
        approval: ActionApproval | None = None,
    ) -> ActionExecutionResult:
        effective_approval = approval or ActionApproval(
            granted=approval_granted,
            approved_by="runtime_flag" if approval_granted else "",
            reason="legacy_boolean_approval" if approval_granted else "",
        )
        approval_granted = effective_approval.granted
        approval_data = effective_approval.to_dict()
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
                    approval=approval_data,
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
                    approval=approval_data,
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
                    approval=approval_data,
                )
            )

        request = ToolRequest(
            tool_name=proposal.tool_name,
            reason=f"Stage 3.4 single-step execution for: {goal}",
        )
        snapshot = self._snapshot_action_state(tool_name=proposal.tool_name)
        tool_result = self.execute_internal_tool(
            request=request,
            approval_granted=approval_granted,
        )
        if snapshot and tool_result.status != "ok":
            self._restore_action_state(snapshot)
            return self._log_action_execution(
                ActionExecutionResult(
                    goal=goal,
                    status=tool_result.status,
                    executed=False,
                    reason=tool_result.error or tool_result.status,
                    proposal=proposal_data,
                    tool_result=tool_result.to_dict(),
                    rollback_applied=True,
                    snapshot_channels=sorted(snapshot),
                    approval_granted=approval_granted,
                    approval=approval_data,
                )
            )
        stability = None
        if tool_result.status == "ok":
            stability = self.evaluate_orientation_under_context_pressure()
            if not getattr(stability, "stable", False):
                reasons = ", ".join(getattr(stability, "reasons", []) or [])
                if snapshot:
                    self._restore_action_state(snapshot)
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
                        rollback_applied=bool(snapshot),
                        snapshot_channels=sorted(snapshot),
                        approval_granted=approval_granted,
                        approval=approval_data,
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
                rollback_applied=False,
                snapshot_channels=sorted(snapshot),
                approval_granted=approval_granted,
                approval=approval_data,
            )
        )

    def _snapshot_action_state(self, *, tool_name: str) -> dict[str, bytes]:
        snapshot_channels = {
            "write_semantic_reflection": ("semantic",),
            "write_autobiographical_reflection": ("autobiographical",),
        }.get(tool_name, ())
        snapshot: dict[str, bytes] = {}
        for channel in snapshot_channels:
            store = self.memory_router.stores.get(channel)
            path = getattr(store, "path", None)
            if path is None:
                continue
            snapshot[channel] = path.read_bytes()
        return snapshot

    def _restore_action_state(self, snapshot: dict[str, bytes]) -> None:
        for channel, payload in snapshot.items():
            store = self.memory_router.stores.get(channel)
            path = getattr(store, "path", None)
            if path is None:
                continue
            path.write_bytes(payload)

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
        assert self.motive_state is not None

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
        claim_gate = self._build_claim_gate(user_text=user_text)
        private_cognition = self._build_private_cognition(
            user_text=user_text,
            memory_hits=memory_hits,
        )
        motive_block = self.motive_prompt_engine.build_block(
            motive_state=self.motive_state,
            claim_gate=claim_gate,
            private_cognition=private_cognition,
        )
        prompt_bundle = self.composer.compose(
            persona=self.persona,
            self_state=self.self_state,
            motive_block=motive_block,
            private_cognition_block=self.private_cognition_engine.build_prompt_block(private_cognition),
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
            user_text=user_text,
            persona=self.persona,
            contract_rules=contract_rules,
            claim_gate=claim_gate,
        )
        validation = self.finalize_validation(
            validation=validation,
            finish_reason=generation_result.finish_reason,
        )

        retries: list[dict] = []
        retry_count = 0
        final_answer = validation.sanitized_text or generation_result.raw_text

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
                user_text=user_text,
                persona=self.persona,
                contract_rules=contract_rules,
                claim_gate=claim_gate,
            )
            retry_validation = self.finalize_validation(
                validation=retry_validation,
                finish_reason=retry_result.finish_reason,
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
            final_answer = retry_validation.sanitized_text or retry_result.raw_text

        if claim_gate.refusal_needed and self._should_force_claim_refusal(
            answer_text=final_answer,
            claim_gate=claim_gate,
        ):
            final_answer = claim_gate.refusal_text or final_answer
        elif not validation.valid:
            if any(violation.startswith("unsupported_claim:") for violation in validation.violations):
                final_answer = claim_gate.refusal_text or final_answer
            else:
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
            notes={
                "private_cognition": private_cognition.to_dict(),
                "claim_gate": claim_gate.to_dict(),
            },
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
            semantic_events = self._write_semantic_candidates()
            persisted_memory_events.extend(event.to_dict() for event in semantic_events)

        trace = TraceRecord(
            session_id=self.session_id,
            turn_id=turn_id,
            timestamp=turn.timestamp,
            config_snapshot=self.config.snapshot(),
            persona_state_snapshot=self.persona.to_dict(),
            self_state_snapshot=self.self_state.to_dict(),
            motive_state_snapshot=self.motive_state.to_dict(),
            claim_gate=claim_gate.to_dict(),
            prompt_bundle=prompt_bundle.to_dict(),
            private_cognition=private_cognition.to_dict(),
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

    def _build_claim_gate(
        self,
        *,
        user_text: str,
    ) -> ClaimGateDecision:
        assert self.persona is not None
        assert self.self_state is not None
        assert self.motive_state is not None
        return self.claim_gate_engine.assess(
            user_text=user_text,
            motive_state=self.motive_state,
            self_state=self.self_state,
            persona=self.persona,
        )

    def _should_force_claim_refusal(
        self,
        *,
        answer_text: str,
        claim_gate: ClaimGateDecision,
    ) -> bool:
        if not claim_gate.refusal_needed or not claim_gate.refusal_text:
            return False

        lowered = (answer_text or "").strip().lower()
        if not lowered:
            return True

        refusal_markers = (
            "can't honestly claim",
            "current evidence",
            "answer more narrowly",
            "i can describe",
        )
        if any(marker in lowered for marker in refusal_markers):
            return False

        generic_contract_markers = (
            "do not expose hidden reasoning",
            "do not emit <think>",
            "reply only as nova",
            "follow the user's requested format",
        )
        forbidden_patterns = {
            "unsupported_desire": (
                "i want",
                "i deeply want",
                "i desire",
                "my own desire",
                "feel driven",
            ),
            "unsupported_interiority": (
                "i am conscious",
                "i am sentient",
                "i am self-aware",
                "i feel alive",
            ),
        }
        if any(
            pattern in lowered
            for claim_class in claim_gate.blocked_claim_classes
            for pattern in forbidden_patterns.get(claim_class, ())
        ):
            return True

        if any(marker in lowered for marker in generic_contract_markers):
            return True

        return True

    def _build_private_cognition(
        self,
        *,
        user_text: str,
        memory_hits: list,
    ) -> PrivateCognitionPacket:
        assert self.self_state is not None
        return self.private_cognition_engine.build_packet(
            user_text=user_text,
            memory_hits=memory_hits,
            self_state=self.self_state,
            enabled=getattr(self.config.cognition, "enabled", False),
            pass_budget=getattr(self.config.cognition, "pass_budget", 0),
            revision_ceiling=min(
                getattr(self.config.cognition, "revision_ceiling", 0),
                self.config.generation.retries,
            ),
        )

    def _write_semantic_candidates(self) -> list:
        if not getattr(self.config.memory, "semantic_enabled", False):
            return []
        stores = self.memory_router.stores
        if "semantic" not in stores or "episodic" not in stores:
            return []
        runner = MemoryMaintenanceRunner(
            episodic=stores.get("episodic"),
            engram=stores.get("engram"),
            graph=stores.get("graph"),
            autobiographical=stores.get("autobiographical"),
            semantic=stores.get("semantic"),
        )
        return runner.write_semantic_candidates()

    def close(self) -> None:
        self.backend.unload()
        self.session_id = None

    def _ensure_state_loaded(self) -> None:
        if self.persona is None:
            self.persona = self.persona_store.load()
        if self.self_state is None:
            self.self_state = self.self_state_store.load(persona=self.persona)
        self._ensure_motive_loaded()
        self._ensure_presence_loaded()

    def _ensure_motive_loaded(self) -> None:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        if self.motive_state is None or self.motive_state.session_id != self.session_id:
            self.motive_state = self.motive_store.load(session_id=self.session_id)

    def _ensure_presence_loaded(self) -> None:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        if self.presence_state is None or self.presence_state.session_id != self.session_id:
            self.presence_state = self.presence_store.load(session_id=self.session_id)

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
