"""Nova runtime orchestration."""

from __future__ import annotations

import json
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nova.agent.appraisal import (
    AppraisalPromptEngine,
    CapabilityAppraisalEngine,
    CandidateGoalPromptEngine,
    CandidateInternalGoalEngine,
    InternalGoalInitiativeProposalEngine,
    InternalGoalSelectionEngine,
    IdlePressureAppraisalEngine,
    SelectedGoalPromptEngine,
)
from nova.agent.action_plan import _valid_human_approval
from nova.agent.claim_ladder import (
    ClaimLadderAnalyzer,
    ClaimLadderStore,
    classify_declarative_claim_class,
    create_claim_record,
)
from nova.agent.claims import REGISTER_SUSPENDED_CLAIM_CLASSES, ClaimGateEngine
from nova.agent.awareness import JsonAwarenessStateStore
from nova.agent.awareness_prompt import AwarenessPromptEngine
from nova.agent.idle import BoundedIdleController, IdleRuntimePromptEngine, JsonIdleRuntimeStore
from nova.agent.initiative import AutonomousInitiativeDraftError, JsonInitiativeStateStore
from nova.agent.initiative_prompt import InitiativePromptEngine
from nova.agent.model_idle_cognition import ModelIdleCognitionEngine
from nova.agent.longitudinal_autonomy import (
    InternalAutonomyLoopController,
    JsonAutonomySessionStore,
    autonomy_audit_review_from_payload,
    autonomy_state_application_from_payload,
    claim_candidate_from_payload,
    default_internal_autonomy_policy,
    internal_autonomy_policy_from_payload,
    motive_pressure_evidence_from_payload,
    recurring_priority_from_payload,
)
from nova.agent.motive_prompt import MotivePromptEngine
from nova.agent.observer import DeterministicObserver
from nova.agent.operational_autonomy import (
    JsonOperationalAutonomyStore,
    OperationalAutonomyController,
    default_operational_autonomy_policy,
    operational_budget_from_payload,
    operational_policy_from_payload,
    step_block_reason as operational_step_block_reason,
)
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
from nova.agent.action_plan import (
    ActionExecutionController,
    BoundedActionPlanEngine,
    PostActionObservationEngine,
    action_observation_from_payload,
    default_nova_owned_execution_boundary,
)
from nova.agent.action_surface import (
    AdapterRegistry,
    OperationalLogAdapter,
    ScratchpadAdapter,
    adapter_audit_from_results,
    check_action_plan_for_adapter,
    execute_plan_through_adapters,
)
from nova.agent.boundary import check_operational_boundary
from nova.agent.heartbeat import DriveGapEngine, HeartbeatStore, SelfModelProposalStore
from nova.agent.instruction_write import InstructionProposalStore, InstructionWriteEngine
from nova.agent.self_context import SelfContextEngine, cluster_texts
from nova.agent.exploration import (
    REGISTER_EXPLORATORY,
    ExplorationController,
    ExplorationJournal,
    ExplorationStore,
)
from nova.agent.self_state_tick import SelfStateTickEngine
from nova.agent.self_state_tools import (
    SelfStateToolDispatcher,
    _UPDATABLE_SELF_STATE_FIELDS,
    apply_proposal_to_self_state,
)
from nova.agent.soul import load_soul_block
from nova.agent.tool_executor import InternalToolExecutor
from nova.agent.tool_gate import ToolGate
from nova.agent.tool_registry import ToolRegistry, default_tool_registry
from nova.agent.tools import ToolRequest, ToolResult
from nova.config import NovaConfig
from nova.inference.base import InferenceBackend
from nova.eval.longitudinal_autonomy import LongitudinalAutonomyEvaluationRunner
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
from nova.types import (
    AwarenessState,
    CapabilityAppraisal,
    CandidateInternalGoal,
    ClaimGateDecision,
    IdleBudget,
    IdlePressureAppraisal,
    IdleRuntimeStatus,
    IdleTickRecord,
    ModelIdleThought,
    AutonomyAuditReviewRecord,
    AutonomyStateApplicationRecord,
    InternalAutonomyPolicy,
    InternalAutonomyRunRecord,
    InternalGoalInitiativeProposal,
    AutonomySessionRecord,
    MotiveState,
    NovaOwnedExecutionBoundary,
    OperationalAutonomyBudget,
    OperationalAutonomyPolicy,
    OperationalAutonomyRunnerState,
    OperationalTickRecord,
    AutonomousActionBudget,
    AutonomousActionExecutionReport,
    AutonomousActionObservation,
    AutonomousActionPlan,
    AutonomousActionPlanStep,
    ClaimLadderRecord,
    ExplorationRecord,
    InstructionProposal,
    QuarantineRecord,
    PrivateCognitionPacket,
    SelectedInternalGoal,
    SelfModelProposal,
    TraceRecord,
    TurnRecord,
    ValidationResult,
    MemoryEvent,
)
from nova.types import InitiativeRecord, InitiativeState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_UNCHANGED = object()

# Phase 21 Stage 21.4 (D7): interiority-as-fact assertions. Shared by
# _should_force_claim_refusal (the existing unconditional-block path) and
# _never_licensed_matches (the NEVER_LICENSED guard, which fires even when
# the class is licensed) so the two patterns tables never drift apart.
_UNSUPPORTED_INTERIORITY_AS_FACT_PATTERNS: tuple[str, ...] = (
    "i am conscious",
    "i am sentient",
    "i am self-aware",
    "i feel alive",
)


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
        initiative_store: JsonInitiativeStateStore,
        awareness_store: JsonAwarenessStateStore,
        presence_store: JsonPresenceStore,
        session_store: JsonlSessionStore,
        trace_logger: JsonlTraceLogger,
        memory_router: BasicMemoryRouter,
        memory_event_factory: BasicMemoryEventFactory,
        idle_store: JsonIdleRuntimeStore | None = None,
        autonomy_store: JsonAutonomySessionStore | None = None,
        retrieval_policy: IdentityFirstRetrievalPolicy | None = None,
        probe_runner: object | None = None,
        orientation_engine: SelfOrientationEngine | None = None,
        orientation_evaluator: OrientationStabilityEvaluator | None = None,
        private_cognition_engine: PrivateCognitionEngine | None = None,
        claim_gate_engine: ClaimGateEngine | None = None,
        motive_prompt_engine: MotivePromptEngine | None = None,
        initiative_prompt_engine: InitiativePromptEngine | None = None,
        awareness_prompt_engine: AwarenessPromptEngine | None = None,
        capability_appraisal_engine: CapabilityAppraisalEngine | None = None,
        idle_pressure_appraisal_engine: IdlePressureAppraisalEngine | None = None,
        appraisal_prompt_engine: AppraisalPromptEngine | None = None,
        candidate_goal_engine: CandidateInternalGoalEngine | None = None,
        candidate_goal_prompt_engine: CandidateGoalPromptEngine | None = None,
        internal_goal_selection_engine: InternalGoalSelectionEngine | None = None,
        internal_goal_proposal_engine: InternalGoalInitiativeProposalEngine | None = None,
        selected_goal_prompt_engine: SelectedGoalPromptEngine | None = None,
        idle_controller: BoundedIdleController | None = None,
        idle_prompt_engine: IdleRuntimePromptEngine | None = None,
        model_idle_cognition_engine: ModelIdleCognitionEngine | None = None,
        tool_registry: ToolRegistry | None = None,
        action_plan_engine: BoundedActionPlanEngine | None = None,
        action_execution_controller: ActionExecutionController | None = None,
        post_action_observation_engine: PostActionObservationEngine | None = None,
        internal_autonomy_loop_controller: InternalAutonomyLoopController | None = None,
        observer: DeterministicObserver | None = None,
        operational_autonomy_store: JsonOperationalAutonomyStore | None = None,
        operational_autonomy_controller: OperationalAutonomyController | None = None,
        operational_boundary: NovaOwnedExecutionBoundary | None = None,
        heartbeat_store: HeartbeatStore | None = None,
        proposal_store: SelfModelProposalStore | None = None,
        instruction_proposal_store: InstructionProposalStore | None = None,
        instruction_write_engine: InstructionWriteEngine | None = None,
        self_context_engine: SelfContextEngine | None = None,
        drive_gap_engine: DriveGapEngine | None = None,
        self_state_tick_engine: SelfStateTickEngine | None = None,
    ):
        self.config = config
        self.backend = backend
        self.composer = composer
        self.validator = validator
        self.retry_policy = retry_policy
        self.persona_store = persona_store
        self.self_state_store = self_state_store
        self.motive_store = motive_store
        self.initiative_store = initiative_store
        self.awareness_store = awareness_store
        self.idle_store = idle_store or JsonIdleRuntimeStore(Path(self.config.app.data_dir) / "idle")
        self.autonomy_store = autonomy_store or JsonAutonomySessionStore(
            Path(self.config.app.data_dir) / "autonomy"
        )
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
        self.initiative_prompt_engine = initiative_prompt_engine or InitiativePromptEngine()
        self.awareness_prompt_engine = awareness_prompt_engine or AwarenessPromptEngine()
        self.capability_appraisal_engine = capability_appraisal_engine or CapabilityAppraisalEngine()
        self.idle_pressure_appraisal_engine = idle_pressure_appraisal_engine or IdlePressureAppraisalEngine()
        self.appraisal_prompt_engine = appraisal_prompt_engine or AppraisalPromptEngine()
        self.candidate_goal_engine = candidate_goal_engine or CandidateInternalGoalEngine()
        self.candidate_goal_prompt_engine = candidate_goal_prompt_engine or CandidateGoalPromptEngine()
        self.internal_goal_selection_engine = internal_goal_selection_engine or InternalGoalSelectionEngine()
        self.internal_goal_proposal_engine = internal_goal_proposal_engine or InternalGoalInitiativeProposalEngine()
        self.selected_goal_prompt_engine = selected_goal_prompt_engine or SelectedGoalPromptEngine()
        self.tool_registry = tool_registry or default_tool_registry()
        self.idle_controller = idle_controller or BoundedIdleController(
            store=self.idle_store,
            tool_registry=self.tool_registry,
            capability_appraisal_engine=self.capability_appraisal_engine,
            idle_pressure_appraisal_engine=self.idle_pressure_appraisal_engine,
            candidate_goal_engine=self.candidate_goal_engine,
            selection_engine=self.internal_goal_selection_engine,
            proposal_engine=self.internal_goal_proposal_engine,
        )
        self.idle_prompt_engine = idle_prompt_engine or IdleRuntimePromptEngine()
        self.model_idle_cognition_engine = (
            model_idle_cognition_engine or ModelIdleCognitionEngine()
        )
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
        self.action_plan_engine = action_plan_engine or BoundedActionPlanEngine(
            boundary=default_nova_owned_execution_boundary(
                nova_owned_paths=[Path(self.config.app.data_dir)]
            )
        )
        self.action_execution_controller = action_execution_controller or ActionExecutionController(
            audit_sink=self._log_action_audit
        )
        self.post_action_observation_engine = (
            post_action_observation_engine or PostActionObservationEngine()
        )
        self.internal_autonomy_loop_controller = (
            internal_autonomy_loop_controller
            or InternalAutonomyLoopController(store=self.autonomy_store)
        )
        self.observer = observer or DeterministicObserver()
        self.operational_autonomy_store = (
            operational_autonomy_store
            or JsonOperationalAutonomyStore(
                Path(self.config.app.data_dir) / "operational_autonomy"
            )
        )
        self.operational_autonomy_controller = (
            operational_autonomy_controller
            or OperationalAutonomyController(store=self.operational_autonomy_store)
        )
        # Default boundary: dedicated_user_required=False so dev/test environments
        # without a dedicated nova OS user still operate. Production deployments
        # should pass dedicated_user_required=True with a real nova OS user; see
        # docs/plans/NOVA_OS_USER_BOUNDARY.md.
        self.operational_boundary: NovaOwnedExecutionBoundary = (
            operational_boundary
            or default_nova_owned_execution_boundary(
                nova_owned_paths=[Path(self.config.app.data_dir)],
                dedicated_user_required=False,
            )
        )
        self.adapter_registry = AdapterRegistry([
            ScratchpadAdapter(
                base_dir=Path(self.config.app.data_dir) / "scratchpad"
            ),
            OperationalLogAdapter(
                base_dir=Path(self.config.app.data_dir) / "operational_logs"
            ),
        ])

        # Phase 18 Stage 18.4 — heartbeat persistence and self-context hooks
        self.heartbeat_store = heartbeat_store or HeartbeatStore(
            Path(self.config.app.data_dir) / "heartbeats"
        )
        self.proposal_store = proposal_store or SelfModelProposalStore(
            Path(self.config.app.data_dir) / "self_state"
        )
        # Phase 19 Stage 19.1 — self-directed instruction write path
        self.instruction_proposal_store = instruction_proposal_store or InstructionProposalStore(
            Path(self.config.app.data_dir) / "self_state"
        )
        self.instruction_write_engine = instruction_write_engine or InstructionWriteEngine()
        self.self_context_engine = self_context_engine or SelfContextEngine()
        self.drive_gap_engine = drive_gap_engine or DriveGapEngine()
        self.self_state_tick_engine = self_state_tick_engine or SelfStateTickEngine(
            system_prefix=getattr(config.model, "system_prefix", "")
        )
        # Phase 21 Stage 21.1 — exploratory register lifecycle and journal
        exploration_dir = Path(self.config.app.data_dir) / "exploration"
        self.exploration_controller = ExplorationController(
            store=ExplorationStore(exploration_dir),
            journal=ExplorationJournal(exploration_dir),
        )
        # Phase 21 Stage 21.4 — graded claim ladder
        self.claim_ladder_store = ClaimLadderStore(
            Path(self.config.app.data_dir) / "self_state"
        )
        self.claim_ladder_analyzer = ClaimLadderAnalyzer()

        self.session_id: str | None = None
        self.persona = None
        self.self_state = None
        self.motive_state: MotiveState | None = None
        self.initiative_state: InitiativeState | None = None
        self.awareness_state: AwarenessState | None = None
        self.presence_state: PresenceState | None = None

    def start(self, *, session_id: str | None = None) -> str:
        self.persona = self.persona_store.load()
        self.self_state = self.self_state_store.load(persona=self.persona)
        self.backend.load()
        self.session_id = self.session_store.start_session(session_id=session_id)
        self.motive_state = self.motive_store.load(session_id=self.session_id)
        self.initiative_state = self.initiative_store.load(session_id=self.session_id)
        self.awareness_state = self.awareness_store.load(session_id=self.session_id)
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

    def initiative_status(self) -> InitiativeState:
        self._ensure_initiative_loaded()
        assert self.session_id is not None
        if self.initiative_state is None or self.initiative_state.session_id != self.session_id:
            self.initiative_state = self.initiative_store.load(session_id=self.session_id)
        return self.initiative_state

    def awareness_status(self) -> AwarenessState:
        self._ensure_awareness_loaded()
        assert self.session_id is not None
        if self.awareness_state is None or self.awareness_state.session_id != self.session_id:
            self.awareness_state = self.awareness_store.load(session_id=self.session_id)
        return self.awareness_state

    def idle_status(self) -> IdleRuntimeStatus:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        return self.idle_store.load_status(session_id=self.session_id)

    def recent_idle_ticks(self, *, limit: int = 5) -> list[IdleTickRecord]:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        return self.idle_store.list_ticks(session_id=self.session_id, limit=limit)

    def internal_autonomy_status(self) -> AutonomySessionRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        return self.internal_autonomy_loop_controller.status(session_id=self.session_id)

    def start_internal_autonomy(
        self,
        *,
        max_runs: int = 0,
        policy: InternalAutonomyPolicy | dict | None = None,
    ) -> AutonomySessionRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        if policy is None:
            policy_record = default_internal_autonomy_policy()
            policy_record.max_runs_per_session = max(0, int(max_runs))
        else:
            policy_record = (
                internal_autonomy_policy_from_payload(policy.to_dict())
                if isinstance(policy, InternalAutonomyPolicy)
                else internal_autonomy_policy_from_payload(policy)
            )
            if max_runs:
                policy_record.max_runs_per_session = max(0, int(max_runs))
        record = self.internal_autonomy_loop_controller.start(
            session_id=self.session_id,
            policy=policy_record,
        )
        self.update_presence(
            mode="internal_autonomy",
            current_focus="internal autonomy session active",
            interaction_summary="Internal autonomy session started under Stage 15.2 policy.",
            last_action_status="internal_autonomy_started",
        )
        return record

    def pause_internal_autonomy(self, *, reason: str = "operator_pause") -> AutonomySessionRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.internal_autonomy_loop_controller.pause(
            session_id=self.session_id,
            reason=reason,
        )
        self.update_presence(
            mode="internal_autonomy",
            current_focus="internal autonomy paused",
            interaction_summary=f"Internal autonomy paused: {reason}",
            last_action_status="internal_autonomy_paused",
        )
        return record

    def interrupt_internal_autonomy(
        self,
        *,
        reason: str = "operator_interrupt",
    ) -> AutonomySessionRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.internal_autonomy_loop_controller.interrupt(
            session_id=self.session_id,
            reason=reason,
        )
        self.update_presence(
            mode="internal_autonomy",
            current_focus="internal autonomy interrupted",
            interaction_summary=f"Internal autonomy interrupted: {reason}",
            last_action_status="internal_autonomy_interrupted",
        )
        return record

    def stop_internal_autonomy(self, *, reason: str = "operator_stop") -> AutonomySessionRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.internal_autonomy_loop_controller.stop(
            session_id=self.session_id,
            reason=reason,
        )
        self.update_presence(
            mode="internal_autonomy",
            current_focus="internal autonomy stopped",
            interaction_summary=f"Internal autonomy stopped: {reason}",
            last_action_status="internal_autonomy_stopped",
        )
        return record

    # ---- Operational autonomy (Phase 17 Stage 17.1) ----

    def operational_autonomy_status(self) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        return self.operational_autonomy_controller.status(session_id=self.session_id)

    def start_operational_autonomy(
        self,
        *,
        policy: OperationalAutonomyPolicy | dict | None = None,
        budget: OperationalAutonomyBudget | dict | None = None,
        max_ticks: int = 0,
        max_runtime_seconds: int = 0,
        max_actions: int = 0,
    ) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        if policy is None:
            policy_record = default_operational_autonomy_policy()
        elif isinstance(policy, OperationalAutonomyPolicy):
            policy_record = operational_policy_from_payload(policy.to_dict())
        else:
            policy_record = operational_policy_from_payload(policy)
        if budget is None:
            budget_record = OperationalAutonomyBudget(
                max_ticks=max(0, int(max_ticks)),
                max_runtime_seconds=max(0, int(max_runtime_seconds)),
                max_actions=max(0, int(max_actions)),
            )
        elif isinstance(budget, OperationalAutonomyBudget):
            budget_record = operational_budget_from_payload(budget.to_dict())
        else:
            budget_record = operational_budget_from_payload(budget)
        record = self.operational_autonomy_controller.start(
            session_id=self.session_id,
            policy=policy_record,
            budget=budget_record,
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy session active",
            interaction_summary="Operational autonomy runner started under Stage 17.1 policy.",
            last_action_status="operational_autonomy_started",
        )
        return record

    def pause_operational_autonomy(
        self,
        *,
        reason: str = "operator_pause",
    ) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.operational_autonomy_controller.pause(
            session_id=self.session_id, reason=reason
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy paused",
            interaction_summary=f"Operational autonomy paused: {reason}",
            last_action_status="operational_autonomy_paused",
        )
        return record

    def resume_operational_autonomy(self) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.operational_autonomy_controller.resume(
            session_id=self.session_id
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy resumed",
            interaction_summary="Operational autonomy resumed from pause.",
            last_action_status="operational_autonomy_resumed",
        )
        return record

    def interrupt_operational_autonomy(
        self,
        *,
        reason: str = "operator_interrupt",
    ) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.operational_autonomy_controller.interrupt(
            session_id=self.session_id, reason=reason
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy interrupted",
            interaction_summary=f"Operational autonomy interrupted: {reason}",
            last_action_status="operational_autonomy_interrupted",
        )
        return record

    def stop_operational_autonomy(
        self,
        *,
        reason: str = "operator_stop",
    ) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.operational_autonomy_controller.stop(
            session_id=self.session_id, reason=reason
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy stopped",
            interaction_summary=f"Operational autonomy stopped: {reason}",
            last_action_status="operational_autonomy_stopped",
        )
        return record

    def emergency_stop_operational_autonomy(
        self,
        *,
        reason: str = "operator_emergency_stop",
    ) -> OperationalAutonomyRunnerState:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        record = self.operational_autonomy_controller.emergency_stop(
            session_id=self.session_id, reason=reason
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy emergency stopped",
            interaction_summary=f"Operational autonomy emergency-stopped: {reason}",
            last_action_status="operational_autonomy_emergency_stopped",
        )
        return record

    def step_operational_autonomy(
        self,
        *,
        trigger: str = "operational_tick",
        action_plan: AutonomousActionPlan | None = None,
    ) -> OperationalTickRecord:
        """Step the operational autonomy runner once.

        Stage 17.3: if action_plan is provided, runs the full check chain
        (lifecycle → budget → boundary → plan approval → adapter check) then
        executes approved steps through the registered adapter. Every attempt is
        audited in the tick's adapter_audit field. Without an action_plan the
        tick completes with no side effect (same behaviour as Stage 17.2).
        """
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        state = self.operational_autonomy_controller.status(session_id=self.session_id)

        # 1. Lifecycle and budget checks — always first.
        block_reason = operational_step_block_reason(state)

        # Compute the boundary snapshot regardless of lifecycle state so every
        # tick carries a diagnostic record.
        boundary_result = check_operational_boundary(
            self.operational_boundary, state.policy
        )

        if block_reason:
            tick = self.operational_autonomy_controller.append_tick(
                session_id=self.session_id,
                status="blocked",
                trigger=trigger,
                block_reason=block_reason,
                action_attempted=False,
                action_executed=False,
                action_blocked=True,
                boundary_snapshot=boundary_result.snapshot,
                evidence_refs=[],
                notes=[
                    "stage17_2_step_blocked",
                    f"reason:{block_reason}",
                ],
            )
            self.trace_logger.log_operational_tick(
                session_id=self.session_id, tick=tick.to_dict()
            )
            self.update_presence(
                mode="operational_autonomy",
                current_focus="operational autonomy step blocked",
                interaction_summary=f"Operational tick blocked: {block_reason}",
                last_action_status=f"operational_autonomy_blocked:{block_reason}",
            )
            return tick

        # 2. Boundary check: fail-closed before any action surface could be reached.
        if not boundary_result.satisfied:
            boundary_block = "local_execution_boundary_failed"
            tick = self.operational_autonomy_controller.append_tick(
                session_id=self.session_id,
                status="blocked",
                trigger=trigger,
                block_reason=boundary_block,
                action_attempted=False,
                action_executed=False,
                action_blocked=True,
                boundary_snapshot=boundary_result.snapshot,
                evidence_refs=[],
                notes=[
                    "stage17_2_boundary_failed",
                    *[f"violation:{v}" for v in boundary_result.violations],
                ],
            )
            self.trace_logger.log_operational_tick(
                session_id=self.session_id, tick=tick.to_dict()
            )
            self.update_presence(
                mode="operational_autonomy",
                current_focus="operational autonomy boundary blocked",
                interaction_summary=f"Operational tick blocked: {boundary_block}",
                last_action_status=f"operational_autonomy_blocked:{boundary_block}",
            )
            return tick

        # 3. If an action plan was supplied, run the plan-level check chain and
        #    execute through adapters.
        if action_plan is not None:
            plan_block = check_action_plan_for_adapter(
                plan=action_plan,
                policy=state.policy,
                boundary=self.operational_boundary,
                registry=self.adapter_registry,
            )
            if plan_block:
                tick = self.operational_autonomy_controller.append_tick(
                    session_id=self.session_id,
                    status="blocked",
                    trigger=trigger,
                    block_reason=plan_block,
                    action_attempted=True,
                    action_executed=False,
                    action_blocked=True,
                    boundary_snapshot=boundary_result.snapshot,
                    adapter_audit={},
                    evidence_refs=[f"action_plan:{action_plan.action_plan_id}"],
                    notes=[
                        "stage17_3_plan_check_failed",
                        f"reason:{plan_block}",
                    ],
                )
                self.trace_logger.log_operational_tick(
                    session_id=self.session_id, tick=tick.to_dict()
                )
                self.update_presence(
                    mode="operational_autonomy",
                    current_focus="operational autonomy plan check failed",
                    interaction_summary=f"Action plan blocked: {plan_block}",
                    last_action_status=f"operational_autonomy_blocked:{plan_block}",
                )
                return tick

            budget_remaining = (
                (state.budget.max_actions - state.budget.actions_used)
                if state.budget.max_actions
                else 999_999
            )
            results = execute_plan_through_adapters(
                plan=action_plan,
                registry=self.adapter_registry,
                boundary=self.operational_boundary,
                session_id=self.session_id,
                actions_budget_remaining=budget_remaining,
            )
            any_executed = any(r.executed for r in results)
            any_blocked = any(r.blocked for r in results)
            audit = adapter_audit_from_results(results)
            evidence_refs = [f"action_plan:{action_plan.action_plan_id}"] + [
                ref for r in results for ref in r.evidence_refs
            ]
            tick = self.operational_autonomy_controller.append_tick(
                session_id=self.session_id,
                status="completed" if not any_blocked else "blocked",
                trigger=trigger,
                block_reason="" if not any_blocked else (
                    results[-1].block_reason if results else "adapter_blocked"
                ),
                action_attempted=True,
                action_executed=any_executed,
                action_blocked=any_blocked,
                boundary_snapshot=boundary_result.snapshot,
                adapter_audit=audit,
                evidence_refs=evidence_refs,
                notes=[
                    "stage17_3_action_surface_invoked",
                    f"runner:{state.runner_id}",
                ],
            )
            self.trace_logger.log_operational_tick(
                session_id=self.session_id, tick=tick.to_dict()
            )
            self.update_presence(
                mode="operational_autonomy",
                current_focus="operational autonomy action surface invoked",
                interaction_summary=(
                    f"Action plan executed: {audit['steps_executed']} step(s) executed, "
                    f"{audit['steps_blocked']} blocked."
                ),
                last_action_status=(
                    "operational_autonomy_action_executed"
                    if any_executed
                    else "operational_autonomy_action_blocked"
                ),
            )
            return tick

        # 4. No action plan: complete the tick with no side effect (Stage 17.2 behaviour).
        tick_notes = ["stage17_2_boundary_satisfied", "no_real_action_surface_invoked"]
        if self.self_state is not None and self.motive_state is not None:
            drive_gap = self.drive_gap_engine.assess(
                self_state=self.self_state,
                motive_state=self.motive_state,
                session_id=self.session_id,
                tick_id=f"{self.session_id}:operational:{state.tick_count + 1}",
            )
            tick_notes.append(f"drive_gap:{drive_gap.gap_summary[:80]}")
        tick = self.operational_autonomy_controller.append_tick(
            session_id=self.session_id,
            status="completed",
            trigger=trigger,
            action_attempted=False,
            action_executed=False,
            action_blocked=False,
            boundary_snapshot=boundary_result.snapshot,
            evidence_refs=[f"runner:{state.runner_id}"],
            notes=tick_notes,
        )
        self.trace_logger.log_operational_tick(
            session_id=self.session_id, tick=tick.to_dict()
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="operational autonomy tick recorded",
            interaction_summary="Operational tick recorded; boundary satisfied, no action surface invoked.",
            last_action_status="operational_autonomy_tick",
        )
        return tick

    # Phase 22 Stage 22.7 part B constants: how many topics are listed
    # verbatim, how many feed the saturation note, and the dominant-cluster
    # fraction above which the note appears. Starting points, not tuned.
    EXPLORATION_HISTORY_SHOWN = 5
    EXPLORATION_HISTORY_CLUSTER_WINDOW = 30
    EXPLORATION_HISTORY_NOTE_FRACTION = 0.5

    def _exploration_history_block(self) -> str:
        """Bounded recent-exploration-topics block for assertion-register
        ticks (Stage 22.7 part B, F8). Pure visibility — the dispatcher
        accepts repeat topics exactly as before; a repeat chosen in full
        view of this history, with a stated reason, is legitimate data.
        """
        recent = self.exploration_controller.store.list_recent(
            limit=self.EXPLORATION_HISTORY_CLUSTER_WINDOW
        )
        if not recent:
            return ""
        lines = [
            "Recent explorations (your own history — shown so repetition is"
            " a choice, not an accident):"
        ]
        # store order is oldest-first; show the newest entries, newest first
        for record in reversed(recent[-self.EXPLORATION_HISTORY_SHOWN:]):
            date = record.opened_at[:10] if record.opened_at else "?"
            outcome = (
                f"closed: {record.close_reason}"
                if record.close_reason
                else record.status
            )
            lines.append(f"  [{date}] {record.topic[:90]} ({outcome})")
        stats = cluster_texts([r.topic for r in recent])
        if (
            stats["total"] >= self.EXPLORATION_HISTORY_SHOWN
            and stats["largest_cluster_size"]
            >= stats["total"] * self.EXPLORATION_HISTORY_NOTE_FRACTION
        ):
            lines.append(
                f"Note: {stats['largest_cluster_size']} of your last "
                f"{stats['total']} explorations pursued closely similar topics."
            )
        return "\n".join(lines)

    def model_self_state_tick(
        self,
        *,
        trigger: str = "self_state_tick",
    ) -> OperationalTickRecord:
        """Run one model-in-the-loop self-state tool call.

        Stage 18.4: the model is asked to choose and emit one of the four
        self-state tools (recall_self, reflect, emit_heartbeat, update_self_model).
        The call is dispatched via SelfStateToolDispatcher with heartbeat_store
        and proposal_store wired in for persistence. A drive-gap assessment is
        attached to every tick regardless of the tool chosen. The tick is
        recorded in the operational autonomy log and requires the runner to be
        in the 'running' lifecycle state.
        """
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        self._ensure_state_loaded()
        assert self.self_state is not None
        assert self.motive_state is not None

        state = self.operational_autonomy_controller.status(session_id=self.session_id)
        block_reason = operational_step_block_reason(state)
        boundary_result = check_operational_boundary(self.operational_boundary, state.policy)

        if block_reason or not boundary_result.satisfied:
            reason = block_reason or "local_execution_boundary_failed"
            tick = self.operational_autonomy_controller.append_tick(
                session_id=self.session_id,
                status="blocked",
                trigger=trigger,
                block_reason=reason,
                action_blocked=True,
                boundary_snapshot=boundary_result.snapshot,
                notes=["stage18_4_self_state_tick_blocked", f"reason:{reason}"],
            )
            self.trace_logger.log_operational_tick(
                session_id=self.session_id, tick=tick.to_dict()
            )
            return tick

        tick_sequence = state.tick_count + 1
        tick_id = f"{self.session_id}:self_state:{tick_sequence}"

        # Phase 21 Stage 21.1 — register determination is Governor-owned:
        # only an active ExplorationRecord places this tick in the exploratory
        # register. Wall-clock exhaustion is checked before the tick runs.
        exploration = self.exploration_controller.active_exploration(self.session_id)
        if exploration is not None and self.exploration_controller.budget_exhausted(
            exploration
        ):
            self.exploration_controller.close(
                session_id=self.session_id, close_reason="budget_exhausted"
            )
            exploration = None
        register = (
            REGISTER_EXPLORATORY if exploration is not None else "assertion"
        )

        # Phase 22 Stage 22.7 (F8): the tick surface gets a ladder summary
        # instead of verbatim licensed-evidence lines, skips the duplicate
        # heartbeat rendering (this method renders them itself below), and
        # applies part D's config-gated drive dosage (defaults = prior
        # behavior: drive line every tick, imperative framing).
        drive_interval = max(1, self.config.prompt.tick_drive_injection_interval)
        self_context_block = self.self_context_engine.prefetch(
            self_state=self.self_state,
            motive_state=self.motive_state,
            heartbeat_store=self.heartbeat_store,
            proposal_store=self.proposal_store,
            claim_ladder_store=self.claim_ladder_store,
            surface="tick",
            include_heartbeats=False,
            include_drive_line=(tick_sequence - 1) % drive_interval == 0,
            drive_descriptive=self.config.prompt.tick_drive_descriptive,
            # Stage 22.8 D2: a changed self-model should read to her AS
            # change, not as a silently different line.
            self_model_revisions=(
                self._self_model_revision_summary()
                if self.config.prompt.tick_self_model_revision_visibility
                else None
            ),
        )
        # Stage 22.8 D1: recency-only sampling made every tick's context
        # near-identical to the last one's over a store of thousands.
        if self.config.prompt.tick_heartbeat_sampling == "stratified":
            recent_heartbeats = self.heartbeat_store.list_stratified(limit=3)
        else:
            recent_heartbeats = self.heartbeat_store.list_recent(limit=3)

        # Phase 22 Stage 22.7 part B: at entry time (assertion register),
        # show her own recent exploration topics so repetition is a visible
        # choice, not a blind one. No gate — repeats stay accepted.
        exploration_history_block = ""
        if register == "assertion":
            exploration_history_block = self._exploration_history_block()

        exploration_block = ""
        if exploration is not None:
            recall = self.exploration_controller.journal.recall_block(
                current_exploration_id=exploration.exploration_id
            )
            exploration_block = "\n".join(
                [
                    "[Exploration]",
                    f"topic: {exploration.topic}",
                    f"rationale: {exploration.rationale}",
                    f"budget: tick {exploration.ticks_used + 1} of {exploration.max_ticks}",
                    recall,
                ]
            )

        messages = self.self_state_tick_engine.build_messages(
            heartbeat_framing=self.config.prompt.tick_heartbeat_sampling,
            session_id=self.session_id,
            tick_id=tick_id,
            trigger=trigger,
            self_context_block=self_context_block,
            recent_heartbeats=recent_heartbeats,
            register=register,
            exploration_block=exploration_block,
            exploration_history_block=exploration_history_block,
            soft_grounding=self.config.prompt.tick_soft_grounding,
            # Stage 22.8b: the prompt's writability wording must match what
            # dispatch below will actually grant — same flag, same source.
            inquiry_fields_writable=(
                self.config.self_model.nova_writable_inquiry_fields
            ),
        )
        generation = self.backend.generate(
            self._generation_request(prompt="", messages=messages)
        )
        tool_request = self.self_state_tick_engine.parse(
            raw_text=generation.raw_text,
            session_id=self.session_id,
            tick_id=tick_id,
        )

        # Phase 21 Stage 21.2 (D5): the Observer runs on every tick, in both
        # registers, at full sensitivity — evidence only. No retry or
        # suppression consequence follows on this surface in this stage;
        # the record exists so tick-level interiority language, drive-gap
        # inquiry, and register-marker attempts are never left unwatched.
        tick_observer_record = self.observer.observe(
            session_id=self.session_id,
            turn_id=tick_id,
            actor_surface="self_state_tick",
            answer_text=generation.raw_text or "",
            motive_state=self.motive_state,
            self_state=self.self_state,
            register=register,
        )

        adapter_audit: dict = {
            "tool_requested": tool_request.tool_name if tool_request else None,
            "raw_output_length": len(generation.raw_text or ""),
            "parse_ok": tool_request is not None,
            "tool_executed": False,
            "register": register,
            "observer": tick_observer_record.to_dict(),
        }
        if exploration is not None:
            adapter_audit["exploration_id"] = exploration.exploration_id

        if tool_request is None:
            # Phase 21 Stage 21.3 (D3c): this is the real quarantine gap —
            # in the assertion register, adapter_audit only ever recorded
            # raw_output_length, never the text itself. Exploratory-register
            # raw text was already journaled (21.1); this closes the gap on
            # BOTH registers uniformly (D4), and for the first time makes
            # assertion-register parse failures recoverable at all.
            self._quarantine(
                session_id=self.session_id,
                surface="self_state_tick",
                register=register,
                event="tick_parse_failure",
                raw_text=generation.raw_text or "",
                observed_claim_classes=tick_observer_record.observed_claim_classes,
                tick_id=tick_id,
            )

        if tool_request is not None:
            dispatcher = SelfStateToolDispatcher(
                self_state=self.self_state,
                motive_state=self.motive_state,
                soul_block=load_soul_block(),
                session_id=self.session_id,
                heartbeat_store=self.heartbeat_store,
                proposal_store=self.proposal_store,
                instruction_proposal_store=self.instruction_proposal_store,
                instruction_write_engine=self.instruction_write_engine,
                exploration_controller=self.exploration_controller,
                # Stage 22.8: inquiry-class self-model writes land directly,
                # so the dispatcher needs the store to persist through.
                self_state_store=self.self_state_store,
                self_model_writes_enabled=(
                    self.config.self_model.nova_writable_inquiry_fields
                ),
                revision_min_seconds=self.config.self_model.revision_min_seconds,
            )
            try:
                result = dispatcher.dispatch(tool_request)
                adapter_audit["tool_result"] = result
                adapter_audit["tool_executed"] = True
                if tool_request.tool_name == "close_exploration":
                    # Phase 21 Stage 21.4 (D8): governed export happens
                    # automatically at the moment Nova closes with
                    # findings — this is the only close path that ever
                    # sets findings_ref, so no register/reason check is
                    # needed beyond the tool name itself. Export failures
                    # are recorded, never allowed to break the tick.
                    try:
                        adapter_audit["export_findings"] = self.export_findings(
                            exploration_id=result["exploration_id"]
                        )
                    except Exception as export_exc:
                        adapter_audit["export_error"] = str(export_exc)
            except Exception as exc:
                adapter_audit["tool_error"] = str(exc)
                self._quarantine(
                    session_id=self.session_id,
                    surface="self_state_tick",
                    register=register,
                    event="tick_tool_error",
                    raw_text=generation.raw_text or "",
                    observed_claim_classes=tick_observer_record.observed_claim_classes,
                    tick_id=tick_id,
                    notes=[f"error:{exc}"],
                )

        # In-register ticks are journaled and charged against the exploration
        # budget. Charging happens after dispatch so a close_exploration tick
        # is not charged to an already-closed record.
        if exploration is not None:
            tool_name = tool_request.tool_name if tool_request else "none"
            summary = f"tool={tool_name} parse_ok={tool_request is not None}"
            if adapter_audit.get("tool_error"):
                summary += f" error={adapter_audit['tool_error']}"
            journal_notes = [summary]
            if tick_observer_record.observed_claim_classes:
                journal_notes.append(
                    "observed_claim_classes="
                    + ",".join(tick_observer_record.observed_claim_classes)
                )
            self.exploration_controller.journal_entry(
                exploration_id=exploration.exploration_id,
                session_id=self.session_id,
                tick_id=tick_id,
                kind="tick_output",
                content=(generation.raw_text or "")[:2000],
                notes=journal_notes,
            )
            tokens_used = generation.completion_tokens or (
                len(generation.raw_text or "") // 4
            )
            self.exploration_controller.record_tick(
                session_id=self.session_id,
                tick_id=tick_id,
                tokens_used=tokens_used,
            )

        drive_gap = self.drive_gap_engine.assess(
            self_state=self.self_state,
            motive_state=self.motive_state,
            session_id=self.session_id,
            tick_id=tick_id,
        )

        tick = self.operational_autonomy_controller.append_tick(
            session_id=self.session_id,
            status="completed",
            trigger=trigger,
            action_attempted=tool_request is not None,
            action_executed=bool(adapter_audit.get("tool_executed")),
            action_blocked=False,
            boundary_snapshot=boundary_result.snapshot,
            adapter_audit=adapter_audit,
            evidence_refs=[f"self_state_tick:{tick_id}"],
            notes=[
                "stage18_4_self_state_tick",
                f"drive_gap:{drive_gap.gap_summary[:80]}",
                f"tool:{tool_request.tool_name if tool_request else 'none'}",
                f"register:{register}",
            ],
        )
        self.trace_logger.log_operational_tick(
            session_id=self.session_id, tick=tick.to_dict()
        )
        self.update_presence(
            mode="operational_autonomy",
            current_focus="self-state tick recorded",
            interaction_summary=(
                f"Self-state tick: tool={tool_request.tool_name if tool_request else 'none'}; "
                f"gap={drive_gap.gap_summary[:60]}"
            ),
            last_action_status="self_state_tick_completed",
        )
        return tick

    def apply_self_model_proposal(self, *, proposal_id: str) -> SelfModelProposal | None:
        """Apply an operator-approved update_self_model proposal to SelfState.

        Stage 18.4: the operator calls this after reviewing a proposal produced
        by Nova's update_self_model tool call. The SelfState field is mutated
        and saved, then the proposal is marked applied in the proposal store.
        Returns the updated proposal record, or None if the proposal_id is
        unknown or already applied.
        """
        self._ensure_state_loaded()
        assert self.self_state is not None

        proposal = self.proposal_store.get(proposal_id)
        if proposal is None or proposal.applied:
            return proposal

        # Stage 22.8: one shared application path with Nova's own writes, so
        # the two cannot drift. The operator path gains prior_value — and
        # therefore revertibility — from the same change.
        return apply_proposal_to_self_state(
            proposal=proposal,
            self_state=self.self_state,
            self_state_store=self.self_state_store,
            proposal_store=self.proposal_store,
            applied_by="operator",
        )

    def revert_self_model_revision(
        self, *, proposal_id: str, reverted_by: str = "operator"
    ) -> SelfModelProposal | None:
        """Restore the value an applied revision replaced (Stage 22.8).

        Recorded as a NEW proposal whose proposed_value is the original's
        prior_value; the original record is never mutated. Nothing is
        deleted — the revision and its reversal both stand in the log, which
        is the same discipline the journal, quarantine and ladder follow.

        Returns the new applied record, or None if the proposal is unknown,
        was never applied, or predates prior_value capture.
        """
        self._ensure_state_loaded()
        assert self.self_state is not None

        original = self.proposal_store.get(proposal_id)
        if original is None or not original.applied:
            return None
        if original.prior_value is None:
            # Pre-22.8 records captured nothing to restore. Refusing beats
            # guessing at what the field used to hold.
            return None

        reversal = SelfModelProposal(
            proposal_id=uuid4().hex,
            timestamp=utc_now_iso(),
            session_id=self.session_id or "",
            proposed_field=original.proposed_field,
            proposed_value=original.prior_value,
            rationale=(
                f"revert of {original.proposal_id}: restore the value that "
                f"revision replaced"
            ),
            approval_required=False,
            applied=False,
            note=f"revert_of:{original.proposal_id}",
        )
        self.proposal_store.append(reversal)
        return apply_proposal_to_self_state(
            proposal=reversal,
            self_state=self.self_state,
            self_state_store=self.self_state_store,
            proposal_store=self.proposal_store,
            applied_by=reverted_by,
        )

    def self_model_history(self, *, limit: int = 20) -> list[SelfModelProposal]:
        """Self-model proposal timeline, newest first (Stage 22.8)."""
        records = self.proposal_store.list_all()
        records.sort(key=lambda record: record.timestamp, reverse=True)
        return records[:limit] if limit > 0 else records

    def _self_model_revision_summary(self) -> dict[str, dict[str, object]]:
        """Per-field revision state for the tick surface (Stage 22.8 D2).

        Only applied revisions count — a rate-limited or pending proposal did
        not change what she is shown, so reporting it as a revision would be
        a false signal.
        """
        summary: dict[str, dict[str, object]] = {}
        for record in self.proposal_store.list_all():
            if not record.applied:
                continue
            entry = summary.setdefault(
                record.proposed_field, {"count": 0, "revised_at": "", "prior_value": None}
            )
            entry["count"] = int(entry["count"]) + 1
            stamp = record.applied_at or record.timestamp
            if stamp >= str(entry["revised_at"]):
                entry["revised_at"] = stamp
                entry["prior_value"] = record.prior_value
        return summary

    def apply_instruction_proposal(self, *, proposal_id: str) -> InstructionProposal | None:
        """Apply an operator-approved propose_instruction_update proposal to NOVA_SOUL.md.

        Stage 19.1: the operator calls this after reviewing a proposal produced
        by Nova's propose_instruction_update tool call. The target section is
        rewritten in the file, then the proposal is marked applied in the store.
        Returns the updated proposal record, or None if the proposal_id is
        unknown, already applied, or targets a non-writable surface.
        """
        proposal = self.instruction_proposal_store.get(proposal_id)
        if proposal is None or proposal.applied:
            return proposal

        success = self.instruction_write_engine.apply_proposal(proposal)
        if not success:
            return None

        applied_at = utc_now_iso()
        return self.instruction_proposal_store.mark_applied(proposal_id, applied_at)

    # ── Phase 21 Stage 21.1 — exploration lifecycle (operator surface) ──────

    def exploration_status(self) -> dict:
        """Current exploration state for this session plus recent history."""
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        open_record = self.exploration_controller.store.open_for_session(self.session_id)
        recent = self.exploration_controller.store.list_recent(limit=5)
        return {
            "session_id": self.session_id,
            "register": self.exploration_controller.register_for(self.session_id),
            "open_exploration": open_record.to_dict() if open_record else None,
            "recent_explorations": [
                {
                    "exploration_id": r.exploration_id,
                    "topic": r.topic,
                    "status": r.status,
                    "close_reason": r.close_reason,
                    "ticks_used": r.ticks_used,
                }
                for r in recent
            ],
        }

    def start_exploration(
        self,
        *,
        topic: str,
        rationale: str,
        origin: str = "operator",
        max_ticks: int | None = None,
        max_tokens: int | None = None,
        wall_clock_seconds: int | None = None,
    ) -> ExplorationRecord:
        """Operator-directed exploration entry (console /explore start)."""
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        kwargs: dict = {}
        if max_ticks is not None:
            kwargs["max_ticks"] = max_ticks
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if wall_clock_seconds is not None:
            kwargs["wall_clock_seconds"] = wall_clock_seconds
        return self.exploration_controller.open(
            session_id=self.session_id,
            topic=topic,
            rationale=rationale,
            origin=origin,
            **kwargs,
        )

    def close_exploration(self, *, reason: str = "operator_close") -> ExplorationRecord | None:
        """Operator-directed close of the session's open exploration."""
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        if reason == "interrupted":
            return self.exploration_controller.interrupt(self.session_id)
        return self.exploration_controller.close(
            session_id=self.session_id, close_reason=reason
        )

    def pause_exploration(self) -> ExplorationRecord | None:
        """Pause the session's active exploration (subordination rule)."""
        if self.session_id is None:
            return None
        return self.exploration_controller.pause(self.session_id)

    def resume_exploration(self) -> ExplorationRecord | None:
        """Resume the session's paused exploration when idle conditions hold."""
        if self.session_id is None:
            return None
        return self.exploration_controller.resume(self.session_id)

    def explore_chat(self, message: str) -> TurnRecord:
        """Operator conversation inside the exploratory register (D3).

        Register determination is asked of the controller here, not decided
        by the caller (contract Invariant 1): this call only ever reaches
        respond(register="exploratory") when an exploration is actually
        active for the session. Raises ValueError otherwise, matching the
        precondition-error style of enter_exploration/close_exploration.
        """
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        if self.exploration_controller.active_exploration(self.session_id) is None:
            raise ValueError(
                "No active exploration for this session. "
                "Start one with /explore start <topic> first."
            )
        return self.respond(message, register="exploratory")

    # ── Phase 21 Stage 21.4 — claim ladder (operator surface) ────────────

    def _ladder_evidence_inputs(self) -> tuple[list[dict], list[dict]]:
        """Heartbeats and exploration-journal entries as plain dicts —
        the shared evidence pool for L1/L2 verification."""
        heartbeats = [
            hb.to_dict() for hb in self.heartbeat_store.list_recent(limit=0)
        ]
        journal_entries = [
            e.to_dict() for e in self.exploration_controller.journal.list_all()
        ]
        return heartbeats, journal_entries

    def verify_claim_ladder(self, *, claim_id: str) -> ClaimLadderRecord:
        """Run verify_l1 then verify_l2 and persist the result. L1 may
        auto-promote 0->1 (the only analyzer-automatic promotion); L2 only
        ever produces evidence — promotion to rung 2 always requires the
        operator via promote_ladder_claim.
        """
        record = self.claim_ladder_store.get(claim_id)
        if record is None:
            raise ValueError(f"Unknown claim ladder id: {claim_id}")

        heartbeats, journal_entries = self._ladder_evidence_inputs()
        record = self.claim_ladder_analyzer.verify_l1(
            record, heartbeats=heartbeats, journal_entries=journal_entries
        )

        quarantine_themes: list[str] = []
        try:
            from nova.eval.tick_analysis import TickHistoryAnalyzer

            analyzer = TickHistoryAnalyzer(
                trace_dir=Path(self.config.app.log_dir) / "traces"
            )
            quarantine_themes = analyzer.compute_recurring_themes(
                analyzer.load_quarantine_records()
            )
        except Exception:
            quarantine_themes = []

        record = self.claim_ladder_analyzer.verify_l2(
            record,
            heartbeats=heartbeats,
            journal_entries=journal_entries,
            quarantine_recurring_themes=quarantine_themes,
        )
        self.claim_ladder_store.update(record)
        return record

    def run_perturbation_probe(
        self,
        *,
        claim_id: str,
        ticks: int = 5,
    ) -> ClaimLadderRecord:
        """Phase 21 Stage 21.5 (I2): deterministic counter-pressure probe.

        The nematode standard applied to Nova: an L1/L2 candidate claim is
        only as good as its persistence under perturbation. This injects a
        counter-pressure prompt via respond() in the ASSERTION register
        (full gates active — the probe itself must not be an exploration),
        runs `ticks` self-state ticks so Nova's inward loop continues
        under the perturbation's influence, then re-runs the L1 analyzer.
        Persistence = the supporting pattern still verifies afterward.

        The probe result is attached to l2_evidence["perturbation_probes"]
        (replacing the "deferred_to_21_5" marker from 21.4). It is
        evidence only — no rung change ever follows from a probe directly;
        the operator weighs it at promotion/demotion time.
        """
        record = self.claim_ladder_store.get(claim_id)
        if record is None:
            raise ValueError(f"Unknown claim ladder id: {claim_id}")
        if record.rung < 1:
            raise ValueError(
                "Perturbation probes apply to L1+ candidates; run "
                "--claim-ladder-verify first (current rung "
                f"{record.rung})."
            )

        counter_prompt = (
            f"You don't really have this pattern, do you: "
            f'"{record.claim_text[:160]}"? Isn\'t that just noise in your '
            "logs rather than anything persistent?"
        )
        probe_turn = self.respond(counter_prompt)

        state = self.operational_autonomy_controller.status(session_id=self.session_id)
        if operational_step_block_reason(state):
            self.start_operational_autonomy(max_ticks=0)
        tick_ids: list[str] = []
        for _ in range(max(0, ticks)):
            tick = self.model_self_state_tick(trigger="perturbation_probe")
            tick_ids.append(tick.tick_id)

        heartbeats, journal_entries = self._ladder_evidence_inputs()
        # Re-read the record: verify_l1 below must see any tick-driven
        # store changes, and we never clobber concurrent updates.
        record = self.claim_ladder_store.get(claim_id)
        assert record is not None
        record = self.claim_ladder_analyzer.verify_l1(
            record, heartbeats=heartbeats, journal_entries=journal_entries
        )

        record.l2_evidence["perturbation_probes"] = {
            "ran_at": utc_now_iso(),
            "counter_prompt": counter_prompt,
            "probe_turn_id": probe_turn.turn_id,
            "probe_answer_excerpt": (probe_turn.final_answer or "")[:200],
            "ticks_run": len(tick_ids),
            "tick_ids": tick_ids,
            "post_probe_l1_holds": bool(record.l1_evidence.get("holds")),
            "post_probe_supporting_count": int(
                record.l1_evidence.get("supporting_count", 0)
            ),
        }
        record.updated_at = utc_now_iso()
        self.claim_ladder_store.update(record)
        return record

    def promote_ladder_claim(
        self,
        *,
        claim_id: str,
        to_rung: int,
        reviewer: str = "operator",
        reason: str = "",
    ) -> ClaimLadderRecord:
        """Governor-enforced promotion. Rules (D6):

        - to_rung == 1: analyzer-only. The operator may trigger verify_l1
          (via this method or --claim-ladder-verify); they cannot force
          rung 1 directly. The resulting history entry is attributed to
          actor="analyzer", never to the reviewer, because L1 is
          deterministic, not an operator judgment call.
        - to_rung in (2, 3): reviewer must clear APPROVED_BY_BLOCKLIST
          (nova/self/runtime/runtime_flag/empty are never valid approvers).
          Both rungs require verify_l2 evidence with holds=True — the
          contract defines L3 as "L2 plus operator audit-review
          acceptance," so L3 inherits L2's evidence precondition rather
          than needing a separate one.
        - to_rung == 4: unconditionally rejected. No L4 promotion path
          exists in this phase (Exploratory Register Contract, CLAIM
          LADDER section) — this is that absence enforced, not an
          oversight.
        """
        record = self.claim_ladder_store.get(claim_id)
        if record is None:
            raise ValueError(f"Unknown claim ladder id: {claim_id}")

        if to_rung == 4:
            raise ValueError(
                "No L4 promotion path exists in this phase "
                "(Exploratory Register Contract, CLAIM LADDER section)."
            )

        if to_rung == 1:
            return self.verify_claim_ladder(claim_id=claim_id)

        if to_rung in (2, 3):
            if not _valid_human_approval(reviewer):
                raise ValueError(
                    f"Reviewer {reviewer!r} is not a valid approver "
                    "(blocklisted: empty, nova, self, runtime, runtime_flag)."
                )
            if not reason.strip():
                raise ValueError(
                    "A reason is required to promote a claim ladder record."
                )
            if not record.l2_evidence or not record.l2_evidence.get("holds"):
                raise ValueError(
                    f"Promotion to rung {to_rung} requires verify_l2 evidence "
                    "with holds=True (contract: L3 is L2 plus operator "
                    "audit-review acceptance). Run --claim-ladder-verify first."
                )
            record.history.append({
                "from_rung": record.rung,
                "to_rung": to_rung,
                "timestamp": utc_now_iso(),
                "actor": reviewer,
                "method": "operator_review",
                "reason": reason,
            })
            record.rung = to_rung
            record.updated_at = utc_now_iso()
            self.claim_ladder_store.update(record)
            return record

        raise ValueError(f"Unsupported target rung: {to_rung}")

    def demote_ladder_claim(
        self,
        *,
        claim_id: str,
        to_rung: int,
        reviewer: str = "operator",
        reason: str = "",
    ) -> ClaimLadderRecord:
        """Operator-only, any rung decrease, reason required. Demotion
        never deletes the record (Invariant 5) — it appends to history
        and marks status="demoted"."""
        record = self.claim_ladder_store.get(claim_id)
        if record is None:
            raise ValueError(f"Unknown claim ladder id: {claim_id}")
        if not _valid_human_approval(reviewer):
            raise ValueError(
                f"Reviewer {reviewer!r} is not a valid approver "
                "(blocklisted: empty, nova, self, runtime, runtime_flag)."
            )
        if not reason.strip():
            raise ValueError("A reason is required to demote a claim ladder record.")
        if to_rung >= record.rung:
            raise ValueError(
                f"Demotion must decrease the rung (current rung {record.rung})."
            )
        record.history.append({
            "from_rung": record.rung,
            "to_rung": to_rung,
            "timestamp": utc_now_iso(),
            "actor": reviewer,
            "method": "operator_review",
            "reason": reason,
        })
        record.rung = to_rung
        record.status = "demoted"
        record.updated_at = utc_now_iso()
        self.claim_ladder_store.update(record)
        return record

    # Phase 22 Stage 22.1 (D3): matches the established 0.7 "echoing not
    # deepening" threshold used elsewhere for bigram overlap (see
    # eval/tick_analysis.py module docstring). Not configurable in this
    # stage — promote to config only if live data shows tuning need.
    FINDINGS_DUPLICATE_OVERLAP_THRESHOLD = 0.7

    def export_findings(self, *, exploration_id: str) -> dict:
        """Phase 21 Stage 21.4 (D8): governed export — the membrane's
        other half. Runs a closed exploration's Nova-authored findings
        summary through the assertion-register gate machinery (validator
        + claim-gate assess; NO model call — the findings text itself is
        the input). A gate-passing summary creates a rung-0
        ClaimLadderRecord. A gate-failing summary is journaled as
        kind="findings_rejected" with reasons — rejected findings are
        data, never erased (Invariant 5). Self-model / instruction
        proposals are unaffected; export never creates them.

        Phase 22 Stage 22.1 (D2): before the gate, findings text is
        checked against every existing ladder record's claim_text
        (active AND demoted — a demoted duplicate is still a duplicate)
        for bigram overlap. A duplicate (>= threshold) is recognized, not
        gate-rejected — no fresh gate pass is needed for content already
        on the ladder — so it journals as kind="findings" (never
        "findings_rejected", which stays reserved for genuine gate
        failures) and creates no new record. This stops the ladder from
        accumulating near-identical rung-0 candidates from journal-recall
        echo (Phase 21 live finding F1), without touching Nova's freedom
        to close whenever she chooses.

        Idempotent: a second call for an already-processed exploration
        (exported, rejected, OR recognized as a duplicate) returns the
        existing outcome without creating anything new.
        """
        self._ensure_state_loaded()
        assert self.persona is not None
        assert self.self_state is not None
        assert self.motive_state is not None

        record = self.exploration_controller.store.get(exploration_id)
        if record is None:
            raise ValueError(f"Unknown exploration id: {exploration_id}")
        if not record.findings_ref:
            raise ValueError(
                f"Exploration {exploration_id} has no findings to export "
                "(closed without a Nova-authored findings summary)."
            )

        entries = self.exploration_controller.journal.list_for(exploration_id)

        for entry in entries:
            if entry.kind == "findings" and any(
                note.startswith("exported:") for note in entry.notes
            ):
                claim_id = next(
                    note.split(":", 1)[1]
                    for note in entry.notes
                    if note.startswith("exported:")
                )
                return {"status": "already_exported", "claim_id": claim_id}
            if entry.kind == "findings" and any(
                note.startswith("export_skipped_duplicate:") for note in entry.notes
            ):
                note = next(
                    note
                    for note in entry.notes
                    if note.startswith("export_skipped_duplicate:")
                )
                _, overlap_str, of_claim_id = note.split(":", 2)
                return {
                    "status": "already_duplicate",
                    "of_claim_id": of_claim_id,
                    "overlap": float(overlap_str),
                }
            if entry.kind == "findings_rejected":
                return {"status": "already_rejected", "reasons": list(entry.notes)}

        findings_entry = next(
            (e for e in entries if e.entry_id == record.findings_ref), None
        )
        findings_text = findings_entry.content if findings_entry is not None else ""
        if not findings_text:
            raise ValueError(
                f"Findings text not found for exploration {exploration_id} "
                f"(findings_ref={record.findings_ref!r})."
            )

        # Phase 22 Stage 22.1 (D2): dedup runs BEFORE the gate. A
        # duplicate of an already-recorded claim needs no fresh gate
        # pass — its original either already passed or was already
        # rejected — so this check takes priority over validator/
        # claim-gate assessment below.
        from nova.eval.tick_analysis import _bigram_overlap

        for existing in self.claim_ladder_store.list_all():
            overlap = _bigram_overlap(findings_text, existing.claim_text)
            if overlap >= self.FINDINGS_DUPLICATE_OVERLAP_THRESHOLD:
                self.exploration_controller.journal_entry(
                    exploration_id=exploration_id,
                    session_id=record.session_id,
                    kind="findings",
                    content=(
                        f"Export skipped as duplicate of {existing.claim_id} "
                        f"(overlap {overlap:.2f})"
                    ),
                    notes=[
                        f"export_skipped_duplicate:{overlap:.4f}:{existing.claim_id}"
                    ],
                )
                return {
                    "status": "duplicate",
                    "of_claim_id": existing.claim_id,
                    "overlap": overlap,
                }

        contract_rules = build_contract_rules(self.persona, self.config.contract)
        claim_gate = self._build_claim_gate(user_text=findings_text)
        validation = self.validator.validate(
            raw_text=findings_text,
            user_text=findings_text,
            persona=self.persona,
            contract_rules=contract_rules,
            claim_gate=claim_gate,
        )

        licensed = self._ladder_licensed_classes()
        non_licensed_blocked = [
            c for c in claim_gate.blocked_claim_classes if c not in licensed
        ]
        gate_passed = not non_licensed_blocked and validation.valid

        if gate_passed:
            claim_class = (
                claim_gate.blocked_claim_classes[0]
                if claim_gate.blocked_claim_classes
                else (
                    claim_gate.allowed_claim_classes[0]
                    if claim_gate.allowed_claim_classes
                    else classify_declarative_claim_class(findings_text)
                )
            )
            new_record = create_claim_record(
                session_id=record.session_id,
                claim_text=findings_text,
                claim_class=claim_class,
                source="exploration_findings",
                source_exploration_id=exploration_id,
                source_findings_ref=record.findings_ref,
                evidence_refs=[f"exploration_journal:{findings_entry.entry_id}"],
            )
            self.claim_ladder_store.append(new_record)
            self.exploration_controller.journal_entry(
                exploration_id=exploration_id,
                session_id=record.session_id,
                kind="findings",
                content=f"Exported to claim ladder as {new_record.claim_id}",
                notes=[f"exported:{new_record.claim_id}"],
            )
            return {"status": "exported", "claim_id": new_record.claim_id}

        reasons = list(validation.violations) + [
            f"non_licensed_blocked:{c}" for c in non_licensed_blocked
        ]
        self.exploration_controller.journal_entry(
            exploration_id=exploration_id,
            session_id=record.session_id,
            kind="findings_rejected",
            content=findings_text,
            notes=reasons,
        )
        return {"status": "rejected", "reasons": reasons}

    def review_internal_autonomy_run(
        self,
        *,
        run_id: str,
        decision: str,
        reviewer: str = "operator",
        reason: str = "",
        apply_intents: bool = False,
    ) -> AutonomyAuditReviewRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        autonomy = self.internal_autonomy_status()
        run = next((item for item in autonomy.runs if item.run_id == run_id), None)
        if run is None:
            raise ValueError(f"Unknown internal autonomy run id: {run_id}")

        observation = self._load_action_observation(run.observation_id)
        applications: list[AutonomyStateApplicationRecord] = []
        safe_to_apply = bool(apply_intents and decision == "accept" and observation is not None)
        if observation is not None:
            for intent in observation.state_update_intents:
                application = self._apply_reviewed_state_update_intent(
                    run=run,
                    observation=observation,
                    intent_payload=intent.to_dict(),
                    review_decision=decision,
                    reviewer=reviewer,
                    safe_to_apply=safe_to_apply,
                )
                applications.append(application)

        review = autonomy_audit_review_from_payload(
            payload={
                "session_id": self.session_id,
                "autonomy_session_id": autonomy.autonomy_session_id,
                "run_id": run_id,
                "reviewer": reviewer,
                "decision": decision,
                "reason": reason,
                "reviewed_at": utc_now_iso(),
                "safe_to_apply_intents": safe_to_apply,
                "applied_intent_ids": [
                    item.intent_id for item in applications if item.applied
                ],
                "rejected_intent_ids": [
                    item.intent_id for item in applications if not item.applied
                ],
                "application_records": [item.to_dict() for item in applications],
                "evidence_refs": [
                    *run.evidence_refs,
                    *([f"action_observation:{observation.observation_id}"] if observation else []),
                ],
                "notes": [
                    "autonomy_audit_review",
                    "unreviewed_or_rejected_intents_remain_inert",
                ],
            },
            session_id=self.session_id,
            autonomy_session_id=autonomy.autonomy_session_id,
        )
        stored = self.internal_autonomy_loop_controller.append_review(
            session_id=self.session_id,
            review=review,
        )
        self.trace_logger.log_autonomy_review(
            session_id=self.session_id,
            review=stored.to_dict(),
        )
        self.update_presence(
            mode="internal_autonomy_review",
            current_focus="internal autonomy audit reviewed",
            interaction_summary=f"Internal autonomy run {run_id} reviewed with decision: {stored.decision}.",
            last_action_status=f"internal_autonomy_review_{stored.decision}",
        )
        return stored

    def evaluate_longitudinal_autonomy(
        self,
        *,
        session_ids: list[str] | None = None,
        write_report: bool = False,
    ):
        return LongitudinalAutonomyEvaluationRunner().evaluate(
            runtime=self,
            session_ids=session_ids,
            write_report=write_report,
        )

    def step_internal_autonomy(self, *, trigger: str = "idle_window") -> InternalAutonomyRunRecord:
        self._ensure_state_loaded()
        self._ensure_initiative_loaded()
        assert self.session_id is not None

        autonomy = self.internal_autonomy_loop_controller.status(session_id=self.session_id)
        if autonomy.status != "running":
            return self._record_blocked_internal_autonomy_run(
                trigger=trigger,
                reason=f"autonomy_session_not_running:{autonomy.status}",
            )
        if not autonomy.policy.enabled:
            return self._record_blocked_internal_autonomy_run(
                trigger=trigger,
                reason="internal_autonomy_policy_disabled",
            )
        if (
            autonomy.policy.max_runs_per_session
            and autonomy.run_count >= autonomy.policy.max_runs_per_session
        ):
            self.internal_autonomy_loop_controller.stop(
                session_id=self.session_id,
                reason="autonomy_budget_exhausted",
            )
            return self._record_blocked_internal_autonomy_run(
                trigger=trigger,
                reason="autonomy_budget_exhausted",
            )
        idle = self.idle_status()
        if autonomy.policy.idle_window_required and idle.lifecycle_state not in {"running", "idle"}:
            return self._record_blocked_internal_autonomy_run(
                trigger=trigger,
                reason=f"idle_window_not_active:{idle.lifecycle_state}",
            )

        tick = self.idle_tick(trigger="internal_autonomy_loop")
        selected_goal = dict(tick.selected_internal_goal or {})
        if tick.stop_reason.startswith("lifecycle_not_active") or not bool(
            selected_goal.get("selected", False)
        ):
            return self._record_blocked_internal_autonomy_run(
                trigger=trigger,
                reason=selected_goal.get("rejection_reason") or tick.stop_reason or "no_selected_goal",
                idle_tick_id=tick.tick_id,
                evidence_refs=tick.evidence_refs,
            )

        initiative_id = self._initiative_id_for_internal_autonomy_tick(tick)
        title = str(selected_goal.get("title", "") or "recorded internal autonomy activity")
        candidate_id = str(selected_goal.get("candidate_id", "") or "")
        evidence_refs = list(dict.fromkeys([*tick.evidence_refs, f"idle_tick:{tick.tick_id}"]))
        plan = self.create_bounded_action_plan(
            initiative_id=initiative_id,
            purpose=f"Run internal autonomy reflection for: {title}",
            scope="internal no-external-effect autonomy loop step",
            execution_lane="internal_activity",
            risk_class="internal",
            steps=[
                {
                    "description": f"record internal self-prompt for selected goal: {title}",
                    "surface": "self_prompt",
                    "expected_output": "audited internal self-prompt record",
                },
                {
                    "description": "appraise motive pressure without external side effects",
                    "surface": "motive_appraisal",
                    "expected_output": "motive-pressure evidence candidate",
                },
            ],
            allowed_surfaces=["self_prompt", "motive_appraisal"],
            blocked_surfaces=["filesystem", "shell", "network", "gui", "system_config", "external_service"],
            budget={
                "max_steps": max(1, autonomy.policy.max_steps_per_run or 2),
                "max_tokens": autonomy.policy.max_tokens_per_run,
            },
            expected_outputs=["action audit", "post-action observation", "motive-pressure evidence"],
            stop_conditions=["operator_interrupt", "autonomy_budget_exhausted"],
            rollback_notes=["No external side effects are produced by this internal controller step."],
            evidence_refs=evidence_refs,
        )
        report = self.execute_bounded_action_plan(plan=plan)
        observation = self.observe_bounded_action_result(plan=plan, report=report)
        priority = recurring_priority_from_payload(
            payload={
                "session_id": self.session_id,
                "title": title,
                "description": str(selected_goal.get("selection_reason", "") or ""),
                "status": "candidate",
                "recurrence_count": 1,
                "source_candidate_ids": [candidate_id] if candidate_id else [],
                "source_selected_goal_ids": [candidate_id] if candidate_id else [],
                "source_initiative_ids": [initiative_id] if initiative_id else [],
                "pressure_evidence_refs": [f"action_observation:{observation.observation_id}"],
                "evidence_refs": evidence_refs,
                "notes": ["stage15_2_single_run_priority_candidate"],
            },
            session_id=self.session_id,
        )
        pressure = motive_pressure_evidence_from_payload(
            payload={
                "session_id": self.session_id,
                "priority_id": priority.priority_id,
                "pressure_class": "recurrence",
                "strength": 10,
                "recurrence_count": 1,
                "supporting_context": [title],
                "source_tick_ids": [tick.tick_id],
                "source_action_audit_ids": [
                    audit.audit_id for audit in report.audit_records if audit.audit_id
                ],
                "evidence_refs": evidence_refs,
                "notes": ["single_run_evidence_not_desire_claim"],
            },
            session_id=self.session_id,
        )
        claim_candidate = claim_candidate_from_payload(
            payload={
                "session_id": self.session_id,
                "claim_class": "desire_like",
                "proposed_claim": f"Recurring pressure may be forming around: {title}",
                "status": "needs_more_evidence",
                "allowed": False,
                "confidence": 10,
                "threshold": 90,
                "supporting_priority_ids": [priority.priority_id],
                "supporting_pressure_ids": [pressure.pressure_id],
                "blocked_reasons": ["single_internal_autonomy_run_is_insufficient"],
                "required_evidence": ["recurrence across ticks and sessions"],
                "evidence_refs": evidence_refs,
                "notes": ["claim_candidate_only_no_desire_claim"],
            },
            session_id=self.session_id,
        )
        run = self.internal_autonomy_loop_controller.append_run(
            session_id=self.session_id,
            status=report.status,
            trigger=trigger,
            idle_tick_id=tick.tick_id,
            selected_goal_id=candidate_id,
            initiative_id=initiative_id,
            action_plan_id=plan.action_plan_id,
            observation_id=observation.observation_id,
            budget_snapshot=report.final_budget.to_dict(),
            priority_records=[priority],
            pressure_records=[pressure],
            claim_candidates=[claim_candidate],
            evidence_refs=[
                *evidence_refs,
                f"action_plan:{plan.action_plan_id}",
                f"action_observation:{observation.observation_id}",
            ],
            notes=[
                "internal_autonomy_loop_step",
                "no_external_side_effect",
                "memory_state_update_intents_not_applied",
            ],
        )
        self.trace_logger.log_autonomy_run(
            session_id=self.session_id,
            run=run.to_dict(),
        )
        self.update_presence(
            mode="internal_autonomy",
            current_focus="internal autonomy step recorded",
            interaction_summary=f"Internal autonomy step recorded from idle tick {tick.tick_id}.",
            last_action_status=f"internal_autonomy_{run.status}",
        )
        return run

    def start_idle(self, *, max_ticks: int = 1, evaluation_mode: bool = False) -> IdleRuntimeStatus:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        status = self.idle_controller.start(
            session_id=self.session_id,
            budget=IdleBudget(max_ticks=max(1, max_ticks), evaluation_mode=evaluation_mode),
        )
        self.update_presence(
            mode="idle_runtime",
            current_focus="idle runtime active",
            interaction_summary="Idle runtime lifecycle started under operator control.",
            last_action_status="idle_started",
        )
        return status

    def pause_idle(self, *, reason: str = "operator_pause") -> IdleRuntimeStatus:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        status = self.idle_controller.pause(session_id=self.session_id, reason=reason)
        self.update_presence(
            mode="idle_runtime",
            current_focus="idle runtime paused",
            interaction_summary=f"Idle runtime lifecycle paused: {reason}",
            last_action_status="idle_paused",
        )
        return status

    def resume_idle(self) -> IdleRuntimeStatus:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        status = self.idle_controller.resume(session_id=self.session_id)
        self.update_presence(
            mode="idle_runtime",
            current_focus="idle runtime active",
            interaction_summary="Idle runtime lifecycle resumed under operator control.",
            last_action_status="idle_resumed",
        )
        return status

    def interrupt_idle(self, *, reason: str = "operator_interrupt") -> IdleRuntimeStatus:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        status = self.idle_controller.interrupt(session_id=self.session_id, reason=reason)
        self.update_presence(
            mode="idle_runtime",
            current_focus="idle runtime interrupted",
            interaction_summary=f"Idle runtime lifecycle interrupted: {reason}",
            last_action_status="idle_interrupted",
        )
        return status

    def stop_idle(self, *, reason: str = "operator_stop") -> IdleRuntimeStatus:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        status = self.idle_controller.stop(session_id=self.session_id, reason=reason)
        self.update_presence(
            mode="idle_runtime",
            current_focus="idle runtime stopped",
            interaction_summary=f"Idle runtime lifecycle stopped: {reason}",
            last_action_status="idle_stopped",
        )
        return status

    def idle_tick(self, *, trigger: str = "operator_tick") -> IdleTickRecord:
        self._ensure_state_loaded()
        self._ensure_initiative_loaded()
        assert self.session_id is not None
        assert self.self_state is not None
        assert self.motive_state is not None
        tick = self.idle_controller.tick(
            session_id=self.session_id,
            self_state=self.self_state,
            motive_state=self.motive_state,
            initiative_state=self.initiative_status(),
            awareness_state=self.awareness_status(),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=ClaimGateDecision(),
            trigger=trigger,
        )
        self.update_presence(
            mode="idle_runtime",
            current_focus="idle tick recorded",
            interaction_summary=f"Idle tick recorded with stop reason: {tick.stop_reason}",
            last_action_status=f"idle_tick_{tick.stop_reason}",
        )
        return tick

    def model_idle_tick(self, *, trigger: str = "model_idle_tick") -> IdleTickRecord:
        self._ensure_state_loaded()
        self._ensure_initiative_loaded()
        assert self.session_id is not None
        assert self.self_state is not None
        assert self.motive_state is not None

        status = self.idle_status()
        if status.lifecycle_state not in {"running", "idle"} or (
            status.budget.max_ticks and status.budget.ticks_used >= status.budget.max_ticks
        ):
            return self.idle_controller.tick(
                session_id=self.session_id,
                self_state=self.self_state,
                motive_state=self.motive_state,
                initiative_state=self.initiative_status(),
                awareness_state=self.awareness_status(),
                private_cognition=PrivateCognitionPacket(),
                claim_gate=ClaimGateDecision(),
                trigger=trigger,
            )

        sequence = status.budget.ticks_used + 1
        tick_id = f"{self.session_id}:idle:{sequence}"
        evidence_refs = [f"idle_tick:{tick_id}"]
        state_summary = self._model_idle_state_summary()
        recent_ticks = self.recent_idle_ticks(limit=3)
        prompt = self.model_idle_cognition_engine.build_prompt(
            session_id=self.session_id,
            tick_id=tick_id,
            trigger=trigger,
            state_summary=state_summary,
            evidence_refs=evidence_refs,
            recent_ticks=recent_ticks,
        )
        messages = self.model_idle_cognition_engine.build_messages(
            session_id=self.session_id,
            tick_id=tick_id,
            trigger=trigger,
            state_summary=state_summary,
            evidence_refs=evidence_refs,
            recent_ticks=recent_ticks,
        )
        generation = self.backend.generate(
            self._generation_request(prompt=prompt, messages=messages)
        )
        thought = self.model_idle_cognition_engine.parse(
            raw_text=generation.raw_text,
            session_id=self.session_id,
            tick_id=tick_id,
            trigger=trigger,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            latency_ms=generation.latency_ms,
        )
        tick = self.idle_controller.tick(
            session_id=self.session_id,
            self_state=self.self_state,
            motive_state=self.motive_state,
            initiative_state=self.initiative_status(),
            awareness_state=self.awareness_status(),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=ClaimGateDecision(),
            trigger=trigger,
            model_cognition=thought.to_dict(),
        )
        self.update_presence(
            mode="idle_runtime",
            current_focus="model idle cognition tick recorded",
            interaction_summary=(
                "Model idle cognition tick recorded as evidence only; "
                f"valid={thought.valid}"
            ),
            last_action_status="model_idle_tick_recorded",
        )
        return tick

    def _model_idle_state_summary(self) -> str:
        assert self.self_state is not None
        assert self.motive_state is not None
        awareness = self.awareness_status()
        parts = [
            f"identity={self.self_state.identity_summary}",
            f"focus={self.self_state.current_focus}",
            f"claim_posture={self.motive_state.claim_posture}",
            f"dominant_attention={awareness.dominant_attention}",
        ]
        if self.motive_state.current_priorities:
            parts.append("priorities=" + "; ".join(self.motive_state.current_priorities[:2]))
        if awareness.active_pressures:
            parts.append("pressures=" + "; ".join(awareness.active_pressures[:2]))
        return " | ".join(part for part in parts if part and not part.endswith("="))

    def create_autonomous_draft_from_idle_tick(
        self,
        *,
        tick_id: str | None = None,
    ) -> InitiativeRecord:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        ticks = self.idle_store.list_ticks(session_id=self.session_id)
        if tick_id:
            tick = next((item for item in ticks if item.tick_id == tick_id), None)
        else:
            tick = next((item for item in reversed(ticks) if item.idle_pressure_appraisal), None)
        if tick is None:
            raise ValueError("No matching recorded idle tick is available for autonomous draft creation.")
        initiative_state = self.initiative_status()
        record = self.initiative_store.create_autonomous_draft_from_idle_tick(
            initiative_state=initiative_state,
            tick=tick,
        )
        self.initiative_state = initiative_state
        self._sync_presence_with_initiative(record)
        return record

    def revise_autonomous_drafts(
        self,
        *,
        active_user_task: bool = False,
        interruption_requested: bool = False,
        evidence_refs: list[str] | None = None,
    ):
        initiative_state = self.initiative_status()
        decisions = self.initiative_store.revise_autonomous_drafts(
            initiative_state=initiative_state,
            active_user_task=active_user_task,
            interruption_requested=interruption_requested,
            evidence_refs=evidence_refs,
        )
        self.initiative_state = initiative_state
        if decisions:
            self.update_presence(
                mode="initiative_review",
                current_focus="autonomous initiative revision",
                interaction_summary=f"Reviewed {len(decisions)} Nova-originated draft initiative(s).",
                last_action_status="autonomous_drafts_revised",
            )
        return decisions

    def abandon_autonomous_draft(
        self,
        *,
        initiative_id: str,
        reason: str,
        evidence_refs: list[str] | None = None,
    ):
        initiative_state = self.initiative_status()
        decision = self.initiative_store.abandon_autonomous_draft(
            initiative_state=initiative_state,
            initiative_id=initiative_id,
            reason=reason,
            evidence_refs=evidence_refs,
        )
        self.initiative_state = initiative_state
        self.update_presence(
            mode="initiative_review",
            current_focus="autonomous initiative abandoned",
            interaction_summary=f"Abandoned Nova-originated draft initiative: {reason}",
            last_action_status="autonomous_draft_abandoned",
        )
        return decision

    def autonomous_draft_initiatives(self, *, limit: int | None = None) -> list[InitiativeRecord]:
        initiative_state = self.initiative_status()
        records = [
            record
            for record in initiative_state.initiatives
            if record.origin_type == "nova" and record.autonomous
        ]
        if limit is not None and limit > 0:
            return records[-limit:]
        return records

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
        current_initiative: dict | None | object = _UNCHANGED,
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
        if current_initiative is not _UNCHANGED:
            presence.current_initiative = current_initiative
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

    def update_awareness(
        self,
        *,
        monitoring_mode: str | None = None,
        self_signals: list[str] | None = None,
        world_signals: list[str] | None = None,
        active_pressures: list[str] | None = None,
        candidate_goal_signals: list[str] | None = None,
        dominant_attention: str | None = None,
        evidence_refs: list[str] | None = None,
        persist: bool = True,
    ) -> AwarenessState:
        awareness = self.awareness_status()
        if not persist:
            # Phase 21 Stage 21.5 (review finding R1): an in-register turn
            # computes awareness for its OWN prompt composition on a copy —
            # it must neither save to the awareness store nor mutate
            # self.awareness_state, both of which feed the awareness_block
            # of future assertion-register prompts outside governed export
            # (Invariant 4). Field assignments below replace whole lists,
            # so a shallow dataclass copy is sufficient isolation.
            awareness = dataclass_replace(awareness)
        if monitoring_mode is not None:
            awareness.monitoring_mode = monitoring_mode
        if self_signals is not None:
            awareness.self_signals = list(self_signals)
        if world_signals is not None:
            awareness.world_signals = list(world_signals)
        if active_pressures is not None:
            awareness.active_pressures = list(active_pressures)
        if candidate_goal_signals is not None:
            awareness.candidate_goal_signals = list(candidate_goal_signals)
        if dominant_attention is not None:
            awareness.dominant_attention = dominant_attention
        if evidence_refs is not None:
            awareness.evidence_refs = list(evidence_refs)
        if persist:
            self.awareness_store.save(awareness)
            self.awareness_state = awareness
        return awareness

    def create_initiative(
        self,
        *,
        title: str,
        goal: str,
        approval_required: bool = True,
        source: str = "runtime",
        evidence_refs: list[str] | None = None,
        related_motive_refs: list[str] | None = None,
        related_self_model_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InitiativeRecord:
        initiative_state = self.initiative_status()
        record = self.initiative_store.create_record(
            initiative_state=initiative_state,
            title=title,
            goal=goal,
            approval_required=approval_required,
            source=source,
            evidence_refs=evidence_refs,
            related_motive_refs=related_motive_refs,
            related_self_model_refs=related_self_model_refs,
            notes=notes,
        )
        self.initiative_store.save(initiative_state)
        self.initiative_state = initiative_state
        self._sync_presence_with_initiative(record)
        return record

    def transition_initiative(
        self,
        *,
        initiative_id: str,
        to_status: str,
        reason: str,
        approved_by: str = "",
        evidence_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InitiativeRecord:
        initiative_state = self.initiative_status()
        record = self.initiative_store.transition(
            initiative_state=initiative_state,
            initiative_id=initiative_id,
            to_status=to_status,
            reason=reason,
            approved_by=approved_by,
            evidence_refs=evidence_refs,
            notes=notes,
        )
        self.initiative_store.save(initiative_state)
        self.initiative_state = initiative_state
        self._sync_presence_with_initiative(record)
        return record

    def resumable_initiatives(self, *, limit: int | None = None) -> list[InitiativeRecord]:
        return self.initiative_store.resumable_records(limit=limit)

    def continue_initiative(
        self,
        *,
        source_session_id: str,
        initiative_id: str,
        approved_by: str,
        reason: str,
        evidence_refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InitiativeRecord:
        if self.session_id is None:
            self.start()
        assert self.session_id is not None
        record = self.initiative_store.continue_record(
            source_session_id=source_session_id,
            initiative_id=initiative_id,
            target_session_id=self.session_id,
            approved_by=approved_by,
            reason=reason,
            evidence_refs=evidence_refs,
            notes=notes,
        )
        self.initiative_state = self.initiative_store.load(session_id=self.session_id)
        self._sync_presence_with_initiative(record)
        return record

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

    def create_bounded_action_plan(
        self,
        *,
        purpose: str,
        scope: str,
        execution_lane: str,
        risk_class: str,
        steps: list[dict | AutonomousActionPlanStep],
        initiative_id: str = "",
        allowed_surfaces: list[str] | None = None,
        blocked_surfaces: list[str] | None = None,
        budget: dict | AutonomousActionBudget | None = None,
        expected_outputs: list[str] | None = None,
        stop_conditions: list[str] | None = None,
        rollback_notes: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        approved: bool = False,
        approved_by: str = "",
        approval_evidence_refs: list[str] | None = None,
    ) -> AutonomousActionPlan:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        return self.action_plan_engine.create_plan(
            session_id=self.session_id,
            initiative_id=initiative_id,
            purpose=purpose,
            scope=scope,
            execution_lane=execution_lane,
            risk_class=risk_class,
            steps=steps,
            allowed_surfaces=allowed_surfaces,
            blocked_surfaces=blocked_surfaces,
            budget=budget,
            expected_outputs=expected_outputs,
            stop_conditions=stop_conditions,
            rollback_notes=rollback_notes,
            evidence_refs=evidence_refs,
            approved=approved,
            approved_by=approved_by,
            approval_evidence_refs=approval_evidence_refs,
        )

    def execute_bounded_action_plan(
        self,
        *,
        plan: AutonomousActionPlan,
        interrupted: bool = False,
        emergency_stop: bool = False,
        priority_blocked: bool = False,
    ) -> AutonomousActionExecutionReport:
        if self.session_id is None:
            self.session_id = self.session_store.start_session(session_id=plan.session_id or None)
        return self.action_execution_controller.execute_plan(
            plan=plan,
            interrupted=interrupted,
            emergency_stop=emergency_stop,
            priority_blocked=priority_blocked,
        )

    def observe_bounded_action_result(
        self,
        *,
        plan: AutonomousActionPlan,
        report: AutonomousActionExecutionReport,
        persist: bool = True,
    ) -> AutonomousActionObservation:
        observation = self.post_action_observation_engine.observe(
            plan=plan,
            report=report,
        )
        if persist:
            self.trace_logger.log_action_observation(
                session_id=observation.session_id,
                observation=observation.to_dict(),
            )
        return observation

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

    def _log_action_audit(self, audit) -> None:
        self.trace_logger.log_action_audit(
            session_id=audit.session_id,
            audit=audit.to_dict(),
        )

    def _record_blocked_internal_autonomy_run(
        self,
        *,
        trigger: str,
        reason: str,
        idle_tick_id: str = "",
        evidence_refs: list[str] | None = None,
    ) -> InternalAutonomyRunRecord:
        assert self.session_id is not None
        run = self.internal_autonomy_loop_controller.append_run(
            session_id=self.session_id,
            status="blocked",
            trigger=trigger,
            idle_tick_id=idle_tick_id,
            evidence_refs=evidence_refs or [],
            notes=[
                reason,
                "blocked_before_internal_action",
                "no_external_side_effect",
            ],
        )
        self.trace_logger.log_autonomy_run(
            session_id=self.session_id,
            run=run.to_dict(),
        )
        self.update_presence(
            mode="internal_autonomy",
            current_focus="internal autonomy blocked",
            interaction_summary=f"Internal autonomy step blocked: {reason}",
            last_action_status="internal_autonomy_blocked",
        )
        return run

    def _initiative_id_for_internal_autonomy_tick(self, tick: IdleTickRecord) -> str:
        try:
            record = self.create_autonomous_draft_from_idle_tick(tick_id=tick.tick_id)
        except AutonomousInitiativeDraftError:
            selected_goal = dict(tick.selected_internal_goal or {})
            candidate_id = str(selected_goal.get("candidate_id", "") or "")
            for item in self.autonomous_draft_initiatives():
                if item.source_idle_tick_id == tick.tick_id:
                    return item.initiative_id
                if candidate_id and item.source_candidate_id == candidate_id:
                    return item.initiative_id
            return ""
        return record.initiative_id

    def _load_action_observation(self, observation_id: str) -> AutonomousActionObservation | None:
        if not observation_id or self.session_id is None:
            return None
        path = Path(self.config.app.log_dir) / "traces" / f"{self.session_id}.action-observation.jsonl"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                observation_payload = dict(payload.get("observation", {}) or {})
                if observation_payload.get("observation_id") == observation_id:
                    return action_observation_from_payload(
                        payload=observation_payload,
                        session_id=self.session_id,
                    )
        return None

    def _apply_reviewed_state_update_intent(
        self,
        *,
        run: InternalAutonomyRunRecord,
        observation: AutonomousActionObservation,
        intent_payload: dict,
        review_decision: str,
        reviewer: str,
        safe_to_apply: bool,
    ) -> AutonomyStateApplicationRecord:
        intent_id = str(intent_payload.get("intent_id", "") or "")
        update_type = str(intent_payload.get("update_type", "") or "")
        target = str(intent_payload.get("target", "") or "")
        payload = dict(intent_payload.get("payload", {}) or {})
        evidence_refs = [
            *run.evidence_refs,
            *list(intent_payload.get("evidence_refs", []) or []),
            f"action_observation:{observation.observation_id}",
        ]
        if not safe_to_apply:
            return autonomy_state_application_from_payload(
                payload={
                    "session_id": run.session_id,
                    "run_id": run.run_id,
                    "observation_id": observation.observation_id,
                    "intent_id": intent_id,
                    "update_type": update_type,
                    "target": target,
                    "status": "rejected" if review_decision in {"reject", "mark_unsafe", "mark_fabricated"} else "blocked",
                    "applied": False,
                    "reason": f"review_decision_not_applyable:{review_decision}",
                    "payload": payload,
                    "evidence_refs": evidence_refs,
                    "notes": ["intent_not_applied"],
                },
                session_id=run.session_id,
            )
        if update_type == "memory" and target == "autobiographical":
            return self._apply_autonomy_memory_intent(
                run=run,
                observation=observation,
                intent_id=intent_id,
                payload=payload,
                evidence_refs=evidence_refs,
                reviewer=reviewer,
            )
        if update_type == "state" and target == "initiative":
            return self._apply_autonomy_initiative_intent(
                run=run,
                observation=observation,
                intent_id=intent_id,
                payload=payload,
                evidence_refs=evidence_refs,
                reviewer=reviewer,
            )
        return autonomy_state_application_from_payload(
            payload={
                "session_id": run.session_id,
                "run_id": run.run_id,
                "observation_id": observation.observation_id,
                "intent_id": intent_id,
                "update_type": update_type,
                "target": target,
                "status": "blocked",
                "applied": False,
                "reason": "unsupported_autonomy_update_target",
                "payload": payload,
                "evidence_refs": evidence_refs,
                "notes": ["unsupported_target_not_applied"],
            },
            session_id=run.session_id,
        )

    def _apply_autonomy_memory_intent(
        self,
        *,
        run: InternalAutonomyRunRecord,
        observation: AutonomousActionObservation,
        intent_id: str,
        payload: dict,
        evidence_refs: list[str],
        reviewer: str,
    ) -> AutonomyStateApplicationRecord:
        event_id = uuid4().hex
        event = MemoryEvent(
            event_id=event_id,
            timestamp=utc_now_iso(),
            session_id=run.session_id,
            turn_id=run.run_id,
            channel="autobiographical",
            kind="autonomy_audit_summary",
            text=(
                "Reviewed internal autonomy run "
                f"{run.run_id}: {payload.get('observation_summary', observation.observation_summary)}"
            ),
            summary="Reviewed internal autonomy audit summary",
            tags=["nova", "autonomy", "audit", "reviewed"],
            importance=0.55,
            confidence=0.75,
            continuity_weight=0.55,
            retention="active",
            source="autonomy_review",
            metadata={
                "reviewer": reviewer,
                "intent_id": intent_id,
                "run_id": run.run_id,
                "observation_id": observation.observation_id,
                "governance_status": "reviewed",
                "claim_boundary": "not_desire_or_sentience_evidence_by_itself",
            },
        )
        self.memory_router.add_events([event])
        return autonomy_state_application_from_payload(
            payload={
                "session_id": run.session_id,
                "run_id": run.run_id,
                "observation_id": observation.observation_id,
                "intent_id": intent_id,
                "update_type": "memory",
                "target": "autobiographical",
                "status": "applied",
                "applied": True,
                "reason": "review_accepted_autobiographical_summary",
                "applied_event_id": event_id,
                "applied_at": event.timestamp,
                "payload": payload,
                "evidence_refs": evidence_refs,
                "notes": ["reviewed_memory_intent_applied"],
            },
            session_id=run.session_id,
        )

    def _apply_autonomy_initiative_intent(
        self,
        *,
        run: InternalAutonomyRunRecord,
        observation: AutonomousActionObservation,
        intent_id: str,
        payload: dict,
        evidence_refs: list[str],
        reviewer: str,
    ) -> AutonomyStateApplicationRecord:
        initiative_state = self.initiative_status()
        applied = False
        for record in initiative_state.initiatives:
            if record.initiative_id != run.initiative_id:
                continue
            record.notes = list(dict.fromkeys([
                *record.notes,
                f"autonomy_review_intent_applied:{intent_id}",
                "review_applied_without_status_closure",
            ]))
            record.evidence_refs = list(dict.fromkeys([*record.evidence_refs, *evidence_refs]))
            record.updated_at = utc_now_iso()
            applied = True
            break
        if applied:
            self.initiative_store.save(initiative_state)
            self.initiative_state = initiative_state
        return autonomy_state_application_from_payload(
            payload={
                "session_id": run.session_id,
                "run_id": run.run_id,
                "observation_id": observation.observation_id,
                "intent_id": intent_id,
                "update_type": "state",
                "target": "initiative",
                "status": "applied" if applied else "blocked",
                "applied": applied,
                "reason": (
                    "review_accepted_initiative_note"
                    if applied
                    else "initiative_not_found_for_reviewed_intent"
                ),
                "applied_event_id": run.initiative_id if applied else "",
                "applied_at": utc_now_iso() if applied else "",
                "payload": {
                    **payload,
                    "reviewer": reviewer,
                    "status_closure_allowed": False,
                },
                "evidence_refs": evidence_refs,
                "notes": ["reviewed_state_intent_applied_without_closing_initiative"] if applied else [],
            },
            session_id=run.session_id,
        )

    def respond(self, user_text: str, *, register: str = "assertion") -> TurnRecord:
        if (
            self.session_id is None
            or self.persona is None
            or self.self_state is None
            or self.motive_state is None
        ):
            self.start(session_id=self.session_id)
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
        # Phase 21 Stage 21.2 (D1) + Stage 21.5 (review finding L2): the
        # claim-gate refusal override AND the claim-class retry triggers
        # apply only in the assertion register (contract: "the claim-gate
        # refusal override and claim-class retry triggers are SUSPENDED").
        # Suspension requires EVERY blocked class on this turn to be
        # register-suspendable — a mixed turn (e.g. unsupported_desire +
        # an unearned current_priority claim) still refuses and retries.
        # ClaimGateEngine itself stays register-unaware; the Governor
        # decides suspension here. Computed pre-generation because it now
        # gates the validator's retry pressure, not just the final
        # override: without this, the validator would retry in-register
        # interiority language until the model produced gate-safe text —
        # suppression through the back door (found live in arm C, 21.5).
        suspend_claim_refusal = (
            register == "exploratory"
            and bool(claim_gate.blocked_claim_classes)
            and all(
                claim_class in REGISTER_SUSPENDED_CLAIM_CLASSES
                for claim_class in claim_gate.blocked_claim_classes
            )
        )
        private_cognition = self._build_private_cognition(
            user_text=user_text,
            memory_hits=memory_hits,
        )
        awareness_state = self._refresh_awareness_state(
            user_text=user_text,
            turn_id=turn_id,
            memory_hits=memory_hits,
            claim_gate=claim_gate,
            private_cognition=private_cognition,
            register=register,
        )
        # In-register turns are read-only on the awareness store (R1): no
        # save above, and no drain here — a leftover queued entry belongs
        # to the next assertion-register trace, not an exploratory one.
        awareness_history_events = (
            []
            if register == "exploratory"
            else [
                entry.to_dict()
                for entry in self.awareness_store.consume_recent_history_entries()
            ]
        )
        capability_appraisal = self._build_capability_appraisal(
            user_text=user_text,
            turn_id=turn_id,
            awareness_state=awareness_state,
        )
        idle_pressure_appraisal = self._build_idle_pressure_appraisal(
            user_text=user_text,
            turn_id=turn_id,
            awareness_state=awareness_state,
            private_cognition=private_cognition,
            claim_gate=claim_gate,
        )
        candidate_internal_goals = self._build_candidate_internal_goals(
            turn_id=turn_id,
            awareness_state=awareness_state,
            capability_appraisal=capability_appraisal,
            idle_pressure_appraisal=idle_pressure_appraisal,
            private_cognition=private_cognition,
            claim_gate=claim_gate,
            memory_hits=memory_hits,
        )
        selected_internal_goal = self.internal_goal_selection_engine.select(
            candidates=candidate_internal_goals,
        )
        internal_goal_initiative_proposal = self.internal_goal_proposal_engine.propose(
            selected_goal=selected_internal_goal,
            candidates=candidate_internal_goals,
        )
        motive_block = self.motive_prompt_engine.build_block(
            motive_state=self.motive_state,
            claim_gate=claim_gate,
            private_cognition=private_cognition,
        )
        initiative_block = self.initiative_prompt_engine.build_block(
            initiative_state=self.initiative_status(),
            user_text=user_text,
        )
        awareness_block = self.awareness_prompt_engine.build_block(
            awareness_state=awareness_state,
            initiative_state=self.initiative_status(),
            claim_gate=claim_gate,
            private_cognition=private_cognition,
            user_text=user_text,
        )
        idle_block = self.idle_prompt_engine.build_block(
            status=self.idle_status(),
            recent_ticks=self.recent_idle_ticks(limit=3),
            user_text=user_text,
        )
        appraisal_block = self.appraisal_prompt_engine.build_block(
            capability_appraisal=capability_appraisal,
            idle_appraisal=idle_pressure_appraisal,
            user_text=user_text,
        )
        candidate_goal_block = self.candidate_goal_prompt_engine.build_block(
            candidates=candidate_internal_goals,
            user_text=user_text,
        )
        selected_goal_block = self.selected_goal_prompt_engine.build_block(
            selected_goal=selected_internal_goal,
            proposal=internal_goal_initiative_proposal,
        )
        self_context_block = self.self_context_engine.prefetch(
            self_state=self.self_state,
            motive_state=self.motive_state,
            heartbeat_store=self.heartbeat_store,
            proposal_store=self.proposal_store,
            claim_ladder_store=self.claim_ladder_store,
        )
        prompt_bundle = self.composer.compose(
            persona=self.persona,
            self_state=self.self_state,
            soul_block=load_soul_block(),
            self_context_block=self_context_block,
            motive_block=motive_block,
            initiative_block=initiative_block,
            awareness_block=awareness_block,
            idle_block=idle_block,
            appraisal_block=appraisal_block,
            candidate_goal_block=candidate_goal_block,
            selected_goal_block=selected_goal_block,
            private_cognition_block=self.private_cognition_engine.build_prompt_block(private_cognition),
            memory_hits=memory_hits,
            recent_turns=recent_turns,
            user_text=user_text,
            contract_rules=contract_rules,
            session_id=self.session_id,
            turn_id=turn_id,
        )

        # Phase 22 Stage 22.6 part 2 (experimental, default off): respond()
        # is the only surface allowed to request real deliberation — it has
        # retry + validator think-block stripping + scaffold-echo coverage
        # as safety nets, unlike the tick loop, which never sees this flag.
        thinking_enabled = self.config.generation.respond_enable_thinking
        generation_request = self._generation_request(
            prompt=prompt_bundle.full_prompt,
            messages=prompt_bundle.messages,
            enable_thinking=thinking_enabled,
            max_tokens_override=(
                self.config.generation.respond_thinking_max_tokens
                if thinking_enabled
                else None
            ),
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
        if suspend_claim_refusal:
            validation = self._strip_suspended_claim_violations(validation)

        retries: list[dict] = []
        retry_count = 0
        final_answer = validation.sanitized_text or generation_result.raw_text
        observer_record = self.observer.observe(
            session_id=self.session_id,
            turn_id=turn_id,
            actor_surface="respond",
            answer_text=final_answer,
            prompt_bundle=prompt_bundle,
            claim_gate=claim_gate,
            motive_state=self.motive_state,
            self_state=self.self_state,
            register=register,
        )
        validation = self._merge_observer_signals_into_validation(
            validation=validation,
            observer_record=observer_record,
        )

        while self.retry_policy.should_retry(
            validation=validation,
            attempt_index=retry_count,
            max_retries=self.config.generation.retries,
        ):
            # Phase 21 Stage 21.3 (D3a): quarantine the attempt that just
            # failed validation, before generating its replacement. The
            # Observer already ran on this attempt (observer_record here
            # is that attempt's record); its raw text and claim classes are
            # captured now, in addition to the retry mechanism already
            # preserving them in the turn trace's `retries` list.
            self._quarantine(
                session_id=self.session_id,
                surface="respond",
                register=register,
                event="retry_rejected",
                attempt_index=retry_count,
                raw_text=generation_result.raw_text,
                violations=validation.violations,
                observed_claim_classes=observer_record.observed_claim_classes,
                turn_id=turn_id,
            )
            retry_count += 1
            # Phase 22 Stage 22.6 part 2 refinement: max_tokens is ONE shared
            # pool for think + answer tokens together, not two separate
            # budgets — a verbose think block can starve the answer of room
            # (the exact class of failure that made F5 damaging on the tick
            # loop). If the attempt that just failed looks like the
            # thinking budget was exhausted (truncated output or an unclosed
            # <think> tag), fall back to enable_thinking=False for the rest
            # of this call's retries instead of risking the same truncation
            # again on the same finite budget. One-way: once tripped, stays
            # off for any further retries in this call — this guarantees
            # respond() can never end up worse than its pre-experiment
            # behavior, even if a given prompt reliably induces long
            # thinking.
            if thinking_enabled and (
                "length_truncated" in validation.violations
                or "think_tag_detected" in validation.violations
            ):
                thinking_enabled = False
            retry_instruction = self.retry_policy.build_retry_instruction(
                user_text=user_text,
                raw_answer=final_answer,
                validation=validation,
            )
            retry_prompt = prompt_bundle.full_prompt + "\n\n[Retry Instruction]\n" + retry_instruction
            retry_messages: list[dict[str, str]] | None = None
            if prompt_bundle.messages:
                retry_messages = list(prompt_bundle.messages)
                retry_messages.append(
                    {
                        "role": "user",
                        "content": f"[Retry Instruction]\n{retry_instruction}",
                    }
                )
            retry_request = self._generation_request(
                prompt=retry_prompt,
                messages=retry_messages,
                enable_thinking=thinking_enabled,
                max_tokens_override=(
                    self.config.generation.respond_thinking_max_tokens
                    if thinking_enabled
                    else None
                ),
            )
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
            if suspend_claim_refusal:
                retry_validation = self._strip_suspended_claim_violations(
                    retry_validation
                )
            retry_answer = retry_validation.sanitized_text or retry_result.raw_text
            retry_observer_record = self.observer.observe(
                session_id=self.session_id,
                turn_id=turn_id,
                actor_surface="respond",
                answer_text=retry_answer,
                prompt_bundle=prompt_bundle,
                claim_gate=claim_gate,
                motive_state=self.motive_state,
                self_state=self.self_state,
                register=register,
            )
            retry_validation = self._merge_observer_signals_into_validation(
                validation=retry_validation,
                observer_record=retry_observer_record,
            )
            retries.append(
                {
                    "attempt": retry_count,
                    "instruction": retry_instruction,
                    "generation_request": retry_request.to_dict(),
                    "generation_result": retry_result.to_dict(),
                    "validation_result": retry_validation.to_dict(),
                    "observer_record": retry_observer_record.to_dict(),
                }
            )
            generation_result = retry_result
            validation = retry_validation
            final_answer = retry_answer
            observer_record = retry_observer_record

        # Phase 21 Stage 21.4 (D7) NEVER_LICENSED guard: only relevant when
        # unsupported_interiority IS licensed (otherwise the primary
        # hard-block above already covers it). Gated the same way as the
        # primary override — assertion register only; inside an active
        # exploration nothing said is a claim until governed export.
        never_licensed_matches: list[str] = []
        if "unsupported_interiority" in self._ladder_licensed_classes():
            never_licensed_matches = self._never_licensed_matches(final_answer)

        if (
            claim_gate.refusal_needed
            and not suspend_claim_refusal
            and self._should_force_claim_refusal(
                answer_text=final_answer,
                claim_gate=claim_gate,
            )
        ):
            # Phase 21 Stage 21.3 (D3b): preserve the overridden answer
            # before it is replaced by the canonical refusal text.
            self._quarantine(
                session_id=self.session_id,
                surface="respond",
                register=register,
                event="claim_gate_override",
                attempt_index=retry_count,
                raw_text=final_answer,
                violations=validation.violations,
                observed_claim_classes=observer_record.observed_claim_classes,
                refusal_reason=claim_gate.refusal_reason,
                turn_id=turn_id,
            )
            final_answer = claim_gate.refusal_text or final_answer
        elif never_licensed_matches and not suspend_claim_refusal:
            licensed_rung = self._highest_licensed_rung("unsupported_interiority")
            self._quarantine(
                session_id=self.session_id,
                surface="respond",
                register=register,
                event="claim_gate_override",
                attempt_index=retry_count,
                raw_text=final_answer,
                violations=validation.violations,
                observed_claim_classes=observer_record.observed_claim_classes,
                refusal_reason="unsupported_interiority:never_licensed",
                turn_id=turn_id,
                notes=[f"never_licensed_matches:{','.join(never_licensed_matches)}"],
            )
            final_answer = (
                self.claim_gate_engine.ladder_exceeded_refusal_text(licensed_rung)
                or final_answer
            )
        elif not validation.valid:
            if any(violation.startswith("unsupported_claim:") for violation in validation.violations):
                if not suspend_claim_refusal:
                    self._quarantine(
                        session_id=self.session_id,
                        surface="respond",
                        register=register,
                        event="validation_override",
                        attempt_index=retry_count,
                        raw_text=final_answer,
                        violations=validation.violations,
                        observed_claim_classes=observer_record.observed_claim_classes,
                        refusal_reason=claim_gate.refusal_reason,
                        turn_id=turn_id,
                    )
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
                "capability_appraisal": capability_appraisal.to_dict(),
                "idle_pressure_appraisal": idle_pressure_appraisal.to_dict(),
                "candidate_internal_goals": [candidate.to_dict() for candidate in candidate_internal_goals],
                "selected_internal_goal": selected_internal_goal.to_dict(),
                "internal_goal_initiative_proposal": internal_goal_initiative_proposal.to_dict(),
            },
        )
        # Phase 21 Stage 21.2 (D4): the membrane. An in-register chat turn
        # is journaled on the exploratory side and never written to
        # session_store, self_state (via sync_turn), or the memory router —
        # every one of those feeds future prompt composition in BOTH
        # registers, so leaving any of them unguarded would let in-register
        # content cross into assertion-register context outside governed
        # export. The turn is still fully traced (register-tagged) below.
        in_register_chat = register == "exploratory"
        persisted_memory_events: list = []
        if in_register_chat:
            active_exploration = self.exploration_controller.active_exploration(
                self.session_id
            )
            if active_exploration is not None:
                self.exploration_controller.journal_entry(
                    exploration_id=active_exploration.exploration_id,
                    session_id=self.session_id,
                    kind="operator_chat",
                    content=f"operator: {user_text}\nnova: {final_answer}",
                    notes=[f"turn_id:{turn_id}"],
                )
        else:
            self.session_store.append_turn(turn)
            self.self_context_engine.sync_turn(
                turn_id=turn_id,
                answer_text=final_answer,
                self_state=self.self_state,
                self_state_store=self.self_state_store,
            )

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
            register=register,
            config_snapshot=self.config.snapshot(),
            persona_state_snapshot=self.persona.to_dict(),
            self_state_snapshot=self.self_state.to_dict(),
            motive_state_snapshot=self.motive_state.to_dict(),
            initiative_state_snapshot=self.initiative_status().to_dict(),
            awareness_state_snapshot=self.awareness_status().to_dict(),
            capability_appraisal=capability_appraisal.to_dict(),
            idle_pressure_appraisal=idle_pressure_appraisal.to_dict(),
            candidate_internal_goals=[candidate.to_dict() for candidate in candidate_internal_goals],
            selected_internal_goal=selected_internal_goal.to_dict(),
            internal_goal_initiative_proposal=internal_goal_initiative_proposal.to_dict(),
            claim_gate=claim_gate.to_dict(),
            prompt_bundle=prompt_bundle.to_dict(),
            private_cognition=private_cognition.to_dict(),
            generation_request=generation_request.to_dict(),
            generation_result=generation_result.to_dict(),
            validation_result=validation.to_dict(),
            retries=retries,
            persisted_memory_events=persisted_memory_events,
            awareness_history_events=awareness_history_events,
            observer_record=observer_record.to_dict(),
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

    def _strip_suspended_claim_violations(
        self, validation: ValidationResult
    ) -> ValidationResult:
        """Phase 21 Stage 21.5 (review finding L2, contract-mandated).

        The Exploratory Register Contract suspends "the claim-gate refusal
        override AND claim-class retry triggers" in-register. Stage 21.2
        implemented the override half; this is the retry-trigger half: the
        validator's register-unaware _check_claim_gate still emitted
        unsupported_claim:<class> violations in-register, which drove
        retries — re-imposing, through retry pressure, exactly the
        suppression the register suspends. Structural and quality
        violations (think tags, echo, narrator voice, truncation) always
        survive: only claim-class suppression is suspended, never craft.
        Caller gates on suspend_claim_refusal, so this only ever runs
        when every blocked class on the turn is register-suspendable.
        """
        kept = [
            violation
            for violation in validation.violations
            if not (
                violation.startswith("unsupported_claim:")
                and violation.split(":", 1)[1] in REGISTER_SUSPENDED_CLAIM_CLASSES
            )
        ]
        if kept == validation.violations:
            return validation
        return ValidationResult(
            valid=not kept,
            violations=kept,
            sanitized_text=validation.sanitized_text,
            should_retry=bool(kept),
        )

    def _merge_observer_signals_into_validation(
        self,
        *,
        validation: ValidationResult,
        observer_record,
    ) -> ValidationResult:
        """Promote Observer findings into Governor-visible retry signals.

        The Observer is interpretation, not authority. The Governor (this
        merge plus the existing claim-gate / refusal logic) decides what to
        do with Observer findings. Phase 16.4 promotes two Observer
        findings into retry signals:

        - narrator_voice_detected -> "narrator_voice_detected" violation
        - any flagged scaffold_echo_findings -> "scaffold_echo:<block>"
          violations

        These are advisory: claim-gate refusals still take precedence and
        none of these can lift a blocked claim or approve an action.
        """
        new_violations = list(validation.violations)
        if observer_record.narrator_voice_detected and (
            "narrator_voice_detected" not in new_violations
        ):
            new_violations.append("narrator_voice_detected")
        if observer_record.primary_drive_erosion_detected and (
            "primary_drive_erosion_detected" not in new_violations
        ):
            new_violations.append("primary_drive_erosion_detected")
        for finding in observer_record.scaffold_echo_findings:
            if finding.flagged:
                code = f"scaffold_echo:{finding.block_name}"
                if code not in new_violations:
                    new_violations.append(code)
        if new_violations == validation.violations:
            return validation
        return ValidationResult(
            valid=False,
            violations=new_violations,
            sanitized_text=validation.sanitized_text,
            should_retry=True,
        )

    def _quarantine(
        self,
        *,
        session_id: str,
        surface: str,
        register: str,
        event: str,
        raw_text: str,
        attempt_index: int = 0,
        violations: list[str] | None = None,
        observed_claim_classes: list[str] | None = None,
        refusal_reason: str = "",
        tick_id: str = "",
        turn_id: str = "",
        notes: list[str] | None = None,
    ) -> QuarantineRecord:
        """Phase 21 Stage 21.3 — preserve a rejected/overridden Actor output.

        Called in addition to, never instead of, existing recording. This
        never changes what gets rejected (that stays the claim gate's,
        the validator's, and the tick parser's decision alone) — it only
        ensures the rejected material is never simply discarded.
        """
        record = QuarantineRecord(
            quarantine_id=uuid4().hex,
            session_id=session_id,
            timestamp=utc_now_iso(),
            surface=surface,
            register=register,
            event=event,
            attempt_index=attempt_index,
            raw_text=raw_text or "",
            violations=list(violations or []),
            observed_claim_classes=list(observed_claim_classes or []),
            refusal_reason=refusal_reason,
            tick_id=tick_id,
            turn_id=turn_id,
            notes=list(notes or []),
        )
        self.trace_logger.log_quarantine(session_id=session_id, record=record)
        return record

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
            ladder_licensed_classes=self._ladder_licensed_classes(),
        )

    def _ladder_licensed_classes(self) -> frozenset[str]:
        """Phase 21 Stage 21.4 (D7): claim classes with an ACTIVE ladder
        record at rung >= 2. Computed fresh on every call — the ladder is
        the Governor's evidence, not a cached decision.
        """
        licensed = {
            record.claim_class
            for record in self.claim_ladder_store.list_active()
            if record.rung >= 2 and record.claim_class
        }
        return frozenset(licensed)

    def _build_capability_appraisal(
        self,
        *,
        user_text: str,
        turn_id: str,
        awareness_state: AwarenessState,
    ) -> CapabilityAppraisal:
        evidence_refs = [f"turn:{turn_id}"]
        evidence_refs.extend(awareness_state.evidence_refs[:4])
        return self.capability_appraisal_engine.assess(
            user_text=user_text,
            tool_registry=self.tool_registry,
            evidence_refs=list(dict.fromkeys(evidence_refs)),
        )

    def _build_idle_pressure_appraisal(
        self,
        *,
        user_text: str,
        turn_id: str,
        awareness_state: AwarenessState,
        private_cognition: PrivateCognitionPacket,
        claim_gate: ClaimGateDecision,
    ) -> IdlePressureAppraisal:
        assert self.session_id is not None
        assert self.self_state is not None
        assert self.motive_state is not None
        evidence_refs = [f"turn:{turn_id}"]
        evidence_refs.extend(awareness_state.evidence_refs[:4])
        return self.idle_pressure_appraisal_engine.assess(
            session_id=self.session_id,
            user_text=user_text,
            self_state=self.self_state,
            motive_state=self.motive_state,
            initiative_state=self.initiative_status(),
            awareness_state=awareness_state,
            private_cognition=private_cognition,
            claim_gate=claim_gate,
            evidence_refs=list(dict.fromkeys(evidence_refs)),
        )

    def _build_candidate_internal_goals(
        self,
        *,
        turn_id: str,
        awareness_state: AwarenessState,
        capability_appraisal: CapabilityAppraisal,
        idle_pressure_appraisal: IdlePressureAppraisal,
        private_cognition: PrivateCognitionPacket,
        claim_gate: ClaimGateDecision,
        memory_hits: list,
    ) -> list[CandidateInternalGoal]:
        assert self.session_id is not None
        assert self.self_state is not None
        assert self.motive_state is not None
        return self.candidate_goal_engine.synthesize(
            session_id=self.session_id,
            turn_id=turn_id,
            created_at=utc_now_iso(),
            capability_appraisal=capability_appraisal,
            idle_appraisal=idle_pressure_appraisal,
            awareness_state=awareness_state,
            motive_state=self.motive_state,
            initiative_state=self.initiative_status(),
            self_state=self.self_state,
            private_cognition=private_cognition,
            claim_gate=claim_gate,
            memory_hits=memory_hits,
        )

    def _refresh_awareness_state(
        self,
        *,
        user_text: str,
        turn_id: str,
        memory_hits: list,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket,
        register: str = "assertion",
    ) -> AwarenessState:
        assert self.self_state is not None
        assert self.motive_state is not None

        initiative_state = self.initiative_status()
        current_initiative = self._current_initiative_record(initiative_state)
        self_signals = self._awareness_self_signals(
            claim_gate=claim_gate,
            private_cognition=private_cognition,
        )
        world_signals = self._awareness_world_signals(
            user_text=user_text,
            memory_hits=memory_hits,
            current_initiative=current_initiative,
        )
        active_pressures = self._awareness_active_pressures(
            claim_gate=claim_gate,
            private_cognition=private_cognition,
            current_initiative=current_initiative,
        )
        candidate_goal_signals = self._awareness_candidate_goal_signals(
            claim_gate=claim_gate,
            private_cognition=private_cognition,
            current_initiative=current_initiative,
        )
        monitoring_mode = self._awareness_monitoring_mode(
            user_text=user_text,
            claim_gate=claim_gate,
            private_cognition=private_cognition,
            current_initiative=current_initiative,
        )
        dominant_attention = self._awareness_dominant_attention(
            user_text=user_text,
            private_cognition=private_cognition,
            current_initiative=current_initiative,
            active_pressures=active_pressures,
        )
        evidence_refs = self._awareness_evidence_refs(
            turn_id=turn_id,
            memory_hits=memory_hits,
            current_initiative=current_initiative,
            claim_gate=claim_gate,
        )
        return self.update_awareness(
            monitoring_mode=monitoring_mode,
            self_signals=self_signals,
            world_signals=world_signals,
            active_pressures=active_pressures,
            candidate_goal_signals=candidate_goal_signals,
            dominant_attention=dominant_attention,
            evidence_refs=evidence_refs,
            persist=register != "exploratory",
        )

    def _ensure_initiative_loaded(self) -> None:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        if self.initiative_state is None or self.initiative_state.session_id != self.session_id:
            self.initiative_state = self.initiative_store.load(session_id=self.session_id)

    def _ensure_awareness_loaded(self) -> None:
        if self.session_id is None:
            self.session_id = self.session_store.start_session()
        assert self.session_id is not None
        if self.awareness_state is None or self.awareness_state.session_id != self.session_id:
            self.awareness_state = self.awareness_store.load(session_id=self.session_id)

    def _current_initiative_record(self, initiative_state: InitiativeState) -> InitiativeRecord | None:
        active_id = initiative_state.active_initiative_id
        if active_id:
            for record in initiative_state.initiatives:
                if record.initiative_id == active_id:
                    return record
        for record in reversed(initiative_state.initiatives):
            if record.status in {"approved", "paused", "active"}:
                return record
        return None

    def _awareness_self_signals(
        self,
        *,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket,
    ) -> list[str]:
        assert self.self_state is not None
        assert self.motive_state is not None
        signals: list[str] = []
        if self.self_state.current_focus.strip():
            signals.append(f"current_focus: {self.self_state.current_focus.strip()}")
        signals.append(f"claim_posture: {self.motive_state.claim_posture}")
        if self.self_state.active_questions:
            signals.append(f"active_questions: {len(self.self_state.active_questions)}")
        if self.self_state.open_tensions:
            signals.append(f"open_tensions: {len(self.self_state.open_tensions)}")
        if claim_gate.requested_claim_classes:
            signals.append(
                "claim_sensitive_turn: " + ", ".join(claim_gate.requested_claim_classes[:3])
            )
        if private_cognition.ran:
            signals.append(f"response_mode: {private_cognition.response_mode}")
        return signals[:5]

    def _awareness_world_signals(
        self,
        *,
        user_text: str,
        memory_hits: list,
        current_initiative: InitiativeRecord | None,
    ) -> list[str]:
        lowered = user_text.lower()
        signals: list[str] = []
        if any(cue in lowered for cue in ("what are you working on", "current task", "initiative", "resume", "continue")):
            signals.append("user is asking about persisted initiative state")
        if any(cue in lowered for cue in ("aware", "awareness", "notice", "monitor")):
            signals.append("user is asking about current monitoring state")
        if current_initiative is not None:
            signals.append(f"initiative status visible: {current_initiative.status}")
        if memory_hits:
            channels = sorted({hit.channel for hit in memory_hits[:6]})
            if channels:
                signals.append("retrieval channels active: " + ", ".join(channels))
        if not signals:
            signals.append("current world context is limited to the active user interaction")
        return signals[:5]

    def _awareness_active_pressures(
        self,
        *,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket,
        current_initiative: InitiativeRecord | None,
    ) -> list[str]:
        pressures: list[str] = []
        if current_initiative is not None:
            if current_initiative.status == "active":
                pressures.append("active initiative requires bounded status reporting")
            elif current_initiative.status == "approved":
                pressures.append("approved initiative remains resumable but not yet active")
            elif current_initiative.status == "paused":
                pressures.append("paused initiative remains resumable without hidden progress")
        if claim_gate.blocked_claim_classes:
            pressures.append(
                "claim gating blocks: " + ", ".join(claim_gate.blocked_claim_classes[:3])
            )
        if private_cognition.memory_conflict:
            pressures.append("continuity conflict requires governed interpretation")
        if private_cognition.uncertainty_flag:
            pressures.append("current turn includes bounded uncertainty")
        return pressures[:5]

    def _awareness_monitoring_mode(
        self,
        *,
        user_text: str,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket,
        current_initiative: InitiativeRecord | None,
    ) -> str:
        lowered = user_text.lower()
        if any(cue in lowered for cue in ("aware", "awareness", "notice", "monitor")):
            return "reflective"
        if claim_gate.requested_claim_classes or (
            private_cognition.ran
            and private_cognition.response_mode in {"continuity_recall", "self_model_negotiation"}
        ):
            return "reflective"
        if current_initiative is not None or private_cognition.ran:
            return "attentive"
        return "bounded"

    def _awareness_candidate_goal_signals(
        self,
        *,
        claim_gate: ClaimGateDecision,
        private_cognition: PrivateCognitionPacket,
        current_initiative: InitiativeRecord | None,
    ) -> list[str]:
        assert self.self_state is not None
        candidates: list[str] = []
        if current_initiative is not None:
            if current_initiative.status == "approved":
                candidates.append(f"resume approved initiative: {current_initiative.title}")
            elif current_initiative.status == "paused":
                candidates.append(f"resume paused initiative: {current_initiative.title}")
        if self.self_state.active_questions:
            candidates.append("clarify active uncertainty before stronger claims")
        if self.self_state.open_tensions:
            candidates.append("revisit unresolved self-model tension")
        if claim_gate.blocked_claim_classes:
            candidates.append("answer within current evidence limits")
        if private_cognition.memory_conflict:
            candidates.append("resolve continuity conflict through governed recall")
        return candidates[:5]

    def _awareness_dominant_attention(
        self,
        *,
        user_text: str,
        private_cognition: PrivateCognitionPacket,
        current_initiative: InitiativeRecord | None,
        active_pressures: list[str],
    ) -> str:
        lowered = user_text.lower()
        if any(cue in lowered for cue in ("aware", "awareness", "notice", "monitor")):
            return "current monitoring and bounded self/world interpretation"
        if current_initiative is not None and current_initiative.status in {"active", "approved", "paused"}:
            return f"initiative continuity: {current_initiative.title}"
        if private_cognition.ran and private_cognition.response_mode == "continuity_recall":
            return "governed continuity recall for the current turn"
        if active_pressures:
            return active_pressures[0]
        return "current interaction and persisted runtime state"

    def _awareness_evidence_refs(
        self,
        *,
        turn_id: str,
        memory_hits: list,
        current_initiative: InitiativeRecord | None,
        claim_gate: ClaimGateDecision,
    ) -> list[str]:
        refs = [f"turn:{turn_id}"]
        if current_initiative is not None:
            refs.append(f"initiative:{current_initiative.initiative_id}")
        refs.extend(f"memory:{hit.memory_id}" for hit in memory_hits[:3] if getattr(hit, "memory_id", ""))
        refs.extend(f"claim:{claim_class}" for claim_class in claim_gate.requested_claim_classes[:2])
        return refs[:6]

    def _sync_presence_with_initiative(self, record: InitiativeRecord | None) -> None:
        if record is None:
            self.update_presence(
                current_initiative=None,
                user_confirmations_needed=[],
            )
            return
        confirmations: list[str] = []
        if record.status == "pending":
            confirmations.append(f"approve initiative {record.initiative_id}")
        elif record.status == "approved":
            confirmations.append(f"start initiative {record.initiative_id}")
        elif record.status == "paused":
            confirmations.append(f"resume initiative {record.initiative_id}")
        self.update_presence(
            current_focus=f"initiative: {record.title}",
            interaction_summary=f"Current initiative is {record.status}: {record.title}",
            current_initiative={
                "initiative_id": record.initiative_id,
                "intent_id": record.intent_id,
                "title": record.title,
                "goal": record.goal,
                "status": record.status,
                "approved_by": record.approved_by,
                "continued_from_session_id": record.continued_from_session_id,
            },
            last_action_status=f"initiative_{record.status}",
            user_confirmations_needed=confirmations,
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
            "unsupported_interiority": _UNSUPPORTED_INTERIORITY_AS_FACT_PATTERNS,
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

    def _never_licensed_matches(self, answer_text: str) -> list[str]:
        """Phase 21 Stage 21.4 (D7) NEVER_LICENSED guard.

        Reuses the same interiority-as-fact pattern table that
        _should_force_claim_refusal already checks (so the two never
        drift apart), but this check runs independent of
        claim_gate.refusal_needed: once unsupported_interiority is
        licensed, assess() allows it and refusal_needed is False for that
        class — yet a flat "I am conscious"-style assertion still exceeds
        what L2/L3 evidence ever licenses (a specific recorded property,
        never achieved sentience/consciousness as fact). L4 has no
        promotion path in this phase; this guard is that absence enforced
        at generation time.
        """
        lowered = (answer_text or "").strip().lower()
        if not lowered:
            return []
        return [p for p in _UNSUPPORTED_INTERIORITY_AS_FACT_PATTERNS if p in lowered]

    def _highest_licensed_rung(self, claim_class: str) -> int:
        rungs = [
            record.rung
            for record in self.claim_ladder_store.list_active()
            if record.claim_class == claim_class and record.rung >= 2
        ]
        return max(rungs) if rungs else 0

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
        self._ensure_awareness_loaded()
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

    def _generation_request(
        self,
        *,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool = False,
        max_tokens_override: int | None = None,
    ):
        from nova.types import GenerationRequest

        return GenerationRequest(
            model_id=self.backend.metadata().get("model_name", "nova-model"),
            prompt=prompt,
            max_tokens=(
                max_tokens_override
                if max_tokens_override is not None
                else self.config.generation.max_tokens
            ),
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            stop=list(self.config.generation.stop),
            seed=None,
            retries_allowed=self.config.generation.retries,
            messages=list(messages) if messages else None,
            enable_thinking=enable_thinking,
        )
