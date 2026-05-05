"""Nova runtime orchestration."""

from __future__ import annotations

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
from nova.agent.claims import ClaimGateEngine
from nova.agent.awareness import JsonAwarenessStateStore
from nova.agent.awareness_prompt import AwarenessPromptEngine
from nova.agent.idle import BoundedIdleController, IdleRuntimePromptEngine, JsonIdleRuntimeStore
from nova.agent.initiative import JsonInitiativeStateStore
from nova.agent.initiative_prompt import InitiativePromptEngine
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
from nova.agent.action_plan import (
    BoundedActionPlanEngine,
    default_nova_owned_execution_boundary,
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
from nova.types import (
    AwarenessState,
    CapabilityAppraisal,
    CandidateInternalGoal,
    ClaimGateDecision,
    IdleBudget,
    IdlePressureAppraisal,
    IdleRuntimeStatus,
    IdleTickRecord,
    InternalGoalInitiativeProposal,
    MotiveState,
    AutonomousActionBudget,
    AutonomousActionPlan,
    AutonomousActionPlanStep,
    PrivateCognitionPacket,
    SelectedInternalGoal,
    TraceRecord,
    TurnRecord,
    ValidationResult,
)
from nova.types import InitiativeRecord, InitiativeState


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
        initiative_store: JsonInitiativeStateStore,
        awareness_store: JsonAwarenessStateStore,
        presence_store: JsonPresenceStore,
        session_store: JsonlSessionStore,
        trace_logger: JsonlTraceLogger,
        memory_router: BasicMemoryRouter,
        memory_event_factory: BasicMemoryEventFactory,
        idle_store: JsonIdleRuntimeStore | None = None,
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
        tool_registry: ToolRegistry | None = None,
        action_plan_engine: BoundedActionPlanEngine | None = None,
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
    ) -> AwarenessState:
        awareness = self.awareness_status()
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
        )
        awareness_history_events = [
            entry.to_dict() for entry in self.awareness_store.consume_recent_history_entries()
        ]
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
        prompt_bundle = self.composer.compose(
            persona=self.persona,
            self_state=self.self_state,
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
                "capability_appraisal": capability_appraisal.to_dict(),
                "idle_pressure_appraisal": idle_pressure_appraisal.to_dict(),
                "candidate_internal_goals": [candidate.to_dict() for candidate in candidate_internal_goals],
                "selected_internal_goal": selected_internal_goal.to_dict(),
                "internal_goal_initiative_proposal": internal_goal_initiative_proposal.to_dict(),
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
