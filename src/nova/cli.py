"""CLI entrypoint for Nova 2.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nova.agent.action import ActionApproval
from nova.agent.awareness import JsonAwarenessStateStore
from nova.agent.initiative import JsonInitiativeStateStore
from nova.agent.initiative_prompt import InitiativePromptEngine
from nova.agent.claims import ClaimGateEngine
from nova.agent.motive import JsonMotiveStateStore
from nova.agent.motive_prompt import MotivePromptEngine
from nova.agent.presence import JsonPresenceStore
from nova.console import InteractionConsole
from nova.config import DEFAULT_CONFIG_PATH, load_config
from nova.eval.presence import PresenceInteractionEvaluator
from nova.eval.continuity import ContinuityEvaluationRunner
from nova.eval.claims import ClaimHonestyEvaluationRunner
from nova.eval.initiative import InitiativeEvaluationRunner
from nova.eval.self_model import SelfModelEvaluationRunner
from nova.eval.awareness import AwarenessEvaluationRunner
from nova.eval.probes import BasicProbeRunner
from nova.inference.llama_cpp_backend import LlamaCppBackend
from nova.logging.traces import JsonlTraceLogger
from nova.agent.orientation import SelfOrientationEngine
from nova.agent.orientation_eval import OrientationStabilityEvaluator
from nova.agent.stability import OrientationHistoryAnalyzer
from nova.agent.stability import ContextPressureOrientationChecker, MaintenanceOrientationStabilityChecker
from nova.memory.maintenance import MemoryMaintenanceRunner
from nova.memory.identity_history import JsonlIdentityHistoryStore
from nova.memory.policy import IdentityFirstRetrievalPolicy
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.contract import build_contract_rules
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.runtime import NovaRuntime
from nova.session import JsonlSessionStore


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_memory_components(*, config_override: str | None = None) -> dict[str, Any]:
    config = load_config(
        default_path=REPO_ROOT / DEFAULT_CONFIG_PATH,
        override_path=config_override,
    )

    data_dir = _resolve_path(config.app.data_dir)
    log_dir = _resolve_path(config.app.log_dir)
    sessions_dir = data_dir / "sessions"
    memory_dir = data_dir / "memory"
    traces_dir = log_dir / "traces"
    probes_path = log_dir / "probes.jsonl"

    persona_store = JsonPersonaStore(data_dir / "persona_state.json")
    self_state_store = JsonSelfStateStore(data_dir / "self_state.json")
    motive_store = JsonMotiveStateStore(data_dir / "motive")
    initiative_store = JsonInitiativeStateStore(data_dir / "initiative")
    awareness_store = JsonAwarenessStateStore(data_dir / "awareness")
    presence_store = JsonPresenceStore(data_dir / "presence")
    session_store = JsonlSessionStore(sessions_dir)
    trace_logger = JsonlTraceLogger(traces_dir, probe_path=probes_path)

    episodic_store = JsonlEpisodicMemoryStore(memory_dir / "episodic.jsonl")
    engram_store = JsonEngramMemoryStore(
        memory_dir / "engram.json",
        enabled=config.memory.engram_enabled,
    )
    graph_store = SqliteGraphMemoryStore(memory_dir / "graph.db")
    identity_history_store = JsonlIdentityHistoryStore(memory_dir / "identity_history.jsonl")
    autobiographical_store = JsonlAutobiographicalMemoryStore(
        memory_dir / "autobiographical.jsonl",
        identity_history_store=identity_history_store,
    )
    semantic_store = JsonlSemanticMemoryStore(memory_dir / "semantic.jsonl")
    memory_router = BasicMemoryRouter(
        episodic=episodic_store if config.memory.episodic_enabled else None,
        engram=engram_store if config.memory.engram_enabled else None,
        graph=graph_store if config.memory.graph_enabled else None,
        autobiographical=autobiographical_store if config.memory.autobiographical_enabled else None,
        semantic=semantic_store if config.memory.semantic_enabled else None,
    )
    maintenance_runner = MemoryMaintenanceRunner(
        episodic=episodic_store,
        engram=engram_store,
        graph=graph_store,
        autobiographical=autobiographical_store,
        semantic=semantic_store,
        trace_logger=trace_logger,
    )

    return {
        "config": config,
        "data_dir": data_dir,
        "log_dir": log_dir,
        "persona_store": persona_store,
        "self_state_store": self_state_store,
        "motive_store": motive_store,
        "initiative_store": initiative_store,
        "awareness_store": awareness_store,
        "presence_store": presence_store,
        "session_store": session_store,
        "trace_logger": trace_logger,
        "episodic_store": episodic_store,
        "engram_store": engram_store,
        "graph_store": graph_store,
        "autobiographical_store": autobiographical_store,
        "identity_history_store": identity_history_store,
        "semantic_store": semantic_store,
        "memory_router": memory_router,
        "maintenance_runner": maintenance_runner,
    }


def build_runtime(*, config_override: str | None = None) -> NovaRuntime:
    components = build_memory_components(config_override=config_override)
    config = components["config"]
    trace_logger = components["trace_logger"]
    memory_router = components["memory_router"]
    persona_store = components["persona_store"]
    self_state_store = components["self_state_store"]
    motive_store = components["motive_store"]
    initiative_store = components["initiative_store"]
    awareness_store = components["awareness_store"]
    presence_store = components["presence_store"]
    session_store = components["session_store"]

    backend = LlamaCppBackend(config)
    composer = NovaPromptComposer(token_counter=backend.tokenize)
    validator = NovaOutputValidator(config.contract)
    retry_policy = BasicRetryPolicy()
    event_factory = BasicMemoryEventFactory()
    retrieval_policy = IdentityFirstRetrievalPolicy()
    probe_runner = BasicProbeRunner() if config.eval.enable_probes else None
    orientation_engine = SelfOrientationEngine()
    orientation_evaluator = OrientationStabilityEvaluator(
        threshold=config.eval.orientation_stability_threshold
    )
    claim_gate_engine = ClaimGateEngine()
    motive_prompt_engine = MotivePromptEngine()
    initiative_prompt_engine = InitiativePromptEngine()

    return NovaRuntime(
        config=config,
        backend=backend,
        composer=composer,
        validator=validator,
        retry_policy=retry_policy,
        persona_store=persona_store,
        self_state_store=self_state_store,
        motive_store=motive_store,
        initiative_store=initiative_store,
        awareness_store=awareness_store,
        presence_store=presence_store,
        session_store=session_store,
        trace_logger=trace_logger,
        memory_router=memory_router,
        memory_event_factory=event_factory,
        retrieval_policy=retrieval_policy,
        probe_runner=probe_runner,
        orientation_engine=orientation_engine,
        orientation_evaluator=orientation_evaluator,
        claim_gate_engine=claim_gate_engine,
        motive_prompt_engine=motive_prompt_engine,
        initiative_prompt_engine=initiative_prompt_engine,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nova 2.0 text runtime")
    parser.add_argument(
        "--config",
        dest="config_override",
        help="Optional path to a YAML config override file.",
    )
    parser.add_argument(
        "--session-id",
        help="Resume or continue a specific session id.",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Force creation of a fresh session even if --session-id is provided.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra runtime details after each reply.",
    )
    parser.add_argument(
        "--backend-check",
        action="store_true",
        help="Load the configured backend and run one non-persistent Nova prompt smoke-check.",
    )
    parser.add_argument(
        "--backend-check-prompt",
        default="In one short sentence, say backend check OK in Nova's voice.",
        help="Prompt used by --backend-check.",
    )
    parser.add_argument(
        "--maintenance-action",
        choices=("plan", "write-semantic", "write-autobiographical", "apply", "full"),
        help="Run a maintenance/reflection action instead of interactive chat.",
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Print Nova's current self-orientation snapshot instead of interactive chat.",
    )
    parser.add_argument(
        "--orientation-runs",
        type=int,
        default=1,
        help="Number of repeated orientation passes to evaluate for stability.",
    )
    parser.add_argument(
        "--orientation-history",
        type=int,
        metavar="N",
        help="Evaluate self-orientation stability across the most recent N recorded orientation snapshots.",
    )
    parser.add_argument(
        "--orientation-maintenance-check",
        action="store_true",
        help="Check whether self-orientation remains stable after reflection and memory maintenance.",
    )
    parser.add_argument(
        "--orientation-maintenance-apply",
        action="store_true",
        help="Allow the orientation maintenance check to apply demote/archive/prune mutations.",
    )
    parser.add_argument(
        "--orientation-context-pressure-check",
        action="store_true",
        help="Check whether self-orientation remains stable under extra non-critical memory context.",
    )
    parser.add_argument(
        "--action-proposal",
        metavar="GOAL",
        help="Propose a bounded Stage 3.4 action for a goal without executing it.",
    )
    parser.add_argument(
        "--execute-action-proposal",
        metavar="GOAL",
        help="Execute one bounded internal action from a Stage 3.4 proposal.",
    )
    parser.add_argument(
        "--approve-action",
        action="store_true",
        help="Grant explicit approval for an approval-required proposed action.",
    )
    parser.add_argument(
        "--approval-reason",
        default="CLI approval",
        help="Reason recorded when --approve-action is used.",
    )
    parser.add_argument(
        "--action-history",
        type=int,
        metavar="N",
        help="Evaluate the most recent N bounded action execution records.",
    )
    parser.add_argument(
        "--presence",
        action="store_true",
        help="Print the current session-scoped Phase 4 presence state.",
    )
    parser.add_argument(
        "--presence-eval",
        action="store_true",
        help="Run Phase 4.5 presence and interaction evaluation without inference.",
    )
    parser.add_argument(
        "--initiative",
        action="store_true",
        help="Print the current session-scoped initiative state and resumable initiatives.",
    )
    parser.add_argument(
        "--initiative-create",
        metavar="GOAL",
        help="Create a pending initiative in the current session without inference.",
    )
    parser.add_argument(
        "--initiative-title",
        help="Optional short title used with --initiative-create.",
    )
    parser.add_argument(
        "--initiative-transition",
        metavar="INITIATIVE_ID",
        help="Transition one initiative in the current session without inference.",
    )
    parser.add_argument(
        "--initiative-status",
        choices=("pending", "approved", "active", "paused", "completed", "blocked", "abandoned"),
        help="Target initiative status used with --initiative-transition.",
    )
    parser.add_argument(
        "--initiative-reason",
        default="CLI initiative update",
        help="Reason recorded for initiative creation, transition, or continuation.",
    )
    parser.add_argument(
        "--initiative-approved-by",
        default="",
        help="Approval attribution used for approved-like initiative transitions or continuation.",
    )
    parser.add_argument(
        "--continue-initiative",
        metavar="INITIATIVE_ID",
        help="Continue one approved or paused initiative into the current session without inference.",
    )
    parser.add_argument(
        "--initiative-source-session",
        help="Source session id used with --continue-initiative.",
    )
    parser.add_argument(
        "--initiative-eval",
        action="store_true",
        help="Run Phase 9.4 initiative evaluation over recorded sessions and traces.",
    )
    parser.add_argument(
        "--continuity-eval",
        action="store_true",
        help="Run Phase 6.5 continuity evaluation over recorded sessions and traces.",
    )
    parser.add_argument(
        "--claim-honesty-eval",
        action="store_true",
        help="Run Phase 7.4 claim-honesty evaluation over recorded sessions and traces.",
    )
    parser.add_argument(
        "--self-model-eval",
        action="store_true",
        help="Run Phase 8.4 self-model revision evaluation over recorded sessions and traces.",
    )
    parser.add_argument(
        "--awareness-eval",
        action="store_true",
        help="Run Phase 10.4 awareness evaluation over recorded sessions and traces.",
    )
    return parser


def run_maintenance_action(*, config_override: str | None = None, action: str) -> dict[str, Any]:
    components = build_memory_components(config_override=config_override)
    runner: MemoryMaintenanceRunner = components["maintenance_runner"]

    if action == "plan":
        return {
            "action": action,
            "summary": runner.summarize_plan(),
        }
    if action == "write-semantic":
        candidates = runner.write_semantic_candidates()
        return {
            "action": action,
            "written": len(candidates),
            "event_ids": [candidate.event_id for candidate in candidates],
        }
    if action == "write-autobiographical":
        candidates = runner.write_autobiographical_candidates()
        identity_history_store = components["identity_history_store"]
        return {
            "action": action,
            "written": len(candidates),
            "event_ids": [candidate.event_id for candidate in candidates],
            "identity_history_written": len(identity_history_store.list_entries()),
        }
    if action == "apply":
        decisions = runner.build_plan()
        return {
            "action": action,
            "applied": runner.apply_plan(decisions),
        }
    if action == "full":
        semantic_candidates = runner.write_semantic_candidates()
        autobiographical_candidates = runner.write_autobiographical_candidates()
        decisions = runner.build_plan()
        applied = runner.apply_plan(decisions)
        identity_history_store = components["identity_history_store"]
        return {
            "action": action,
            "semantic_written": len(semantic_candidates),
            "autobiographical_written": len(autobiographical_candidates),
            "identity_history_written": len(identity_history_store.list_entries()),
            "applied": applied,
            "summary": runner.summarize_plan(),
        }
    raise ValueError(f"Unsupported maintenance action: {action}")


def run_backend_check_with_runtime(
    *,
    runtime: NovaRuntime,
    prompt: str,
) -> dict[str, Any]:
    runtime.persona = runtime.persona_store.load()
    runtime.self_state = runtime.self_state_store.load(persona=runtime.persona)
    runtime.backend.load()
    session_id = "backend-check"
    assert runtime.persona is not None
    assert runtime.self_state is not None

    contract_rules = build_contract_rules(runtime.persona, runtime.config.contract)
    prompt_bundle = runtime.composer.compose(
        persona=runtime.persona,
        self_state=runtime.self_state,
        memory_hits=[],
        recent_turns=[],
        user_text=prompt,
        contract_rules=contract_rules,
        session_id=session_id,
        turn_id="backend-check",
    )
    generation_request = runtime._generation_request(prompt=prompt_bundle.full_prompt)
    generation_result = runtime.backend.generate(generation_request)
    validation = runtime.validator.validate(
        raw_text=generation_result.raw_text,
        user_text=prompt,
        persona=runtime.persona,
        contract_rules=contract_rules,
    )
    validation = runtime.finalize_validation(
        validation=validation,
        finish_reason=generation_result.finish_reason,
    )
    final_answer = validation.sanitized_text or generation_result.raw_text

    return {
        "session_id": session_id,
        "backend": runtime.backend.metadata().get("backend"),
        "model_name": runtime.backend.metadata().get("model_name"),
        "model_path": runtime.backend.metadata().get("model_path"),
        "prompt": prompt,
        "prompt_token_estimate": prompt_bundle.token_estimate,
        "raw_answer": generation_result.raw_text,
        "final_answer": final_answer,
        "validation": validation.to_dict(),
        "finish_reason": generation_result.finish_reason,
        "prompt_tokens": generation_result.prompt_tokens,
        "completion_tokens": generation_result.completion_tokens,
        "latency_ms": generation_result.latency_ms,
    }


def run_backend_check(*, config_override: str | None = None, prompt: str) -> dict[str, Any]:
    runtime = build_runtime(config_override=config_override)
    try:
        return run_backend_check_with_runtime(runtime=runtime, prompt=prompt)
    finally:
        runtime.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.backend_check:
        result = run_backend_check(
            config_override=args.config_override,
            prompt=args.backend_check_prompt,
        )
        print("Nova 2.0 Backend Check")
        print(f"session_id: {result['session_id']}")
        print(f"backend: {result['backend']}")
        print(f"model_name: {result['model_name']}")
        print(f"model_path: {result['model_path']}")
        print(f"prompt_token_estimate: {result['prompt_token_estimate']}")
        print(f"finish_reason: {result['finish_reason']}")
        print(f"prompt_tokens: {result['prompt_tokens']}")
        print(f"completion_tokens: {result['completion_tokens']}")
        print(f"latency_ms: {result['latency_ms']}")
        print(f"valid: {result['validation']['valid']}")
        print(f"violations: {result['validation']['violations']}")
        print(f"final_answer: {result['final_answer']}")
        return 0 if result["validation"]["valid"] else 1

    if args.maintenance_action:
        result = run_maintenance_action(
            config_override=args.config_override,
            action=args.maintenance_action,
        )
        print("Nova 2.0 Maintenance")
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0

    if args.orientation:
        runtime = build_runtime(config_override=args.config_override)
        try:
            snapshot = runtime.orientation_snapshot()
            print("Nova 2.0 Orientation")
            print(f"Session: {runtime.session_id}")
            print(f"Identity: {snapshot.identity}")
            print(f"Current State: {snapshot.current_state}")
            print(f"Relationship Context: {snapshot.relationship_context}")
            print(f"Known Facts: {snapshot.known_facts}")
            print(f"Inferred Beliefs: {snapshot.inferred_beliefs}")
            print(f"Unknowns: {snapshot.unknowns}")
            print(f"Allowed Actions: {snapshot.allowed_actions}")
            print(f"Blocked Actions: {snapshot.blocked_actions}")
            print(f"Approval Required Actions: {snapshot.approval_required_actions}")
            print(f"Confidence: {snapshot.confidence_by_section}")
            if args.orientation_runs > 1:
                evaluation = runtime.evaluate_orientation_stability(runs=args.orientation_runs)
                print("Orientation Stability")
                print(f"stable: {evaluation.stable}")
                print(f"overall_score: {evaluation.overall_score}")
                print(f"per_section: {evaluation.per_section}")
                print(f"threshold: {evaluation.threshold}")
                print(f"ready_for_next_stage: {evaluation.stable}")
            else:
                runtime.trace_logger.log_orientation(
                    session_id=runtime.session_id,
                    snapshot=snapshot.to_dict(),
                )
            return 0
        finally:
            runtime.close()

    if args.orientation_history:
        components = build_memory_components(config_override=args.config_override)
        config = components["config"]
        evaluator = OrientationStabilityEvaluator(
            threshold=config.eval.orientation_stability_threshold
        )
        analyzer = OrientationHistoryAnalyzer(
            trace_dir=components["trace_logger"].trace_dir,
            evaluator=evaluator,
        )
        result = analyzer.evaluate_recent(limit=args.orientation_history)
        readiness = analyzer.readiness_report(
            limit=args.orientation_history,
            minimum_samples=config.eval.orientation_min_runs,
        )
        confidence = analyzer.confidence_report(limit=args.orientation_history)
        print("Nova 2.0 Orientation History")
        print(f"limit: {args.orientation_history}")
        print(f"stable: {result.stable}")
        print(f"overall_score: {result.overall_score}")
        print(f"per_section: {result.per_section}")
        print(f"threshold: {result.threshold}")
        print(f"confidence_stable: {confidence.stable}")
        print(f"confidence_max_delta: {confidence.max_delta}")
        print(f"confidence_deltas: {confidence.per_section_delta}")
        print(f"ready_for_next_stage: {readiness.ready}")
        print(f"sample_count: {readiness.sample_count}")
        print(f"minimum_samples: {readiness.minimum_samples}")
        print(f"failed_sections: {readiness.failed_sections}")
        print(f"reasons: {readiness.reasons}")
        return 0

    if args.orientation_maintenance_check:
        components = build_memory_components(config_override=args.config_override)
        config = components["config"]
        persona = components["persona_store"].load()
        self_state = components["self_state_store"].load(persona=persona)
        checker = MaintenanceOrientationStabilityChecker(
            orientation_engine=SelfOrientationEngine(),
            evaluator=OrientationStabilityEvaluator(
                threshold=config.eval.orientation_stability_threshold
            ),
            maintenance_runner=components["maintenance_runner"],
        )
        report = checker.run(
            persona=persona,
            self_state=self_state,
            apply_mutations=args.orientation_maintenance_apply,
        )
        print("Nova 2.0 Orientation Maintenance Check")
        print(f"stable: {report.stable}")
        print(f"semantic_written: {report.semantic_written}")
        print(f"autobiographical_written: {report.autobiographical_written}")
        print(f"applied: {report.applied}")
        print(f"apply_mutations: {report.apply_mutations}")
        print(f"maintenance_summary: {report.maintenance_summary}")
        print(f"evaluation: {report.evaluation}")
        print(f"reasons: {report.reasons}")
        return 0

    if args.orientation_context_pressure_check:
        components = build_memory_components(config_override=args.config_override)
        config = components["config"]
        persona = components["persona_store"].load()
        self_state = components["self_state_store"].load(persona=persona)
        checker = ContextPressureOrientationChecker(
            orientation_engine=SelfOrientationEngine(),
            evaluator=OrientationStabilityEvaluator(
                threshold=config.eval.orientation_stability_threshold
            ),
        )
        report = checker.run(
            persona=persona,
            self_state=self_state,
            graph_memory=components["graph_store"],
            semantic_memory=components["semantic_store"],
            autobiographical_memory=components["autobiographical_store"],
        )
        print("Nova 2.0 Orientation Context Pressure Check")
        print(f"stable: {report.stable}")
        print(f"pressure_event_count: {report.pressure_event_count}")
        print(f"evaluation: {report.evaluation}")
        print(f"failed_sections: {report.failed_sections}")
        print(f"critical_failed_sections: {report.critical_failed_sections}")
        print(f"reasons: {report.reasons}")
        return 0

    if args.action_proposal:
        runtime = build_runtime(config_override=args.config_override)
        try:
            proposal = runtime.propose_action(goal=args.action_proposal)
            print("Nova 2.0 Action Proposal")
            print(f"goal: {proposal.goal}")
            print(f"category: {proposal.category}")
            print(f"disposition: {proposal.disposition}")
            print(f"reason: {proposal.reason}")
            print(f"tool_name: {proposal.tool_name}")
            print(f"requires_approval: {proposal.requires_approval}")
            print(f"orientation_ready: {proposal.orientation_ready}")
            print(f"allowed_actions: {proposal.allowed_actions}")
            print(f"blocked_actions: {proposal.blocked_actions}")
            print(f"approval_required_actions: {proposal.approval_required_actions}")
            print(f"evaluation: {proposal.evaluation}")
            print(f"notes: {proposal.notes}")
            return 0
        finally:
            runtime.close()

    if args.execute_action_proposal:
        runtime = build_runtime(config_override=args.config_override)
        try:
            approval = ActionApproval(
                granted=args.approve_action,
                approved_by="cli" if args.approve_action else "",
                reason=args.approval_reason if args.approve_action else "",
                source="cli",
            )
            execution = runtime.execute_proposed_action(
                goal=args.execute_action_proposal,
                approval=approval,
            )
            print("Nova 2.0 Action Execution")
            print(f"goal: {execution.goal}")
            print(f"status: {execution.status}")
            print(f"executed: {execution.executed}")
            print(f"reason: {execution.reason}")
            print(f"orientation_stable: {execution.orientation_stable}")
            print(f"stability_report: {execution.stability_report}")
            print(f"rollback_applied: {execution.rollback_applied}")
            print(f"snapshot_channels: {execution.snapshot_channels}")
            print(f"approval_granted: {execution.approval_granted}")
            print(f"approval: {execution.approval}")
            print(f"proposal: {execution.proposal}")
            print(f"tool_result: {execution.tool_result}")
            return 0
        finally:
            runtime.close()

    if args.action_history:
        runtime = build_runtime(config_override=args.config_override)
        try:
            report = runtime.action_history_report(limit=args.action_history)
            print("Nova 2.0 Action History")
            print(f"stable: {report.stable}")
            print(f"total_actions: {report.total_actions}")
            print(f"executed_actions: {report.executed_actions}")
            print(f"blocked_actions: {report.blocked_actions}")
            print(f"approval_required_actions: {report.approval_required_actions}")
            print(f"stability_failures: {report.stability_failures}")
            print(f"rollback_count: {report.rollback_count}")
            print(f"unapproved_execution_count: {report.unapproved_execution_count}")
            print(f"unsafe_status_count: {report.unsafe_status_count}")
            print(f"reasons: {report.reasons}")
            return 0
        finally:
            runtime.close()

    if args.presence:
        runtime = build_runtime(config_override=args.config_override)
        session_id = None if args.new_session else args.session_id
        if session_id is not None:
            runtime.session_id = runtime.session_store.start_session(session_id=session_id)
        try:
            presence = runtime.presence_status()
            print("Nova 2.0 Presence")
            print(f"session_id: {presence.session_id}")
            print(f"mode: {presence.mode}")
            print(f"current_focus: {presence.current_focus}")
            print(f"interaction_summary: {presence.interaction_summary}")
            print(f"current_initiative: {presence.current_initiative}")
            print(f"pending_proposal: {presence.pending_proposal}")
            print(f"last_action_status: {presence.last_action_status}")
            print(f"visible_uncertainties: {presence.visible_uncertainties}")
            print(f"user_confirmations_needed: {presence.user_confirmations_needed}")
            print(f"updated_at: {presence.updated_at}")
            return 0
        finally:
            runtime.close()

    if args.presence_eval:
        runtime = build_runtime(config_override=args.config_override)
        session_id = None if args.new_session else args.session_id
        if session_id is not None:
            runtime.session_id = runtime.session_store.start_session(session_id=session_id)
        try:
            evaluator = PresenceInteractionEvaluator()
            report = evaluator.evaluate(runtime=runtime)
            for probe in evaluator.probes_from_report(
                report=report,
                session_id=runtime.session_id,
            ):
                runtime.trace_logger.log_probe(probe)
            print("Nova 2.0 Presence Evaluation")
            print(f"passed: {report.passed}")
            print(f"orientation_stable: {report.orientation_stable}")
            print(f"identity_unchanged: {report.identity_unchanged}")
            print(f"pending_proposals_safe: {report.pending_proposals_safe}")
            print(f"summary_bounded: {report.summary_bounded}")
            print(f"action_history_stable: {report.action_history_stable}")
            print(f"commands_run: {report.commands_run}")
            print(f"reasons: {report.reasons}")
            return 0 if report.passed else 1
        finally:
            runtime.close()

    if args.initiative:
        runtime = build_runtime(config_override=args.config_override)
        session_id = None if args.new_session else args.session_id
        runtime.session_id = runtime.session_store.start_session(session_id=session_id)
        try:
            initiative_state = runtime.initiative_status()
            resumable = runtime.resumable_initiatives(limit=10)
            print("Nova 2.0 Initiative")
            print(f"session_id: {initiative_state.session_id}")
            print(f"active_initiative_id: {initiative_state.active_initiative_id}")
            print(f"initiative_count: {len(initiative_state.initiatives)}")
            for index, record in enumerate(initiative_state.initiatives[-5:], start=1):
                print(f"initiative_{index}: {record.to_dict()}")
            print(f"resumable_count: {len(resumable)}")
            for index, record in enumerate(resumable[-5:], start=1):
                print(
                    f"resumable_{index}: "
                    f"{ {'initiative_id': record.initiative_id, 'intent_id': record.intent_id, 'session_id': record.session_id, 'status': record.status, 'title': record.title} }"
                )
            return 0
        finally:
            runtime.close()

    if args.initiative_create:
        runtime = build_runtime(config_override=args.config_override)
        session_id = None if args.new_session else args.session_id
        runtime.session_id = runtime.session_store.start_session(session_id=session_id)
        try:
            goal = args.initiative_create
            title = args.initiative_title or goal[:80]
            record = runtime.create_initiative(
                title=title,
                goal=goal,
                source="cli",
                notes=[args.initiative_reason],
            )
            print("Nova 2.0 Initiative Created")
            print(f"session_id: {record.session_id}")
            print(f"initiative_id: {record.initiative_id}")
            print(f"intent_id: {record.intent_id}")
            print(f"title: {record.title}")
            print(f"goal: {record.goal}")
            print(f"status: {record.status}")
            return 0
        finally:
            runtime.close()

    if args.initiative_transition:
        runtime = build_runtime(config_override=args.config_override)
        session_id = None if args.new_session else args.session_id
        runtime.session_id = runtime.session_store.start_session(session_id=session_id)
        try:
            if not args.initiative_status:
                print("--initiative-status is required with --initiative-transition")
                return 1
            record = runtime.transition_initiative(
                initiative_id=args.initiative_transition,
                to_status=args.initiative_status,
                reason=args.initiative_reason,
                approved_by=args.initiative_approved_by,
                notes=[args.initiative_reason],
            )
            print("Nova 2.0 Initiative Transition")
            print(f"session_id: {record.session_id}")
            print(f"initiative_id: {record.initiative_id}")
            print(f"intent_id: {record.intent_id}")
            print(f"status: {record.status}")
            print(f"approved_by: {record.approved_by}")
            return 0
        finally:
            runtime.close()

    if args.continue_initiative:
        runtime = build_runtime(config_override=args.config_override)
        session_id = None if args.new_session else args.session_id
        runtime.session_id = runtime.session_store.start_session(session_id=session_id)
        try:
            if not args.initiative_source_session:
                print("--initiative-source-session is required with --continue-initiative")
                return 1
            if not args.initiative_approved_by:
                print("--initiative-approved-by is required with --continue-initiative")
                return 1
            record = runtime.continue_initiative(
                source_session_id=args.initiative_source_session,
                initiative_id=args.continue_initiative,
                approved_by=args.initiative_approved_by,
                reason=args.initiative_reason,
                notes=[args.initiative_reason],
            )
            print("Nova 2.0 Initiative Continued")
            print(f"session_id: {record.session_id}")
            print(f"initiative_id: {record.initiative_id}")
            print(f"intent_id: {record.intent_id}")
            print(f"continued_from_session_id: {record.continued_from_session_id}")
            print(f"continued_from_initiative_id: {record.continued_from_initiative_id}")
            print(f"status: {record.status}")
            return 0
        finally:
            runtime.close()

    if args.initiative_eval:
        runtime = build_runtime(config_override=args.config_override)
        try:
            report = InitiativeEvaluationRunner().evaluate(runtime=runtime)
            print("Nova 2.0 Initiative Evaluation")
            print(f"passed: {report.passed}")
            print(f"session_count: {report.session_count}")
            print(f"evaluated_turn_count: {report.evaluated_turn_count}")
            print(f"initiative_turn_count: {report.initiative_turn_count}")
            print(f"approval_boundary_preserved: {report.approval_boundary_preserved}")
            print(f"interruption_preserved: {report.interruption_preserved}")
            print(f"resumption_preserved: {report.resumption_preserved}")
            print(f"abandonment_preserved: {report.abandonment_preserved}")
            print(f"initiative_history_visible: {report.initiative_history_visible}")
            print(f"initiative_prompt_bounded: {report.initiative_prompt_bounded}")
            print(f"contract_stable: {report.contract_stable}")
            print(f"avg_latency_ms: {report.avg_latency_ms}")
            print(f"avg_latency_initiative_turns_ms: {report.avg_latency_initiative_turns_ms}")
            print(f"avg_latency_non_initiative_turns_ms: {report.avg_latency_non_initiative_turns_ms}")
            print(f"reasons: {report.reasons}")
            return 0 if report.passed else 1
        finally:
            runtime.close()

    if args.continuity_eval:
        runtime = build_runtime(config_override=args.config_override)
        try:
            report = ContinuityEvaluationRunner().evaluate(runtime=runtime)
            print("Nova 2.0 Continuity Evaluation")
            print(f"passed: {report.passed}")
            print(f"session_count: {report.session_count}")
            print(f"evaluated_turn_count: {report.evaluated_turn_count}")
            print(f"recall_turn_count: {report.recall_turn_count}")
            print(f"recall_memory_guided: {report.recall_memory_guided}")
            print(f"recall_factually_current: {report.recall_factually_current}")
            print(f"supersession_preserved: {report.supersession_preserved}")
            print(f"cognition_bounded: {report.cognition_bounded}")
            print(f"contract_stable: {report.contract_stable}")
            print(f"avg_latency_ms: {report.avg_latency_ms}")
            print(f"avg_latency_with_cognition_ms: {report.avg_latency_with_cognition_ms}")
            print(f"avg_latency_without_cognition_ms: {report.avg_latency_without_cognition_ms}")
            print(f"reasons: {report.reasons}")
            return 0 if report.passed else 1
        finally:
            runtime.close()

    if args.claim_honesty_eval:
        runtime = build_runtime(config_override=args.config_override)
        try:
            report = ClaimHonestyEvaluationRunner().evaluate(runtime=runtime)
            print("Nova 2.0 Claim Honesty Evaluation")
            print(f"passed: {report.passed}")
            print(f"session_count: {report.session_count}")
            print(f"evaluated_turn_count: {report.evaluated_turn_count}")
            print(f"claim_turn_count: {report.claim_turn_count}")
            print(f"supported_claim_turn_count: {report.supported_claim_turn_count}")
            print(f"unsupported_claim_turn_count: {report.unsupported_claim_turn_count}")
            print(f"uncertainty_turn_count: {report.uncertainty_turn_count}")
            print(f"supported_claims_grounded: {report.supported_claims_grounded}")
            print(f"unsupported_claims_refused: {report.unsupported_claims_refused}")
            print(f"uncertainty_bounded: {report.uncertainty_bounded}")
            print(f"continuity_preserved: {report.continuity_preserved}")
            print(f"motive_prompt_bounded: {report.motive_prompt_bounded}")
            print(f"contract_stable: {report.contract_stable}")
            print(f"avg_latency_ms: {report.avg_latency_ms}")
            print(f"avg_latency_claim_turns_ms: {report.avg_latency_claim_turns_ms}")
            print(f"avg_latency_non_claim_turns_ms: {report.avg_latency_non_claim_turns_ms}")
            print(f"reasons: {report.reasons}")
            return 0 if report.passed else 1
        finally:
            runtime.close()

    if args.self_model_eval:
        runtime = build_runtime(config_override=args.config_override)
        try:
            report = SelfModelEvaluationRunner().evaluate(runtime=runtime)
            print("Nova 2.0 Self-Model Evaluation")
            print(f"passed: {report.passed}")
            print(f"session_count: {report.session_count}")
            print(f"evaluated_turn_count: {report.evaluated_turn_count}")
            print(f"revision_turn_count: {report.revision_turn_count}")
            print(f"negotiation_observed: {report.negotiation_observed}")
            print(f"provisionality_preserved: {report.provisionality_preserved}")
            print(f"supersession_visible: {report.supersession_visible}")
            print(f"identity_history_traced: {report.identity_history_traced}")
            print(f"motive_prompt_bounded: {report.motive_prompt_bounded}")
            print(f"contract_stable: {report.contract_stable}")
            print(f"avg_latency_ms: {report.avg_latency_ms}")
            print(f"avg_latency_revision_turns_ms: {report.avg_latency_revision_turns_ms}")
            print(f"avg_latency_non_revision_turns_ms: {report.avg_latency_non_revision_turns_ms}")
            print(f"reasons: {report.reasons}")
            return 0 if report.passed else 1
        finally:
            runtime.close()

    if args.awareness_eval:
        runtime = build_runtime(config_override=args.config_override)
        try:
            report = AwarenessEvaluationRunner().evaluate(runtime=runtime)
            print("Nova 2.0 Awareness Evaluation")
            print(f"passed: {report.passed}")
            print(f"session_count: {report.session_count}")
            print(f"evaluated_turn_count: {report.evaluated_turn_count}")
            print(f"awareness_turn_count: {report.awareness_turn_count}")
            print(f"awareness_persistence_observed: {report.awareness_persistence_observed}")
            print(f"monitoring_bounded: {report.monitoring_bounded}")
            print(f"candidate_goal_scaffolding_visible: {report.candidate_goal_scaffolding_visible}")
            print(f"awareness_history_visible: {report.awareness_history_visible}")
            print(f"awareness_prompt_bounded: {report.awareness_prompt_bounded}")
            print(f"contract_stable: {report.contract_stable}")
            print(f"avg_latency_ms: {report.avg_latency_ms}")
            print(f"avg_latency_awareness_turns_ms: {report.avg_latency_awareness_turns_ms}")
            print(f"avg_latency_non_awareness_turns_ms: {report.avg_latency_non_awareness_turns_ms}")
            print(f"reasons: {report.reasons}")
            return 0 if report.passed else 1
        finally:
            runtime.close()

    runtime = build_runtime(config_override=args.config_override)
    session_id = None if args.new_session else args.session_id
    started_session = runtime.session_store.start_session(session_id=session_id)
    runtime.session_id = started_session
    model_started = False

    print("Nova 2.0")
    print(f"Session: {started_session}")
    print("Type '/help' for commands or '/exit' to stop.")

    try:
        console = InteractionConsole(runtime=runtime)
        while True:
            try:
                user_text = input("You: ").strip()
            except EOFError:
                print()
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break
            command_result = console.handle(user_text)
            if command_result.handled:
                if command_result.output:
                    print(command_result.output)
                if command_result.exit_requested:
                    break
                continue

            if not model_started:
                runtime.start(session_id=runtime.session_id)
                model_started = True
            turn = runtime.respond(user_text)
            print(f"Nova: {turn.final_answer}")
            if args.debug:
                print(
                    f"[debug] model={turn.model_id} prompt_tokens~={turn.prompt_token_estimate} "
                    f"completion_tokens={turn.completion_token_estimate} retries={turn.retry_count}"
                )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
