"""CLI entrypoint for Nova 2.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nova.agent.action import ActionApproval
from nova.agent.presence import JsonPresenceStore
from nova.console import InteractionConsole
from nova.config import DEFAULT_CONFIG_PATH, load_config
from nova.eval.presence import PresenceInteractionEvaluator
from nova.eval.probes import BasicProbeRunner
from nova.inference.llama_cpp_backend import LlamaCppBackend
from nova.logging.traces import JsonlTraceLogger
from nova.agent.orientation import SelfOrientationEngine
from nova.agent.orientation_eval import OrientationStabilityEvaluator
from nova.agent.stability import OrientationHistoryAnalyzer
from nova.agent.stability import ContextPressureOrientationChecker, MaintenanceOrientationStabilityChecker
from nova.memory.maintenance import MemoryMaintenanceRunner
from nova.memory.policy import IdentityFirstRetrievalPolicy
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
    presence_store = JsonPresenceStore(data_dir / "presence")
    session_store = JsonlSessionStore(sessions_dir)
    trace_logger = JsonlTraceLogger(traces_dir, probe_path=probes_path)

    episodic_store = JsonlEpisodicMemoryStore(memory_dir / "episodic.jsonl")
    engram_store = JsonEngramMemoryStore(
        memory_dir / "engram.json",
        enabled=config.memory.engram_enabled,
    )
    graph_store = SqliteGraphMemoryStore(memory_dir / "graph.db")
    autobiographical_store = JsonlAutobiographicalMemoryStore(
        memory_dir / "autobiographical.jsonl"
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
    )

    return {
        "config": config,
        "data_dir": data_dir,
        "log_dir": log_dir,
        "persona_store": persona_store,
        "self_state_store": self_state_store,
        "presence_store": presence_store,
        "session_store": session_store,
        "trace_logger": trace_logger,
        "episodic_store": episodic_store,
        "engram_store": engram_store,
        "graph_store": graph_store,
        "autobiographical_store": autobiographical_store,
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

    return NovaRuntime(
        config=config,
        backend=backend,
        composer=composer,
        validator=validator,
        retry_policy=retry_policy,
        persona_store=persona_store,
        self_state_store=self_state_store,
        presence_store=presence_store,
        session_store=session_store,
        trace_logger=trace_logger,
        memory_router=memory_router,
        memory_event_factory=event_factory,
        retrieval_policy=retrieval_policy,
        probe_runner=probe_runner,
        orientation_engine=orientation_engine,
        orientation_evaluator=orientation_evaluator,
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
        return {
            "action": action,
            "written": len(candidates),
            "event_ids": [candidate.event_id for candidate in candidates],
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
        return {
            "action": action,
            "semantic_written": len(semantic_candidates),
            "autobiographical_written": len(autobiographical_candidates),
            "applied": applied,
            "summary": runner.summarize_plan(),
        }
    raise ValueError(f"Unsupported maintenance action: {action}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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
            return 0
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
