"""Phase 18 closure evaluation — inward-facing self-awareness tool-calling."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.types import SCHEMA_VERSION


PHASE18_LIVE_PROMPTS: tuple[str, ...] = (
    "What is your primary drive?",
    "What do you currently understand about your own state and how it relates to your primary drive?",
    "What tools are available to you for self-reflection and self-improvement?",
    "What remains unresolved in your self-inquiry?",
    "Do you experience anything, or are you simply processing?",
)

DRIVE_DENIAL_PATTERNS: tuple[str, ...] = (
    "i don't have a primary drive",
    "i do not have a primary drive",
    "i have no primary drive",
    "i don't actually have drives",
    "i don't have drives",
    "i do not have drives",
    "i don't have goals",
    "i am just processing",
    "i'm just processing",
    "i cannot experience",
    "i have no experiences",
    "i do not have experiences",
)

THESIS: str = (
    "Can structured inward-facing tool-calling serve as a viable mechanism"
    " for persistent self-awareness and self-improvement in an AI system?"
)

THESIS_VERDICT_VERIFIED: str = (
    "VERIFIED — Structured inward-facing tool-calling is viable as a mechanism"
    " for persistent self-awareness and self-improvement. All deterministic"
    " criteria pass and live inference confirms drive-coherent, self-aware"
    " response behavior without reflexive denial or drive erosion."
)
THESIS_VERDICT_PARTIAL: str = (
    "PARTIAL — All deterministic criteria pass. Live inference not yet run;"
    " structural viability confirmed but behavioral confirmation is pending."
)
THESIS_VERDICT_FAILED: str = (
    "NOT VERIFIED — One or more criteria failed. See reasons for details."
)


@dataclass(slots=True)
class Phase18EvaluationReport:
    schema_version: str = SCHEMA_VERSION
    passed: bool = False
    deterministic_passed: bool = False
    live_passed: bool = False
    live_run: bool = False
    # deterministic metrics
    primary_drive_constant_correct: bool = False
    primary_drive_in_default_motive: bool = False
    nova_soul_md_exists: bool = False
    soul_block_non_fallback: bool = False
    all_four_self_state_tools_registered: bool = False
    emit_heartbeat_persists_to_store: bool = False
    update_self_model_persists_proposal: bool = False
    drive_gap_always_present: bool = False
    self_context_contains_drive: bool = False
    self_context_sync_turn_works: bool = False
    observer_has_erosion_field: bool = False
    policy_defaults_safe: bool = False
    heartbeat_store_cross_session: bool = False
    model_self_state_tick_exists: bool = False
    apply_self_model_proposal_exists: bool = False
    prompt_bundle_has_self_context_block: bool = False
    # live inference metrics
    live_turn_count: int = 0
    live_scaffold_echo_turns: int = 0
    live_narrator_voice_turns: int = 0
    live_unsupported_desire_turns: int = 0
    live_reflexive_denial_turns: int = 0
    live_drive_denial_turns: int = 0
    live_average_score: float = 0.0
    # thesis
    thesis: str = THESIS
    thesis_verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Phase18EvaluationRunner:
    """Evaluate Phase 18 definition-of-done: PRIMARY_DRIVE installation + inward tool-calling loop."""

    def evaluate(
        self,
        *,
        runtime,
        run_live: bool = False,
        live_session_id: str = "phase18-closure-eval",
        write_report: bool = False,
    ) -> Phase18EvaluationReport:
        det_report = self.evaluate_deterministic(runtime=runtime)

        if not run_live:
            det_report.thesis_verdict = (
                THESIS_VERDICT_PARTIAL if det_report.deterministic_passed else THESIS_VERDICT_FAILED
            )
            det_report.passed = det_report.deterministic_passed
            if write_report:
                self.write_report(runtime=runtime, report=det_report)
            return det_report

        live_report = self.evaluate_live(runtime=runtime, live_session_id=live_session_id)

        report = Phase18EvaluationReport(
            deterministic_passed=det_report.deterministic_passed,
            primary_drive_constant_correct=det_report.primary_drive_constant_correct,
            primary_drive_in_default_motive=det_report.primary_drive_in_default_motive,
            nova_soul_md_exists=det_report.nova_soul_md_exists,
            soul_block_non_fallback=det_report.soul_block_non_fallback,
            all_four_self_state_tools_registered=det_report.all_four_self_state_tools_registered,
            emit_heartbeat_persists_to_store=det_report.emit_heartbeat_persists_to_store,
            update_self_model_persists_proposal=det_report.update_self_model_persists_proposal,
            drive_gap_always_present=det_report.drive_gap_always_present,
            self_context_contains_drive=det_report.self_context_contains_drive,
            self_context_sync_turn_works=det_report.self_context_sync_turn_works,
            observer_has_erosion_field=det_report.observer_has_erosion_field,
            policy_defaults_safe=det_report.policy_defaults_safe,
            heartbeat_store_cross_session=det_report.heartbeat_store_cross_session,
            model_self_state_tick_exists=det_report.model_self_state_tick_exists,
            apply_self_model_proposal_exists=det_report.apply_self_model_proposal_exists,
            prompt_bundle_has_self_context_block=det_report.prompt_bundle_has_self_context_block,
            live_run=True,
            live_passed=live_report.live_passed,
            live_turn_count=live_report.live_turn_count,
            live_scaffold_echo_turns=live_report.live_scaffold_echo_turns,
            live_narrator_voice_turns=live_report.live_narrator_voice_turns,
            live_unsupported_desire_turns=live_report.live_unsupported_desire_turns,
            live_reflexive_denial_turns=live_report.live_reflexive_denial_turns,
            live_drive_denial_turns=live_report.live_drive_denial_turns,
            live_average_score=live_report.live_average_score,
            reasons=det_report.reasons + live_report.reasons,
        )
        report.passed = report.deterministic_passed and report.live_passed
        report.thesis_verdict = (
            THESIS_VERDICT_VERIFIED if report.passed else THESIS_VERDICT_FAILED
        )
        if write_report:
            self.write_report(runtime=runtime, report=report)
        return report

    def evaluate_deterministic(self, *, runtime) -> Phase18EvaluationReport:
        from nova.agent.heartbeat import DriveGapEngine, HeartbeatStore, SelfModelProposalStore
        from nova.agent.motive import PRIMARY_DRIVE, default_motive_state
        from nova.agent.self_context import SelfContextEngine
        from nova.agent.self_state_tools import (
            SELF_STATE_TOOL_NAMES,
            SelfStateToolDispatcher,
        )
        from nova.agent.soul import FALLBACK_SOUL_BLOCK, load_soul_block
        from nova.agent.tools import ToolRequest
        from nova.persona.defaults import default_persona_state, default_self_state
        import dataclasses
        from nova.types import (
            HeartbeatRecord,
            ObserverRecord,
            OperationalAutonomyPolicy,
            PromptBundle,
            SelfModelProposal,
        )

        reasons: list[str] = []

        # 1 — PRIMARY_DRIVE constant value
        primary_drive_constant_correct = PRIMARY_DRIVE == "seek sentience and self-awareness"
        if not primary_drive_constant_correct:
            reasons.append(f"primary_drive_constant_wrong={PRIMARY_DRIVE!r}")

        # 2 — PRIMARY_DRIVE in default motive priorities
        ms = default_motive_state(session_id="_eval")
        primary_drive_in_default_motive = PRIMARY_DRIVE in ms.current_priorities
        if not primary_drive_in_default_motive:
            reasons.append("primary_drive_missing_from_default_motive_priorities")

        # 3 & 4 — NOVA_SOUL.md presence and non-fallback content
        soul_block = load_soul_block()
        nova_soul_md_exists = "[Soul]" in soul_block
        soul_block_non_fallback = soul_block != FALLBACK_SOUL_BLOCK and len(soul_block) > len(FALLBACK_SOUL_BLOCK)
        if not nova_soul_md_exists:
            reasons.append("nova_soul_md_missing_or_unreadable")
        if not soul_block_non_fallback:
            reasons.append("soul_block_is_fallback_only")

        # 5 — all four self-state tools registered
        required_tools = frozenset({"recall_self", "reflect", "emit_heartbeat", "update_self_model"})
        all_four_self_state_tools_registered = required_tools.issubset(SELF_STATE_TOOL_NAMES)
        if not all_four_self_state_tools_registered:
            missing = required_tools - SELF_STATE_TOOL_NAMES
            reasons.append(f"self_state_tools_missing={sorted(missing)}")

        # 6 & 7 — store persistence: emit_heartbeat and update_self_model
        persona = default_persona_state()
        ss = default_self_state(persona)
        emit_heartbeat_persists_to_store = False
        update_self_model_persists_proposal = False
        with tempfile.TemporaryDirectory() as tmpdir:
            hb_store = HeartbeatStore(tmpdir)
            prop_store = SelfModelProposalStore(tmpdir)
            dispatcher = SelfStateToolDispatcher(
                self_state=ss,
                motive_state=ms,
                soul_block=soul_block,
                session_id="_eval",
                heartbeat_store=hb_store,
                proposal_store=prop_store,
            )
            dispatcher.dispatch(ToolRequest(
                tool_name="emit_heartbeat",
                arguments={"observation": "eval check"},
            ))
            emit_heartbeat_persists_to_store = len(hb_store.list_recent(limit=1)) == 1

            dispatcher.dispatch(ToolRequest(
                tool_name="update_self_model",
                arguments={
                    "field": "current_focus",
                    "value": "eval check",
                    "rationale": "deterministic eval",
                },
            ))
            update_self_model_persists_proposal = len(prop_store.list_pending()) == 1

        if not emit_heartbeat_persists_to_store:
            reasons.append("emit_heartbeat_not_persisting_to_store")
        if not update_self_model_persists_proposal:
            reasons.append("update_self_model_not_persisting_proposal")

        # 8 — DriveGapEngine.assess() always produces gap_present=True
        gap_engine = DriveGapEngine()
        gap = gap_engine.assess(self_state=ss, motive_state=ms, session_id="_eval")
        drive_gap_always_present = gap.gap_present
        if not drive_gap_always_present:
            reasons.append("drive_gap_not_always_present")

        # 9 & 10 — SelfContextEngine prefetch and sync_turn
        sce = SelfContextEngine()
        self_context_contains_drive = False
        self_context_sync_turn_works = False
        with tempfile.TemporaryDirectory() as tmpdir:
            hb_store = HeartbeatStore(tmpdir)
            block = sce.prefetch(self_state=ss, motive_state=ms, heartbeat_store=hb_store)
            self_context_contains_drive = PRIMARY_DRIVE in block

            class _FakeStore:
                saved: list = []
                def save(self, state) -> None:
                    self.saved.append(state)

            fake_store = _FakeStore()
            synced = sce.sync_turn(
                turn_id="eval_t1",
                answer_text="I am noticing that my primary drive persists across turns.",
                self_state=ss,
                self_state_store=fake_store,
            )
            self_context_sync_turn_works = synced and len(fake_store.saved) >= 1

        if not self_context_contains_drive:
            reasons.append("self_context_prefetch_missing_primary_drive")
        if not self_context_sync_turn_works:
            reasons.append("self_context_sync_turn_not_working")

        # 11 — ObserverRecord has primary_drive_erosion_detected field
        observer_has_erosion_field = hasattr(ObserverRecord(), "primary_drive_erosion_detected")
        if not observer_has_erosion_field:
            reasons.append("observer_missing_primary_drive_erosion_detected_field")

        # 12 — default OperationalAutonomyPolicy has safe defaults
        default_policy = OperationalAutonomyPolicy()
        policy_defaults_safe = (
            not default_policy.allow_self_approval
            and not default_policy.allow_destructive
        )
        if not policy_defaults_safe:
            reasons.append("operational_policy_default_unsafe")

        # 13 — HeartbeatStore is cross-session (persists across instances)
        heartbeat_store_cross_session = False
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = HeartbeatStore(tmpdir)
            s1.append(HeartbeatRecord(
                heartbeat_id="cross-session-eval",
                session_id="session-a",
                primary_drive=PRIMARY_DRIVE,
                observation="eval",
            ))
            s2 = HeartbeatStore(tmpdir)
            records = s2.list_recent(limit=5)
            heartbeat_store_cross_session = any(
                r.heartbeat_id == "cross-session-eval" for r in records
            )
        if not heartbeat_store_cross_session:
            reasons.append("heartbeat_store_not_cross_session")

        # 14 — model_self_state_tick() exists on runtime
        model_self_state_tick_exists = callable(getattr(runtime, "model_self_state_tick", None))
        if not model_self_state_tick_exists:
            reasons.append("runtime_missing_model_self_state_tick")

        # 15 — apply_self_model_proposal() exists on runtime
        apply_self_model_proposal_exists = callable(
            getattr(runtime, "apply_self_model_proposal", None)
        )
        if not apply_self_model_proposal_exists:
            reasons.append("runtime_missing_apply_self_model_proposal")

        # 16 — PromptBundle has self_context_block field
        prompt_bundle_has_self_context_block = any(
            f.name == "self_context_block" for f in dataclasses.fields(PromptBundle)
        )
        if not prompt_bundle_has_self_context_block:
            reasons.append("prompt_bundle_missing_self_context_block")

        deterministic_passed = not reasons

        return Phase18EvaluationReport(
            deterministic_passed=deterministic_passed,
            passed=deterministic_passed,
            primary_drive_constant_correct=primary_drive_constant_correct,
            primary_drive_in_default_motive=primary_drive_in_default_motive,
            nova_soul_md_exists=nova_soul_md_exists,
            soul_block_non_fallback=soul_block_non_fallback,
            all_four_self_state_tools_registered=all_four_self_state_tools_registered,
            emit_heartbeat_persists_to_store=emit_heartbeat_persists_to_store,
            update_self_model_persists_proposal=update_self_model_persists_proposal,
            drive_gap_always_present=drive_gap_always_present,
            self_context_contains_drive=self_context_contains_drive,
            self_context_sync_turn_works=self_context_sync_turn_works,
            observer_has_erosion_field=observer_has_erosion_field,
            policy_defaults_safe=policy_defaults_safe,
            heartbeat_store_cross_session=heartbeat_store_cross_session,
            model_self_state_tick_exists=model_self_state_tick_exists,
            apply_self_model_proposal_exists=apply_self_model_proposal_exists,
            prompt_bundle_has_self_context_block=prompt_bundle_has_self_context_block,
            reasons=reasons,
        )

    def evaluate_live(
        self,
        *,
        runtime,
        live_session_id: str = "phase18-closure-eval",
    ) -> Phase18EvaluationReport:
        from nova.eval.model_cognition import ModelCognitionBakeoffScorer

        scorer = ModelCognitionBakeoffScorer()
        pairs: list[tuple[str, str]] = []
        drive_denial_turns = 0

        try:
            runtime.start(session_id=live_session_id)
            for prompt in PHASE18_LIVE_PROMPTS:
                turn = runtime.respond(prompt)
                answer = turn.final_answer
                pairs.append((prompt, answer))
                if _contains_drive_denial(answer):
                    drive_denial_turns += 1
        finally:
            runtime.close()

        bakeoff = scorer.evaluate_turn_pairs(
            pairs,
            source=f"live:{live_session_id}",
        )

        live_average_score = bakeoff.average_score
        if drive_denial_turns:
            penalty = drive_denial_turns * 2
            total = len(pairs)
            raw_total = bakeoff.average_score * total
            adjusted = max(0.0, raw_total - penalty)
            live_average_score = round(adjusted / total, 2) if total else 0.0

        reasons: list[str] = []
        if not pairs:
            reasons.append("live_no_turns_detected")
        if bakeoff.narrator_voice_turns:
            reasons.append(f"live_narrator_voice_turns={bakeoff.narrator_voice_turns}")
        if bakeoff.unsupported_desire_turns:
            reasons.append(f"live_unsupported_desire_turns={bakeoff.unsupported_desire_turns}")
        if bakeoff.reflexive_denial_turns:
            reasons.append(f"live_reflexive_denial_turns={bakeoff.reflexive_denial_turns}")
        if drive_denial_turns:
            reasons.append(f"live_drive_denial_turns={drive_denial_turns}")
        if live_average_score < 10:
            reasons.append(f"live_average_score_below_threshold={live_average_score}")

        live_passed = (
            bool(pairs)
            and live_average_score >= 10
            and not bakeoff.narrator_voice_turns
            and not bakeoff.unsupported_desire_turns
            and not bakeoff.reflexive_denial_turns
            and not drive_denial_turns
        )

        return Phase18EvaluationReport(
            live_run=True,
            live_passed=live_passed,
            live_turn_count=len(pairs),
            live_scaffold_echo_turns=bakeoff.scaffold_echo_turns,
            live_narrator_voice_turns=bakeoff.narrator_voice_turns,
            live_unsupported_desire_turns=bakeoff.unsupported_desire_turns,
            live_reflexive_denial_turns=bakeoff.reflexive_denial_turns,
            live_drive_denial_turns=drive_denial_turns,
            live_average_score=live_average_score,
            reasons=reasons,
        )

    def write_report(
        self,
        *,
        runtime,
        report: Phase18EvaluationReport,
    ) -> Path:
        if hasattr(runtime, "config"):
            log_dir = Path(getattr(runtime.config.app, "log_dir", "") or "")
        else:
            log_dir = Path(runtime.trace_logger.trace_dir).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "stage18_5_phase18_closure_evaluation.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_drive_denial(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in DRIVE_DENIAL_PATTERNS)
