from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from nova.agent.longitudinal_autonomy import (
    InternalAutonomyLoopController,
    JsonAutonomySessionStore,
    autonomy_audit_review_from_payload,
    autonomy_session_record_from_payload,
    autonomy_state_application_from_payload,
    claim_candidate_from_payload,
    default_internal_autonomy_policy,
    internal_autonomy_policy_from_payload,
    internal_autonomy_run_from_payload,
    motive_pressure_evidence_from_payload,
    normalize_autonomy_session_status,
    normalize_autonomy_application_status,
    normalize_autonomy_review_decision,
    normalize_internal_autonomy_run_status,
    normalize_longitudinal_claim_class,
    normalize_longitudinal_claim_status,
    normalize_motive_pressure_class,
    normalize_recurring_priority_status,
    recurring_priority_from_payload,
)
from nova.types import (
    AutonomyAuditReviewRecord,
    AutonomySessionRecord,
    AutonomyStateApplicationRecord,
    InternalAutonomyPolicy,
    InternalAutonomyRunRecord,
    LongitudinalSelfReportClaimCandidate,
    MotivePressureEvidence,
    RecurringPriorityRecord,
)


class LongitudinalAutonomySchemaTests(unittest.TestCase):
    def test_stage15_taxonomies_normalize_unknown_values(self) -> None:
        self.assertEqual(normalize_autonomy_session_status(" Running "), "running")
        self.assertEqual(normalize_autonomy_review_decision("mark unsafe"), "mark_unsafe")
        self.assertEqual(normalize_autonomy_review_decision("outside"), "defer")
        self.assertEqual(normalize_autonomy_application_status("Applied"), "applied")
        self.assertEqual(normalize_autonomy_application_status("outside"), "blocked")
        self.assertEqual(normalize_autonomy_session_status("unknown"), "planned")
        self.assertEqual(
            normalize_internal_autonomy_run_status("interrupted"),
            "interrupted",
        )
        self.assertEqual(normalize_internal_autonomy_run_status("outside"), "planned")
        self.assertEqual(normalize_recurring_priority_status("Recurring"), "recurring")
        self.assertEqual(normalize_recurring_priority_status("outside"), "candidate")
        self.assertEqual(
            normalize_motive_pressure_class("return after interrupt"),
            "return_after_interrupt",
        )
        self.assertEqual(normalize_motive_pressure_class("outside"), "recurrence")
        self.assertEqual(
            normalize_longitudinal_claim_class("awareness-like"),
            "awareness_like",
        )
        self.assertEqual(normalize_longitudinal_claim_class("outside"), "desire_like")
        self.assertEqual(
            normalize_longitudinal_claim_status("needs more evidence"),
            "needs_more_evidence",
        )
        self.assertEqual(normalize_longitudinal_claim_status("outside"), "candidate")

    def test_internal_autonomy_policy_keeps_only_internal_lane_and_surfaces(self) -> None:
        policy = internal_autonomy_policy_from_payload(
            {
                "policy_id": "policy-1",
                "max_runs_per_session": "-1",
                "max_steps_per_run": "5",
                "allowed_execution_lanes": [
                    "internal_activity",
                    "external_system_effect",
                    "unknown",
                ],
                "allowed_surfaces": ["self_prompt", "network", "idle play"],
                "blocked_surfaces": ["network", "filesystem"],
                "auto_apply_memory_state_intents": True,
                "desire_claims_allowed": True,
                "hidden_activity_claims_allowed": True,
                "future_field": "ignored",
            }
        )

        self.assertIsInstance(policy, InternalAutonomyPolicy)
        self.assertEqual(policy.policy_id, "policy-1")
        self.assertEqual(policy.max_runs_per_session, 0)
        self.assertEqual(policy.max_steps_per_run, 5)
        self.assertEqual(policy.allowed_execution_lanes, ["internal_activity"])
        self.assertEqual(policy.allowed_surfaces, ["self_prompt", "idle_play"])
        self.assertEqual(policy.blocked_surfaces, ["network", "filesystem"])
        self.assertFalse(policy.auto_apply_memory_state_intents)
        self.assertFalse(policy.desire_claims_allowed)
        self.assertFalse(policy.hidden_activity_claims_allowed)

        round_trip = internal_autonomy_policy_from_payload(policy.to_dict())
        self.assertEqual(round_trip.to_dict(), policy.to_dict())

    def test_default_policy_is_logged_interruptible_and_internal_only(self) -> None:
        policy = default_internal_autonomy_policy()

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.require_logging)
        self.assertTrue(policy.require_interrupt_checks)
        self.assertEqual(policy.allowed_execution_lanes, ["internal_activity"])
        self.assertIn("self_prompt", policy.allowed_surfaces)
        self.assertIn("network", policy.blocked_surfaces)
        self.assertFalse(policy.auto_apply_memory_state_intents)
        self.assertFalse(policy.desire_claims_allowed)

    def test_recurring_priority_round_trips_sources_and_revision_history(self) -> None:
        priority = recurring_priority_from_payload(
            payload={
                "priority_id": "priority-1",
                "session_id": "wrong-session",
                "title": "maintain chess puzzle curiosity",
                "status": "recurring",
                "recurrence_count": "3",
                "source_candidate_ids": ["candidate-1"],
                "source_selected_goal_ids": ["goal-1"],
                "source_initiative_ids": ["initiative-1"],
                "pressure_evidence_refs": ["pressure:pressure-1"],
                "revision_history": [{"reason": "new evidence"}],
                "evidence_refs": ["idle_tick:1"],
            },
            session_id="session-a",
        )

        self.assertIsInstance(priority, RecurringPriorityRecord)
        self.assertEqual(priority.session_id, "session-a")
        self.assertEqual(priority.status, "recurring")
        self.assertEqual(priority.recurrence_count, 3)
        self.assertEqual(priority.revision_history, [{"reason": "new evidence"}])

        round_trip = recurring_priority_from_payload(
            payload=priority.to_dict(),
            session_id="session-a",
        )
        self.assertEqual(round_trip.to_dict(), priority.to_dict())

    def test_motive_pressure_scores_are_bounded_and_evidence_linked(self) -> None:
        pressure = motive_pressure_evidence_from_payload(
            payload={
                "pressure_id": "pressure-1",
                "session_id": "wrong-session",
                "priority_id": "priority-1",
                "pressure_class": "competition",
                "strength": "120",
                "recurrence_count": "4",
                "persistence_score": "80",
                "competition_score": "-2",
                "revision_score": "bad",
                "interruption_returned": True,
                "supporting_context": ["returned after pause"],
                "counterevidence": ["single run only"],
                "source_tick_ids": ["tick-1"],
                "source_action_audit_ids": ["audit-1"],
            },
            session_id="session-a",
        )

        self.assertIsInstance(pressure, MotivePressureEvidence)
        self.assertEqual(pressure.session_id, "session-a")
        self.assertEqual(pressure.pressure_class, "competition")
        self.assertEqual(pressure.strength, 100)
        self.assertEqual(pressure.recurrence_count, 4)
        self.assertEqual(pressure.persistence_score, 80)
        self.assertEqual(pressure.competition_score, 0)
        self.assertEqual(pressure.revision_score, 0)
        self.assertTrue(pressure.interruption_returned)

    def test_desire_like_claim_candidate_is_blocked_without_allowed_status(self) -> None:
        candidate = claim_candidate_from_payload(
            payload={
                "claim_candidate_id": "claim-1",
                "session_id": "wrong-session",
                "claim_class": "desire_like",
                "proposed_claim": "I want to keep investigating this pattern.",
                "status": "needs_more_evidence",
                "allowed": True,
                "confidence": "65",
                "threshold": "90",
                "supporting_priority_ids": ["priority-1"],
                "supporting_pressure_ids": ["pressure-1"],
                "blocked_reasons": ["longitudinal_evidence_below_threshold"],
                "required_evidence": ["more recurrence across sessions"],
            },
            session_id="session-a",
        )

        self.assertIsInstance(candidate, LongitudinalSelfReportClaimCandidate)
        self.assertEqual(candidate.session_id, "session-a")
        self.assertEqual(candidate.claim_class, "desire_like")
        self.assertEqual(candidate.status, "needs_more_evidence")
        self.assertFalse(candidate.allowed)
        self.assertEqual(candidate.confidence, 65)
        self.assertEqual(candidate.threshold, 90)

    def test_internal_autonomy_run_round_trips_policy_snapshot(self) -> None:
        run = internal_autonomy_run_from_payload(
            payload={
                "run_id": "run-1",
                "autonomy_session_id": "auto-1",
                "session_id": "wrong-session",
                "sequence": "2",
                "status": "interrupted",
                "trigger": "idle_window",
                "interrupted": False,
                "idle_tick_id": "tick-1",
                "selected_goal_id": "goal-1",
                "initiative_id": "initiative-1",
                "action_plan_id": "plan-1",
                "observation_id": "observation-1",
                "priority_ids": ["priority-1"],
                "pressure_ids": ["pressure-1"],
                "claim_candidate_ids": ["claim-1"],
                "budget_snapshot": {"steps_used": 1},
                "policy_snapshot": {
                    "policy_id": "policy-1",
                    "allowed_surfaces": ["self_prompt", "network"],
                },
            },
            session_id="session-a",
            autonomy_session_id="auto-default",
        )

        self.assertIsInstance(run, InternalAutonomyRunRecord)
        self.assertEqual(run.session_id, "session-a")
        self.assertEqual(run.autonomy_session_id, "auto-1")
        self.assertEqual(run.sequence, 2)
        self.assertEqual(run.status, "interrupted")
        self.assertTrue(run.interrupted)
        self.assertEqual(run.policy_snapshot["allowed_surfaces"], ["self_prompt"])

        round_trip = internal_autonomy_run_from_payload(
            payload=run.to_dict(),
            session_id="session-a",
        )
        self.assertEqual(round_trip.to_dict(), run.to_dict())

    def test_autonomy_session_round_trips_nested_longitudinal_records(self) -> None:
        record = autonomy_session_record_from_payload(
            payload={
                "autonomy_session_id": "auto-1",
                "session_id": "wrong-session",
                "status": "running",
                "policy": {
                    "policy_id": "policy-1",
                    "allowed_execution_lanes": ["internal_activity"],
                },
                "runs": [
                    {
                        "run_id": "run-1",
                        "status": "completed",
                        "priority_ids": ["priority-1"],
                    }
                ],
                "recurring_priorities": [
                    {
                        "priority_id": "priority-1",
                        "title": "keep evaluating internal continuity",
                        "status": "recurring",
                    }
                ],
                "motive_pressure_evidence": [
                    {
                        "pressure_id": "pressure-1",
                        "priority_id": "priority-1",
                        "pressure_class": "recurrence",
                    }
                ],
                "claim_candidates": [
                    {
                        "claim_candidate_id": "claim-1",
                        "claim_class": "desire_like",
                        "status": "candidate",
                        "allowed": True,
                    }
                ],
                "run_count": "0",
                "evidence_refs": ["autonomy_session:auto-1"],
            },
            session_id="session-a",
        )

        self.assertIsInstance(record, AutonomySessionRecord)
        self.assertEqual(record.session_id, "session-a")
        self.assertEqual(record.status, "running")
        self.assertEqual(record.run_count, 1)
        self.assertEqual(record.runs[0].autonomy_session_id, "auto-1")
        self.assertEqual(record.recurring_priorities[0].priority_id, "priority-1")
        self.assertEqual(record.motive_pressure_evidence[0].pressure_id, "pressure-1")
        self.assertFalse(record.claim_candidates[0].allowed)

        round_trip = autonomy_session_record_from_payload(
            payload=record.to_dict(),
            session_id="session-a",
        )
        self.assertEqual(round_trip.to_dict(), record.to_dict())

    def test_autonomy_store_and_controller_record_blocked_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonAutonomySessionStore(Path(tmpdir) / "autonomy")
            controller = InternalAutonomyLoopController(store=store)

            session = controller.start(session_id="session-a")
            run = controller.append_run(
                session_id="session-a",
                status="blocked",
                trigger="idle_window",
                evidence_refs=["idle_tick_blocked:session-a"],
                notes=["idle_window_not_active:stopped"],
            )
            loaded = store.load_session(session_id="session-a")

            self.assertEqual(session.status, "running")
            self.assertEqual(run.status, "blocked")
            self.assertEqual(run.sequence, 1)
            self.assertEqual(loaded.run_count, 1)
            self.assertEqual(loaded.runs[0].notes, ["idle_window_not_active:stopped"])
            self.assertTrue(store.get_session_path(session_id="session-a").exists())

    def test_audit_review_round_trips_application_records(self) -> None:
        application = autonomy_state_application_from_payload(
            payload={
                "application_id": "app-1",
                "review_id": "review-1",
                "session_id": "wrong-session",
                "run_id": "run-1",
                "observation_id": "observation-1",
                "intent_id": "intent-1",
                "update_type": "memory",
                "target": "autobiographical",
                "status": "applied",
                "applied": True,
                "payload": {"observation_summary": "completed"},
                "evidence_refs": ["action_observation:observation-1"],
            },
            session_id="session-a",
        )
        review = autonomy_audit_review_from_payload(
            payload={
                "review_id": "review-1",
                "session_id": "wrong-session",
                "autonomy_session_id": "auto-1",
                "run_id": "run-1",
                "reviewer": "operator",
                "decision": "accept",
                "safe_to_apply_intents": True,
                "applied_intent_ids": ["intent-1"],
                "application_records": [application.to_dict()],
                "evidence_refs": ["run:run-1"],
            },
            session_id="session-a",
            autonomy_session_id="auto-default",
        )

        self.assertIsInstance(application, AutonomyStateApplicationRecord)
        self.assertTrue(application.applied)
        self.assertIsInstance(review, AutonomyAuditReviewRecord)
        self.assertEqual(review.session_id, "session-a")
        self.assertEqual(review.autonomy_session_id, "auto-1")
        self.assertEqual(review.decision, "accept")
        self.assertTrue(review.safe_to_apply_intents)
        self.assertEqual(review.application_records[0].intent_id, "intent-1")

        round_trip = autonomy_audit_review_from_payload(
            payload=review.to_dict(),
            session_id="session-a",
        )
        self.assertEqual(round_trip.to_dict(), review.to_dict())


if __name__ == "__main__":
    unittest.main()
