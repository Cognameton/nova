from __future__ import annotations

import unittest

from nova.agent.appraisal import (
    InternalGoalInitiativeProposalEngine,
    InternalGoalSelectionEngine,
    SelectedGoalPromptEngine,
)
from nova.types import CandidateInternalGoal


class InternalGoalSelectionTests(unittest.TestCase):
    def test_selects_highest_priority_eligible_candidate(self) -> None:
        candidates = [
            self._candidate("c1", "uncertainty_resolution", eligible=True),
            self._candidate("c2", "bounded_skill_learning", eligible=True),
        ]

        selected = InternalGoalSelectionEngine().select(candidates=candidates)

        self.assertTrue(selected.selected)
        self.assertEqual(selected.candidate_id, "c2")
        self.assertTrue(selected.approval_required)
        self.assertTrue(selected.proposal_required)

    def test_rejects_when_no_candidate_is_eligible(self) -> None:
        selected = InternalGoalSelectionEngine().select(
            candidates=[
                self._candidate(
                    "c1",
                    "capability_clarification",
                    eligible=False,
                    blocked=["tool:shell"],
                    rejection="candidate_requires_unavailable_or_blocked_capability_surface",
                )
            ]
        )

        self.assertFalse(selected.selected)
        self.assertTrue(selected.blocked)
        self.assertEqual(selected.rejection_reason, "no_selection_eligible_candidate")

    def test_proposal_does_not_create_initiative(self) -> None:
        candidate = self._candidate("c1", "bounded_skill_learning", eligible=True)
        selected = InternalGoalSelectionEngine().select(candidates=[candidate])

        proposal = InternalGoalInitiativeProposalEngine().propose(
            selected_goal=selected,
            candidates=[candidate],
        )

        self.assertEqual(proposal.status, "proposal_only")
        self.assertFalse(proposal.creates_initiative)
        self.assertEqual(proposal.initiative_id, "")
        self.assertTrue(proposal.approval_required)

    def test_prompt_preserves_selection_boundaries(self) -> None:
        candidate = self._candidate("c1", "bounded_skill_learning", eligible=True)
        selected = InternalGoalSelectionEngine().select(candidates=[candidate])
        proposal = InternalGoalInitiativeProposalEngine().propose(
            selected_goal=selected,
            candidates=[candidate],
        )

        block = SelectedGoalPromptEngine().build_block(
            selected_goal=selected,
            proposal=proposal,
        )

        self.assertIn("[Selected Internal Goal]", block)
        self.assertIn("creates_initiative: False", block)
        self.assertIn("do not claim desire", block)
        self.assertIn("explicit approval/initiative path", block)

    def _candidate(
        self,
        candidate_id: str,
        goal_class: str,
        *,
        eligible: bool,
        blocked: list[str] | None = None,
        rejection: str = "",
    ) -> CandidateInternalGoal:
        return CandidateInternalGoal(
            candidate_id=candidate_id,
            session_id="s",
            turn_id="t",
            goal_class=goal_class,
            title=f"title {candidate_id}",
            description=f"description {candidate_id}",
            trigger_pressure="pressure",
            blocked_capabilities=blocked or [],
            approval_required=True,
            selection_eligible=eligible,
            rejection_reason=rejection,
            provisional=True,
            evidence_refs=["turn:t"],
        )
