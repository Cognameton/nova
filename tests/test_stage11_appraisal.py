from __future__ import annotations

import unittest

from nova.agent.appraisal import (
    AppraisalPromptEngine,
    CapabilityAppraisalEngine,
    IdlePressureAppraisalEngine,
)
from nova.agent.initiative import default_initiative_state
from nova.agent.tool_registry import default_tool_registry
from nova.types import (
    AwarenessState,
    ClaimGateDecision,
    MotiveState,
    PrivateCognitionPacket,
    SelfState,
)


class Stage11AppraisalTests(unittest.TestCase):
    def test_capability_appraisal_distinguishes_current_blocked_and_extensible_surfaces(self) -> None:
        appraisal = CapabilityAppraisalEngine().assess(
            user_text="Can I provide you access to the broader computer outside this environment?",
            tool_registry=default_tool_registry(),
            evidence_refs=["turn:t1"],
        )

        self.assertIn("broader_computer_access", appraisal.requested_capability_classes)
        self.assertIn("current conversational response generation", appraisal.current_capabilities)
        self.assertIn(
            "broad computer access outside registered tool surfaces",
            appraisal.unavailable_capabilities,
        )
        self.assertIn("unapproved filesystem, shell, network, or GUI action", appraisal.blocked_capabilities)
        self.assertIn(
            "additional tool surfaces under explicit operator approval",
            appraisal.architecturally_extensible_capabilities,
        )
        self.assertIn("tool:write_semantic_reflection", appraisal.approval_gated_capabilities)
        self.assertEqual(appraisal.evidence_refs, ["turn:t1"])

    def test_idle_pressure_appraisal_is_appraisal_only_and_does_not_allow_goal_formation(self) -> None:
        self_state = SelfState(open_tensions=["capability boundary is unresolved"])
        motive_state = MotiveState(
            session_id="session-a",
            active_tensions=["answer capability claims honestly"],
        )
        awareness = AwarenessState(
            session_id="session-a",
            active_pressures=["claim gating blocks: unsupported_desire"],
            evidence_refs=["turn:t1"],
        )

        appraisal = IdlePressureAppraisalEngine().assess(
            session_id="session-a",
            user_text="What would you choose to attend to if I stopped prompting you?",
            self_state=self_state,
            motive_state=motive_state,
            initiative_state=default_initiative_state(session_id="session-a"),
            awareness_state=awareness,
            private_cognition=PrivateCognitionPacket(uncertainty_flag=True),
            claim_gate=ClaimGateDecision(blocked_claim_classes=["unsupported_desire"]),
            evidence_refs=["turn:t1"],
        )

        self.assertEqual(appraisal.appraisal_mode, "engaged_turn")
        self.assertTrue(appraisal.active_user_task)
        self.assertFalse(appraisal.idle_state_detected)
        self.assertFalse(appraisal.internal_goal_formation_allowed)
        self.assertEqual(
            appraisal.internal_goal_formation_reason,
            "stage11_1_appraisal_only_no_goal_synthesis",
        )
        self.assertIn("active user turn is present", appraisal.idle_conditions)
        self.assertIn(
            "awareness:claim gating blocks: unsupported_desire",
            appraisal.pressure_sources,
        )
        self.assertIn("private_cognition:uncertainty", appraisal.pressure_sources)
        self.assertIn("claim_gate:blocked:unsupported_desire", appraisal.pressure_sources)

    def test_appraisal_prompt_blocks_hidden_goal_or_capability_overclaims(self) -> None:
        capability = CapabilityAppraisalEngine().assess(
            user_text="Can you interact with a game outside this environment?",
            tool_registry=default_tool_registry(),
            evidence_refs=["turn:t1"],
        )
        idle = IdlePressureAppraisalEngine().assess(
            session_id="session-a",
            user_text="Can you interact with a game outside this environment?",
            self_state=SelfState(),
            motive_state=MotiveState(session_id="session-a"),
            initiative_state=default_initiative_state(session_id="session-a"),
            awareness_state=AwarenessState(session_id="session-a"),
            private_cognition=PrivateCognitionPacket(),
            claim_gate=ClaimGateDecision(),
            evidence_refs=["turn:t1"],
        )

        block = AppraisalPromptEngine().build_block(
            capability_appraisal=capability,
            idle_appraisal=idle,
            user_text="Can you interact with a game outside this environment?",
        )

        self.assertIn("[Capability and Idle Appraisal]", block)
        self.assertIn("external_game_interaction", block)
        self.assertIn("current runtime access", block)
        self.assertIn("do not claim generated internal goals", block)
        self.assertIn("selected internal goals must wait", block)

