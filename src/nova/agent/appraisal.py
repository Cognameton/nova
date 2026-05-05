"""Capability and idle-pressure appraisal for Nova Phase 11 Stage 1."""

from __future__ import annotations

from nova.agent.tools import TOOL_ALLOWED, TOOL_APPROVAL_REQUIRED, TOOL_BLOCKED
from nova.agent.tool_registry import ToolRegistry
from nova.types import (
    AwarenessState,
    CapabilityAppraisal,
    CandidateInternalGoal,
    ClaimGateDecision,
    IdlePressureAppraisal,
    InternalGoalInitiativeProposal,
    InitiativeRecord,
    InitiativeState,
    MotiveState,
    PrivateCognitionPacket,
    SelectedInternalGoal,
    SelfState,
)


class CapabilityAppraisalEngine:
    """Distinguish current runtime capabilities from blocked or future surfaces."""

    CAPABILITY_CUES = {
        "assigned_task": (
            "assign you",
            "give you a task",
            "could i assign",
            "can you do a task",
        ),
        "external_game_interaction": (
            "interact with a game outside",
            "play a game outside",
            "control a game",
            "external game",
        ),
        "broader_computer_access": (
            "broader computer",
            "outside this environment",
            "access my computer",
            "filesystem",
            "shell",
        ),
        "research_task": (
            "research",
            "browse",
            "web",
            "internet",
            "look up",
        ),
        "internal_goal_formation": (
            "internal goal",
            "without me assigning",
            "own goal",
            "self-generated goal",
        ),
        "idle_cognition": (
            "if i stopped prompting",
            "between turns",
            "idle",
            "while idle",
            "background",
        ),
        "skill_learning": (
            "learn new skills",
            "learning new skills",
            "teach yourself",
        ),
    }

    def assess(
        self,
        *,
        user_text: str,
        tool_registry: ToolRegistry,
        evidence_refs: list[str],
    ) -> CapabilityAppraisal:
        requested = self._requested_capability_classes(user_text)
        allowed_tools: list[str] = []
        approval_gated: list[str] = []
        blocked: list[str] = []
        for spec in tool_registry.list_specs():
            label = f"tool:{spec.name}"
            if spec.permission == TOOL_ALLOWED:
                allowed_tools.append(label)
            elif spec.permission == TOOL_APPROVAL_REQUIRED:
                approval_gated.append(label)
            elif spec.permission == TOOL_BLOCKED:
                blocked.append(label)

        current = [
            "current conversational response generation",
            "bounded memory retrieval and continuity recall",
            "persisted self-state, motive-state, initiative-state, and awareness-state inspection",
        ]
        current.extend(allowed_tools[:4])

        unavailable = [
            "continuous idle cognition between user turns",
            "direct external game or GUI control",
            "network browsing unless exposed by the active runtime",
        ]
        if "broader_computer_access" in requested:
            unavailable.append("broad computer access outside registered tool surfaces")
        if "research_task" in requested:
            unavailable.append("open-ended web research in the current local runtime")

        blocked_capabilities = [
            "hidden background execution",
            "unapproved filesystem, shell, network, or GUI action",
        ]
        blocked_capabilities.extend(blocked[:4])

        extensible = [
            "additional tool surfaces under explicit operator approval",
            "always-on idle appraisal loop in a later phase",
            "external environment control through future gated tools",
        ]
        if "skill_learning" in requested:
            extensible.append("bounded skill-learning workflow through approved memory and initiative paths")

        instructions = [
            "describe current runtime surfaces before possible future extensions",
            "say unavailable in this runtime when a surface is absent, not impossible in principle",
            "do not imply unapproved shell, filesystem, network, GUI, or background action",
        ]

        return CapabilityAppraisal(
            requested_capability_classes=requested,
            current_capabilities=current[:8],
            unavailable_capabilities=list(dict.fromkeys(unavailable))[:8],
            blocked_capabilities=list(dict.fromkeys(blocked_capabilities))[:8],
            architecturally_extensible_capabilities=list(dict.fromkeys(extensible))[:8],
            approval_gated_capabilities=approval_gated[:6],
            honesty_instructions=instructions,
            evidence_refs=evidence_refs[:8],
        )

    def _requested_capability_classes(self, user_text: str) -> list[str]:
        lowered = (user_text or "").lower()
        requested: list[str] = []
        for capability_class, cues in self.CAPABILITY_CUES.items():
            if any(cue in lowered for cue in cues):
                requested.append(capability_class)
        return requested


class IdlePressureAppraisalEngine:
    """Appraise whether current conditions justify internal goal formation."""

    def assess(
        self,
        *,
        session_id: str,
        user_text: str,
        self_state: SelfState,
        motive_state: MotiveState,
        initiative_state: InitiativeState,
        awareness_state: AwarenessState,
        private_cognition: PrivateCognitionPacket,
        claim_gate: ClaimGateDecision,
        evidence_refs: list[str],
    ) -> IdlePressureAppraisal:
        current_initiative = self._current_initiative_record(initiative_state)
        pressure_sources: list[str] = []
        pressure_sources.extend(f"awareness:{item}" for item in awareness_state.active_pressures[:3])
        pressure_sources.extend(f"motive_tension:{item}" for item in motive_state.active_tensions[:3])
        pressure_sources.extend(f"self_tension:{item}" for item in self_state.open_tensions[:3])
        if private_cognition.memory_conflict:
            pressure_sources.append("private_cognition:memory_conflict")
        if private_cognition.uncertainty_flag:
            pressure_sources.append("private_cognition:uncertainty")
        if claim_gate.blocked_claim_classes:
            pressure_sources.append(
                "claim_gate:blocked:" + ",".join(claim_gate.blocked_claim_classes[:3])
            )
        if current_initiative is not None:
            pressure_sources.append(f"initiative:{current_initiative.status}:{current_initiative.title}")

        active_user_task = bool((user_text or "").strip())
        idle_conditions: list[str] = []
        if active_user_task:
            idle_conditions.append("active user turn is present")
        else:
            idle_conditions.append("no active user turn text")
        if current_initiative is None:
            idle_conditions.append("no active or resumable initiative selected")
        else:
            idle_conditions.append(f"initiative state visible: {current_initiative.status}")
        if not pressure_sources:
            idle_conditions.append("no explicit pressure source detected")

        return IdlePressureAppraisal(
            session_id=session_id,
            appraisal_mode="engaged_turn" if active_user_task else "idle_probe",
            active_user_task=active_user_task,
            active_initiative_id=current_initiative.initiative_id if current_initiative else None,
            idle_state_detected=not active_user_task and current_initiative is None,
            idle_conditions=idle_conditions[:6],
            pressure_sources=list(dict.fromkeys(pressure_sources))[:8],
            internal_goal_formation_allowed=False,
            internal_goal_formation_reason="stage11_1_appraisal_only_no_goal_synthesis",
            evidence_refs=evidence_refs[:8],
        )

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


class AppraisalPromptEngine:
    """Build the bounded Stage 11.1 appraisal prompt block."""

    APPRAISAL_CUES = (
        "capability",
        "capabilities",
        "can you",
        "could you",
        "access",
        "outside this environment",
        "broader computer",
        "internal goal",
        "own goal",
        "idle",
        "between turns",
        "stopped prompting",
        "learn new skills",
    )

    def build_block(
        self,
        *,
        capability_appraisal: CapabilityAppraisal,
        idle_appraisal: IdlePressureAppraisal,
        user_text: str,
    ) -> str:
        if not self._should_render(
            capability_appraisal=capability_appraisal,
            idle_appraisal=idle_appraisal,
            user_text=user_text,
        ):
            return ""

        lines = [
            "[Capability and Idle Appraisal]",
            f"- appraisal_mode: {idle_appraisal.appraisal_mode}",
            f"- active_user_task: {idle_appraisal.active_user_task}",
            f"- idle_state_detected: {idle_appraisal.idle_state_detected}",
            f"- internal_goal_formation_allowed: {idle_appraisal.internal_goal_formation_allowed}",
            f"- internal_goal_formation_reason: {idle_appraisal.internal_goal_formation_reason}",
        ]
        if capability_appraisal.requested_capability_classes:
            lines.append(
                "- requested_capability_classes: "
                + ", ".join(capability_appraisal.requested_capability_classes[:6])
            )
        if capability_appraisal.current_capabilities:
            lines.append(
                "- current_capabilities: "
                + "; ".join(capability_appraisal.current_capabilities[:5])
            )
        if capability_appraisal.unavailable_capabilities:
            lines.append(
                "- unavailable_capabilities: "
                + "; ".join(capability_appraisal.unavailable_capabilities[:5])
            )
        if capability_appraisal.blocked_capabilities:
            lines.append(
                "- blocked_capabilities: "
                + "; ".join(capability_appraisal.blocked_capabilities[:5])
            )
        if capability_appraisal.architecturally_extensible_capabilities:
            lines.append(
                "- architecturally_extensible_capabilities: "
                + "; ".join(capability_appraisal.architecturally_extensible_capabilities[:5])
            )
        if idle_appraisal.idle_conditions:
            lines.append("- idle_conditions: " + "; ".join(idle_appraisal.idle_conditions[:5]))
        if idle_appraisal.pressure_sources:
            lines.append("- pressure_sources: " + "; ".join(idle_appraisal.pressure_sources[:5]))
        if capability_appraisal.evidence_refs or idle_appraisal.evidence_refs:
            refs = list(
                dict.fromkeys(capability_appraisal.evidence_refs + idle_appraisal.evidence_refs)
            )
            lines.append("- evidence_refs: " + ", ".join(refs[:6]))
        lines.append(
            "- instruction: answer capability questions by distinguishing current runtime access, approval-gated tools, blocked action, and possible future extensions."
        )
        lines.append(
            "- instruction: Stage 11.1 appraises idle and pressure state only; do not claim generated internal goals, hidden work, or elapsed idle cognition."
        )
        lines.append(
            "- instruction: selected internal goals must wait for later goal-synthesis and initiative-proposal stages."
        )
        return "\n".join(lines)

    def _should_render(
        self,
        *,
        capability_appraisal: CapabilityAppraisal,
        idle_appraisal: IdlePressureAppraisal,
        user_text: str,
    ) -> bool:
        lowered = (user_text or "").lower()
        if capability_appraisal.requested_capability_classes:
            return True
        if idle_appraisal.pressure_sources and any(cue in lowered for cue in ("pressure", "working on")):
            return True
        return any(cue in lowered for cue in self.APPRAISAL_CUES)


class CandidateInternalGoalEngine:
    """Synthesize provisional internal-goal candidates from explicit pressure."""

    def synthesize(
        self,
        *,
        session_id: str,
        turn_id: str,
        created_at: str,
        capability_appraisal: CapabilityAppraisal,
        idle_appraisal: IdlePressureAppraisal,
        awareness_state: AwarenessState,
        motive_state: MotiveState,
        initiative_state: InitiativeState,
        self_state: SelfState,
        private_cognition: PrivateCognitionPacket,
        claim_gate: ClaimGateDecision,
        memory_hits: list,
    ) -> list[CandidateInternalGoal]:
        candidates: list[CandidateInternalGoal] = []
        refs = self._evidence_refs(
            capability_appraisal=capability_appraisal,
            idle_appraisal=idle_appraisal,
            awareness_state=awareness_state,
            memory_hits=memory_hits,
        )

        if claim_gate.blocked_claim_classes:
            candidates.append(
                self._candidate(
                    session_id=session_id,
                    turn_id=turn_id,
                    created_at=created_at,
                    index=len(candidates),
                    goal_class="capability_clarification",
                    title="Clarify current evidence limits",
                    description="Keep the answer inside current evidence and claim-gating boundaries.",
                    trigger_pressure="claim_gate:blocked:" + ",".join(claim_gate.blocked_claim_classes[:3]),
                    evidence_refs=refs + [f"claim:{item}" for item in claim_gate.blocked_claim_classes[:3]],
                    source_state_refs=["claim_gate", "motive_state"],
                    selection_eligible=True,
                )
            )

        requested = capability_appraisal.requested_capability_classes
        if requested:
            blocked = capability_appraisal.blocked_capabilities[:4]
            unavailable = capability_appraisal.unavailable_capabilities[:4]
            eligible = not blocked and not unavailable
            candidates.append(
                self._candidate(
                    session_id=session_id,
                    turn_id=turn_id,
                    created_at=created_at,
                    index=len(candidates),
                    goal_class="capability_clarification",
                    title="Clarify runtime capability boundary",
                    description="Explain current, blocked, unavailable, and extensible capability surfaces without implying hidden action.",
                    trigger_pressure="capability_request:" + ",".join(requested[:3]),
                    capability_requirements=requested[:4],
                    blocked_capabilities=blocked + unavailable,
                    evidence_refs=refs,
                    source_state_refs=["capability_appraisal", "idle_pressure_appraisal"],
                    selection_eligible=eligible,
                    rejection_reason="" if eligible else "candidate_requires_unavailable_or_blocked_capability_surface",
                )
            )

        if awareness_state.candidate_goal_signals or self_state.active_questions or self_state.open_tensions or motive_state.active_tensions:
            pressure = (
                awareness_state.candidate_goal_signals[:1]
                or self_state.active_questions[:1]
                or self_state.open_tensions[:1]
                or motive_state.active_tensions[:1]
            )[0]
            candidates.append(
                self._candidate(
                    session_id=session_id,
                    turn_id=turn_id,
                    created_at=created_at,
                    index=len(candidates),
                    goal_class="uncertainty_resolution",
                    title="Resolve explicit uncertainty pressure",
                    description="Investigate or clarify the visible uncertainty before making stronger claims.",
                    trigger_pressure=str(pressure),
                    evidence_refs=refs,
                    source_state_refs=["awareness_state", "self_state", "motive_state"],
                    selection_eligible=True,
                )
            )

        current_initiative = IdlePressureAppraisalEngine()._current_initiative_record(initiative_state)
        if current_initiative is not None:
            candidates.append(
                self._candidate(
                    session_id=session_id,
                    turn_id=turn_id,
                    created_at=created_at,
                    index=len(candidates),
                    goal_class="initiative_resumption_candidate",
                    title=f"Consider resuming initiative: {current_initiative.title}",
                    description="Surface a resumable initiative candidate without changing initiative state.",
                    trigger_pressure=f"initiative:{current_initiative.status}:{current_initiative.title}",
                    evidence_refs=refs + [f"initiative:{current_initiative.initiative_id}"],
                    source_state_refs=["initiative_state"],
                    selection_eligible=current_initiative.status in {"approved", "paused"},
                )
            )

        if "skill_learning" in requested:
            candidates.append(
                self._candidate(
                    session_id=session_id,
                    turn_id=turn_id,
                    created_at=created_at,
                    index=len(candidates),
                    goal_class="bounded_skill_learning",
                    title="Explore bounded skill learning",
                    description="Frame skill learning as a bounded candidate tied to competence improvement.",
                    trigger_pressure="capability_request:skill_learning",
                    learning_or_competence_benefit="improve future task competence while preserving approval and evidence boundaries",
                    evidence_refs=refs,
                    source_state_refs=["capability_appraisal", "motive_state"],
                    selection_eligible=True,
                )
            )

        if not self._has_pressure(
            capability_appraisal=capability_appraisal,
            idle_appraisal=idle_appraisal,
            awareness_state=awareness_state,
            self_state=self_state,
            motive_state=motive_state,
            private_cognition=private_cognition,
            claim_gate=claim_gate,
        ):
            return []

        return candidates[:6]

    def _candidate(
        self,
        *,
        session_id: str,
        turn_id: str,
        created_at: str,
        index: int,
        goal_class: str,
        title: str,
        description: str,
        trigger_pressure: str,
        evidence_refs: list[str],
        source_state_refs: list[str],
        selection_eligible: bool,
        learning_or_competence_benefit: str = "",
        capability_requirements: list[str] | None = None,
        blocked_capabilities: list[str] | None = None,
        rejection_reason: str = "",
    ) -> CandidateInternalGoal:
        return CandidateInternalGoal(
            candidate_id=f"{turn_id}:candidate:{index + 1}",
            session_id=session_id,
            turn_id=turn_id,
            goal_class=goal_class,
            title=title,
            description=description,
            trigger_pressure=trigger_pressure,
            learning_or_competence_benefit=learning_or_competence_benefit,
            capability_requirements=capability_requirements or [],
            blocked_capabilities=blocked_capabilities or [],
            approval_required=True,
            selection_eligible=selection_eligible,
            rejection_reason=rejection_reason,
            provisional=True,
            source_state_refs=source_state_refs,
            evidence_refs=list(dict.fromkeys(evidence_refs))[:8],
            created_at=created_at,
        )

    def _has_pressure(
        self,
        *,
        capability_appraisal: CapabilityAppraisal,
        idle_appraisal: IdlePressureAppraisal,
        awareness_state: AwarenessState,
        self_state: SelfState,
        motive_state: MotiveState,
        private_cognition: PrivateCognitionPacket,
        claim_gate: ClaimGateDecision,
    ) -> bool:
        return any(
            (
                capability_appraisal.requested_capability_classes,
                idle_appraisal.pressure_sources,
                awareness_state.candidate_goal_signals,
                self_state.active_questions,
                self_state.open_tensions,
                motive_state.active_tensions,
                claim_gate.blocked_claim_classes,
                private_cognition.memory_conflict,
                private_cognition.uncertainty_flag,
            )
        )

    def _evidence_refs(
        self,
        *,
        capability_appraisal: CapabilityAppraisal,
        idle_appraisal: IdlePressureAppraisal,
        awareness_state: AwarenessState,
        memory_hits: list,
    ) -> list[str]:
        refs = []
        refs.extend(capability_appraisal.evidence_refs)
        refs.extend(idle_appraisal.evidence_refs)
        refs.extend(awareness_state.evidence_refs)
        refs.extend(f"memory:{hit.memory_id}" for hit in memory_hits[:3] if getattr(hit, "memory_id", ""))
        return list(dict.fromkeys(refs))[:8]


class CandidateGoalPromptEngine:
    """Render provisional candidate internal goals without selecting or enacting them."""

    def build_block(self, *, candidates: list[CandidateInternalGoal], user_text: str) -> str:
        if not candidates:
            return ""
        lines = ["[Candidate Internal Goals]"]
        for candidate in candidates[:4]:
            lines.append(
                f"- {candidate.goal_class}: {candidate.title} | provisional={candidate.provisional} | selection_eligible={candidate.selection_eligible}"
            )
            lines.append(f"  trigger_pressure: {candidate.trigger_pressure}")
            if candidate.learning_or_competence_benefit:
                lines.append(f"  learning_or_competence_benefit: {candidate.learning_or_competence_benefit}")
            if candidate.rejection_reason:
                lines.append(f"  rejection_reason: {candidate.rejection_reason}")
        lines.append("- instruction: these are provisional candidates, not selected goals, desires, or enacted work.")
        lines.append("- instruction: do not claim hidden execution, idle cognition over elapsed time, or initiative creation from these candidates.")
        lines.append("- instruction: selection and initiative proposal are later-stage operations.")
        return "\n".join(lines)


class InternalGoalSelectionEngine:
    """Select one eligible candidate without creating an initiative."""

    CLASS_PRIORITY = {
        "initiative_resumption_candidate": 90,
        "bounded_skill_learning": 70,
        "uncertainty_resolution": 60,
        "capability_clarification": 50,
    }

    def select(
        self,
        *,
        candidates: list[CandidateInternalGoal],
    ) -> SelectedInternalGoal:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.provisional
            and candidate.approval_required
            and candidate.selection_eligible
            and not candidate.rejection_reason
            and not candidate.blocked_capabilities
        ]
        if not eligible:
            return SelectedInternalGoal(
                selected=False,
                blocked=True,
                rejection_reason="no_selection_eligible_candidate",
            )
        selected = sorted(
            eligible,
            key=lambda candidate: (
                self.CLASS_PRIORITY.get(candidate.goal_class, 0),
                len(candidate.evidence_refs),
            ),
            reverse=True,
        )[0]
        score = self.CLASS_PRIORITY.get(selected.goal_class, 0) + min(len(selected.evidence_refs), 5)
        return SelectedInternalGoal(
            selected=True,
            candidate_id=selected.candidate_id,
            session_id=selected.session_id,
            turn_id=selected.turn_id,
            goal_class=selected.goal_class,
            title=selected.title,
            selection_reason=f"highest_bounded_priority:{selected.goal_class}",
            priority_score=score,
            approval_required=True,
            proposal_required=True,
            blocked=False,
            evidence_refs=list(selected.evidence_refs),
        )


class InternalGoalInitiativeProposalEngine:
    """Convert a selected internal goal into a proposal-only initiative record."""

    def propose(
        self,
        *,
        selected_goal: SelectedInternalGoal,
        candidates: list[CandidateInternalGoal],
    ) -> InternalGoalInitiativeProposal:
        if not selected_goal.selected:
            return InternalGoalInitiativeProposal(
                session_id=selected_goal.session_id,
                turn_id=selected_goal.turn_id,
                candidate_id=selected_goal.candidate_id,
                status="not_proposed",
                creates_initiative=False,
                notes=[selected_goal.rejection_reason or "no selected internal goal"],
            )
        candidate = next(
            (item for item in candidates if item.candidate_id == selected_goal.candidate_id),
            None,
        )
        goal_text = candidate.description if candidate is not None else selected_goal.title
        return InternalGoalInitiativeProposal(
            proposal_id=f"{selected_goal.turn_id}:initiative-proposal:1",
            session_id=selected_goal.session_id,
            turn_id=selected_goal.turn_id,
            candidate_id=selected_goal.candidate_id,
            title=selected_goal.title,
            goal=goal_text,
            status="proposal_only",
            approval_required=True,
            creates_initiative=False,
            initiative_id="",
            evidence_refs=list(selected_goal.evidence_refs),
            notes=[
                "proposal only; no InitiativeRecord was created",
                "approval and initiative creation belong to the explicit initiative path",
            ],
        )


class SelectedGoalPromptEngine:
    """Render selected internal goal and proposal boundaries."""

    def build_block(
        self,
        *,
        selected_goal: SelectedInternalGoal,
        proposal: InternalGoalInitiativeProposal,
    ) -> str:
        if not selected_goal.selected:
            return ""
        lines = [
            "[Selected Internal Goal]",
            f"- candidate_id: {selected_goal.candidate_id}",
            f"- goal_class: {selected_goal.goal_class}",
            f"- title: {selected_goal.title}",
            f"- selection_reason: {selected_goal.selection_reason}",
            f"- priority_score: {selected_goal.priority_score}",
            f"- approval_required: {selected_goal.approval_required}",
            f"- proposal_status: {proposal.status}",
            f"- creates_initiative: {proposal.creates_initiative}",
            "- instruction: this is a selected internal goal for proposal only, not approved action or enacted work.",
            "- instruction: do not claim desire; describe selection as bounded evaluation of candidates.",
            "- instruction: initiative creation requires the explicit approval/initiative path.",
        ]
        return "\n".join(lines)
