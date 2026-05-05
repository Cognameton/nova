"""Capability and idle-pressure appraisal for Nova Phase 11 Stage 1."""

from __future__ import annotations

from nova.agent.tools import TOOL_ALLOWED, TOOL_APPROVAL_REQUIRED, TOOL_BLOCKED
from nova.agent.tool_registry import ToolRegistry
from nova.types import (
    AwarenessState,
    CapabilityAppraisal,
    ClaimGateDecision,
    IdlePressureAppraisal,
    InitiativeRecord,
    InitiativeState,
    MotiveState,
    PrivateCognitionPacket,
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
