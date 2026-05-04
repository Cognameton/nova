"""Shared Nova 2.0 data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class PersonaState:
    schema_version: str = SCHEMA_VERSION
    name: str = "Nova"
    core_description: str = ""
    tone: str = ""
    values: list[str] = field(default_factory=list)
    commitments: list[str] = field(default_factory=list)
    style_rules: list[str] = field(default_factory=list)
    disallowed_output_patterns: list[str] = field(default_factory=list)
    identity_anchors: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelfState:
    schema_version: str = SCHEMA_VERSION
    identity_summary: str = ""
    current_focus: str = ""
    active_questions: list[str] = field(default_factory=list)
    stable_preferences: list[str] = field(default_factory=list)
    relationship_notes: list[str] = field(default_factory=list)
    continuity_notes: list[str] = field(default_factory=list)
    open_tensions: list[str] = field(default_factory=list)
    last_reflection_at: str | None = None
    stability_version: int = 1
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MotiveState:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    current_priorities: list[str] = field(default_factory=list)
    active_tensions: list[str] = field(default_factory=list)
    local_goals: list[str] = field(default_factory=list)
    claim_posture: str = "conservative"
    evidence_refs: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InitiativeTransition:
    schema_version: str = SCHEMA_VERSION
    from_status: str = ""
    to_status: str = "pending"
    reason: str = ""
    timestamp: str = ""
    approved_by: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InitiativeRecord:
    schema_version: str = SCHEMA_VERSION
    initiative_id: str = ""
    intent_id: str = ""
    session_id: str = ""
    origin_session_id: str = ""
    continued_from_session_id: str | None = None
    continued_from_initiative_id: str | None = None
    title: str = ""
    goal: str = ""
    status: str = "pending"
    approval_required: bool = True
    approved_by: str = ""
    source: str = "runtime"
    created_at: str = ""
    updated_at: str = ""
    last_transition_at: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    related_motive_refs: list[str] = field(default_factory=list)
    related_self_model_refs: list[str] = field(default_factory=list)
    continuation_session_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    transitions: list[InitiativeTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InitiativeState:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    active_initiative_id: str | None = None
    initiatives: list[InitiativeRecord] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AwarenessState:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    monitoring_mode: str = "bounded"
    self_signals: list[str] = field(default_factory=list)
    world_signals: list[str] = field(default_factory=list)
    active_pressures: list[str] = field(default_factory=list)
    candidate_goal_signals: list[str] = field(default_factory=list)
    dominant_attention: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClaimGateDecision:
    schema_version: str = SCHEMA_VERSION
    requested_claim_classes: list[str] = field(default_factory=list)
    allowed_claim_classes: list[str] = field(default_factory=list)
    blocked_claim_classes: list[str] = field(default_factory=list)
    evidence_score_by_class: dict[str, int] = field(default_factory=dict)
    threshold_by_class: dict[str, int] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    claim_posture: str = "conservative"
    refusal_needed: bool = False
    refusal_reason: str = ""
    refusal_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryEvent:
    schema_version: str = SCHEMA_VERSION
    event_id: str = ""
    timestamp: str = ""
    session_id: str = ""
    turn_id: str = ""
    channel: str = ""
    kind: str = ""
    text: str = ""
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.0
    confidence: float = 1.0
    continuity_weight: float = 0.0
    retention: str = "active"
    supersedes: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphFact:
    schema_version: str = SCHEMA_VERSION
    fact_id: str = ""
    timestamp: str = ""
    subject_type: str = ""
    subject_key: str = ""
    relation: str = ""
    object_type: str = ""
    object_key: str = ""
    subject_name: str | None = None
    object_name: str | None = None
    weight: float = 1.0
    confidence: float = 1.0
    continuity_weight: float = 0.0
    active: bool = True
    superseded_by: str | None = None
    evidence_text: str | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalHit:
    channel: str
    text: str
    score: float
    kind: str | None = None
    source_ref: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PrivateCognitionPacket:
    schema_version: str = SCHEMA_VERSION
    enabled: bool = False
    ran: bool = False
    trigger: str = ""
    continuity_risk: str = "low"
    memory_conflict: bool = False
    response_mode: str = "direct_answer"
    revise_needed: bool = False
    uncertainty_flag: bool = False
    pass_budget: int = 0
    pass_budget_used: int = 0
    revision_ceiling: int = 0
    relevant_channels: list[str] = field(default_factory=list)
    governing_memory_ids: list[str] = field(default_factory=list)
    current_claims: list[str] = field(default_factory=list)
    conflict_claim_axes: list[str] = field(default_factory=list)
    provisional_claim_axes: list[str] = field(default_factory=list)
    revision_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptBundle:
    session_id: str
    turn_id: str
    persona_block: str
    self_state_block: str
    motive_block: str
    initiative_block: str
    private_cognition_block: str
    memory_blocks: dict[str, str]
    recent_turns_block: str
    user_block: str
    response_contract_block: str
    full_prompt: str
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationRequest:
    model_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    retries_allowed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    model_id: str
    raw_text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    violations: list[str] = field(default_factory=list)
    sanitized_text: str | None = None
    should_retry: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TurnRecord:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    turn_id: str = ""
    timestamp: str = ""
    user_text: str = ""
    final_answer: str = ""
    raw_answer: str = ""
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(valid=False)
    )
    memory_hits: list[RetrievalHit] = field(default_factory=list)
    prompt_token_estimate: int = 0
    completion_token_estimate: int | None = None
    latency_ms: int | None = None
    model_id: str = ""
    retry_count: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class IdentityHistoryEntry:
    schema_version: str = SCHEMA_VERSION
    entry_id: str = ""
    timestamp: str = ""
    session_id: str = ""
    source_event_id: str = ""
    governance_scope: str = ""
    claim_axis: str = ""
    self_model_status: str = ""
    revision_class: str = ""
    text: str = ""
    superseded_by: str | None = None
    prior_event_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceRecord:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    turn_id: str = ""
    timestamp: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    persona_state_snapshot: dict[str, Any] = field(default_factory=dict)
    self_state_snapshot: dict[str, Any] = field(default_factory=dict)
    motive_state_snapshot: dict[str, Any] = field(default_factory=dict)
    initiative_state_snapshot: dict[str, Any] = field(default_factory=dict)
    awareness_state_snapshot: dict[str, Any] = field(default_factory=dict)
    claim_gate: dict[str, Any] = field(default_factory=dict)
    prompt_bundle: dict[str, Any] = field(default_factory=dict)
    private_cognition: dict[str, Any] = field(default_factory=dict)
    generation_request: dict[str, Any] = field(default_factory=dict)
    generation_result: dict[str, Any] = field(default_factory=dict)
    validation_result: dict[str, Any] = field(default_factory=dict)
    retries: list[dict[str, Any]] = field(default_factory=list)
    persisted_memory_events: list[dict[str, Any]] = field(default_factory=list)
    identity_history_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProbeResult:
    schema_version: str = SCHEMA_VERSION
    probe_id: str = ""
    timestamp: str = ""
    session_id: str | None = None
    model_id: str = ""
    probe_type: str = ""
    prompt: str = ""
    answer: str = ""
    score: float | None = None
    passed: bool | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
