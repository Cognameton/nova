"""Agent-facing self-orientation components for Nova."""

from nova.agent.action import (
    ActionApproval,
    ActionHistoryAnalyzer,
    ActionHistoryReport,
    ActionExecutionResult,
    ActionProposal,
    ActionProposalEngine,
    ActionProposalEvaluation,
)
from nova.agent.action_plan import (
    ACTION_RISK_CLASSES,
    ACTION_SURFACES,
    ActionExecutionController,
    ActionPlanBoundaryError,
    BoundedActionPlanEngine,
    EXECUTION_LANES,
    action_audit_record_from_payload,
    action_budget_from_payload,
    action_permission_from_payload,
    action_plan_from_payload,
    action_plan_step_from_payload,
    approval_required_for_action,
    default_nova_owned_execution_boundary,
    execution_boundary_from_payload,
    normalize_action_risk_class,
    normalize_action_surface,
    normalize_execution_lane,
)
from nova.agent.initiative import (
    AutonomousInitiativeDraftError,
    AutonomousInitiativeRevisionEngine,
    InitiativeTransitionError,
    JsonInitiativeStateStore,
)
from nova.agent.idle import BoundedIdleController, IdleRuntimePromptEngine, JsonIdleRuntimeStore
from nova.agent.awareness import AwarenessClassifier, AwarenessResult
from nova.agent.boundaries import BoundaryPolicy, OperationalLatitude
from nova.agent.orientation import OrientationSnapshot, SelfOrientationEngine
from nova.agent.orientation_eval import OrientationEvaluationResult, OrientationStabilityEvaluator
from nova.agent.presence import JsonPresenceStore, PresenceState
from nova.agent.stability import (
    ContextPressureOrientationChecker,
    ContextPressureOrientationReport,
    MaintenanceOrientationReport,
    MaintenanceOrientationStabilityChecker,
    OrientationConfidenceReport,
    OrientationHistoryAnalyzer,
    OrientationReadinessReport,
)
from nova.agent.tool_gate import ToolGate
from nova.agent.tool_executor import InternalToolExecutor
from nova.agent.tool_registry import ToolRegistry, default_tool_registry
from nova.agent.tools import ToolGateDecision, ToolRequest, ToolResult, ToolSpec

__all__ = [
    "ActionApproval",
    "ActionHistoryAnalyzer",
    "ActionHistoryReport",
    "ActionExecutionResult",
    "ActionProposal",
    "ActionProposalEngine",
    "ActionProposalEvaluation",
    "ACTION_RISK_CLASSES",
    "ACTION_SURFACES",
    "ActionExecutionController",
    "ActionPlanBoundaryError",
    "EXECUTION_LANES",
    "AutonomousInitiativeDraftError",
    "AutonomousInitiativeRevisionEngine",
    "AwarenessClassifier",
    "AwarenessResult",
    "BoundedActionPlanEngine",
    "BoundaryPolicy",
    "BoundedIdleController",
    "ContextPressureOrientationChecker",
    "ContextPressureOrientationReport",
    "InternalToolExecutor",
    "InitiativeTransitionError",
    "JsonIdleRuntimeStore",
    "JsonInitiativeStateStore",
    "JsonPresenceStore",
    "MaintenanceOrientationReport",
    "MaintenanceOrientationStabilityChecker",
    "OperationalLatitude",
    "OrientationConfidenceReport",
    "OrientationEvaluationResult",
    "OrientationHistoryAnalyzer",
    "IdleRuntimePromptEngine",
    "OrientationReadinessReport",
    "OrientationSnapshot",
    "OrientationStabilityEvaluator",
    "PresenceState",
    "SelfOrientationEngine",
    "ToolGate",
    "ToolGateDecision",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
    "ToolSpec",
    "action_audit_record_from_payload",
    "action_budget_from_payload",
    "action_permission_from_payload",
    "action_plan_from_payload",
    "action_plan_step_from_payload",
    "approval_required_for_action",
    "default_nova_owned_execution_boundary",
    "default_tool_registry",
    "execution_boundary_from_payload",
    "normalize_action_risk_class",
    "normalize_action_surface",
    "normalize_execution_lane",
]
