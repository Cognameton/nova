"""Action surface adapters for Phase 17 Stage 17.3.

Adapter interface, registry, scratchpad and log adapters, and the pre-execution
check chain that runs before any adapter is invoked. Adapters are the only path
through which approved action plans reach the filesystem.

Invariants:
- No adapter writes outside nova_owned_paths (fail-closed on confinement check).
- Adapters are never called without a prior approved plan passing
  check_action_plan_for_adapter.
- No adapter registers or handles external-effect surfaces (filesystem, shell,
  network, gui, system_config, external_service).
- Each write creates a new unique file; no existing file is modified or deleted.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nova.agent.action_plan import NOVA_OWNED_ENVIRONMENT_SURFACES
from nova.types import (
    AutonomousActionPlan,
    NovaOwnedExecutionBoundary,
    OperationalAutonomyPolicy,
    SCHEMA_VERSION,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Request / result types ----

@dataclass
class AdapterActionRequest:
    """A single action request routed to a specific surface adapter."""
    surface: str
    content: dict[str, Any]
    step_id: str = ""
    plan_id: str = ""
    session_id: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class AdapterActionResult:
    """The outcome of one adapter execution attempt."""
    surface: str
    adapter_name: str
    executed: bool
    blocked: bool
    block_reason: str
    artifact_path: str = ""
    artifact_id: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "adapter_name": self.adapter_name,
            "executed": self.executed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "artifact_path": self.artifact_path,
            "artifact_id": self.artifact_id,
            "evidence_refs": list(self.evidence_refs),
            "notes": list(self.notes),
        }


# ---- Adapter base class ----

class ActionSurfaceAdapter(ABC):
    """Base class for all action surface adapters."""

    @property
    @abstractmethod
    def surface_name(self) -> str:
        ...

    @abstractmethod
    def execute(
        self,
        *,
        request: AdapterActionRequest,
        boundary: NovaOwnedExecutionBoundary,
        session_id: str,
    ) -> AdapterActionResult:
        """Execute the action and return a result. Must not raise on expected failure."""
        ...


# ---- Concrete adapters ----

class ScratchpadAdapter(ActionSurfaceAdapter):
    """Write JSON entries to the Nova-owned scratchpad directory.

    Each call creates a new unique file under base_dir/<session_id>/.
    No existing file is ever modified or deleted by this adapter.
    """

    @property
    def surface_name(self) -> str:
        return "nova_scratchpad"

    def __init__(self, *, base_dir: Path):
        self._base_dir = Path(base_dir)

    def execute(
        self,
        *,
        request: AdapterActionRequest,
        boundary: NovaOwnedExecutionBoundary,
        session_id: str,
    ) -> AdapterActionResult:
        entry_id = uuid4().hex
        target_dir = self._base_dir / session_id
        target_path = target_dir / f"{entry_id}.json"

        if not _path_is_confined(target_path, boundary.nova_owned_paths):
            return AdapterActionResult(
                surface=self.surface_name,
                adapter_name="ScratchpadAdapter",
                executed=False,
                blocked=True,
                block_reason=f"path_outside_nova_owned:{target_path}",
                notes=["scratchpad_confinement_failed"],
            )

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "entry_id": entry_id,
                "session_id": session_id,
                "surface": self.surface_name,
                "step_id": request.step_id,
                "plan_id": request.plan_id,
                "written_at": _utc_now_iso(),
                "content": request.content,
                "notes": list(request.notes),
            }
            target_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            return AdapterActionResult(
                surface=self.surface_name,
                adapter_name="ScratchpadAdapter",
                executed=False,
                blocked=True,
                block_reason=f"io_error:{type(exc).__name__}",
                notes=["scratchpad_io_error"],
            )

        return AdapterActionResult(
            surface=self.surface_name,
            adapter_name="ScratchpadAdapter",
            executed=True,
            blocked=False,
            block_reason="",
            artifact_path=str(target_path),
            artifact_id=entry_id,
            evidence_refs=[f"scratchpad_entry:{entry_id}"],
            notes=["scratchpad_entry_written"],
        )


class OperationalLogAdapter(ActionSurfaceAdapter):
    """Write JSON entries to the Nova-owned operational log directory.

    Each call creates a new unique file under base_dir/<session_id>/.
    No existing file is ever modified or deleted by this adapter.
    """

    @property
    def surface_name(self) -> str:
        return "nova_logs"

    def __init__(self, *, base_dir: Path):
        self._base_dir = Path(base_dir)

    def execute(
        self,
        *,
        request: AdapterActionRequest,
        boundary: NovaOwnedExecutionBoundary,
        session_id: str,
    ) -> AdapterActionResult:
        entry_id = uuid4().hex
        target_dir = self._base_dir / session_id
        target_path = target_dir / f"{entry_id}.json"

        if not _path_is_confined(target_path, boundary.nova_owned_paths):
            return AdapterActionResult(
                surface=self.surface_name,
                adapter_name="OperationalLogAdapter",
                executed=False,
                blocked=True,
                block_reason=f"path_outside_nova_owned:{target_path}",
                notes=["log_confinement_failed"],
            )

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "entry_id": entry_id,
                "session_id": session_id,
                "surface": self.surface_name,
                "step_id": request.step_id,
                "plan_id": request.plan_id,
                "written_at": _utc_now_iso(),
                "content": request.content,
                "notes": list(request.notes),
            }
            target_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            return AdapterActionResult(
                surface=self.surface_name,
                adapter_name="OperationalLogAdapter",
                executed=False,
                blocked=True,
                block_reason=f"io_error:{type(exc).__name__}",
                notes=["log_io_error"],
            )

        return AdapterActionResult(
            surface=self.surface_name,
            adapter_name="OperationalLogAdapter",
            executed=True,
            blocked=False,
            block_reason="",
            artifact_path=str(target_path),
            artifact_id=entry_id,
            evidence_refs=[f"operational_log_entry:{entry_id}"],
            notes=["operational_log_entry_written"],
        )


# ---- Registry ----

class AdapterRegistry:
    """Map surface names to adapter instances."""

    def __init__(self, adapters: list[ActionSurfaceAdapter] | None = None):
        self._adapters: dict[str, ActionSurfaceAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: ActionSurfaceAdapter) -> None:
        self._adapters[adapter.surface_name] = adapter

    def get(self, surface_name: str) -> ActionSurfaceAdapter | None:
        return self._adapters.get(surface_name)

    def surfaces(self) -> list[str]:
        return sorted(self._adapters.keys())


# ---- Pre-execution check chain ----

def check_action_plan_for_adapter(
    *,
    plan: AutonomousActionPlan,
    policy: OperationalAutonomyPolicy,
    boundary: NovaOwnedExecutionBoundary,
    registry: AdapterRegistry,
) -> str:
    """Return a non-empty block reason if the plan cannot proceed through adapters.

    Checks (in order):
    1. plan.status must be "approved"
    2. plan.permission.approved must be True
    3. plan.execution_lane must be "nova_owned_environment"
    4. Each surface in plan.allowed_surfaces must be:
       a. in NOVA_OWNED_ENVIRONMENT_SURFACES
       b. in policy.allowed_surfaces
       c. in boundary.allowed_surfaces  (Stage 17.3 rule 4 — placed here, not
          in boundary.check_operational_boundary, because this is a plan-level
          check: the policy may list nova_owned surfaces without triggering a
          blanket boundary violation; we only enforce the subset requirement when
          an actual plan requests execution on those surfaces)
       d. not in boundary.blocked_surfaces
    5. plan must have at least one step
    6. each step.surface must have a registered adapter
    """
    if plan.status != "approved":
        return f"plan_not_approved:{plan.status}"
    if not plan.permission.approved:
        return "plan_permission_not_approved"
    if plan.execution_lane != "nova_owned_environment":
        return f"plan_lane_not_nova_owned:{plan.execution_lane}"

    policy_surfaces = set(policy.allowed_surfaces)
    boundary_allowed = set(boundary.allowed_surfaces)
    boundary_blocked = set(boundary.blocked_surfaces)

    for surface in plan.allowed_surfaces:
        if surface not in NOVA_OWNED_ENVIRONMENT_SURFACES:
            return f"surface_not_nova_owned_environment:{surface}"
        if surface not in policy_surfaces:
            return f"surface_not_in_policy_allowed:{surface}"
        if surface not in boundary_allowed:
            return f"surface_not_in_boundary_allowed:{surface}"
        if surface in boundary_blocked:
            return f"surface_in_boundary_blocked:{surface}"

    if not plan.steps:
        return "plan_has_no_steps"

    for step in plan.steps:
        if registry.get(step.surface) is None:
            return f"no_adapter_for_surface:{step.surface}"

    return ""


# ---- Execution ----

def execute_plan_through_adapters(
    *,
    plan: AutonomousActionPlan,
    registry: AdapterRegistry,
    boundary: NovaOwnedExecutionBoundary,
    session_id: str,
    actions_budget_remaining: int,
) -> list[AdapterActionResult]:
    """Execute each step of an approved plan through the registered adapter.

    Stops at the first blocked step or when the action budget is exhausted.
    Returns results for all attempted steps (executed and blocked).
    """
    results: list[AdapterActionResult] = []
    budget_remaining = max(0, actions_budget_remaining)

    for step in plan.steps:
        if budget_remaining == 0:
            results.append(AdapterActionResult(
                surface=step.surface,
                adapter_name="none",
                executed=False,
                blocked=True,
                block_reason="action_budget_exhausted_mid_plan",
                notes=["budget_exhausted_stopping_plan"],
            ))
            break

        adapter = registry.get(step.surface)
        if adapter is None:
            results.append(AdapterActionResult(
                surface=step.surface,
                adapter_name="none",
                executed=False,
                blocked=True,
                block_reason=f"no_adapter_for_surface:{step.surface}",
                notes=["adapter_not_found_at_execution_time"],
            ))
            break

        request = AdapterActionRequest(
            surface=step.surface,
            content={
                "description": step.description,
                "expected_output": step.expected_output,
            },
            step_id=step.step_id,
            plan_id=plan.action_plan_id,
            session_id=session_id,
            notes=list(step.notes),
        )
        result = adapter.execute(
            request=request,
            boundary=boundary,
            session_id=session_id,
        )
        results.append(result)
        if result.executed:
            budget_remaining -= 1
        if result.blocked:
            break

    return results


def adapter_audit_from_results(results: list[AdapterActionResult]) -> dict[str, Any]:
    """Build an adapter_audit dict for OperationalTickRecord.adapter_audit."""
    return {
        "steps_attempted": len(results),
        "steps_executed": sum(1 for r in results if r.executed),
        "steps_blocked": sum(1 for r in results if r.blocked),
        "results": [r.to_dict() for r in results],
    }


# ---- Path confinement helper ----

def _path_is_confined(path: Path, nova_owned_paths: list[str]) -> bool:
    """Return True iff path is under at least one nova_owned_path."""
    if not nova_owned_paths:
        return False
    for owned in nova_owned_paths:
        try:
            path.relative_to(Path(owned))
            return True
        except ValueError:
            continue
    return False
