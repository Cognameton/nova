"""Local execution boundary check for Phase 17 Stage 17.2.

Provides a deterministic, fail-closed check of the Nova-owned execution
boundary before each operational tick. Stage 17.3 action adapters call
the same function before executing any approved surface action.

Fail-closed rules (any violation → satisfied=False):
  1. boundary and policy objects must be valid instances
  2. dedicated_user_required and not dedicated_user_detected → blocked
  3. policy must name at least one allowed surface
  4. no policy allowed_surface may appear in boundary.blocked_surfaces

Stage 17.3 note: the boundary.allowed_surfaces subset check for
nova_owned_environment surfaces lives in check_action_plan_for_adapter
(nova.agent.action_surface). The boundary module checks policy-level
invariants only; whether a plan's surfaces are within the boundary's
allowed list is a plan-level concern checked just before adapter execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nova.types import NovaOwnedExecutionBoundary, OperationalAutonomyPolicy


@dataclass(slots=True)
class BoundaryCheckResult:
    satisfied: bool
    violations: list[str] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)


def check_operational_boundary(
    boundary: NovaOwnedExecutionBoundary | None,
    policy: OperationalAutonomyPolicy | None,
) -> BoundaryCheckResult:
    """Check the Nova-owned execution boundary before an operational tick.

    Returns a result with satisfied=False if any rule is violated. The
    snapshot field is always populated so blocked ticks carry a full
    diagnostic record regardless of which rule triggered.
    """
    if not isinstance(boundary, NovaOwnedExecutionBoundary):
        return BoundaryCheckResult(
            satisfied=False,
            violations=["invalid_boundary_object"],
            snapshot={},
        )

    if not isinstance(policy, OperationalAutonomyPolicy):
        return BoundaryCheckResult(
            satisfied=False,
            violations=["invalid_policy_object"],
            snapshot=_build_snapshot(boundary, ["invalid_policy_object"]),
        )

    violations: list[str] = []

    # Rule 1: dedicated OS user must be detected when required
    if boundary.dedicated_user_required and not boundary.dedicated_user_detected:
        violations.append(
            "dedicated_user_not_detected:"
            f"expected={boundary.expected_os_user!r},"
            f"active={boundary.active_os_user!r}"
        )

    # Rule 2: policy must name at least one allowed surface
    policy_surfaces = set(policy.allowed_surfaces)
    if not policy_surfaces:
        violations.append("policy_has_no_allowed_surfaces")
    else:
        # Rule 3: no policy surface may appear in the boundary blocklist
        boundary_blocked = set(boundary.blocked_surfaces)
        for surface in sorted(policy_surfaces & boundary_blocked):
            violations.append(f"surface_in_boundary_blocklist:{surface}")

    snapshot = _build_snapshot(boundary, violations)
    return BoundaryCheckResult(
        satisfied=not violations,
        violations=violations,
        snapshot=snapshot,
    )


def _build_snapshot(
    boundary: NovaOwnedExecutionBoundary,
    violations: list[str],
) -> dict[str, Any]:
    data = boundary.to_dict()
    data["satisfied"] = not violations
    data["violations"] = list(violations)
    return data
