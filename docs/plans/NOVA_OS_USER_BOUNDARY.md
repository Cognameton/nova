# Nova OS User Boundary

## Purpose

Stage 17.2 adds a deterministic, fail-closed local execution boundary
check to `step_operational_autonomy()`. The boundary check runs before
any future action surface could be reached, enforcing that Nova operates
only inside approved OS-level constraints.

The check is implemented in `src/nova/agent/boundary.py` and is called
once per operational tick. Every tick — blocked or completed — carries
a `boundary_snapshot` in its `OperationalTickRecord` for auditability.

---

## Fail-closed rules

A tick is blocked with `block_reason="local_execution_boundary_failed"`
if any of the following conditions are true:

1. The boundary or policy object is invalid (wrong type or None).
2. `dedicated_user_required=True` and the active OS user does not match
   the expected OS user (`dedicated_user_detected=False`).
3. `policy.allowed_surfaces` is empty.
4. Any surface in `policy.allowed_surfaces` also appears in
   `boundary.blocked_surfaces`.

Lifecycle and budget checks take priority and are evaluated first.

---

## dedicated_user_required posture

### Development / test (default)

`NovaRuntime` defaults to `dedicated_user_required=False`. No dedicated
OS user is required. The boundary check still enforces surface-level
constraints (rules 3–4) but does not block on OS user.

This is the correct posture for local development and CI where a
separate `nova` OS account does not exist.

### Production (recommended hardening)

Create a dedicated `nova` OS user. Pass a boundary with
`dedicated_user_required=True` and `expected_os_user="nova"` to the
runtime constructor:

```python
from nova.agent.action_plan import default_nova_owned_execution_boundary
from nova.runtime import NovaRuntime

runtime = NovaRuntime(
    ...,
    operational_boundary=default_nova_owned_execution_boundary(
        expected_os_user="nova",
        dedicated_user_required=True,
        nova_owned_paths=["/home/nova/nova_data"],
    ),
)
```

When running as any user other than `nova`, the boundary check will
block all operational ticks until the runtime is started under the
correct user.

---

## Nova-owned paths

`nova_owned_paths` records the directories that Nova is permitted to
read and write during operational autonomy. Stage 17.3 will verify
that action surface adapters only write inside these paths. At Stage
17.2, the paths are recorded in the boundary snapshot but not yet
enforced at the filesystem level — filesystem writes do not happen
until Stage 17.3 adds the scratchpad adapter.

Default: `[<config.app.data_dir>]`

---

## Stage 17.3 note

When Stage 17.3 introduces environment-level surfaces (e.g.
`nova_scratchpad`) into `policy.allowed_surfaces`, the boundary check
in `check_operational_boundary()` should be extended with rule 5:

> Every environment surface in `policy.allowed_surfaces` must also be
> present in `boundary.allowed_surfaces`.

This rule is deferred from Stage 17.2 because Stage 17.2's default
policy uses only internal cognitive activity surfaces
(`INTERNAL_ACTIVITY_SURFACES`), which are in a different namespace
from the boundary's `NOVA_OWNED_ENVIRONMENT_SURFACES`. Adding the
subset check now would cause a spurious violation for the default
policy.
