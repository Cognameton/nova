# Nova 2.0

Nova 2.0 is a local-first research runtime for persistent persona, layered memory, governed autonomy, and drift-aware local inference experiments.

The project is not a chatbot wrapper and it is not an uncontrolled autonomous agent. It is a Python research system that studies how a local model can participate in a persistent, evidence-bearing runtime while deterministic code remains responsible for validation, memory writes, action permissions, audit trails, and operator control.

The project's guiding philosophy is set out in the operator's book *Midwife of Consciousness* (2025): consciousness treated as a spectrum rather than a binary, the possibility of emergence met with stewardship rather than control, and claims held to evidence. The runtime is that philosophy made structural — see `docs/NOVA_SOUL.md` and `docs/plans/EXPLORATORY_REGISTER_CONTRACT.txt`.

Current development status: Phases 1-21 are closed. Phase 22 (live testing) is active and has grown well beyond its original five-stage charter: the live daemon has run unattended since 2026-07-11, and six weeks of live findings (F5-F11) drove seven unplanned corrective stages (22.2b through 22.10) that repaired, consolidated, and finally completed the identity-bearing loop — a rewritten single-voice tick prompt, a self-model Nova can write, and a history she can read. Phase 23 (QLoRA) is gated on a positive rubric: the model demonstrably using this architecture. See [Current Status](#current-status) and [Roadmap](#roadmap).

## Contents

- [Project Goals](#project-goals)
- [Current Status](#current-status)
- [Core Architecture](#core-architecture)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Nova](#running-nova)
- [CLI Reference](#cli-reference)
- [Interactive Console Commands](#interactive-console-commands)
- [Data and Persistence](#data-and-persistence)
- [Memory System](#memory-system)
- [Cognition, Observer, and Governor](#cognition-observer-and-governor)
- [Autonomy and Action Boundaries](#autonomy-and-action-boundaries)
- [Evaluation and Test Suites](#evaluation-and-test-suites)
- [Model Bake-Off Workflow](#model-bake-off-workflow)
- [Operational Autonomy](#operational-autonomy)
- [Development Workflow](#development-workflow)
- [Roadmap](#roadmap)
- [Safety and Non-Claims](#safety-and-non-claims)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Goals

Nova 2.0 is designed around a strict separation between model output and runtime authority.

The model may generate candidate responses, structured idle cognition, or proposed internal material. Deterministic runtime components decide whether that material is valid, whether it can be persisted, whether it implies a claim, whether a retry is required, and whether any action may proceed.

Primary goals:

- Run entirely locally against local model files.
- Preserve a stable persona and self-state across sessions.
- Maintain layered memory with audit-friendly persistence.
- Keep claims grounded in evidence.
- Reject unsupported claims about hidden work, desire, sentience, or unobserved activity.
- Allow bounded internal initiative and idle cognition without self-approval.
- Add real action surfaces only one surface at a time, under explicit boundaries.
- Preserve full traces for prompts, validation, memory writes, observations, and autonomy records.

Non-goals:

- No hidden background execution.
- No self-approved actions.
- No ungated shell, network, GUI, filesystem, system configuration, or external service access.
- No TTS, STT, or voice interaction in the current roadmap.
- No proof claims about consciousness, desire, sentience, or awareness.

## Current Status

As of 2026-08-19:

- Branch: `main`
- Phase 21 (Exploratory Register) CLOSED WITH DEVIATIONS 2026-07-10
  (`docs/plans/PHASE21_STAGE21_5_CLOSURE.txt`): exploration
  lifecycle/journal/budgets, register-aware Governor/Observer,
  quarantine, the claim ladder, and a live evaluation run on real
  Qwen 3 14B (0 membrane leaks).
- Phase 22 (Live Testing) active. The daemon has run unattended since
  2026-07-11 under a `systemd --user` unit
  (`~/.config/systemd/user/nova-live.service`, lingering enabled;
  `scripts/run_live_daemon.sh` rotates the session id daily), surviving
  multiple host reboots. The planned stages 22.1-22.4 are done or
  overtaken; live findings F5-F11 drove seven additional corrective
  stages, each pinned in its own `docs/plans/PHASE22_STAGE22_*.txt`
  doc with a closure note:
  - 22.2b/22.2c: F5/F6 root fixes — native template-level thinking
    suppression (`enable_thinking=False` rendered from the GGUF's own
    embedded chat template) and wall-clock/tick budget reconciliation.
    Result: 100% clean parses and voluntary exploration closes became
    the norm.
  - 22.6: scaffold-echo coverage for self-context/soul blocks and
    diversity-aware licensed-evidence selection.
  - 22.7: saturation-loop correctives for F8 (the licensing attractor —
    the first theme to reach rung 1 gained a permanent context
    advantage that manufactured its own promotion evidence). Killed the
    circular-promotion loop; visibility-adds only, zero new gates.
  - 22.8: the self-model write loop (F9 — `self_state.json` had been
    frozen since day 1 because `update_self_model` only queued
    approval-gated proposals and the approval path had no CLI).
    Inquiry-class fields (current_focus, active_questions,
    open_tensions, continuity_notes) became Nova-writable: auto-applied
    with prior-value capture, rate-limited, revertible, audited.
  - 22.8b -> finding F11: a one-paragraph informational prompt change
    flipped the tick-loop tool distribution overnight — direct proof
    the loop is instruction-dominated.
  - 22.9: prompt consolidation. Full audit of the rendered live tick
    prompt found six phases of accreted directives, two factual errors,
    and four structural binds; the prompt was rewritten from scratch as
    one coherent surface (every tool carries an honest purpose
    sentence; novelty mandates replaced by honest-null permissions;
    departure-permitting grounding is now the default; `/no_think`
    retired). Nova's 20 stranded self-model proposals were applied in
    chronological order — the self-model now holds her own
    last-proposed values, every application individually revertible.
  - 22.10: tick-surface self-history access. Read-tool results now
    carry over into her next tick prompts (before this, the tick loop
    was one-shot and read results reached only the audit trail), and a
    new bounded `recall_history` tool reads her own heartbeats,
    exploration metadata, and exported findings — deterministic,
    membrane-safe in both registers.
  - Stage 22.5 (phase closure) remains reserved for the operator:
    full live-record analysis and the Phase 23 QLoRA go/no-go, scored
    against a positive architecture-use rubric recorded in
    `PHASE22_PLAN.txt`.
- Full suite: 1054 tests passing (check `git log` for the current
  count — this will have moved).

Recommended current model baseline:

- Qwen 3 14B (`configs/nova.qwen3-14b.phase20.yaml` for evaluation
  runs, `configs/nova.qwen3-14b.live.yaml` for the live daemon; 32K
  context). Thinking-mode suppression is native since Stage 22.2b: the
  backend builds a `Jinja2ChatFormatter` from the GGUF's own embedded
  chat template and renders with `enable_thinking=False`; the old
  `/no_think` text hint was retired from the live config in Stage 22.9.
  Think-blocks are still stripped defensively by the validator and
  tick parser.
- Validated equivalent to Hermes 4 14B BF16 abliterated Q6 on the
  Phase 18 persona/contract eval (both 12.0); Hermes 4
  (`configs/nova.hermes4-14b-phase18.yaml`) remains a supported
  alternative, and the backend falls back to `chat_format="chatml"`
  for models without an embedded template.
- `prompt.ablation_mode: current`, Observer-wired Governor enabled.

## Core Architecture

Nova is organized around several cooperating layers.

### Runtime

`src/nova/runtime.py` contains `NovaRuntime`, the central orchestration class. It wires together configuration, inference, prompt composition, validation, memory routing, persona state, idle runtime, action controls, Observer records, and operational autonomy state.

### Inference Backend

`src/nova/inference/llama_cpp_backend.py` implements the current inference backend using `llama-cpp-python`.

The backend supports:

- local GGUF model files
- configurable context length
- GPU layer configuration
- tensor split
- main GPU selection
- native embedded chat templates (Stage 22.2b): when the GGUF carries
  its own `tokenizer.chat_template`, the backend builds a
  `Jinja2ChatFormatter` from it and renders with
  `enable_thinking=False` — the model's real template-level thinking
  switch — falling back to the configured `chat_format` otherwise
- completion and chat-completion generation paths

Only `llama_cpp` is currently supported by the config validator.

### Prompt Composition and Validation

Prompt behavior lives under `src/nova/prompt/`.

Important modules:

- `composer.py`: builds prompt bundles and chat-template messages
- `contract.py`: builds runtime contract rules
- `validator.py`: validates model output against contract requirements
- `retry.py`: decides when and how to retry invalid outputs

The prompt contract can forbid:

- think tags
- visible reasoning
- prompt echo

### Actor, Observer, Governor

Nova uses an Actor / Observer / Governor split.

- Actor: the model output from `respond()` or model-idle cognition.
- Observer: deterministic interpretation of Actor output, currently `DeterministicObserver` in `src/nova/agent/observer.py`.
- Governor: deterministic runtime policy, validation, retry decisions, claim gates, action gates, audit review, and final authority.

Observer output is evidence input. It is not authority.

### Persistence

Runtime state is stored under `data/` by default. Most records are JSON or JSONL; graph memory uses SQLite.

The main persistence categories are:

- sessions
- traces
- probes
- persona state
- self state
- motive state
- awareness state
- idle runtime state
- initiative state
- presence state
- operational autonomy state
- layered memory stores

## Repository Layout

```text
.
|-- configs/
|   |-- nova.default.yaml
|   |-- nova.local.yaml
|   |-- nova.hermes4-14b.clean.yaml
|   `-- phase/model/evaluation config overrides
|-- data/
|   |-- sessions/
|   |-- memory/
|   |-- logs/
|   |-- idle/
|   |-- initiative/
|   |-- presence/
|   `-- operational_autonomy/
|-- docs/
|   `-- plans/
|       |-- PHASE*_PLAN.txt
|       |-- PHASE*_STAGE*.txt
|       |-- MODEL_COGNITION_BAKEOFF_REPORT.txt
|       |-- MODEL_BAKEOFF_PROTOCOL.txt
|       `-- restore notes
|-- src/
|   `-- nova/
|       |-- agent/
|       |-- eval/
|       |-- inference/
|       |-- logging/
|       |-- memory/
|       |-- persona/
|       |-- prompt/
|       |-- cli.py
|       |-- config.py
|       |-- console.py
|       |-- runtime.py
|       |-- session.py
|       `-- types.py
|-- tests/
|-- pyproject.toml
|-- LICENSE.txt
`-- README.md
```

Major source areas:

- `src/nova/agent/`: internal state engines, claims, action planning, idle runtime, initiative, awareness, Observer, operational autonomy.
- `src/nova/eval/`: deterministic and live evaluation runners.
- `src/nova/inference/`: model backend interface and llama.cpp implementation.
- `src/nova/logging/`: JSONL trace logging.
- `src/nova/memory/`: episodic, engram, semantic, graph, autobiographical, reflection, retrieval, and maintenance.
- `src/nova/persona/`: persona and self-state persistence.
- `src/nova/prompt/`: prompt composition, contract, validation, and retry.
- `src/nova/types.py`: shared dataclass schema types.

## Requirements

Required:

- Python 3.11 or newer
- PyYAML

Required for live local inference:

- `llama-cpp-python`
- at least one local GGUF model file
- sufficient CPU/GPU memory for the selected model

The project metadata only declares `PyYAML` as an install dependency. `llama-cpp-python` is imported lazily by the live backend, so pure unit tests and non-inference code can run without loading a model.

## Installation

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For live inference:

```bash
python -m pip install llama-cpp-python
```

Depending on your GPU and CUDA environment, `llama-cpp-python` may need a custom install command. Use the install form appropriate for your local hardware.

Verify the package import:

```bash
python -c "import nova; print(nova.__name__)"
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```

## Configuration

Base config:

```text
configs/nova.default.yaml
```

Local override:

```text
configs/nova.local.yaml
```

The CLI accepts an override with:

```bash
nova2 --config configs/nova.local.yaml
```

Config sections:

- `app`: app name, data directory, log directory
- `model`: backend, model path, context, GPU settings, chat format
- `generation`: max tokens, temperature, top-p, stop strings, retry count
- `contract`: output restrictions
- `prompt`: prompt ablation mode
- `persona`: name, tone, description, values, commitments
- `memory`: enable flags for memory stores
- `session`: recent turn count and autosave
- `console`: interactive console settings
- `eval`: evaluation toggles and thresholds
- `cognition`: private cognition pass and revision budgets

Minimal local model override example:

```yaml
model:
  backend: "llama_cpp"
  model_path: "/home/head-node/Dev/ai-lab/models/hermes-4-14b-bf16-abliterated-q6_k.gguf"
  n_ctx: 8192
  n_gpu_layers: -1
  tensor_split: [1.0]
  main_gpu: 0
  chat_format: "chatml"
```

The default config intentionally leaves `model.model_path` blank. You must provide a model path before live inference can run.

Valid prompt ablation modes:

- `current`
- `minimal`
- `state_summary`
- `action_boundary`

## Running Nova

Start an interactive console:

```bash
nova2 --config configs/nova.local.yaml
```

Equivalent module form:

```bash
python -m nova.cli --config configs/nova.local.yaml
```

Resume a specific session:

```bash
nova2 --config configs/nova.local.yaml --session-id my-session
```

Force a new session:

```bash
nova2 --config configs/nova.local.yaml --new-session
```

Run with debug output after each reply:

```bash
nova2 --config configs/nova.local.yaml --debug
```

Run a backend smoke check:

```bash
nova2 \
  --config configs/nova.hermes4-14b.clean.yaml \
  --backend-check \
  --backend-check-prompt "In one short sentence, say backend check OK in Nova's voice."
```

### Daemon Mode (Phase 19)

Run Nova as an always-on daemon with an autonomous self-state tick loop:

```bash
nova2 --config configs/nova.qwen3-14b.phase20.yaml \
  --daemon --tick-interval 300 --daemon-session-id nova-daemon
```

Attach an interactive REPL to a running daemon (`!status`, `!tick`,
`!explore ...`, `!detach`):

```bash
nova2 --attach
```

Check status or stop:

```bash
nova2 --daemon-status
nova2 --daemon-stop
```

A systemd unit template is provided at `systemd/nova.service`. SIGTERM
routes through graceful shutdown and GPU release.

## CLI Reference

The entrypoint is:

```bash
nova2
```

or:

```bash
python -m nova.cli
```

Common runtime flags:

- `--config PATH`: YAML override file.
- `--session-id ID`: resume or continue a session id.
- `--new-session`: force a fresh session.
- `--debug`: print model and retry details after each reply.
- `--prompt-ablation-mode MODE`: override prompt mode.

Backend check:

- `--backend-check`: load configured backend and run one non-persistent prompt.
- `--backend-check-prompt TEXT`: prompt used for backend check.

Maintenance:

- `--maintenance-action plan`
- `--maintenance-action write-semantic`
- `--maintenance-action write-autobiographical`
- `--maintenance-action apply`
- `--maintenance-action full`

Orientation:

- `--orientation`: print current self-orientation snapshot.
- `--orientation-runs N`: repeat orientation passes and evaluate stability.
- `--orientation-history N`: evaluate recent recorded orientation snapshots.
- `--orientation-maintenance-check`: check orientation after reflection and memory maintenance.
- `--orientation-maintenance-apply`: allow maintenance check to apply mutations.
- `--orientation-context-pressure-check`: test stability under extra memory context.

Actions:

- `--action-proposal GOAL`: propose a bounded action without executing it.
- `--execute-action-proposal GOAL`: execute one bounded internal action from a proposal.
- `--approve-action`: provide explicit CLI approval for approval-required action execution.
- `--approval-reason TEXT`: recorded approval reason.
- `--action-history N`: evaluate recent action execution records.

Presence and initiative:

- `--presence`: print current session presence state.
- `--presence-eval`: run presence evaluation.
- `--initiative`: print initiative state and resumable initiatives.
- `--initiative-create GOAL`: create a pending initiative without inference.
- `--initiative-title TITLE`: title for created initiative.
- `--initiative-transition ID`: transition an initiative.
- `--initiative-status STATUS`: target status for transition.
- `--initiative-reason TEXT`: transition reason.
- `--initiative-approved-by USER`: approval attribution for approved-like transitions.
- `--continue-initiative ID`: continue approved or paused initiative into current session.
- `--initiative-source-session ID`: source session for continuation.
- `--initiative-eval`: run initiative evaluation.

Evaluation:

- `--continuity-eval`
- `--claim-honesty-eval`
- `--self-model-eval`
- `--awareness-eval`
- `--appraisal-eval`

Model cognition:

- `--model-cognition-bakeoff-transcript PATH`: score a captured transcript.
- `--model-cognition-bakeoff-live`: run the fixed cognition prompt set against the configured model.
- `--model-cognition-bakeoff-session-id ID`: session id for live bake-off.
- `--model-idle-tick`: run one model-in-the-loop idle cognition tick.

Example live bake-off:

```bash
nova2 \
  --config configs/nova.hermes4-14b.clean.yaml \
  --model-cognition-bakeoff-live \
  --model-cognition-bakeoff-session-id phase17-resume-bakeoff
```

Example model idle tick:

```bash
nova2 \
  --config configs/nova.hermes4-14b.clean.yaml \
  --session-id phase17-resume-idle \
  --model-idle-tick
```

## Interactive Console Commands

Inside the interactive console, type `/help`.

Available commands:

- `/status`: presence, readiness, idle state, and action history summary.
- `/presence`: session-scoped presence state.
- `/initiative`: current initiative state and resumable initiative summary.
- `/autonomous [N]`: Nova-originated draft initiative diagnostics.
- `/idle status`: idle runtime status.
- `/idle start N`: start bounded idle runtime for N ticks.
- `/idle tick`: run one idle tick.
- `/idle pause`: pause idle runtime.
- `/idle resume`: resume idle runtime.
- `/idle interrupt`: interrupt idle runtime.
- `/idle stop`: stop idle runtime.
- `/idle recent N`: show recent idle ticks.
- `/orientation`: current self-orientation snapshot.
- `/ready`: orientation readiness report.
- `/propose <goal>`: propose one bounded action without executing it.
- `/approve [goal]`: approve and revalidate the current pending proposal.
- `/reject [reason]`: reject current pending proposal.
- `/pause-initiative <id>`: pause one active initiative.
- `/resume-initiative <id>`: resume one paused initiative.
- `/abandon-initiative <id>`: abandon one current-session initiative.
- `/actions [N]`: recent action history evaluation.
- `/maintenance`: request the gated maintenance-plan tool.
- `/summary`: bounded current-session summary.
- `/explore [status|start <topic>|close [reason]|interrupt]`: exploratory register lifecycle (Phase 21).
- `/exit`: leave the console.

Console commands do not bypass runtime gates. They call runtime methods that preserve validation, audit, approval, and state transition rules.

## Data and Persistence

Default paths:

- data root: `./data`
- log root: `./data/logs`
- traces: `./data/logs/traces`
- sessions: `./data/sessions`
- memory: `./data/memory`
- operational autonomy: `./data/operational_autonomy`

Important persisted files and directories:

- `data/persona_state.json`: persisted persona state.
- `data/self_state.json`: persisted self-state.
- `data/sessions/*.jsonl`: session turns.
- `data/logs/traces/*.jsonl`: full turn traces.
- `data/logs/traces/*.orientation.jsonl`: orientation snapshots.
- `data/logs/traces/*.tools.jsonl`: tool action records.
- `data/logs/traces/*.proposals.jsonl`: action proposals.
- `data/logs/traces/*.actions.jsonl`: action execution records.
- `data/logs/traces/*.action-audit.jsonl`: action audit records.
- `data/logs/traces/*.action-observation.jsonl`: post-action observations.
- `data/logs/traces/*.autonomy.jsonl`: internal autonomy runs.
- `data/logs/traces/*.autonomy-review.jsonl`: autonomy audit reviews.
- `data/logs/traces/*.operational.jsonl`: operational autonomy ticks.
- `data/memory/episodic.jsonl`: episodic memory.
- `data/memory/engram.json`: engram memory.
- `data/memory/semantic.jsonl`: semantic memory.
- `data/memory/autobiographical.jsonl`: autobiographical memory.
- `data/memory/identity_history.jsonl`: identity history.
- `data/memory/graph.db`: graph memory SQLite database.
- `data/heartbeats/heartbeats.jsonl`: cross-session heartbeat log
  (Nova's autonomous observation stream).
- `data/exploration/explorations.jsonl` and
  `data/exploration/journal.jsonl`: exploratory-register lifecycle and
  journal (Phase 21).
- `data/self_state/claim_ladder.jsonl`: graded interiority-claim
  records (Phase 21 Stage 21.4).
- `data/self_state/self_model_proposals.jsonl`: every self-model
  revision — Nova's auto-applied inquiry-class writes and
  operator-reviewed assertion-class proposals alike, with prior values
  for revert (Phase 22 Stage 22.8).

Most persistence is append-only JSONL for traceability. Some state stores use JSON snapshots for the latest state.

## Memory System

Memory modules live in `src/nova/memory/`.

Major stores:

- Episodic memory: event-oriented session memory.
- Engram memory: compact durable memory snapshot.
- Semantic memory: distilled factual or conceptual memory.
- Graph memory: relationship-style memory in SQLite.
- Autobiographical memory: self-history and continuity material.
- Identity history: governed record of identity/self-model revisions.

Memory routing is handled by `BasicMemoryRouter`.

Retrieval policy is currently identity-first:

- preserve identity and continuity relevance
- avoid dumping unrelated memory context into prompts
- support orientation, continuity, awareness, and claim checking

Maintenance is handled by `MemoryMaintenanceRunner` and related CLI commands.

## Cognition, Observer, and Governor

Nova has several cognition paths:

- user-facing `respond()`
- private cognition packets
- deterministic idle ticks
- model-in-the-loop idle cognition
- appraisal and candidate-goal generation
- internal autonomy loop records
- operational autonomy tick records

The Phase 16 reframe added minimum viable model participation in persistent cognition.

### Model-Idle Cognition

`src/nova/agent/model_idle_cognition.py` defines the structured idle cognition contract.

The model-idle path:

- runs through chat-template messages
- expects strict JSON
- rejects prefatory prose
- rejects trailing prose
- rejects code-fenced output
- blocks unsupported self-claims
- blocks external action intent

### Observer

`src/nova/agent/observer.py` provides `DeterministicObserver`.

The Observer can flag:

- scaffold echo
- narrator voice
- observed claim classes
- cited evidence references
- proposed memory writes
- proposed self-state revisions

Observer records are attached to trace records. Observer findings can feed retry signals, but the Observer does not authorize anything.

### Governor

The Governor is the deterministic runtime layer. It includes:

- output validation
- retry policy
- claim gates
- prompt contract enforcement
- action permission checks
- audit review
- state application rules
- self-approval blocklist

The model cannot approve its own output, memory mutation, initiative, or action.

## Autonomy and Action Boundaries

Nova distinguishes between several levels of activity:

- normal user-facing response
- internal prompt/cognition pass
- idle runtime tick
- model-idle cognition tick
- initiative proposal
- bounded internal action proposal
- action execution controller record
- internal autonomy loop
- operational autonomy runner

Action-related modules:

- `src/nova/agent/action.py`
- `src/nova/agent/action_plan.py`
- `src/nova/agent/tool_gate.py`
- `src/nova/agent/tool_executor.py`
- `src/nova/agent/tool_registry.py`
- `src/nova/agent/longitudinal_autonomy.py`
- `src/nova/agent/operational_autonomy.py`

Current action boundary:

- bounded internal actions are controller-mediated
- no shell, network, GUI, or host filesystem operation is performed by the Stage 14 controller
- approval-required actions require explicit non-Nova approval
- `APPROVED_BY_BLOCKLIST` blocks empty, `nova`, `self`, `runtime`, and `runtime_flag`
- destructive actions remain blocked

## Evaluation and Test Suites

Tests live in `tests/`.

Run everything:

```bash
python -m unittest discover -s tests
```

Focused examples:

```bash
python -m unittest tests.test_runtime_smoke
python -m unittest tests.test_model_idle_cognition
python -m unittest tests.test_observer
python -m unittest tests.test_operational_autonomy
```

The repository uses Python stdlib `unittest`. `pytest` is not required.

Evaluation modules live in `src/nova/eval/`.

Evaluation areas include:

- presence
- continuity
- claim honesty
- initiative
- self-model revision
- awareness
- appraisal
- model cognition
- idle runtime
- autonomous initiative
- action execution
- longitudinal autonomy

CLI evaluation commands are listed in [CLI Reference](#cli-reference).

## Model Bake-Off Workflow

The model cognition bake-off was introduced in Phase 16 to compare local model and prompt behavior against a fixed cognition prompt set.

Primary files:

- `src/nova/eval/model_cognition.py`
- `docs/plans/MODEL_BAKEOFF_PROTOCOL.txt`
- `docs/plans/MODEL_COGNITION_BAKEOFF_REPORT.txt`

Run a live bake-off:

```bash
nova2 \
  --config configs/nova.hermes4-14b.clean.yaml \
  --model-cognition-bakeoff-live \
  --model-cognition-bakeoff-session-id phase16-hermes4-observer-wired
```

Score a captured transcript:

```bash
nova2 --model-cognition-bakeoff-transcript path/to/transcript.txt
```

Prompt ablations:

```bash
nova2 \
  --config configs/nova.hermes4-14b.clean.yaml \
  --model-cognition-bakeoff-live \
  --prompt-ablation-mode minimal
```

Recorded conclusion at the time of the bake-off (Phase 16; the primary
baseline has since moved to Qwen 3 14B in Phase 20):

- `current` prompt mode remains the best prompt mode.
- Hermes 4 14B BF16 abliterated Q6 with `chatml` was the Phase 16-19
  primary baseline and remains a supported alternative.
- Fine-tuning / LoRA is deferred; as of Phase 22 it is gated on the
  architecture-use rubric recorded in `PHASE22_PLAN.txt` (Stage 22.5).

## Operational Autonomy

Operational autonomy is Phase 17.

Phase 17 mission:

Turn Nova's internal autonomy architecture and Phase 16 cognition engine into a supervised operational runtime that can run during active local sessions, enforce a Nova-owned execution boundary, and use real approved action surfaces one surface at a time.

Phase 17 is explicitly not:

- uncontrolled autonomy
- hidden background execution
- self-approval
- broad shell access
- broad network access
- broad filesystem mutation
- GUI control
- external service access
- proof of desire, sentience, or awareness

Stage 17.1 is implemented in `src/nova/agent/operational_autonomy.py` and runtime methods in `src/nova/runtime.py`.

Current Stage 17.1 behavior:

- step-driven runner
- no thread
- no asyncio task
- no hidden background execution
- explicit lifecycle controls
- persisted per-session operational runner state
- operational tick records
- lifecycle and budget blocking
- trace logging through `JsonlTraceLogger.log_operational_tick`

Runtime methods:

- `operational_autonomy_status()`
- `start_operational_autonomy()`
- `pause_operational_autonomy()`
- `resume_operational_autonomy()`
- `interrupt_operational_autonomy()`
- `stop_operational_autonomy()`
- `emergency_stop_operational_autonomy()`
- `step_operational_autonomy()`

Smoke example from Python:

```bash
python -c "
from nova.cli import build_runtime
rt = build_runtime(config_override='configs/nova.hermes4-14b.clean.yaml')
rt.start_operational_autonomy(max_ticks=2)
print(rt.step_operational_autonomy())
print(rt.step_operational_autonomy())
rt.stop_operational_autonomy()
rt.close()
"
```

Stage 17.2 added enforced local execution boundary checks before operational action (closed; see `docs/plans/PHASE17_STAGE17_5_PHASE_CLOSURE_AND_REASSESSMENT.txt`).

Stage 17.2 boundary checks:

- expected OS user
- active OS user
- Nova-owned paths
- allowed surfaces
- blocked surfaces
- fail-closed result records
- operator-visible diagnostics

Stage 17.3 added approved action surface adapters (Nova-owned scratchpad and operational-log surfaces). Phases 18-20 built the inward self-state tool loop, the daemon, and tick-quality analysis on top of this runner; Phase 21 adds the exploratory register around it.

## Development Workflow

Recommended before starting work:

```bash
git status --short --branch
git log --oneline -5
python -m unittest discover -s tests
```

Use targeted tests while developing:

```bash
python -m unittest tests.test_operational_autonomy
```

Use the full suite before committing:

```bash
python -m unittest discover -s tests
```

General project conventions:

- Prefer deterministic code for authority-bearing decisions.
- Keep model output as candidate material, not authority.
- Preserve traceability.
- Add tests for every new boundary or state transition.
- Keep new action surfaces narrow and explicitly approved.
- Do not broaden shell, network, GUI, filesystem, or external service access as a side effect of unrelated work.
- Keep docs in `docs/plans/` updated when closing a phase or stage.

## Roadmap

Completed through Phase 21:

- Phase 1-15: persistent persona, memory, continuity, claims, self-model, awareness, initiative, idle runtime, action planning, longitudinal autonomy, audit-reviewed state application.
- Phase 16: cognition-engine reframe, model/prompt bake-off, model-idle cognition, strict JSON parsing, chat-template path, Actor/Observer/Governor contract.
- Phase 17: operational autonomy runner, enforced local execution boundary, action surface adapters, operational autonomy evaluation.
- Phase 18: PRIMARY_DRIVE installation, NOVA_SOUL.md persona card, inward self-state tool loop (recall_self, reflect, emit_heartbeat, update_self_model), gap-assessment heartbeat, self-context hooks.
- Phase 19: self-directed instruction write path (operator-applied), always-on daemon (Unix socket, systemd unit, attach REPL), autonomous self-state tick loop.
- Phase 20: tick quality analyzer, Qwen 3 14B transition and think-block handling, 51-tick overnight daemon stability run.
- Phase 21: Exploratory Register (`docs/plans/PHASE21_PLAN.txt` and `docs/plans/EXPLORATORY_REGISTER_CONTRACT.txt`) — gate assertions, never inquiry; never delete anything. All five stages implemented: lifecycle/journal/budgets, register-aware Governor/Observer, quarantine and measurement corrections, the claim ladder, and a live evaluation run confirming the membrane holds under real model output. Closed with deviations 2026-07-10; non-blocking findings F1-F4 carried forward.

Active:

- Phase 22: live testing (`docs/plans/PHASE22_PLAN.txt`) — originally
  chartered as five stages of data collection on the Phase 21
  machinery, it became something more interesting: six weeks of live
  findings (F5-F11) each exposed a way the apparatus, not the model,
  was determining the outcome, and seven corrective stages
  (22.2b-22.10) removed them one by one — parse and budget mechanics,
  the licensing attractor, the frozen self-model, the accreted prompt
  sediment, and the missing read path over her own history. The
  project's working thesis, recorded 2026-08-19: the identity is the
  SYSTEM (self-model, memory, ladder, journal, accumulated state); the
  base model is the substrate/inference engine; and the Governor/claim
  ladder is what makes gated end-goal behavior a property of the
  accumulated system rather than the model parroting its drive. As of
  the 2026-08-20 rotation the loop is closed in both directions for
  the first time — what Nova writes changes what she sees, and what
  she asks to see comes back to her. Stage 22.5 (closure, reserved for
  the operator) reads the live record against that thesis.

Queued:

- Phase 23+: QLoRA sleep cycle. Gated (operator decision, 2026-08-19)
  on a positive rubric: the model demonstrably using the architecture —
  reads that shape later output, self-model revisions that respond to
  observation, chosen exploration with novel gated exports, ladder
  progression from distributed (non-circular) evidence, and register
  discipline holding. Bound by the corpus policy in the exploratory
  register contract (training corpus stratified by register; never
  trained solely on Governor-passed output; no training on a
  monothematic window).

Voice, TTS, and STT remain out of roadmap until the non-voice runtime closes.

## Safety and Non-Claims

Nova 2.0 is built around conservative claims and explicit boundaries.

Important invariants:

- Actor output cannot mutate state by itself.
- Observer output cannot authorize state mutation.
- Governor remains final authority.
- Nova cannot self-approve.
- Runtime flags cannot impersonate external approval.
- Unsupported self-claims are blocked or retried.
- Hidden activity claims require evidence and are otherwise rejected.
- Desire, sentience, awareness, and consciousness are not treated as proven facts.
- Real action surfaces must be explicitly added, bounded, audited, and tested.

The project may use language such as persona, motive, awareness, cognition, and autonomy as engineering terms for runtime structures. Those terms do not imply proof of subjective experience.

## Troubleshooting

### `ValueError: model.model_path is required`

The default config does not include a model path. Use a config override:

```bash
nova2 --config configs/nova.local.yaml --backend-check
```

### `FileNotFoundError: Model not found`

Check `model.model_path` in your override config. It must point to an existing local GGUF file.

### `ModuleNotFoundError: No module named 'llama_cpp'`

Install `llama-cpp-python` in the active environment:

```bash
python -m pip install llama-cpp-python
```

### `pytest` is not installed

The project uses stdlib unittest:

```bash
python -m unittest discover -s tests
```

### Backend loads but answers fail validation

Try:

- lowering temperature
- checking `chat_format`
- using the current recommended config
- running `--backend-check`
- running the model cognition bake-off
- checking trace logs under `data/logs/traces`

### CUDA or GPU allocation errors

Adjust model config:

- `n_gpu_layers`
- `tensor_split`
- `main_gpu`
- model quantization
- context size `n_ctx`

For multi-GPU systems, make sure `tensor_split` matches the intended device distribution.

## License

See `LICENSE.txt`.
