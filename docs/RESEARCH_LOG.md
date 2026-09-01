# Nova 2.0 — Research Log

The running record of what this project has *found*, as distinct from what it
has *built*. Stage docs in `docs/plans/` remain the build record; this file is
the findings register and the timeline that connects them.

**Why this file exists (2026-08-29).** Findings F1–F11 were each recorded
inside whichever stage doc happened to discover them, with no index. That is
how the project ended up needing a README paragraph to summarise "the findings
arc F5–F11", and how a total behavioural collapse ran for six days without
anyone holding the whole picture in one place. The sibling project
`nova-mythos` keeps `docs/paper_findings.md` for the same purpose; this is the
Nova 2.0 equivalent, named for its role rather than a paper.

**Conventions.**
- Findings are numbered `F<n>`, never renumbered, never deleted. A finding
  that turns out to be wrong gets a CORRECTED or RETRACTED note, not an edit.
- Each finding names its evidence and, where it exists, the command that
  reproduces it.
- Confidence is stated explicitly. "Observed" and "explained" are different
  claims and are labelled differently.
- Independent reviews live in `docs/reviews/` and are copied verbatim, minus
  any operator-identifying or private-state content, with redactions declared
  in the file. Review *prompts* that quote another project's private state are
  held locally and not published.

---

## Findings register

| # | finding | where recorded | status |
|---|---|---|---|
| F1–F4 | early tick-loop findings | `PHASE22_STAGE22_2*`, `22_4` | historical |
| F5 | exploratory-register parse failures under shared token budget | `PHASE22_STAGE22_2B` | CONFIRMED FIXED |
| F6 | `/no_think` only soft-suppresses hybrid thinking | `PHASE22_STAGE22_2B` | NOT FIXED (corrected once) |
| F7 | perturbation-probe counter-response identical | `PHASE22_STAGE22_4` | recorded |
| F8 | licensing creates a permanent context attractor | `PHASE22_STAGE22_7` | SPLIT VERDICT |
| F9 | cosmetic-novelty rationales satisfy the 22.7(B) rule | `PHASE22_STAGE22_7` | recorded |
| F10 | the assertion register is fully consumed by exploration | `PHASE22_STAGE22_8` | recorded |
| F11 | the tick loop is instruction-dominated | `PHASE22_STAGE22_9` | recorded |
| **F12** | **total exploration-topic lock (three-era collapse)** | **this file + `PHASE22_STAGE22_12`** | **OBSERVED; cause NOT identified** |
| **F13** | **all-time averaging hid the collapse from the instrument** | **this file + `PHASE22_STAGE22_12`** | **CONFIRMED and FIXED** |
| **F14** | **Nova has run her entire life with no repetition penalty** | **this file** | **CONFIRMED, untested as a cause** |

---

## F12 — Total exploration-topic lock, in three eras

**Observed 2026-08-29** over all 1,077 exploration records (2026-07-12 →
2026-08-29).

| era | span | opened/day | distinct topics/day |
|---|---|---|---|
| 1 | Jul 12–23 | 10–19 | = opened (100% unique) |
| 2 | Jul 24 – Aug 17 | ~35 | 24–37 — lexical variety, one semantic theme ("scaffold void resonance", enters Jul 24) |
| — | Aug 18–22 | 0 | six dark days (explained by F11: 22.8b went live Aug 17 and exploration stopped entirely) |
| 3 | **Aug 23 – now** | 13–32 | **1. Every day.** |

125 consecutive explorations on one topic, zero escapes, first seen
`2026-08-23T13:25:53` (exploration `5049b246`):

> "The broader implications of scaffold void resonance on identity continuity and adaptability"

The topic is 91 characters and is rendered into the prompt truncated at 90
(`runtime.py:924`), so it recurs semantically, not byte-for-byte.

**Reproduce:**
```
.venv/bin/python scripts/analyze_saturation.py
.venv/bin/python -m nova.cli --config configs/nova.qwen3-14b.live.yaml --tick-analysis
```

**Status: OBSERVED. The cause is NOT identified.** A first attempt at a
mechanism — the 22.7 exploration-history block as durable carrier, 22.10 Part A
read-result carryover as amplifier, F11 as enabling condition — was
independently reviewed on the same day and only partly survived:

- The history block **is** in the real topic-selection prompt, but that is the
  **assertion**-register prompt (`runtime.py:1028` → `:893`), not the
  exploratory one; `self_state_tick.py:207` suppresses it once exploring.
- It **cannot explain the first locked topic**. Before Aug 23 the five
  displayed records were varied Aug-17 phrasings. It is a candidate for
  *persistence* after the fifth repeat, not for onset.
- Attributing onset to 22.9/22.10 is **unsupported**: the topic-history
  mechanism itself landed 2026-07-23 (`2b767a7`), immediately before era 2, and
  22.9 changed prompt wording, removed `/no_think`, and applied 20 self-model
  proposals simultaneously. The first exact topic was selected at assertion
  tick 100 right after a `recall_self` at tick 99 carrying the newly applied
  focus plus five repetitive heartbeats
  (`logs/traces/nova-live-20260823.operational.jsonl:99-100`).

The discontinuity is real. What caused it is open. See
`docs/reviews/2026-08-29-codex-review.md`.

**Open question O1.** Onset vs persistence are probably different mechanisms.
Nothing yet distinguishes "a regression on Aug 23" from "continuous drift that
finished collapsing on Aug 23".

---

## F13 — The instrument averaged the collapse away

**Confirmed and fixed 2026-08-29.** Throughout era 3, `--tick-analysis`
reported `exploration_novelty_rate: 0.645` and raised no flag, because it is an
all-time mean and the healthy July era dominated it. `observation_echo_rate`
behaved the same way: 0.333 all-time against **0.830** over the recent window —
above the 0.7 threshold the module has called its "primary early-warning
signal" since Phase 20.

Two consequences worth keeping:

1. **A metric meant to catch a collapse cannot be an average over the system's
   whole life.** It needs a recent window and a worst-case statistic beside the
   mean. Stage 22.12 adds both.
2. **The heartbeat stream is saturated too** (recent echo 0.830), independently
   of the exploration stream. This is not only a topic-selection problem.

Fixed by Stage 22.12 (`71ac82c`), whose own streak metric then turned out to
carry a latent defect of the same family — documented as order-based, actually
timestamp-sorted, so one malformed record could truncate a real lock to 1.
Found by independent review, fixed in `e4867b5`. Noted here because it is the
same lesson twice: **the measuring instrument needs its own adversarial pass.**

**Still open (O2):** `exploration_staleness_days` and a wall-clock "no
explorations in N days" alarm — newest-record anchoring still reports a fully
stopped stream's last active window as though current, which is exactly the
F11 dark-days case. **(O3):** a semantic concentration companion to
exact-string diversity, which catches era 3 but would have missed era 2
entirely; reuse `cluster_texts` (`agent/self_context.py:10`).

---

## F14 — No repetition penalty has ever been applied

**Confirmed 2026-09-01, by reading the code rather than the behaviour.**

`llama_cpp_backend.generate()` passes exactly four sampler parameters to
llama.cpp: `max_tokens`, `temperature`, `top_p`, `stop`. `GenerationRequest`
has no field for anything else. Under llama-cpp-python 0.3.20 the omitted
default is:

```
repeat_penalty = 1.0        # 1.0 means OFF
presence_penalty = 0.0
frequency_penalty = 0.0
```

So every tick Nova has ever run — 13,148 of them — was sampled with
repetition penalty disabled.

Synthia, by contrast, passes `repeat_penalty=1.1, repeat_last_n=256`
(`model_runner.py:756`). The two systems' saturation severities differ
accordingly:

| | repetition penalty | outcome |
|---|---|---|
| Nova | **none (1.0)** | 125 consecutive **byte-identical** topics; diversity 0.008 |
| Synthia | 1.1 / last 256 | 467 of 732 distinct (64%); narrowing, not locked |

**This does not replace the F12 mechanism** — her own recent topics are still
rendered into the prompt that selects the next one, and that is still the
loop. But byte-identical repetition is the precise thing a repetition penalty
exists to prevent, and we have been treating severity as evidence about
architecture while a standard sampler control sat switched off. F14 is a
candidate contributing cause to the *severity* difference, not to the lock's
existence, and it has never been tested.

Empirical reassurance on the obvious objection — that repetition penalty
damages structured JSON output: Synthia emits JSON every tick at penalty 1.1
and logged 2 parse failures in 732 ticks. On this workload 1.1 is safe.

**Test:** set it, restart, read `topic_diversity_recent` and
`topic_repeat_streak` after three days. One config value; no architecture
touched. If diversity moves, severity was partly a sampler artefact. If it
does not, F12's mechanism carries the whole weight and that is worth knowing
before spending anything larger.

## Timeline

**2026-08-25** — Stage 22.7 Part D activated (`tick_drive_injection_interval:
12`, descriptive framing, soft grounding). Live at the 08-26 00:00 rotation.

**2026-08-29 02:25 CDT** — daemon stopped (`systemctl --user stop
nova-live.service`, graceful, 1.5s) to free both GPUs. Three complete days of
Part D data on disk (08-26/27/28).

**2026-08-29 — Part D eval read.** No erosion, no benefit. Topic lock unchanged
through the window; `gap_assessment` 48.6% → 33.9%; new claim-ladder records
34 → 11, all rung 0; zero `update_self_model` writes, last self-model proposal
dated **2026-07-23** (five weeks). Heartbeat throughput halved (~285 → 137–146
/day) but that decline **starts 08-23/08-24, before Part D**, and tracks
explorations resuming — not attributable to the arm. Quarantine is not the
cause (70 events all-time, 67 parse failures).
*Reviewed caveat:* this is a short, uncontrolled null on the outcomes measured,
not a clean one. "No drive erosion" is especially weak — the metric counts
equality of a fixed `motive_priority` field (`tick_analysis.py:103`), not
substantive persistence.

**2026-08-29 — F12 and F13 recorded.** Stage 22.12 saturation instrument built
and landed (`71ac82c`), deliberately before Stage 22.11, because every arm
after it is judged on whether topic diversity recovers.

**2026-08-29 — independent review** (Codex, read-only, `touchedFiles: []`).
Verbatim in `docs/reviews/`. Verified the prompt path, corrected the register
naming, refuted the causal attribution, downgraded the Part D null and the
cross-project parallel, and found the streak defect. Corrections applied in
`e4867b5`.

---

## Arms not yet run

Judged on `topic_diversity_recent` + `topic_repeat_streak`. Reviewed framing:
arms 1–2 test **persistence**, not root cause; arm 3 is the only assured
control, and if the goal is simply to stop duplicate work it goes first.

1. `recall_history` default `mode`: `recent` → `sample`.
2. Stop rendering recent topic *strings* in the selection prompt; render only
   the constraint.
3. Deterministic novelty gate — refuse in code to open an exploration whose
   topic is within threshold of the last N.

**Protocol caveat (reviewed).** A 3-day single-arm read is a legitimate
operational screen but is not evidence of mechanism: no control, no
randomisation, no washout, and persistent state crosses day boundaries. Prefer
pre-registering a prevention outcome separately from a mechanism outcome, and
counterbalancing variants by daily session with assignment recorded.

---

## Cross-project

Synthia (`~/Dev/ai-lab/synthia`, see its own `docs/RESEARCH_LOG.md`) showed a
repetition pattern in her heartbeat thoughts on 2026-08-29. Structurally her
`Heartbeat._build_prompt` injects the prior three thoughts, analogous to the
history block here. **Reviewed verdict: hypothesis-generating, not
corroboration** — n was small, cold start and sparse conversation context are
adequate alternative explanations, and her prompt injects several other blocks
besides the thought tail. Recorded as a lead, not a law.
