# Independent review — Codex, 2026-08-29

**Verbatim, with two redactions noted below.** Copied from the Codex job record
(`task-mte4mdmf-8dcb2w`, thread `01a04ca9-c6c6-71f2-9ae7-d949cea36813`), which lived
only in a plugin state directory outside this repo. Preserved here so the
review survives a cache clear and can be checked against the claims it
corrects.

- Run: read-only (`write: false`), `touchedFiles: []`.
- Verified before and after: `nova2.0` and `synthia` HEADs and working trees
  unchanged.
- Prompt/handoff: held locally by operator decision, not published. It quoted
  four of Synthia's private heartbeat thoughts verbatim; those come from a
  gitignored self-state directory that has never been pushed anywhere, so
  publishing them here would have crossed a boundary the other project keeps
  deliberately. The review's reasoning stands without them.
- Corrections adopted: `docs/plans/PHASE22_STAGE22_12_SATURATION_INSTRUMENT.txt`
  (CORRECTIONS section) and findings F12/F13 in `docs/RESEARCH_LOG.md`.
- Defect it found in our code, since fixed: commit `e4867b5`.

**Redactions:** absolute `/home/...` paths normalised to repo-relative (the
citations are more useful that way regardless), and one occurrence of the
operator's first name replaced with "the operator". No claim, verdict, file
reference or line number was altered.

---

## Bottom line

The central code claim is substantially correct, with one literal caveat: recent exploration topics are rendered into the assertion-register prompt that selects `enter_exploration`, not into the subsequent exploratory-register prompt. The locked topic is also truncated to 90 characters, so the full 91-character string is not rendered byte-for-byte.

### A. Prompt path — verified

The runtime determines the register before generation; an active exploration yields `exploratory`, otherwise `assertion` ([runtime.py](src/nova/runtime.py:982)). A new topic can only be selected on the assertion tick because that menu includes `enter_exploration`; the exploratory menu instead has `close_exploration` ([self_state_tick.py](src/nova/agent/self_state_tick.py:75), [self_state_tick.py](src/nova/agent/self_state_tick.py:168)).

On every assertion tick, runtime calls `_exploration_history_block()` ([runtime.py](src/nova/runtime.py:1028)), which reads the latest 30 persistent records, emits the newest five, and emits the saturation note ([runtime.py](src/nova/runtime.py:893)). It passes that block to `build_messages` ([runtime.py](src/nova/runtime.py:1050)). The tick engine explicitly suppresses the block only when already in the exploratory register ([self_state_tick.py](src/nova/agent/self_state_tick.py:207)), then appends it to the user prompt ([self_state_tick.py](src/nova/agent/self_state_tick.py:232)).

So: **yes, the recent-topic history is in the real topic-selection prompt.** The claim should be reworded from “exploratory-register tick prompt” to “assertion-register entry-selection prompt.”

Literal caveat: topics are rendered as `record.topic[:90]` ([runtime.py](src/nova/runtime.py:924)). The locked topic is 91 characters, hence appears without its final `y`. This does not negate semantic priming, but it does refute any byte-identical prompt-string claim.

The history block cannot explain the *first* Aug. 23 locked topic: before it, the five displayed records were varied Aug. 17 scaffold-resonance phrasings. It becomes a very plausible persistence mechanism only after the fifth repeat, when all displayed items are the same 90-character prefix. The store is persistent (`list_recent` reads the JSONL store, [exploration.py](src/nova/agent/exploration.py:111)), so it survives rotation.

### B. Regression versus continuous drift

**Not identifiable as “22.9/22.10 caused it.”** The record supports a sharp behavioral discontinuity, but not a unique cause.

Verified timing:

- The topic-history mechanism itself landed July 23 (`2b767a7`), immediately before the July 24 semantic-lock era.
- 22.9 and 22.10 were committed Aug. 19 and designed to become live together at the Aug. 20 rotation. Their own plan explicitly says attribution between them is impossible.
- Aug. 18–22 had no entries; traces show almost exclusively `emit_heartbeat`.
- On Aug. 23, tools diversify and exploration resumes. The first exact topic is chosen at assertion tick 100 immediately after a `recall_self` at tick 99.
- That recall result contains the newly applied current focus and active question, plus five repetitive recent heartbeats. The trace itself documents this sequence: [nova-live-20260823.operational.jsonl](data/live/qwen3-14b/logs/traces/nova-live-20260823.operational.jsonl:99), [nova-live-20260823.operational.jsonl](data/live/qwen3-14b/logs/traces/nova-live-20260823.operational.jsonl:100).

The major confound is larger than the diagnosis states. 22.9 simultaneously changed prompt wording, removed `/no_think`, and applied 20 accumulated self-model proposals, including the current focus and active question ([PHASE22_STAGE22_9_PROMPT_CONSOLIDATION.txt](docs/plans/PHASE22_STAGE22_9_PROMPT_CONSOLIDATION.txt:81), [PHASE22_STAGE22_9_PROMPT_CONSOLIDATION.txt](docs/plans/PHASE22_STAGE22_9_PROMPT_CONSOLIDATION.txt:143)). Those state changes flow into every tick through `SelfContextEngine.prefetch` ([runtime.py](src/nova/runtime.py:1003)).

My assessment:

- **Supported:** Aug. 23 is a discrete transition from lexical variation to exact repetition.
- **Supported:** a prior, continuous semantic collapse began around July 24.
- **Unsupported:** assigning the Aug. 23 discontinuity to 22.9, 22.10, or the history block specifically.
- **Plausible but unproven:** the history block sustains the post-fifth-repeat lock.
- **Unsupported:** the “showing the string to avoid pins it” priming explanation.

The reviewed quarantine traces add no competing causal event: available quarantine records are old parse-failure-era material, not Aug. 18–23. The operational traces do add the recall-self/self-context confound above.

### C. Protocol

A three-day, single-arm read is fine as an **operational screen**: hundreds of highly autocorrelated ticks make a total lock or obvious safety regression easy to see quickly. It is not sound evidence of mechanism or causality. There is no control, no randomization, no independent unit, no washout, and persistent state crosses day boundaries.

A practical stronger design would be:

- Pre-register a primary prevention outcome separately from a mechanism outcome.
- Randomize or counterbalance prompt variants by daily session, recording assignment, rather than comparing one chronological block with an earlier era.
- Use an ABAB-style reversal only for reversible, low-risk prompt arms; retain the persistent history and state rather than pretending sessions are independent.
- For the deterministic novelty gate, evaluate it as a safety/control property (“duplicates refused”) rather than waiting three days to prove it changes model behavior.

I would not call Part D a “clean null.” It is a short, uncontrolled null for the particular outcomes observed. “No drive erosion” is especially weak because the metric counts equality of a fixed `motive_priority` field ([tick_analysis.py](src/nova/eval/tick_analysis.py:103)), not substantive persistence or understanding of the drive.

### D. Synthia comparison

The original n=4 inference is over-read. The structural observation is real: Synthia injects the prior three thoughts directly into each heartbeat prompt ([heartbeat.py](synthia:heartbeat.py:240), [heartbeat.py](synthia:heartbeat.py:262)). But it also injects self-state, skills, recent conversation, and active wants ([heartbeat.py](synthia:heartbeat.py:232)), so the tail is not an isolated explanation.

At inspection time the journal had 17 records, and the later Aug. 29 records continue repeating the operator/continuity theme. That strengthens “there is a current repetition pattern,” but still does not establish a cross-architecture law or identify history injection as its cause. Cold start, sparse/new conversation context, self-state wording, and an idle-only prompt are adequate alternatives. Call it a hypothesis-generating parallel, not corroboration.

### E. Metric design

Newest-record anchoring is appropriate for reproducible historical analysis, but wrong as the sole live-health metric. As implemented, a stopped exploration stream keeps reporting a historical “recent” window ([tick_analysis.py](src/nova/eval/tick_analysis.py:515)). Add both:

- `exploration_staleness_days` / timestamp of newest exploration, wall-clock anchored.
- A wall-clock recent count and a “no explorations in N days” alarm.
- Retain record-anchored measures, explicitly labeled as activity-window measures.

Exact-string diversity correctly detects the Aug. 23 byte lock but would miss the July 24 semantic lock. Add a semantic/concentration companion, not a replacement: e.g. dominant-cluster share using the existing `cluster_texts` machinery, with a calibrated threshold. The current helper was specifically built to group reworded exploration topics ([self_context.py](src/nova/agent/self_context.py:10)). An embedding measure can be useful, but it needs versioned model/configuration and retained inputs to remain comparable.

One implementation edge case: the stated “order-based” streak is actually timestamp-sorted, with unparseable timestamps sorted after valid ones ([tick_analysis.py](src/nova/eval/tick_analysis.py:536)). That can make a malformed late record determine `newest_topic`; it is not material to the current valid live data, but the documentation overstates robustness.

### F. Material missed / prioritization

The most important correction is causal scope:

1. The recent-topic block is real and load-bearing for persistence, but cannot explain the initial Aug. 23 exact phrase.
2. The first exact selection had at least three simultaneous inputs: repetitive heartbeat context, newly applied self-model content surfaced by `recall_self`, and the changed 22.9/22.10 apparatus.
3. Therefore arms 1 and 2 are diagnostic, but cheapest-first is reasonable only if their evaluation is framed as “does this break persistence?” rather than “does this establish root cause.”
4. If the immediate goal is to stop duplicate work, Arm 3 is the only assured control. If the goal is diagnosis, run Arms 1/2 first with a stronger counterbalanced protocol.
