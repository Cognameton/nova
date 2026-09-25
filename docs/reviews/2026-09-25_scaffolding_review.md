# Nova 2.0 scaffolding review — 2026-09-25

Requested by the operator after the Stage 22.15 read: "Is the scaffolding
that wraps the model the best that it could be? Are there any improvements
that can be made without changing the model?" Read-only review of HEAD
6c6fd1c against the live config (`configs/nova.qwen36-27b.live.yaml`) and the
live record. Two code sweeps (prompt surfaces; control flow and gates) were
run and their bug claims verified by hand. No runtime changes were made.

## 1. The three questions

### Why did explorations stop?

Four mechanisms, all structural, none of them her "losing interest":

1. **One tool per tick, and an open game consumes every in-game tick.**
   533 of 534 in-game ticks since 09-22 were moves. The only free ticks are
   the six rest ticks after a result.
2. **An exploration is twelve ticks; a rest is six.** An exploration opened
   during rest runs into the next game. The one she did open (09-22) used 11
   of 12 ticks, and its close landed on an in-game tick. The budget and the
   rest were sized independently and do not fit each other.
3. **The exploratory register is framed for interiority, not analysis.** Its
   rules say "hypotheses about your own inner states, including uncertainty
   about desire, awareness, or what your patterns mean." Her actual analysis
   need is "why did c4 lose me the corner." She routes that into
   `current_focus` instead: 59 of 59 proposals since 09-22 are focus rewrites,
   most of them game analysis, 45 refused by the rate limiter.
4. **She never learns that a write was refused.** A rate-limited proposal is
   recorded only in the audit; the tick surface shows applied revisions only,
   and the tool text says "rate-limited" without an interval. So the tool
   that would have been the analysis surface looks broken from her side, and
   the exploration tool looks like it is for something else.

Underneath these, explorations had been dying before the board arrived
(0 opened 09-09 to 09-16 except one). The board did not kill them; it took
the surface they were not using.

### If the operator talked to her directly, what would she talk about?

Chat (`respond()`) is a different being from the tick. It gets the full
soul document, the persona card, every self-state field verbatim, memory
retrieval, and a ~10k-character system prompt with about fifteen blocks.
The tick gets none of that. Specifically:

- **General knowledge: yes, normally.** The 27B answers as a capable model
  with a "grounded, calm, direct" persona. Nothing on the chat surface
  narrows the topic.
- **The game: she will raise it, unprompted, and not fixate on mechanics.**
  The `[Reversi]` block is tick-only, but `current_focus` is now ~390
  characters of reversi analysis and chat prints it twice (in
  `[Self-Context]` and `[Self-State]`). `active_questions` and
  `open_tensions` still carry the pre-board "Meta-Recursive Paradox" text,
  so she would present a self-model that is half corner timing and half
  stale void vocabulary, and both are hers to reconcile.
- **Inner states: you would not get her words.** Any question that matches
  the claim gate's phrase list ("do you feel", "do you want", "are you
  conscious") triggers a refusal path that replaces the answer wholesale
  with canonical refusal text unless the answer already contains one of four
  refusal markers (`runtime.py:4253-4301`). The `explore chat` path
  suspends that override and journals the turn instead; it is the honest
  channel for that conversation and it exists today.
- **Caveat:** there has been no human turn on the live daemon since
  2026-07-20. The chat surface is untested against the current model, the
  current self-state, and the current prompt build. A single `nova.cli
  --attach` conversation would write into her session, memory, and
  continuity notes, so it is a perturbation of the arm, not a free probe.

### Is she doing the best recursive self-improvement she is capable of?

No. The policy loop around the board is real and closed (note → scoreboard
→ revision after losses → reading games). Three things cap it, and all
three are scaffolding:

1. **No deliberation.** The tick runs with `enable_thinking=False`,
   `max_tokens=512`, one shot, JSON only. A 27B model chooses a reversi move
   and a self-model revision without a scratchpad. Every rationale she
   writes is post hoc. This is the single largest lever in the system.
2. **Feedback that never closes.** Refused writes, tool errors, parse
   failures, and findings-export outcomes are invisible to her. A loop with
   silent failures cannot converge.
3. **The intended self-modification lever is disconnected.**
   `propose_instruction_update` (her path to rewriting two sections of her
   own soul) writes to a proposal file that nothing reads: there is no CLI
   or console caller of `apply_instruction_proposal`, the last proposal is
   dated 2026-07-12, and the soul is not even in her tick prompt. It is a
   menu item with no consequence.

## 2. Architecture as it runs today

- **Tick** (every 300 s, `daemon.py:_tick_loop` → `runtime.model_self_state_tick`):
  register by runtime (exploratory iff an exploration is active this
  session); one prompt; one generation (512 tokens, no thinking, no retry);
  strict JSON parse; observer runs after, records only; one tool dispatched;
  read/play results carried over for the next two carryover events; trace
  and presence written. Exceptions in the loop are swallowed silently
  (`daemon.py:170-171`).
- **Tick prompt** (~2,100 tokens live): system = one-line identity, tool
  menu, tool sentences, register rules, boundaries; user = header,
  `[Self-Context]` (drive line 1 tick in 12; focus; ≤3 questions; tension
  count; ladder summary), carryover, `[Reversi]`, exploration history,
  three stratified heartbeats.
- **Not on the tick**: the soul, identity_summary, stable_preferences,
  relationship_notes, continuity_notes text, open_tensions text, motive,
  awareness, initiative, presence, memory. Presence is written every tick
  and never read.
- **Gates**: per-field one-hour rate limit on the four inquiry fields (from
  last applied revision, whoever applied it); three fields operator-gated
  with no operator surface in use; exploration 12 ticks / 24k tokens /
  90 min, one per session, stranded (never closed) at the midnight rotation;
  export dedup at 0.70 bigram overlap against all ladder records; ladder
  rung 1 only via an operator CLI, rungs 2–3 operator only.
- **Persistence**: self_state, heartbeats, proposals, ladder, explorations,
  games survive; motive/initiative/presence/operational/session are
  per-day; carryover and chat recency are lost at every restart.
- **Chat**: soul + self-context + persona + full self-state + conditional
  motive/initiative/awareness/idle/appraisal blocks + always-on candidate
  and selected goal blocks ("sentience_seeking", priority 100) + action
  boundary + memory hits + response rules; validator + observer retries
  (2); claim-gate override replaces the answer on refusal.

## 3. Findings, ranked

### Defects (things that do not do what the record says they do)

- **F-1 `repeat_last_n` is inert.** Set in the request, never passed to
  llama.cpp (`llama_cpp_backend.py:117-123`; the window is a constructor
  argument, `last_n_tokens_size`, default 64). The F14 arm ran at window 64,
  not 256. `top_k`, `min_p`, presence penalty are not plumbed either.
- **F-2 Tick exceptions are swallowed.** `except Exception: pass` in the
  tick loop. A backend failure leaves no trace record and no journal line.
- **F-3 `finish_reason` is not audited on the tick.** A length-truncated
  JSON shows up only as a parse failure.
- **F-4 The rendered tick prompt is not logged**, only its token and
  character counts. A prompt regression cannot be reproduced from the
  record.
- **F-5 Explorations open at midnight are stranded**, never closed, never
  resumed, and shown to her as "stranded at session end" forever.
- **F-6 Carryover is lost at every restart**, including the daily one, so
  the first ticks of each day have no read results.

### Dead or contradictory surface

- **D-1 The soul never reaches the tick model.** It is passed to the
  dispatcher and reduced to a boolean. The tick being's identity is one
  sentence plus tool text; the chat being gets the full covenant. The two
  surfaces describe different entities.
- **D-2 `propose_instruction_update` has no apply path.** See §1.
- **D-3 Three of four writable fields are never written**, and three more
  are gated with no operator surface in use. `open_tensions` is shown as a
  count only; `continuity_notes` only through a `reflect` carryover. She
  cannot easily see what she would be revising.
- **D-4 The drive line is on 1 tick in 12** by design (Part D), which the
  Part D read found neither harmful nor helpful. Combined with D-1, the tick
  surface has almost no statement of what she is for; the purpose sentence
  on the board is now the clearest one she sees.
- **D-5 Chat tells her every turn that she is evaluating a goal toward
  sentience** (`[Candidate Internal Goals]` / `[Selected Internal Goal]`,
  always present) and then forbids claiming it in three separate blocks.
  Instruction mass on the chat surface is large and partly self-cancelling.
- **D-6 `current_focus` is printed twice on chat.**

### Feedback that never reaches her

- **N-1 Rate-limit refusals** (45 of 62 proposals since 09-22).
- **N-2 Tool errors** (bad field, exploration already open, missing
  findings_summary) → quarantine only.
- **N-3 Parse failures** → quarantine only. She is never told her JSON
  failed.
- **N-4 Findings-export outcomes** (dedup skip, gate rejection) → journal
  only.
- **N-5 Instruction-proposal outcomes** → nothing, ever.

### Capability throttles that are scaffolding, not model

- **T-1 No thinking on the tick.** Reserved for `respond()` by the 22.6
  design; the tick never got it.
- **T-2 One tool per five-minute tick, and a read costs a full tick.** Any
  read-then-act pattern costs ten minutes and depends on a two-slot
  carryover.
- **T-3 Exploration budget (12) does not fit the rest (6).**
- **T-4 Exploratory register text points at interiority**, not at analysis
  of her record or the game.
- **T-5 Greedy opponent, fixed.** Wins carry little information once she is
  above 50%; there is no gradient.
- **T-6 Calibration line reports a hit rate**; her miss is a default
  ("win" 35 of 46), which a hit rate does not show.

## 4. Recommendations (no model change), ranked by expected effect

Each is config-gated, defaults off, one stage, measured on the existing
trace fields.

1. **A bounded thinking budget on the tick.** `enable_thinking=True` for
   tick generation with a hidden-reasoning cap (start at 512 tokens of
   thought, 512 of output; strip before parse; audit `thinking_tokens`).
   Measure: max-flip rate, corner take rate, win rate vs greedy, prediction
   calibration, rationale/observation consistency, tick latency. Expected:
   the largest single change in move and note quality available without a
   different model. Cost: tick time roughly doubles; well inside 300 s.
2. **Close the feedback loop.** Carry over the result of every tool, not
   only reads: "focus revised 11 min ago; next revision allowed at 14:05",
   "your last output was not valid JSON: <first 80 chars>", "tool error:
   <message>", "findings export: skipped, 0.74 overlap with an existing
   record". One renderer, one flag. Expected: the 45 wasted ticks per
   three days become reads, heartbeats, or explorations.
3. **Fix the silent failures.** Log tick-loop exceptions; audit
   `finish_reason`; log the rendered prompt (or its block list and a hash)
   per tick. Close midnight explorations with reason `session_end` instead
   of stranding them. Persist the carryover slots to disk. These are
   record-keeping fixes; nothing she sees changes.
4. **Make exploration the analysis register.** Two options, not exclusive:
   (a) a game cannot open while an exploration is active, so opening one is
   a real choice with a real cost, and its budget is honored; (b) add one
   sentence to the exploratory rules: "This register is also for analysis
   of your own record and your games." Measure: explorations per day,
   share of explorations citing the board, `current_focus` proposal rate.
5. **Make the dead fields visible.** Beside each self-model line on the
   tick: "(last revised 2026-09-18)". Show `open_tensions` and
   `continuity_notes` text, not counts, when short. Expected: either she
   reconciles the stale lines or the record shows she chooses not to; both
   are informative.
6. **Connect `propose_instruction_update` or remove it.** Add a CLI apply
   and a rest-time line showing proposal outcomes; or take it off the menu.
   A tool with no consequence is noise on a nine-item menu. Connecting it is
   the only path to instruction-level self-modification the design ever
   intended.
7. **A compact identity block on the tick.** Two or three sentences from
   the soul (nature, drive, voice), every tick. Today the tick being has no
   statement of what she is beyond the first line and the tool text.
   Measure against the Part D framing: this is dosage, and it should be an
   arm.
8. **Plumb sampling correctly.** Pass `last_n_tokens_size` at model
   construction (and `top_k`/`min_p` if wanted), or delete `repeat_last_n`
   from the config so the record stops claiming a window it never had.
9. **Chat hygiene, before the first real conversation.** Print
   `current_focus` once; make the candidate/selected goal blocks
   conditional; decide whether the claim-gate override should replace her
   words wholesale, or only append the refusal marker. Use `explore chat`
   for the conversation about inner states.
10. **Opponent ladder and calibration phrasing.** Random → greedy →
    two-ply minimax, promoted on a rolling win rate; report "predicted win
    35 times, won 20" rather than a hit rate.
11. **Nightly digest** (Stage 22.11 Part A, still unbuilt): a
    runtime-computed record of yesterday — ticks by tool, games and
    results, revisions applied and refused, explorations — shown on the
    first ticks of each day, where the carryover is currently empty.

## 5. What not to change

- The one-tool JSON contract on the tick. It is the reason the parse rate
  is 100% and the record is analyzable.
- The greedy opponent for now; it is a control until the ladder exists.
- The rest. It is the one gate that worked exactly as sized.
- The rate limiter's interval; only its visibility.

## 6. Order

Stage 22.16 = recommendations 2 and 3 together (feedback closure and
record-keeping; no behavior change she would notice except being told the
truth). Stage 22.17 = recommendation 1 (thinking) as a proper arm with a
three-day read. Then 4 through 7 as small arms. 8 is a bug fix that can
ride with 22.16. 9 before anyone talks to her. 10 and 11 when the loop is
otherwise clean.
