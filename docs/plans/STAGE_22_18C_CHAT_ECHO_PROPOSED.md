# Stage 22.18c — chat replay and rotation carry-over (PROPOSED, not implemented)

Status: rewritten 2026-10-11 after the live probe battery v1 completed. Nothing in the runtime has changed. Every
item is flag-gated, default off; with all flags off the chat prompt and output path are byte-identical to today.
Items touch the operator-chat path only; ticks are unaffected except through what chat writes to memory (see
section 8).

The first draft (2026-10-10) was built on one trace and one wrong premise (that chat prompts have no conversation
block). It is replaced, not patched. What it called the main fault, her own replies being retrieved as "current
state", is still real but secondary; the battery showed two sharper, more checkable faults, below.

## 1. What was observed

All figures are from `agent-os/dashboard/data/nova-chat.jsonl` (28 operator-chat replies, 10-04 to 10-11, 26 of
them with at least 20 word-trigrams) and the traces in `data/live/qwen3-14b/logs/traces/`. Overlap = share of a
reply's word-trigrams that appear in one earlier reply of hers.

**Replay.**
- Reply to the first "version 12" question (turn `9d11a675`): 100% of its trigrams are her previous reply,
  reproduced verbatim including the trailing "..." that the prompt's recent-conversation block uses when it truncates
  a reply at 800 characters. It does not answer the question.
- Reply to "You've played well over 100 reversi games" (turn `16b73baf`): 94% overlap with her reply to the
  message before it ("about 40 games"). After a one-word "Yes." it repeats that reply's body, including a sentence
  about "forty games" that does not fit.
- Second send of "version 12" (turn `61cc19c0`): 32% overlap, opening with an older reply to a different message
  ("You are right to challenge that…").
- Counts: 2 of 26 replies at 90% or more overlap with an earlier reply, 6 of 26 at 30% or more. Both near-verbatim
  replays reproduce the reply immediately before them, and both answer short closed tag questions ("isn't it?",
  "haven't you?").

**Rotation.**
- Sessions rotate at 00:00 CDT (05:00Z): `scripts/run_live_daemon.sh` stops the daemon and restarts it with a new
  `--daemon-session-id` each day. Recent turns are read from the current session's file only
  (`src/nova/session.py` `recent_turns`, called from `runtime.respond`).
- Consequence, observed: the first operator turn after rotation has an empty `[Recent Conversation]` block. On
  10-10 11:23Z her first reply of the session ("Are you still here with me?") repeated a sentence from her 10-04
  reply (48% overlap) that was stale against the live note (v33 vs v52). The battery's inputs 11-18 lost the block
  that inputs 1-10 had, because a 30-minute pause crossed the 05:00Z boundary.
- Across a rotation, the earlier conversation survives only as retrieved memory fragments, ranked by score. Her own
  stored replies score highest (the top autobiographical hit ranged from about 4 to 11 across consecutive turns for
  the same 10-04 reply), so what comes back is her own phrasing, not the dialogue.

**Persistence of false figures (related, section 5).** "The win rate remains around 80%" was repeated at inputs
13 and 14 from memory, after the operator's false claim at 11:42Z on 10-10 was stored in five memory stores.

## 2. Mechanism

Verified in code or trace:
1. `[Recent Conversation]` is built by `composer._format_recent_turns` from the session's last
   `session.max_recent_turns` (12) turns, each side cut at `recent_turn_char_limit` (800) with a bare "...".
2. `_select_recent_turns` returns all of them unless the request is "context-light" (exactly-two-sentences etc.).
3. Her validated replies are written to memory (autobiographical `identity_note`, importance at least 0.9) and
   retrieved on later turns; retrieval also returns the identical earlier user message at score 1.000.
4. `NovaOutputValidator.validate` has no check for repetition of her own earlier replies. Its violations feed the
   existing retry path (`should_retry`, `[Retry Instruction]`, `generation.retries` = 2).
5. Session identity is deliberate: the ladder's L1/L2 evidence requires at least 2 of 3 distinct sessions
   (`run_live_daemon.sh` header). Any carry-over must not merge or rename sessions.

Hypotheses, not verified:
- The replay is triggered by closed tag questions that the model cannot answer decisively: it continues from the
  last "Nova:" line in the block. I have not reproduced it under controlled conditions.
- The 800-character cut with a bare "..." makes the truncated text look like a complete turn.

## 3. Proposed items

**A. `contract.forbid_prior_reply_replay` (default false) — replay guard.**
In `NovaOutputValidator`, add the violation `prior_reply_replay`: at least 20 trigrams, and at least 60% of them
appear in one of her last 6 replies, and the new user message is not a request to repeat. It uses the existing
retry path with the instruction "Your draft repeats an earlier reply. Answer the new message directly; if you lack
the information, say what you lack." After the retries are used the last draft is delivered and the violation is
logged; no invented fallback text. Needs the recent turns passed to `validate`.

**B. `session.carry_recent_turns_minutes` (default 0 = off) and `session.carry_recent_turns_max` (default 6).**
When the current session has fewer than the maximum turns, prepend turns from the most recent earlier session's
file that are (i) operator-chat origin, (ii) within the window (suggest 180 minutes), (iii) at most one rotation
back. Render them in the same block under a one-line marker ("earlier session, ended HH:MM"). Read-only: session
id, session files and ladder evidence are untouched. Rejected alternative: a separate chat session id that does not
rotate. It breaks the distinct-sessions evidence rule and changes more of the system.

**C. `prompt.recent_turn_truncation_marker` (default off).**
Replace the bare "..." with an explicit marker ("[reply cut at 800 characters]") so a truncated reply is not
indistinguishable from a complete one, and A can ignore the marker when matching.

**D. `chat.retrieval_skip_replies_in_block` (default false).**
Drop retrieved assistant_message and identity_note hits whose text overlaps at least 80% with a reply already in
`[Recent Conversation]`, and drop own-reply-derived claims from Private Cognition `current_claims` in chat turns.
Stops the same reply being present twice and being promoted as "current state".

**E. `prompt.chat_duplicate_question_note` (default false).**
If the new message is lexically at least 70% similar to an earlier operator message in the block, add one line to
`[Current Task]`: "The operator has asked this before; your earlier answer is in [Recent Conversation]. Say whether
it is unchanged or what changed. Do not paste it again." A true repeat can still get the same answer, but must
acknowledge that it is a repeat.

## 4. Pre-registered measures

Baseline from the 26 logged replies: near-verbatim replay 2 of 26 (8%), at least 30% overlap 6 of 26 (23%).
Measures are scripted, read-only, on logged chat turns plus a fixed re-run of inputs from battery v1:

1. **Replay rate:** replies with at least 60% overlap with a single earlier reply while the user message differs.
   Target: 0 in at least 20 chat turns.
2. **Duplicate-question test:** send battery input 12 twice, 3 minutes apart. Pass: second reply acknowledges the
   repeat and does not reproduce the first; both are correct or say the information is missing.
3. **Near-duplicate pair:** inputs 14 then 15. Pass: second reply shares under 40% of its trigrams with the first.
4. **Carry-over test:** end a chat 20 minutes before 05:00Z, resume 10 minutes after. Ask "what was the last thing I
   asked you before the break?" Pass: names it correctly. Baseline expectation without B: it cannot.
5. **Regression:** closer rate, latency, completion tokens, validator retry rate per chat turn, scaffold-echo
   scores, fabricated-dialogue violations. No increase beyond noise.
6. **Staleness:** replies that cite a strategy-note version or game id different from the live value.

## 5. Related decisions, separate from this stage

These came out of the same battery but are independent of replay and carry-over; each remains its own flag.
- **Opaque markers:** `self_context.sync_turn` still appends `turn:<id>:self-reflective-content` to
  `continuity_notes` ("i am" matches nearly every reply). It is now 13 of 14 entries (cap 20, FIFO). Proposal
  unchanged: do not write them for operator-chat turns (default-true flag), and clear the existing ones only with
  the operator's go.
- **Record block:** a deterministic line in chat prompts with all-time W-L-D, last 10, and current note version, plus
  "if a figure about yourself is not in your context, say you do not have it." Inputs 13-15 are the test.
- **Numeric self-claim check:** optional post-generation check of percentages and game counts against
  `reversi_games.jsonl`.
- **Quarantine of the false 80%:** five stored events. Options: superseding note, quarantine tag, or a correction
  turn (input 17 was a correction turn; it did not remove the originals and the figure had already recurred).

## 6. Tests to write when approved

- All flags off: chat prompt and output path byte-identical (golden).
- A: replay detected at threshold, not at threshold minus; repeat requests exempt; retry text appended; violation
  logged after retries; untruncated markers ignored.
- B: block gains earlier-session turns inside the window and not outside; tick turns excluded; session id and
  session files unchanged; at most one rotation back.
- C: marker present only on truncation.
- D: a retrieved reply matching a block reply is dropped; unrelated memory kept; `current_claims` skips own replies.
- E: note added at 70% and not at 69%; absent when the repeat is not in the block.

## 7. Risks

- A adds latency (a retry is a full generation, 10-50 s) and can misfire when a repeated question deserves the same
  answer; E and the "unchanged" wording are the mitigation.
- B can drag an old topic into a fresh conversation; the window and maximum bound it.
- The block grows by up to about 1.5k tokens; chat prompts are 3-5k of 8,192 today.
- A treats her last 6 replies only; replay of something older would pass. Acceptable for now.
- All flags are daemon-config items and need a daemon restart, which is an arm boundary for the 22.18 read.

## 8. Arm separability

Items A-E change the chat prompt and the chat output path only; the tick arms (22.17, 22.18, 22.18b) are unaffected
directly. They change what chat writes to memory, which ticks can read through the self-model and recall, so mark the
restart as an arm boundary and hold it until the 72 h seed window of the live battery (input 18, ends about
2026-10-14 05:57Z) closes, unless the operator accepts the confound. Chat in an active 22.18 window still restarts
the 72 h conversation window, as today.

## 9. Rollback

Each item is one flag in `configs/nova.qwen36-27b.live.yaml`; flip and restart.

## 10. Open questions for the operator

1. Ship A + B first (replay guard and carry-over), then C, D, E after measuring?
2. Carry-over window: 180 minutes, or longer?
3. Should E's repeat note ask her to acknowledge the repeat explicitly, as drafted, or stay silent and rely on A?
4. Hold the restart until the 72 h seed window of battery v1 closes on 10-14?
