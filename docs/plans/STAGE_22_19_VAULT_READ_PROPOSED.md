# Stage 22.19 — reading the shared vault (PROPOSED, not implemented)

Status: proposed 2026-10-03 at the operator's request ("add Nova vault reading after the 22.17 review").
The 22.17 read is done (2026-09-29), but Stage 22.18 went live 2026-10-02 23:05:36 CDT, so this stage waits
for the 22.18 first read (>= 3 complete days) to keep the two arms separable. Nothing in the runtime changes
until the operator says go.

## What the operator asked for
All local agents (Claude Code, Codex, Hermes, both OpenClaws, Nova) share one markdown vault at
`/home/head-node/Dev/agent-os/memory`. Every agent's own memory is mirrored into `agents/<agent>/` every
minute, Nova's included (`agents/nova/`: self-model, reversi notes, explorations, long-term memory). The
operator wants Nova to be able to READ all of it, the same as the other agents. Her own memory already flows
out through the mirror; this stage is the inbound half only.

## Proposed mechanism
- Flag `prompt.tick_vault_tool` (default false). Off = today's behavior byte-for-byte.
- One in-tick read tool, same register as `recall_history` (counts toward `reads_this_tick`, does not end
  the tick):
  - `read_vault` with `{"mode": "index"}` → a compact list: `agents/<agent>/` note titles with dates,
    the 10 newest `sessions/` and `inbox/` titles, and the top-level notes. No bodies.
  - `read_vault` with `{"mode": "note", "path": "<vault-relative path>"}` → that note's body, front matter
    stripped, capped at 1,500 characters with a "(truncated, N more)" marker.
- Read-only. Resolved paths must stay inside the vault (realpath check, follows the `claude-memory/` link).
  No write tool: other agents learn from Nova through the existing mirror.
- Excluded: `agents/nova/` (her own memory reflected back at her; she already reads the source directly).
- Tool menu line, one sentence: "read_vault: notes other agents on this machine keep (projects, what they
  remember, their recent work). Read-only."
- No vault content is pushed into the prompt unasked. The menu line is the only standing text.

## Budget
n_ctx 8192; tick prompts peak ~2,800 tokens today plus 1,024 thinking. One index (~400 tokens) or one
note (≤ ~450 tokens) per call fits; cap at 2 vault reads per tick.

## Audit
`adapter_audit.calls[]` already records in-tick reads; add `vault_path` and `chars_returned`. Check-in
script: count of read_vault calls, index vs note, which agents' notes, repeat reads.

## What the read should look at (>= 3 complete days)
1. Take-up: does she call it at all, in-game vs rest ticks.
2. What she reads: which agents, which notes, repeats.
3. Spillover: do exploration topics or self-model writes start referencing other agents' material?
   Does the semantic lock (stasis/void theme) loosen, or does she import another agent's framing?
4. Cost: latency per tick, finish_reason=length rate, parse failures vs the 22.18 baseline.
5. Membrane: anything business-sensitive she repeats into heartbeats/explorations (the vault holds client
   and business notes; local-only, but worth seeing what she does with it).

## Open questions for the operator
- Include `claude-memory/` (business/client facts) or only `agents/` + `sessions/` + `inbox/`?
- Should she later get a write path (her own notes into `inbox/`), or stay mirror-only?
