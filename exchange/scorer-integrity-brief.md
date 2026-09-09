# Brief — can a loop candidate still game the gate?

**Deliverable: `~/Projects/mojo-baro/exchange/scorer-integrity-report.md`.** Write there and
reply only `written to <path>`. Terminal output is status, never content.

## Why you are here

Iteration 004's `cand-1` commented out a profiling guard. Iteration 005's `cand-2` deleted
`tq = perf_counter_ns()` outright — it attacked the instrument instead of the kernel. Both
were caught, and `tools/loop-gate.sh` grew a stage-0 scope check because of them.

A research pass on exactly this problem produced a ranked set of controls and, more useful,
the list of attacks that survive them. It is at

    ~/Brain/agents/2026-09-08-candidate-cannot-own-the-scorer.md

Read it first. Its organising rule is: **the candidate knows what success means, can measure
an approximate version of it, and cannot alter how authoritative success is measured.**

## The task

Audit this repository's self-optimising loop against that rule and report. In scope:

    tools/loop-gate.sh · tools/loop-propose.py · tools/diff-normalise.py
    bench/loop-protocol.md · PROTOCOL-RULES.md (wherever it lives) · docs/BASELINE.md
    the receipts under .work/loop/ if any survive

For each of the note's seven controls, say plainly: **present, partial, or absent here** —
with the line that implements it, or the reason it cannot be. Do not restate the note.

Four leads found while writing this brief. Confirm or refute each with evidence from the
code; do not take them on my word, and say so when one is wrong:

1. **The metric is self-reported by the artifact under test.** Stage 3 reads `tok/s_gen`
   back out of the engine's own stdout — the binary the candidate wrote prints its own
   score. Stage 0's grep bans edits to `tok/s`/`print(`/`perf_counter_ns`, which mitigates
   but does not remove it. Is an independent wall-clock measurement outside the engine
   possible here, and what would it cost in fidelity?
2. **One fixture.** Identity is 64 greedy tokens against one reference. A candidate that
   detects that exact input, or that shape, is not caught by anything in the ladder. What
   would a second, unpublished workload cost to add?
3. **Scoring runs in a copy, not a fresh checkout.** `$work/src` is `cp -r` of `$dir/src`.
   Say what that lets through that a clean checkout of the candidate's commit would not.
4. **Stage 0 is a denylist of tokens plus an allowlist of files.** The allowlist is the
   strong half. Is the token denylist reachable around — by touching a file on the
   allowlist in a way none of those patterns match?

Then: name the attacks from the note's residual list that this ladder does **not** cover,
ranked by how cheap they are to execute here. "Move work outside the timed region" is the
one I would look at hardest.

## Rules

- **Do not change the frozen acceptance rule** (`bench/loop-protocol.md`, 2026-09-01).
  Changing what counts as a win changes the meaning of every past receipt — that is
  the maintainer's call. Propose, with the diff you would apply, and stop.
- Gate hardening that does not alter acceptance (a stricter scope check, a fresh checkout,
  a second fixture behind a flag) you may implement, each as its own commit, only if it is
  small and you can show it still passes on a known-good candidate.
- **Every GPU workload goes through the waiting room**: `gpu-wait run [--priority N] -- <cmd>`,
  never bare. That includes any `mojo build` + engine run, and llama-server. If
  `GPU_WAITING_ROOM_JOB` is already set you are admitted and run bare. Check with
  `gpu-wait gpu` / `gpu-wait list` before assuming the GPU is free.
- The loop's own choreography stands: the server on 8083 is the proposer, not the
  instrument, and must be stopped before any timing stage.
- Commit as work lands, conventional subject and a why-body, on the current branch.
  **Never add a `Co-Authored-By` or any model attribution line.** Never a git remote.
- If you find a real bug, fix it and name it in the report. Ask first only if the fix is
  large, changes an interface, or is really a design call.
- Report what you did not do and why. A gap you name is worth more than a gap you paper over.
