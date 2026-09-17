# Lane P4B follow-up: B1's identity receipt is too thin to carry the P4 claim

Driver check of your status (2026-09-18). C1, C3, C2, D1, D2: verified by use, nothing to redo.
B1: the freeze order held (amendment `3d3e362` at 00:18:20, first run started 00:19:06), TMAX 4096
read back on both engines, placement a=10 b=10, 3 of 3 PASS. Two defects, both yours to close
before the final report:

## 1. The identity check compares about 70 tokens per run

`.work/p4b/gate1-run1/identity.json`: 20 rows = 5 distinct prompts x 4 repeats, answers like
`\n\namber`, 272 characters in total. Team A's gate 1 is a PLACEMENT gate with a small identity
check; that was right for P0b. The P4 protocol's identity gate is 20 prompts x 64 tokens
(`bench/mtp-prompts/p*.tokens`, token ids, T=0, `spec=false`), which is about 35 times more
compared tokens. Three passes of a 70-token check do not put "the repeat-rule receipt" behind
P4, and the report may not say so.

Do this:
1. Amend `bench/p4-multigpu-protocol.md` again: the XTX identity gate sends the 20
   `bench/mtp-prompts/p*.tokens` prompts as token ids, `max_tokens` 64, `temperature` 0,
   `spec` false, THROUGH THE ROUTER, and compares each response's token ids with the same prompt
   sent directly to one engine (the solo arm), by `cmp` on the ids, never on text. 3 runs on
   identical binaries, 20/20 each. State plainly that the three 00:19 to 00:21 runs were the
   placement gate and count as placement receipts only. COMMIT, then run. Tightening a gate is
   allowed; running before the commit is not.
2. Write the gate as `bench/p4-router-identity.sh` (strict mode, loud failures, CPU preflight
   mode, gate-dryrun stop, explicit PATH for gpu-wait). Reuse the launch half of team A's gate
   script by calling it or sourcing its functions if it allows, do not fork a copy of the router
   bring-up if you can avoid it; say in the report which you did. The payload and token
   extraction exist in `.work/p4/run-two-engines.sh` of the MAIN checkout (`payload`, `tokens`,
   `request`).
3. Placement must still spread: the receipt shows how many of the 20 went to each engine, and the
   gate FAILS if either engine served zero (an identity pass with one engine doing all the work
   proves nothing about the split). That is this gate's reverse arm; see the
   `reverse-arm-gates` skill.
4. Exercise the comparison on CPU before the queue with a stub: two token files that differ in
   one id must produce FAIL and a non-zero exit. Put that negative control's output in the report.

## 2. The device read-back is prose, not a file

The report says `rocm-smi --showpids` was read mid-run-2, but no showpids output is on disk under
`.work/p4b/`. No receipt, no arm (`bench/PROTOCOL-RULES.md` P1). The new gate script writes
`rocm-smi --showpids` and the two engine child PIDs to its OUT dir on every run and fails if
either PID is missing from it. Also read back `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT`
(or whatever cap team A's launch uses) from `/proc/<pid>/environ` of each engine, into the receipt.

## Rules

Same as `briefs/2026-09-17-p4b-skills-itools-protocol.md`. GPU budget for this follow-up: 30
minutes. If any of the 3 runs misses identity, P4 stays FAILED: report the miss with both token
files, do not re-run until it passes. Then finish `exchange/lane-P4B-report.md` with B1 rewritten
around these runs, the `./run-tests.sh` receipt, and reply `written to <path>`.
