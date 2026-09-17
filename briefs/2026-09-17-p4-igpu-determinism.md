# P4 lane: verify and fix the iGPU determinism failure

You are a fable 5.1 lane. Worktree `$HOME/Projects/mojo/mojo-baro-lanes/p4`,
branch `lane-p4`, forked from `main` at `41f60f8`. Work only in that worktree.
Read `CLAUDE.md` at the repo root first; its rules bind you, especially the
kernel-comment rule, the receipt rules, and DONE-means-a-real-test-landed.

## The failure you are inheriting

`exchange/lane-P4-report.md` (merged, honest, do not rewrite its history) records
P4 as FAILED. Wiring, fixtures, CPU preflight and device pinning all passed. The
kill line fired on the identity check:

```
p01-water,xtx,PASS
p02-python-fib,igpu,PASS
p03-story,xtx,PASS
p04-list-planets,igpu,PASS
p05-math,xtx,PASS
p06-translate,igpu,FAIL
```

On `p06-translate` the iGPU solo and split streams agree for the first six token
ids (`220 16 15 15 15 15`) and then diverge. The solo stream degenerates into
repeated token 15; the later split request produces varied tokens. Same
long-lived iGPU process, same prompt, temperature 0, `spec=false`.

Receipts: `.work/team-B/codex/p4/gate/round-robin.csv`,
`.work/team-B/codex/p4/gate/p06-translate.igpu-solo.tokens`,
`.work/team-B/codex/p4/gate/p06-translate.split.tokens` (in the `team-b`
worktree; copy what you need, do not edit that tree).

Protocol `bench/p4-multigpu-protocol.md`, fixtures
`bench/p4-fixtures/manifest.tsv`, gate body `.work/p4/run-two-engines.sh`.

## The lead to test FIRST, before any new hypothesis

That gate ran with main parent `c0e9297`. The `save_state` cross-request state
leak was fixed AFTER it, in `f9048f6`: `save_state` in `serve/engine.mojo` picked
the checkpoint by POSITION ALONE, so an export could carry another prompt's conv
and SSM state whenever two resident checkpoints shared a position. "Same
long-lived process, solo and split diverge" is exactly that bug's shape.

So step 1 is: re-run the SAME gate, unchanged, on a tree that contains `f9048f6`,
and see whether p06 still fails. If it passes, the fix is already on main and your
job is a receipt plus a regression test, not a new fix. Do not start editing the
engine before you have run this.

Report the re-run result either way. A pass here is a real result and closes the
lane; do not go looking for a bug to fix if the bug is already fixed.

## If it still fails

Then it is a live determinism defect and yours to find. Constraints:

- Do NOT replace the Qwen7 iGPU arm and do NOT weaken the identity check. The
  previous lane's closing instruction, and it stands.
- The suspect area named by that lane is the engine request-reset/state path.
  Degenerate repetition of a single token id (15) on gfx1030 specifically is also
  consistent with a numerics/kernel problem on that device, not state bleed.
  Distinguish the two before fixing: a state-bleed bug reproduces with two
  requests and vanishes with one; a gfx1030 numerics bug reproduces with ONE
  request in a fresh process.
- Run that discriminator first. It is cheap and it decides which code you read.
- gfx1030 is the iGPU. `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `HIP_VISIBLE_DEVICES=0`
  and `ROCR_VISIBLE_DEVICES=0` hide or misidentify it; the gate already unsets
  those for topology discovery and reapplies per arm. Keep that behavior.
- `~/iTools/bin/igpu-env` exists for iGPU probing; use it rather than improvising.

## Rules that bind this lane

- Every GPU launch goes through `gpu-wait run [--priority N] [--vram GB]
  [--timeout S] -- <cmd>`, never bare. `gpu-wait` drops the shell env: pass an
  explicit `PATH` the way the previous lane's final submission did.
- Preflight on CPU first. A GPU job never discovers a build, flag or config error.
- Read every arm-defining parameter back from the running system and record it.
  Passing a flag is not evidence it took effect. No receipt, no arm.
- Failures are loud: `set -euo pipefail`, `FAIL <step>: <reason>` plus a log path,
  non-zero exit. A skip is never exit 0.
- Commit as work lands, on `lane-p4`, conventional subject and a why-body, staged
  by explicit pathspec only. Never `git add .` or `git add -A` (this is a shared
  checkout and other lanes stage files). Never add a `Co-Authored-By` or any
  model attribution line.
- No em dashes anywhere in anything you write.
- Do not mark anything done without naming the check that passed. A green build,
  an active unit and a 200 are not done.

## Deliverable

Write the report to `exchange/lane-P4-report-round2.md` in your worktree and
commit it. Reply in the pane with one line only: `written to <path>`. Terminal
output is status, never content.

The report states, in this order: whether the re-run on `f9048f6`-containing main
passed or failed, with the round-robin receipt; the discriminator result if you
ran it; the root cause if you found one, with the evidence that distinguishes it
from the alternative; the fix and its commit; and the gate re-run that proves the
fix, by the same protocol, unweakened. If P4 still fails at the end, say FAILED
and say exactly what is unresolved. A failed lane reported honestly is worth more
than a passed one that moved a threshold.
