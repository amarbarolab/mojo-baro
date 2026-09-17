# Lane P4B: turn the P4 round 2 lessons into skills, one iTool, and the P4 protocol amendment

You are a sonnet lane. Worktree `$HOME/Projects/mojo/mojo-baro-lanes/p4b`, branch `lane-p4b`.
Read `CLAUDE.md` at the repo root and `~/.claude/CLAUDE.md` first; both bind you. the maintainer approved
every item below on 2026-09-17 ("all of them"). Do them in the order given, one at a time, each
with its check, and commit as each lands.

## Source material (read these first, in one burst, then do not re-read)

- `exchange/lane-P4-report-round2.md` (what happened, with receipts under `.work/p4/` in the MAIN
  checkout `$HOME/Projects/mojo/mojo-baro/.work/p4/`)
- `~/Brain/mojo/mojo-baro/2026-09-17-evolve-options.md` (the options table; your items are C1, C2,
  C3, D1, D2, B1)
- `~/Brain/mojo/mojo-baro/2026-09-17-p4-igpu-transient-corruption.md`
- `bench/p4-soak.sh`, `bench/p4-gate-watch.sh`, `bench/p4-igpu-load.sh`, `tools/p4-row-diff.py`,
  `tools/p4-trace-diff.py` (the instruments; D1 lifts two functions out of `p4-soak.sh`)
- `~/Brain/Skills/gate-authoring/SKILL.md`, `~/Brain/Skills/_index.md`,
  `~/iTools/hardware/igpu-env/` (neighbour to copy for D1: own dir, `tool.toml`, a `.sh`)

## Items

### C1. Update the `gate-authoring` skill (S)
Add four rules, each with its one-line receipt from the report, in the skill's existing voice:
1. An identity PASS on any device needs a soak or repeated runs behind it. Receipt: the unchanged
   P4 gate went FAIL, FAIL, PASS on identical binaries.
2. Never edit a script while its job runs. Receipt: job `mu5vx2nsz9yy` exited 2 on a shifted tail.
3. Exercise every poll or watch loop on CPU against a stub before the queue. Receipt: the first
   `p4-gate-watch.sh` job died in its first poll and orphaned a gate run outside the queue.
4. Under `set -euo pipefail`, `[ test ] && cmd` as the LAST statement of a loop or function body
   returns 1 when the test is false and kills the script silently, often only for one input
   (it passed for the iGPU node and died for the XTX node). Write `if`. Also `sed` on a missing
   file returns 2 inside a pipeline.
Check: the skill file parses (frontmatter intact), `_index.md` line still accurate, and
`~/iTools/bin/synth-gate gate-authoring` (or the index's own check, whichever exists) passes.

### C3. One rule in `lane-dispatch` and `mixed-pair-teams` (XS)
"A brief's lead names the binary or harness it applies to, and the dispatcher greps for it
first." Receipt: the P4 brief's `save_state` lead pointed at `serve/engine.mojo` while the failing
arm ran `serve/spark.mojo` (0 hits for save_state there); the stale-KV lead pointed at a
sliding-window branch that `SWA_WIN = 0` never takes. Same check as C1.

### C2. New skill `determinism-triage` (M)
Scaffold with `~/iTools/skills-tooling/skill-new/skill-new.sh determinism-triage "<desc>"`.
Content = the ladder that worked, each rung with the instrument that implements it and what its
result rules in or out:
1. Which harness and binary does the failing arm run? grep before reading any code.
2. Fresh-process vs history discriminator (`bench/p4-discriminator.sh`): state bleed reproduces
   with two requests and vanishes with one; numerics or transients reproduce in a fresh process.
3. Bitwise row capture across fresh processes (`p4-glitch-capture.sh` + `p4-row-diff.py`); note
   that a dump path adds a host sync and can mask a race.
4. Non-perturbing device checksums (`-D BARO_TRACE_SUM=1` + `p4-trace-diff.py`): first differing
   (position, layer, stage) cell.
5. Control device: same source, other target, matched by token count AND stated wall time.
6. Load arms with an engine-time read-back from fdinfo; holding a render node is not using it.
7. Before calling anything clean, compute P(0 events | observed rate); 10k clean tokens at 1 in
   3,700 is still 6 percent luck. Read file times before writing any claim about when.
Include a "what does not count" section: a single PASS, a holder list, a clean run under a
perturbing probe. Check: skill-new's own gate passes, `_index.md` has the row, the symlink
`~/.claude/skills/.user/determinism-triage` resolves.

### D1. New iTool `gpu-clients` (S, bash, under `~/iTools/gpu-telemetry/gpu-clients/`)
Bash, not Mojo and not Python: it is /proc and sysfs reads, the same shape as `igpu-env.sh`.
`gpu-clients [--seconds N] [--node /dev/dri/renderD129]`: for each render node (or the one
given) print the PCI device and gfx target, every process holding it, each holder's DRM engine
time DELTA over N seconds per engine (gfx, compute, dma, from `/proc/PID/fdinfo`, children not
needed), and for every KFD compute process `evicted_ms` per gpu_id. A holder with zero delta is
printed as `idle`. Lift `load_ns` and the node lookup from `bench/p4-soak.sh` (both already
fixed for strict mode). `set -euo pipefail`, loud failures, `--help` from the header comment.
`tool.toml` copied from a neighbour, then `cd ~/iTools && ./index-gen --write` and verify the row
and the `bin/gpu-clients` shim landed. NEVER hand-edit `INDEX.md`.
Check (no GPU job needed): run it for 3 s on this desktop; it must list at least the compositor
on the XTX node with a non-zero gfx delta and must exit 0 when a node has no holders. Put the
output in your report.

### D2. `igpu-env`: say what it is not (XS)
In `~/iTools/hardware/igpu-env/igpu-env.sh` header and its printed banner add one line: the
Raphael iGPU under this override is NOT bit-reproducible, 3 one-token deviations in about 11,000
tokens measured 2026-09-17, never gate identity on it; pointer to the Brain note. Add the same
paragraph to `~/Brain/OS/2026-09-17-raphael-igpu-under-rocm-and-mojo.md`. Regenerate the iTools
index. Check: `igpu-env` still prints eval-able lines (the new line must be a `#` comment), and
`igpu-env --probe` still passes. The probe touches the iGPU: run it as
`gpu-wait run --timeout 300 --vram 1 -- env PATH=/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin HOME=$HOME igpu-env --probe`.

### B1. Amend the P4 protocol (M, no new engine code)
the maintainer's decision: the iGPU arm stays as the PINNING AND WIRING receipt, reported, not gated;
identity under split is gated on two engine processes on the XTX. Steps:
1. Read team A's router report (find it with `ls exchange/ | grep -i -E "team-a|P0b|router"`) and
   confirm from its receipts, not its words, that P0b gate 1 already ran 20 prompts through the
   router across two real engines on the XTX at `BARO_TMAX=4096` with every response identical
   to its single-engine run. If the receipts are on disk, that run IS the identity-under-split
   receipt and you do not rebuild it. If they are not, say so and stop B1 there.
2. Amend `bench/p4-multigpu-protocol.md`: status, the split of what each arm proves, the iGPU
   finding with a pointer to the round 2 report, the repeat rule (the identity gate is run 3
   times on identical binaries, all 3 must pass, because this gate has passed by luck before),
   and that the state-move gate no longer exists (LatentOS is an exploration since `50f9d34`;
   check the whiteboard card and `docs/PLATFORM-PLAN.md` say the same). Update the P4 section of
   `docs/PLATFORM-PLAN.md` to match. COMMIT THIS BEFORE ANY RUN: the amendment is frozen by
   commit, the hash goes in your report.
3. Only then run the router-backed identity gate 3 times, one `gpu-wait` job each or one job
   looping 3 times, with team A's own gate script and launch config, unchanged. CPU preflight
   and `gate-dryrun` first. Every arm-defining parameter read back (TMAX, the memory-manager
   cap, both PIDs on the XTX in `rocm-smi --showpids`). Record 3 verdicts. If any run fails
   identity, P4 stays FAILED: report it, do not retry until it passes.
Kill line: any identity miss in the 3 runs, or a missing read-back. GPU budget: 45 minutes.

## Rules that bind this lane

- Every GPU launch through `gpu-wait run [--priority N] [--vram GB] [--timeout S] -- <cmd>`,
  with an explicit `PATH` (gpu-wait drops the shell env). Re-read the whiteboard head for a GPU
  hold before EVERY launch. Preflight on CPU first.
- Web search, if any: firecrawl only. Local recall: `~/iTools/bin/brain-ask`, `brain-recall`.
- Failures are loud; never edit a script while it runs; a skip is never exit 0.
- Commits: conventional subject plus a why-body, staged by explicit pathspec, never `git add .`,
  never a `Co-Authored-By` or any model attribution line. mojo-baro changes on `lane-p4b`;
  `~/Brain` and `~/iTools` are their own local-only git repos, commit there by pathspec too.
  No git remotes, ever.
- No em dashes anywhere in anything you write.
- DONE means a check that exercises the thing the way a user meets it passed, named in the same
  sentence. Otherwise write UNVERIFIED and what is missing.
- Do not touch `kernels/` or `serve/*.mojo`. If an item seems to need it, stop and report.

## Deliverable

`exchange/lane-P4B-report.md` in your worktree, committed: one section per item (C1, C3, C2, D1,
D2, B1) with what changed, the commit hash in whichever repo, and the check output. For B1 the
three verdicts and the frozen-amendment hash. End with the `./run-tests.sh` receipt line only if
you changed anything under `bench/`, `tools/` or `docs/` that ci-checks covers (run
`tools/ci-checks.sh` regardless). Reply in the pane with one line: `written to <path>`.
