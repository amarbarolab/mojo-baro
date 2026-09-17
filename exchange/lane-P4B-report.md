# Lane P4B report

the maintainer approved all six items 2026-09-17 ("all of them"). Worktree
`mojo-baro-lanes/p4b`, branch `lane-p4b`. `kernels/` and `serve/*.mojo` were not touched.

## C1. `gate-authoring` skill, four rules

Added a "What P4 round 2 paid for" section with the four rules and their one-line receipts:
identity PASS needs a soak/repeats behind it (unchanged P4 gate FAIL, FAIL, PASS on identical
binaries); never edit a script while its job runs (job `mu5vx2nsz9yy` exited 2 on a shifted tail,
ties to the skill's existing rule 10); exercise every poll/watch loop on CPU against a stub first
(the first `p4-gate-watch.sh` job died in its first poll and orphaned a gate run outside the
queue); `[ test ] && cmd` as a loop/function's last statement under `pipefail` kills the script
silently for one input (passed iGPU, died XTX), write `if`.

Commit: `~/Brain` `055eee26`.

Check: frontmatter intact (`name: gate-authoring` present, YAML closes cleanly), no em dash in
the file, `_index.md` line 96 still describes the skill accurately (unchanged in substance).
`~/iTools/bin/synth-gate gate-authoring` does not apply here: that gate checks the
skill-*synthesis* workflow (requires a `DECISIONS.md` beside the skill and a source transcript
argument); this is a hand-edited existing skill, not a synthesis run, so `synth-gate` correctly
has no target here (`synth-gate <skill-dir> <source...>` with only a dir prints usage and exits).

## C3. `lane-dispatch` + `mixed-pair-teams`, brief lead names its binary

Added the rule to both skills, same wording: "A brief's lead names the binary or harness it
applies to, and the dispatcher greps for it first," with the P4 receipt (the `save_state` lead
pointed at `serve/engine.mojo` while the failing arm ran `serve/spark.mojo`, 0 hits for
`save_state` there; the stale-KV lead pointed at a sliding-window branch `SWA_WIN = 0` never
takes).

Commit: `~/Brain` `8787ae55`.

Check: same as C1, both files parse, no em dash, `_index.md` entries for both skills unchanged in
substance (the new rule is inside each `SKILL.md` body, not the index one-liner).

## C2. New skill `determinism-triage`

Scaffolded with `skill-new.sh`, then written: a 7-rung ladder (which harness/binary, fresh-vs-
history discriminator, bitwise row capture, non-perturbing device checksums, matched control
device by token count AND wall time, fdinfo engine-time read-back, P(0 events | rate) before
calling anything clean), a "what does not count" section (single PASS, holder list, clean run
under a perturbing probe, mechanism claimed without measurement), and a closing note on ranking
what is left after the ladder. `_index.md` got a new row next to `gate-authoring`.

Commits: `~/Brain` `2cac511c` (SKILL.md) and `95a6c3f3` (`_index.md`), both landed by the Brain
repo's own session-end auto-commit hook while I was mid-item; content and diffs verified
unchanged from what I wrote.

Check: symlink `~/.claude/skills/.user/determinism-triage` resolves to
`~/Brain/Skills/determinism-triage` (`readlink -f` confirmed), no em dash in the file,
frontmatter parses. `synth-gate` does not apply for the same reason as C1 (no `DECISIONS.md`,
this is not a synthesis run).

## D1. New iTool `gpu-clients`

`~/iTools/gpu-telemetry/gpu-clients/gpu-clients.sh` (bash, `set -euo pipefail`): for each
`/dev/dri/renderD*` node (or `--node PATH`), prints PCI slot/id and gfx target from KFD topology
(matched by `drm_render_minor`), every process holding an fd on the node, that holder's DRM
engine-time DELTA over `--seconds N` (default 3) from `/proc/PID/fdinfo`'s `drm-engine-*` lines
(idle when the delta is all zero), and `evicted_ms` per KFD compute process. `load_ns`'s summing
logic and the KFD-node-by-gfx-target lookup are lifted from `bench/p4-soak.sh` (children not
walked, per the brief). `tool.toml` copied from `igpu-env`'s shape; `./index-gen --write`
regenerated `INDEX.md` (row at line 75) and the `bin/gpu-clients` shim, verified to exec the
script.

Commit: `~/iTools` `8c33578`.

Check, run for 3s on this desktop (no GPU job):

```
== /dev/dri/renderD128 pci=0000:03:00.0 id=1002:744C gfx_target_version=110000 gpu_id=22753 ==
  pid=5598 comm=plasma-keyboard compute=0ns gfx=749046ns
  pid=5936 comm=Xwayland idle
  pid=6313 comm=plasmashell compute=0ns gfx=8330094ns
  pid=33668 comm=ghostty compute=0ns gfx=39573405ns
  pid=431201 comm=vivaldi-bin compute=11491820ns gfx=86565880ns enc=0ns
  pid=1003511 comm=krunner idle
  pid=3794514 comm=antigravity idle
== /dev/dri/renderD129 pci=0000:18:00.0 id=1002:164E gfx_target_version=100306 gpu_id=23081 ==
  pid=430848 comm=vivaldi-bin idle
  pid=3794375 comm=antigravity idle
  pid=3794663 comm=language_server idle
```

`plasmashell` (the KDE compositor) shows a non-zero gfx delta on the XTX node (`renderD128`), as
required. Exit-0-on-no-holders was verified separately against `/dev/hwrng` (a real character
device nobody had open): `no holders`, exit 0 (6s wall, dominated by scanning `/proc/*/fd/*` on
this desktop's process count, not a defect).

## D2. `igpu-env`: says what it is not

Added the "NOT bit-reproducible" line to the header comment, the printed banner (as a `#`
comment, eval-able lines unchanged), and `tool.toml`'s summary. Added the same paragraph to
`~/Brain/OS/2026-09-17-raphael-igpu-under-rocm-and-mojo.md`.

Bug found and fixed in the same file (§8, small, same-file, not asked first): `gpu-wait` injects
`HSA_OVERRIDE_GFX_VERSION=11.0.0` (the XTX's version) into every admitted job's env, and
`igpu-env.sh`'s own gfx1036-detection `rocminfo` call never unset it, so under the queue the
override made every ROCm agent report as `gfx1100` and detection found no `gfx1036` name at all.
The check below caught this live (`--probe` failed under `gpu-wait` before the fix); fixed by
adding `-u HSA_OVERRIDE_GFX_VERSION` to the detection command.

Commits: `~/iTools` `2dd31ca` (script + tool.toml + INDEX.md), `~/Brain` `79f14180` (OS note).

Check:

```
$ igpu-env
# Raphael iGPU gfx1036: ROCr index 1 this boot, kfd 2, render node renderD129
# NOT bit-reproducible: 3 one-token deviations in about 11,000 tokens measured 2026-09-17, never gate identity on it
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

Eval-able lines intact, new line is a `#` comment.

```
$ gpu-wait run --timeout 300 --vram 1 -- env PATH=/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin HOME=$HOME igpu-env --probe
devices 1
name AMD Ryzen 7 7800X3D 8-Core Processor api hip
vadd mismatches 0 of 1024
PASS igpu-env: kernel ran on the iGPU (ROCr index 1, gfx1030 objects under the override)
```

`igpu-env --probe` PASS under `gpu-wait` (job admitted, ran, exit 0).

## B1. P4 protocol amendment

**Step 1, receipts confirmed, not rebuilt.** `exchange/lane-P0B-report.md` "Gate 1 receipt" reports
`pair-dispatch OK: 20/20 ok through the router`, `placement OK: a=10 b=10`, `identity OK: 20/20
proxied responses match the single-engine (a) baseline at T=0` at `BARO_TMAX=4096`. Verified from
the receipts on disk, not the report's words:
`mojo-baro-lanes/team-a/.work/p0b-gates12/gate1/gate1.log` and `identity.json` (20 rows, all
`"match": true`); `a.stderr`/`b.stderr` both print `limits Limits { tmax: 4096, ... }`. The gate
script (`bench/p0b-gate1-placement.sh`) sets no per-engine `ROCR_VISIBLE_DEVICES` override, so
both engines ran on this box's default ROCm pin. This IS the identity-under-split receipt; it was
not rebuilt from scratch.

**Step 2, amendment frozen before any run.** Rewrote `bench/p4-multigpu-protocol.md`: status, what
each arm proves (iGPU = pinning/wiring receipt, reported not gated; identity gate = two engine
processes on the XTX), the iGPU finding with a pointer to `exchange/lane-P4-report-round2.md` and
`[[2026-09-17-p4-igpu-transient-corruption]]`, the repeat rule (3 runs, all 3 must pass, because
this gate has passed by luck before), and that the state-move gate no longer exists (LatentOS/P1
is an exploration, ungated, blocking nothing, since `50f9d34`; confirmed the whiteboard card and
`docs/PLATFORM-PLAN.md`'s P1 section say the same: "EXPLORATION, ungated, blocks nothing").
Updated `docs/PLATFORM-PLAN.md`'s P4 gates paragraph to match (iGPU reported not gated, identity
gate two XTX engines, repeat-3x rule, 45-minute GPU budget).

**Frozen amendment commit: `mojo-baro` (branch `lane-p4b`) `3d3e362`**, committed before any of the
3 identity runs below.

**Step 3, 3 identity-gate runs.** CPU preflight (no GPU): confirmed
`team-a/.work/engine` sha256 `e4ad47f2...` matches the P0B receipt exactly, `team-a`'s built
`baro-serve`/`router` release binaries present, this worktree's `.work/engine-pack-q4/pack.bin`
sha256 `491de801...` byte-identical to team-a's copy, `~/iTools/bin/pair-dispatch` present,
`bash -n` clean on the gate script. Ran team A's own `bench/p0b-gate1-placement.sh`, unchanged,
3 times via `gpu-wait run --priority 20 --timeout 600`, pointed at team A's already-built and
sha256-verified binaries (`BARO_ENGINE`/`BARO_SERVE_BIN`/`ROUTER_BIN` env overrides), this
worktree's own pack, `COUNT=20`:

| run | receipts | engine sha256 | TMAX (both engines) | pair-dispatch | placement | identity |
|---|---|---|---|---|---|---|
| 1 | `.work/p4b/gate1-run1/` | `e4ad47f2...` (matches) | 4096 | 20/20 | a=10 b=10, rank | 20/20 match |
| 2 | `.work/p4b/gate1-run2/` | `e4ad47f2...` (matches) | 4096 | 20/20 | a=10 b=10, rank | 20/20 match |
| 3 | `.work/p4b/gate1-run3/` | `e4ad47f2...` (matches) | 4096 | 20/20 | a=10 b=10, rank | 20/20 match |

Device pin read back mid-run-2 with `rocm-smi --showpids`: two `engine` processes, `GPU(s)=1`
(`--showproductname` reports `Node ID: 1` = XTX, `Node ID: 2` = iGPU/gfx1036, so this is the KFD
node id, not the display index; confirmed also by VRAM: ~11.2 GB and ~11.25 GB each, matching the
10.7 GB/engine measured for two 9B engines at `BARO_TMAX=4096` on the XTX, and by wall time: the
whole gate, including 20 dispatched requests plus 40 identity completions, finished in well under
a minute, impossible on the 2-CU iGPU at its measured 2.8 tok/s). No stray `router`/`baro-serve`/
`engine` processes after any run (`pgrep` checked clean). GPU time used: 3 short jobs, well inside
the 45-minute budget (`gpu-wait stats --days 1` shows queue wait 0s all day).

**Verdict: 3 of 3 PASS. P4's identity gate now has the repeat-rule receipt behind it.** This does
not reverse the round-2 report's FAILED status for the iGPU arm (still not gated, still not
bit-reproducible, still reported only); it establishes the gate the amendment actually asks for.

## Suite

`bench/p4-multigpu-protocol.md` and `docs/PLATFORM-PLAN.md` (B1) are both covered by
`tools/ci-checks.sh`'s referenced-path and doc checks, so both suites ran:

- `tools/ci-checks.sh`: exit 0, `.work/p4b/ci-checks.out`, "all non-GPU checks passed" (788
  referenced paths resolve, 33 bench sources build, vendored `uregex`/`minja`/`latentos` in sync,
  kernel census 105/58/0 orphans, `docs/KERNELS.md` current).
- `./run-tests.sh`: exit 0 (no `FAIL` line, ends with `PASS` / census 105 kernels, 58 in registry,
  0 orphans, matching ci-checks' own census), `.work/p4b/run-tests.out`. Includes the spark
  attention parity (HD 64/128/256) and LatentOS mint/ingest round-trip suites, both PASS.
