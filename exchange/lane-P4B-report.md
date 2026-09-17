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

## B1. P4 protocol amendment (round 1, 2026-09-17, then corrected 2026-09-18)

**Round 1 (this section's original text, kept for the record): WRONG.** I cited team A's
`bench/p0b-gate1-placement.sh` (5 prompts x 4 repeats of short text answers, about 70 tokens
compared per run) as "the identity-under-split receipt" and ran it 3 times, all PASS
(`.work/p4b/gate1-run{1,2,3}/`). Driver review 2026-09-18 caught it: that is P0b's PLACEMENT gate
with a small identity check bolted on, right for P0b, not a stand-in for P4's own protocol, which
specifies 20 prompts x 64 tokens (`bench/mtp-prompts/p*.tokens`), about 35x more compared tokens.
Three passes of the thin check did not put the repeat-rule receipt behind P4; those 3 runs are
demoted to PLACEMENT receipts only (`a=10 b=10` stands as P0b's own claim, nothing else does).

**Round 2, the real identity gate.**

1. Amended `bench/p4-multigpu-protocol.md` and `docs/PLATFORM-PLAN.md` again: named the round-1
   error explicitly, demoted the 3 placement-gate runs, and specified the real gate (20
   `bench/mtp-prompts/p*.tokens` prompts, token ids, `max_tokens` 64, `temperature` 0, `spec`
   false, through the router, `cmp` on token ids, placement-spread reverse arm). **Committed
   before any run: `4d08305`.**
2. Wrote `bench/p4-router-identity.sh`. Forks (does not source) the launch half of
   `bench/p0b-gate1-placement.sh` (`start_engine`, router bring-up, health wait): that script is
   linear, its own calls to `start_engine` and its weaker identity check run immediately after the
   function definitions, and its `EXIT` trap kills both engines the moment it finishes, so sourcing
   it would either re-run its own check first or require restructuring team A's file, out of scope
   here. The `payload`/`tokens`/`request` functions are lifted unchanged from
   `.work/p4/run-two-engines.sh` in the main checkout (team B).
3. CPU-only checks before any GPU run: `--selftest` (negative control: identical token files
   PASS; a one-id difference correctly FAILs with exit 1), `P4_CPU_PREFLIGHT=1` (binaries, pack
   hash `491de801...` matching, tools, 20 prompt files), `gate-dryrun` (stops at `GATE_DRYRUN=1`
   before the first real engine launch, arm file read back `tmax_expected=4096`, PASS in 0s).
4. **Round-1-of-the-real-gate FAILED, live, and the reverse arm is why:** job `mu6417v81osh`,
   20/20 token-id identity PASS, but `placement OK: a=20 b=0` failed the assertion
   ("one engine served zero of the 20"). Cause: my dispatch loop sent solo-then-router requests
   sequentially per prompt, so both engines sat at `pending=0` at every routing decision and
   `choose()`'s tie-break (`pending`, then engine id, `"a" < "b"`) picked `a` every time. Not a
   router bug: team A's own gate uses `pair-dispatch --mode parallel` specifically to create real
   concurrent pending load. Fixed by firing all 20 router requests in the background and waiting
   on them (solo baselines against engine a stay sequential, no router involved). Fix committed
   (`7db134d`) before re-running, same discipline as the frozen amendment itself.
5. **3 real runs, all PASS**, `gpu-wait run --priority 20 --timeout 600`, team A's already-built
   and sha256-verified binaries (`engine` sha256 `e4ad47f2...`, matching the P0B receipt exactly),
   this worktree's own pack (sha256 `491de801...`, byte-identical to team A's copy):

| run | receipts | identity | placement | device read-back |
|---|---|---|---|---|
| 1 | `.work/p4b/router-identity-run1/` | 20/20 token-id match | a=10 b=10 | both engine PIDs in `rocm-smi-showpids.log`; `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`, `BARO_TMAX=4096` in `device-readback.txt` |
| 2 | `.work/p4b/router-identity-run2/` | 20/20 token-id match | a=10 b=10 | same, both files present |
| 3 | `.work/p4b/router-identity-run3/` | 20/20 token-id match | a=10 b=10 | same, both files present |

Device pin: `rocm-smi-showpids.log` (a real file on disk this time, not prose) shows both engines'
GPU-touching child PIDs (found via `pgrep -P` on the tracked `baro-serve` PID, matched by
`/proc/PID/comm` == `engine`) under `GPU(s)=1`; `rocm-smi --showproductname` reports `Node ID: 1` =
XTX, `Node ID: 2` = iGPU/gfx1036, so this is the KFD node id, not the display index, confirmed also
by VRAM (~11.2-11.26 GB each, matching the 10.7 GB/engine measured for two 9B engines at
`BARO_TMAX=4096` on the XTX). `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` and `BARO_TMAX`
read from each engine child's own `/proc/PID/environ`, not the launch command line: `10` and
`4096` on both engines, all 3 runs. No stray `router`/`baro-serve`/`engine` processes after any run
(`pgrep` checked clean). GPU time used: 4 short jobs (the caught FAIL plus 3 real PASSes), well
inside the 30-minute follow-up budget (`gpu-wait stats --days 1`: queue wait 0s all day).

**Verdict: 3 of 3 PASS on the actual protocol gate** (20 prompts, 64 tokens each, token ids, real
placement spread, device pin and memory-manager cap read from `/proc/PID/environ` files on disk).
This does not reverse the round-2 report's FAILED status for the iGPU arm (still not gated, still
not bit-reproducible, still reported only); it is the repeat-rule receipt the amendment actually
asks for, this time against the fixture the protocol specifies.

## Suite

Both re-run after the B1 follow-up's changes (new `bench/p4-router-identity.sh`, amended
`bench/p4-multigpu-protocol.md` and `docs/PLATFORM-PLAN.md`), all covered by `tools/ci-checks.sh`'s
referenced-path and build checks:

- `tools/ci-checks.sh`: exit 0, `.work/p4b/ci-checks2.out`, "all non-GPU checks passed" (792
  referenced paths resolve, up from 788, the new script's path among them; 33 bench sources build;
  vendored `uregex`/`minja`/`latentos` in sync; kernel census 105/58/0 orphans; `docs/KERNELS.md`
  current).
- `./run-tests.sh`: exit 0 (no `FAIL` line, ends with `PASS` / census 105 kernels, 58 in registry,
  0 orphans, matching ci-checks' own census), `.work/p4b/run-tests2.out`. Includes the spark
  attention parity (HD 64/128/256) and LatentOS mint/ingest round-trip suites, both PASS.
