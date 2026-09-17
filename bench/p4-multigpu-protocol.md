# P4 multi-GPU protocol

Status: AMENDED 2026-09-17 (round 2, lane P4B), frozen by commit before any identity run. The
iGPU arm stays as the pinning-and-wiring receipt, reported, never gated. The identity gate runs
as two engine processes on the XTX. The state-move gate does not exist. This supersedes the
"wiring preflight, timed gate pending" status below it replaces; the effective-device preflight,
fixture manifest, and receipt discipline are unchanged.

## Scope

P4 tests process-level data parallel wiring, not cross-model quality or throughput. The discrete
arm uses the dense q4 fixture and the iGPU arm uses self-describing Qwen2.5-7B-Instruct BARO
`b4271cd`. Each arm is compared only with its own solo reference. The iGPU is a functional harness
device, not a performance arm, and (see "The iGPU finding" below) not an identity-gated one.

The fixture manifest is `bench/p4-fixtures/manifest.tsv`. It records model and pack paths and
hashes, but does not copy model bytes. The 20 prompts are the existing
`bench/mtp-prompts/p*.tokens` set, whose sorted-file aggregate hash is recorded there.
The Qwen7 pack is extracted from the GGUF's embedded `engine-pack.py` and `--dense` metadata,
matching the `tools/baro serve` cache builder. The import receipt reports min 96.9%, mean 99.4%
on its 20-prompt gate and pack index hash `e2ef587f...`; the generated local pack must reproduce
that index hash before a GPU run.

## What each arm proves

- **iGPU arm** (Qwen2.5-7B via `serve/spark.mojo`, gfx1030 code objects loaded under
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`): proves the harness can pin, launch and read back a second,
  physically distinct GPU process (`ROCR_VISIBLE_DEVICES` set, `HIP_VISIBLE_DEVICES` unset), and
  that its kernels are deterministic when nothing goes wrong (768 logits rows bit-identical across
  12 fresh processes; 60 of 60, then 5 of 5, per-request checksum tables identical). It is
  REPORTED, not gated: see "The iGPU finding" below for why.
- **Identity gate**: two real engine processes on the same XTX at `BARO_TMAX=4096`, 20 prompts
  dispatched by the router, each response compared against a single-engine baseline at T=0. This
  is the only gated claim P4 makes, and the receipt for it already exists (see "Gate 1" below):
  it does not exercise a second physical device.

## The iGPU finding (why it moved out of the gate)

Full report: `exchange/lane-P4-report-round2.md`; summary: `[[2026-09-17-p4-igpu-transient-corruption]]`
(Brain, mojo-baro). The unchanged gate on identical iGPU binaries went FAIL, FAIL, PASS over three
runs: 3 one-token deviations in about 11,000 iGPU tokens, each one wrong logit off by more than 26
followed by the stream continuing from it. Ruled out with receipts: request-state bleed (a fresh
single-request process still deviated; history-bearing requests did not), `f9048f6`/save_state
(not in the failing binary, which is `spark.mojo`, not `engine.mojo`), systematic
gfx1030-on-gfx1036 miscompute (768 logits rows bit-identical across 12 fresh processes when
nothing went wrong), and KFD queue eviction as a sufficient cause (about 100 evict/restore cycles
produced no corruption). The same kernels, same source, same pack, built for gfx1100 and pinned to
the XTX: 51,200 tokens, 0 deviations. Cause not placed; open candidates (untested or only
partially tested) are power/clock state transitions under bursty desktop use and transient DDR5
read faults, both outside the engine, not a per-execution race in the kernels. A rig that lies
about 1 in 3,700 tokens hides a regression; never gate identity on this device.

## Repeat rule

The identity gate has passed by luck before: FAIL, FAIL, PASS on the unchanged iGPU gate, same
binaries, 2026-09-17 (a single PASS is luck, not evidence, per the finding above). Any PASS claim
for P4's identity gate is run 3 times on identical binaries; all 3 must pass. A miss in any of the
3 keeps P4 FAILED; it is not retried until it passes.

## State movement: no longer a P4 gate

The state-move gate (moving state XTX to iGPU and back through P1, comparing ids with the
single-node arm) does not exist. LatentOS (P1) is an exploration, ungated, blocking nothing, since
`50f9d34` (2026-09-17, "LatentOS rules removed, P1 becomes an exploration"); this is the same rule
dropped for LatentOS everywhere else (`docs/PLATFORM-PLAN.md`'s P1 section: "EXPLORATION, ungated,
blocks nothing"; whiteboard card, same wording). P4's kill line is the identity gate alone.

## Effective-device preflight

Before any prompt is sent, the receipt must contain all of:

1. `igpu-env --probe` exit 0 and its printed `ROCR_VISIBLE_DEVICES`, unset
   `HIP_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION=10.3.0` (iGPU wiring receipt only).
2. `rocminfo` supplies the gfx1100 `Uuid: GPU-*` selector for the XTX; the bare hex
   `rocm-smi --showuniqueid` value is retained as a suffix cross-check. `rocm-smi --showpids`
   covers every running engine process, with PID-to-device mapping (both engine PIDs on the XTX,
   for the identity gate).
3. For the iGPU wiring receipt: the process's engine log showing the expected HIP API path, plus
   external `rocminfo`/`rocm-smi` output naming the gfx1030 object and mapping the running PID to
   it. MAX's `DeviceContext.name()` is not accepted as the device identity: under this override it
   reports the host CPU name even while `ctx.api()` reports HIP. `ROCR_VISIBLE_DEVICES=1` alone
   is insufficient because MAX lists the Raphael device as gfx1030 objects under this override
   and does not list gfx1036. The misleading engine self-name is retained in the receipt.
4. The per-process environment read-back from the Rust spawn seam, including the server port,
   selector, and pack path, and (for the identity gate) `BARO_TMAX` as printed in each engine's
   own ready line.

Any missing or contradictory read-back stops the gate before prompts.

## Gate 1: identity, two engine processes on the XTX

The receipt for this gate already exists and is not rebuilt: `exchange/lane-P0B-report.md`
("Gate 1 receipt", team A), receipts on disk at
`mojo-baro-lanes/team-a/.work/p0b-gates12/gate1/` (`gate1.log`, `identity.json`,
`a.stderr`/`b.stderr`). It reports, at `BARO_TMAX=4096`, `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`
(two 9B `baro-serve` processes at 10.7 GB each on the XTX): `pair-dispatch OK: 20/20 ok through
the router`, `placement OK: a=10 b=10`, `identity OK: 20/20 proxied responses match the
single-engine (a) baseline at T=0`. Both engine processes ran on the XTX under this box's default
ROCm pin (no per-engine `ROCR_VISIBLE_DEVICES` override in the gate script); `a.stderr`/`b.stderr`
show `limits Limits { tmax: 4096, ... }` for both. This is the identity-under-split receipt the
repeat rule above applies to: run team A's own gate script and launch config, unchanged, 2 more
times, for 3 total.

The timed invocation is from the lane root, using an absolute script path and an explicit
runtime `PATH`. `gpu-wait` admits jobs with a clean environment and may not preserve the caller's
cwd, so the short relative form is not a valid submission receipt:

```text
cd $HOME/Projects/mojo/mojo-baro-lanes/team-a
$HOME/.local/bin/gpu-wait run --priority 20 --timeout 900 -- env PATH=/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin $HOME/Projects/mojo/mojo-baro-lanes/team-a/bench/p0b-gates12-live.sh
```

The full harness is one serialized GPU job per run. No nested `gpu-wait` is allowed inside the
admitted job. The iGPU wiring receipt, when it is (re-)collected, is included in the queue like
any other GPU job; it is never exempt.

## Kill line and receipts

Kill the item on any identity miss (in any of the 3 repeat runs), wrong device pin, missing
effective-parameter read-back, or stale/missing fixture hash. A gate that lacks its dependency
remains explicitly blocked and cannot be replaced by a weaker claim.

The command, exit code, preflight, device mapping, and all 3 verdicts land in
`exchange/lane-P4B-report.md`. The CPU dry-run must stop before the first GPU step and is recorded
before each queued invocation.
