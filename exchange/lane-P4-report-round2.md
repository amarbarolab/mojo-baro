# Lane P4 report, round 2: the iGPU determinism failure

Status: **FAILED, UNRESOLVED.** The identity miss is real, it is not request-state bleed, and it
is not fixed. The unchanged gate is flaky on unchanged binaries: FAIL, FAIL, PASS over three
runs. No threshold was moved, the Qwen7 iGPU arm was not replaced, no engine fix is claimed.

Branch `lane-p4`, forked from `main` at `41f60f8`. Fable 5.1 lane, plus one sonnet research
helper in herdr pane `w80:p2` (the maintainer's request, mid-lane), whose two findings files are cited
below as leads, not as results.

## 1. Re-run of the same gate on a tree that contains `f9048f6`: FAILED

The brief's lead could not have applied. The iGPU arm (Qwen2.5-7B) is served by
`serve/spark.mojo`; `f9048f6` touches only `serve/engine.mojo`, and `spark.mojo` has no
`save_state`, no checkpoint chain and no restore path. The re-run was done anyway, as briefed.

Binaries rebuilt from `41f60f8` with `bench/p4-build.sh` (`.work/p4/bin/BUILD.txt`):
`engine-qwen` sha256 `33c10c93...ea55a`, `engine-dense` `76ca1990...4f35`, `baro-serve`
`3a77eccf...1066`. Gate body `.work/p4/run-two-engines.sh` byte-identical to team-b's
(sha256 `2d3c9ee4...`), binaries and the Qwen pack passed in through the body's own env overrides.
CPU preflight with gpu-wait's injected variables: PASS.

Job `mu5uslx9aoez`, exit 1, receipts `.work/p4/gate-rerun/`:

```
p01-water,xtx,PASS
p02-python-fib,igpu,PASS
p03-story,xtx,PASS
p04-list-planets,igpu,PASS
p05-math,xtx,PASS
p06-translate,igpu,PASS
p07-json,xtx,PASS
p08-sql,igpu,FAIL
```

Read-back: iGPU child PID 1012222, `ROCR_VISIBLE_DEVICES=1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`,
`HIP_VISIBLE_DEVICES` unset; XTX child PID 1013903, `ROCR_VISIBLE_DEVICES=GPU-859baafa301986cb`;
both PIDs in `rocm-smi --showpids`; runtime identity `41f60f8`. Decode 2.6 to 2.8 tok/s on the
iGPU arm is a second, independent device receipt (the XTX cannot be that slow).

The failure moved: round 1 failed p06 and passed nothing after it, this run passed p06 and
failed p08. On p08 the solo stream counts "200, 300, 400 ..." and the split stream is identical
for 41 tokens, then emits one garbage id (`110952`) inside that confident context and carries on
from it. Round 1's p06 shows the same shape with the roles swapped: p06 solo was all `15` in both
rounds, so in round 1 it was the SPLIT stream that took the hit (`57905`), not the solo.

A third run of the same body and the same binaries, watched from outside by
`bench/p4-gate-watch.sh` (the job after the crashed `mu5wpfrhvhfd`, receipts `.work/p4/gate-watch1/`):
**PASS 20/20**, gate exit 0. That pass is luck, not a fix, and is not claimed.

## 2. Discriminator: NOT state bleed

`bench/p4-discriminator.sh` drives `engine-qwen` over stdin with no baro-serve in the loop.

- Run 1 (`.work/p4/discriminator/`): 3 fresh single-request processes plus 4 history-bearing
  p06 requests. `fresh1` left the stream at generated token 1 (`220 17 15 17 18 ...` against
  `220 16 15 15 15 ...`); the other 2 fresh runs and all 4 history-bearing requests were
  identical to each other. Verdict `NONDETERMINISTIC-FRESH`: one request, fresh process, no
  history, and it still deviates.
- Run 2 (`.work/p4/discriminator2/`): 10 fresh runs, p06 after a LONGER prior request
  (p17, 59 prompt tokens, then p06 twice) and after a SHORTER one (p14, 7 tokens). 13 of 13
  streams identical. The coordinator's stale-KV lead (the comment-only invariant at
  `serve/spark.mojo:281`) is cleared: fresh = after-shorter = after-longer, and this profile has
  `SWA_WIN = 0`, so the window branch of `amar_attn_decode_swa_gated` is never taken and the
  kernel reads positions `0..T-1` only, all written by the current request
  (`kernels/attn.mojo:87-103`, out-of-range threads score `-3.4e38`).

## 3. What the defect is, as far as measured

A rare, transient, single-step corruption on the iGPU arm only. Observed 3 times in roughly
11,000 undumped tokens: gate round 1 at 10:43, then the gate re-run (p08 split written 20:25:12,
token 39 of 64) and `fresh1` (log closed 20:26:22, token 1), which puts the second and third
events about one minute apart. Zero in everything run after 20:27, about 10,000 tokens. Two of
three events inside one minute and none in the next three hours is the strongest evidence here
for a bursty external cause rather than a steady per-token fault rate. (Corrected 2026-09-17: the
first version of this report said "all three before 17:45"; I had not read the file times.)

The corruption is large, not a near-tie flip: in a clean capture of the same prompt
(`.work/p4/glitch-capture/run1/row-39.bin`) the row at the p08 glitch step has token `15` at
23.35, the runner-up at 17.09 and the emitted garbage id `110952` at -3.51, so that one logit,
or the argmax over it, was off by more than 26.

| run | engine requests | tokens | deviations |
|---|---|---|---|
| gate round 1 (team B) | 9 iGPU | 576 | 1 (p06 split) |
| gate re-run `mu5uslx9aoez` | 12 iGPU | 768 | 1 (p08 split) |
| discriminator 1 | 9 | 576 | 1 (`fresh1`, token 1) |
| discriminator 2 | 16 | 1024 | 0 |
| logits capture (dump path, extra sync per step) | 12 fresh | 768 | 0, and all 768 logits rows bit-identical across 12 processes |
| soak, one process | 40 | 2560 | 0 |
| gate run 3, watched | 30 iGPU | 1920 | 0 |
| trace soak, checksum build, one process | 60 | 3840 | 0, and all 60 checksum tables identical |

What is ruled out, with the evidence:

- **Request-state bleed**: section 2.
- **`f9048f6` / save_state**: not in the failing binary.
- **Systematic gfx1030-on-gfx1036 miscompute**: 768 logits rows bit-identical across 12 fresh
  processes (`bench/p4-glitch-capture.sh`, `tools/p4-row-diff.py`, `.work/p4/glitch-capture/REPORT.txt`).
  The kernels are deterministic when nothing goes wrong.
- **KFD queue eviction as a sufficient cause**: the soak logged `evicted_ms` per request
  (`.work/p4/soak/soak.csv`). 9.8 s of eviction in total, 8.3 s of it in the first 86 s of process
  life, request 29 carried 1277 ms; all 40 streams identical. At least about a hundred
  evict-and-restore cycles produced no corruption. The CWSR size fix the research helper found
  (rocm-systems PR 2200 class) appears present: node 2 exports `cwsr_size 700416` and
  `ctl_stack_size 4096`. No amdgpu, KFD, MCE or EDAC line in `journalctl -k` for the period.

### Graphics-load soak (added 2026-09-17 late, the maintainer: do it)

`bench/p4-soak.sh` with `P4_SOAK_LOAD_CMD=bench/p4-igpu-load.sh`: the trace build serving p08-sql
while `vkcube --gpu_number 1 --present_mode 0` (Vulkan GPU1 = RADV RAPHAEL, checked with a
negative control) and a 60 fps VAAPI scale ran on the same iGPU. Job `mu5zh83w0nht`, receipts
`.work/p4/gfx-soak1/`. Read-back from fdinfo at t=83 s: vkcube `drm-engine-gfx` 66.2 s and
`drm-engine-dma` 50.2 s, ffmpeg `drm-engine-compute` 7.7 s, so the iGPU graphics engine was about
80 percent busy beside the engine. The contention was real: decode fell from 2.8 to 1.0 tok/s,
prefill from 9 s to 25.5 s, 67 to 89 s per request.

Result: **0 deviations in 5 complete requests**; all 5 checksum tables identical (5 x 89
positions x 85 cells) and identical to the unloaded stream. The job then exited 1 by design at
t=450 s because the vkcube window went away (no coredump; most likely closed on the desktop), so
the planned 30 requests became 5. The first submission (`mu5zgb95ts2t`) failed in 1 s on my own
device check (`grep -A3` did not reach the deviceType line), fixed and re-tested on CPU with a
negative control.

Reading: 445 tokens is a small sample against a base rate of 3 in 11,000, so by token count this
proves little. By preemption count it says more: the engine shared two CUs with an unthrottled
renderer for 450 s, orders of magnitude more graphics submissions than Discord or Vivaldi could
have produced in the one minute that held two of the three events, and nothing broke. Sustained
graphics preemption does not corrupt this engine at any rate that could explain the failures.
What this arm does NOT cover is bursty light use (idle to busy transitions, clock and power
state changes), which is what desktop clients actually do.

### Control arm: the same spark engine on the XTX (added 2026-09-17 late, the maintainer: do it)

`bench/p4-build.sh` with `P4_QWEN_NATIVE=1` builds the same `serve/spark.mojo` + Qwen2.5-7B
profile + trace define for gfx1100 (`.work/p4/bin-xtx-trace/`, engine sha256 `03341a74...`), and
`bench/p4-soak.sh` with `P4_SOAK_DEVICE=xtx` pins it to `GPU-859baafa301986cb`. Same pack, same
prompt, same kernel source; only the device and the compile target differ. Read-back: 22,192 MB
mapped through the XTX render node, 67.7 tok/s decode (the iGPU does 2.8).

| run | requests | tokens | deviations | checksum tables |
|---|---|---|---|---|
| `.work/p4/xtx-soak1` | 160 | 10,240 | 0 | 160 of 160 identical |
| `.work/p4/xtx-soak2` | 640 | 40,960 | 0 | 640 of 640 identical |

The XTX token stream is also identical to the iGPU's clean stream for this prompt. At the iGPU's
observed per-token rate (3 in about 11,000) the chance of 51,200 clean tokens is about one in a
million, so **a per-execution race in the spark kernels at that rate is ruled out on gfx1100**.
The fault belongs to the iGPU rig (the gfx1036 part under the gfx1030 override, its system-memory
weights, its power states), not to the engine code that also serves llama, qwen2 and granite
models on the XTX. Limit of this control: it matches the iGPU by token count, not by wall time
(14 minutes against about 75), so it says nothing about faults that arrive per minute rather
than per kernel launch; those are rig faults by definition.

What is NOT ruled out (ranked by my own estimate, none measured):

1. **Power or clock state transitions on the iGPU** (bursty desktop use, GFXOFF exits, DPM
   switches). Consistent with two events one minute apart during desktop activity and with a
   sustained load being harmless. Not tested; a bursty load arm (1 s on, 3 s off) is the test.
   Sustained graphics preemption itself is now substantially weakened, see the soak above.
   Holding `renderD129` is not using it: at review time the only holders were Antigravity and
   its language server, with zero gfx engine time in fdinfo.
2. **Transient DDR5 read faults.** The iGPU streams about 7.6 GB of weights from system memory
   per token; the XTX arm never touches DIMMs for weights and never fails. No way to test this
   from the lane without a reboot (memtest) and no EDAC on this board.
3. **A race that only the gfx1030 code object or the 2-CU timing exposes.** The XTX control
   clears the kernels as written and as compiled for gfx1100; it cannot clear the gfx1030 build
   on the gfx1036 part. Still the least likely of the three: 768 bit-identical rows and 60 plus 5
   identical checksum tables on the iGPU itself.

## 4. Fix

None. There is nothing I can honestly fix yet: the fault has not been placed in a kernel, and
two of the three live candidates are outside the engine.

## 5. What landed for the next round

- `bench/p4-build.sh`: the three gate binaries from the current tree. The spark harness needs
  `-I serve` and must be built under `igpu-env` (first attempt failed on exactly that, on CPU).
- `bench/p4-discriminator.sh`, `bench/p4-glitch-capture.sh` + `tools/p4-row-diff.py`,
  `bench/p4-soak.sh` (KFD `evicted_ms` polling, arm env passthrough with render-node read-back,
  analyze-only mode), `bench/p4-gate-watch.sh` (runs the UNCHANGED gate body as a child and
  records both engines' eviction time at 10 Hz; verdict and exit code pass through).
- `-D BARO_TRACE_SUM=1` in `serve/spark.mojo` + `amar_trace_sum` in `kernels/spark_kernels.mojo`
  + `tools/p4-trace-diff.py`: one FNV checksum per (position, layer, stage) and per logits row,
  written on the device with no host sync, dumped per request. The first differing cell of a
  deviant request names the kernel group and the layer. This is the instrument that turns the
  next caught glitch into a location; it costs 85 tiny launches per token.
  First run (`.work/p4/trace-soak1/`, 60 requests of p08-sql in one process): 60 of 60 tables
  identical over 89 positions x 85 cells, token stream identical to the untraced engine's, so
  the trace build is identity-checked against the untraced build on that prompt and it caught
  nothing because nothing happened. The DEFAULT build of the edited `spark.mojo` (trace off)
  compiles (`.work/p4/bin-default-check/`) but has NOT been run on the GPU since the edit:
  UNVERIFIED, re-run the gate before merging this branch anywhere.

## 6. Exactly what is unresolved

1. The cause. Next action: the gfx-load soak above on the trace build, then, if it reproduces,
   read the first differing cell. If the cause is gfx preemption or DIMMs, the honest outcome
   for P4 is that the Raphael iGPU under the gfx1030 override is not a bit-reproducible device
   while the desktop uses it, and the protocol's iGPU arm needs either an idle iGPU (no
   graphics clients on `renderD129`) recorded as a preflight read-back, or a different second
   device. That is a protocol decision for the maintainer, not something this lane may decide by itself.
2. The gate as written cannot distinguish "fixed" from "lucky": it passed 1 of 3 on identical
   binaries. Any future PASS claim for P4 needs repeated runs, or the trace soak, behind it.
3. The router gate stays `BLOCKED: P0b` and the state gate was not run, as in round 1.

## 7. GPU budget and my own failures

8 GPU jobs from this lane. 1 real gate verdict (exit 1). 2 failures were mine and both were
CPU-discoverable: the first soak job exited 2 because I edited `p4-soak.sh` while it ran (data
complete, report recomputed with `P4_SOAK_ANALYZE=1`); the first gate-watch job exited 2 in its
first poll because `sed` on a not-yet-created log returned 2 under `pipefail`, and it orphaned a
gate run outside the queue, which I stopped by hand. The watcher now guards the missing file,
kills its child on exit, and was exercised on CPU against a stub gate body before the re-submit.
Machine-wide for the day: `gpu-wait stats --days 1` = 278 jobs, 180 ok, 85 failed, 13 cancelled,
421 min busy.
