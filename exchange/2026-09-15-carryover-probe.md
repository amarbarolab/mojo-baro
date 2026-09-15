# Carry-over probe: the 3.3% is not a clock artifact (2026-09-15)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-carryover-probe.md`. Repo
`main` @ `a063d19`, tracked `kernels/ssm.mojo` and `serve/registry.mojo`
untouched (byte-identical to `a9c0106`). Nothing committed. Preregistration:
`bench/ssm-occupancy-protocol.md`, section "Carry-over probe 2026-09-15" plus
Amendment 1, hashes in `.work/carryover/prereg.sha256` (both stamped before
the first timed run). Artifacts: `.work/carryover/` (generator, sources,
binaries, ISA receipts, run logs, analysis).

## Verdict

1. **H1 (clock or power carry-over) is falsified.** The FFN GEMVs run at the
   same shader clock in both arms, measured inside the kernels over every
   block of every launch: 2949 / 2954 / 2976 MHz (control) against 2959 /
   2968 / 2993 MHz (split), ratio 1.003 to 1.006, the split arm marginally
   higher. The sysfs reads outside the window agree: post-decode gfxclk 3020
   vs 3026 MHz, uclk 2500 both, package power 304 vs 303 W with the 290 W cap
   binding equally in both arms. Nothing the split does leaves the card in a
   different clock or power state.
2. **The cost is real cycles at equal clock, paid in kernels the change never
   touches (H2 class).** Per run (median of 5, ranges disjoint): gate GEMV
   +2.34 ms (+3.8%), down GEMV +4.15 ms (+6.2%), the rest of each layer
   (residual add, next layer's norm and attn or SSM sub-block including the
   split kernel) +5.08 ms (+2.2%); the up GEMV is flat (-0.19%, overlapping)
   and so are both launch gaps. The trunk windows account for the whole
   decode difference (+11.95 ms of windows against +10.9 ms of decode). What
   memory-side state carries it is not identified by this probe; what is
   excluded is listed below.
3. **Gate-2-style per-kernel timing cannot judge a geometry change on this
   card: no.** Gate 2 measured the split kernel 2.3% faster at MR=3 with host
   syncs around it (09-15, disjoint ranges) while the same change costs
   +12 ms per run in the stream, at identical clock, spread over kernels
   downstream of it. A sync isolates the kernel from the stream in which its
   cost is paid. The instruments that see it are the untraced end-to-end
   tok/s (which caught it) and the in-stream device timeline used here.
4. **MSPEC: not exposed to this artifact class, but its correction is
   unverified.** MSPEC step 1 read its window gap share off a rocprofv3
   timeline, which is in-stream (hardware begin/end stamps, no host syncs),
   so downstream cycle costs would appear in its kernel durations. Its
   exposure is the tracer's own 7% perturbation, corrected as a bound
   (8.4 to 9.2% from 15.65% traced). How to tell: stamp every dispatch of one
   verify window with this probe's helper (no tracer, no syncs) and read the
   untraced gap share; if it falls inside 8.4 to 9.2% the correction stands.
   Nothing in MSPEC was touched.

Standing: one prompt (`p09-explain-gpu`), 5 alternating rounds per stamped
arm, 3 per unstamped arm. These are receipts under the protocol's own gates,
not P4 20-prompt medians; they settle the hypothesis question, not a
BASELINE number.

## Instruments (both outside the forbidden zone)

**Stamps.** `.work/carryover/mkprobe.py` copies `kernels/*.mojo` and
`serve/*.mojo` into `.work/carryover/src-C` and `src-D2` (D2: `ssm.mojo` and
`registry.mojo` from `b31f7b5`, `SSM_JSPLIT = 2`) and applies one patch to
both: an uncommitted copy `amar_matmul_skinny_q4rowb_st` of the q4 row GEMV
that reads `llvm.readsteadycounter` (100 MHz REALTIME) and
`llvm.readcyclecounter` (20-bit SHADER_CYCLES, `s_getreg`) at wave entry and,
after an explicit `s_waitcnt(0)`, at wave exit. Per block, wave 0 adds its
wall span and cycle span into two per-launch sums (native
`global_atomic_add_u64`); the last block to arrive at a per-launch counter
stores the launch end; block 0 stores the launch start. Dispatched only at
the three trunk FFN sites (gate, up, down) of the launch path; every other
kernel, including the split kernel, is the tracked code. After the run the
harness prints one `STAMP` line per launch (3360 per run = 35 windows x 32
layers x 3 sites). Shader clock per launch = sum of cycles / sum of wall over
all 1536 (or 512) blocks; the single-wave block-0 ratio was noisy (2923 to
3274 MHz on one run) and is not used.

**sysfs.** The harness reads `gpu_metrics` (v1.3) and `pp_dpm_sclk` at three
points outside the decode window: before `t0`, after the prefill sync and
before `t_prefill_end` is stamped, and after the decode-end sync once
`gpu_total_s` is taken. Read cost 130 to 420 us, printed per read, charged to
prefill or outside the stopwatch. No clock read of any kind inside the window.

**Receipts** (`tools/isa-receipt.py`, `.work/carryover/isa-*.txt`):
`amar_ssm_delta_step` spill ladder MR=1..8 `81 167 235 312 380 458 526 603`
in C and Cs, `78 155 227 297 366 435 504 573` in D2 and D2s (the 09-15
numbers). Stamped GEMV at MR=3: 157/159 VGPR, 0 spills, 0 scratch against
156/158 unstamped. Binaries: C `fc976cd91f4d`, D2 `614968b64033`, Cs
`e5e95f6f9c00`, D2s `7fdd2a0a5428`, each read back per run.

## Read-back, every one of the 16 timed runs

`gpu-wait list` empty before each launch (no WARN line in
`.work/carryover/runs-all.log`); `power1_cap` 290 W; `BARO_SPEC: True`,
`spec k: 2`, `BARO_MEGA: True`, `pack q4 trunk: True`; `mega fail word: 0`;
`GENERATED` sha `c00468774758` (identical, and identical to the 09-15
identity); stamped runs: 3360 launches, 35 windows, `m` in {2, 3} only (the
MR=8 variant that spills is never dispatched), arrivals 1536/1536/512 on
every launch, 0 wrap-unsafe blocks.

## Phase R: the control reproduces (unstamped C / D2, 3 alternating rounds)

| arm | tok/s_gen median | min..max | spread | decode_s |
|---|---|---|---|---|
| C | 118.89 | 118.82..118.98 | 0.13% | 0.5299 |
| D2 | 115.96 | 115.40..116.16 | 0.66% | 0.5433 |

Recorded 09-15: C 118.90, D2 115.00. C reproduces to 0.01%. D2/C = 0.975,
inside the frozen 0.967 +- 0.010, disjoint. P0 holds; downstream may be read.

## Phase S: stamped Cs / D2s, 5 alternating rounds

| arm | tok/s_gen median | min..max | spread | decode_s |
|---|---|---|---|---|
| Cs | 116.14 | 115.40..116.33 | 0.80% | 0.5424 |
| D2s | 113.85 | 113.69..114.20 | 0.45% | 0.5534 |

D2s/Cs = 0.980 against D2/C = 0.975: within the frozen 0.010, disjoint. The
instrument preserves the arm difference.

**Deviation, stated:** Cs sits 2.3% under C, outside the 1.0% stamp-cost
gate. The preregistration said "rebuild with block-0 start / last-block end"
naming the CAS-loop min/max as the suspect; that suspect was already removed
in Amendment 1 (the first stamped build, whose smoke run is not read, cost
4.6%; the rebuilt one costs 2.3%). The remaining cost is the waited arrival
atomic at each block's exit plus the summed adds, 3.5 us per launch. I did
not rebuild a third time: the reading below depends on the instrument
perturbing both arms equally, and the ratio gate above is the evidence that
it does. A reader who wants the letter of the gate discounts the absolute
spans by 2.3% and keeps the differences.

## Decomposition (device-side, no host syncs, ms per run, median of 5)

| region | Cs | D2s | D2s minus Cs | ranges |
|---|---|---|---|---|
| gate GEMV spans | 61.56 (61.12..61.91) | 63.90 (63.70..63.97) | **+2.34 (+3.8%)** | disjoint |
| up GEMV spans | 56.88 (56.71..57.01) | 56.77 (56.59..56.81) | -0.11 (-0.2%) | overlap |
| down GEMV spans | 67.01 (66.10..68.88) | 71.16 (70.99..71.73) | **+4.15 (+6.2%)** | disjoint |
| gap gate end to up start | 4.16 | 4.12 | -0.05 | overlap |
| gap up end to down start (swiglu) | 8.84 | 8.88 | +0.04 | overlap |
| down end to next gate start (rest of layer) | 233.80 (232.69..234.89) | 238.88 (237.96..239.81) | **+5.08 (+2.2%)** | disjoint |
| window totals | 431.85 (431.21..435.68) | 443.80 (442.39..444.56) | **+11.95 (+2.8%)** | disjoint |
| decode_s | 542.4 | 553.4 | +10.9 | disjoint |

Per-launch medians (us): gate 54.84 -> 57.16, up 50.76 -> 50.68, down
60.22 -> 63.41.

Shader clock during the GEMVs (sum of cycles / sum of wall, all blocks,
median over launches then runs, MHz):

| site | Cs | D2s | ratio |
|---|---|---|---|
| gate | 2949 | 2959 | 1.0033 |
| up | 2954 | 2968 | 1.0047 |
| down | 2976 | 2993 | 1.0059 |

sysfs, median over runs (Cs / D2s): pre-decode gfxclk 3155 / 3116 MHz at
223 / 219 W; post-decode gfxclk 3020 / 3026 MHz, uclk 2500 / 2500, power
304 / 303 W (cap 290), hotspot 67 / 68 C, throttle_status identical
(0x30002 post-decode in both). The card runs at the power cap in both arms
and the SMU gives both the same clock.

## What is excluded and what is left standing

Excluded by this probe: a clock difference (in-kernel and sysfs), a memory
clock difference, a power-cap difference, launch bubbles (both gaps flat),
the split kernel's own duration (Gate 2: faster). Left standing: something
memory-side that the split kernel's execution leaves behind and that costs
the following weight-streaming kernels cycles. Its structure is the lead for
whoever picks it up, and it is not what the simple "dirty state evicted by
the next GEMV" story predicts: the first GEMV after the SSM sub-block (gate)
pays 3.8%, the identical GEMV right after it (up, same input, same weights
shape) pays nothing, and down, six kernels after the split kernel with only
its own swiglu input in between, pays the most at 6.2%. The 09-15
`BARO_PROFILE=1` attribution (11 to 14 of 20 ms in the FFN sub-block) was
directionally right and mis-split: with host syncs it charged the FFN
sub-block; the device timeline puts 6.4 ms in the three FFN GEMVs and their
gaps and 5.1 ms in the rest of the layer. No mechanism is proposed here; the
brief said settle the hypothesis, not fix it.

## Files touched (working tree, nothing committed)

- `bench/ssm-occupancy-protocol.md`: appended preregistration, Amendment 1,
  Results (append-only; earlier text untouched).
- `exchange/2026-09-15-carryover-probe.md`: this file.
- `.work/carryover/` (gitignored): `mkprobe.py`, `src-C/`, `src-D2/`,
  `engine-Cs`, `engine-D2s`, `build-engine-*.log`, `isa-*.txt`, `isa-*/`,
  `prereg.sha256`, `run.sh`, `analyze.py`, `runs/` (16 timed logs plus
  `SMOKE-Cs.log`), `runs-all.log`, `analysis-S.txt`, `probe_intrinsics.mojo`,
  `probe_waitcnt.mojo` and their binaries.
- Tracked `kernels/` and `serve/`: untouched (`git status`: only the two
  files above).
