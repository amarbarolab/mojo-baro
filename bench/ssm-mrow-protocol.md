# SSM per-row floor probe — frozen before the first timed run

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 (read-back
> receipts), P4 (decode verdicts are 20-prompt medians), P5 (row scaling),
> P6 (harness before kernel).

Milestone item 1 of `~/Brain/mojo-baro/2026-09-05-engine-next-milestone.md`.

## Question

Of the SSM sub-block's per-window cost at m=2, how much is sequential by
construction? `amar_ssm_delta_step[MR]` carries `for r in range(MR)` inside
one kernel (`kernels/ssm.mojo:193`): the state update of row r+1 reads the
state row r wrote, so its m=2 cost cannot fall below 2x its m=1 cost without
changing the recurrence. Every other stage of the sub-block (the four q8
GEMMs + two reduces, `rgates`, `conv`, `l2`, `gated`, the out GEMM + add) is
row-parallel and should be far closer to 1x.

The answer sets the ceiling for milestone item 2 (multi-row window
1.28x -> 1.1x): the sequential part is a floor no window rework can remove.

## Instrument

`serve/engine.mojo` `BARO_PROFILE=2`, which already synchronizes at the seven
SSM stage boundaries and prints `ssm-kernel: <name> <s> <share>` for
`gemm4+reduce2 / rgates / conv / l2 / delta / gated / out_gemm+add`. No new
kernel and no new bench binary (P6): the split needed for this question
exists.

`BARO_PROFILE=2` inserts a `ctx.synchronize()` between stages, so the stage
times are serialized and their SUM exceeds the unsynchronized sub-block time.
This probe compares stages **within one profiled arm** and compares the SAME
stage across m, both under identical synchronization. Absolute per-stage
milliseconds from this probe may not be added into any end-to-end budget, and
no `tok/s` figure from a `BARO_PROFILE=2` run may be quoted anywhere.

## Arms

Both arms: same binary, same pack, same prompt, `BARO_PROFILE=2`.

- **A (m=1)**: `BARO_SPEC=0`. Every window is one row.
- **B (m=2)**: `BARO_SPEC=1`, `BARO_SPEC_K=1` -> `m = kcfg + 1 = 2`.

Prompt-phase windows also run at m>1; the arms are compared on the decode
phase only, which for this prompt dominates the window count.

## Receipts required before either arm's numbers are recorded (P1)

Read back from the run's own stdout, not from the command line:
`pack loaded in`, `prompt tokens:`, `tokens:`, `spec k:` (arm B must print
1, arm A must not report spec windows), and the engine binary rebuilt in the
same command as the run. Recorded alongside: `gpu-wait gpu` VRAM/power before
the run, and the fact that the CoiOS bge-m3 embedding server is resident
(~3 GB, idle) — it is a deviation from `ssm-occupancy-protocol.md`'s
"GPU exclusive" instrument and is disclosed rather than hidden.

3 repeats per arm, drop the first, median of 2 reported per stage; spread
gate <5% on the delta stage (the quantity under test). A wider spread voids
the arm.

## Predictions — frozen at commit time, before any run

1. `delta` ratio B/A >= 1.8. This is the mechanism claim; the loop is
   sequential and each row's arithmetic is identical.
2. Every row-parallel stage's B/A ratio <= 1.5, and `rgates` + `l2`
   specifically <= 1.2 (their grids are m-independent or nearly so).
3. `delta` share of the profiled sub-block rises from arm A to arm B.

## Decision rule — what the answer buys

Let `d` = delta B/A ratio and `S_delta` = delta's share of the m=2 profiled
sub-block.

- **Sequential-dominated**: `S_delta >= 0.35` and `d >= 1.8` -> the +1.06 ms
  at m=2 is mostly floor; milestone item 2's ceiling is **~105 tok/s**, and
  item 2 is rescoped to the non-delta stages only.
- **Recoverable**: `S_delta <= 0.20` -> the loss is outside the state update;
  item 2's ceiling stands at **115-120 tok/s** and it proceeds as written.
- **Mixed** (anything between): item 2 proceeds but its predicted band is
  restated as 105-115 before it is run, not after.

Prediction on the outcome itself, so it is falsifiable: **mixed**, with
`S_delta` between 0.20 and 0.35 and `d` near 2.0.

## What this probe cannot answer

It does not measure whether a different recurrence formulation (chunked scan,
associative scan over rows) would beat the sequential floor. That is a
separate item; this one only prices the floor as the kernel is written.
