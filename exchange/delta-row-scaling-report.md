# SSM sub-block row scaling at m = 1, 2, 4, 8 (fable, 2026-09-16)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-fable-delta-row-scaling.md`. Preregistration
`b7969eb`, result section in `bench/ssm-mrow-protocol.md` round 2, receipts `.work/mrow2/`,
scripts `bench/ssm-mrow-run.sh` and `bench/ssm-mrow-summarize.py`.

## One-line recommendation

**A2 + A3 as planned, no delta kernel round first; set the batching target at N = 4, not 8.**

## Why

- The decision rule as frozen: delta at m = 4 costs 2.10x its m = 1 (bound 2.6x); delta's share
  of the serialized SSM sub-block at m = 8 is 0.029 (bound 0.30). The recurrence is small
  against the sub-block at every m.
- The aggregate tokens-per-step ceiling relative to single stream, from sub-block totals per
  window: **1.56x at m = 2, 2.31x at m = 4, 1.46x at m = 8.** It peaks at 4 and falls at 8.
- What falls at m = 8 is every q4 weight GEMM: gemm4 1.63x at m = 4 then 4.53x at m = 8;
  ffn gate/up/down 1.5 to 1.9x at m = 4 then 5.8 to 5.9x at m = 8; the head 7x. The
  elementwise stages (conv, l2, gated, rmsnorm, swiglu, r_add) stay under 1.25x through
  m = 8. This is the row-parallel GEMM at the SM = 8 tile losing efficiency, not the SSM.
  The linear model's falsifier fired for exactly these stages at m = 8 (1.8 to 2.4x their
  prediction), so the preregistered m = 8 ceiling is withdrawn and the measured one stands.

## Correction to round 1

rocprofv3 device time per launch of `amar_ssm_delta_step[m]`: 10.68 / 21.48 / 41.24 / 75.48 us
at m = 1 / 2 / 4 / 8, ratios 2.01 / 3.86 / 7.07: **linear in m.** Round 1 measured 1.46x on
the host-synced stage timer, which reads 31 us per launch at m = 1 against 10.7 us of device
time; its "state columns reused across rows" explanation is withdrawn. The verdict it
supported (delta is a small share) still holds, for the reason that the kernel is tens of
microseconds against a sub-block of milliseconds. Memory `ssm-per-row-floor-measured` updated.

## Receipts and voids

- Engine sha `808243c5...` printed in the same command as the runs; `BARO_MEGA: False`,
  `BARO_SPEC`, `spec k`, `mtp: drafted D accepted A k K`, `tokens`, `prompt tokens` read back
  from every one of the 36 stdouts; windows per run from D / K. Delta spreads under 1%.
- Round 1's m = 2 row reproduced: delta 1.43x (band 1.39 to 1.53).
- Two void runs, disclosed in the protocol: the dense engine's megakernel default at m = 1
  bypassed the stage timers (the read-back list lacked `BARO_MEGA`; it now voids on it), and a
  patch script that aborted before writing. Third run valid.
- The rocprofv3 trace of the m = 1 arm hung at engine exit after flushing its CSV (867 of 1512
  launches); killed after 10 minutes on the coordinator's call, the traces for m = 2, 4, 8
  completed. Recorded as a hung step; if traces are needed again they run with a timeout.
- GPU: 36 profiled runs plus 4 traces at about 3 s each, about 3 minutes, plus the two void
  runs of the same size and the 10-minute hang.

## Full tables

In `bench/ssm-mrow-protocol.md` round 2 result (SSM stages, FFN stages, sub-block totals,
device times).
