# qwen35moe decode, round 2: launches and expert kernels (preregistered 2026-09-15, before any run)

Base: `main` after `4209cc4` (R1 to R3 landed, 93.46 tok/s_gen 20-prompt
median, `bench/moe-perf-protocol.md`). Bar: llama.cpp 109.4.

## Receipt: the token timeline at HEAD (rocprofv3, p09, 43 decode tokens, `.work/moe-perf/timeline-head.txt`)

Per token, medians: **1216 launches**, wall 12.5 ms (traced; 10.7 untraced),
kernel spans 8.5 ms, **gaps 4.0 ms (32% of wall)**, inter-dispatch gap
median 3.1 us. Kernel classes: q8_0 projections 2.3 ms over 160 calls (the
17.8 MB matrix at 18.7 us = 950 GB/s, near the floor); expert gate+up
1.13 ms / 40 calls (28 us for 9.4 MB = 335 GB/s); expert down 1.10 ms / 37
calls (30 us for 4.7 MB = 157 GB/s); rmsnorm 0.37 ms / 81 calls (4.6 us
each, launch-bound); delta 0.33, reduce_gates 0.22, router 0.19, sigmoid
gate 0.15; **`fillBufferAligned` 60 calls and `copyBuffer` 40 calls per
token (0.26 ms plus their gaps)**, host-enqueued memsets and copies in the
MoE layer path; head 0.6 ms.

## Rounds, in order, each its own commit with the R1-R3 gates

Gates for every round (unchanged from `bench/moe-perf-protocol.md`):
`test_moe_block` parity; 20-prompt agreement mean within 52.70..53.70;
`run-tests` and `ci-checks` green; 20-prompt tok/s A/B against the previous
round with clocks read back; kill line: median below +5% is a no-op and is
reverted (lower than round 1's +10% because these are smaller levers).
GPU per gate under 10 minutes.

- **R4, launches from the host (S, `serve/window.mojo`, delegated to the
  opus lane which owns that file):** remove the per-layer `enqueue_memset`
  and `enqueue_copy` in the MoE FFN path (60 + 40 per token) by having the
  consuming kernels write full outputs (no zero-init) and by pointing at
  buffers instead of copying. Prediction: launches 1216 -> about 1116,
  wall -0.5 to -0.6 ms, **+5 to +6%**.
- **R5, expert down kernel (M, fable):** `amar_moe_down_q4k` runs one
  output column per wave over 8 experts x K=512 (2 super-blocks), so 16 of
  32 lanes idle per block-dot and the 8 expert dots serialize per wave.
  Change: two experts per pass on the two lane halves, 4 passes per column,
  same per-element arithmetic. Prediction: down 30 -> under 18 us per call,
  **+6 to +9%**. The gate+up kernel gets the same look (28 us at 335 GB/s;
  one row per wave over K=2048 = 8 super-blocks, 2 passes) only if a
  measured cause is found; not promised here.
- **R6, MoE persistent token kernel (L, fable):** the MoE profile runs the
  per-kernel launch path (`MEGA_ALLOWED` off); build the MoE token kernel on
  `kernels/mega.mojo`'s phases plus expert phases (router, gathered gate/up,
  down, shared expert), R1/R2 dots inside. Prediction: launches 1216 -> 1,
  gaps 4.0 -> under 0.5 ms, **10.7 -> about 7.5 ms per token, about 130
  tok/s**, past llama.cpp's 109.4. Fingerprint read before the number
  (`isa-loops`); residency ceiling from the ISA receipt before launch
  (`persistent-kernel-gfx11`).

Falsifier for the whole round: if R4 lands and the wall does not move by at
least the removed launches x 3 us, the gap accounting is wrong and R6 is not
opened on this evidence.

### R5 result (2026-09-15): below the kill line, reverted; mechanism confirmed, prediction wrong

Two experts per pass on the two lane halves (`q4k_dot_pair`, shared
`q4k_block_partial`). Gate 1 parity PASS; gate 2 agreement **53.35/64**;
gate 4: 20-prompt tok/s **94.43 -> 98.55 (1.0436x)**, ranges 92.81..95.00 vs
98.32..98.86, sclk med 3267 MHz, GENERATED equal 17/20. Kernel receipt
(rocprofv3, `.work/moe-perf/trace-r5`): **down 30.5 -> 17.7 us per call**,
exactly the "under 18" predicted; gate+up unchanged 29.0 -> 29.8. The
token-level prediction (+6 to +9%) was wrong by arithmetic: 12.8 us x 37
calls = 0.47 ms of 10.7 = 4.4%. Under the frozen +5% line the arm does not
land; patch kept at `.work/moe-perf/r5-down-pair.patch`. It re-enters only
as one combined arm with a gate+up change (28 us at 335 GB/s, K = 2048, one
row per wave) under a new frozen prediction stated in kernel microseconds
and in token percent from the launch counts, not a guessed range.

### R5b, combined expert arm (preregistered 2026-09-15, before any run)

Change: (1) the R5 down pair (`.work/moe-perf/r5-down-pair.patch`, down
30.5 -> 17.7 us measured); (2) gate+up: one wave computes its gate row and
its up row in a single loop with all four 16-byte loads per lane (gate block
b, up block b, gate b+4, up b+4) issued before any arithmetic, instead of two
sequential dots each with dependent iterations. Same per-element arithmetic
and bf16 rounding; accumulation order within a row unchanged (blocks still
summed in lane order).
Predictions, frozen: gate+up **29.8 -> under 20 us per call** (40 calls:
at least 0.39 ms per token), down 17.7 us (37 calls: 0.47 ms); token 10.7 ->
under 9.85 ms, **+8 to +10%** on the 20-prompt median against
`.work/moe-perf/engine-head`. Kill line +5% as for the round. Gates as for
the round (parity, agreement band, tok/s A/B with clocks, run-tests,
ci-checks). Falsifier: gate+up not under 22 us in the trace means the loads
were already overlapped and the cause is elsewhere (the receipt decides,
not the tok/s).
