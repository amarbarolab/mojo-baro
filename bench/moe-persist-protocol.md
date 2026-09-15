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
