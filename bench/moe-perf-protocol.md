# qwen35moe decode performance round (preregistered 2026-09-15, before any kernel change)

Bar: llama.cpp RegesCore-35B Q4_K_S decode 109.445 tok/s, 20-prompt median
(`exchange/lane-MOE-BASE-report.md`). Ours today: 42.66 served, 40.0 in
`BARO_PROFILE=1` on p09 (`.work/m5/moe-prof1.log`), engine `-D BARO_MODEL=qwen35moe`
at `97059f0`, pack `.work/moe-w1/pack` rebuilt from the GGUF this session
(733 tensors, 21,005,191,680 bytes, same size as W1's receipt).

## Receipt: where the 22 ms per token go

rocprofv3 kernel trace of one decode (p09, prefill 19 + 63 tokens, tracer on;
`.work/moe-perf/trace/moe_results.db`, summary `.work/moe-perf/kernels-per-call.txt`):

| kernel | per call | calls per token | ms per token | bytes per call | rate |
|---|---|---|---|---|---|
| `moe_gate_up_q4k_pack` (8 experts x 512 rows, gate and up) | 132 us | 40 | 5.3 | 9.4 MB | 71 GB/s |
| `moe_matmul_q8_0_m1` (ssm/attn projections, three variants) | 34 to 46 us | ~160 | 6.2 | 9 to 18 MB | ~320 GB/s |
| `amar_moe_down_q4k` (8 experts, 2048 rows, K=512) | 83 us | 40 | 3.3 | 4.7 MB | 57 GB/s |
| `amar_moe_router_top8` (grid 1, 256 threads) | 73 us | 40 | 2.9 | 2 MB f32 logits already computed | latency |
| `moe_gate_up_q8_0` shared expert + `amar_moe_sig_gate` (grid 1) | 28 + 19 us | 40 | 1.9 | 3.3 MB | |
| everything else (norms, delta, gates, adds) | | | ~1.5 | | |

Bytes per token from the pack index: experts 0.78 GB, ssm/attn projections
1.36 GB, head 0.54 GB, about 2.7 GB; at the q8 row kernel's 855 GB/s that is
3.2 ms, so the floor is near 300 tok/s and llama.cpp's 109 is not a ceiling.

Cause, read from the kernel bodies (`kernels/moe.mojo`): `q8_0_row_dot`,
`q4k_row_dot` and `q6k_row_dot` stride k by lane and call a per-element
`*_value` that loads the block scale(s) and ONE quant byte per lane per
iteration, so a wave moves 32 bytes per load instruction and re-reads scales
per element. The dense path's `q4_dot_blocks` loads 16 bytes per lane per
block and reads each scale once. The router top-8 is a single 256-thread block
doing a serial selection; the sigmoid gate is a single wave.

## Arms and order (each its own commit, each gated before the next)

- **R1 experts, q4_k block dot.** New `q4k_dot_blocks` sharing the
  per-element arithmetic of `q4k_value` exactly (same `d*sc*q - dm*mn`, same
  bf16 round-trip per element) but loading 16 quant bytes per lane and the
  144-byte super-block header once per block; `moe_gate_up_q4k_pack` and
  `amar_moe_down_q4k` call it. `amar_moe_down_q6k` (3 layers) same treatment
  or left as is, stated. Prediction: gate+up 132 -> under 30 us, down 83 ->
  under 20 us per call; token 22 -> about 15 ms.
- **R2 projections, q8_0 block dot.** `q8_0_row_dot` loads 16 quant bytes per
  lane per half-block and the f16 scale once per block, same per-element
  bf16 rounding. Prediction: 34 to 46 -> 12 to 20 us per call (near the
  dense q8 row kernel's rate for the same bytes); token about 15 -> 10 ms.
- **R3 latency kernels.** Router top-8 as a wave-parallel selection (8 rounds
  of warp argmax over 256 logits, or a block-wide bitonic top-k); sigmoid gate
  folded into the router launch. Prediction: 73 + 19 -> under 10 us per
  layer; token about 10 -> 7 ms, i.e. about 140 tok/s.

## Gates, every arm

1. `kernels/test_moe_block.mojo` parity vs `tools/moe-ref.py` (fixtures
   regenerated this session into `.work/gguf/`): expert ids exact 8/8, routed,
   shared and y rel < 5e-3, the frozen bar from W3 gate 1.
2. 20-prompt teacher-forced agreement vs llama.cpp (`bench/moe-gate2-force.sh`
   shape, reference ids from llama-server on the same GGUF): mean within
   +-0.5 of 53.20 (P14 bar; accumulation order may move single tokens,
   dequant math may not). Below 52.70 the arm is reverted.
3. `./run-tests.sh` and `tools/ci-checks.sh` green; `isa-receipt` census of
   the changed kernels recorded (VGPR, scratch) as a receipt, not a target.
4. tok/s: 20-prompt median (P4), `bench/ab-prompts.sh` base vs arm with
   `AB_ENGINE_B`, both `-D BARO_MODEL=qwen35moe`, `BARO_MEGA=0 BARO_SPEC=0`,
   clock and power cap read back, identity per prompt (GENERATED equal is
   NOT required here, gate 2 is the identity: record equal/unequal per prompt).
5. Kill line per arm: median below +10% over its base is a no-op and is
   reverted with the numbers in this file.

Fixed for the round: no change to the router math, the top-8 renormalisation,
the shared-expert gate formula, or the pack format. Prefill (m>1) keeps its
path; only m=1 decode kernels change.
