# Split-K on the N=4096 phases (frozen 2026-09-06, after the q4 ALU round, before any change)

Follows `bench/q4-alu-protocol.md` (closed at 118.3 no-spec, `784caa5`). Pool named there: the three
residual-add GEMM phases with N=4096 output rows (ssm out K=4096, attn o K=4096, ffn down K=12288) =
2.35 ms of the 8.33 ms token, streaming at 0.7-0.75x the wide phases' rate because 4096 rows over
1536 wave-slots (96 blocks x 16 waves) is 2.67 rows per wave: a third of the waves run a third row
while the rest wait at the barrier.

## Design (both paths change, identity by construction)

- Work item = (row, half): 8192 items per phase, 5.33 per wave, K split in two halves of 32-blocks
  (64+64 for K=4096, 192+192 for K=12288). Item order is row-major with the two halves adjacent, so
  both halves of a row land in the same block, in waves 2k and 2k+1.
- Megakernel (`split_gemm_add`, Q4 only): half 0 writes its partial p0 to `Pk[row]` and publishes a
  per-row flag (release, tag = barrier generation + 1); half 1 waits on the flag (acquire, bounded spin
  + fail word like `grid_barrier`), clears it, and writes `X = X + (p0 + t1)`. No extra grid barrier.
- Launch path: `amar_matmul_skinny_q4rowb[UNROLL, MR, KSPLIT=2]` writes `Cp[part, r, row]`;
  `amar_skinny_reduce_add[.., NSPLIT=2]` computes `acc = (0 + p0) + p1; X = X + acc`. Same real
  expression `X + (p0 + p1)`, same lane-to-block mapping (one shared inline dot body
  `q4_dot_blocks` over a block range), so the bits agree.
- q8 packs untouched (Q4=False keeps the unsplit loop; q8 launch unchanged).

## Stages

| stage | check |
|---|---|
| K1 kernel | `test_mega_block` m=1 q4 bit-identical (its launch emulation uses the split kernels too); q8 cases unchanged |
| K2 engine | q4 pack mega == launch spec 0/1; launch vs `tools/model-ref.py` 64/64 (the numerics moved: the token stream is the judge); run-tests green |
| K3 A/B | 20 prompts, old binary (`.work/engine-a2`) vs new, one stint, clock-probe; identity 20/20; per-phase profile |

## Frozen prediction

Pool phases at 0.72 -> 0.92 of the wide rate: ssm out 583 -> 460, attn o 162 -> 130, ffn down 1600 ->
1280 us = -475 us on 8331 = **118.3 -> 125-127 no-spec (+6%)**. Costs counted against it: the flag
handshake and the second X read (negligible), the half-1 waves' spin (short: both halves start
together). Land >= 123 (+4%); close < 121 (+2%). Stop rules: any K1 mismatch = the two paths' block
mapping differs, fix the mapping, never the check; K3 spread > 5% = re-run, not a verdict.
