# Megakernel, multi-row window (frozen 2026-09-06 before any timed run)

Bound by `PROTOCOL-RULES.md`. Follows `megakernel-protocol.md` (m=1 round:
646 -> 1 launch/token, +22%, default since `ed92d84`) and
`mrow-gemm-protocol.md` (window cost at m=3 is FMA-bound, not traffic;
k=2 20-prompt median **102.4 tok/s_gen**, 69% acceptance, is the champion).

## Problem on the record (measured, `BARO_PROFILE=3`, ref prompt, k=2)

Per spec window (m = k+1 = 3 rows): trunk 32 layers + head **21.3 ms** as
646 launches on the launch path; draft path (`blk32_forward` process + 1
draft step: MTP layer + 1 GB head read each) **4.6 ms**; total ~26 ms for
up to 3 tokens. Launch cost inside the trunk window: 646 x 2.43 us =
**1.57 ms** (6.0% of the window). The m=1 megakernel does not fire under
spec (`use_mega` requires m == 1).

## Shape

`amar_mega_token[MR=3]`: the same per-token kernel with every phase body
generalised to MR rows (runtime m <= MR, row guards as in
`amar_matmul_skinny_q8row[UNROLL, MR]`): GEMM phases accumulate MR rows
per wave (`QV` stays 16 for MR <= 5, so per-row arithmetic order is the
launch kernels' at any m <= 5); rmsnorm per row by every block; attention
(head, row) -> block (r*NQH + h), `T = pos + 1 + r`; ssm conv/l2/delta/
gated loop rows inside the block exactly as the launch kernels do (slot
(ring+r) -> (ring+r+1)); ffn elementwise over m*FFN; head GEMM MR rows and
per-row argmax written to `Dtok[r]` (the `argmax_d` contract), plus the
f32 final-norm rows into `hn_d` that the next window's MTP process step
reads (`amar_rmsnorm` body). The engine takes this path when
`win_spec and m == MR` (k=2 shipping config); any other m, prefill, and
the draft path keep the launch path. `BARO_MEGA=0` = parity reference.

## Prediction (frozen)

Trunk window 21.3 -> 19.7 ms (launches) minus the fixed-grid GEMM gain,
which at m=1 was 5-16% of the GEMM phases and is unmeasured at m=3
(assume 0..1.0 ms). Window 26.0 -> 23.4..24.4 ms = **+6.5% .. +11%**
tokens/s at fixed acceptance.

| stage | prediction | land rule |
|---|---|---|
| W1 kernel gate | `test_mega_block` at m=3: X, conv/ssm state, KV cache, Dtok bit-identical to the launch sequence | 0 mismatches |
| W2 engine, k=2 | 20-prompt k=2 median **102.4 -> 109..114** | `BARO_SPEC=1` GENERATED equal to `BARO_MEGA=0 BARO_SPEC=1` on 20/20 prompts AND median >= +6%, spread < 5%, baseline re-run same stint |

Stop rules: **W1** any mismatch that is not a phase-ordering slip ->
report, do not paper over. **W2** < +3% -> close, code stays on the branch
(`BARO_MEGA` keeps the m=1 path only). **S0b** the MR=3 kernel's VGPR/LDS
must keep 1 block/CU at G=96 (256 VGPR cap, `flat_work_group_size=512`);
if it spills into the GEMM loops (scratch ops near `load_b128` > the m=1
kernel's), measure before wiring.

## Not in this round

Folding the draft path (`blk32_forward`, ~4.6 ms/window, 2 x 1 GB head
reads) -- separate freeze; MR > 3; the q8dot m>=3 FFN kernel (`BARO_DOT`,
closed in `mrow-gemm-protocol.md`).

## W1/W2 receipt (2026-09-06 01:20) -- CLOSED at W2

W1 PASSES: `amar_mega_token[MR=3]` bit-identical to the launch sequence at
m=3 (X, conv windows, ssm states, KV cache, 3 argmax rows, f32 final-norm
rows), `kernels/test_mega_block.mojo` runs m=1 and m=3.

W2 FAILS the stop rule: ref prompt k=2 spec, `BARO_MEGA_WIN=1` 121.0 vs
launch 122.2 tok/s_gen (-1%); synthetic 4-layer window m=3 mega 1.10x the
launch time (m=1: 0.87x). Per-phase: the GEMM phases scale 1.5-1.6x from
m=1 to m=3 inside the persistent kernel where the native `q8row[4,3]`
scales ~1.15x (M0 receipt). Cause is occupancy, not spills: the persistent
kernel runs 1 block/CU = 2 waves/SIMD (256-VGPR budget forced by the delta
phase's 128-register column); at m=1 the loop is bandwidth-bound and does
not care, at m=3 it is FMA/latency-bound and half the native occupancy
cannot hide the load latency. Tried: runtime row loop in delta (spills
457 -> 185, no speedup); software-pipelined dot loop (spills 92/244, 4%
SLOWER at both m). Not tried (separate freeze): shrink the delta phase to
<= 128 VGPRs so G=192 (2 blocks/CU) becomes resident; that is the only
lever that changes the occupancy.

Kept: the MR-generic kernel (m=1 path unchanged, spills 84 -> 77), the m=3
kernel gate, and the engine window path behind `BARO_MEGA_WIN=1`
(default 0). `BARO_MEGA=1` stays the default for m=1 decode.
