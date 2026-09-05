# Megakernel round (frozen 2026-09-05; engine 68.8 tok/s_gen q8 no-spec, 14.5 ms/token)

Bound by `PROTOCOL-RULES.md`. Follows `launch-fusion-protocol.md`: stage 0
there put the launch floor at 2.4-2.6 us (646 launches/token = 10.8% at the
q8 token), stage 2 (`bench/bench_gridbar.mojo`, `0e3880d`) put a grid-wide
atomic barrier at 0.60 us @96 blocks .. 1.10 us @384. Idea source: Luce
megakernel (Qwen3.5-0.8B, one persistent kernel per token, atomic barrier
between phases, LM head as a second launch). Their token is ~2 ms so
launches were ~half of it; ours is 14.5 ms, so this is a +8-12% round, not
a 1.5x one. the maintainer chose the full per-token shape over per-layer.

## Shape

One persistent kernel per token, m = 1, fixed grid G x 256 threads (8 waves,
= `ROW_THREADS`), G <= resident-block ceiling. Every launch in the decode loop
becomes a phase; phases are separated by the `bench_gridbar` barrier
(agent-scope counter + monotonic generation word, no reset between tokens).
Phase bodies are the EXISTING kernel bodies, hoisted into `@always_inline`
device functions that both the standalone `amar_*` kernels and the megakernel
call, with the block index replaced by a block-strided loop
(`for tile in range(block_idx.x, ntiles, grid_dim.x)`). Per-element
arithmetic and summation order are unchanged, so 64/64 token identity vs the
launch path holds by construction; the launch path stays in the engine as
the parity reference (`BARO_MEGA=0`).

Phases that collapse (not just fuse): the six `amar_skinny_reduce*[..., 1]`
passes are `NSPLIT=1` copies/casts (`Cp[0] -> Y`) -- the GEMM phase writes
its destination directly; `reduce_add` becomes one add at the GEMM tail (same
single addition, same order).

Per-token launches: 646 -> 3 (megakernel, LM-head GEMM over VOCAB=248320 rows,
argmax); stage 3 folds the last two in.

Barriers per token: attn layer 8 (rmsc | q,k,v GEMMs | split+hrms+rope+append
| att | gmul+o-GEMM+add) x 8, ssm layer 7 (rmsc | qkv,z,a,b GEMMs | rgates+conv
| l2 | delta | gated | out-GEMM+add) x 24, ffn 4 (rmsc | gate,up GEMMs |
swiglu | down-GEMM+add) x 32, head 2 = ~360 barriers x ~0.7 us = 0.25 ms.

## Prediction (frozen)

Launch cost removed: 643 x 2.43 us = 1.56 ms. Barriers added: 0.25 ms.
Net 1.31 ms of 14.5 = 9.0%, minus whatever the fixed grid costs the GEMM
phases (stage 0 measures it; ceiling assumes 0).

| stage | prediction | land rule |
|---|---|---|
| S2 engine, m=1 no-spec | **+8% to +12%** (68.8 -> 74.3..77.1 tok/s_gen) | 64/64 identity vs `BARO_MEGA=0` AND 20-prompt median >= +6%, spread < 5%, baseline re-run in the same stint |

Stop rules: **S0a** block-strided q8row at the best resident G loses >= 8% vs
the native launch on the ffn shape -> the GEMM phases eat the launch saving,
close the round at stage 0. **S0b** the combined kernel's VGPR count forces
G < 96 -> close (occupancy, Luce's S_TILE lesson; `delta_step` already spills
122 at m=1, `ssm-occupancy-protocol.md`). **S2** < +4% -> close, code stays
on the branch, receipt in `docs/BASELINE.md`.

## Stage 0: fixed-grid GEMM cost + occupancy (S, `bench/bench_mega_gemm.mojo`)

Cold-cache protocol (`coldcache-protocol.md`: W >= 96 MB rotated). q8row m=1
block-strided at G in {96, 144, 192} (see occupancy receipt) vs native `ceildiv(N, 8)` grid, shapes
(N x K): 4096x4096, 8192x4096, 12288x4096, 4096x12288. Receipt: us per
shape per G, ratio to native, clock via `clock-probe.sh`.

Occupancy receipt: a kernel containing the q8row body + the delta_step body
(no barrier logic needed), built with `mojo build --emit=asm`; read
`.vgpr_count` / `.sgpr_count` / `.lds_size` from the code-object metadata;
resident blocks (wave64, 8 waves/block) = floor(768 / vgprs_granule8) waves per
SIMD x 4 SIMD / 8 waves = blocks per CU, x 96 CUs. Cross-check with
`bench/bench_mega_occ.mojo <G>` (q8row phase + barrier), whose barrier has a
bounded spin (`SPIN_LIMIT`) and a fail word: an overshoot prints
`NOT-RESIDENT` and returns. **NEVER launch an unbounded grid barrier past the
computed ceiling: a deadlocked barrier is not a timeout, it is a gfx ring
hang -> MODE1 reset -> every GL client on the desktop dies (2026-09-05 23:25,
`mega_occ_probe 240`, hard reboot).** Receipt 2026-09-05: vgpr 182 -> 4
waves/SIMD -> 2 blocks/CU -> 192; probe: 96/144/192 complete, 216
NOT-RESIDENT. Stage 0 G set is therefore {96, 144, 192}; 288/384 are out.


### Stage 0 receipt (2026-09-05 23:40, `ce3576d`+grid edit, `.work/mega-stage0.log`)

Arm: 290 W cap, -100 mV, engine-pack-q8 blk.0, NBUF=8 rotation, ITERS=200, 5 reps;
clock-probe sclk med 2802 (2694-3122) MHz, power 281-346 W, Tj 74 C. All G
bit-exact vs native (0 mismatches).

| shape (NxK) | native us | G=96 | G=144 | G=192 | best/native |
|---|---|---|---|---|---|
| 12288x4096 | 71.8 | 64.3 | 64.3 | 64.7 | 0.895 |
| 4096x12288 | 77.4 | 65.3 | 68.6 | 74.8 | 0.845 |
| 8192x4096 | 47.0 | 44.6 | 44.2 | 44.7 | 0.941 |
| 4096x4096 | 27.9 | 24.3 | 24.2 | 24.5 | 0.868 |

**S0a does not fire**: the fixed grid is 5-16% FASTER than the native
`ceildiv(N,8)` launch at G=96 on every shape (one block per CU, no tail
wave); G=192 is worst but still <= native. Best resident G for the
megakernel = 96. This is a gain the frozen prediction assumed to be 0, so
the stage-2 ceiling is +8-12% from launches PLUS the GEMM-phase gain; the
frozen numbers stand as the land rule regardless. -> stage 1.

## Stage 1: one ssm layer as a persistent kernel (L, `kernels/mega.mojo`)

Hoist `matmul_skinny_q8row`, `ssm_*`, `rmsnorm_cast`, `residual_add` bodies
into inline device functions (kernel files stay comment-free per CLAUDE.md);
`amar_mega_ssm_layer` = 7 phases. Gate: `kernels/test_mega_block.mojo`
bit-exact vs the launch sequence on random inputs, layer time at G vs the
14-launch sequence (device events).

### Stage 1 receipt (2026-09-05 23:49, `kernels/mega.mojo` + `kernels/test_mega_block.mojo`)

`amar_mega_ssm_layer`: 7 phases (rmsc | qkv,z,a,b GEMMs | rgates+conv | l2 |
delta | gated | out-GEMM+add), 6 grid barriers, G=96 x 512 threads. GEMM
phases write their destinations directly (NSPLIT=1 reduce collapsed); the
head phases map head -> block 0..31 on threads 0..127 with the original
wave-sum order. ISA: vgpr 192 (spill 121, delta_step's), sgpr 87, LDS 1 KB
-> 1 block/CU at 512 threads, G=96 resident, S0b does not fire.

| gate | result |
|---|---|
| residual X (4096) | 0 mismatches |
| conv window (2 slots x 3 x 8192) | 0 mismatches |
| ssm state (2 x 32 x 128 x 128) | 0 mismatches |
| layer time, 200 iters, synthetic q8 weights | launch(14) 120.8 us, mega 87.1 us, **0.72x** |

Per-layer saving 33.7 us x 24 ssm layers = 0.81 ms/token from the ssm
sub-blocks alone (14 launches x 2.43 us = 34 us predicted; measured 33.7,
so barrier cost is hidden under the fixed-grid GEMM gain). -> stage 2.

## Stage 2: per-token kernel (L-XL)

Add attn phases (`attn.mojo` bodies), ffn, embed, head-norm; device-side
layer loop reading a weight-offset table (`List[Int]` -> device buffer) and
building TileTensors from `wbuf` pointers in-kernel (**verify in stage 1 that
`TileTensor(ptr, layout)` constructs in device code; if not, pass per-layer
pointer structs**). `serve/engine.mojo`: `BARO_MEGA=1` replaces the layer
loop with one `enqueue_function`; LM head + argmax stay as launches. Gate =
the prediction table. Profile receipt: `BARO_PROFILE` per-phase device
timestamps (block 0 thread 0 writes `clock64` at each barrier) so the
post-mortem says which phases moved.

## Stage 3: fold LM head + argmax; MR > 1 (M, only if S2 lands)

Head GEMM as a phase (VOCAB rows block-strided, 31040 row-groups), argmax via
per-block partials + last-block reduce; 1 launch/token. MR > 1 for the MTP
window mirrors `gemm_q8`'s dispatch (comptime MR) -- separate prediction,
separate freeze, because the m>1 kernels have different occupancy.

## Not in this round

q4/q8dot weights, prefill, MTP verify width, hipLaunchCooperativeKernel (0.1
us, not needed), Mojo `Semaphore` (unprobed).
