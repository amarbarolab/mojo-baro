# R6 lane report: the qwen35moe persistent token kernel

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-r6-persistent-moe-lane.md`.
Design `docs/design/moe-persistent-kernel.md`; protocol
`bench/moe-persist-protocol.md`. Champion at lane open: `38ee0b7`, launch
path, 727 launches per token, 111.89 tok/s_gen.

## R6.1: the full MoE body at parity (speed not gated)

### What was built

`kernels/mega_moe.mojo`, `amar_mega_moe_token`: one launch per token over
the 40 layers, G = 96 blocks of 256 threads, the dense kernel's bounded grid
barrier and fail word (`grid_barrier` from `kernels/mega.mojo`), the MoE
logical offset walk (16 entries per attention layer, 19 per SSM layer, the
order `serve/engine.mojo` builds from `index.txt`). The head stays on the
launch path: the kernel ends after layer 39's FFN, and the existing
`rmsc_k` + head GEMM + `r_head` + argmax run after it as before.

Phases per SSM layer, one grid barrier after each: norm (every block computes
the full row sum, block-strided bf16 + f32 writes, the `amar_rmsnorm_cast2`
form), projections (qkv 8192 rows, gate 4096 rows with the raw-block
`q8_0_row_dot`; alpha and beta 32 rows each with the f32 `skinny_m1_row2`
loop, one wave per row, wave-strided over 768 resident waves), gates + conv,
l2 norm, delta (the `delta_col` form the launch kernel uses at m = 1), gated
out, ssm_out with the residual add. Per attention layer: norm, q/k/v rows,
head norm + rope + KV append, attention (`attn_head_span` below
`att_split`, the dattn split + combine above it), gate multiply, out
projection with the residual add. Per FFN: norm, router GEMV (256 rows),
top-8 + shared sigmoid (block 0 wave 0, the one-wave kernel's body moved into
`router_top8_sig_body` and called from both the launch kernel and here),
gathered gate+up (4096 waves of work) and shared gate+up (512) in one
wave-strided list, then one wave per output column does the eight routed
down dots (q4_k, or q6_k on layers 34, 38, 39), the shared down dot, and
the `(X + routed) + shared` add in the launch kernel's order.

Every dot product is the launch path's own function (`q8_0_row_dot`,
`q4k_dot_blocks`, `q6k_row_dot`, the f32 row loop copied statement for
statement), reading the same global activations, so the per-element
arithmetic is the launch kernel's by construction; only the work
distribution changed.

Host side: `BARO_MEGA=1` on the MoE profile now selects this kernel (default
off there; it used to raise). It refuses to combine with `BARO_TIER` (the
expert tier is a host round trip per layer) and `BARO_EXPERTS` (the trace is
not written). The device offset table grew from 512 to 1024 entries
(`OFF_CAP`): the MoE table has 733, and it is re-uploaded after the engine
builds the logical order, since `alloc_bufs` had uploaded the lexical one.
Five `if MEGA_ALLOWED or cfg.mega` guards on the dense FFN launches in
`window.mojo` became `comptime if MEGA_ALLOWED`; with `cfg.mega` now
reachable on the MoE they would have run the dense `gemm_w` on MoE offsets
during prompt replay.

### ISA receipt (`tools/isa-receipt.py`, `.work/r6/isa-r61`)

| kernel | vgpr | vgpr spills | sgpr | sgpr spills | scratch B | instr |
|---|---|---|---|---|---|---|
| `amar_mega_moe_token` | 256 | 256 | 107 | 102 | 940 | 18439 |

Residency at 256 VGPRs, wave32, 8 waves per block: 6 waves per SIMD, 3
blocks per CU, ceiling 288 blocks; G = 96 is inside it with 3x slack. The
spills are R6.2's first pool, not a parity matter.

### Gate 1: per-layer dump compare, one prompt (p09, 64 tokens)

Same binary `.work/r6/engine-r61` (sha `a26584c5851eda14` before the
offset-walk fix, rebuilt after it), `BARO_MEGA=0` against `BARO_MEGA=1`,
`BARO_DUMP` on both, `tools/dump-diff.py --layers 40 --hidden 2048`:
**identical over all 64 dumped tokens and 80 slots per token**, GENERATED
identical (64 ids), fail word 0 on both arms.

Two defects found by this gate, both fixed in the lane:

- First build: layer 0 was bit-exact (both slots) and every later slot was
  NaN. The line `w += W_ATT if is_att else W_SSM` advanced the offset walk
  by 16 on SSM layers as well, so layer 1 read `up_shexp` bytes as its norm
  gamma. Written as an explicit if/else.
- `tools/dump-diff.py` tested `abs(d).max() > 0`, which is False for NaN, so
  it skipped the all-NaN slots and reported a false first divergence at
  token 1 layer 0. It now treats any NaN as a divergence.

### Gate 2: 20-prompt teacher-forced identity vs champion `38ee0b7`

`bench/force-ab.sh` gained a fifth argument (candidate-only env) because the
champion binary refuses `BARO_MEGA=1`. Reference `.work/r6/engine-ref-38ee0b7`
(sha `74e249fb65beab20`, built from a `git worktree` of `38ee0b7`), greedy on
the launch path; candidate `.work/r6/engine-r61` with `BARO_MEGA=1`,
teacher-forced on the reference's GENERATED ids.

Candidate sha `02ccf54004c68d2d` (`.work/r6/force-r61/arm.txt`, `.work/r6/force-r61.log`):
**20/20 prompts, forced agreement 64/64 on every prompt, min 100.0% mean
100.0%, no voids.** Fail word 0 on all 20 candidate runs; arm identity read
back from every log (`BARO_MEGA: False` x20 reference, `BARO_MEGA: True`
x20 candidate). The candidate's own tok/s_gen under teacher forcing read
108.07 median: an instrument reading from a forced run, not a number, and
not the R6.2 A/B (that one is preregistered first, then run, both arms in
one stint).

### Repo gates

`tools/ci-checks.sh` exit 0 (census 100 kernels, 0 orphans, `docs/KERNELS.md`
regenerated; 26 bench sources build; vendored copies in sync).
`./run-tests.sh` exit 0, 104 PASS (`.work/r6/run-tests-r61.log`).

### GPU minutes

44 engine runs (2 dump-compare runs, 2 dump4 probe runs, 40 identity runs),
97 s of pack load plus forward on the GPU, about 1.6 GPU minutes, plus
`run-tests.sh`. Every gate well under the 10-minute line.

## R6.2: speed (preregistered `b5bbab0`, result below the kill line)

### Receipts before timing

- Per-run device receipt, the maintainer's P1 gap after R6.1: the engine prints the
  grid-barrier generation counter from `ctr_d` (only a persistent kernel's
  barriers advance it), 470 per token on the MoE profile; `force-ab.sh` and
  `ab-prompts.sh` record it and the fail word per run for both arms
  (`6af3fcc`). rocprofv3 launches per token: launch arm 727.0, persistent arm
  8.0 (`.work/r6/lc-*.log`).
- Phase stamps of the kernel as landed, kernel span 8391 us, table in the
  protocol: the q8 projection phases ran at about 400 GB/s against the launch
  kernels' 950 for the same bytes.
- Disclosed instrument reading from the R6.1 identity runs: 0.971 (forced
  candidate vs greedy reference, host-synced per token).

### Lever 1, grid size: falsified by residency

`BARO_MOE_G` builds at 96/192/288 on three prompts: **G = 192 and 288 are
NOT-RESIDENT at the first barrier (fail word 1, gen 0) on 6/6 runs**; G = 96
runs (fail word 0, gen 470 per token). My preregistration's 3-blocks-per-CU
arithmetic used the wrong register-file size; the `persistent-kernel-gfx11`
formula (vgpr 256 -> ceiling 96) was right. Occupancy is not a knob at 256
VGPRs. In the ledger.

### Lever 2, latency hiding inside the wave: bit-exact, 8391 -> 7986 us

Three builds, each gate-1 dump compare identical:

| build | change | vgpr / spills / scratch | kernel span us | note |
|---|---|---|---|---|
| r61 | as landed | 256 / 256 / 940 | 8391 | |
| r62 | q8 U=4 + q4k U=2 | 256 / 315 / 1140 | 8743 | SSM proj -230, down +376 (spills moved into it) |
| r63 | q8 U=4, q4k original, chunked-fma delta | 256 / 236 / 936 | 7986 | delta 457 -> 253, down 1689 -> 1534, q8 phases flat |

The chunked delta is the `RELOAD=True` form of the dense kernel; its
`fma()` spelling matches the launch `amar_ssm_delta_step` ISA (fma 1061 /
mul 346 / add 0 read from the same binary), and the dump compare and the
20-prompt identity confirm it. `isa-loops` fingerprint of the timed kernel:
hot loops dual 361/361/51/51/51/227/229/26, totals fma 1691 mul 262 add 387
scratch 395.

### Confirmation, 20 prompts, one stint

Identity vs champion `38ee0b7`: 20/20 at 64/64, gen 30080 on every
candidate run (`.work/r6/force-r63/`). A/B one binary (sha
`7c27c2a82214b485`), `BARO_MEGA=0` vs `1`, clock probe sclk med 3134 MHz,
cap 290 W, -100 mV, junction 71 C:

| arm | median tok/s_gen | spread | gen per run | fail word |
|---|---|---|---|---|
| launch | 110.90 | 1.4% | 0 | 0 |
| persistent | 113.94 | 1.4% | 30080 | 0 |

**Ratio 1.027, identity PASS 20/20.** Kill line +5% not met; prediction
(130 to 145) falsified. Default stays `BARO_MEGA=0` on the MoE profile; the
kernel is in the tree as an opt-in. The protocol's result section names the
three pools for a next round; the largest is the q8 projection phases, 2.7
of 8.0 ms at 400 GB/s, which the per-wave unroll did not move.

### GPU minutes, whole lane

R6.1 about 1.6 min plus run-tests; R6.2: launch counts (4 traced serve
runs), 12 sweep runs, 3 gate runs, 40 identity runs, 40 A/B runs, about 3 s
of GPU each including the 1.4 s pack load: about 5 minutes, plus two
run-tests. No gate over 10 minutes.
