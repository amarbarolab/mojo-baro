# Prefill GEMM protocol: bf16-lds as the trunk prefill GEMM (lane pfgemm, 2026-09-08)

Binds to `bench/PROTOCOL-RULES.md` P1-P6 and continues
`bench/prefill-protocol.md` (R1-R4 on main; R5/R5a/R5-abl on branch
`lane-int8`, commit `181d6c6`, where the LDS-pipelined schedule was measured:
bf16-lds 1.86x over the R4 kernel at n=1024 on the ffn shape, 1.75x at
n=256, 1.02x at 128, 0.77-0.81x at n=16/64; int8 MMQ closed negative).

## Arms

- **main** (`b3da39a`): `amar_matmul_prefill_q4/q8` (R4 kernel, register-
  staged, `kernels/matmul_prefill.mojo`) for every prefill GEMM; chunk
  `CP = 1024`.
- **lds** (this branch): `amar_matmul_prefill_lds[WDT, 4, 2, 2, 4]`
  (`kernels/matmul_prefill_lds.mojo` = the bf16 arm of lane-int8 `813c999`
  with the int8 path removed and an int8-weight loader added for the q8
  pack; 8 waves 4x2, wave tile 2x4, 128x128 block, BLK_K 32, two LDS
  buffers, XOR-swizzled rows, one barrier per K-step) for prefill GEMMs
  with m > `PF_LDS_MIN = 128`; m <= 128 stays on the R4 kernel (R5 receipt:
  lds is slower below 128 rows). Same maths as R4 (nibble/int8 -> bf16 by
  `f32x16_to_bf16_trunc`, WMMA lo then hi with the lo result as C-input,
  `acc = fma(t2, d, acc)`), so the engine output is bit-identical by
  construction. The R5a uniform loader (`8347c55`) is not taken: it measured
  a loss on bf16-lds inside the spread.
- Decode path untouched (`kernels/mega.mojo`, the skinny kernels).

## Gates (before any timed run)

1. `kernels/test_prefill.mojo` section 1b: lds vs R4 bit-exact (zero
   differing floats over 1024 x 12288) for q4 128x128 / 64x128 at M=1024,
   M=24 padded, n=32, ACC epilogue; q8 128x128 at M=1024 and M=24.
2. `tools/merge-gate.sh` green (engine + test builds, run-tests, ci-checks,
   test_prefill, one-shot q4/q8 64/64, p0512, mtp 20/20, server suite).
3. Prefill identity: `GENERATED` on p0512 / p8192 / p32768 byte-identical
   between the main engine and the lds engine (same pack, same env).
4. Decode A/B `bench/ab-prompts.sh` main engine vs lds engine over the 20
   mtp prompts: identity 20/20, ratio within the run's spread (the prompts
   are <= 24 tokens, so the prefill path is the R4 kernel in both arms and
   the decode kernels are untouched).

## Measurement

- `prefill_s` printed by the engine (wall from the prompt ids on the device
  to the last prompt token's megakernel, `serve/engine.mojo`), `BARO_PACK=
  .work/engine-pack-q4`, `BARO_TMAX=102400`, `BARO_PREFILL_C=1024`
  (default `CP`), 64 generated tokens. Three runs per length, median with
  min-max; exclusive GPU (`gpu-wait run --priority 90 --vram 23`, queue
  empty, pasted in the status file).
- P1 receipts: `TMAX:` / `prefill chunk:` / `prefill rows:` lines printed
  by the run; the binary built in the same stint from the commit named in
  the Result; `gpu-wait list` before each timed batch.
- GEMM share: `BARO_PROFILE=1` now makes `prefill_forward` synchronize
  around every prefill GEMM and print per chunk `prefill profile: rows m
  gemm_s chunk_s share` (serialized, so chunk_s > the unsynchronized chunk;
  only the share is read). Measured on a variant of the lds binary with
  `PF_LDS_MIN = 1 << 30` (every GEMM on the R4 kernel) before the
  prediction is frozen, and on the lds binary afterwards as the receipt.

## Receipts before the freeze (2026-09-08 13:25-13:32, `.work/logs/`)

- Parity: `test-prefill-2.log` 7 lds checks bit-exact, PASS.
- Kernel level, this session, try 1 (`bench-prefill-1.log`; a `mojo build`
  job from another lane was admitted beside it, `gpu-list-bench.txt`):
  us per GEMM, ffn shape, wmma 128x128 / bf16-lds 128x128: n=128 644 /
  991, n=256 2525 / 1779, n=512 3452 / 2160, n=1024 4444 / 2557 = **1.74x
  at n=1024** (R5 on lane-int8: 4183 / 2250 = 1.86x). Try 2 runs in the
  same stint as the timed runs, queue empty.
- Main bar, `.work/engine-main` built from `b3da39a` sources in this stint,
  q4 pack, `BARO_TMAX=102400`, chunk 1024, 3 runs (`main-p*-*.log`; P1:
  `TMAX: 102400`, `prefill chunk: 1024`, `prefill rows: 8191 / 32767`
  printed by each run):

  | length | prefill_s median (min-max) | M0d recorded |
  |---|---|---|
  | 8192 | 8.32 (8.31-8.34) | 8.30 |
  | 32768 | 51.40 (50.55-51.49) | 50.24 |

- GEMM share on the R4 path (`engine-r4prof`, this branch with
  `PF_LDS_MIN = 1 << 30`, `BARO_PROFILE=1`, `r4prof-p*.log`): per
  1024-row chunk the GEMMs take 0.515 s regardless of position; the rest
  of the chunk grows with position (attention over the KV).

  | length | chunks | gemm_s | chunk_s (serialized) | share | prefill_s of the same run |
  |---|---|---|---|---|---|
  | 8192 | 8 | 5.14 | 8.37 | **0.614** | 8.38 |
  | 32768 | 32 | 21.10 | 51.57 | **0.409** | 51.58 |

  Serialization costs < 1 % here (chunk_s sum = the unsynchronized
  prefill_s), so the share is read as-is.

## Frozen prediction (commit before the lds timed runs)

GEMM time on the lds path = R4 GEMM time / 1.86 (the R5 kernel-level
ratio at n=1024, every chunk being 1024 rows); everything else unchanged.

| length | main (this stint) | R4 gemm_s | predicted lds gemm_s | **predicted prefill_s** | ratio | with 1.74x (try 1) |
|---|---|---|---|---|---|---|
| 8192 | 8.32 | 5.14 | 2.76 | **5.95** | 1.40x | 6.14 |
| 32768 | 51.40 | 21.10 | 11.34 | **41.6** | 1.24x | 42.4 |
| 100000 | 298.6 (M0d) | 0.515 x 98 = 50.5 | 27.1 | **~275** | 1.09x | ~277 |

Gate (landed): prefill_s 8192 <= 6.6 s and 32768 <= 44 s, i.e. the in-situ
GEMM speedup is >= 1.5x. Falsifiers: 8192 > 6.6 s = the engine's shapes do
not transfer (the KV projections N=1024 are 64 blocks on 96 CUs, the ssm
a/b projections N=32 are 8 blocks; the ffn shape is 768) or the engine
GEMMs are not B-load bound the way the bench's are; 8192 < 5.7 s = the
share was undercounted (serialization hid overlap). Identity: every lds
run's `GENERATED` equals the main engine's on the same prompt (bit-exact
maths); a single differing token voids the arm. Decode: 20-prompt A/B
ratio within its spread and identity 20/20 (the prompts are <= 24 rows,
so they never reach the lds kernel; anything else is a harness change).
llama.cpp bars (chat-protocol M0c, `prompt_ms`): 2.59 s at 8192, 12.8 s
at 32768 -- not re-run here; the gap after this round is predicted 2.3x /
3.3x, still the attention and the non-GEMM chunk work, not the GEMM.
