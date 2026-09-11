# Decode attention, generalised (frozen 2026-09-11, before any kernel change)

Binds: `bench/PROTOCOL-RULES.md` P1-P6, `bench/coldcache-protocol.md`, repo `CLAUDE.md`.

## Why this round

The llama.cpp harvest (`results/ggml-harness-headroom-2026-09-11.md`, note
`2026-09-11-llama-cpp-kernel-harvest`) puts decode attention furthest from the 7900 XTX
roofline of any kernel our models run: `flash_attn_ext_vec` at 13-31% of 960 GB/s,
cache-cold. It is also the one kernel a granite engine is missing: `kernels/attn.mojo`
hard-codes Qwythos (`HD=256`, `NQH=16`, `NKVH=4`, `KVT=f32`).

## Scope

A decode attention kernel generic over `HD`, `NQH`, `NKVH` and `KVT` (f32, f16, bf16) at
comptime, runtime `scale`, one query token, reading the paged KV layout `kv_off` defines
(128-token pages, layer-in-page, LatentOS page format). Split across blocks for long
caches the way `attn_phases` does in `kernels/mega.mojo` (partials + combine), exact
single-span path for short ones.

Out of scope: prompt attention, masks, sliding window, gating, the megakernel phase
itself (a later round wires the winner in).

Base: `main` (contains `1e7ab91`, the page-arithmetic fix). **Nothing from `lane-attn`**:
its Round A V addressing is the bug `1e7ab91` deleted. Its hoisted-load idea is allowed
only on per-element `kv_off`.

## Arms

| arm | what | instrument |
|---|---|---|
| **R** | llama.cpp `flash_attn_ext_vec` via `bench/ggml-harness/op_bench flash_attn`, re-run in the SAME stint as O | rocprofv3 device time per iteration |
| **O** | ours, standalone bench with the same KV rotation (>= 4 x 96 MB) and the same shapes | rocprofv3 device time per iteration |

KV dtype f16 on both arms. Shapes (NH NKVH HD KV), one query token, KV = 4096:

| shape | model | R receipt (us, 2026-09-11) |
|---|---|---|
| S1 16 4 256 4096 | qwen35 (Qwythos, Ornith, 27B) | 55.99 |
| S2 40 8 64 4096 | granite-4.2-3b | 34.78 |
| S3 28 4 128 4096 | qwen2.5-7b | 67.03 |

Plus a scaling receipt at KV = 512, 1024, 16384 on S1 (split vs exact path crossover).

## Gates (all must pass before any timed run counts)

1. **P1 read-back:** O's bench echoes HD, NQH, NKVH, KVT, KV length, grid and block dims,
   split count, bytes rotated; R's harness line (`arm ... | rotated ... exceeds`) recorded.
2. **Numerics:** O vs an fp64 numpy reference of softmax(QK^T * scale) V over the same f16
   KV bytes: max abs error <= 2e-3 relative to max|out|, on all three shapes and on
   KV = 1, 127, 128, 129, 4096 (page edges).
3. **No regression on the shipped path:** instantiated at HD=256 / NQH=16 / NKVH=4 /
   KVT=f32, O is bit-identical to today's `amar_attn_decode` on `kernels/test_attn_block.mojo`,
   and the engine gate stays green (`./run-tests.sh`, `tools/model-ref.py` 64/64,
   mega == launch).

## Frozen predictions (device us per iteration, cache-cold)

Ideal time = KV bytes / 960 GB/s: S1 17.5, S2 8.7, S3 8.7 us.

| shape | R | O predicted | ratio R/O |
|---|---|---|---|
| S1 | 55.99 | 24-30 | 1.9-2.3x |
| S2 | 34.78 | 13-18 | 1.9-2.7x |
| S3 | 67.03 | 13-18 | 3.7-5.2x |

**Land:** O beats R by >= 1.5x on at least two shapes and is not slower on any, all gates
pass. **Close negative:** O < 1.1x on two or more shapes. Between: report, no land, one
more round only with a new frozen prediction.

## Stop rules

- Gate 3 mismatch: the generic kernel changed the shipped arithmetic. Find the difference;
  do not re-freeze the reference.
- R's rotation reads "BELOW": the R row is void, re-run, never compared.
- Any run whose fail word or parameter echo is missing is void (repo `CLAUDE.md`).

## Result

(empty until the lane reports)
