# A2 step 2: int8 KV on the block table (2026-09-17, lane-a2s2, opus)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-17-A2-step2-int8-kv.md`. Protocol `bench/a2-protocol.md`
section "A2 step 2" (bar measured, predictions frozen `773a75a` before any int8 GPU run). Code `80d4b2c`,
bar harness `f45a7dd`, RULER harness `2a231c0` + `aa74429`. Receipts under `.work/a2s2/`.

Engines (all from `80d4b2c`; the freeze commit changed only the protocol text):
reference `a8ab6d2903176a42` (`ed8c7f7`, before this lane), f32 default `829c25a55d4c5421`,
int8 `dcf1564e082c5286` (`-D BARO_KVQ=int8`), bf16 `38ee0084a8e8ac14` (`-D BARO_KVQ=bf16`, bar arm only).

## Verdict

**Arm 1 (per-row int8, post-RoPE) passes the bar; arm 2 (KIVI axes) was not needed.** KV bytes per
token 65536 -> 16640 (0.254x), agreement above what llama.cpp's own q8_0 KV reaches on the same 60
prompts, decode after 32k **117.69 vs 100.54 tok/s (1.171x)** in the same session, short context
1.011x. The code is opt-in (`BARO_KVQ=int8`); the default build is unchanged at the ISA level.

## The bar (P14)

| arm vs its own f32 KV | 8k min / mean | 16k min / mean | 32k min / mean | receipt |
|---|---|---|---|---|
| llama.cpp q8_0 (block-32 scales) | 96.9 / 98.52 | 95.3 / 98.12 | **93.8** / 97.97 | `.work/a2s2/bar/full/` |
| llama.cpp f16 | 100 / 100 | 100 / 100 | 100 / 100 | same |
| ours bf16 | 98.4 / 99.8 | 96.9 / 99.7 | 98.4 / 99.6 | `.work/a2s2/gates/bar-bf16/` |
| **ours int8 (arm 1)** | **96.9 / 99.3** | **96.9 / 99.2** | **96.9 / 99.5** | `.work/a2s2/gates/int8-id/` |

llama.cpp side: `bench/a2-bar-llama.sh` (new tool `tools/llama-force.cpp`, libllama driver; C++ because
the API passes structs by value). f32 self-check 3840/3840. A first version forced the 63 reference
ids as one batch and read 127/128 against f32 itself: llama.cpp kernels are not batch-invariant, so
forcing is one decode step per token, like our engine. llama.cpp's f16 equals its f32 because its
flash-attention kernel runs in f16 either way. Frozen: `MIN_PCT=93.75` per prompt and set mean >= 97.9.

## Layout

- `BARO_KVQ=f32|bf16|int8` comptime (default f32) sets `KVT` in `kernels/attn.mojo`; start-up line
  `BARO_KVQ: int8  kv dtype: int8  kv pool bytes: 545259520  kv bytes/token: 16640` (TMAX 32768).
- One (page, attention layer, KV head) stride = 128 x 256 int8 then 128 float32 scales
  (`KVPAD = KVPAGE * 4`). Scale = row absmax / 127, round half away from zero.
- **Scales in the page tail, not a separate plane**: the block table, pool sizing, state paging and any
  future page move (A3, B3) carry them unchanged, no kernel gains an argument, and there is one
  address function. The reverse-table arm proves the address math (below).
- Quantize at write: `amar_kv_append`, `append2`, the token and window megakernels (one extra barrier:
  wave max, row max). Dequantize at read: split kernel at span load (so `dattn_step` is unchanged),
  exact span and `attn_head_span` (scale hoisted per key row), prefill and WMMA prefill per row.
- Refusals: state save/load and LatentOS KV pages raise on non-f32 KV; MoE engine and spark raise at
  start-up. `serve/latent.mojo` and `bench/latent_ingest_bench.mojo` now take `DeviceBuffer[KVT]`
  (a conditional comptime `KVT` does not unify with `f32` in signatures).

## Gates

Preflight PASS before every GPU job (`c071ac51e8fb` on the frozen tree); 290 W cap; refcache hit on
the reference ids for every candidate after the first.

| prediction | command | result | verdict |
|---|---|---|---|
| P-S2a int8 identity table | `MIN_PCT=93.75 bench/a2-gate.sh engine-ref engine-int8 .work/a2s2/gates/int8-id` | exit 0; min 96.9 at 8k/16k/32k, means 99.3 / 99.2 / 99.5 (band [97.0, 99.5], point 98.5), restored 59/60, `kv bytes/token: 16640` | PASS (means at the top of the band) |
| P-S2b int8 reverse table | same with `BARO_KVTAB=reverse`, `.work/a2s2/gates/int8-rev` | exit 0; per-prompt counts **identical** to P-S2a on all 60 (`diff` of results) | PASS |
| P-S2c decode after 32k | gate median | int8 **117.69** (reverse 118.17); band [108, 128], point 120, bar 101 | PASS |
| P-S2d short context | `AB_ENGINE_B=engine-int8 bench/clock-probe.sh bench/ab-prompts.sh engine-ref .work/a2s2/ab "BARO_SPEC=0 BARO_MEGA=1" (x2) ref int8` | ref 136.07, int8 137.62, **ratio 1.011**; greedy identity 16/20 (not gated); megakernel on both (`BARO_MEGA: True`); sclk med 2948 MHz | MISS on my band [0.97, 1.00] (faster than predicted); inside the standing +-2% |
| P-S2e f32 default | `bench/a2-gate.sh engine-ref engine-f32 .work/a2s2/gates/f32-id` (MIN_PCT 100) | 60/60 at 100.0%, restored 59/60, decode after 32k 100.54 vs reference 101.38 (the reference's run in the bar job, same session; 99.2%) | PASS |
| P-S2f tests | `run-tests.sh` (gpu-wait job `mu4qyllkt948`); `tools/ci-checks.sh` and MoE/spark builds inside preflight | run-tests exit 0 (all PASS, 36.8 s); preflight PASS | PASS |
| P-S2f state refusal | `BARO_STATE_SAVE=... engine-int8` | exit 1, `BARO_STATE_SAVE: state files store f32 KV; this engine has BARO_KVQ=int8 (quantized KV state is A2 step 3)`, no file written | PASS |
| P-S2g RULER | `bench/a2-ruler.sh ENGINE OUT SIZE 5` (niah_single, N=5, spec off) | int8 **100.0** at 65536 and **100.0** at 131072; f32 100.0 at 65536; f32 at 131072 started (`kv pool bytes: 8724152320`, ready, 24.4 GB VRAM) and its prompt run was cancelled by decision: int8 at the ceiling leaves nothing for f32 to show. int8 128k prefill 104 s per prompt | PASS (128k comparison not run, see below) |

ISA (`tools/isa-receipt.py` + `isa-diff`, `.work/a2s2/isa/`): **f32 default vs reference: PASS**, no
resource change on any kernel, token megakernel 124/79/79/59 SAME, window and split SAME. int8 build
(recorded, not gated): q4 token megakernel dual 135/84/84/60, 0 spills (fast class); split kernel
165 -> 180 VGPR, 0 spills; attention decode 38 -> 44 VGPR; WMMA prefill spills 25 -> 16; window
megakernel spills 42 -> 51 and 48 -> 60 (behind `BARO_MEGA_WIN=0`).

## What the numbers say

- Long-context decode is byte-bound at 32k: bf16 (x0.5 bytes) bought 13.3%, int8 (x0.254) 17.1%.
  My arithmetic (bytes proportional) predicted 120; the dequantization cost ate about 2.3 tok/s of it.
- Per-row int8 beats llama.cpp's block-32 q8_0 on the worst prompt (96.9 vs 93.8) and every set mean.
  The reference here is a true f32 KV, llama.cpp's is f16 inside its FA kernel, so the comparison
  flatters neither side's precision; it only anchors what "known-good lossy" reaches.
- Short context got faster, not slower (1.011x): the megakernel's extra barrier costs less than the
  smaller cache footprint saves even at a few hundred tokens. Recorded as a prediction miss.

## Process defects (logged, fixed)

- `bench/a2-gate.sh REF CAND OUT ""`: an empty ENV argument makes `gpu-wait run` refuse the re-exec
  (`cmd must be a non-empty list of strings`); two gates ran zero GPU time and were re-run without
  the argument. Fixed in `a2-gate.sh` (the re-exec names its arguments), verified by a queued call.
- `bench/a2-ruler.sh` exits 143 after a clean, fully scored run. `aa74429` preserved the status
  through the EXIT trap and the reruns still exit 143, so the trap was not the cause (suspect: the
  gpu-wait reclaim signalling the job's process group after the server is killed). **Open**, scores
  and responses are complete on every 143 run; do not read that exit as a failure until fixed.
- A host low-memory event killed this session's background shells mid-lane; the running gpu-wait job
  (run-tests) finished and its log came back through `gpu-wait logs`; the queued RULER job was
  cancelled and re-run detached.
- Three RULER starts died with `hipErrorOutOfMemory`: the comfy lane's engine was running bare
  (outside gpu-wait, ~20 GB). Coordinator confirmed and restarted it under the queue; the arms were
  re-run clean.

## RULER decision (the maintainer, 2026-09-17)

The frozen P-S2g compares int8 with f32 on the same prompts. At 131072 int8 scored 100.0, so f32
could not show a loss; the f32 prompt run was cancelled after its start-up proved the f32 pool fits
(8.7 GB pool, 24.4 GB VRAM used, no headroom). Rule used: run the f32 128k arm only if int8 < 100.

## Left for step 3

- State files and prefix checkpoints of int8 KV (save/load refuse today; the page-tail layout means a
  page is copyable as bytes, scales included).
- LatentOS KV page export of int8 pages (`serve/latent.mojo` refuses).
- MTP spec on int8 KV (the draft head's `kc32` cache uses the same append/decode kernels, so it is
  quantized too; spec was off in every gate here) and the int8 default decision.
- int4 (KIVI axes) as its own lane; arm 2 of this lane was not run because arm 1 passed.
- MoE megakernel and spark: f32 only.

## GPU budget

Lane jobs (gpu-wait): llama.cpp bar 3 (smoke that failed its self-check, smoke, full 306 s); a2
gates 4 (bf16 bar, int8 identity, int8 reverse, f32 default; two further calls were refused before
the queue by the empty-ENV bug); short-context A/B + state refusal 1; run-tests 1 (36.8 s); RULER 7
(int8 64k, three OOM starts from the bare foreign engine, f32 64k, int8 128k at ~10 min, f32 128k
cancelled after start-up); gate-fix verification 1 (CPU, exits at the in-job preflight check by
design). `gpu-wait stats --days 1` machine-wide at report time: 231 jobs, 157 ok / 69 failed / 5
cancelled, GPU busy 414 min.
