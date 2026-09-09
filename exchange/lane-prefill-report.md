# Lane prefill — report (2026-09-06)

Worktree `$HOME/Projects/mojo-baro-lanes/prefill`, branch `lane-prefill`, six commits on top of `main` (`751bc3c`, unchanged during the lane). Working tree clean.

## What landed

Real prompt ingestion through a prefill path, measured, parity-tested, gated, with the decode path untouched.

- `kernels/matmul_prefill.mojo` (new): `amar_matmul_prefill_q4` / `_q8` — bf16 WMMA GEMM (`mma` 16x16x16 on gfx1100) straight over the weight-native Q4_0 / Q8_0 layout: nibbles/int8 converted in registers (f32 cvt + `llvm.amdgcn.perm` truncation, exact), 32-block scale applied on the f32 accumulator, `ACC` epilogue for residual adds; `amar_prefill_swiglu_bf16`. Template `WTM, WTN, WAVES_M` (8 waves per block). Dispatch: 64x128 tile for m <= 64, 128x128 above.
- `kernels/attn.mojo`: `amar_attn_prefill` — causal chunk attention, online softmax, GQA 16/4, HD=256, reads the bf16 KV cache the decode path writes; `attn_head_body` untouched; `MAX_T` 128 -> 1088.
- `kernels/ssm.mojo`: `amar_ssm_gates_rows`, `amar_ssm_conv_chunk`, `amar_ssm_qk_l2norm_rows`, `amar_ssm_delta_chunk` (sequential recurrence per head over the chunk, bit-identical to the per-row kernel), `amar_ssm_gated_out_rows_bf16`.
- `kernels/test_prefill.mojo` (new): parity of every kernel vs fp64 host references computed in the test (rel < 1e-3 at every intermediate; delta chunk vs per-row 0.0; GEMM 3.5e-5 vs fp64 / 4.4e-5 vs `q4rowb`).
- `bench/bench_prefill.mojo` (new): kernel arm on the ffn shape, 4 WMMA configs vs the `q4rowb` MR=8 row loop, NBUF=8 cold rotation, correctness gate.
- `bench/prefill-protocol.md` (new): frozen measurement + predictions (`a488766`, before any timed run), results R1-R4.
- `serve/registry.mojo`: TMAX 128 -> 1088 (comptime-asserted <= MAX_T in `delta_dispatch`), CP=1024, PF_MIN=16, prefill layouts and bindings, `gemm_prefill_q4/q8` dispatch.
- `serve/engine.mojo`: `prefill_forward` + `gemm_pw`; prompt rows 0..L-2 in chunks of `BARO_PREFILL_C` (default 1024) when L-1 >= 16; last prompt token still through the megakernel; the last chunk writes the post-final-norm rows of the final `(L-1) mod 8` window into `hn` and sets `pos_prev`, so the MTP draft sees the same window as before. `BARO_PREFILL=0` = the old chunk-8 path, bit for bit. Prints `TMAX / prefill chunk / prefill rows` as the P1 read-back.
- `docs/KERNELS.md` regenerated (63 kernels, 35 in registry, 0 orphans).

## Gate

`.work/prefill-gate.txt` (worktree), run after the final build:

| step | command | exit |
|---|---|---|
| repo tests | `gpu-wait run --priority 60 --timeout 1800 -- ./run-tests.sh` | 0 |
| ci | `tools/ci-checks.sh` | 0 |
| prefill parity | `gpu-wait run … -- ./.work/test_prefill` | 0 (`PASS: prefill kernels`) |
| default identity | `tools/check-tokens.sh .work/engine-pack-q4/ref-tokens-64.txt .work/prefill-v2/default.log` | 0 (64/64) |

## TTFT (ms, `prefill_s` = prompt ids on device -> first argmax on host; 3 runs, q4 pack)

Command per run: `BARO_PROMPT=bench/prefill-prompts/pNNNN.tokens gpu-wait run --priority 60 --timeout 3600 -- .work/engine` (inside `.work/run-stage3.sh`, log `.work/stage3.log`, per-run logs `.work/prefill-v2/pNNNN.{1,2,3}.log`). Read-back on every run: `prompt tokens: N`, `TMAX: 1088  prefill chunk: 1024  prefill rows: N-1`, `mega fail word: 0`, `tokens: 64`, `pack q4 trunk: True`, `BARO_MEGA: True`, `BARO_SPEC: False`.

| length | ours/prefill (final) | ours/chunk8 (before, TMAX=1088) | llama.cpp Q4_0-pure `prompt_ms` | frozen prediction (prefill / llama) |
|---|---|---|---|---|
| 32 | 65.3 / 66.0 / 66.8 | 216.7 / 214.4 / 216.5 | 77.7 / 56.9 / 52.4 | 30 / 40 |
| 128 | 124.7 / 124.6 / 124.3 | 831.5 / 834.2 / 833.2 | 73.3 / 73.4 / 73.3 | 55 / 80 |
| 512 | 415.4 / 418.0 / 414.6 | 3306 / 3326 / 3333 | 165.5 / 167.5 / 166.8 | 170 / 250 |
| 1024 | 761.4 / 759.7 / 762.7 | 6701 / 6675 / 6686 | 315.3 / 313.8 / 315.1 | 330 / 500 |

- chunk8 arm: `.work/engine-tmax` (source of `751bc3c`, TMAX=1088 only), logs `.work/prefill-tmax/`; TMAX=128 baseline (`.work/engine-base`) could only run 32 tokens: 214.5 / 217.2 / 215.9.
- llama.cpp arm: `.work/run-llama-ttft.sh` — `llama-server` on the Q4_0-pure gguf with the flags of `tools/llama-mtp-prompts.sh`, port 8097, `/completion` with the same token ids, `n_predict 64`, `cache_prompt false`; `.work/prefill-llama/props.json` (`n_ctx 8192, n_batch 2048, n_ubatch 512, flash_attn on`); per run `prompt_n == N`, `draft_n 0`; table in `.work/prefill-new/llama-ttft.txt`. Each engine alone on the GPU.
- Verdict vs the frozen predictions: direction right (prefill beats chunk8 at every length, 3.3x at 32 to 8.8x at 1024), magnitude wrong on both sides: our GEMM reaches 24.6 not 50 TFLOP/s and is weight-stream-bound below 64 rows; llama.cpp's MMQ prompt path is 2.3x faster than predicted. Final: 1.16x slower than llama.cpp at 32 tokens, 1.7x at 128, 2.5x at 512, 2.4x at 1024.
- Intermediate v1 (tail window still on the decode path): 125 / 184 / 489 / 857 — the 8-row `q4rowb` window costs ~50 ms per prompt, hence the tail change.

## Kernel arm (`bench/bench_prefill.mojo`, ffn shape N=12288 K=4096 Q4_0, us per GEMM, median of 5, `.work/prefill-v2/bench.txt`)

| n | wmma 64x128 | wmma 128x128 | rowloop q4rowb MR8 x n/8 | best TFLOP/s | predicted wmma / rowloop |
|---|---|---|---|---|---|
| 16 | 445 | 435 | 590 | 3.7 | 60 / 270 |
| 64 | 512 | 605 | 2222 | 12.9 (32x256: 498) | 130 / 1100 |
| 256 | 2056 | 1629 | 8910 | 15.8 | 500 / 4300 |
| 1024 | 4829 | 4183 | 35710 | 24.6 | 1900 / 17000 |

`correct: true` (WMMA vs row loop rel < 1e-4 at every n). Both predictions were optimistic; WMMA wins at every n, so the wave-per-row n-row variant was not pursued (its MR=8 register ceiling makes it the n/8-pass loop, measured 2x slower than predicted for the same reason the chunk8 baseline was 4x slower than predicted).

## Identity and decode receipts (`.work/stage3.log`, `.work/prefill-v2/`)

- First generated token identical to the chunk8 arm at 32/128/512/1024 (264 / 2469 / 16 / 3706); 64-token stream identical at all four lengths.
- Default prompt 64/64 vs `ref-tokens-64.txt` (5 prompt tokens: below PF_MIN, so this gate does not exercise the prefill path).
- `BARO_PREFILL=0` on p07-json and p12-rust reproduces the pre-lane `engine-base` stream 64/64 (decode path unchanged).
- Chunk-size invariance: `BARO_PREFILL_C=16` vs 1024 on p12-rust (24 rows -> 16+8) and p1024 (64 chunks) 64/64 — chunk boundaries carry the conv / SSM / KV state exactly.
- 20 prompts `bench/mtp-prompts/`, no-spec (A) and spec k=2 (B), `.work/prefill-v2/decode.txt`:
  - no-spec median 130.71 tok/s_gen (min 129.9, max 131.6) vs 131.29 before the lane (`.work/prefill-base/decode.txt`): -0.45%, within the 1% bound.
  - spec k=2 median 151.0, `identity_B` (spec vs no-spec) 20/20.
  - first generated token identical to `engine-base` on 20/20 prompts.
  - 64-token stream identical to `engine-base` on 18/20: p07-json diverges at generated token 1, p09-explain-gpu at token 37. Nine of the 20 prompts are long enough (>= 17 tokens) to take the prefill path; the flips are among those nine, and the set moved between v1 and v2 (p12-rust and p15-bash differed in v1 and match in v2), i.e. near-ties, not a defect.
  - Third opinion, numpy fp32 `tools/model-ref.py decode` on the same prompt ids (`.work/ref-p07-json.log`): the reference produces `198 79871 763 328 760 8252 469 34340 …` = the prefill path's stream on all 16 tokens computed, while `engine-base` produced `198 220 328 2034 …`. p09-explain-gpu, 40 reference tokens (`.work/ref-p09-explain-gpu.log`): reference == prefill path on all 40, `engine-base` differs at token 37 (383 vs 436). Both 64-token divergences are cases where the prefill path agrees with the fp32 reference and the old decode path does not.

The identity gate for a non-bit-exact numerics change is, as in the q8 and q4 rounds, agreement with `tools/model-ref.py`; that plus first-token identity 20/20 and 64-token identity on the four protocol prompts is what this lane claims. The prefill GEMM sums each row in a different order (16-wide WMMA products, scale on the block accumulator) from `q4rowb`, so bit-identity of long streams after a near-tie is not achievable without making the kernel the row kernel.

## Commits (lane branch)

| sha | subject |
|---|---|
| `a488766` | prefill-protocol: freeze the TTFT measurement and predictions before the first timed run (+ `bench/prefill-prompts/p{0032,0128,0512,1024}.tokens`) |
| `b6b5efd` | prefill kernels: bf16-WMMA GEMM over the Q4_0/Q8_0 weight layout, causal chunk attention, SSM conv/delta batched per chunk (+ test, bench, KERNELS.md) |
| `06fabb7` | prefill gemm: dequant nibbles via f32 cvt + perm truncation (+8% at n=1024) |
| `635b99b` | engine: prefill path — chunked prompt ingestion through the WMMA GEMM, chunk attention and batched SSM kernels (registry + engine) |
| `5de6446` | prefill-protocol: R3/R4 results |
| `2132b29` | prefill-protocol: p09 reference verdict |

No attribution trailers. Nothing committed from the main repo.

## Disclosures

- Parity references are fp64 computed inside `kernels/test_prefill.mojo` (the plan named the `tools/*-ref.py` numpy pattern); done in-test to stay inside the lane's file ownership. The numpy `tools/model-ref.py` was used unchanged for the end-to-end third opinion.
- `kernels/test_prefill.mojo`'s "attn prefill vs amar_attn_decode" gate is 1e-3 relative with a 1e-2 floor (max observed 2.7e-4, at an output of magnitude 9e-5, i.e. 3e-6 absolute — f32 noise).
- `$HOME/Projects/mojo-baro/.work/build-engine.log` (a throwaway build log in the main repo) was overwritten once at 10:35 through a `.work` symlink before I switched to `.work/build-engine-prefill.log`. No source or artifact in the main repo was touched.
- `tools/model-ref.py` needs ~46 GB RAM (f32 dequant cache of the 9B pack); running it concurrently with anything else got two of my waiters killed for memory. Only p07 and p09 (the two divergent prompts) were run; p12/p15 already match `engine-base` in the final build.
- The bench's n=16 timings moved between sessions (333-443 us in R2, 435-470 in R4) — latency-bound shape, not resolved.

## What is left

- GEMM is at 24.6 TFLOP/s (20% of the card's matrix peak) and 85 GB/s of weight stream at n <= 64. The next steps, in order of expected payoff: LDS-staged A tiles + k-loop unroll for memory-level parallelism (small n), then the 1024-row shape. This is what closes the 2.4x gap to llama.cpp at 512/1024.
- 32-token TTFT is now dominated by the weight stream at 31 rows (~45 ms of the 66) plus the megakernel token.
- `BARO_PREFILL_C` is exposed but only 16 and 1024 were exercised for identity; TTFT was measured at 1024 only (frozen choice).

## Questions

1. Identity policy: the plan says "20/20 on `bench/mtp-prompts/`" and in the same sentence "must not change the first generated token". First tokens are 20/20; 64-token streams are 18/20, and on both divergent prompts the fp32 reference sides with the prefill path. Is model-ref agreement + first-token 20/20 the accepted gate for this lane (as for q8/q4), or do you want the 64-token 20/20 literally (which would require the prefill GEMM to reproduce `q4rowb`'s summation order)?
2. Spend the next round on the GEMM (LDS staging / unroll, towards llama.cpp's prompt rate) or ship as is and move to the server lane's integration?

