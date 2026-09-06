# Prefill protocol — prompt ingestion and TTFT

Binds to `bench/PROTOCOL-RULES.md` P1-P6. Question: how long from prompt token
ids to the first generated token (TTFT), for prompt lengths 32 / 128 / 512 /
1024 on the q4 pack, ours vs llama.cpp Q4_0-pure on the same token ids, and
does the prefill path change any generated token.

## Arms

- **ours/chunk8** — the pre-round engine: prompt rows through the decode
  window path, `m = min(8, remaining)` per window, `amar_matmul_skinny_q4rowb`
  MR=8, last prompt token through the megakernel. TMAX=128 admits only the
  32-token prompt; TMAX is raised (step 2 of the lane) before 128/512/1024
  can run on this arm.
- **ours/prefill** — the same engine with the prefill path: chunk C of prompt
  rows through `kernels/matmul_prefill.mojo` (bf16 WMMA over the Q4_0
  nibbles, scale applied per 32-block on the accumulator; exact products),
  causal prefill attention with online softmax (`amar_attn_prefill`), SSM
  conv + delta recurrence batched per chunk (`amar_ssm_conv_chunk`,
  `amar_ssm_delta_chunk`). The last `(L-1) mod 8` rows (8 when that is 0)
  still go through the decode window path so the MTP draft sees the same
  final window as before; the last prompt token still goes through the
  megakernel.
- **llama.cpp** — `llama-server` on `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf`
  with the flags of `tools/llama-mtp-prompts.sh` (`-c 8192 -ngl 99 -fa on
  -b 2048 -ub 512 -t 8 -ctk q8_0 -ctv q8_0`), no speculation,
  `/completion` with the token ids, `n_predict 64`, `cache_prompt false`.

## Measurement

- Prompt ids: `bench/prefill-prompts/p{0032,0128,0512,1024}.tokens`, the
  first N ids of `llama-tokenize` (Q4_0-pure gguf, same tokenizer as the
  packs; `p01-water` re-tokenised identically to its `.tokens`) over the
  concatenation of the 20 `bench/mtp-prompts/*.txt` plus two docs files.
- ours: `prefill_s` printed by the engine = wall time from the prompt ids on
  device to the first argmax synchronised on the host (includes the last
  prompt token's megakernel and the head GEMM). 3 runs per length, median,
  min-max beside it. Run: `BARO_PROMPT=bench/prefill-prompts/pNNNN.tokens
  gpu-wait run --priority 60 --timeout 1800 -- .work/engine`.
- llama.cpp: `timings.prompt_ms` of the response (prompt ingestion wall
  time), `timings.prompt_n` must equal N. 3 runs per length, median.
- Both engines alone on the GPU (llama-server down for ours; ours not
  running for llama.cpp); every run through `gpu-wait run`.

## P1 receipts (read back, never assumed)

- ours: `prompt tokens: N` (N == length), `pack q4 trunk: True`,
  `BARO_MEGA: True`, `BARO_SPEC: False`, `TMAX:` and `prefill chunk:` lines
  printed by the binary (added with the integration), `mega fail word: 0`,
  and the binary rebuilt in the same command as the timed run for any
  comptime change (TMAX, C).
- llama.cpp: `GET /props` (`n_ctx`, `n_batch`, `n_ubatch`, flash_attn) saved
  next to the results; per-request `timings.prompt_n == N`, `draft_n`
  absent/0.
- kernel bench: grid/block dims and template params printed by
  `bench/bench_prefill.mojo`.

## Frozen predictions (commit before the first timed run)

TTFT in ms:

| length | ours/chunk8 | ours/prefill (C=1024) | llama.cpp Q4_0-pure |
|---|---|---|---|
| 32 | 57 | 30 | 40 |
| 128 | 200 | 55 | 80 |
| 512 | 780 | 170 | 250 |
| 1024 | 1550 | 330 | 500 |

Reasoning: chunk8 streams the 3.75 GB q4 trunk once per 8 rows at ~2.3x the
m=1 window time (mrow receipt), ~12 ms/window; prefill is compute-bound above
~64 rows at 13.8 GFLOP/token on a matrix path measured at 70% of 512
FLOP/clk/CU (`wmma-fp16-protocol.md`), taken at ~50 TFLOP/s effective after
dequant and scale epilogue; llama.cpp from its MMQ prompt rate on this card
(~2000 tok/s at Q4_0, ub 512).

Kernel arm on the ffn shape (N=12288, K=4096, q4, us per GEMM, NBUF=8
rotation so the weight stream is cold):

| n | WMMA (predicted) | wave-per-row n/8 passes (predicted) |
|---|---|---|
| 16 | 60 | 270 |
| 64 | 130 | 1100 |
| 256 | 500 | 4300 |
| 1024 | 1900 | 17000 |

Prediction: WMMA wins at every n >= 16; the wave-per-row kernel cannot hold
more than 8 rows of accumulators per lane (MR=8 is its register ceiling,
`mrow-gemm-protocol.md`), so its n-row variant IS the n/8-pass loop.

Decode must not move: no-spec 20-prompt median tok/s_gen before and after,
same session, within 1%; `bench/mtp-prompts/` identity 20/20 and
`ref-tokens-64` 64/64 on the q4 pack with the prefill path active.

## Results

(filled after the runs; see the lane report for the receipts)
