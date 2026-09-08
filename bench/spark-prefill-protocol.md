# Spark X2.5 4B prefill: prompt ingestion tok/s, ours vs llama.cpp master

Preregistered 2026-09-08 ~22:30, engine 5f5297d, before any timed run. Binds to
`bench/PROTOCOL-RULES.md` P1-P6. Question: prompt tokens per second at 64 / 256 / 1024 /
2048 ids, ours (token-by-token today; chunked prefill after the round) vs llama.cpp master
b10860 fast config, and does the prefill path change any generated token.

## Arms
- **ours/tbt**: `.work/spark/spark-engine`, prompt rows through the decode kernels one token
  at a time (`BARO_PF_CHUNK=0` once the flag exists). `prefill_s` = wall from the first
  prompt-token enqueue to the sync after the last prompt token (printed by the engine),
  `prefill rows` = n_prompt - 1 (the last prompt token is timed with generation).
- **ours/prefill**: same binary after steps 2-4, `BARO_PF_CHUNK=64` (and 128), rows 0..L-2
  in chunks through `amar_matmul_prefill_q8` + the SWA/gated prefill attention.
- **llama.cpp fast**: `~/llama.cpp-master/build/bin/llama-server`, `-fa on -ctk f16 -ctv f16
  -b 2048 -ub 512 -c 4096 -ngl 99 -t 8`, `/completion` on the ids, `n_predict 1`,
  `cache_prompt false`, `timings.prompt_ms` median of 3.
- **llama.cpp f32** (identity reference only): `-fa off -ctk f32 -ctv f32`, `n_predict 64`.
- Prompt set: `bench/spark-prefill-prompts/p{0064,0256,1024,2048}.txt` from
  `tools/spark-prefill-prompts.py` (mtp-prompts + docs text, sliced to exactly 64/256/1024/2048
  Mojo-tokenizer ids). Each arm tokenizes its own text; id-list identity is part of the gate.
- Read-back before any number is read (P1): engine sha256, power cap, vddgfx, `prefill rows`,
  chunk size; llama `props.json` per pass (fa, KV types, n_batch, n_ubatch), `prompt_n == N`.
  No receipt, no arm. Never both arms on the GPU at once. The RegesCore :8099 server is down
  for every timed run.

## Frozen prediction
- llama.cpp fast prompt tok/s: 64: 1500-3000 (launch-bound), 256: 3500-5500, 1024 and 2048:
  **5000-8000** (a 9B Q4 ran 3300 here; 4B Q8 has 2.2x fewer FLOPs per token).
- ours/tbt: **130-145 tok/s** at every length (decode path, 7 ms/token; attention cost at
  2048 is a few percent).
- ours/prefill after step 4: 64: 800-1500 (two chunks below the GEMM knee), 256: 1500-2500,
  1024: **2500-3500**, 2048: 2500-3500. Derivation: 8.2 GFLOP/token (4.1 B params x 2), GEMM at
  the Qwythos prefill kernel's 22.8 TFLOP/s = 2780 tok/s GEMM-bound; attention + elementwise
  + the last-token decode step eat the rest. Below 64 rows the path is weight-stream-bound
  (Qwythos R2), so chunk 64 is the floor.
- Ratio ours/prefill vs llama fast at 1024: **0.35-0.7**. Claim only if the band is cleared
  from below; the honest expected outcome is "3x faster than today, still behind llama.cpp".
- Identity: generated 64 tokens at each length must equal ours/tbt AND llama f32; a flip on
  the prefill arm is a WMMA accumulation-order near-tie and is REPORTED, not tuned around
  (round-2 rule: non-bit-exact changes need a logit-tolerance oracle).
- Decode must not move: `.work/spark/gate.sh` tok/s within +-2 % of 144 one-shot.

## Gates for step 1 (this commit)
G1 this file; G2 prompt files at exactly 64/256/1024/2048 ids (tools/spark-prefill-prompts.py:
64/256/1024/2048, all OK); G3 `tools/spark-prefill-ref.sh` end to end with id identity between
passes; G4 engine prints `prefill_s` + `prefill rows`, gate.sh 64/64 + 43/43; G5 committed
before any timing is read.

## Result
(filled after the runs)
