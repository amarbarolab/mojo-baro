# ggml mul_mat_vec_q (Q8_0) vs amar_gemv_q8 on the Spark m=1 shapes

Preregistered 2026-09-08 ~20:05, before any run. Question: is our GEMV faster per shape than
llama.cpp master's (b10860) own kernel, or is the 1.20x decode win all launch count / graph?

## Arms, one stint, rocprofv3 kernel trace around each
- **ggml**: `bench/ggml-mmvq/mmvq_shapes` -- public ggml backend API, one `ggml_mul_mat`
  (Q8_0 [K,N] x F32 [K,1]) per shape, 20 warm + 200 timed iters; per-kernel us from the
  trace (`quantize_q8_1` + `mul_mat_vec_q<Q8_0,1,...>`), wall us/iter printed as a cross-check.
- **ours**: `.work/spark/spark-engine` 8-token run under the same rocprofv3 flags; `amar_gemv_q8`
  rows by shape (already measured 2026-09-08 17:00: 33.5/33.5/33.1 ffn+down, 21.0 qkv,
  15.6 o, 3.75 headgate, 386 lmhead us).
- Shapes: headgate 16x2560, o 2560x4096, qkv 6144x2560, ffn_gate 10240x2560, down 2560x10240,
  lmhead 131072x2560. Power cap / vddgfx read back first.

## Frozen prediction
- ggml `mul_mat_vec_q` on ffn_gate (26.2 MB): **30-40 us** (700-870 GB/s), plus quantize_q8_1
  ~2 us. Ours 33.5. Ratio ours/theirs per shape **0.9-1.15** on the three big shapes.
- Small shapes (headgate, o): theirs faster by up to 2x -- their block geometry covers more
  rows per block; ours is one wave per row with a fixed 4-unroll.
- lmhead: within 10 % either way (pure bandwidth, 335 MB).
- Verdict rule: if ours < theirs on all three big shapes, the kernel hook into llama.cpp is
  worth an M; if within +-10 % the decode win is launch-path only and the hook is closed.

## Result
(filled after the run)
