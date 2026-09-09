# Web-model consult, 2026-09-08 — six briefs to ChatGPT, Gemini, DeepSeek

Briefs and raw answers: `~/Projects/WebArea/BrowserExtension/AgentJack/.work/research/brief-{08..13}-*.md`
(18 answers, 3 recovered from the tab after short extractions). Structured digest of every claim,
number, disagreement and error: `exchange/web-model-consult-2026-09-08-digest.md`. This file is the
synthesis: what changes in our design, what is settled, what still needs a measurement.

Reliability, measured on this batch: **ChatGPT** did research with named sources on every brief
(gfx1100 attention tile data, llama.cpp function names, KV-quant papers, AlphaEvolve/Sakana/
KernelBench numbers) and recomputed our own arithmetic, once catching a wrong premise in a brief.
**Gemini** is fluent prior with one mechanism-level contribution (CWSR preemption) and one
load-bearing error (llama.cpp's prompt path). **DeepSeek** is prior with fabrications: MFMA on
RDNA3, a bf8 intrinsic for bf16, a 6-hour prefill from a CDNA3 number, checkpoint sizes off 18x,
benchmark names nobody else knows. Agreement of all three is a shared prior, not evidence.

## Settled here, after the consult (verified locally where marked)

1. **llama.cpp's Q4_0 prompt path on gfx1100 is int8 MMQ, not dequant + hipBLASLt.** Verified in
   `~/llama.cpp/ggml/src/ggml-cuda/mmq.cu` `ggml_cuda_should_use_mmq`: on RDNA3 with WMMA the
   default for Q4_0 is `return true` at every batch size (only Q2_K/Q6_K/IQ2 fall back above 128
   rows). Activations quantised to q8_1, weights stay q4, dot products on
   `v_wmma_i32_16x16x16_iu8`. Our prefill dequantises q4 to bf16 and runs bf16 WMMA. That, not the
   0.71 roof, is the 2.4x. (ChatGPT and DeepSeek right, Gemini wrong.)
2. **Flash-Next indexer cache at 1M is ~0.8 GB, not 12 GB.** My brief premise was wrong: the header
   says `attention.compress_ratios` = 4 on every attention layer and the indexer key is one shared
   D=128 head, so 12 x (1M/4) x 128 x 2 B = 768 MB bf16. Verified in the gguf header. (ChatGPT
   recomputed it; the other two accepted the wrong number.)
3. **Checkpoint size for the 9B is 26.25 MiB** (24 x (16x128x128x4 + 4x6144x4) B). ChatGPT and
   Gemini agree; DeepSeek's 1.5 MB is an 18x error. 32 per slot = 840 MiB, host RAM, not VRAM.
4. **Bandwidth arithmetic for the megakernel:** 6.2 GB / 7.5 ms = 827 GB/s; at 900 GB/s the token
   is 6.89 ms; the 0.58 ms gap ≈ the 260 µs of rmsnorm barriers + the 306 µs GDN regression
   (219 → 525 µs). The pools we already named are the whole gap. (ChatGPT's sum, our numbers.)

## Design changes to adopt

### Long-context attention (brief 08)
- Decode unit of work = one KV head + its four Q heads in one wave: K and V loaded once for four
  dot products and four accumulators. 32 lanes per 256-dim row, 8 dims per lane, one packed q4
  dword per lane, block scales broadcast in 4-lane subgroups. 2 rows in flight, benchmark 4.
- 128-token physical pages, 64 pages (8192 tokens) per split, ~123 splits per KV head at 1M; producer
  writes (m, l, O) partials (0.5 MB per head), one merge workgroup per head. No LDS in decode.
- Budget: block-32 fp16 scales make q4 4.5 bits/value, so 1M q4 KV = 9.2 GB and ~13 ms/token at
  700 GB/s, not 10. Target 650-780 GB/s, accept 600 first.
- KV precision: K8/V4 conservative, K4/V4 fast, chosen by a needle test at 128k+; keys need more
  bits than values (KIVI, KVQuant, KVTuner, RotateKV). The llama.cpp "lossless q4" claim was
  measured at 33k only. Optionally keep the ~2 % lowest-entropy sink heads at 8-bit.
- Prefill attention: 64 query rows x 4 waves (measured gfx1100 data: 43 → 93 TFLOP/s fp16 when
  BLOCK_M went 16 → 64; 128 rows regressed; 8 waves lost 2x; a 128-token K/V tile exceeds 64 KiB
  LDS). Plan on ~70 TFLOP/s for bf16 D=256 causal. Keep a lane-striped scalar D=256 kernel as a
  real competitor, since gfx11 WMMA replicates A/B operands across the wave.
- Cold 1M prefill stays 35-50 minutes on this GPU whatever we do to decode. Prefix reuse first.
- Sparsity without retraining: optional tier only. Quest-style 16-token selector blocks over
  128-token pages, sinks + recent window always kept; selector metadata is linear in T (62,500
  blocks, ~128 MB/layer at int8 for D=256), so measure before adding a hierarchy.

### Prefix reuse and chat contract (brief 09)
- Checkpoint = {position, prefix_hash(tokens[0:position]), model/ABI hash, generation, conv[24],
  delta[24]}. Valid only for the exact prefix; restore requires the hash, never position alone.
- Create at every 1024-token boundary and at every prompt end (the 7.9k system-prompt end is the
  one every agent call restarts from). 32 per slot in three classes: 8 pinned anchors, 16 recent,
  8 historical with exponential spacing. Never evict pinned. Never trim; a diverged prefix falls
  back to the nearest earlier snapshot.
- CI test, byte-exact: cold run of A||B vs restore(A)+replay(B): conv, delta, logical KV and
  next-token logits `memcmp` equal, at positions 0/1/1023/1024/1025/7900/8191/8192 and mutations
  at first token, checkpoint±1, last token; a corrupted hash must fail restoration. Slot save
  files carry the checkpoints.
- Speculative sampling: Leviathan accept min(1, p/q), residual max(0, p-q) on reject; penalties
  applied identically to draft and target from the same running context; counter-based RNG
  (Philox-style); distribution-exactness and same-seed equality are different guarantees. Turn
  speculation off by an EWMA of effective tok/s, not a fixed acceptance threshold.
- Stop/cancel: 64-byte aligned control block in pinned host memory {generation, cancel, state,
  produced}, polled once per token on device. One request at a time with prefix reuse; no
  continuous batching on one 24 GB card.
- Tool calls: state-machine lexer over the raw tag stream, copy vLLM's/llama.cpp's OpenAI
  streaming code; parallel tool calls and many-optional-parameter tools are the known breakage.

### Megakernel (brief 10)
- Remove most of the 64 rmsnorm+quant grid barriers: each GEMV workgroup computes the RMS scalar
  and quantises its own activation tile into LDS in its prologue (K-tiled for K=12288), WG barrier
  only. The invariant that survives is "previous GEMV output globally complete", one barrier per
  GEMV, not two.
- Barrier: two-word sense-reversing, leader-only atomics, release before arrival, acquire on
  sense; poll 4x bare, then `s_sleep 1`, then 2-4. Do not aggregate per CU (blockIdx ≠ CU). Floor
  2-4 µs; the 4 µs we see is tail arrival, not the atomic.
- GDN delta instability = register-allocation cost-model drift. Stamp
  `amdgpu_flat_work_group_size(512,512)`, sweep `waves_per_eu` and an explicit VGPR ceiling, put
  the delta step behind `noinline` (separate object if possible), ISA fingerprint in CI (scratch,
  ds_read/write, v_readlane, s_waitcnt counts), hand-written ISA if it still drifts. Only two state
  passes are needed if r = qᵀS is computed alongside u = kᵀS.
- Cooperative launch does not protect against compositor preemption (CWSR saves/restores waves
  regardless). On barrier timeout: discard the token, re-run from the last committed state, same
  kernel; never resume piecemeal, never a second code path.
- FFN-down spills: K-tile 2048-4096 double-buffered in ~40 KiB LDS, or keep packed bf16 (24 KiB
  for K=12288) in LDS and unpack 8 at the point of use; benchmark whether the spills cost anything
  before restructuring.

### Prefill (brief 11)
- Build the Q4 x Q8 int8-WMMA GEMM first, isolated at M=512, K=4096, N=12288: 4 waves, output
  tile N=64, M=64, K-tile 256, LDS `A_q8[64][256]` + `W_q8[64][256]` + scales; interleave four
  accumulator tiles; `s_waitcnt vmcnt(k>0)`; layer-major over chunks so weight tiles stay warm in
  Infinity Cache. Expected 1.15-1.4 ms for the 103 GOP shape vs ~2.5 ms today.
- Chunk size is disputed (512 vs 1024 vs 4096-8192); measure 512/1024/2048/4096 once the int8
  path exists. GDN chunked prefill is ~36-48 GFLOP per 1024 tokens across 24 layers, irrelevant
  unless its LDS traffic makes it so; FLA's chunk 64 formulation is the reference.
- 0.71 → 0.85 of the hipBLASLt roof is the practical ceiling for hand-written fp16 WMMA; not
  worth another round.
- Open numeric dispute to settle by microbench: RDNA3 int8 WMMA peak, 123 TOPS (ChatGPT) or
  ~245 TOPS (Gemini). It sets the target for the new kernel.

### Proposer loop (brief 12)
- The 27B is the wrong protocol more than the wrong size (Kevin-32B: 56 → 82 % correct after
  RL; KernelBench R1 12/36/2 % one-shot → 43/72/18 % with 10 feedback turns). Four identities are
  not a population.
- If we run it again: SEARCH/REPLACE with 3-10 verbatim anchor lines inside an evolve-block, 50-200
  editable lines plus helpers and one caller, `NO_CHANGE` allowed; stage A returns a
  machine-readable hypothesis (target, bottleneck, evidence, transformation, predicted ISA effect,
  expected phase gain), stage B edits; one transformation class per branch from a 12-entry menu.
- Gate additions: fresh randomised hidden fixtures after compile; state hashes at random
  positions, not 64 argmax tokens; interleaved C A A C timing with a bootstrap interval above
  1.02 (our +2 % over 3 runs is a 147 µs lottery, as 006 showed); canary buffers; resource caps.
- Decision: KernelBench-Verified 2026 puts the best frontier model at 0.88x geometric mean on
  realistic baselines. A human writes the three diagnosed kernels first (barrier fold, delta
  pinning, FFN-down spills). The loop returns, if at all, as a 100-200-proposal bake-off on one
  tractable region with three arms (27B new protocol / frontier / human), stop rule < 5 % valid.

### All models and Flash-Next (brief 13)
- One source, per-shape instantiation: 128-bit structural id from the pack header
  (architecture, dims, layer pattern, expert count, quant ABI) → `switch` → `Engine<Shape>`.
  Compile-time: tile shapes, unroll, LDS budgets, head dims, block-quant size. Runtime: layer
  count, T, page addresses, expert ids.
- Flash-Next decode: routed experts on CPU from pinned RAM. Correct traffic is 816 MB/token
  (41.8 GB x 10/512), so the DDR ceiling is 63.7 tok/s and 30-45 realistic; the cost that matters
  is 48 synchronisations per token, ~15 µs each way, not bytes. Mailbox structs per layer
  (job: expert ids + weights + q8 activation; result: bf16 y[2560]). Thread CPU work over output
  rows. Better than "all experts on CPU": whole expert layers resident on GPU, ~871 MB each,
  ~9 layers per 8 GiB, which removes 9 of the 48 handshakes.
- MoE prefill: stream all experts once per chunk over PCIe, 1.63 s per pass; crossover vs CPU at
  chunk 384-512 (from llama.cpp's 175 tok/s and the decode floor), so chunks of 2048 are fine.
  Two staging buffers of ~900 MB, one-layer lookahead, layer-major order.
- Keep IQ formats native on the CPU side (re-expanding IQ2_XXS/IQ1_M to q4 grows them 2.2-2.6x
  and kills the offload). llama.cpp's IQ x Q8 vector dots are the reference kernels.

## Corrections to our own notes
- The 12 GB indexer-cache figure in `2026-09-08-llamacpp-threads-long-context-moe.md` and in
  brief 13 is wrong; it is ~768 MB (compression 4, one shared key head).
- "1 GB of expert weights per token" is 816 MB; the tok/s ceiling is 64, not 50.

## Immediate actions, in order
1. Fold rmsnorm+quant into the GEMV prologues (barrier count 64 → ~0 for that class). Human.
2. Pin the GDN delta phase: attributes, noinline, ISA fingerprint in CI; recover 219 µs. Human.
3. Int8 WMMA microbench (settles 123 vs 245 TOPS), then the Q4 x Q8 prefill GEMM.
4. Rewrite the chat-engine lane brief around: shape table, 128-token paged K8/V4-or-K4/V4 KV,
   GQA-grouped decode kernel, checkpoint struct + byte-exact CI test, stop/cancel control block,
   speculative rejection sampling with shared penalties, tool-call lexer.
5. AgentJack: extraction came back short 3 of 18 times (role badge, research preamble, 13.6k of
   19.3k); the tab still holds the answer; ChatGPT virtualises old turns, so re-read soon.
