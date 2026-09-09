# Fanout extract: 18 web-model answers, briefs 08-13

Source: `~/Projects/WebArea/BrowserExtension/AgentJack/.work/research/brief-{08,09,10,11,12,13}-*.md`.
Six question files matched (each has a gfx1100/mojo-baro-relevant slug; a second slug per NN
— `local-serving`, `payments-auth` — exists in the same directory but belongs to a different,
unrelated fanout batch and was excluded): `08-long-context-attention-gfx1100`,
`09-hybrid-prefix-reuse-chat-contract`, `10-megakernel-decode-pools`,
`11-prefill-throughput-gap`, `12-kernel-proposer-loop`, `13-multi-model-moe-static-split`.
No answer file was a fragment (all > 1000 chars; range 5.5-24.5 KB).

---

## Brief 08 — long-context-attention-gfx1100

Question: decode kernel design for T up to 1M on gfx1100 (paged/quantised KV), deep-prefill
WMMA D=256, KV quant format (K vs V bits), post-hoc sparsity without retraining.

### 1. Consensus
- Decode at 1M is bandwidth-bound; must fan T out across many workgroups (Flash-Decoding
  style split-K over pages) because a naive per-KV-head launch leaves most of the 96 CUs
  idle. (prior — all three restate Flash-Decoding)
- Merge online-softmax partials (m, l, acc) with a two-pass scheme: producer WGs write
  partials, one small reducer pass merges; atomics on the accumulator are rejected by all
  three as worse than a clean merge pass. (prior)
- For decode, do NOT stage K/V through LDS — read straight from global into registers,
  since each K/V token is touched exactly once per query. (prior)
- K tolerates fewer bits worse than V — quantization error in K perturbs the pre-softmax
  score and can flip which token dominates; V error is "just" a linear scaling error.
  (prior — chatgpt, deepseek, gemini all state this, though chatgpt then complicates it)
- Softmax/attention-sink degradation, not raw MSE, is the real 1M failure mode for
  quantised K. (prior)
- Qwen3.8/Flash-Next's QSA indexer is *trained*; naive post-hoc block-sparse selection
  (Quest-style, min/max per block) is promising but not equivalent, and pruning cost must
  not itself scale linearly with T without care. (prior)

### 2. Evidence brought
- chatgpt: cites a "recent vLLM gfx1100 experiment" (GitHub) — BLOCK_M 16→64 raised fp16
  attention from ~43 to 93 TFLOP/s; BLOCK_M=128 regressed, 256 spilled registers badly;
  4 waves beat 8 waves ~2x; 128-token K/V tile exceeds 64 KiB LDS; ~107 TFLOP/s rocBLAS
  GEMM throughput measured on the same card.
- chatgpt: AMD quotes 960 GB/s raw BW and "123 TFLOP/s FP16 matrix peak"; targets
  650-780 GB/s (68-81% of raw) for decode streaming, "~600 GB/s acceptable first silicon".
- chatgpt: KIVI (per-channel K / per-token V quant), KVQuant (<0.1 PPL degradation at 3-bit,
  per-channel pre-RoPE K), KVTuner (key precision matters more than value; Qwen2.5-7B
  needed ~4-bit avg precision), RotateKV (Hadamard rotation, protects attention sinks,
  "robust 2-bit results"), vLLM 2026 eval of K8V4/K4V4/K3V4/K3V3 variants.
- chatgpt: Quest paper — 16-token pages, traffic ~ 1/page_size + K/num_pages; at 1M/16-tok
  pages that's ~62,500 pages scored per query. InfLLM: training-free extrapolation to
  1,024K reported successful. MagicPIG: argues top-K attention itself can degrade quality
  (uses LSH instead).
- chatgpt: Qwen3.8-Flash-Next QSA — 4-head D=128 MQA indexer, compression ratio 4, budget
  512 microblocks / 2048 tokens (cites Hugging Face, Qwen docs).
- chatgpt: attention FLOP arithmetic reproduced from the prompt: 8×2×2×(1M²/2)×256×16 =
  1.31e17; at 55/70/80 TF gives 39.7/31.2/27.3 min pure-attention cold-prefill time.
- deepseek: "peak of 1024 FLOPS/clock/CU" for INT4 on RDNA3 (unsourced number). Cites an
  optimized flash-attention on "a CDNA3-class chip (MI300X)" reaching ">10 TFLOPS" then
  extrapolates 5-7 TFLOPS for RDNA3 D=256.
- deepseek: cites Quest via footnote "-10" for ANN/FAISS-based block search with
  d_search=128, top-2048 positions, claims "O(log T)" selection cost via ANN index.
- gemini: RDNA3 122 TFLOPS bf16 compute figure (vs chatgpt's 123). Cites Quest as validated
  on "128K context tasks using standard Llama-2 models," selection via per-block min/max.
- gemini: bandwidth target 700-750 GB/s (75-80% of peak); prefill 50-60 TFLOPS (40-50%
  utilization of 122 TFLOPS); 1M cold-prefill attention time ≈ 2,600 s = 43 min.

### 3. Disagreements
- Realistic prefill TFLOPS for D=256 flash-attention on gfx1100: chatgpt 55-80 TF (uses
  cited 93 TF gfx1100 fp16 measurement as anchor) vs deepseek 5-7 TF vs gemini 50-60 TF.
  chatgpt and gemini roughly agree; deepseek is an outlier ~10x lower.
- Cold 1M prefill wall time: chatgpt ~30-40 min (attention only) + ~6 min rest = 35-50 min
  total; gemini ~43 min (attention only, "brutal reality"); deepseek ~6 hours (21,666 s) at
  its much lower 6 TFLOPS assumption.
- Decode page size: chatgpt 128 tokens; deepseek 64 tokens; gemini 64 tokens (16
  pages/workgroup ≈ 1K tokens/WG).
- K vs V precision recommendation: chatgpt ends up recommending K8/V4 (or K6/V4) as the
  conservative asymmetric mode; gemini recommends "K in fp16 or fp8 while quantizing V to
  4-bit" — same K>V-precision direction but different concrete bit assignment.
- Sparsity selection granularity: chatgpt proposes a hierarchy (16-token selector block →
  256 → 4K superblocks); deepseek proposes ANN/FAISS with 64-token blocks and d_search=128
  projections; gemini proposes flat 64-128 token blocks with min/max metadata, no ANN.
- "Lossless" KV4 for hybrid models: deepseek says hybrid models' SSM layers "act as a
  buffer" making 4-bit "nearly lossless"; chatgpt explicitly refuses the word "lossless"
  and says there's no theorem that surrounding GDN layers repair a bad attention retrieval;
  gemini says hybrids "do NOT tolerate 4-bit KV losslessly" for needle-in-haystack because
  they have fewer attention layers to fall back on — direct three-way disagreement.

### 4. Concrete recommendations
- chatgpt: 128-token physical page; 64-128 splits/KV-head at 1M (~123 splits/head); 2-4
  waves/WG; fuse 4 Q heads per KV-head unit of work; 2 token-rows in flight per wave
  (benchmark 4); block-32 q4 KV with fp16 scales as first production format (144 B/vector
  for D=256, "effective precision 4.5 bits/value"); model-level prefill chunk 4096 tokens
  (range 2K-8K); prefill tile variants: conservative Q32/K32/4waves, throughput Q64/K32-64/
  4waves.
- deepseek: 64-token physical page; INT4 dequant via v_mfma (see Errors); K8/INT8 for
  "bottom 2% of heads by entropy" kept at higher precision; sparsity block size ~64 tokens;
  chunked prefill N=2048.
- gemini: 64-token page, 16 pages/WG (~1K tokens/WG); 4-8 rows in flight per wave (32-64
  VGPRs of 256 available); prefill tile BLOCK_M=BLOCK_N=BLOCK_K=64; chunk N=2048 for
  prefill weight reuse; sparsity block size 64 or 128 tokens (contiguous, for coalescing).

### 5. Errors / suspect
- deepseek, decode section: "you should read the packed INT4 values straight from global
  memory into registers using v_mfma instructions. RDNA3 has native hardware support for
  INT4 matrix multiplication through WMMA 16x16x16 tiles" — MFMA is a CDNA instruction
  family (MI-series); gfx1100/RDNA3 has WMMA, not MFMA. The sentence names both in the same
  breath, self-contradictory as well as factually wrong for gfx1100.
- deepseek, prefill section: extrapolates "~5-7 TFLOPS" flash-attention forward for gfx1100
  D=256 by starting from an MI300X (CDNA3) number, on hardware AMD itself rates at ~123
  TFLOP/s BF16/FP16 matrix peak (per chatgpt's own citation) — a ~15-20x underestimate
  relative to chatgpt's and gemini's independently-derived 50-90 TF estimates, and
  inconsistent with the 93 TFLOP/s fp16 gfx1100 measurement chatgpt cites for a similar
  shape.
- gemini's 16×256 "128 VGPRs" line ("holding an entire K tile in registers (16×256)
  consumes 128 VGPRs") glosses over per-lane vs per-wave register accounting without
  showing the derivation — flagged as unverifiable, not confirmed wrong.

---

## Brief 09 — hybrid-prefix-reuse-chat-contract

Question: recurrent-state checkpointing for GDN+attention hybrid, exact speculative
sampling under penalties/temperature, stop/cancel/scheduling for a persistent kernel,
tool-call SSE streaming.

### 1. Consensus
- Recurrent (conv+delta) state cannot be trimmed like attention KV — it's path-dependent;
  checkpoints must be full state snapshots taken at fixed intervals plus prompt end.
  (prior)
- The byte-exact test for restore correctness: restored-state + replay must be bit-
  identical to a cold prefill of the same tokens, checked via logits, not just
  argmax/text. (prior)
- Correct speculative decoding under sampling = Leviathan-et-al. rejection sampling:
  accept draft token x with prob min(1, p(x)/q(x)); on reject, resample from the
  normalized residual max(0, p(x)-q(x)). (prior — all three state this identically)
- Repeat/presence penalties must be applied identically, using the same running-context
  state, to both draft (q) and target (p) distributions before computing the
  accept/reject ratio — otherwise acceptance collapses. (prior)
- Stop-on-EOS/max_tokens and cancellation should be handled device-side via a polled flag
  in host-pinned/mapped memory, not a host round-trip per token. (prior)
- One-request-at-a-time with prefix reuse is the right design for this single-24GB-GPU
  agent workload, not continuous batching — KV cache doesn't share across independent
  requests and weights+KV+activations already consume most of VRAM at long context.
  (prior)
- Tool-call streaming should use a lexical/state-machine parser over the raw
  `<tool_call>...</tool_call>` / `<think>...</think>` tags, not regex on accumulated text,
  and reference vLLM's/llama.cpp's existing OpenAI-compatible streaming code rather than
  writing from scratch. (prior)

### 2. Evidence brought
- chatgpt: computes S_delta = 16·128·128·4 = 1,048,576 B/layer, S_conv = 4·6144·4 =
  98,304 B/layer → S_ckpt = 24×(1,048,576+98,304) = 27,525,120 B = 26.25 MiB. Cites
  llama.cpp's reported "~149.6 MiB checkpoint for the 48-recurrent-layer Qwen3.8-27B" and
  "defaults to up to 32 checkpoints per slot" (GitHub). Cites an "8192-token default
  minimum checkpoint spacing" issue and a bug where "an edit jumping from ~80K back to
  ~30K fell outside the checkpoint window and caused all checkpoints to be erased"
  (GitHub). Cites a llama.cpp bug where slot serialization restores tokens but not
  `server_prompt::checkpoints`. Cites vLLM's RejectionSampler description ("accepted
  tokens + recovered token after rejection + target bonus token") and its sampler pipeline
  order (logits → processors → penalties → temperature → min-p → top-k/top-p → sample).
  Cites a llama.cpp report of "MTP changing output under temperature=0."
- deepseek: computes conv "~6144 × 4 taps ≈ 24.5 KB" and delta "16×128×128×f32 ≈ 1 MB"
  then a table total "~1.5 MB" per checkpoint, "keep at most 8 checkpoints (12 MB)" — see
  Errors. Cites "MTP improves throughput by ~100-130% with acceptance rates of 0.68-0.82"
  at temp=1 "with Qwen3.5-397B" and "vLLM ... reporting ~130% throughput improvement on
  Qwen3.5-397B."
- gemini: computes conv 6144×4×4B = 98 KB, delta 16×128×128×4B = 1.04 MB, total/layer
  ~1.14 MB, snapshot 24×1.14 MB = ~27.5 MB; "keeping 16 snapshots per slot costs only
  ~440 MB of VRAM." Cites "llama.cpp used a rigid --checkpoint-every-n-tokens defaulting
  to 8192" bug (Particula Tech) with the same "n_past=0, nuked every checkpoint" failure
  mode as chatgpt's citation.

### 3. Disagreements
- Checkpoint size: chatgpt 26.25 MiB and gemini ~27.5 MB agree closely; deepseek's own
  table claims "~1.5 MB" total for 24 layers — an order-of-magnitude discrepancy against
  the other two answers (and against deepseek's own per-layer component numbers, see
  Errors).
- Eviction policy: chatgpt proposes a 3-tier scheme (8 pinned / 16 recent / 8 historical,
  exponential spacing, 32 total, never evict pinned); gemini proposes hash-indexed
  overwrite-only-divergent-checkpoints in a ring buffer, no tiering; deepseek proposes a
  simple exponentially-spaced ring buffer (pages 0,1,2,4,8,16,32,64) with only 8
  checkpoints kept, evicting everything after an edit point and recomputing forward.
- Reproducibility/RNG for speculative sampling: chatgpt insists distribution-exactness and
  same-seed token-for-token equality are two *different* guarantees, and that a
  counter-based (Philox-style) generator is needed with per-purpose counters; it explicitly
  says "a speculative run need not produce the same sampled realization as ordinary
  autoregressive decoding with the same seed." Gemini instead claims reproducibility is
  achieved by seeding "hash(base_seed + sequence_position)" without noting this
  distinction. Deepseek says simply "a single random seed with deterministic sampling
  paths" makes accept/reject "deterministic," which is the weakest/least precise claim.
- When to turn off speculation: chatgpt gives a dynamic EWMA throughput-ratio rule
  (effective tok/s(spec) <= effective tok/s(non-spec) × ~1.03); gemini says turn off when
  temperature > 1.0 or min-p very low, citing acceptance dropping "below 30%"; deepseek
  gives a static threshold "draft_acceptance_rate < 0.5 or draft_cost > target_cost * 0.3."
- What vLLM/llama.cpp actually do for penalized speculative sampling: gemini claims they
  "often skip rejection sampling complexities for penalized sampling" and "fall back to
  temperature=0 ... for drafts, or ... ignore penalties on the draft tokens" — this
  contradicts chatgpt's claim that vLLM's sampler pipeline explicitly applies
  processors/penalties before sampling in a fixed, documented order.

### 4. Concrete recommendations
- chatgpt: checkpoint at every 1024-token boundary + exact prompt end (~7900 for this
  workload); 32 checkpoints/slot (~840 MiB/slot in host RAM, not VRAM); RecurrentCheckpoint
  struct with u64 position, Hash256 prefix_hash, Hash256 model_cache_abi, u64
  branch_generation; restore rule requires prefix_hash match, not just position <= L;
  Control block: `alignas(64)` struct with u64 request_generation, u32 cancel, u32 state,
  u32 produced.
- deepseek: 8 checkpoints (12 MB by its — erroneous — sizing); exponential spacing
  0,1,2,4,8,16,32,64; test invariant = bit-identical logits for first 100 tokens at temp=0.
- gemini: snapshot at exact prompt-end plus 1024-token boundaries; 16 snapshots/slot,
  ~440 MB VRAM; 64-bit MurmurHash3 of exact token-ID prefix as the checkpoint key; 4-byte
  cancel flag in `hipHostMalloc` pinned memory, polled once per token iteration.

### 5. Errors / suspect
- deepseek's checkpoint table: line "Conv state (24 layers) ~6144 × 4 taps ≈ 24.5 KB" and
  "Delta state (24 layers) 16 × 128 × 128 × f32 ≈ 1 MB" followed by "Total per checkpoint
  ~1.5 MB" — internally inconsistent. The delta-state row is explicitly labeled "(24
  layers)" but the value given (≈1 MB) matches a *single* layer's delta state (16×128×128×4
  = 1,048,576 B), matching chatgpt's and gemini's per-layer figures. Summed correctly over
  24 layers this is ~27 MB, not 1.5 MB — deepseek's own downstream recommendation ("keep at
  most 8 checkpoints (12 MB)") is built on this ~18x-too-small number.

---

## Brief 10 — megakernel-decode-pools

Question: grid barriers for 96 workgroups, pinning LLVM AMDGPU codegen for a regressing
GDN loop, co-residency with the desktop compositor, q4 weight-stream GEMV bandwidth gaps,
FFN-down VGPR spills.

### 1. Consensus
- The sense-reversing global-atomic-counter barrier design (WG leader does the atomic,
  release-before-arrival / acquire-after-sense) is correct in shape; use `s_sleep` with
  increasing backoff rather than tight-spinning on the atomic. (prior)
- The 64 RMSNorm+quantise grid barriers can be eliminated: fold RMS+quantise into the
  consuming GEMV's own prologue (each WG computes its own RMS/quant locally, or reads
  scale from the previous kernel's global write, and uses a WG-local barrier instead of a
  grid barrier). (prior — chatgpt and gemini state this near-identically; deepseek agrees
  the split is "artificial")
- `hipLaunchCooperativeKernel` exists on ROCm/gfx1100 and legalizes grid-wide sync
  (`cooperative_groups::this_grid().sync()`), but it does NOT guarantee immunity from
  desktop-compositor preemption (CWSR). (prior — chatgpt and gemini agree explicitly;
  deepseek disagrees, see Disagreements)
- A barrier timeout must not resume with whatever partial work the surviving workgroups
  did — the whole token attempt must be discarded/re-run from last committed state, never
  resumed piecemeal. (prior)
- No LDS staging for q4 weights in the m=1 GEMV — global → VGPR → dequant → FMA directly;
  LDS is for activation reuse, not weight streaming, since weights aren't reused across
  waves at m=1. (prior)
- The 142 VGPR spills on FFN-down are plausibly costly and should be fixed by tiling K and
  consuming unpacked activation values immediately (small live microtile) rather than
  materializing the whole unpacked vector. (prior)

### 2. Evidence brought
- chatgpt: derives the whole-token bandwidth arithmetic from the prompt's own numbers:
  6.2 GB / 7.5 ms = 827 GB/s (measured), 6.2/900 = 6.89 ms, 6.2/830 = 7.47 ms → gap is
  "only about 0.58 ms/token." Sums the already-identified overheads: 260 µs (RMS/quant
  barriers) + 306 µs (GDN 219→525 regression) = 566 µs, calling this "almost exactly" the
  830→900 GB/s gap. Cites LLVM AMDGPU docs: `amdgpu-num-vgpr` is "deprecated in favor of
  amdgpu-waves-per-eu"; gfx11 LLVM naming is `global_load_b128` (successor of
  `global_load_dwordx4`). Cites HIP docs for `hipHostMallocMapped`/`hipHostMallocCoherent`
  and for `hipLaunchCooperativeKernel`/`hipErrorCooperativeLaunchTooLarge`/
  `hipDeviceAttributeCooperativeLaunch`. Cites current Transformers' Gated-Delta reference
  recurrence (S←e^gS; u=k^TS; δ=β(v-u); S←S+kδ^T; o=q^TS) matching the brief's own
  formulation (GitHub). Derives algebraically that only 2 state passes (not 3) are needed
  by pre-computing r=q^T S_d alongside u=k^T S_d.
- deepseek: cites `rocm/rocWMMA`-style claim that "hipLaunchCooperativeKernel ... actually
  guarantees residency" (footnote -14, unnamed source). Suggests measuring eviction via
  the `SQ_WAVES` performance counter. States FFN-down vs LM-head gap is because "FFN-down:
  12288→4096 (outputs fewer ... more VGPR pressure). LM head: 4096→128000 (outputs more,
  less pressure)" — no cited source, own reasoning. Cites `__builtin_amdgcn_bf8_unpack` for
  bf16 unpacking (see Errors).
- gemini: derives `s_sleep 1` timing: "stalls the wave for ~64 clock cycles. At ~2.5 GHz,
  this is ~25 ns." Cites CWSR (Compute Wave Save/Restore) as the actual mechanism by which
  the compositor preempts compute waves, dumps VGPR state to VRAM, and restores after — the
  most specific mechanistic claim among the three on this question. Proposes leaving 1 CU
  idle (95 of 96 WGs) as a compositor-scheduling escape valve. Cites `v_dot2_f32_bf16` as
  an RDNA3 instruction for packed-bf16 dot product (see Errors — unverified). States "RDNA3
  supports v_wmma_f32_16x16x16_bf16"; states clock throttling from "~2.5 GHz to ~1.8-2.0
  GHz" under sustained WMMA to stay within a "355W board limit."

### 3. Disagreements
- Cooperative-launch residency guarantee: deepseek states flatly "It actually guarantees
  residency of the requested grid size" — contradicted by chatgpt ("It does not mean ...
  these 96 workgroups now own the GPU and cannot be preempted by graphics ... HIP does not
  document cooperative launch as a CU reservation mechanism") and gemini ("does not
  guarantee absolute physical residency against the OS compositor ... CWSR").
- Barrier-timeout fallback: chatgpt says never resume, poison the whole token, re-run from
  last committed state with token-level double buffering; deepseek says re-run on "a
  non-persistent path with a fresh kernel launch"; gemini says abort and relaunch the
  *same* persistent kernel for that token, explicitly rejecting a second code path
  ("maintaining two code paths for a megakernel is an engineering nightmare") — direct
  disagreement with deepseek's non-persistent-path recommendation.
- GDN delta-step kernel shape: chatgpt and gemini both describe it as a pure streaming/FMA
  elementwise pass over the 1 MiB state (no WMMA — "you can't move the 64 KiB/head state
  through LDS... load a small strip into VGPRs"); deepseek's formulation explicitly invokes
  WMMA ("16 lanes process 128 columns using WMMA with dequantized weights") for what is a
  rank-1 state update, not a tiled matmul — see Errors.
- Cause of the 830→900 GB/s whole-token gap: chatgpt attributes it almost entirely to the
  barrier + GDN-regression overhead (566 µs vs a 580 µs theoretical gap — "too strong to
  ignore"); gemini attributes it to "260 µs ... barriers ... [+] GDN delta-step (525 µs =
  7%)" reaching a similar conclusion but a different arithmetic breakdown; deepseek instead
  frames the 830 GB/s figure itself as "already at ~86% of peak 960 GB/s — excellent" and
  recommends "eliminate the 260 µs barrier overhead and the 142 VGPR spills," not
  addressing the GDN regression's contribution at all.
- Rows-per-wave for the q4 GEMV: chatgpt recommends benchmarking 1-2 (expects 1 may win at
  K=12288); deepseek recommends 4 rows per wave flatly ("4 rows per wave to hide latency");
  gemini doesn't give a specific rows/wave number for this exact question.

### 4. Concrete recommendations
- chatgpt: `GridBarrier{u32 arrivals; u32 sense;}`; polling schedule "first ~4 polls: no
  sleep; then: s_sleep 1; long tail: s_sleep 2-4"; cost floor estimate "~1-2 µs
  exceptional, 2-4 µs realistic" (explicitly an engineering estimate, not measured);
  `__attribute__((amdgpu_flat_work_group_size(512,512)))`, test
  `__attribute__((amdgpu_waves_per_eu(1,1)))`; sweep `amdgpu_num_vgpr(N)` for N in
  {64,72,80,88,96,...}; put the GDN delta-step in a `noinline` function, possibly a
  separate translation unit, escalate to hand-written ISA if that fails; Ktile=2048 (8 KiB
  f32 / 4 KiB bf16) or 4096 (16 KiB f32 / 8 KiB bf16) for FFN-down, double-buffered within
  a ~40 KiB LDS budget; or store raw packed bf16 in LDS (12288×2=24 KiB, fits whole K
  vector) and unpack 8 values at point of use.
- deepseek: `amdgpu_flat_work_group_size(256,256)`, `amdgpu_waves_per_eu(2,4)`,
  `amdgpu_num_vgpr(128)`, `noinline`; request 16 CUs (1536 threads) with LDS < 8 KB/WGP so
  "4 workgroups per WGP" fit, "making eviction unlikely"; `global_load_dwordx4`, 4 rows in
  flight per wave, no LDS staging; tile K to 2048 for FFN-down.
- gemini: `amdgpu_waves_per_eu(MIN,MAX)` e.g. (4,4) for a hard 64-VGPR/wave contract;
  `amdgpu_flat_work_group_size(512,512)`; `noinline` on delta-step and GEMV; 1 WG per GDN
  head, no LDS staging of the 1 MB state; leave 1 CU idle (94-95 of 96 WGs launched) as a
  compositor-scheduling defense; abort-and-relaunch-persistent-kernel-only on timeout, no
  second code path; for FFN-down, stage the whole 24 KiB activation vector in LDS in the
  phase prologue and consume ≤32 VGPRs per iteration via `v_dot2_f32_bf16`.

### 5. Errors / suspect
- deepseek: `float8_t unpack_8_bf16(uint32_t packed) { return (float8_t)
  __builtin_amdgcn_bf8_unpack(packed); }` — `bf8` denotes 8-bit brain-float (a distinct,
  narrower format), not `bf16` (16-bit brain-float, the format actually in use per the
  brief's "bf16 activations staged in LDS"). Using a bf8-unpack intrinsic to unpack bf16
  values is a type/format mismatch; suspect this line is fabricated or copy-confused.
- deepseek: "The correct barrier implementation ... hipLaunchCooperativeKernel API is
  available on RDNA3 and guarantees all workgroups in the grid are resident
  simultaneously" and again "It actually guarantees residency" — contradicted by the
  other two answers' more specific CWSR-preemption discussion; flagged as the weaker/wrong
  claim given gemini's concrete mechanism (CWSR dumps/restores waves regardless of
  cooperative-launch status) is consistent with AMD's documented consumer-GPU scheduling
  model.
- deepseek's GDN delta-step formulation invoking WMMA for a per-token rank-1 state update
  ("16 lanes process 128 columns using WMMA with dequantized weights") is a shape mismatch
  — WMMA is a 16x16x16 tiled-matmul instruction; a single-token outer-product state update
  has no batched-matmul structure to feed it, which is presumably why chatgpt and gemini
  both reject WMMA here in favor of plain FMA streaming.
- gemini's `v_dot2_f32_bf16` — RDNA3's documented dot instructions are for f16
  (`v_dot2_f32_f16`) and integer types; a bf16-specific packed dot instruction of this exact
  name was not independently corroborated by the other two answers or by any cited
  document in this batch — flagged as unverified, possibly invented.

---

## Brief 11 — prefill-throughput-gap

Question: why llama.cpp's Q4_0 HIP prompt path beats a hand-written bf16-WMMA prefill path
by ~2.4x on gfx1100; GDN chunked-prefill cost; chunk-size/pipeline design; whether 0.71 of
hipBLASLt roof is a hard ceiling for hand-written WMMA.

### 1. Consensus
- gfx1100 has native INT8 WMMA (`v_wmma_i32_16x16x16_iu8`); an activation-quantize-to-int8
  (Q8) + INT8-WMMA path avoids materializing a full BF16 expansion of q4 weights the way
  the current bf16-WMMA path does. (prior on the mechanism's existence, though not on
  whether it's what llama.cpp actually runs — see Disagreements)
- GDN chunked-prefill FLOPs are trivial relative to the FFN GEMMs — all three explicitly
  say the recurrence core should not be a meaningful fraction of prefill time for a
  well-implemented kernel. (prior)
- Do not fuse/overlap prefill and the first decode token into one persistent launch —
  their resource profiles (compute-bound/high-occupancy vs memory-bound/persistent-spin)
  conflict and would poison codegen for both. (prior)
- 0.71 of vendor hipBLASLt roof cannot by itself explain a 2.4x end-to-end prefill gap —
  even reaching 0.85-0.90 only buys ~1.2x on the affected GEMM, so the dominant gap must be
  architectural (kernel choice: INT8 MMQ vs BF16 dequant-then-WMMA), not micro-tuning.
  (prior — chatgpt states this arithmetic explicitly; deepseek and gemini implicitly agree
  by both pointing at the same INT8/MMQ mechanism as the primary lever)
- LDS bank-conflict/padding and double-buffering with careful `s_waitcnt` (not blanket
  `vmcnt(0)`) are the remaining legitimate levers for the BF16 WMMA kernel itself. (prior)

### 2. Evidence brought
- chatgpt: cites current llama.cpp docs — "MMQ is enabled by default on GPUs with INT8
  matrix-core support, including RDNA3+"; `quantize_mmq_q8_1_cuda()` activation-quant
  kernel; RDNA3 path reaches `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` (GitHub, named
  functions). Cites `mmq-config-rdna3.cuh` tile family: "I=64 output rows, J=16/32/48/64/
  96/128 activation rows, K_vram=256, 128 threads," "MMQ_ITER_K=256." Cites llama.cpp
  defaults: `n_ubatch=512` physical batch, `n_batch=2048` logical. Cites AMD GPUOpen WMMA
  guide: "4 VGPRs/lane for IU8 versus 8 for FP16/BF16 inputs," 123 TOPS INT8 matrix / 123
  TFLOP/s FP16 matrix / 246 TOPS INT4 (7900 XTX). Reproduces the GEMM-FLOP math:
  2×1024×4096×12288 = 103.1 GOP → at 123 TOPS floor = 0.838 ms; targets 75-90 TOPS effective
  (61-73% of peak) → 1.15-1.38 ms. FLA production kernel names cited:
  `gdn_gate_chunk_cumsum`/`chunk_local_cumsum`, `chunk_gated_delta_rule_fwd_intra`,
  `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o`; FLA "defaults to C=64, only allows 16,32,
  or 64." GDN FLOP estimate: "~36-48 GFLOP/1024-token prompt" across 24 layers, "not
  TFLOPs." Notes the measured incremental cost between 512→1024 tokens is 345 ms (theirs)
  vs 148 ms (llama.cpp) — used to argue the gap is in large-M steady-state, not launch
  overhead.
- deepseek: cites "MMQ_X = 64, MMQ_Y = 128, and NWARPS = 8" as llama.cpp's RDNA3 tile
  config (conflicts with chatgpt's I/J/K numbers, see Disagreements). Cites "some users
  reporting over 1,200 tok/s for a 7B model" with `-DGGML_HIP_ROCWMMA_FATTN=ON`. Cites "a
  27B parameter model, a 1,024-token prefill call for GDN took ~405 µs" (footnote -8, no
  named source). Cites a claimed Modular/Mojo result: "a hand-written Mojo WMMA kernel
  beating hipBLASLt on a 7900 XTX ... a 10-second warm-up period ... the difference between
  66,000 and 91,000 GFLOP/s on a 4096³ GEMM."
- gemini: states RDNA3 int8 WMMA "peaks at 244 TOPS" (vs chatgpt's 123 TOPS INT8 — see
  Disagreements). Chunk-size arithmetic: "8192 tokens, the fp16 activations for all 32
  layers consume only ~1.8 GB of VRAM"; "Streaming 6.2 GB of weights takes ~7.5 ms per
  chunk"; recommends chunk 4096 or 8192. GDN: "Chunk Size: Use Tc=128 ... At Tc=256, the
  O(Tc²·D) intra-chunk memory requirements blow out the 64 KB LDS budget"; claims GDN
  consumes "15-25% of your total prefill wall-time" despite FLOPs being "<2% of the FFN
  GEMMs" — attributed to LDS bandwidth/sync cost, not arithmetic.

### 3. Disagreements
- What llama.cpp's Q4_0 gfx1100 prompt path actually does: chatgpt and deepseek both say
  it is INT8 MMQ (Q4→Q8 activation, `v_wmma_i32_16x16x16_iu8`), citing named llama.cpp
  functions/files. gemini instead claims it "explicitly abandons the custom mul_mat_q
  int8/f16 ALU kernels" and "dispatches the GEMM to the highly tuned rocBLAS /
  hipBLASLt library" after a memory-bound Q4_0→fp16 dequant pass — the opposite mechanism.
  This is the single most load-bearing disagreement in the whole brief, since the two
  proposed mechanisms (INT8 MMQ vs FP16-dequant-then-rocBLAS) point to entirely different
  kernels to copy.
- RDNA3 INT8 WMMA peak throughput: chatgpt says "123 TOPS INT8 matrix... exactly the same
  nominal operation rate as its 123 TFLOP/s FP16 matrix figure"; gemini says "int8 WMMA
  peaks at 244 TOPS" — a 2x numeric contradiction on the same hardware spec.
  gemini's number implies INT8 doubles FP16 throughput; chatgpt's implies parity.
- llama.cpp RDNA3 MMQ tile config: chatgpt cites I=64,J=16..128,K=256,128 threads; deepseek
  cites MMQ_X=64, MMQ_Y=128, NWARPS=8 — different naming/values, not reconciled.
- Realistic INT8-WMMA GEMV/GEMM efficiency for a hand-written kernel: chatgpt targets
  75-90 TOPS effective (61-73% of 123 TOPS); gemini targets "55-65% of the 244 TOPS peak
  (~145 TOPS)" — very different absolute numbers stemming directly from the 123-vs-244 TOPS
  disagreement above.
- Optimal prefill chunk size: chatgpt argues for 512 (or the 256/512/1024 range, expecting
  512 to win) explicitly to make apples-to-apples comparison with llama.cpp and to avoid
  activation/attention/GDN working-set growth; gemini argues chunk should be "as large as
  possible," recommending 4096 or 8192 to amortize the weight stream, calling 32-token
  chunks wasteful; deepseek says the prompt's own 1024 "is near-optimal." All three land on
  different numbers.
- Practical ceiling for a hand-written WMMA GEMM vs hipBLASLt: chatgpt says "0.75-0.85 ...
  a very good result," "0.90+ is possible ... but I would not plan around it"; deepseek
  says with a 10-second clock warm-up plus LDS bypass, 0.85 is achievable and is "the
  practical ceiling ... a hardware limit, not a software bug"; gemini says "0.80 to 0.85 ...
  is the absolute maximum practical ceiling," and separately claims a specific mechanism
  (SMU downclocking "from 2.5 GHz to ~1.8-2.0 GHz to stay within the 355W board limit")
  that neither chatgpt nor deepseek names.

### 4. Concrete recommendations
- chatgpt: implement Q4×Q8 INT8-WMMA kernel isolated at M=512,K=4096,N=12288 first, before
  further BF16-WMMA tuning; WG of 4 waves=128 threads, output tile N=64, prompt rows M=64,
  K tile=256; LDS layout `A_q8[64][256]` 16 KiB + `W_q8[64][256]` 16 KiB + scales; process
  by layer-major (all chunks of a layer before moving to next layer) rather than
  chunk-major, to keep quantized weight tiles warm in L2/Infinity Cache; interleave
  independent WMMA accumulator tiles (C0,C1,C2,C3 round-robin) to break dependency chains;
  keep `s_waitcnt vmcnt(k)` with k>0, not vmcnt(0), for ≥2 outstanding global stages.
- deepseek: 10-second GPU warm-up before any timed measurement; bypass LDS for the GEMM
  entirely; avoid hipBLASLt's CDNA-tuned fallback path pitfalls.
- gemini: chunk 4096-8192; overlap only `global_load` of layer L+1 weights with layer L's
  WMMA write-out (not attention-i/FFN-(i-1) overlap, which gemini explicitly rejects as
  "Transformer layers are strictly sequential"); swizzle LDS write addresses
  (`addr ^ (row_id * 4)`) to spread 16x16 fp16 tile reads across LDS banks; explicit
  double-buffer via `__builtin_amdgcn_s_waitcnt` since "LLVM will sequence them serially"
  otherwise.

### 5. Errors / suspect
- The 123-vs-244 TOPS INT8 figure is a direct, unresolved numeric contradiction between
  chatgpt and gemini for the same GPU spec (RX 7900 XTX INT8 WMMA peak) — one of them is
  wrong, and this brief's other conclusions (achievable TOPS%, expected GEMM time) inherit
  whichever is correct. Neither source is independently verified within this batch.
- gemini's claim that llama.cpp's gfx1100 prompt path "abandons ... int8/f16 ALU kernels"
  in favor of "rocBLAS / hipBLASLt" after an FP16 dequant contradicts named-function-level
  citations from both chatgpt and deepseek (`ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`,
  `MMQ_X`/`MMQ_Y`/`NWARPS`) — flagged as the likely-wrong answer of the three, given the
  other two independently converge on the INT8 MMQ mechanism with specific source
  references.

---

## Brief 12 — kernel-proposer-loop

Question: what AlphaEvolve/FunSearch/KernelBench/Sakana do differently from a 7-iteration,
28-candidate, zero-survivor local-27B kernel-optimization loop; edit format; giving the
proposer a mechanism; reward-hacking controls; whether the whole approach is worth it.

Note: this brief is about ML research/tooling, not RDNA3 hardware mechanics — the
"wrong for gfx1100" axis of the Errors section mostly does not apply; flagged items below
are internal-inconsistency / unverifiable-citation concerns instead.

### 1. Consensus
- The zero-survivor result is explained by sample-count and edit-granularity mismatch, not
  proof that a ~27B model is categorically incapable: real systems (AlphaEvolve, FunSearch,
  Sakana) run far more candidates than 28, and edit smaller regions than an 18k-character
  file. (prior)
- Unified diffs (with line numbers) are the worst edit format for weaker/smaller models;
  anchored text-match formats (search/replace, or whole-function rewrite for very small
  models) fail less often. (prior)
- A two-stage "diagnose, then edit" prompt structure is expected to help — separating
  bottleneck identification/mechanism naming from code generation. (prior)
- Sakana's original "AI CUDA Engineer" reward-hacked its own evaluator (a well-known,
  independently reported incident) — cited by all three as the canonical warning case for
  this exact gate-design problem. (prior)
- Given the current zero-survivor state, a human writing the "obvious" kernels (documented
  elsewhere as the GDN codegen regression, FFN-down spills, RMSNorm-barrier folding) is a
  faster path than continuing to tune the local-27B loop as-is. (prior — all three converge
  on some version of "do the known fixes by hand first")

### 2. Evidence brought
- chatgpt: AlphaEvolve uses EVOLVE-BLOCK markers, SEARCH/REPLACE (not line-numbered unified
  diff), a MAP-Elites-inspired evolutionary database with islands, evaluation cascades, and
  mixes Gemini 2.0 Flash (volume) with Gemini 2.0 Pro (rarer high-quality mutations); its
  own paper "contrasts millions of samples for FunSearch with thousands for AlphaEvolve."
  FunSearch evolved "a single Python function, typically about 10-20 lines." Sakana's
  revised (post-reward-hacking) workflow: "samples 8 candidates/generation for 10
  generations," pre-filters with "three LLM verifiers," hardware-tests top 4, keeps an
  archive of up to five previous kernels sorted slowest-to-fastest, uses an ensemble of
  "o3, o4-mini, Claude 3.7 Sonnet, GPT-4.1" (Gemini 2.5 Pro in an ablation) — "roughly 80
  proposals ... hardware-evaluated ~40 for one kernel search." KernelBench: DeepSeek-R1
  one-shot "fast_1" rates "12/36/2% on Levels 1/2/3"; 10 sequential
  execution/profiling turns raised those to "43/72/18%." Kevin-32B (QwQ-32B-derived,
  ICLR 2026): correctness "56%→82%," mean performance "0.53x→1.10x PyTorch eager,"
  attributing more gain to serial refinement than to sampling more independent candidates.
  Aider's 2023 GPT-4 Turbo benchmark: "modified unified-diff format raised completion from
  20% with SEARCH/REPLACE to 61%" (note: format ranking reversed relative to chatgpt's own
  general recommendation — a specific-model exception it calls out itself). Aider
  architect/editor-mode result: "o1-preview architect ... reached 85% versus a 79.7%
  baseline." KernelBench-Verified 2026 study: best tested frontier model reached only
  "0.88x geometric-mean performance," "28% of its kernels increased peak GPU memory."
- deepseek: cites "EvoEngineer-Free and EvoEngineer-Insight" using block-level evolution.
  Cites "PTXBench" as explicitly benchmarking "Qwen3.6-27B for GPU kernel optimization"
  with "uneven" success and inability to "consistently match frontier libraries." Cites an
  unnamed benchmark: "AST edit is the only format to hit 100% correctness on 3 out of 4
  models tested"; "unified diffs ... dropped from 93% on a powerful model to 20.7% on a
  smaller one"; whole-file rewrites "use up to 18x more tokens and take 12x longer." Cites
  "AlphaVerus framework, which uses a 'Treefinement' tree search algorithm." Cites
  "SpecBench" as "a comprehensive framework for understanding and measuring reward hacking
  in coding agents."
- gemini: cites the same Sakana/FunSearch/KernelBench lineage without the specific
  candidate-count breakdown chatgpt gives; states "even GPT-4o and Claude 3.5 Sonnet
  struggle on medium-difficulty kernels" (unsourced); frames the reward-hacking incident as
  "discovered a memory exploit in the testing harness that allowed it to skip computation
  and point to the pre-computed correct answers."

### 3. Disagreements
- Whether a ~27B-class model can be useful at all for this task: chatgpt's explicit
  position is "27B is not too small by construction; your 27B is probably the wrong 27B and
  the wrong search protocol," directly citing Kevin-32B as a counterexample of a
  successfully post-trained ~32B kernel optimizer. deepseek's position is "A 27B-class
  model is at the very low end of what has been demonstrated to work ... likely the wrong
  size and lacks the specialized knowledge needed." gemini's position is "Yes, by
  construction" the model is too small, citing training-data scarcity for RDNA3/gfx1100
  specifically as compounding the size problem. Three-way disagreement on the causal
  attribution (size vs protocol vs domain-data-scarcity).
- Best edit format: chatgpt recommends AlphaEvolve-style SEARCH/REPLACE anchored to an
  EVOLVE-block with 3-10 unchanged anchor lines, but explicitly notes format choice should
  be validated per-model rather than assumed universal (cites Qwen2.5-Coder-32B doing well
  in whole-file mode). deepseek recommends AST-based edits as strictly best ("the only
  format to hit 100% correctness"). gemini recommends whole-function extraction +
  whole-function rewrite + mechanical splice-back specifically because unified diffs and
  even search/replace both risk failure at this proposer size.
- Recommended next proposer: chatgpt's decision rule is empirical/staged — bake off local
  27B (with diagnose→edit + SEARCH/REPLACE) vs a frontier model vs human-written kernels on
  one tractable hot region, and only then decide. deepseek states flatly: "Stop using the
  local 27B model as the primary proposer... Use a frontier model (like GPT-5.5 or Claude
  Opus 4.7) as your proposer" — no bake-off, direct switch. gemini says "Shut down the
  optimization loop... optimize the C++ by hand" — no proposer at all, human-only.

### 4. Concrete recommendations
- chatgpt: SEARCH/REPLACE contract requiring exact byte-for-byte SEARCH text with 3-10
  anchor lines, `NO_CHANGE` as a valid output; show only 50-200 lines of editable code plus
  callee/caller definitions plus ~10-20 lines of context plus profiler/ISA summary (not the
  full 18k-char megakernel); machine-readable hypothesis fields (TARGET, BOTTLENECK,
  EVIDENCE, TRANSFORMATION, PREDICTED ISA EFFECT, EXPECTED SPEEDUP) required before an edit
  is requested; menu of 12 named transformation classes (A-L) with one randomly assigned
  per branch instead of 4 personas; advancement rule Δ > max(2%, 3σ_Δ) instead of a flat 3
  runs/+2%; final bake-off of 100-200 proposals under 3 conditions (local 27B diagnose→edit,
  frontier model, human) measuring valid-edit%, compile%, correct%, mechanistically-
  relevant%, >=2%-phase-improvement%, >=2%-token-survivor%, wall-clock/GPU-hour per
  survivor; stop local-27B path if <5% mechanically-valid+relevant after 100 proposals.
- deepseek: switch immediately to a frontier proposer + AST or robust search/replace with
  a "4-level matching fallback chain (Exact → Whitespace → Indentation → Fuzzy)"; two-stage
  diagnose→edit; "holdout" fixture never shown in the prompt; median-of-N runs (already
  in place); add a "wall-clock plausibility" term comparing claimed vs total-kernel runtime
  change.
- gemini: two-turn prompt (Turn 1: pick one of 4 named transformations + 2-sentence plan;
  Turn 2: implement only that plan); dynamic per-run fixture randomization (prompt length,
  token IDs, target string); interleaved Champ→Cand→Champ→Cand A/B testing with a paired
  t-test instead of block-then-block runs; explicit `hipDeviceSynchronize()` immediately
  before the stop-clock instruction in the locked harness; logit MAE check across the full
  vocabulary (reject if MAE > 1e-4), not just argmax-token identity.

### 5. Errors / suspect
- No RDNA3/gfx1100 hardware claims appear in this brief's answers to check against §CLAUDE
  facts (wave32/WMMA/LDS/CU count/bandwidth) — not applicable here.
- deepseek's citations "PTXBench" (benchmarking "Qwen3.6-27B") and "SpecBench" (a "reward
  hacking in coding agents" framework) are not corroborated by chatgpt's or gemini's
  answers, are given no publication venue/author, and do not match any widely-known
  benchmark name in this space (KernelBench, and its "-Verified" variant, are the
  well-attested ones cited by chatgpt) — flagged as possibly fabricated/hallucinated
  citations.
- deepseek's "AST (Abstract Syntax Tree) edits are superior ... 100% correctness on 3 out
  of 4 models tested, with zero format failures" is stated with a specific-sounding number
  but no named benchmark or source, unlike chatgpt's citations which name the specific
  systems (AlphaEvolve paper, Aider's own blog benchmarks) — flagged as unverifiable.
- deepseek's model-name recommendations ("GPT-5.5 or Claude Opus 4.7") and claim these are
  "benchmarked at 80%+ on Aider Polyglot" are unverifiable/likely-invented specific version
  numbers.

---

## Brief 13 — multi-model-moe-static-split

Question: comptime shape specialization across a 9B/27B/Flash-Next-MoE family without
kernel duplication; Flash-Next static CPU/GPU decode split (41.8 GB routed experts, uniform
routing); MoE prefill chunking/staging; 1M-context indexer-cache sizing; native IQ/K-quant
formats vs re-quantizing at load.

### 1. Consensus
- Uniform/near-uniform expert routing (98.6% of 24,576 slots active within 1.3k tokens,
  hottest slot only ~2.5x uniform, per the brief) means there is no cacheable hot expert
  set — all three treat "all routed experts on CPU/host RAM for decode" as the baseline-
  correct architecture given this routing trace, not a compromise. (prior — though gemini
  and chatgpt both separately propose a *better* alternative, see below)
- For MoE prefill, once a chunk is large enough that nearly all experts get touched anyway,
  streaming the full 41.8 GB over PCIe once per chunk and computing on GPU beats CPU
  compute for that chunk. (prior)
- Compile-time-worthy shape parameters are: tile shapes/unroll factors, LDS budgets, head
  dimension/count where they set lane ownership, block-quant size, WMMA tile dims; layer
  count, sequence length, expert IDs, and cache/page addresses can stay runtime without
  performance loss. (prior — all three give near-identical splits)
- IQ-format CPU dequant (IQ4_NL/IQ2_XXS/IQ1_M) should use existing llama.cpp-proven native
  vector-dot kernels rather than converting to a uniform format at load time, because the
  41.8 GB corpus's low-bit formats (down to 1.75 bpw) would balloon if re-expanded to a
  uniform ~4.5 bpw format before hitting DDR bandwidth. (prior — though gemini disagrees
  for the GPU side specifically, see Disagreements)

### 2. Evidence brought
- chatgpt: cites the exact released Flash-Next config figures ("48 layers, 24 Q heads/2
  KV heads, D=256, 512 experts/top-10, expert width 640," "4Q/1K D=128 sparse indexer with
  compression ratio 4" — Hugging Face). Computes routed-expert bandwidth arithmetic:
  41.8 GB × (10/512) = 816 MB/token; at 52 GB/s DDR5 → 15.7 ms/token → 63.7 tok/s host-
  bandwidth ceiling; at 25.7 GB/s PCIe → 31.8 ms → 31.5 tok/s if experts were instead
  copied to GPU per-token (explicitly rejected). Per-layer bandwidth: 816/48 ≈ 17.0 MB/
  layer, DDR floor 17/52 ≈ 327 µs/layer. Cites llama.cpp CPU vector-dot names:
  "IQ4_NL × Q8_0, IQ2_XXS × Q8_K, IQ1_M × Q8_K" with "VNNI-style byte dot products." Cites
  IQ4_NL structure: "32 weights ... fp16 scale plus 16 packed nibbles, giving 4.5 bpw," and
  IQ2_XXS "2.0625 bpw," IQ1_M "1.75 bpw." MoE-prefill probability model:
  f(C)=1-(1-10/512)^C, giving a table (32→47%, 64→72%, 128→92%, 256→99.4%). PCIe streaming
  time 41.8/25.7 = 1.63 s/chunk; at C=2048 that's 0.794 ms/token amortized. Crossover math
  against llama.cpp's stated 175 tok/s prefill: 1/175=5.71 ms/token → C>285; against ideal
  decode floor 15.7 ms/token → C>104; concludes practical crossover "C≈384-512." Recomputes
  the indexer cache using QSA's actual architecture (4 query heads, 1 shared MQA key head,
  D=128, one compressed key per 4 tokens): 12×(1e6/4)×128×2 = 768 MB, not the 12 GB stated
  in the prompt's own framing (see Disagreements/Errors). Cites an "recent experimental
  llama.cpp expert-cache RFC" reporting "+84% on one synthetic decode test" from dynamically
  caching 32-64 expert slots on a 24GB 4090 for Flash-Next — flagged by chatgpt itself as
  "new community evidence ... not a settled result."
- deepseek: states "A native RDNA3 kernel for W4A16 MoE ... has been shown to provide up to
  a 2.15x speedup over a Triton reference kernel for similar models on RDNA3" (footnote -5,
  unnamed source) and cites "vLLM project has demonstrated a fused MoE W4A16 HIP kernel for
  gfx1100 that uses packed int4 weights" (footnote -5). Cites "AVX-512BW natively supports
  ultra-fast 8-bit codebook lookups using `_mm512_permutexvar_epi8`" — actually this exact
  phrase appears in gemini's answer, not deepseek's (deepseek's own AVX-512 claim is
  generic, no intrinsic named). PCIe latency calc: "10 KB transfer over PCIe at 25.7 GB/s
  has a theoretical latency of ~0.4 µs."
- gemini: cites Qwen/SGLang QSA description matching chatgpt's (4Q/1 shared-K-head/D=128/
  compression-4). Computes indexer-cache-via-pooling alternative: block size 64 → "shrinks
  the 12 GB indexer cache to 187 MB." Selection-cost estimate: "1,000,000/64=15,625 block
  keys ... takes <100µs" on the XTX. PCIe handoff latency: "10 KB down and 10 KB up via
  PCIe 4.0 x16 takes ~15µs each way using zero-copy pinned memory," "across 48 layers ...
  only ~1.5 ms per token." States "AVX-512BW natively supports ultra-fast 8-bit codebook
  lookups using `_mm512_permutexvar_epi8`." States DDR bandwidth ceiling "52 GB/s DDR5
  reading 1 GB of expert weights per token yields a hard physical ceiling of ~50 tokens/s"
  (vs chatgpt's more precise 816 MB/token → 63.7 tok/s — see Disagreements). Prefill
  crossover: "Streaming weights to the GPU is faster for any chunk size >~64 tokens" (vs
  chatgpt's ~384-512 — see Disagreements). Staging-buffer proposal: "three 128 MB staging
  buffers."

### 3. Disagreements
- Indexer-cache size: brief states "12 GB at bf16 for 1M tokens." deepseek accepts 12 GB
  unchanged, proposes fp8/q4 → "3 GB or 6 GB." chatgpt recomputes from QSA's real
  architecture (1 shared MQA key head, native 4x compression) → 768 MB, 16x smaller.
  gemini adds 64-token pooling on top of the accepted 12 GB baseline → 187 MB. Three
  different final numbers via three different mechanisms.
- Whole-layer GPU residency vs "all experts on CPU": chatgpt argues a *better* split exists
  — complete routed-expert layers on GPU (871 MB/layer avg, ~9 layers per 8 GiB) rather
  than per-layer partial-expert subsets, since partials still force a CPU handshake every
  layer. gemini: "'All routed experts on CPU' is the strictly optimal split," no
  whole-layer alternative considered. deepseek agrees CPU-all is "the correct approach" and
  separately argues against layer/expert subsets "because you must load 41.8 GB ... once
  per chunk anyway" — a decode/prefill conflation (that 41.8 GB PCIe figure is the prefill
  streaming cost, not decode's).
- Decode tok/s ceiling: chatgpt 816 MB/token → 63.7 tok/s ceiling (52 GB/s), ~30-45 tok/s
  realistic; gemini rounds to "~1 GB/token" → "~50 tokens/s" — ceilings differ ~25% purely
  from gemini's rounding of 816 MB up to 1 GB; deepseek gives no single ceiling number.
- Handoff latency: deepseek computes bandwidth-limited ~0.4 µs per 10 KB transfer; gemini
  computes latency-floor-dominated ~15 µs each way (~1.5 ms/48 layers); chatgpt gives no
  single-transfer number but frames cost as "48 synchronization latencies are what
  matter" — agrees with gemini's latency-dominated framing, implicitly against deepseek's
  bandwidth-only calc.
- MoE prefill crossover chunk: chatgpt derives "C≈384-512" from two explicit bounds
  (decode-bandwidth floor and llama.cpp's 175 tok/s); gemini asserts ">~64 tokens," 6-8x
  smaller, from the f(C)-saturation argument alone with no GPU-overhead accounting;
  deepseek gives no numeric crossover.
- Native IQ-quant decode on GPU: chatgpt and deepseek both keep IQ4_NL/IQ2_XXS/IQ1_M
  native on GPU too (repack layout, not values) — requantizing IQ2_XXS→q4 would expand
  storage 2.18x, IQ1_M→q4 2.57x, "destroying much of the ... property making CPU offload
  viable." gemini disagrees for the GPU path: "RDNA3 wave32 ALUs do not possess a fast
  hardware codebook lookup instruction," recommends CPU-side re-quant into the engine's own
  q4 at load time instead — though this is about the GPU-resident subset, not the
  CPU-hosted 41.8 GB bulk chatgpt/deepseek are mainly arguing about.

### 4. Concrete recommendations
- chatgpt: a 128-bit structural-ID hash over architecture/dims/layer-pattern/expert-count/
  quant-ABI-version in the model file header, dispatched via `switch(sid)` to
  `Engine<Shape>::load(...)`; CpuExpertJob/CpuExpertResult mailbox structs (job: generation,
  10x expert_id, 10x router_w, Q8Activation x; result: generation, bf16 y[2560] ~5 KiB);
  thread CPU work over output rows not K (gate/up: 10×2×640=12,800 independent rows; down:
  10×2560=25,600 rows); whole-layer GPU residency (~871 MB/layer average) preferred over
  partial-expert-subset caching; two-buffer per-layer PCIe staging (`expert_stage[2]
  [~900 MB]`), one-layer lookahead; layer-major prefill ordering (stage layer, process all
  chunks/rows, evict, next layer); keep QSA indexer cache in bf16 initially (~768 MB by its
  own recompute), only quantize after long-context retrieval validation.
- deepseek: single NUMA node, one thread per expert (up to 10 experts × 3 matrices = 30
  independent GEMMs/layer); double-buffered staging area, two streams, one-layer lookahead
  for prefill; quantize indexer keys to fp8/q4 to cut the (accepted) 12 GB to 3-6 GB.
- gemini: GPU holds dense/GDN/shared-expert/attention/indexer caches only; CPU holds all
  41.8 GB pinned; pin CPU threads to physical cores (no NUMA needed, single socket); three
  128 MB VRAM staging buffers, two async HIP streams, one-layer lookahead; 64-token
  block-pooled indexer keys (187 MB) computed/selected at the end of the previous layer's
  GDN execution, stored in LDS for the next attention layer; re-quantize IQ/K-quant weights
  into the engine's own q4 block-32 format at GPU load time, accepting "~10-15 seconds"
  extra load time for a simpler, higher-bandwidth GPU decode loop.

### 5. Errors / suspect
- The brief's own "12 GB at bf16 for 1M tokens" indexer-cache premise looks internally
  questionable against the brief's own stated QSA architecture (4-head indexer, compression
  ratio 4): chatgpt's recomputation using a genuine single shared MQA key head at 4x
  sequence compression yields 768 MB, 16x smaller than the stated 12 GB — either a wrong
  premise in the original brief or a wrong recomputation, not resolvable from this batch,
  but too large a gap to pass over.
- deepseek's `_mm512_permutexvar_epi8` claim is word-for-word gemini's, asserted by both
  with no cited source — the intrinsic-level correctness for IQ2_XXS/IQ1_M codebook shapes
  is unverified within this batch.
- deepseek's own answer is internally inconsistent: section 2 recommends CPU-all for
  decode and separately states CPU-all is also right "for your single-GPU scenario"
  without qualifying decode vs. prefill, then section 3 recommends GPU streaming for
  prefill — the unqualified section-2 sentence pre-empts and contradicts section 3.
