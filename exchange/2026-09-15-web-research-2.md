# Web research 2: open questions behind the next mojo-baro rounds

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-web-research-2.md`. All
retrieval via firecrawl (`firecrawl_search` / `firecrawl_scrape`,
`onlyMainContent`). Retrieval date for every source below: 2026-09-15 unless
a source's own publish date differs (noted inline). All nine questions were
worked; none were skipped for budget, though Q7 and part of Q9 have gaps
flagged explicitly below rather than filled with a guess.

## Q1 (highest): how llama.cpp spends a MoE decode token on RDNA3

Sources: `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/CMakeLists.txt`,
`ggml/src/ggml-hip/CMakeLists.txt`, `ggml/src/ggml-cuda/common.cuh`
(https://github.com/ggml-org/llama.cpp, `master`, retrieved 2026-09-15); Aman
Bhargava (am17an, llama.cpp CUDA contributor), "Optimizing Token Generation
in llama.cpp's CUDA Backend", https://am17an.bearblog.dev/new-post/,
published 2025-12-01, retrieved 2026-09-15 (mirrors
https://github.com/ggml-org/llama.cpp/discussions/17621).

**HIP graphs are compiled in by default and not disabled for AMD by the
existing arch gate.** `ggml/CMakeLists.txt`: `option(GGML_HIP_GRAPHS "ggml:
use HIP graph" ON)`, requiring `hip_VERSION >= 6.1` (`ggml-hip/CMakeLists.txt`
aborts the build below that). Runtime, `ggml_cuda_graph_set_enabled()`
(`ggml-cuda.cu:4404`) disables graphs when `ggml_cuda_info().devices[dev].cc <
GGML_CUDA_CC_VOLTA` (700). AMD device codes are built as
`GGML_CUDA_CC_OFFSET_AMD (0x1000000) + arch_code` (`common.cuh:63`), always
far above 700, so this particular gate never fires for gfx1100. HIP graphs
are a distinct build flag from `GGML_CUDA_GRAPHS` (NVIDIA path, default OFF
unless the top-level build sets it), so "CUDA graphs on AMD" really means
"HIP graphs", which is a separate, default-on option.

**The MoE op (`MUL_MAT_ID`) is graph-compatible exactly when the routed
batch is small, which decode is.** `ggml_cuda_graph_check_compability()`
(`ggml-cuda.cu:2558`) drops out of graph capture only if
`ggml_cuda_mul_mat_id_needs_sync()` returns true for that node (comment:
"the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA
graphs", ref PR #18958). That function (`ggml-cuda.cu:1881`) returns false
(no sync needed, stays inside the graph) when `dst->ne[2] <=
MMVQ_MAX_BATCH_SIZE` and the quantized-weight batch limit from
`get_mmvq_mmid_max_batch()` is not exceeded, in which case
`ggml_cuda_mul_mat_id()` (line 1910) dispatches straight to
`ggml_cuda_mul_mat_vec_q()`, the single gathered `mul_mat_vec_q_moe` kernel
(confirmed against the kernel body in web research 1's Q1.4: one launch over
`ids[]`, no per-expert relaunch). At batch 1 decode with a small
`n_expert_used` (top-8 for qwen35moe fits comfortably under typical
`MMVQ_MAX_BATCH_SIZE`), this is the path taken, so the MoE step itself does
not force a stream sync or break graph capture.

**Gate and up expert GEMVs are fused into one kernel.** Per am17an's post:
"we identify operations where `MUL_MAT` follows an `ADD` or more
interestingly, when there is a gated activation i.e. the operation
`o = sigma(Wgate*X) . Wup*X`, so we can re-use the X activation and multiply
by both matrices." This is PR #16715 ("GEMV fusion"), enabled by default.
It answers the brief's fused-gate-up question directly: yes, one kernel
reuses the activation vector against both projection matrices instead of two
separate GEMV launches.

**Router (softmax + top-k) runs fused on GPU, not CPU.** Same source,
"TopK-MoE" section: "This is the common scoring algorithm in MoE models to
select expert weights per token. This usually involves a soft-max over the
logits per expert, then a top-k selection for the experts for the sparse MoE
matrix multiplication (called MUL_MAT_ID in ggml)", fused via PR #16130,
enabled by default. No CPU round trip for routing.

**Per-token launch count: no single number found.** Neither the PRs linked
from the blog post nor the discussion state an exact kernel count per token
for a MoE model; am17an's post only says "none of these PRs increase the TG
by more than 10%, taking them all altogether we get a nice speedup", with
his own llama-bench numbers on RTX 4090/5090 (qwen3moe 30B.A3B Q4_K_M:
198.39 to 271.04 tok/s stock CUDA graphs vs baseline with fusion+
`GGML_CUDA_GRAPH_OPT=1` disabled; gpt-oss 20B MXFP4: 232.05 to 271.99 tok/s).
Those numbers are on NVIDIA (4090/5090), not gfx1100, and are included only
to show the fusion stack's magnitude, not as an AMD data point. Flagged as
unanswered: exact per-token kernel count on gfx1100 for qwen35moe.

Shared expert scheduling: not found in this pass (no source distinguishing
shared-expert handling from routed-expert handling in the CUDA/HIP backend
was located). Flagged as unanswered.

## Q2: int8 activations and V_DOT4 on RDNA3, cost and accuracy

Sources: `ggml/src/ggml-cuda/common.cuh` (retrieved 2026-09-15);
https://github.com/ggml-org/llama.cpp/discussions/14471 (retrieved
2026-09-15, unanswered discussion, cited only for confirming the CUDA
backend's implicit q8_1 conversion is a known, previously-asked-about
behavior, not for any numeric claim).

**Correction to web research 1's more generic claim: on RDNA3 specifically,
`ggml_cuda_dp4a` is NOT the same intrinsic as on RDNA2/CDNA.**
`common.cuh:720`:

```c
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    ... // inline v_mul_i32_i24 / v_add3_u32 asm fallback
```

CDNA/RDNA2/gfx906 use `__builtin_amdgcn_sdot4` (plain signed int8 dot4,
`V_DOT4_I32_I8`). gfx1100 (RDNA3) instead uses `__builtin_amdgcn_sudot4`
with both sign flags set true, which is the ISA's mixed signed/unsigned
dot4 form (`V_DOT4_I32_IU8`, capable of treating each operand's sign
independently, here used with both signed since q8_1 activations and the
k-quant int8 values are both signed). `V_DOT4_I32_IU8` throughput per CU:
per web research 1's Q3 citation (zolotukhin.ai AMD RDNA3/RDNA4 reference,
retrieved 2026-09-15), it sits in the same 1/clk-on-wave32 throughput class
as `V_FMA_F32`.

**No f16/bf16-activation path exists for k-quants specifically.**
`ggml_cuda_mul_mat_vec_f` (seen in `ggml_cuda_mul_mat_id`, web research 1's
Q1.4 excerpt) is the float-activation GEMV path, but it is selected when
`src0` (the weight tensor) is itself a non-quantized float type
(`GGML_CUDA_CC_IS_AMD(cc)` branch at `ggml-cuda.cu:1932`), not as an
alternative for quantized k-quant weights. Every k-quant weight type
(`Q4_K`, `Q5_K`, `Q6_K`, etc.) is routed through `mul_mat_vec_q`, which
always consumes a q8_1-quantized activation (web research 1, Q2). There is
no config knob to keep the activation in bf16/fp16 while dequantizing a
k-quant weight in the CUDA/HIP backend.

**Accuracy impact of q8_1 activation quantization in isolation: not found.**
Searches for a perplexity delta attributable specifically to the activation
quantization step (as opposed to the weight quantization itself) did not
surface a measurement; the one relevant GitHub discussion
(ggml-org/llama.cpp#14471, "Implicit Q8_1 quantization for matrix
multiplications? The CUDA backend implicitly converts src1 to q8_1 if src0
is quantized. Why?") went unanswered. Flagged as unanswered: no isolated
q8_1-activation perplexity number was found in this pass.

## Q3: speculative decoding under sampling, reference algorithms

Sources: vLLM docs, https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/sample/
rejection_sampler.html (retrieved 2026-09-15, docstring content, sourced
from `vllm/v1/sample/rejection_sampler.py`); `common/sampling.cpp`
(https://github.com/ggml-org/llama.cpp, `master`, retrieved 2026-09-15,
already fetched in full for web research 1's Q5); Rost Glukhov, "Speculative
Decoding: 20-50% Faster LLM Inference",
https://www.glukhov.org/llm-performance/optimization/speculative-decoding/,
published 2026-07-01, retrieved 2026-09-15 (secondary blog source, cited
only where it corroborates the primary vLLM docstring's formula, marked as
such below).

**Original rule and its formula, corroborated by two independent sources.**
vLLM's own docstring states its `RejectionSampler` "strictly follows the
algorithm described in https://arxiv.org/abs/2211.17192" (Leviathan et al.,
"Fast Inference from Transformers via Speculative Decoding"), and defines
three token classes: "accepted tokens: tokens that are accepted based on
the relationship between the raw draft and target probabilities. recovered
tokens: tokens that are sampled based on the adjusted probability
distribution, which is derived from both the draft and target
probabilities. bonus tokens: If all proposed tokens are accepted, the bonus
token is added to the end of the sequence. The bonus token is only sampled
from the target probabilities." The glukhov.org piece (secondary, not code)
states the same formula in closed form: acceptance probability
`min(1, p(x) / p_draft(x))`, and on rejection, resample from the residual
`r(x) = max(0, p(x) - p_draft(x)) / sum_y max(0, p(y) - p_draft(y))`. Both
sources agree on the shape of the rule; the vLLM docstring is the primary
citation, the closed-form quote is secondary corroboration.

**vLLM does not apply top-p/top-k inside the accept/recover stage, only to
the bonus token.** Same vLLM docstring: "we can use top_p, top_k sampling
for bonus tokens, while spec decode does not support these sampling
strategies." So under sampling (temperature and nucleus/top-k) in vLLM,
only the raw-probability accept/recover math runs on the drafted positions;
top-p/top-k truncation is confined to the one bonus token sampled after a
fully-accepted draft run. This is a real, currently-documented limitation,
not a rare edge case.

**llama.cpp does not implement the same two-distribution rejection-sampling
math. It samples fresh from the full target chain and compares for
equality.** From `common/sampling.cpp`'s `common_sampler_sample_and_accept_n`
(already fetched in full for web research 1):

```c
for (; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);
    if (draft[i] != id) {
        break;
    }
}
```

`common_sampler_sample` runs the position's full sampler chain (temperature,
top-p, top-k, whatever is configured) independently of the draft, and only
uses `draft[i]` to decide whether to keep extending the accepted run; the
token actually appended (`id`) always comes from that independent sample of
the target chain, never from the draft token itself. This is the well-known
simplification sometimes called "sample-and-compare": each individual
emitted token is a legitimate draw from the target's per-position
distribution (so it is not silently biased token-by-token the way a
draft-copies-through scheme would be), but it does not use the Leviathan/
Chen ratio `min(1, p/q)` or the residual-distribution resampling, so its
expected acceptance rate for a given (draft, target) pair is lower than
proper rejection sampling would achieve for the same pair (proper rejection
sampling exploits the correlation between draft and target more
efficiently). This is inference from reading the source directly, not a
quoted claim from an llama.cpp doc; no llama.cpp document was found that
states this tradeoff explicitly, so treat the efficiency comparison as
reasoned analysis, and the mechanism description (what the code does) as
a direct source read.

**The documented pitfall the brief asked about, restated precisely**: naive
"accept the draft token if it equals an independently-resampled target
token" is a distinct algorithm from "accept the draft token itself with
probability `min(1, p/q)`". llama.cpp's code performs the former. Both are
unbiased at the single-token level under llama.cpp's implementation (because
the emitted token is always freshly sampled from the target chain, whichever
branch fires), but the true Leviathan/Chen rule additionally raises the
acceptance rate by directly using `p/q`, which llama.cpp's implementation
does not do, so it should be expected to have a lower expected accepted-run
length than vLLM's rejection sampler for the same draft/target pair and
acceptance-rate baseline.

## Q4: raising MTP / draft acceptance at batch 1

Sources: Rost Glukhov, "Speculative Decoding: 20-50% Faster LLM Inference"
(cited above, published 2026-07-01, retrieved 2026-09-15, secondary blog
source, all numbers below attributed to it explicitly); vLLM rejection
sampler docstring (cited above) for the underlying algorithm the formula
below derives from.

**EAGLE-3 acceptance figures (glukhov.org, unverified against a primary
EAGLE-3 paper or benchmark in this pass):** "EAGLE-3 typically achieves
60-80% acceptance rates on in-distribution workloads, compared to 40-60%
for standalone draft models. On code generation workloads with high
repetition, acceptance can exceed 85%." Acceptance-by-method table from the
same source: draft model (same family) 40-60%, EAGLE-3 60-80%, P-EAGLE
65-85%, n-gram 10-90%+ (workload dependent), MTP 50-70% ("Qwen 3.6 models
specifically"), self-speculative 30-50%.

**Prompt-lookup (n-gram) decoding, llama.cpp specifics.**
`examples/lookup/README.md` (https://github.com/ggml-org/llama.cpp, `master`,
retrieved 2026-09-15): implements
https://github.com/apoorvumang/prompt-lookup-decoding via PR #4484; key
params `ngram_min`, `ngram_max` (size of n-grams searched for in the prompt)
and `n_draft` (how many subsequent tokens to draft on a match). Per
glukhov.org's numbers (secondary, unverified): acceptance "10-90%+
(workload-dependent, high on repetitive, near zero on novel)", best for
code editing and template filling, useless for novel generation.

**The expected-accepted-tokens formula, and why 42% acceptance at k=2
should already be enough.** glukhov.org states a simplified linear form,
`E[accepted] = alpha * K`, but the exact formula implied by the Leviathan/
Chen algorithm (the same arxiv 2211.17192 vLLM cites) for `K` drafted
tokens with i.i.d. per-token acceptance probability `alpha` is the geometric
sum

```
E[tokens per verify pass] = (1 - alpha^(K+1)) / (1 - alpha)
```

(this is standard algebra on a geometric series, not itself a web claim; it
reduces to `1 + alpha` at K=1 and to `alpha*K` only in the small-alpha
limit, so glukhov.org's linear form is an approximation, not this exact
result). Plugging in mojo-baro's own stated numbers, `alpha = 0.42`, `K = 2`:

```
E[tokens] = (1 - 0.42^3) / (1 - 0.42) = (1 - 0.074088) / 0.58 = 0.925912 / 0.58 ~= 1.597
```

That is already a ~1.6x tokens-per-verify-pass multiple over plain
autoregressive decoding (1 token per step), comfortably past the "1.3x+"
target stated in the brief, assuming draft generation cost is negligible
next to the verify pass (true for an MTP head riding along in the same
forward pass, per the brief's own framing). Solving the same formula for
the acceptance rate that exactly hits 1.3x at K=2 (using `(1-a^3)/(1-a) =
1+a+a^2`): `a^2 + a - 0.3 = 0`, `a = (-1 + sqrt(1 + 1.2)) / 2 =
(-1 + sqrt(2.2)) / 2 ~= 0.242`, i.e. only about 24% acceptance is needed at
k=2 for the idealized formula to clear 1.3x. Since mojo-baro is already at
42%, well above that 24% floor, the idealized math says 1.3x+ should already
be realized; if it is not showing up in wall-clock, the gap is most likely
overhead (draft cost, verify-kernel launch cost, or how "42%" is measured)
rather than the raw acceptance probability itself. This computation is
mine, built on the cited formula and the brief's own stated 42%/k=2 numbers,
not a quoted external claim.

## Q5: KV cache quantization and paging on consumer GPUs

Sources: `ggml-common.h` block layout confirmed via web research 1
(`QK_K`-style block-scale pattern; the KV-specific `q8_0`/`q4_0` KV block
size is 32 elements per block per llama.cpp's own quant-type documentation,
consistent with the general q8_0/q4_0 weight block size already established
in web research 1's Q1); Abhishek Patel, "KV Cache Quantization: Q8 vs FP16
(and Q4 Pitfalls)", https://www.techplained.com/kv-cache-quantization,
published 2026-03-25, last content update noted on-page 2026-09-15, retrieved
2026-09-15 (secondary blog source with its own measured numbers on
Qwen 3.5 9B/32B, not an llama.cpp or vLLM primary source; flagged as such).
KIVI or a similarly-named per-page int8/int4 KV scheme with its own paper or
repo was not located in this pass; flagged as unanswered.

**Layout**: llama.cpp's `-ctk` / `-ctv` KV cache types reuse the same
weight-quant block formats (`q8_0`, `q4_0`, `q5_0`, etc.), each with a
per-block scale (and, for the `_1` variants, an additional min/offset),
block size 32 elements, exactly the `block_q8_0`/`block_q4_0` structs
established in web research 1's Q1 (`ggml-common.h`). llama.cpp's
supported cache types per the techplained.com writeup: `f32, f16, bf16,
q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl, q5_k, q6_k, iq3_xs, iq2_xxs`, of which
it says only `q8_0` and `q4_0` are "the most-tested paths; the others are
research-grade" (its own characterization, not verified against an
llama.cpp doc in this pass).

**Reported quality and speed impact (techplained.com's own measurements on
Qwen 3.5 9B/32B, WikiText-103 + HumanEval, secondary source, not an
llama.cpp benchmark):** Q8 K + Q8 V: +0.05% perplexity vs FP16, "0 to -0.2"
HumanEval delta, called "the closest thing to a free lunch in local
inference." Q4 K + Q8 V: +0.4% perplexity, -0.5 HumanEval. Q8 K + Q4 V:
+1.4% perplexity, -1.5 HumanEval (asymmetric the other way, called out as
"skip"). Q4 K + Q4 V: +2.1% perplexity, -2.8 HumanEval. The asymmetry
(quantize K more conservatively than V, or if forced to choose, quantize
K harder than V per its "Q4 K + Q8 V is the magic combo" framing) is
explained there as: keys feed a softmax, so quantization error there can
change which tokens get attended to at all; values are only
averaged post-softmax, so quantization error there is a milder blur.
Speed: the source says quantized KV needs `--flash-attn` paired with it or
"decoding crawls"; with flash attention it claims "speed parity with FP16
KV... within 5%" on RDNA/consumer setups is not itself RDNA3-specific
in the source (its numbers are not tied to a specific GPU architecture);
no RDNA3-specific speed number for KV quantization was found in this pass.

**vLLM PagedAttention page size and page-table mechanism (primary source):**
https://docs.vllm.ai/en/v0.9.0/design/kernel/paged_attention.html (vLLM
official docs, retrieved 2026-09-15). Default block (page) size is 16
tokens (confirmed independently by
https://prakashkagitha.github.io/llm-stack-book/07-inference-serving/
03-vllm-internals.html, retrieved 2026-09-15: "Block size 16 is the
long-standing default for most models; FP8 KV caches and certain attention
backends prefer other values"). The kernel receives `k_cache` and `v_cache`
shaped `[num_blocks, num_kv_heads, head_size/x, block_size, x]` and
`[num_blocks, num_kv_heads, head_size, block_size]` respectively; the page
table is a per-sequence array of `physical_block_number` values that the
kernel indexes with directly (`k_ptr = k_cache + physical_block_number *
kv_block_stride + ...`), so "paging" here means each sequence carries a
list of physical block indices, and the attention kernel does one indirect
lookup per block visited rather than requiring contiguous KV storage per
sequence.

## Q6: continuous batching for a single-GPU engine, minimum viable design

Source: same vLLM PagedAttention doc as Q5 (primary, retrieved 2026-09-15),
plus the block/page mechanism already described there, which is the
concrete mechanism a from-scratch engine would need to replicate for
gathering different KV lengths into one m>1 decode GEMV. A dedicated vLLM
v1 or SGLang scheduler-loop blog/doc covering prefill chunking and
preemption specifically was not fetched in this pass due to budget; what
follows is limited to what the PagedAttention doc itself establishes plus
terms it defines, not a separate scheduler source. Flagged as partially
unanswered: the scheduler-loop specifics (prefill chunking cadence,
preemption policy) were not sourced this round.

**What the kernel needs from the engine, per terms the vLLM doc defines**:
a **sequence** is one client request contributing one query token per
decode step (`num_seqs` in the kernel's shapes equals total tokens
processed in the batch); the **context** is that sequence's already-
generated tokens; each sequence owns a **block table** (a list of physical
block indices) that lets its KV live in non-contiguous physical blocks.
The **grid** shape is `(num_heads, num_seqs, max_num_partitions)`, so one
thread block handles one head of one sequence, meaning independent
requests with different context lengths are naturally handled: the kernel
just iterates a different number of blocks per sequence (context length
determines the per-sequence loop trip count, not the batch shape). This is
the mechanism by which an m>1 decode GEMV is formed from independent
requests with different KV lengths: the batch dimension is the sequence
index, and each sequence's own block-table length independently bounds
how many KV blocks that thread block visits, no padding to a common length
required inside the kernel itself. For a first single-GPU version, the
minimum state per request implied directly by this design is: the running
KV block table (list of physical block ids) and current context length;
everything else (prefill chunking granularity, preemption/eviction policy)
is scheduler-level policy on top of this mechanism, and was not sourced
this round.

## Q7: 3-bit weight formats worth a kernel

Sources: ExLlamaV3 README, https://raw.githubusercontent.com/turboderp-org/
exllamav3/master/README.md (retrieved 2026-09-15); llama.cpp PR #5676
title/description found via search ("IQ3_S: a much better alternative to
Q3_K", https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/5676/
overview, retrieved 2026-09-15, snippet only, not the full PR body).

**EXL3 dequant cost shape: trellis/Viterbi, not a simple lookup table.**
Per the ExLlamaV3 README: "EXL3 quantization is a streamlined variant of
QTIP from Cornell RelaxML... By computing Hessians on the fly and thanks to
a fused Viterbi kernel, the quantizer can convert a model in a single
step." QTIP (referenced paper https://arxiv.org/abs/2406.11235) encodes
weights through a trellis code, decoded via a Viterbi-style sequential
traversal rather than a flat per-value lookup table (the difference the
brief asked about): a lookup-table dequant (like the IQ series' grid codes)
is embarrassingly parallel per element, while trellis decode has a
sequential dependency along the trellis path, which is a harder fit for a
memory-bound GEMV than a pure per-element table lookup would be. EXL3
supports 2 to 8 bits per weight (README, "EXL3, based on QTIP, plus 2-8
bit cache quantization"). No RDNA3 port or discussion of trellis decode
feasibility on RDNA3 was found in this pass; ExLlamaV3 is stated in its own
README to require CUDA-toolkit-built PyTorch (`torch>=2.6.0`, CUDA
`>=12.4`), with no ROCm/HIP build path mentioned. Flagged as unanswered:
whether anyone has attempted an RDNA3 port of EXL3's trellis decode.

**IQ3_S vs Q3_K: found the PR that introduced it, but not the perplexity
numbers themselves.** Search surfaced llama.cpp PR #5676, "IQ3_S: a much
better alternative to Q3_K", with the search snippet stating: "The existing
Q3_K_XS quantization mix (a mix of Q3_K, Q2_K and Q4_K) is replaced with a
simpler and much better mix of IQ3_XXS and IQ3_S with an approximate bpw of
3.25." The full PR body (with its actual perplexity table across model
sizes) was not fetched in this pass due to budget; the bits-per-weight
figure (~3.25 bpw for the IQ3_XXS/IQ3_S mix) is the only hard number
retrieved. IQ-series formats (including IQ3_S) dequantize via small grid
codebook lookups (per `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_s.comp`,
located but not fetched this pass), which is architecturally the
lookup-table shape the brief contrasted against Q3_K's arithmetic
(scale-and-shift) dequant; this is stated with the source file located but
its contents unread, so treat the dequant-shape claim for IQ3_S specifically
as lower confidence than the other sourced claims in this report. No
Q4_0/Q4_K_M cross-comparison perplexity numbers for either format on
7B-9B models were retrieved. Flagged as unanswered: the actual perplexity
deltas requested.

## Q8: RDNA3 dispatch and persistent-kernel facts

Sources: https://github.com/ROCm/legacy-rocm-build/issues/6409 (opened
2026-07-11, retrieved 2026-09-15, a reproducible benchmark filed by the
hipEngine project against a matched ROCm 7.15.0a20260711 / TheRock build on
a Radeon Pro W7900, which is the workstation SKU of the same gfx1100 Navi 31
die as the RX 7900 XTX); `ggml/CMakeLists.txt` and `ggml/src/ggml-hip/
CMakeLists.txt` (cited under Q1, retrieved 2026-09-15); zolotukhin.ai AMD
RDNA3/RDNA4 reference (cited in web research 1's Q3, retrieved 2026-09-15,
for the LDS bandwidth figure repeated below).

**Measured HIP kernel launch floor on gfx1100 (W7900), from a matched-stack
HIP-vs-Vulkan microbenchmark**: for a single dispatch of 1 thread block
("the tiny kernel adds 1.0 to each in-range output"), HIP's GPU-timed
elapsed was 14.980 microseconds versus Vulkan's 1.480 microseconds on the
same hardware and driver stack (a 10.12x gap). Once warmed inside a
941-node HIP graph replay, the per-dispatch cost dropped to about 3.865
microseconds at 1 block and 3.845 microseconds at 128 blocks (Vulkan's
command-buffer replay stayed near 0.82-0.88 microseconds at the same
sizes). The issue's own framing of this: "HIP graph replay / tiny-dispatch
floor. Vulkan command-buffer replay is faster on both devices: serialized
GPU ratios are 2.44x-10.12x on gfx1100." The reporter explicitly asks
whether "3.87 us/node on gfx1100... for a warmed 941-node graph" is
expected, i.e. this is an open, unresolved ROCm issue, not a settled
number; treat ~3.85-15 microseconds as the measured range (warmed graph
replay to cold single dispatch) on this exact chip, not a vendor-published
constant.

**HIP graph support status on gfx11 in current ROCm: available and
default-on, gated to ROCm/HIP >= 6.1.** From the CMake source (Q1): `option
(GGML_HIP_GRAPHS "ggml: use HIP graph" ON)` with a hard `hip_VERSION <
6.1` fatal-error gate in `ggml-hip/CMakeLists.txt`. The benchmark above was
run on ROCm 7.15.0a20260711 (TheRock nightly), well past that floor, and
explicitly exercises HIP graph replay (not just bare dispatch), so graph
support itself is not in question on gfx1100 today, only its replay-floor
performance relative to Vulkan's command buffers.

**LDS bandwidth per CU**: repeating the figure already sourced in web
research 1's Q3 (zolotukhin.ai AMD RDNA3/RDNA4 reference, retrieved
2026-09-15) since it belongs here too: approximately 64 bytes/clock/CU,
dual-issue capable, 32 banks of 4 bytes each, with N-way bank conflicts
serializing to roughly N times the latency.

**`s_sleep` granularity**: not found in this pass. No RDNA3 ISA guide or
GPUOpen page with `s_sleep`'s specific cycle/microsecond granularity was
successfully fetched (the AMD RDNA3 ISA PDF was blocked by the fetcher's
anti-bot check again this round, same failure mode as web research 1's
Q3). Flagged as unanswered.

## Q9 (small): chat template flags through minijinja

Sources: vLLM docs, https://docs.vllm.ai/en/stable/features/
reasoning_outputs/ (retrieved 2026-09-15); llama.cpp GitHub issue #20409,
"Eval bug: Qwen3.5 enable_thinking=false via --chat-template-kwargs..."
(https://github.com/ggml-org/llama.cpp/issues/20409, retrieved 2026-09-15).

**Field name, both engines**: `chat_template_kwargs` is the OpenAI-
compatible request-body field both vLLM and llama.cpp's server accept, with
`enable_thinking` as a key inside it (e.g. `{"chat_template_kwargs":
{"enable_thinking": false}}`), matching Qwen's own chat template variable
name. vLLM additionally exposes a server-level default via
`--default-chat-template-kwargs` ("You can set default chat_template_kwargs
at the server level using the --default-chat-template-kwargs CLI argument.
This is useful for configuring reasoning behavior across all requests
without requiring clients to specify it in each request", vLLM reasoning
outputs doc). llama.cpp exposes the equivalent as a CLI flag,
`--chat-template-kwargs`, per issue #20409's own repro steps, though that
same issue is a live bug report that setting `enable_thinking: false` for
Qwen3.5 through this path currently does not suppress the thinking block
in llama.cpp, i.e. the flag exists and is documented but is reported broken
for at least this one model family as of the issue's filing.

**Granite 4's expected variable name: not confirmed.** Fetching
`ibm-granite/granite-4.0-h-1b`'s actual `chat_template.jinja`
(https://huggingface.co/ibm-granite/granite-4.0-h-1b/raw/main/
chat_template.jinja, retrieved 2026-09-15) found no `thinking` or
`enable_thinking` variable at all in that template; it only branches on
`available_tools`, `documents`, and message roles, with no reasoning-mode
toggle. This may mean the h-1b checkpoint's template genuinely has no
configurable thinking mode (i.e. Granite 4.0's reasoning toggle, if any,
lives in a different checkpoint's template, such as a larger or
explicitly-reasoning-tuned Granite 4 variant not fetched in this pass), or
that Granite 4 does not use the `enable_thinking` convention at all.
Flagged as unanswered: Granite 4's actual variable name was not located.
