# Web research 3: primary sources only (papers, vendor docs, source code)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-web-research-3.md`. Rule this
round: papers, vendor documents and source code only, no blogs, no
aggregator pages, no secondary write-ups. All retrieval via firecrawl
(`firecrawl_search`, `firecrawl_scrape`, `onlyMainContent`). The
`firecrawl_research_*` tools returned 404 for every call in this session
(no OAuth/API-key connection present per the MCP server's own instructions),
so all paper text below was pulled via `firecrawl_scrape` on each paper's
own `arxiv.org/html/<id>` rendering, which arXiv now serves for most papers
as full HTML. Retrieval date for every source: 2026-09-15. One exception is
noted inline (Hazy Research's own technical blog post on their own
megakernel, treated as a first-party technical report per the brief's own
wording, not a third-party write-up about someone else's work).

All eight sections were worked in order. A few sub-items came back
unconfirmed; each is flagged explicitly at its own point below rather than
filled with a guess.

## P1: speculative decoding, the rules and their acceptance math

### Leviathan et al. 2023, arXiv 2211.17192, "Fast Inference from Transformers via Speculative Decoding"

Source: https://arxiv.org/html/2211.17192, retrieved 2026-09-15.

**Algorithm 1's acceptance rule and residual distribution, quoted from the
paper's own pseudocode** (Section 2.3):

```
for i = 1 to gamma:
    q_i(x) <- M_q(prefix + [x_1,...,x_{i-1}])
    x_i ~ q_i(x)
p_1(x),...,p_{gamma+1}(x) <- M_p(prefix), ..., M_p(prefix + [x_1,...,x_gamma])
r_1 ~ U(0,1), ..., r_gamma ~ U(0,1)
n <- min({i-1 | 1<=i<=gamma, r_i > p_i(x)/q_i(x)} union {gamma})
p'(x) <- p_{n+1}(x)
if n < gamma:
    p'(x) <- norm(max(0, p_{n+1}(x) - q_{n+1}(x)))
t ~ p'(x)
return prefix + [x_1,...,x_n, t]
```

Here `q` is the draft model, `p` is the target model. A guess `x_i` is kept
when the drawn `r_i` does not exceed the ratio `p_i(x_i)/q_i(x_i)`
(equivalent to accepting with probability `min(1, p/q)`); on the first
rejection, the paper resamples from `norm(max(0, p - q))`, exactly the
residual distribution the brief asked to quote.

**Expected-tokens-per-pass expression (their Equation 1, Section 3.1,
Definition 3.1 and the surrounding text, quoted directly)**: with
`alpha = E(beta)` the mean acceptance rate (assumed i.i.d. per position),
"the number of tokens produced by a single run of Algorithm 1 is a capped
geometric variable, with success probability `1-alpha` and cap `gamma+1`,"
and Section 3.3 states the walltime-improvement factor for a
negligible-cost draft model directly as

```
E[#generated tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)
```

bounded above by `1/(1-alpha)` as `gamma` grows.

**Statement on truncation (top-p/top-k) applied to target and draft,
quoted, Section 2**: "while there are many methods and parameters of
sampling, like argmax, top-k, nucleus, and setting a temperature, and
popular implementations usually treat them differently at the logits
level, they can all easily be cast into standard sampling from an adjusted
probability distribution... Going forward we'll assume that `p(x)` and
`q(x)` are the distributions from `M_p` and `M_q` respectively, adjusted
for the sampling method." So the paper's own framework applies whatever
truncation is configured to both `p` and `q` before the accept/reject rule
runs; it is not a separate step layered on top.

### Chen et al. 2023, arXiv 2302.01318, "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind)

Source: https://arxiv.org/html/2302.01318, retrieved 2026-09-15. Note their
notation is the mirror of Leviathan's: here `q` is the target and `p` is
the draft.

**Acceptance rule and residual (Section 4.2, quoted)**: accept
`x_tilde_{n+1}` with probability `min(1, q(x_tilde_{n+1}|prefix) /
p(x_tilde_{n+1}|prefix))`; on rejection, resample from
`(q(x|prefix) - p(x|prefix))_+` where `(f(x))_+ = max(0,f(x)) /
sum_x max(0,f(x))`. Algebraically identical shape to Leviathan's rule with
the model roles swapped in notation.

**Statement on truncation, quoted, Section 4.2**: "With standard sampling
methods such as nucleus, top-k sampling and adjusting temperature, we can
modify the probabilities accordingly before applying this rejection
sampling scheme. We have observed that the overall acceptance rate is
robust to the exact parameters used." Their own Table 1 experiments use
this directly: "XSum was executed with nucleus parameter p=0.8, and
HumanEval with p=0.95 and temperature 0.8," i.e. nucleus sampling combined
with rejection sampling is the paper's actual evaluation setup, not just a
theoretical aside.

### EAGLE, EAGLE-2, EAGLE-3 acceptance length

Sources: https://arxiv.org/html/2401.15077 (EAGLE), 2026-09-15;
https://arxiv.org/html/2406.16858 (EAGLE-2), 2026-09-15;
https://arxiv.org/html/2503.01840 (EAGLE-3), 2026-09-15.

None of the three papers sweeps a draft-token-count parameter `K` from 2 to
6 and reports acceptance length at each value the way the brief assumed;
they report average acceptance length `tau` per target model and task at a
fixed draft-tree configuration instead. Reporting what is actually in the
tables:

- **EAGLE, Table 2** (MT-bench, `tau` = average tokens accepted per target
  forward pass): Vicuna 7B/13B/33B and LLaMA2-Chat 7B/13B/70B at
  temperature=0 range `tau` = 3.62 to 3.98; at temperature=1, 3.17 to 3.46.
  **EAGLE, Table 3** (Mixtral 8x7B Instruct, the paper's own MoE case): only
  a 1.5x speedup, explained in the paper's own words: "MoE models typically
  require reading the weights of only two experts per token during vanilla
  autoregressive decoding. However, during the verification phase of
  speculative sampling, processing multiple tokens may necessitate
  accessing the weights of more than two experts, contrasting with dense
  decoder-only models where all weights are read regardless of the number
  of tokens forwarded." This is the direct, primary-source explanation for
  why MoE speculative decoding gives a smaller win than dense-model
  speculative decoding: verifying a multi-token draft against a MoE target
  forces reading more experts' weights per step than a plain one-token
  decode would, partially offsetting the memory-bandwidth win speculative
  decoding is chasing in the first place. Directly relevant to mojo-baro's
  own 256-expert, top-8 qwen35moe MTP work.
- **EAGLE-2, Table 1/Table 2**: "Each drafting-verification cycle of
  EAGLE-2 generates approximately 4-5.5 tokens" across its tested models
  and tasks (quoted).
- **EAGLE-3, Table 1**: speedup 3.0x-6.5x over vanilla decoding, "20%-40%
  improvement over EAGLE-2"; HumanEval reaches "an average acceptance
  length of up to 7.5" (quoted). EAGLE-3's own restatement of the
  acceptance rule (Section on verification, quoted): acceptance probability
  `min(1, p_{j+i}(t_hat)/p_hat_{j+i}(t_hat))`, rejection resample from
  `norm(max(0, p_{j+i} - p_hat_{j+i}))`, citing "Appendix A.1 of Leviathan
  et al. 2023" directly, confirming EAGLE-3 uses the identical
  Leviathan/Chen rule rather than a relaxed one.

### DeepSeek-V3 MTP acceptance rate, arXiv 2412.19437

Source: https://arxiv.org/html/2412.19437, retrieved 2026-09-15. Model
architecture (Section 4.1, quoted): "Each MoE layer consists of 1 shared
expert and 256 routed experts... Among the routed experts, 8 experts will
be activated for each token... The multi-token prediction depth D is set
to 1, i.e., besides the exact next token, each token will predict one
additional token." This is close to mojo-baro's own qwen35moe shape (256
experts, top-8). Acceptance number, quoted directly from the paper's own
inference section: "Based on our evaluation, the acceptance rate of the
second token prediction ranges between 85% and 90% across various
generation topics, demonstrating consistent reliability. This high
acceptance rate enables DeepSeek-V3 to achieve a significantly improved
decoding speed, delivering 1.8 times TPS."

### Cost model of the verify window: Sequoia, arXiv 2402.12374

Source: https://arxiv.org/html/2402.12374, retrieved 2026-09-15. The
paper's own closed-form speedup expression (Section 5, "hardware-aware
tree optimizer," quoted):

```
Speedup(n,d) = G(n,d) / (t(n) + d*c)
```

where `G(n,d)` is the expected number of tokens generated by verifying a
speculation tree of size `n` and depth `d` (computed via the paper's
dynamic program), `t(n)` is the hardware-measured time to verify `n`
tokens divided by the time to verify 1 token, and `c` is the
hardware-measured time to draft 1 token divided by the time to verify 1
token. This is exactly "tokens per pass divided by per-pass cost" the
brief asked for, with `d*c` accounting for the sequential draft-generation
cost that a wider or deeper speculation tree adds. The paper also proves
(Section 3.1.2) that Sequoia's optimal-tree construction scales the
expected generated-token count roughly logarithmically with tree size,
whereas prior structures (independent `k`-sequence trees, as in SpecInfer)
asymptote and stop improving past a bounded tree size, which is why
Sequoia is cited here rather than SpecInfer directly: SpecInfer's own
paper (arXiv 2305.09781) was not separately fetched this round since
Sequoia's Section 1 already quotes and cites the SpecInfer limitation
being addressed ("SpecInfer constructs a token tree using k independent
sequences, a topology that is bounded by the expected number of tokens it
can accept, regardless of the tree size").

## P2: KV cache quantization

### KIVI, arXiv 2402.02750

Source: https://arxiv.org/html/2402.02750, retrieved 2026-09-15.

**Why keys and values are quantized along different dimensions, quoted**:
"For key cache, there are a few fixed channels whose magnitudes are very
large... key cache should be quantized per-channel, i.e., group elements
along the channel dimension and quantize them together. In this way, it
can confine the error to each individual channel, without impacting the
other normal channels." "For value cache, there is no obvious outlier
pattern. Although value cache has no obvious outlier pattern, we
experimentally show that it can only be quantized per-token because it is
used to calculate the attention output, which is essentially a value cache
mixer... the per-token quantization can confine the error inside each
individual token and ensure that the quantization of one token does not
adversely impact the others."

**2-bit results, Table 1, group size 32, CoQA / TruthfulQA (columns as
labeled in the paper; C = per-channel, T = per-token)**:

| Configuration | CoQA | TruthfulQA |
| --- | --- | --- |
| 2bit (K-T, V-T) | 52.93 | 24.98 |
| 2bit (K-C, V-C) | 2.88 | 0.74 |
| 2bit (K-T, V-C) | 2.80 | 0.26 |
| 2bit (K-C, V-T) | 63.53 | 28.60 |

K-per-channel/V-per-token (the paper's chosen KIVI design) is the best row
on both tasks; quantizing the value cache per-channel collapses accuracy
regardless of how the key cache is quantized (the paper's own OB2).

**Headline numbers**: 2.6x peak memory reduction (Llama-2-7B), up to 4x
larger batch size, 2.35x-3.47x throughput, "with little to no accuracy
drop," tuning-free.

### KVQuant, arXiv 2401.18079

Source: https://arxiv.org/html/2401.18079, retrieved 2026-09-15. Adds two
findings beyond KIVI's per-channel/per-token split: **pre-RoPE key
quantization** ("Keys exhibit outliers in specific channels before
applying RoPE. However, the outlier channel magnitudes become less
consistent after applying RoPE... We address this by quantizing Keys
per-channel before RoPE is applied," since RoPE's per-pair rotation
scrambles which channel holds the large-magnitude outlier) and
**per-vector dense-and-sparse outlier isolation** (separate outlier
threshold per channel or per token rather than one threshold per layer,
"we can efficiently and accurately identify and compress outlier values in
order to store them compactly in a separate sparse representation... By
removing only 1% of outliers, we can attain under 0.1 perplexity
degradation on both Wikitext-2 and C4 for 3-bit KV cache quantization").
Ablation numbers quoted directly: per-channel Key + per-token Value gives
"a 3.82 perplexity improvement on Wikitext-2 for 3-bit LLaMA-7B
quantization" versus the naive per-token/per-token baseline; their
non-uniform (sensitivity-weighted k-means) datatype gives "0.29 perplexity
improvement on Wikitext-2 relative to 3-bit uniform methods." Reported
speedup: up to ~1.7x versus fp16 matrix-vector multiplication baselines on
LLaMA-7B, enabling "up to 1 million" context on a single A100-80GB and "up
to 10 million" on an 8-GPU system.

**Int8 KV with per-block scales on a 7B-9B model (ggml's q8_0 shape,
block 32)**: no dedicated academic paper measuring this exact
configuration was found in this pass. Both KIVI and KVQuant target 2-3 bit
and use per-channel or non-uniform quantization rather than llama.cpp's
plain per-block-32 int8 scheme; no primary source found for that specific
combination, consistent with the same gap already flagged (from a blog,
not a paper) in web research 2.

## P3: int8 activation quantization accuracy

### LLM.int8(), arXiv 2208.07339

Source: https://arxiv.org/html/2208.07339, retrieved 2026-09-15.

**The outlier-channel argument, in the paper's own words**: "large
features with magnitudes up to 20x larger than in other dimensions first
appear in about 25% of all transformer layers and then gradually spread to
other layers as we scale transformers to 6B parameters. At around 6.7B
parameters, a phase shift occurs, and all transformer layers and 75% of
all sequence dimensions are affected by extreme magnitude features. These
outliers are highly systematic: at the 6.7B scale, 150,000 outliers occur
per sequence, but they are concentrated in only 6 feature dimensions
across the entire transformer. Setting these outlier feature dimensions to
zero decreases top-1 attention softmax probability mass by more than 20%
and degrades validation perplexity by 600-1000% despite them only making
up about 0.1% of all input features. In contrast, removing the same amount
of random features decreases the probability by a maximum of 0.3% and
degrades perplexity by about 0.1%." Their outlier threshold is a fixed
magnitude of 6.0 (`alpha=6.0`), and the fix (mixed-precision decomposition:
16-bit for the outlier dimensions, 8-bit for the other 99.9%) is what
LLM.int8() actually ships.

**Table 1 (their own table number), C4 validation perplexity by model
size and quantization method, 125M-13B**: absmax, row-wise, zeropoint, and
vector-wise quantization all degrade sharply once past 6.7B parameters
(13B 8-bit perplexity worse than 6.7B 8-bit perplexity for those methods);
LLM.int8() is described as "the only method that preserves perplexity" and
"the only method with a favorable scaling trend" across the full 125M to
13B range. Weights alone (no activation quantization) are separately
noted, citing the same paper's own framing, to tolerate INT8 or even INT4
without accuracy loss; the outlier problem in this paper is specifically
about **activation** quantization, not weight quantization.

### SmoothQuant, arXiv 2211.10438

Source: https://arxiv.org/html/2211.10438, retrieved 2026-09-15. **Table 1
(their own table number)**, average accuracy on WinoGrande/HellaSwag/
PIQA/LAMBADA for INT8 per-channel activation quantization across increasing
OPT model sizes: 64.8%, 65.6%, 68.0%, 69.4%, 71.4% (matching the FP16
baseline at each size, per the paper's framing that per-channel activation
quantization is the only granularity that "successfully bridges the
accuracy with the FP16 baseline"), but the paper immediately notes
per-channel activation quantization "does not map well to hardware-
accelerated GEMM kernels" since scaling can only be applied along the
matrix multiplication's outer dimensions on real tensor-core hardware,
which is why SmoothQuant instead migrates the outlier magnitude from
activations into weights offline via a per-channel smoothing factor
`s_j = max(|X_j|)`, so that both the smoothed activation and the rescaled
weight are easy to quantize per-tensor or per-token at inference time.
Reported result: "up to 1.56x speedup and 2x memory reduction... with
negligible loss in accuracy" for W8A8 across OPT, BLOOM, GLM, MT-NLG,
Llama-1/2, Falcon, Mistral, and Mixtral.

**Isolating per-block-32 int8 activation quantization (ggml's q8_1 shape)
on a pre-quantized 4-bit-weight model**: neither paper covers this
specific combination. Both LLM.int8() and SmoothQuant are W8A8 schemes
starting from full-precision weights, not W4A8 with an already-quantized
4-bit weight tensor the way ggml's k-quant + q8_1 activation path works.
No primary paper isolating that exact combination was found in this pass,
matching the finding already reported (without a primary source) in web
research 2's Q2.

## P4: MoE decode kernels and persistent kernels

### MegaBlocks, arXiv 2211.15841

Source: https://arxiv.org/html/2211.15841, retrieved 2026-09-15. This is a
**training**-throughput paper (block-sparse dMoE kernels to avoid token
dropping during MoE **training**), not a single-request decode-latency
paper; flagging that distinction since the brief listed it under
"inference kernels." Mechanism: MoE expert computation reformulated as
block-sparse matrix products (128x128 block size, chosen to match the
highest-throughput tile size found for dense CUTLASS kernels on the
authors' hardware), using "blocked-CSR-COO encoding and transpose indices"
to support efficient SDD/DSD/DDS block-sparse products in either
transpose order. Reported gain: "end-to-end training speedups of up to 40%
over MoEs trained with the state-of-the-art Tutel library and 2.4x over
DNNs trained with the highly-optimized Megatron-LM framework." No
per-token decode launch-overhead number is in this paper; it is not the
right primary source for that number.

### Mirage Persistent Kernel (MPK), arXiv 2512.22219

Source: https://arxiv.org/html/2512.22219v1, retrieved 2026-09-15. Abstract
number, quoted: "MPK significantly outperforms existing kernel-per-operator
LLM serving systems by reducing end-to-end inference latency by up to
1.7x." Mechanism, matching exactly what the brief asked for ("grid
barrier, task queue, event counters"): MPK compiles a model into an
SM-level task graph called a `ttGraph`, where "each node represents either
a task or an event... every task only has outgoing edges to triggering
events and incoming edges from dependent events. A task is ready for
execution when its dependent events are all activated and notifies its
triggering event upon completion." At runtime, "the runtime partitions a
GPU's SMs into workers, which maintains a dedicated task queue and
executes all assigned tasks in a first-in-first-out order, and schedulers,
which maintain dependency across tasks and assign tasks when their
prerequisites are satisfied," using "device-memory synchronization
primitives" rather than kernel-launch barriers, all inside one persistent
mega-kernel with no further kernel launches during model execution. The
paper explicitly contrasts this with plain CUDA Graphs: "while CUDA Graphs
capture dependencies only at the kernel level, ttGraphs operate at the
granularity of individual SM tasks and sub-kernel events... This design
allows MPK to exploit parallelism that is inaccessible to CUDA Graphs or
kernel-level execution models."

### Hazy Research "Look Ma, No Bubbles!" megakernel report

Source: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles,
authored by the Stanford Hazy Research group describing their own system
(treated here as a first-party technical report, not a third-party
write-up), published 2025-05-27, retrieved 2026-09-15; code at
https://github.com/HazyResearch/Megakernels. This is the batch-1,
single-model (Llama-3.2-1B) case the brief specifically named.

**The quantified root cause, quoted**: "popular LLM inference engines --
vLLM and SGLang -- are only able to use at most 50% of available GPU
bandwidth when running this workload on an H100. The root of the problem...
is that existing systems break down a model forward pass into around a
hundred separate kernels." Measured per-kernel-launch cost on H100: "about
2.1 microseconds" on a plain CUDA stream, dropping to "around 1.3
microseconds" with CUDA graphs, still non-zero overhead. (Cross-reference:
web research 2's Q8 measured a comparable, but notably higher, per-dispatch
floor on gfx1100/W7900 HIP graphs, about 3.85 microseconds warmed versus
NVIDIA's 1.3 microseconds here, both on the same "warmed graph replay"
condition, though the two measurements come from different benchmark
harnesses and are not a controlled apples-to-apples comparison.)

**Mechanism**: an on-GPU interpreter where each streaming multiprocessor
runs a pre-scheduled sequence of instructions (7 instruction types for the
full Llama-1B forward pass: fused RMSNorm+QKV+RoPE, attention, attention
reduction, O-projection+residual, fused RMSNorm+up-gate+SiLU,
down-projection+residual, final RMSNorm+LM-head), with shared memory
divided into 13 pages of 16KiB on an H100 that instructions explicitly
request and release so the interpreter can start the next instruction's
loads as soon as a page frees up, and dependencies enforced by
"a simple counter system... an array of counters (i.e. integers) in GPU
global memory with a starting value of zero. Whenever an instruction
completes, it increments one of these counters. Similarly, whenever a new
instruction starts, it must wait for some of these counters to reach a
target value" -- this is the "event counters" mechanism the brief asked
about, in the authors' own description.

**Gain, quoted**: "on an H100, we use 78% of memory bandwidth and
outperform existing systems by over 1.5x... On an H100, our megakernel
runs almost 2.5x faster than vLLM and over 1.5x faster than SGLang. On a
B200, the gap with vLLM rises to over 3.5x, and we remain more than 1.5x
faster than SGLang, too," with a detailed 600-microsecond B200
forward-pass breakdown given in the post (activation store/wait/load ~250
microseconds, RMSNorm+matrix-vector compute ~200 microseconds, weight
loading ~30 microseconds, cross-warp synchronization overhead ~40
microseconds, setup/misc ~80 microseconds).

**DeepSpeed-MoE inference section**: not fetched this round due to time
budget; flagged as unanswered rather than guessed at.

## P5: serving systems

### Orca, OSDI 2022

Source: https://www.usenix.org/system/files/osdi22-yu.pdf (official USENIX
proceedings PDF, "ORCA: A Distributed Serving System for Transformer-Based
Generative Models," Yu, Jeong, Kim, Kim, Chun), retrieved 2026-09-15.

**The one mechanism**: two techniques, **iteration-level scheduling**
(schedule and admit/evict requests at the granularity of a single decode
iteration rather than a whole request or a whole static batch, so a
request that finishes does not force the rest of the batch to wait, and a
new request can join mid-flight) combined with **selective batching**
(batch the operations that tolerate ragged, differently-shaped sequences
across requests, such as the linear/FFN layers, while handling attention
per-request separately since its computation cannot be naively batched
across requests with different context lengths sharing one KV cache).

**The one number, quoted**: "showing 36.9x throughput improvement at the
same level of latency compared to NVIDIA FasterTransformer."

### PagedAttention / vLLM, arXiv 2309.06180 (SOSP 2023)

Source: https://arxiv.org/html/2309.06180, retrieved 2026-09-15.

**The one mechanism**: KV cache managed as fixed-size, non-contiguously
stored **blocks** (their own term, "KV block size B"), with a per-request
**block table** mapping logical block indices to physical block indices,
directly analogous to OS virtual-memory paging. This eliminates the three
sources of memory waste the paper names and diagrams in its own Figure 3:
reserved-but-unused slots for a request's maximum possible length,
internal fragmentation from over-provisioning, and external fragmentation
from a buddy-style allocator.

**Block size, from the paper's own ablation (Section 7.2, quoted)**: "In
the ShareGPT trace, block sizes from 16 to 128 lead to the best
performance. In the Alpaca trace, while the block size 16 and 32 work
well, larger block sizes significantly degrade the performance... In
practice, we find that the block size 16 is large enough to efficiently
utilize the GPU and small enough to avoid significant internal
fragmentation in most workloads. Accordingly, vLLM sets its default block
size as 16." This upgrades web research 2's blog-sourced "block size 16"
claim to a primary citation with the paper's own reasoning.

**The one number**: "vLLM improves the throughput of popular LLMs by 2-4x
with the same level of latency compared to the state-of-the-art systems,
such as FasterTransformer and Orca," attributed specifically to reducing
memory waste (near-zero, per the paper's own claim) so more requests fit
in the same KV cache memory budget, which is a complementary axis to
Orca's own iteration-level scheduling rather than a replacement for it
(the paper says as much directly: "Orca and PagedAttention in vLLM are
complementary techniques").

### Sarathi-Serve, arXiv 2403.02310

Source: https://arxiv.org/html/2403.02310, retrieved 2026-09-15 (this
appears to be an OSDI 2024 paper per the brief; the fetched arXiv listing
did not show a venue tag in the scraped content, so venue is stated per
the brief's own framing rather than independently re-verified here).

**The one mechanism**: **chunked-prefills** (split one prefill request
into "near equal sized chunks" processed over multiple iterations instead
of one long iteration) combined with **stall-free scheduling** (each
batch coalesces the ongoing decode tokens with one or more prefill chunks
from newly admitted requests, so a new request's prefill never fully
displaces or pauses in-flight decodes the way naive prefill-prioritizing
schedulers do).

**The one number, quoted**: "For Mistral-7B on single A100 GPUs, we
achieve 2.6x higher serving capacity and up to 3.7x higher serving
capacity for the Yi-34B model on two A100 GPUs as compared to vLLM. When
used with pipeline parallelism on Falcon-180B, Sarathi-Serve provides up
to 5.6x gain in the end-to-end serving capacity."

### SGLang RadixAttention, arXiv 2312.07104

Source: https://arxiv.org/html/2312.07104, retrieved 2026-09-15.

**The one mechanism**: KV cache blocks (page size one token) kept alive
after a request finishes and indexed in a shared **radix tree** across all
requests, with an LRU eviction policy that evicts leaves first "to enable
the re-use of their common ancestors until those ancestors become leaves,"
plus a cache-aware scheduler that sorts waiting requests by matched-prefix
length (equivalent to a depth-first-search visitation order over the tree,
which the paper proves is optimal for cache hit rate given a cache large
enough for the longest request).

**The one number, quoted**: "SGLang achieves up to 6.4x higher throughput
compared to state-of-the-art inference systems," with the cache-aware
scheduler reported to reach "96% of the optimal hit rate on average"
across their benchmark suite (cache hit rates observed ranging 50%-99%
depending on task).

## P6: low-bit weight formats

### QTIP, arXiv 2406.11235

Source: https://arxiv.org/html/2406.11235, retrieved 2026-09-15.

**Correction to web research 2's more cautious framing**: brief 2 (sourced
from ExLlamaV3's README, a secondary description) called trellis decode
"sequential" in general terms. QTIP's own paper is explicit that this is
true of a **generic** trellis but not of QTIP's specific "bitshift
trellis" construction: "TCQ-quantized sequences also cannot generally be
decoded in parallel, as the t-th element of `S_hat` could depend on up to
the first `tk` encoded bits [in a generic trellis]... In QTIP, we solve
these issues by introducing a series of fast compute-based Gaussian codes
designed for the hardware-efficient 'bitshift trellis.' Specifically, the
bitshift trellis supports parallel decoding, does not require storing
[the trellis graph], and our compute-based codes eliminate needing to
store a large node value codebook." The reason parallel decoding becomes
possible: in the bitshift trellis, "each group of V weights only depends
on a contiguous window of L bits in `S_hat`," a fixed, local dependency
window rather than a chain back to the start of the sequence, and
"obtaining the next compressed group of V weights in a sequence only
requires bitshifting by kV bits, which is supported on virtually all
hardware."

**Quantization-time cost (not decode-time)**: encoding still uses the
Viterbi algorithm, "which runs in O(2^L * T) time," i.e. linear in
sequence length T and exponential only in the fixed trellis parameter L
(the paper uses L=16 in its main results), quoted directly from Section 3.

**3-bit / low-bit results vs GPTQ, AQLM, QuIP#**: the paper's headline
claim (Section 1, quoted): "With QTIP, 2 bit models scale better than
theoretically optimal 4 bit models," and Table 1 (their own table number)
shows QTIP's compute-based codes (1MAD, 3INST, HYB) matching a pure-lookup
random-Gaussian trellis code's distortion rate at 2 bits while all
trellis-coded methods "outperform SQ and VQ and are significantly closer
to the infinite-length distortion-rate `D_R`" than either scalar or vector
quantization (QuIP#, AQLM) can get, since VQ's shaping advantage is capped
by codebook size scaling exponentially with dimension (QuIP# and AQLM are
both hardware-limited to dimension <=8, per the paper's own explanation of
why VQ codebooks that large no longer fit in L1 cache).

### AQLM (arXiv 2401.06118) and QuIP# (arXiv 2402.04396)

Not independently fetched this round beyond what QTIP's own paper already
quotes and cites about them (their dimension-8 hardware ceiling and
codebook-size tradeoffs, above); flagging that their own tables were not
independently pulled in this pass, so the comparison above is filtered
through QTIP's framing of them, not each paper's own words.

### llama.cpp PR #5676, "IQ3_S: a much better alternative to Q3_K"

Source: https://github.com/ggml-org/llama.cpp/pull/5676, retrieved
2026-09-15 (GitHub source, explicitly allowed by the brief).

**bpw and PPL vs Q3_K, quoted from the PR's own description**: "This PR
adds `IQ3_S`, a 3.4375 bpw quantization (i.e., the exact same size as
`Q3_K`) that has a significantly lower PPL compared to `Q3_K_S`... The
improvement in quantization error (defined as `PPL(Q)/PPL(fp16)-1`) is
40-70% depending on model," tested on LLaMA-v1, LLaMA-v2, and Mistral-7B.
It also adds `IQ3_M` ("a mix between the new `IQ3_S` and `Q4_K`. It has
basically the same PPL as the existing `Q3_K_M` at 0.15 bpw less") and
replaces the old `Q3_K_XS` mix with "a simpler and much better mix of
`IQ3_XXS` and `IQ3_S` with an approximate bpw of 3.25."

**Dequant cost shape, confirming the lookup-table characterization,
quoted**: "The `IQ` series of quants use 'codebooks' to encode groups of 4
or 8 weights. For `IQ3_S` this requires 4 memory loads from a lookup table
of 2048 bytes to setup one 128-bit SIMD register." This is architecturally
the lookup-table shape the brief asked to contrast against Q3_K's
arithmetic (scale-and-shift) dequant, and the PR's own portability note is
a useful caution for any future RDNA3 port: "Performance on the M2 Max CPU
with ARM_NEON intrinsics is pathetic -- only about 10 t/s for a 7B model
compared to 22.5 t/s for `Q3_K_S`... It seems Apple Silicon does not like
this very much," i.e. this specific lookup-table-heavy dequant shape has
measured, ISA-specific performance cliffs, not just a uniform cost
independent of the target hardware.

## P7: AMD primary documents

Source for the ISA items below: the AMD "RDNA3" Instruction Set
Architecture Reference Guide (document 70650), fetched successfully this
round via https://docs.amd.com/api/khub/documents/UkT_UPQL21KfKAMUBFnZTw/content
(retrieved 2026-09-15; the same document's PDF mirrors were blocked by
anti-bot checks in both prior research rounds, but this API-rendered copy
returned full text this time).

**`s_sleep` semantics and granularity, quoted verbatim from the ISA's own
"Dependency, Delay and Scheduling Instructions" table**: "`S_SLEEP` Cause
a wave to sleep for approx. 64*SIMM16[6:0] clocks. 's_sleep 0' sleeps the
wave for 0 cycles." `SIMM16[6:0]` is a 7-bit immediate (0-127), so the
granularity is 64 clocks per unit and the maximum single-instruction sleep
is about 8128 clocks. `S_WAKEUP` lets one wave in a work-group signal all
other waves in the same work-group to wake from `S_SLEEP` early ("if
waves are not sleeping, they are not affected by this instruction").
`S_NOP` is separately described as "like a short version of `S_SLEEP`"
for short, fixed repeat counts (1-16).

**`S_GETREG SHADER_CYCLES`, does it tick at shader clock, quoted**: the
ISA's own "Time" section states there are two methods, "'TIME' - measure
cycles in graphics core clocks (20 bit counter)" and "'REALTIME' - measure
time based on a fixed frequency, constantly running clock (typically
100MHz)." `SHADER_CYCLES` is the former: "This counter can be read via:
'S_GETREG S0, SHADER_CYCLES' and returns a 20-bit cycle counter value.
This counter is not synchronized across different SIMDs and should only
be used to measure time-delta within one wave. Reading the counter is
handled through the SALU which has a typical latency of around 8 cycles."
So yes, it ticks at the graphics/shader core clock (not a fixed frequency
like REALTIME), wraps at 20 bits, and is per-SIMD, not globally
synchronized across the GPU, matching this repo's existing finding that
`llvm.readsteadycounter` works but `s.memrealtime` crashes isel (that
finding used the REALTIME path; `SHADER_CYCLES`/TIME is the separate,
shader-clock-ticking counter).

**LDS size and banks per CU, quoted**: "LDS: Local Data Share. A 32-bank
scratch memory allocated to waves or work-groups." Size: "A single
work-group may allocate up to 64kB of LDS space," and internally, in the
dual-CU work-group-processor mode, "LDS is composed of two blocks of
memory of 64kB each" (one block affiliated with each of the two CUs in a
WGP), i.e. 64 KB addressable per work-group/CU, 128 KB physically present
per WGP, 32 banks.

**`v_dot4_i32_iu8` description, quoted verbatim (VOP3P instruction
table)**: "Dot product of signed or unsigned bytes," with pseudocode:

```
declare A : 32'I[4]; declare B : 32'I[4];
for i in 0:3 do
    A8 = S0[i*8+7 : i*8]; B8 = S1[i*8+7 : i*8];
    A[i] = NEG[0].u1 ? 32'I(signext(A8.i8)) : 32'I(32'U(A8.u8));
    B[i] = NEG[1].u1 ? 32'I(signext(B8.i8)) : 32'I(32'U(B8.u8));
endfor;
C = S2.i;
D0.i = A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3] + C;
```

`NEG[0]` and `NEG[1]` independently select signed versus unsigned
interpretation for each of the two 4xint8 operands. This confirms and
closes web research 3's own P3 finding about `ggml_cuda_dp4a` on RDNA3
compiling to `__builtin_amdgcn_sudot4(true, a, true, b, c, false)`: the
`true, true` arguments map exactly onto `NEG[0]`/`NEG[1]` here, i.e. ggml
requests the signed/signed interpretation of `V_DOT4_I32_IU8` for its
int8-times-int8 k-quant dot products, which the ISA confirms is the
instruction's own documented per-operand sign-selection mechanism, not an
undocumented or repurposed use of it.

**ROCm HIP graph support matrix for gfx11**: no dedicated ROCm
compatibility-matrix page naming gfx11 (RDNA3) HIP graph support was
located this round; the strongest primary evidence remains what web
research 2 already established from `ggml`'s own CMake source
(`option(GGML_HIP_GRAPHS "ggml: use HIP graph" ON)`, gated on
`hip_VERSION >= 6.1`), which is a build-system fact rather than a ROCm
vendor compatibility statement. Flagged as not independently confirmed
against a ROCm-authored support matrix in this pass.

**`hipIpcGetMemHandle` / `hipIpcOpenMemHandle`, same-process vs
cross-process rules**: source
https://rocmdocs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html
(official current ROCm/HIP API reference, retrieved 2026-09-15). The
documented rule, quoted: "Contexts that may open hipIpcMemHandles are
restricted in the following way. hipIpcMemHandles from each device in a
given process may only be opened by one context per device per other
process." Also documented: "During multiple processes, using the same
memory handle opened by the current context, there is no guarantee that
the same device pointer will be returned in `*devPtr`. This is different
from CUDA," and "This IPC memory related feature API on Windows may
behave differently from Linux." **No RDNA-consumer-specific restriction or
"not supported on consumer cards" statement was found in this API
reference or elsewhere in this pass.** If a documented consumer-RDNA
restriction on `hipIpcOpenMemHandle` exists that closes the B3 kill cause
for good, it was not located in this search; flagged as unanswered rather
than asserted. The API as documented describes cross-process behavior
generically for all HIP devices, with the two caveats above (one handle
opened by at most one context per device per other process; no pointer
stability guarantee across processes, unlike CUDA) being the only
documented irregularities, neither of which is framed as an RDNA-specific
restriction.

## P8: still open from brief 2

**Per-token launch count for a MoE decode in llama.cpp**: still not found
as an explicit stated number in any issue, PR, or benchmark table
inspected in this round or web research 2. What is now confirmed from
primary source (Section P1's EAGLE finding plus web research 2's own
`ggml-cuda.cu` reading) is the *shape* of the answer: llama.cpp's decode
path uses one dedicated, single-launch `mul_mat_vec_q_moe` /
`mul_mat_vec_f` kernel that gathers all selected experts' rows via an
`ids[]` array in one dispatch (established directly from source in web
research 1), plus separate fused kernels for gate+up (PR #16715) and
router softmax+top-k (PR #16130, both established in web research 2 from
am17an's own technical write-up, not re-verified against the PR bodies
themselves in this primary-sources-only round). No PR or issue was found
that states a total per-token kernel count for a MoE forward pass the way
Hazy Research's post states "around a hundred" for dense Llama-1B. Flagged
as unanswered.

**Granite 4.x chat-template thinking variable, from vendor documentation,
resolved**: source https://www.ibm.com/granite/docs/models/granite4-2
(official IBM Granite documentation site, retrieved 2026-09-15; the page
itself carries a banner stating the site "is no longer being updated" in
favor of Hugging Face and GitHub, so treat this as current-at-capture
rather than necessarily still the canonical location). Granite 4.2 (the
reasoning-tuned tier; the earlier `granite-4.0-h-1b` checkpoint fetched
directly in web research 2 has no thinking-mode branch in its own
`chat_template.jinja` at all, which is why that check came back empty)
"supports three thinking modes, selected via chat-template parameters,"
quoted directly from the vendor's own table:

| Mode | Template parameter(s) |
| --- | --- |
| Thinking (default) | `enable_thinking=True` |
| Non-thinking | `enable_thinking=False` |
| Low-effort | `enable_thinking=True, low_effort=True` |

So Granite uses the **identical** `enable_thinking` kwarg name as Qwen,
plus its own additional `low_effort` flag for a third mode not present in
Qwen's convention. IBM's own vLLM serving guidance for Granite 4.2 (same
page) specifies `--reasoning-parser granite_thinking_parser` (a
model-specific parser shipped in each model's own Hugging Face repo,
IBM's own recommendation over the built-in `nemotron_v3` parser "for
better formatting of reasoning output") and `--tool-call-parser
qwen3_coder`.

**vLLM/SGLang scheduler-loop specifics, from papers rather than blogs**:
now covered directly by P5 above rather than needing a separate fetch.
Orca's paper (P5) gives vLLM's scheduling ancestor mechanism directly
(iteration-level scheduling, selective batching); the PagedAttention paper
(P5) gives vLLM's own preemption mechanism in its own words (Section 4,
not separately quoted above): when the KV cache is full, vLLM evicts a
whole request's blocks and either recomputes them from scratch on
resumption or swaps them out to CPU memory, and the paper's own ablation
(Figure 19, referenced but not reproduced above) found "recomputation
overhead is never higher than 20% of swapping's latency" for small block
sizes, swapping is more efficient for large block sizes, with the two
comparable in the 16-64 block-size range vLLM actually uses; the
RadixAttention/SGLang paper (P5) gives SGLang's own scheduling mechanism
directly (radix-tree KV cache reuse plus longest-shared-prefix-first
cache-aware scheduling, proven equivalent to a depth-first-search
visitation order for optimal cache hit rate under a sufficiently large
cache).
