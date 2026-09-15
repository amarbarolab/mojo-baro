# Web research 4: prior art and limits for six headline capabilities

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-web-research-4.md`. Same
rules as brief 3: papers, vendor documents and source code only, no blogs,
no aggregator pages, no secondary write-ups. All retrieval via firecrawl.
`firecrawl_research_*` tools again returned 404 all session (no OAuth/API
key present); every paper below was pulled via `firecrawl_scrape` on its
own `arxiv.org/html/<id>` rendering or, where that failed, the official PDF
with `parsers: ["pdf"]`. Retrieval date for every source: 2026-09-15. All
six items worked in order, each against the same four questions: (a)
closest prior art and its number, (b) what it could not do that mojo-baro
would claim, (c) the hard limit that bounds the claim, (d) the check a
sceptical reviewer would demand.

## I1: a model file that runs itself and proves its numbers

**(a) Closest prior art.** llamafile (Mozilla / Mozilla.ai), source
https://docs.mozilla.ai/llamafile, retrieved 2026-09-15. Combines
llama.cpp with Cosmopolitan Libc into a single executable. Its own
description of what it bundles, quoted: "The weights for an LLM can be
embedded within the llamafile. We added support for PKZIP to the GGML
library. This lets uncompressed weights be mapped directly into memory,
similar to a self-extracting archive. It enables quantized weights
distributed online to be prefixed with a compatible version of the
llama.cpp software, thereby ensuring its originally observed behaviors can
be reproduced indefinitely." Runs on 6 OSes and both AMD64/ARM64 without
installation.

**(b) What it could not do that we would claim.** llamafile bundles the
engine and weights so the same binary always produces the same behavior;
it says nothing about performance claims. It does not embed a hardware
receipt, does not embed the kernel source that produced any benchmark
number, and has no rebuild-and-verify step: it guarantees behavioral
reproducibility (same input, same output, across time), not a
verified-performance claim (this number was measured on this hardware
with this kernel). No self-verification loop exists in the format.

**(c) Hard limit.** The Windows loader caps a runnable single-file
executable at 4GB (`llamafile` documentation's own README note, cited via
the GitHub listing in the same search pass, retrieved 2026-09-15: "Only
executables under 4GB can run on Windows, so any llamafile above 4GB won't
work"), which is a hard format/loader ceiling, not a tunable.

**(d) Sceptic's check.** Does the artifact, on its own, without a
network fetch or a human auditor, reproduce the exact hardware and kernel
identity that produced its own bundled benchmark number, and fail loudly
if run on different hardware or a different kernel build? llamafile
answers "same weights, same engine version, same output" but not this
question; nothing in its own documentation claims it.

**Adjacent primary sources checked, both negative for the specific claim.**

MLPerf Inference rules, https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc,
retrieved 2026-09-15. MLPerf does verify submitted numbers, but through a
human **audit process**, not an artifact-embedded proof: "up to two
submissions will be audited" per round, the auditor gets "two days of
hardware access," and "an audit is expected to be completed within a 90
day period" (all quoted). Reproducibility for "Available" submissions
requires the submitter to separately "specify software version of all
components, hardware configurations, software stacks, dockers, and
settings of all components and stacks" and "include the conversion
routine/scripts" (quoted from the rules' FAQ section). This is the closest
formal precedent for "a benchmark claim that must prove itself," but it is
an external, scheduled, human-mediated audit, not a one-command
self-verify baked into the artifact the way mojo-baro's GGUF hardware
receipt is.

Sigstore `model-signing` v1.0, https://blog.sigstore.dev/model-transparency-v1.0/,
Mihai Maruseac (Google Open Source Security Team), published 2025-04-04,
retrieved 2026-09-15 (first-party project announcement, not a
third-party write-up). Signs model weight files with a short-lived
Sigstore certificate bound to an OIDC identity, recorded in a public
transparency log so "a rogue insider cannot release new models as if they
are signed by the company," verifiable after the certificate expires.
This proves provenance and byte-level integrity of the weights, nothing
about the model's measured performance: the signature has no awareness of
benchmark content, hardware, or kernel source at all. No format found in
this pass (llamafile, GGUF metadata conventions, ONNX/safetensors signing,
or Sigstore) that embeds the kernel source that produced a benchmark
number; that combination appears to be novel to mojo-baro among the
sources checked here.

## I2: agents sharing KV/SSM state instead of text

**(a) Closest prior art, with numbers.**

CacheGen, arXiv 2310.07240, https://arxiv.org/html/2310.07240, retrieved
2026-09-15. Encodes a precomputed KV cache into a compact bitstream for
network transfer (not GPU-memory reduction), reporting "reduces the KV
cache size by 3.5-4.3x and the total delay in fetching and processing
contexts by 3.2-3.7x with negligible impact on the LLM response quality"
(quoted, tested 7B-70B models). CacheBlend, arXiv 2405.16444,
https://arxiv.org/html/2405.16444, retrieved 2026-09-15, solves a
different problem: when several KV caches are concatenated as a
non-prefix input (e.g. RAG chunks), naive concatenation loses
cross-attention between chunks; CacheBlend selectively recomputes "an
update fraction of less than 15%" of tokens to restore full-recompute
quality, "reduces time-to-first-token (TTFT) by 2.2-3.3x and increases the
inference throughput by 2.8-5x from full KV recompute" (quoted). DistServe,
arXiv 2401.09670, https://arxiv.org/html/2401.09670, retrieved 2026-09-15,
moves intermediate KV state between GPUs to disaggregate prefill and
decode phases, reporting "7.4x more requests or 12.6x tighter SLO" versus
state-of-the-art colocated serving, and states the disaggregation
communication overhead is "insubstantial" on modern cluster fabrics
(quoted).

LatentMAS, arXiv 2511.20639, https://arxiv.org/html/2511.20639v1, retrieved
2026-09-15. This is the closest match to "agents sharing KV state
directly," and it is real KV-cache handoff, quoted directly: "we extract
the KV-caches from all L transformer layers of A1 once, and define its
latent working memory... we perform layer-wise concatenation to update
its KV cache by prepending each K/V(l)_{A1,cache} and V(l)_{A1,cache} to
existing K(l)_{A2,cache} and V(l)_{A2,cache}." Reported gain: "LatentMAS
still achieves a 2.6x-7x speedup over the vLLM-optimized TextMAS," since
under 50 latent steps can substitute for the 20K+ output tokens a
text-based multi-agent chain-of-thought trace would otherwise need
(quoted). Interlat, arXiv 2511.09149, https://arxiv.org/html/2511.09149v1,
retrieved 2026-09-15, transmits only the **last hidden state per token**
(a single d-dimensional vector, not the full multi-layer KV cache),
further compressible: "latent messages can be compressed to as few as 8
tokens while maintaining competitive performance, achieving up to a 24x
reduction in communication latency" (quoted). Interlat's own cited
bandwidth comparison for why latent beats text at all: "~15 bits/token
vs. ~40k bits/hidden-state" (quoted, their own citation), meaning a single
hidden state actually carries far more raw bits than one token, and the
win comes from needing far fewer of them, not from each unit being
cheaper.

**(b) What none of them could do that we would claim.** All of the above
operate on plain Transformer attention KV caches. None handles or even
discusses **SSM recurrent state** handoff for a hybrid model (Qwythos-9B's
3-of-4-layers-SSM architecture has no KV cache at all for those layers, only
a fixed-size recurrent state per layer); this is a genuine gap in the
literature surveyed here, not just an unmeasured case. Separately,
LatentMAS's own implementation concatenates KV caches through HuggingFace's
in-process `past_key_values` interface between agents running inside the
same Python process on the same GPU, not a wire-format handoff between
separate OS processes the way mojo-baro's own KV/SSM checkpoint handoff
is built; no cross-process, cross-model-architecture handoff was found in
any of these five papers.

**(c) Hard limit.** For a literal KV/SSM handoff (not a compressed latent
summary), the receiving process must run an architecturally identical
model (same layer count, head configuration, and, for SSM layers, same
state-space dimensions) since the tensors are consumed directly by the
next forward pass with no adapter; Interlat's own "communication adapter"
(a trained self-attention plus projection layer) exists precisely because
a raw hidden-state handoff between differing models does not work without
one, and LatentMAS's linear realignment matrix `W_a` exists for the same
reason within a single model's own latent-to-input-embedding mismatch.

**(d) Sceptic's check.** Does the state transferred allow the receiving
process to continue generation bit-identically (or provably
distributionally identically) to what the sending process would have
produced had it continued itself, for both the attention KV cache and the
SSM recurrent state, verified by diffing continuations rather than by a
downstream task-accuracy proxy? None of CacheGen, CacheBlend, DistServe,
LatentMAS, or Interlat report this exact check; CacheGen/CacheBlend/
DistServe check task-quality metrics (F1, Rouge-L, SLO attainment) as a
proxy, and LatentMAS/Interlat check downstream reasoning benchmark
accuracy, not continuation identity.

## I3: one million tokens of context on a consumer GPU

**(a) Closest prior art.** Jamba, arXiv 2403.19887,
https://arxiv.org/html/2403.19887, retrieved 2026-09-15. Hybrid
Transformer-Mamba-MoE model, quoted: "Our architecture aims to provide not
only a small number of active parameters but also an 8x smaller KV cache
compared to a vanilla Transformer." Concrete number, their own Section 5.1:
at 256K context, Mixtral's KV cache needs 32GB while "Jamba's KV cache
takes only 4GB even at such a long context" (quoted), an 8x reduction
matching the architectural claim. "We have successfully trained Jamba
models with context lengths of up to 1M tokens. The released model
handles context lengths of up to 256K tokens" (quoted); this was on a
single **80GB** A100/H100, not a 24GB consumer card. Separately, KVQuant
(brief 3, arXiv 2401.18079) reports enabling "LLaMA-7B with a context
length of up to 1 million on a single A100-80GB" via 3-bit KV
quantization with under 0.1 perplexity degradation, again an 80GB
datacenter card, not consumer-class.

**(b) What neither could do that we would claim.** Neither Jamba nor
KVQuant demonstrates 1M tokens on a 24GB consumer card; both rely on
80GB of VRAM. Jamba's 8x KV-cache reduction comes from its layer ratio
(mostly Mamba, few attention layers), not from combining a hybrid ratio
with aggressive KV quantization on the remaining attention layers, which
is what a 24GB target would require; no source in this pass combines
both techniques on record.

**(c) Hard limit, KV bytes per token, the exact arithmetic the brief
asked for.** For 10 GQA attention layers (Qwythos-9B's attention-layer
count) at 4 KV heads x 256 dims per head, per token, both K and V:
`2 (K and V) x 10 layers x 4 heads x 256 dims x bytes_per_value`:

```
bf16 (2 bytes):  2 * 10 * 4 * 256 * 2 = 40,960 bytes/token  (40 KB/token)
q8   (1 byte):   2 * 10 * 4 * 256 * 1 = 20,480 bytes/token  (20 KB/token)
q4   (0.5 byte): 2 * 10 * 4 * 256 * 0.5 = 10,240 bytes/token (10 KB/token)
```

At 1,000,000 tokens: bf16 = 40,960,000,000 bytes (~38.15 GiB), q8 =
20,480,000,000 bytes (~19.07 GiB), q4 = 10,240,000,000 bytes (~9.54 GiB).
Against a 24 GB (nominally 25.77 GB, using 1024-based GiB the card
actually reports ~24 GiB = 25.77 GB) card that must also hold the model
weights (35B MoE at even 4-bit is far larger than 24GB alone) plus the SSM
layers' fixed-size recurrent state (small and constant, not scaling with
sequence length, so not part of this arithmetic): **bf16 KV alone at 1M
tokens (38 GiB) does not fit a 24GB card even with zero weights loaded;
q8 (19 GiB) is tight but plausible only if paired with a very small
resident weight footprint; q4 (9.5 GiB) is the only one of the three that
leaves meaningful headroom for weights on a 24GB card.** This computation
is mine, built directly from Qwythos-9B's stated GQA shape (4 KV heads x
256 dims, 10 attention layers) in the brief's own context line, not a
quoted external claim.

**(d) Sceptic's check.** RULER, arXiv 2404.06654,
https://arxiv.org/html/2404.06654, retrieved 2026-09-15, is exactly the
tool a sceptical reviewer would demand here. Its own headline finding
directly warns against taking a "claimed context length" at face value:
"While all models claim context size of 32k tokens or greater, our
results indicate that only half of them can effectively handle sequence
length of 32K by exceeding a qualitative threshold. Moreover, almost all
models fall below the threshold before reaching the claimed context
lengths" (quoted). Most pointed for a hybrid SSM claim specifically,
quoted directly: "we show that non-Transformer architectures, such as
RWKV and Mamba, still lag behind Transformer by large margins on Ruler."
The check: run RULER (or an equivalent needle/multi-hop/aggregation suite,
not just needle-in-a-haystack, which RULER's own Table 1 marks as too
easy a test on its own) at the full claimed length and report the
"effective length," the point past which accuracy drops below a fixed
threshold, not just "it did not crash" or "it accepted the prompt."

## I4: a 100B-class MoE on a 24 GB card by streaming experts from host RAM

**(a) Closest prior art, with numbers.**

Mixtral-Offloading, arXiv 2312.17238, https://arxiv.org/html/2312.17238,
retrieved 2026-09-15. LRU cache of `k` experts kept resident in GPU
memory (quoted: "we use k=2 for 12GB GPUs and k=4 for 16GB ones" on
Mixtral-8x7B) plus speculative prefetch that "guesses which experts are
needed ahead of time to better overlap expert loading with computation."
Result, quoted: interactive generation "at 2-3 tokens per second" on a
T4, RTX 3060, or RTX 3080 Mobile. Fiddler, arXiv 2402.07033,
https://arxiv.org/html/2402.07033, retrieved 2026-09-15, instead executes
cold (not-resident) experts **on the CPU** rather than moving their
weights over PCIe, since a small batch's activations
(`input_size x 4096` for Mixtral) are far cheaper to move than a whole
expert's weight matrices (each expert over 300MB at 16-bit). Result,
quoted, on uncompressed 16-bit Mixtral-8x7B (over 90GB of parameters):
"on average 1.26 times speed up in single batch inference, 1.30 times in
long prefill processing, and 11.57 times in beam search inference,"
tested on PCIe Gen3 x16 (32GB/s) and Gen4 x16 (64GB/s) setups (their own
Table 1). MoE-Infinity, arXiv 2401.14361,
https://arxiv.org/html/2401.14361, retrieved 2026-09-15, is the most
directly comparable to mojo-baro's own hardware class: tested on "single
NVIDIA-A5000-24GB through PCIe4.0 (24GB/s)" (quoted, their own Table 1
caption), reporting "3.1-16.7x per-token latency improvements over
numerous state-of-the-art systems, including vLLM, Ollama, DeepSpeed and
BrainStorm." Its own trace-study finding, quoted: "For MoE models with
around 100 experts (e.g., DeepSeek, QWen-MoE, NLLB, and Switch-MoE),
fewer than 5% of experts are repeatedly activated when decoding tokens
for a single request. Even for MoE models with fewer experts (e.g.,
Mixtral), we observe only 25% activation per request," which is the
working-set-locality assumption its sparsity-aware cache exploits.

**(b) What none of them could do that we would claim.** MoE-Infinity's
headline 24GB-card number is measured on **DeepSeek-V2-Lite**, a
15.7B-total/2.4B-active model, not a 100B-class model. Mixtral-Offloading
and Fiddler both target Mixtral-8x7B, which is 46.7B total parameters
(8 experts, ~13B active), also short of 100B-class. None of the three
papers checked in this pass demonstrates a genuinely 100B+-total-parameter
MoE running interactively on a single 24GB card; the closest hardware
match (MoE-Infinity's A5000-24GB) uses the smallest model of the three.
No DeepSeek-V3/R1-class (600B+) or comparably large MoE was found
benchmarked on a single 24GB consumer card in this pass; llama.cpp's own
`-ot`/`--override-tensor` expert-to-CPU placement flag was located as a
real, shipping mechanism for this class of offload, but no measured
tok/s number for it on a 100B+-class MoE was found in this round
(flagged as unanswered rather than guessed).

**(c) Hard limit.** PCIe bandwidth as actually measured by these papers,
not the marketing peak: Fiddler's own evaluation setups list "PCIe Gen3
x16 (32GB/s), Gen4 x16 (64GB/s)" as the two theoretical link speeds
tested (their Table 1), while MoE-Infinity's own single-A5000 setup
states "PCIe4.0 (24GB/s)" as the number that actually governed their
measured result, matching the brief's own "25 to 28 GB/s measured" figure
for real-world sustained Gen4 x16 traffic rather than the 64GB/s
theoretical ceiling. Per-token expert bytes at 512-wide experts (mojo-baro's
35B MoE, 256 experts, top-8): each expert's gate+up+down FFN weight
matrices, at 512-wide intermediate dimension and the model's hidden size,
multiplied by however many of the 8 selected experts are not already
resident in the GPU cache that step, is the quantity that must clear the
~24-28 GB/s sustained PCIe budget within the token's compute-bound time
window; MoE-Infinity's own finding that fewer than 5% of a ~256-expert
model's experts repeat within one request is the load-bearing assumption
that makes staying under that budget plausible at all, since without high
per-request locality, most top-8 selections would each require a fresh
fetch.

**(d) Sceptic's check.** Measure sustained tok/s on the actual target
model (not a smaller stand-in) at the actual expert count and width,
report the cache hit rate (fraction of selected experts already resident,
not fetched that step) the way MoE-Infinity's own Table 1 does, and
report bytes moved per token against the PCIe link's independently
measured sustained bandwidth (not its rated peak), the same three numbers
MoE-Infinity, Mixtral-Offloading, and Fiddler each report for their own
smaller models.

## I5: forking a conversation into parallel branches at near-zero cost

**(a) Closest prior art, with numbers.** Hydragen, arXiv 2402.05099,
https://arxiv.org/html/2402.05099, retrieved 2026-09-15. An **exact**
(not approximate) attention implementation that decomposes attention over
a shared prefix from attention over each branch's unique suffix, batching
the shared-prefix queries across all branches into one matrix-matrix
product instead of many matrix-vector products. Quoted numbers: "improve
end-to-end CodeLlama-13b throughput by up to 32x against competitive
baselines, with speedup growing with the batch size and shared prefix
length... increasing the prefix length from 1K to 16K tokens decreases
Hydragen throughput by less than 15%, while the throughput of baselines
drops by over 90%." For tree-structured (not just prefix-suffix) sharing,
applied hierarchically to competitive-programming problems: "reduce
inference time on competitive programming problems by 55%" over a
single-level split (quoted). Tree of Thoughts, arXiv 2305.10601,
https://arxiv.org/html/2305.10601, retrieved 2026-09-15, is the
prompting-level analogue (branch into a tree of intermediate "thoughts,"
evaluate each state, search with BFS or DFS, backtrack): "in Game of 24,
while GPT-4 with chain-of-thought prompting only solved 4% of tasks, our
method achieved a success rate of 74%" (quoted).

**(b) What each could not do that we would claim.** Tree of Thoughts's
own framework paper does not address the systems cost of branching at
all: each explored "thought" node is generated as an ordinary LLM call,
with no claim about avoiding recomputation of the shared prefix across
sibling branches; the near-zero-cost part of "fork a tree of thoughts
cheaply" is a systems problem ToT's own paper does not solve, only the
search-and-evaluation logic on top of whatever inference backend is used.
Hydragen solves the systems side but is a fixed prefix-suffix (or
statically hierarchical tree) decomposition computed by the caller; it
does not itself provide a live, mutable-at-any-point conversation-forking
API with snapshot/rollback the way a grammar-engine-integrated fork
primitive would (SGLang's own `fork` primitive, cited by the brief, is
the closer systems-level match for that API shape, though its own paper
body was not independently re-fetched in this round beyond what web
research 3 already established about RadixAttention's underlying radix
tree and LRU eviction).

**(c) Hard limit.** Hydragen's own throughput curve shows the technique's
benefit is bounded by how much of the sequence is actually shared: as
branches diverge and the unique-suffix portion grows relative to the
shared prefix, the matrix-matrix-product advantage over ordinary batched
matrix-vector attention shrinks correspondingly, which is exactly why its
own reported number is a range that "grows with batch size and shared
prefix length" rather than a single constant multiplier; a fork with a
very short shared prefix and long independent continuations gets little
of Hydragen's benefit. A second, orthogonal cap: identity guarantee is
exact-recompute-equivalent only insofar as the underlying decomposition
is bit-exact, which Hydragen claims for its attention math; a forked
branch's own subsequent generation is only reproducible to the extent the
sampler and any batching-order-dependent floating point associativity are
themselves deterministic, which is a separate, unaddressed concern in
Hydragen's own paper.

**(d) Sceptic's check.** Compare the cost (wall-clock and total FLOPs) of
forking N branches from a shared prefix against the cost of N fully
independent recomputes from scratch, at varying shared-prefix lengths and
varying degrees of branch divergence, and separately verify that each
branch's forked continuation is either bit-identical to what a
from-scratch recompute of that same branch alone would have produced, or,
if not bit-identical, that the paper explicitly states what weaker
guarantee (e.g. same distribution, not same sample) it is claiming
instead; Hydragen's own paper claims exactness for the attention
computation itself, which is the standard this checks against.

## I6: measurement as a product

**(a) Closest prior art, with numbers.**

"Benchmarking Crimes: An Emerging Threat in Systems Security," arXiv
1801.02381, https://arxiv.org/pdf/1801.02381, retrieved 2026-09-15.
Surveyed "50 papers" from top systems security venues (quoted, their own
methodology section) and found benchmarking-quality problems
"respectively affecting 80% and 69% of the applicable papers" for their
two most common crime categories, with "only a single paper" in the
50-paper sample committing no high-impact crime at all (quoted, their own
results discussion). This is the closest primary-source precedent for
"measurement rigor itself is the differentiator," applied to a sibling
field (systems security) rather than ML/GPU kernels specifically, but the
taxonomy of crimes (missing information needed to reproduce a number,
cherry-picked configurations, subbenchmarks silently dropped) generalizes
directly to a GPU-kernel benchmarking claim.

NEUTRINO, OSDI 2025, https://www.usenix.org/system/files/osdi25-huang-songlin.pdf,
retrieved 2026-09-15, is the primary source that most directly matches
"in-kernel timestamp instrumentation for attribution... Nsight-style vs
in-kernel stamps." NEUTRINO injects small, platform-independent "probes"
directly into GPU tracepoints at the assembly level (their own Python
"Tracing DSL," compiled down into "raw low-level assembly probes"),
enabling "instruction-level timers" and cross-probe timing by
"leveraging registers as the temporal storage between probes," in
contrast to kernel-exclusive external profilers like Nsight Compute that
only report one aggregate number per kernel launch. Their own worked
example (Table 4, FlashAttention-v2, comparing an SM shared with another
concurrent kernel against one running exclusively) found "5.85x higher
stall cycles due to L1 miss from hardware profiling" and "4.47x higher
exposed stall cycles for compute pipeline contention" when a kernel
shares its SM with another concurrent kernel versus running alone
(quoted); a kernel-level, host-synced timer that only reports one number
per launch cannot see this difference at all, since it averages away
exactly the sub-kernel, contention-dependent effect NEUTRINO's
sub-instruction probes were built to expose.

**(b) What neither could do that we would claim.** The Benchmarking
Crimes paper is a taxonomy and survey, not a tool; it tells you what
categories of mistake are common, but does not itself instrument
anything or prove a specific number is trustworthy. NEUTRINO gives
sub-kernel, instruction-level attribution but is a general-purpose GPU
profiling framework aimed at understanding arbitrary kernel runtime
behavior; nothing in the material reviewed here shows it being used to
answer the specific question mojo-baro's own finding addresses (whether a
timing delta between two arms is a genuine change from a geometry/kernel
change, versus clock-ramp or cache carry-over from the immediately
preceding kernel in the same stream). NEUTRINO's own Table 4 result is
the closest analogue found (cross-kernel SM contention changing measured
stall cycles by 4-6x), but it is framed as an SM-sharing/occupancy
finding, not explicitly as "host-synced per-kernel timing cannot judge a
geometry change" the way mojo-baro's own repo finding states it; no
primary source found in this pass makes that exact claim in those exact
terms.

**(c) Hard limit.** A host-synced (`hipEventRecord`/`cudaEventRecord` or
equivalent, wrapping a single kernel launch with a synchronization
barrier before and after) timer cannot distinguish, within its own
single number, how much of the measured interval is the kernel's own
steady-state execution versus residual effects carried over from the
immediately preceding kernel on the same stream (clock frequency not yet
settled after a preceding kernel's different power/thermal profile, or
data from the preceding kernel still resident in L2/Infinity Cache
biasing the next kernel's memory-access latency); this is a genuine
information-theoretic limit of the measurement technique itself, not a
tooling gap, since the host-synced timer only ever observes one
aggregate wall-clock delta per launch, structurally unable to separate
these components without either finer-grained in-kernel instrumentation
(NEUTRINO's approach) or a controlled experiment that varies only the
preceding kernel while holding the measured kernel fixed.

**(d) Sceptic's check.** Repeat the same timed kernel back-to-back with
different preceding kernels (an idle/cold start, the same kernel, a
kernel that touches disjoint cache lines, a kernel that touches
overlapping cache lines) and show the measured time is invariant to that
choice, or if it is not invariant, report the size of the carry-over
effect directly rather than reporting a single number as if it were
kernel-intrinsic; this is the same shape of check NEUTRINO's own Table 4
comparison performs (same kernel, exclusive-block vs shared-block
condition, reporting the ratio), and is the check a sceptical reviewer
would demand before accepting that a measured timing delta reflects a
genuine geometry change rather than an artifact of what ran immediately
before it in the same stream.
