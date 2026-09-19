# Latent thoughts: exhaustive sweep (2026-09-19)

Scope per brief: every paper on models reasoning, drafting, or communicating in
hidden-state space instead of tokens. Extends
`exchange/2026-09-17-latent-communication-survey.md` (47 ids, 7 families,
narrower scope: models communicating without words). That file's ids are
listed here as KNOWN, one line each, not re-annotated; full text stays there.
Everything below with a paragraph is NEW this pass. Search tool was firecrawl
only (firecrawl_search categories:["research"], firecrawl_scrape on arXiv abs
pages and, for two non-arXiv items, on the alternate primary source). firecrawl
returned empty results on roughly half of individual queries this session
(same transient flakiness the 09-17 survey reported), resolved by retrying,
rephrasing, or scraping `arxiv.org/search/?query=...` directly.

Relevance sentence in each entry answers four things at once, in order:
(a) a drafter fed the target's hidden states over a wire, (b) verified by a
fused multi-row MoE kernel, (c) experts in VRAM or host RAM, (d) on one
consumer GPU. Not every paper touches all four; the sentence says which.

---

## Family 1: Latent / continuous reasoning

KNOWN (full entries in the 09-17 survey, family 5): 2310.02226 (pause tokens),
2311.01460 (implicit CoT via distillation), 2404.15758 (filler tokens),
2412.06769 (Coconut), 2502.03275 (Token Assorted), 2502.05171 (Huginn,
recurrent-depth test-time compute).

**2403.09629** Quiet-STaR: Language Models Can Teach Themselves to Think
Before Speaking. Eric Zelikman. 2024. Generalizes STaR: the model generates a
short rationale at every token position (not just at QA time) via a tokenwise
parallel sampling trick with learnable start/end thought tokens, and is
rewarded when the rationale improves prediction of the actual next tokens.
Relevance: this is the training recipe that would teach a drafter to produce
useful intermediate states unsupervised; irrelevant to MoE kernels or VRAM/RAM
placement.

**2505.15778** Soft Thinking: Unlocking the Reasoning Potential of LLMs in
Continuous Concept Space. Zhen Zhang. 2025. Training-free: replaces the
sampled discrete CoT token with a probability-weighted mixture of embeddings
("concept token") fed back as the next input, so multiple reasoning paths are
implicitly represented in one continuous step. Relevance: a second concrete
mechanism (after Coconut) for "hidden state as the next input instead of a
token" that a hidden-state drafter would reuse; no MoE or offload angle.

**1807.03819** Universal Transformers. Mostafa Dehghani. 2018. Predates the
"latent reasoning" framing by six years: a parallel-in-time recurrent
Transformer variant that reapplies the same block depth-adaptively per
position (adaptive computation time), shown Turing-complete under
assumptions. Relevance: the origin of "iterate the same weights instead of
emitting more tokens," which Huginn (known, 2502.05171) and looped-transformer
work descend from directly; no drafting/MoE/offload content.

---

## Family 2: Speculative decoding drafting from the target's hidden states

Entirely new to this survey; the 09-17 survey did not cover speculative
decoding at all.

**2401.15077** EAGLE: Speculative Sampling Requires Rethinking Feature
Uncertainty. Yuhui Li. 2024. Drafts by autoregressing on the target's
second-to-top-layer feature (hidden state), not on tokens, resolving the
uncertainty this creates by also feeding the draft head the next sampled
token; 2.7-3.5x speedup, distribution-preserving. Relevance: this is
literally "a drafter fed the target's hidden states," the closest published
mechanism to that exact wire; single dense model, no MoE, no offload.

**2406.16858** EAGLE-2: Faster Inference of Language Models with Dynamic
Draft Trees. Yuhui Li. 2024. Replaces EAGLE's static draft tree with a
context-aware dynamic tree built from the draft model's own (well-calibrated)
confidence scores; 3.05-4.26x, 20-40% faster than EAGLE-1. Relevance: same
hidden-state-over-a-wire channel as EAGLE, refines only the verification
tree shape; no MoE or offload content.

**2503.01840** EAGLE-3: Scaling up Inference Acceleration of Large Language
Models via Training-Time Test. Yuhui Li. 2025. Abandons feature-level
prediction (which plateaus with more data) for direct token prediction fused
from multiple target layers ("training-time test"); up to 6.5x, 1.4x over
EAGLE-2. Relevance: shows the target-hidden-state channel itself was the
bottleneck once data scaled, a finding directly load-bearing for any drafter
design that streams target features rather than a single feature vector.

**2401.10774** Medusa: Simple LLM Inference Acceleration Framework with
Multiple Decoding Heads. Tianle Cai. 2024. Adds several parallel decoding
heads on top of the frozen (Medusa-1) or jointly fine-tuned (Medusa-2)
backbone's last hidden state, verified via tree attention; 2.2-3.6x. Relevance:
draft heads read the target's hidden state locally (in-process), the
baseline every later "draft head" paper (Hydra, Clover, ReDrafter) improves
on; no MoE/offload.

**2402.05109** Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.
Zachary Ankner. 2024. Medusa's draft heads speculate independently of each
other; Hydra heads condition each subsequent head on the previously
speculated tokens, closing that gap; 1.31x over Medusa, 2.70x over
autoregressive. Relevance: same in-process hidden-state channel as Medusa,
the sequential-dependency fix a wire-based multi-head drafter would also
need; no MoE/offload.

**2405.00263** Clover: Regressive Lightweight Speculative Decoding with
Sequential Knowledge. Bin Xiao. 2024. Transmits sequential knowledge from
already-speculated tokens through a "Regressive Connection" into an Attention
Decoder, plus an Augmenting Block that re-purposes the target's hidden state
for speculation rather than next-token prediction; beats Medusa by up to 57%.
Relevance: explicit hidden-state re-purposing for the drafting objective, a
design point directly reusable for a cross-process drafter.

**2408.00264** Clover-2: Accurate Inference for Regressive Lightweight
Speculative Decoding. Bin Xiao. 2024. Clover's RNN draft head traded accuracy
for cheapness versus attention-decoder drafters; Clover-2 closes that gap via
architecture changes plus knowledge distillation, without losing the
lightweight RNN's cost advantage. Relevance: the accuracy/cost trade-off
curve for RNN-vs-attention drafters, directly relevant to sizing a drafter
that must also pay wire/serialization cost.

**2408.15766** Learning Harmonized Representations for Speculative Sampling
(HASS). Lefan Zhang. 2024. Diagnoses two mismatches in EAGLE-style drafters:
inconsistent context between training and decoding, and a training/decoding
objective mismatch; fixes both via harmonized objective distillation and
context alignment, 8-20% over EAGLE-2 at no added inference cost. Relevance:
directly cited by this repo's own `docs/BASELINE.md`-adjacent MoE work
(mojo-baro's dot-loop / prompt-lookup rounds); the train/decode-context
mismatch it fixes is exactly the failure mode a hidden-state-over-a-wire
drafter would hit first (context available at train time may not match what
crosses the wire at decode time).

**2404.16710** LayerSkip: Enabling Early Exit Inference and Self-Speculative
Decoding. Mostafa Elhoushi. 2024. Trains with layer dropout (heavier at later
layers) plus a shared early-exit loss, then drafts by exiting early and
verifying/correcting with the remaining layers of the SAME model, no
auxiliary draft model at all; 1.8-2.2x. Relevance: the hidden state never
leaves the process (self-speculation), the opposite end of the design space
from a wire-fed drafter, useful as the zero-transport-cost baseline; no MoE.

**2404.18911** Kangaroo: Lossless Self-Speculative Decoding via Double Early
Exiting. Fangcheng Liu. 2024. Also self-speculative (fixed shallow
sub-network drafts, remaining layers verify), but adds a lightweight adapter
bridging the sub-network to the full model's representation space, plus a
second early-exit inside the draft phase itself when confidence drops; 1.68x
with 88.7% fewer added parameters than Medusa-1. Relevance: the adapter is a
small trained bridge between a partial hidden state and the target's full
representation, structurally similar to any cross-process hidden-state
translator a wire-based drafter would need.

**2403.09919** Recurrent Drafter for Fast Speculative Decoding in Large
Language Models (ReDrafter). Yunfei Cheng. 2024. An RNN draft head
conditioned directly on the target LLM's hidden states, with dynamic tree
attention pruning duplicate beam prefixes and knowledge distillation from the
target; 2.8x on H100, 2.3x on Apple Silicon Metal (on-device). Relevance:
explicitly "RNN conditioned on the LLM's hidden states," and the only paper
in this family with a shipped on-device (consumer-hardware-class) result,
directly relevant to a single-consumer-GPU drafter.

**2412.19437** DeepSeek-V3 Technical Report. DeepSeek-AI. 2024. The 671B/37B-
active MoE model that introduced multi-token prediction (MTP) as a pretraining
objective: extra sequential MTP modules predict tokens 2..k+1 ahead from
shared hidden states, usable at inference for self-speculative decoding.
Relevance: MTP heads read the model's OWN hidden states (in-process, like
LayerSkip) but on an MoE model specifically, the direct link between families
2 and 3 in this brief; the MTP head's cost scales with active experts, not
total experts, which matters for the "verified by a fused multi-row MoE
kernel" question.

Qwen's MTP: no dedicated arXiv paper found distinct from DeepSeek-V3's; Qwen3
and Qwen3.5/Next MTP support is documented in serving-framework blog posts
(vLLM/SGLang) and the model cards, not a standalone paper, so it is not
listed as its own entry (checked, not invented).

KTransformers ("Unleashing the Full Potential of CPU/GPU Hybrid Inference for
MoE Models") is listed under family 3 below, not here, and carries no arXiv
id: three separate arXiv title searches (plain, quoted, "KTransformers
Unleashing") returned zero hits; only a paywalled ACM DL entry
(10.1145/3731569.3764843) and the project's own GitHub/site describe it.
Flagged, not invented.

---

## Family 3: Speculative decoding for MoE and offloaded-weight models

Entirely new to this survey.

**2406.02532** SpecExec: Massively Parallel Speculative Decoding for
Interactive LLM Inference on Consumer Devices. Ruslan Svirschevski. 2024. Not
MoE-specific, but the foundational "speculate to hide offload latency" paper:
builds a wide "cache tree" of the draft model's most probable continuations,
validates it in one target pass, exploiting the fact that an offloaded target
model can score hundreds of tokens almost as cheaply as one. Relevance:
50B+ dense models on one consumer GPU with RAM offload at 4-6 tok/s; the
mechanism every MoE-offload-plus-speculation paper below borrows.

**2402.07033** Fiddler: CPU-GPU Orchestration for Fast Inference of
Mixture-of-Experts Models. Keisuke Kamahori. 2024. Not speculative decoding;
keeps attention/dense weights on GPU and expert FFNs on CPU, picking per-token
whether to compute an expert on CPU or move it to GPU based on a cost model;
1.26-11.57x over baselines depending on scenario. Relevance: experts live in
host RAM by design, the direct precursor to every "predict which experts,
then prefetch" paper below; no speculative drafting.

**2401.14361** MoE-Infinity: Efficient MoE Inference on Personal Machines
with Sparsity-Aware Expert Cache. Leyang Xue. 2024 (rev. 2025). At batch size
1 (personal-machine setting), traces which experts a request actually
activates and uses that trace, not a generic LRU policy, to drive expert
cache replacement and prefetch; 3.1-16.7x per-token latency over vLLM/Ollama/
DeepSpeed/BrainStorm. Relevance: purpose-built for "experts in host RAM, one
consumer machine," the exact deployment target; no speculative decoding
component, a cache policy instead.

**2308.12066** Pre-gated MoE: An Algorithm-System Co-Design for Fast and
Scalable Mixture-of-Expert Inference. Ranggi Hwang. 2023 (rev. 2024). Changes
the MoE algorithm itself (a "pre-gating" function computed one layer ahead)
so which experts will activate is known before they are needed, eliminating
the CPU-to-GPU expert-migration latency that plain offloading pays on the
critical path; runs large MoE LLMs on a single GPU. Relevance: the
algorithm-level answer to "predict experts early enough to hide the fetch,"
the idea every prefetch-based paper below (HybriMoE, ExpertFlow, MoE-SpeQ)
re-derives via prediction instead of an architecture change.

**2504.05897** HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for
Efficient MoE Inference. Shuzhang Zhong. 2025. Built explicitly on top of
KTransformers: dynamic intra-layer CPU/GPU load balancing, impact-driven
inter-layer prefetching, and a score-based expert cache to handle unstable
(non-fixed) expert activation patterns; 1.33x prefill / 1.70x decode over the
prior state of the art. Relevance: experts split VRAM/host-RAM on one machine,
no speculative decoding; the scheduler a drafter-plus-offload system would
sit on top of.

**2410.17954** ExpertFlow: Efficient Mixture-of-Experts Inference via
Predictive Expert Caching and Token Scheduling. Xin He. 2024 (rev. 2026).
Trains a lightweight transformer to predict the routing path for ALL MoE
layers in one forward pass, groups tokens by predicted route to raise expert
utilization, then runs a predictive cache that corrects mispredictions at
runtime; up to 93.7% GPU memory reduction, 10x throughput over strong
offloading baselines on a single GPU. Relevance: closest non-speculative
paper to "predict which experts a future token needs," on one consumer-class
GPU; the predictor here is a small auxiliary model, not the target's own
draft-and-verify loop.

**2604.10152** SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via
Self-Assisted Speculative Decoding. Jehyeon Bang. 2026. CPU-offloaded MoE
inference where the draft model is a small subset of the TARGET's own hot
experts (no separate trained draft model, no fine-tuning), used exactly like
LayerSkip/Kangaroo self-speculation but for the expert-offload setting;
4.30x throughput, reduced memory/interconnect bandwidth. Relevance: the
missing link this brief asked about directly: self-speculation (family 2)
applied to expert-offloaded MoE (family 3) on constrained hardware.

**2508.21706** Accelerating Mixture-of-Experts Inference by Hiding Offloading
Latency with Speculative Decoding (SpecMoEOff). Zhibin Wang. 2025. Uses
speculative decoding purely to enlarge the batch each expert processes per
step (more draft tokens verified at once = better hardware utilization per
expert fetch), with a dedicated CPU chunked-attention verification kernel and
an auto-tuning optimizer; up to 2.5x decode throughput over SOTA MoE
offloading. Relevance: speculative decoding as a batching trick to amortize
the exact expert-fetch cost this repo's MoE kernel rounds are chasing; strong
match to "verified by a fused multi-row MoE kernel."

**2511.14102** MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert
Prefetching and Offloading for Mixture-of-Experts. Wenfeng Wang. 2025. A
small on-device draft model predicts which experts future tokens will need;
an orchestrator prefetches those experts from host memory while the draft
model computes, hiding PCIe latency behind useful work, tuned per-hardware by
an "Amortization Roofline Model"; up to 2.34x over SOTA offloading on
Phi-MoE. Relevance: the fullest realization of "drafter predicts, host RAM
prefetches, consumer hardware" in one system; closest single paper to the
brief's exact combination apart from row-scaling kernel verification.

**2607.12696** Less Experts, Faster Decoding: Cost-Aware Speculative Decoding
for Mixture-of-Experts (EcoSpec). Jincheng Xie. 2026. Identifies "expert
scattering": confidence-driven draft-tree selection can route high-probability
draft tokens to disjoint experts, inflating expert-weight memory traffic even
though acceptance likelihood looks fine; EcoSpec's draft selection factors in
predicted marginal expert-activation cost and reuses already-loaded experts;
up to 1.62x on DeepSeek-V3.1/Qwen3-235B-A22B/GPT-OSS-120B. Relevance: names
and fixes the exact failure mode ("expert-overlap" of the verify window) this
repo's own `bench/expert-overlap` trace work (git log: "expert-overlap trace
for MoE verify-window union size") is measuring; the most directly relevant
paper in the whole sweep to that specific instrument.

---

## Family 4: Model-to-model latent communication

KNOWN (full entries in the 09-17 survey, families 1 and 4, its closest-prior
list already names Cache-to-Cache as nearest): 1704.06960 (Translating
Neuralese), 2110.07904 (SPoT), 2310.06272 (CIPHER), 2506.16196 (POST),
2510.03215 (Cache-to-Cache), 2106.07682 (model stitching), 1411.5908 (Lenc &
Vedaldi, stitching origin), 2209.15162 (LiMBeR), 2209.15430 (relative
representations), 2210.01738 (ASIF), 2311.00664 (latent space translation),
2405.07987 (Platonic Representation Hypothesis), 2505.12540 (vec2vec). Also
KNOWN, tangential (emergent multi-agent communication, not this brief's
focus): the full 1605.06676-2501.00226 cluster in that survey's families 2-3.

No new arXiv-indexed paper surfaced this pass beyond the 09-17 list under
this family; several searches for "cross-model KV cache transfer" and "model
grafting" repeated the 09-17 survey's own conclusion (empty/no results,
resolved by direct-id guesses that hit nothing new).

---

## Family 5: Latent state transfer and reuse in serving

KNOWN (full entries in the 09-17 survey, family 6): 2309.06180 (vLLM /
PagedAttention), 2311.04934 (Prompt Cache), 2310.07240 (CacheGen), 2311.18677
(Splitwise), 2312.07104 (SGLang / RadixAttention), 2401.09670 (DistServe),
2405.16444 (CacheBlend), 2407.00079 (Mooncake).

**2603.15530** DUET: Disaggregated Hybrid Mamba-Transformer LLMs with
Prefill and Decode-Specific Packages. Alish Kanani. 2026. Not a software
serving system but an accelerator architecture: splits prefill (systolic-
array chiplets, off-package memory, good for long-sequence SSM scans) and
decode (vector-unit arrays, high-bandwidth in-package memory, good for
token-by-token SSM recurrence) into disaggregated hardware packages, both
runtime-configurable for mixed Mamba/attention layers; 4x TTFT, 1.4x
throughput, 1.5x lower TBT versus B200. Relevance: this is the SSM/hybrid
state-checkpointing gap the brief named explicitly (family 5); it is a
hardware proposal, not a deployed serving stack, so it doesn't answer how an
SSM recurrent state gets checkpointed and handed across a live prefill-decode
split today (contrast with the software-only vLLM/SGLang hybrid-SSM-disagg
blog posts found alongside it, which describe shipped code but are not
papers).

---

## Family 6: Oversight and interpretability of latent channels

KNOWN (full entries in the 09-17 survey, family 7): 2305.04388 (unfaithful
CoT), 2307.13702 (measuring CoT faithfulness), 2310.18512 (preventing hidden
reasoning / encoded-reasoning steganography), 2402.07510 (secret collusion),
2505.03439 (steganographic potentials of LMs). Also KNOWN, UNVERIFIED (no
arXiv id found in the 09-17 pass or this one): "Language Models can Learn
High-Capacity Secure Steganography" (OpenReview id CjxxRknUd1 only).

**2303.08112** Eliciting Latent Predictions from Transformers with the Tuned
Lens. Nora Belrose. 2023 (rev. 2025). Trains a per-layer affine probe that
decodes any intermediate hidden state into a distribution over the
vocabulary, a learned refinement of the earlier "logit lens" trick (raw
hidden state through the final unembedding, no training); more predictive,
reliable, and unbiased than logit lens, and the latent-prediction trajectory
detects malicious inputs. Relevance: the direct oversight tool for exactly
the channel this brief's families 1-3 propose sending across a wire or into
a fused kernel: if hidden states carry the "thought," the tuned lens is how
an auditor reads them without trusting the model's own decoded text.

Logit lens itself (nostalgebraist, "interpreting GPT: the logit lens," LessWrong,
2020) has no arXiv id; it is a blog post, not a paper, and is cited by name
inside the Tuned Lens abstract above rather than independently verified here.
Flagged, not invented.

**2401.06102** Patchscopes: A Unifying Framework for Inspecting Hidden
Representations of Language Models. Asma Ghandeharioun. 2024. Generalizes
logit-lens-style probing: "patches" a hidden representation into a DIFFERENT
prompt/position (even a different, more capable model) and lets that
target's own generation explain the patched representation in natural
language; subsumes prior projection- and intervention-based inspection
methods as special cases, and supports one model explaining another's
internal state. Relevance: closest published oversight mechanism to
"model-to-model latent communication" (family 4) crossed with oversight
(family 6) at once, since a Patchscope literally injects one model's hidden
state into another model to have it explained; no MoE, no kernel, no wire
protocol, purely an inspection tool.

---

## Closest prior work to the exact combination

No paper found, known or new, combines all four: (1) a drafter fed the
target's hidden states over an actual wire between processes, (2) verified
against a fused multi-row MoE kernel, (3) with experts split across VRAM and
host RAM, (4) on one consumer GPU. The nearest single papers, each missing a
different piece:

- **MoE-SpeQ (2511.14102)** has (3) and (4) fully, and a real drafter, but the
  drafter predicts EXPERT IDENTITY for prefetch, not hidden states for
  generation; there is no wire, the draft model is on the same device.
- **EcoSpec (2607.12696)** has (2) in spirit (draft-tree selection is
  cost-aware of the MoE verify window's expert union, this repo's own
  `expert-overlap` instrument measures the same union) and speculative
  decoding proper, but targets datacenter-scale MoE (DeepSeek-V3.1,
  Qwen3-235B, GPT-OSS-120B), not one consumer GPU with host-RAM experts.
- **ReDrafter (2403.09919)** has the literal "drafter conditioned on the
  target's hidden states" plus a real on-device (Apple Silicon) result, but
  the target is a dense model, no MoE, no expert offload at all.
- **SpecMoEOff (2508.21706)** has MoE offload plus speculative decoding
  batching a fused CPU verification kernel around, closest on the systems
  side, but the draft signal is standard token-level speculation, not
  target hidden states crossing a wire, and its evaluated hardware is
  datacenter-class (roofline-tuned, not framed as consumer GPU).

## What nobody has done

1. **No paper drafts from a target MoE model's hidden states specifically
   for the purpose of predicting the NEXT TOKEN, streamed to a separate
   process, while a fused multi-row kernel handles the MoE verify batch.**
   Every MoE-plus-speculation paper found (SpecMoE, SpecMoEOff, MoE-SpeQ,
   EcoSpec) keeps the draft-and-verify loop inside one process on one
   machine; every hidden-state-drafter paper found (EAGLE family, Clover,
   ReDrafter, HASS) targets a dense model with no offloading. The
   cross-product, an EAGLE-style feature-level draft channel specifically
   engineered to hide MoE expert-fetch latency across a process boundary,
   does not appear to exist yet in what firecrawl surfaced.
2. **No published cost-aware draft-tree selection ties directly to a named,
   reusable "verify-window expert union" metric the way this repo's own
   in-flight `bench/expert-overlap` trace does.** EcoSpec is the closest
   (same underlying quantity, "expert scattering" vs. this repo's union
   size), independently arrived at in the same window (EcoSpec: 14 Jul 2026;
   this repo's trace commit: aa356c2, same week per git log) rather than
   built on each other; worth reading EcoSpec's cost model against the
   local measurement before the next MoE speculation round.
3. **No oversight mechanism (Tuned Lens, Patchscopes, or otherwise) has been
   applied to an MoE expert-offload or speculative-decoding channel.**
   Both interpretability papers found here target dense-model hidden states;
   neither discusses what changes when the "hidden state" in question only
   exists transiently during an expert fetch, or when it is deliberately
   sent across a process boundary for drafting, which is exactly the
   auditability gap the 09-17 survey flagged for KV-cache/embedding channels
   generally (its finding #1) and remains open here for the MoE-specific
   case too.
