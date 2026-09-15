# Next plan (2026-09-15): five capability gaps and six headline capabilities

Status: DRAFT. Part A is detailed. Part B is filled in after web research
brief 4 (`exchange/2026-09-15-web-research-4.md`) lands, then the whole plan
is dispatched to an opus build lane under the fable coordinator. Effort in
LOC (XS < 20, S < 60, M < 150, L < 400, XL beyond). Every item names the
check that makes it done (CLAUDE.md s18); every timed claim is preregistered
in its `bench/*-protocol.md` before the first run (P1, P4).

Ground truth today (`docs/BASELINE.md`): Qwythos-9B q4 136.37 tok/s_gen
greedy, 150.96 with MTP k=2; 32k decode 101 tok/s; qwen35moe 93.46 tok/s
(llama.cpp 109.4); prefill within 1.3x of llama.cpp at 8k to 32k; four dense
families served; sampling on device with two shapes refused pending rounds;
self-describing bakes verified at `8184f7d`; LatentOS KV/SSM handoff 27x
faster than re-prefill at 32k.


## GPU rule for every item (the maintainer, 2026-09-15): short experiments that prove

The GPU is in constant use. No lane gets GPU-days. Every claim is proven by
a preregistered experiment that holds the GPU for minutes, not hours:
- Each gate below is sized to run in under 10 minutes of GPU time
  (one-prompt receipts for direction, the 20-prompt median only for a
  landing decision, never for exploration).
- Anything longer (training, sweeps) runs as `gpu-wait run --preemptible
  --priority 10` in slices that checkpoint every few minutes, so interactive
  work at `--priority 90` preempts it and nothing is lost; it never holds the
  card outright.
- A build lane reports GPU minutes used per gate in its report; a gate that
  needs more than 10 minutes is split or redesigned before it runs.
- Reads, builds, ISA receipts, parity tests on fixtures and host oracles are
  CPU and free; do them first and often.

## Part A: the five gaps

### A1. Sampled speculation (M, sonnet; after the C3 tail round, S)

Rule (Leviathan 2211.17192, Chen 2302.01318, both apply truncation to
target and draft before the rule): accept draft token x with probability
min(1, p(x)/q(x)) on the adjusted distributions, resample from
norm(max(0, p - q)) on the first rejection, bonus token from p. Host
reference `serve/sample_ref.mojo::spec_accept_ref` exists; device
`kernels/sample.mojo::amar_spec_accept` exists. Wire both into the MTP
verify window under `temperature > 0` (today spec is forced off there).
Prerequisite: the C3 tail round (53-bit uniform for the Gumbel key on both
sides, preregistered in `bench/chat-protocol.md`), else the untruncated
distribution carries the 2^-24 floor.
Gates: (1) T=0 byte-identical on `ref-tokens-64` and the 20-prompt set;
(2) device accept == host reference per draw on fixed seeds; (3) 20000-draw
chi-square of emitted tokens against the target distribution on three real
rows at T=1 and at T=0.7/top_p 0.9, p=0.001; (4) 20-prompt tok/s at T=0.7
with spec on within 5% of the T=0 spec gain; (5) refusal for the untruncated
shape lifted in the same commit as the tail fix.
Fallback: llama.cpp's sample-and-compare (unbiased per token, lower
acceptance), S.

### A2. Paged and quantized KV (L, fable for `kernels/dattn.mojo`, sonnet for the allocator)

Step 1 paging: 16-token blocks, per-request block table (vLLM 2309.06180),
`dattn` split kernel reads through the table; prefix checkpoints and the
MTP (k+1)-slot ring keep working. Step 2 quantization on KIVI's axes
(2402.02750): keys per channel quantized before RoPE (KVQuant 2401.18079),
values per token, int8 with per-block scales first, int4 as a second arm.
Gates: forced agreement (`BARO_FORCE`) 20/20 at 8k, 16k, 32k for each arm;
64k and 128k RULER subsets within 24 GB; decode after p32768 not below
101 tok/s; prefix checkpoint restore byte-identical; the MoE profile
unaffected. This lane is also LatentOS C2 (compressed checkpoints).

### A3. Continuous batching (XL; plan first, sonnet on the scheduler, fable on the m>1 GEMV)

Orca (OSDI 2022): iteration-level scheduling and selective batching, i.e.
attention per request, linear layers batched as rows. The engine already
runs m up to 8 rows per launch and the Rust front already has a request
queue and `BARO_POOL`. Needs A2 (block tables let different lengths share a
batch). Design decisions for the plan stage: prefill chunking granularity
(Sarathi), preemption by recompute (vLLM's own ablation favours it at small
block sizes), per-request sampler state.
Gates: N=4 concurrent clients through `baro-serve`, each output identical
to its single-request run at T=0; aggregate tok/s at m=4 against the P5
row-scaling receipt (delta phase scales 1.46x at m=2, so predict, do not
assume); no regression of the single-request 20-prompt median.

### A4. Trained draft head (M code; GPU in preemptible slices, sonnet)

Acceptance is the lever only through training: DeepSeek-V3's trained MTP
head reaches 85 to 90% second-token acceptance (2412.19437) against our 42%.
E13's trainer, converter and Mojo projector are built and verified
(`~/AMDHQ`, mojo-baro `761472a` `627e8da` `ca10db1`); retarget them to
next-token prediction from the trunk's last hidden state. Expected tokens
per pass (1 - a^(K+1)) / (1 - a) divided by the window cost t(K)
(Sequoia 2402.12374) is the frozen prediction; at a=0.8, K=2: 2.44 / 1.45.
Proof before any long run: a 10-minute training smoke on 2000 sequences
must move acceptance on 5 prompts above the untrained 42% (frozen: >= 50%
or the item is killed); only then a preemptible, checkpointed run in
slices. Gates: acceptance on the 20-prompt set; 20-prompt tok/s at k=2 and
k=3; T=0 identity unchanged (the draft never changes the target's argmax).
Caveat from EAGLE Table 3: on the MoE model the win is bounded by extra
expert reads per verified token; predict it, do not promise it.

### A5. Tool calling and structured output through the grammar engine (M, sonnet)

`grammar/` has JSON-schema and regex to PDA, token-mask fill, accept /
snapshot / rollback. Wire the mask into the device sampler per step
(mask before Gumbel), expose `response_format` (json_schema) and `tools` on
`/v1/chat/completions`, pass `chat_template_kwargs` through to minijinja
(Granite 4.2: `enable_thinking`, `low_effort`; same field name as vLLM and
llama.cpp). Spec decode stays on: rollback on rejected drafts is what the
snapshot API is for.
Gates: schema-valid JSON on 50 prompts at T=0 and T=0.7; a real tool-call
round trip through the server with the call parsed and the tool result fed
back; spec on with at least one rollback exercised; T=0 identity unchanged
when no grammar is set.

### A0. Prerequisites and hygiene

- Served-path timing of the MoE model after R1 to R3 (one gpu-wait launch;
  the 93.46 is the one-shot engine).
- MoE persistent path (L, fable): a stamp timeline of one MoE token first
  (`bench/carryover-stamp.py`), then move the expert phases into the
  megakernel; research put llama.cpp's remaining edge in launch structure.
- B3 C-language IPC probe (S): decides Mojo `external_call` ABI versus
  driver before IPC handoff is called dead.

## Part B: six headline capabilities (to be detailed after research 4)

B1 a model file that runs itself and proves its numbers. B2 agents sharing
KV/SSM state instead of text (E14 demo). B3 one million tokens of context on
a consumer GPU (A2 taken all the way). B4 a 100B-class MoE on 24 GB by
streaming experts from host RAM. B5 forking a conversation into parallel
branches at near-zero cost. B6 measurement as a product. Each gets: prior
art and its number, our claim, the hard limit, the reviewer's check, effort,
model, dependencies.
