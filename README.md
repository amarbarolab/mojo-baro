# mojo-baro

An LLM inference engine for AMD RDNA3 GPUs, with every GPU kernel written in
[Mojo](https://www.modular.com/mojo). It serves chat over an OpenAI-compatible
HTTP API on a single RX 7900 XTX. On matched weights it decodes ahead of
llama.cpp on the dense 9B (Q4_0) and level with it on the 35B MoE; on q8 and
on packs built from Q4_K_M it is still behind.

The point of the project is the kernels. Consumer RDNA3 cards are where most
people actually have 24 GB of VRAM, and they are not what the vendor libraries
are tuned for. This repo measures how much of that gap is real, one
preregistered round at a time.

**Have an AMD GPU that isn't a 7900 XTX?** Your card is the most useful thing
you can contribute. See [docs/amd-family.md](docs/amd-family.md): fifteen
minutes, no model weights needed.

## What it can do

Every capability, its status word and the switch that turns it on. The prose,
the evidence and the caveats are in [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md),
which wins on any disagreement. Status words mean exactly one thing each:
**WORKS** (a named check passed), **PARTIAL** (works inside a stated limit),
**DEFAULT-OFF** (implemented, not on), **PRESENT BUT UNUSED** (nothing calls
it), **CLAIMED** (no check behind it), **FAILS** / **PARKED** / **KILLED**.

Failed and default-off entries are here on purpose. Several of them measured at
or below their own preregistered kill lines, and that is as much a part of what
this repo is as the passing ones.

### Engine and decode

| capability | status | switch |
|---|---|---|
| Dense `qwen35` engine (hybrid SSM + attention, MTP head) | WORKS | `serve/engine.mojo`, `-D BARO_MODEL=qwen35` |
| MoE `qwen35moe` engine (256 experts, top 8) | WORKS | `-D BARO_MODEL=qwen35moe` |
| Spark engine, profile-driven dense families | WORKS | `serve/spark.mojo` + generated `profile.mojo` |
| Speculative decode / MTP draft head, dense | WORKS | `BARO_SPEC=1` (default), `BARO_SPEC_K` |
| Speculative decode on MoE | PRESENT BUT UNUSED | silently inert, no draft head |
| Megakernel, one launch per token, dense | WORKS, default on | `BARO_MEGA` |
| Megakernel on MoE | DEFAULT-OFF | `BARO_MEGA=1`, measured 1.027x under a +5% kill line |
| Megakernel over the speculative window | DEFAULT-OFF | `BARO_MEGA_WIN=1`, 0.761x |
| Chunked SSM delta scan | WORKS | part of the dense q4 m=1 kernel |
| rmsnorm fold into the consuming GEMV | WORKS | always on, 65 fewer barriers per token |
| Batched MoE prefill, bit-identical to replay | WORKS, DEFAULT-OFF | `BARO_PREFILL=1` |
| MoE prefill expert streaming, two staging slots | WORKS, default on in tier mode | `BARO_PF_OVERLAP` |
| WMMA attention in MoE prefill | DEFAULT-OFF, FAILS identity | `BARO_PF_ATT=wmma`, 97.92% vs a 99% bar |
| Dense chunk SSM scan in MoE prefill | DEFAULT-OFF, FAILS identity | `BARO_PF_SSM=chunk`, flips a greedy token |
| Expert tier, routed experts in host RAM | WORKS | `BARO_TIER=64 BARO_TIER_PINNED=1` |
| Zero-copy expert misses read from pinned host memory | DEFAULT-OFF | `BARO_TIER_ZC=1`, 1.064x, identity 20/20 |
| Resident MoE pack on 24 GB | PARTIAL | short context only; tier mode past 1k |
| MoE prompt-lookup (ngram) speculation | DEFAULT-OFF, slower | `BARO_NGRAM=1`, 0.76x |
| int8-dot FFN path | DEFAULT-OFF | `BARO_DOT=1`, q8 packs only, closed as no signal |
| R6.3 dense-addressed q8 dot for MoE projections | PRESENT, UNMEASURED | in the launch path, A/B has no numbers |
| Device sampler: temperature, top-p/top-k, penalties, logprobs, grammar mask | WORKS | request fields |

### Quantization and memory

| capability | status | switch |
|---|---|---|
| q4 weights (block-32 int4 + fp16 scale) | WORKS, default | `tools/engine-pack.py --q4` |
| q8 weights, bit-equal to `llama-quantize Q8_0` | WORKS | `--q8`, checked by `tools/q8-check.py` |
| bf16 weights | REMOVED | numbers kept in `bench/q8-protocol.md` |
| f16 weights | never existed | f16 is only a block-scale dtype |
| MoE native K-quant experts (Q4_K / Q8_0 / Q6_K) | WORKS | dedicated `kernels/moe.mojo` kernels |
| int8 MMQ prefill, dequant in kernel | PARTIAL | parity-tested, not the default dispatch |
| int8 WMMA for GEMM speed | KILLED | issues at the same rate as bf16 on gfx1100 |
| Ternary Q2_B3 / TQ1_0 / TQ2_0 | PRESENT BUT UNUSED | kernels and wrappers exist, no caller |
| KV cache f32 | WORKS, default | `BARO_KVQ=f32` |
| KV cache int8, dense only | DEFAULT-OFF | `-D BARO_KVQ=int8`, 1.171x after 32k |
| KV cache bf16 | PRESENT BUT UNUSED | fails the 20-prompt identity gate |
| Paged KV cache, 128-token pages | WORKS | `serve/kvpage.mojo` |
| Long context 32k / 128k | PARTIAL | needs `BARO_TMAX` raised; default is 1088 |
| Split-K decode attention (dattn) | PARTIAL | above `BARO_ATT_SPLIT_T`, or `BARO_ATT_SPLIT=1` |
| WMMA prefill attention (long-context lane) | PARTIAL | registered, that lane not merged |

### Serving APIs

| capability | status | endpoint |
|---|---|---|
| OpenAI completions and chat, SSE streaming | WORKS | `POST /v1/completions`, `/v1/chat/completions` |
| Chat templates (minja), file overrides | WORKS | `BARO_CHAT_TEMPLATE` |
| Tool calls, including streaming name/argument deltas | WORKS | OpenAI shape, from the `<tool_call>` grammar |
| JSON-schema `response_format` enforcement | WORKS | dense and MoE |
| JSON-schema enforcement on Spark profiles | IMPLEMENTED, live gate UNVERIFIED | same field |
| Logprobs and top-logprobs | WORKS | `logprobs`, `top_logprobs` |
| Ollama API (`/api/tags`, `/api/chat`, `/api/generate`, ...) | WORKS | real Ollama clients adopt it |
| Ollama `repeat_penalty`, `num_ctx`, `images`, `tool_calls` | PARTIAL | parsed so clients do not 400, not wired |
| Ollama `POST /api/pull` | WORKS for the loaded pack | 404 otherwise; no import or hot-swap |
| Embeddings, last-token hidden, L2-normalized | WORKS | `/v1/embeddings`, `/api/embeddings` |
| Embeddings on Spark or `BARO_SEQS > 1` | refuses 501, path UNVERIFIED | |
| Raw hidden states per token | WORKS | `hidden: true` |
| Raw pre-penalty top-k logits | WORKS, K <= 20 | `logits_topk: K` |
| Speech in (whisper sidecar) | WORKS | `POST /v1/audio/transcriptions` |
| Speech out (Piper by default) | WORKS | `POST /v1/audio/speech`, `BARO_TTS_BIN` |
| Request bodies up to 64 MiB | WORKS | |
| 62 `BARO_*` settings with a checked template | WORKS | `tools/settings-check.py` |
| Named profiles | WORKS as wiring | `tools/baro --profile NAME`, `profiles/*.toml` |

### State, checkpoints and forking

| capability | status | endpoint |
|---|---|---|
| Named state checkpoints as first-class objects | WORKS | `POST/GET /v1/checkpoints`, `GET/DELETE /{id}` |
| Prefix checkpoint restore, suffix-only prefill | WORKS | `test_prefix`, byte-exact |
| Single-node prompt forking | WORKS | `POST /v1/fork` |
| State export and import (LAT1 `.baro` files) | WORKS on one node | `/v1/state`, `/v1/state/export`, `/import` |
| State export while KV is int8 | refuses 501 | unimplemented case |
| State HMAC signing and rejection | WORKS | `BARO_STATE_HMAC_KEY` |
| Cross-node fork (`target: HOST:PORT`) | FAILS its identity gate | code exists, ids cannot certify a state |
| llama.cpp state bridge | UNCLASSIFIED | 16/13/14 of 20 vs a control that reaches 15/16/16 |
| Fan-out (one prefill, N followers) | designed, no route | `docs/P1-STATE-API.md` |
| LatentOS memfd + SCM_RIGHTS handle handoff | WORKS | `BARO_LATENT_SOCK` |
| LatentOS HIP IPC, GPU to GPU | KILLED | Mojo FFI struct ABI, not a driver problem |
| `latentos-agent --daemon` | WORKS | eviction and watchdog loop |
| LatentOS cross-host TCP transport | untested | |

### Routing and multi-engine

| capability | status | switch |
|---|---|---|
| Engine pool inside one server, shortest queue | WORKS | `BARO_POOL=N` |
| Per-engine device pinning | WORKS as wiring | `--engine-env KEY=VALUE` |
| One-command multi-engine launch | WORKS as CLI wiring | `tools/baro multi-serve --devices 0,1` |
| `baro-router`: rank and place across engines | WORKS, PARTIAL limit | two 9B engines fit only at `BARO_TMAX=4096` |
| Router state locality bias | WORKS | raw-prompt routes only, not `messages` |
| Router failover before the first response byte | WORKS | past that it fails loudly |
| Router prefix affinity, rendezvous hashing, cold-miss locking | WORKS | |
| mDNS discovery, PAIR interop | WORKS | needs a firewalld mdns rule that does not survive reboot |
| Two GPUs in production | FAILS | P4 gate: iGPU diverged after 6 tokens |
| Zero-downtime process handoff | absent | |

### Clients

| capability | status | notes |
|---|---|---|
| Android app as a remote chat client | WORKS | 80 requests byte-identical to curl |
| Android on-device llama.cpp mode | FAILS its gate | 4/5 exact, near-tie logprob divergence |
| Android voice input end to end | UNVERIFIED | only unit-tested against a mock |
| PWA client served by the server itself | WORKS | offline shell, service worker, 400px clean |
| PWA browser voice capture into transcription | WORKS | real whisper sidecar |
| ComfyUI node (chat and JSON-schema) | WORKS | time-sliced, engine stops around each call |
| aarch64: router and CPU tooling | WORKS | no decode engine there |

### Models and pipeline

| capability | status | notes |
|---|---|---|
| Qwythos-9B champion (dense) | both PPL and task PASS | the reference model of the repo |
| RegesCore-35B (MoE) | both PPL and task PASS | weakest agreement basis, 83.1% mean |
| Llama-3.2-1B, lily-7B, Qwen2.5-7B, Qwen2.5-Coder-7B | task PASS, PPL BLOCKED | spark family has no `top_logprobs` wiring |
| Granite-4.2-3B | VOID | script edited mid-sweep, needs a clean rerun |
| Ornith-1.5-9B, Qwythos-v2, Spark-X2.5-4B | not run this round | |
| GGUF to runtime pack | WORKS | `tools/engine-pack.py`, `tools/spark-pack.py` |
| Dense profile generation from a GGUF | WORKS | `tools/gen-profile.mojo` |
| Self-describing bakes (sources + reference tokens in the gguf) | PROVEN | `tools/bake.sh`, refuses a dirty tree |
| Bake closure rebuild and verify | WORKS | `tools/gguf-closure.sh`, `gguf-verify.sh` |
| Verified-receipt ledger | WORKS | only `tools/baro verify --append` writes it |
| Self-contained run from a bake alone | WORKS | `tools/baro run` (self-reported, never a receipt) |
| Quality eval CLI | WORKS | `tools/baro eval`, `eval-all`, `--dry-run` |
| `model-import` script | does not exist | use the documented chain |
| `refcache` from PATH | BROKEN for this repo | call `~/iTools/dev/refcache` by path |

### Training

| capability | status | notes |
|---|---|---|
| LoRA training and GGUF write-back | PASS on an amended bar | rank-16 on `blk.24..31` ffn_down |
| Self-distilled draft head | PARKED, no signal | +2.30 pp against a required +4 |

### Verification

| capability | status | what it is |
|---|---|---|
| `bench/preflight.sh` | WORKS | CPU stamp every gate checks before the GPU |
| `./run-tests.sh` | WORKS | 8 Mojo test binaries + kernel census, needs the card |
| `tools/ci-checks.sh` | WORKS | 12 CPU checks, about 2.5 minutes |
| Kernel census, no orphaned kernel | WORKS | `tools/kernel-census.mojo --check` |
| Teacher-forced agreement | the standing identity gate | `BARO_FORCE`, valid past 256 ids |
| Greedy 64-token equality | never valid past ~256 ids | llama.cpp fails its own reference |
| 20-prompt identity | the basis of any performance claim | `bench/mtp-prompts.sh` |
| Byte checks for saved state | required | ids kept 2 of 5 deliberately wrong states |
| Preregistration, P1 to P20 | binds every protocol in `bench/` | `bench/PROTOCOL-RULES.md` |
| Cold-cache rotation | required above 96 MB working sets | `bench/coldcache-protocol.md` |

## What runs today

| model | architecture | status |
|---|---|---|
| Qwythos-9B | `qwen35` (hybrid SSM + attention, MTP head) | main engine: chat server, speculative decode, prefix checkpoints, RULER at 32k |
| Ornith-1.5-9B | `qwen35` | same engine, packed from a Q4_K GGUF; 98.4% teacher-forced agreement with llama.cpp |
| Qwythos-9B-v2-MTP | `qwen35` | same engine, packed from a Q6_K GGUF; 99.2% median teacher-forced agreement with llama.cpp (`bench/qwythos-v2-protocol.md`, 2026-09-16) |
| Spark-X2.5-4B | `spark2_5` (gated sliding-window attention) | its own engine, `serve/spark.mojo` |
| RegesCore-35B | `qwen35moe` (256 experts, top 8) | decodes and serves; 53.20/64 mean teacher-forced agreement with llama.cpp over 20 prompts, above the dense path's own 51.90 on a quant-matched arm. Decode **111.89 tok/s_gen** (20-prompt median, 727 launches per token, `bench/moe-persist-protocol.md`, 2026-09-15) against llama.cpp's 109.92 on the same GGUF, up from 42.88 at the start of that day. The two numbers come from different stints; in llama.cpp's own stint the previous champion read 0.973x. An experimental expert tier runs it with the routed experts in host RAM (2.68 GB of VRAM instead of 21 GB, output identical on 20/20 prompts) at 39 tok/s, bound by host round trips (`exchange/lane-B4-stage2b-report.md`). |

### Dense families on the Spark path

`serve/spark.mojo` is profile-driven: `tools/gen-profile.mojo` reads a GGUF's
recipe (dims, rope type and base, norm eps, QKV bias, activation, SWA window,
scale multipliers, tied embeddings) into a comptime module the engine imports,
so a new dense model is a generated profile rather than a new engine. Four
targets are verified end to end against llama.cpp on the same GGUF
(`bench/dense-protocol.md`, preregistered; 20 prompts, teacher-forced):

| model | arch | forced agreement | tok/s_gen |
|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | `llama` | 20/20 prompts, 95.3-100% | 451-458 |
| Qwen2.5-7B-Instruct Q4_K_M | `qwen2` | 20/20 prompts, 96.9-100% | 96.8-97.8 |
| granite-4.2-3b BF16 | `granite` | 20/20 prompts, 98.4-100% | 164.5-169.0 |
| lily-cybersecurity-7b Q6_K | `llama` (Mistral-arch, SPM) | 20/20 prompts, 96.9-100% | 94.1-95.0 |

**All four are now servable.** `serve/spark.mojo` speaks the same
request/response protocol as `serve/engine.mojo` (`serve/PROTOCOL.md`), reusing
its byte-scanner JSON reader and request parser (`serve/serve_proto.mojo`)
rather than defining a second wire format. `baro-serve --engine
.work/<target>/spark-engine --pack .work/<target>` reaches any of the four
through the same HTTP surface as `qwen35`/`qwen35moe`. Verified end to end for
two targets so far: a real `POST /v1/chat/completions` against a running
`baro-serve` returned a correct completion for Qwen2.5-7B-Instruct
("The capital of France is Paris.") and granite-4.2-3b
("<think></think>Paris."), both with `finish_reason: "stop"` (spark now acts
on the `stop` field it previously only parsed). Llama-3.2-1B and
lily-cybersecurity-7b are wired through the identical code path but not yet
verified through the HTTP front this round (no `tokenizer.json` fetched for
them; Llama-3.2 is gated on HF and needs an accepted-license token).

Two gaps the four checkpoints did not exercise: Granite's embedding/residual
scale folding at pack time (both multipliers are 1.0 in the only Granite
checkpoint available), while sampling is now covered: at `temperature > 0` all five Spark-path models
draw with the device sampler, with T=0 output unchanged
(`exchange/lane-SAMPLE-report.md` item 2).

Model weights are not distributed with this repo.

## Results

### Decode speed against llama.cpp

Same box, same GGUF, 20-prompt medians unless noted
([`bench/q8-protocol.md`](bench/q8-protocol.md),
[`bench/ornith-protocol.md`](bench/ornith-protocol.md),
[`bench/q4-protocol.md`](bench/q4-protocol.md),
[`bench/attn-latency-protocol.md`](bench/attn-latency-protocol.md)).

| model, weights | llama.cpp tok/s | mojo-baro tok/s | ratio |
|---|---|---|---|
| **Qwythos-9B, Q4_0 both sides (`llama-quantize --pure Q4_0`)** | **110.0** | **130.7** | **1.19x** |
| Qwythos-9B, q8 (5-token race prompt) | 74.1 | 68.8 | 0.94x |
| Ornith-1.5-9B, from Q4_K_M | 88.8 | 80.8 | 0.91x |

The arms disagree because they are different arms, not different days. On
matched Q4_0 weights the megakernel decode path is ahead; on q8 and on a pack
built from Q4_K_M it is behind. Our Q4_0 side has since moved to 136.1 to 136.8
tok/s on the same twenty prompts (`bench/attn-protocol.md` Round A/C A/B), so
the current ratio is higher than 1.19x, but the llama.cpp bar has not been
re-measured since, and a ratio is only worth quoting when both sides were
measured in the same stint.

Sampling runs in the decode loop and composes with speculation
([`bench/spec-sample-protocol.md`](bench/spec-sample-protocol.md), dense q4,
k=2, one stint): T=0.7 top_p 0.9 decodes at 147.15 tok/s_gen with speculation
and 109.19 without, against 134.97 greedy without. The sampler itself costs
about 19% of decode at this 248,320-token vocabulary.

Speculative decode with the model's own MTP head is on by default
(`BARO_SPEC=0` turns it off) and output-identical to plain greedy decode on
every prompt tested. On the q4 pack it gives 1.1042x at k=2 as a 20-prompt
median, 150.96 against 136.72 tok/s_gen
([`bench/mtp-protocol.md`](bench/mtp-protocol.md)). A single prompt read only
1.012x, which is why that number is a median and not one run.
Prefill was the known weak spot and is now within about 1.3x of llama.cpp at 8k
to 32k context, down from 2.5 to 3.3x: 3.29 s at 8k against its 2.56 s (1.29x),
7.17 s at 16k against 5.60 s (1.28x), 17.42 s at 32k against 13.18 s (1.32x).
Against our own previous champion that is 1.9x to 2.5x faster (6.35 / 15.87 /
43.71 s). It is still slower than llama.cpp, just no longer by a wide margin
([`docs/prefill-long-ctx-2026-09-11.md`](docs/prefill-long-ctx-2026-09-11.md):
f16 WMMA flash attention plus a one-wave SSM scan).

### A cold-cache vendor beat at the decode shape

Single-token decode streams weights out of VRAM. The shape that matters is a
skinny GEMM, **M=1, K=4096, N=12288**, with weights not resident in the 96 MB
Infinity Cache ([`bench/coldcache-protocol.md`](bench/coldcache-protocol.md),
10 repeats, 8 rotating buffers):

| kernel | us per launch |
|---|---|
| `amar_matmul_skinny_v2` CPT=8 @ M=1 | 137.8 to 138.9 |
| **`amar_matmul_skinny_m1` CPT=8** | **121.2 to 121.8** |
| hipBLASLt f16 @ M=1 | 122.1 to 123.3 |

The ranges do not overlap; the margin is about 1%. The mechanism is occupancy,
not bytes: specialising for one row drops the 8-row LDS staging and shrinks the
accumulator to `SIMD[CPT]`, so more waves stay resident. It took four
preregistered rounds, three of which falsified their own predictions.

### An fp16 WMMA GEMM ahead of hipBLASLt

A pipelined square fp16 GEMM on RDNA3's WMMA units: two LDS buffers, one
barrier per K-step, a two-deep global prefetch, XOR-swizzled A, 188 VGPR and no
spills. Tile shape is picked per launch from how many blocks the grid would
have, which is what closed the gap at small sizes
([`bench/wmma-fp16-protocol.md`](bench/wmma-fp16-protocol.md), 10 s clock
warm-up):

| size | 256 | 512 | 768 | 1024 | 1536 | 2048 | 2560 | 3072 | 3584 |
|---|---|---|---|---|---|---|---|---|---|
| **ours, GFLOP/s** | 6372 | 30642 | 64204 | 74824 | 93632 | 91300 | 97974 | 99307 | 105786 |
| hipBLASLt | 6288 | 26324 | 54924 | 63623 | 69332 | 80203 | 87147 | 97224 | 85671 |
| ratio | 1.01 | 1.16 | 1.17 | 1.18 | 1.35 | 1.14 | 1.12 | 1.02 | 1.24 |

These sizes fit in the Infinity Cache for both arms, so this is a warm-cache
comparison. 4096³ is left out on purpose: its three buffers total 100.7 MB,
just over the 96 MB cache, and the same binary read anywhere from 91k to 99k
GFLOP/s depending on what else held the cache. fp32 WMMA does not exist on
gfx1100 (an ISA limitation, checked against `llvm-mc`), so fp16 and fp32 GEMM
numbers here are not comparable.

## How numbers get into this repo

Every performance claim is preregistered: the question, the instrument, the
predicted range and the falsifier are committed **before** the run, and the
result is recorded against them whether or not it agreed. Missed predictions
stay in the file. Rules: [`bench/PROTOCOL-RULES.md`](bench/PROTOCOL-RULES.md).

This is not ceremony. An early version of `docs/BASELINE.md` recorded a ~2x
lead over hipBLASLt. It was measuring an untuned vendor call; fixing three
defects in our own shim took hipBLASLt from 2497 to 5201 GFLOP/s and erased the
lead. **A vendor baseline that looks easy to beat is a bug in your harness
until proven otherwise.**

`bench/` and `exchange/` are a lab notebook, not a product surface. They assume
the author's machine: llama.cpp checked out at `$HOME/llama.cpp`, models under
`$HOME/Models`, a local GPU job queue every timed run goes through, and a few
private tools. Read them for the method and the receipts; expect to edit paths
before any of it runs elsewhere. The parts meant to run on your machine are
`./run-tests.sh`, `tools/ci-checks.sh` and the server.

## The server

`baro-serve` (Rust, `serve/src/`) keeps one engine process alive and speaks
the OpenAI API: `/v1/chat/completions` and `/v1/completions` with SSE
streaming, `/v1/models`, `/v1/cancel`, `/v1/fork`, `/tokenize`, `/detokenize`,
`/health`.

- Stop sequences and EOS are handled inside the engine; a request can be
  cancelled mid-generation.
- Each message boundary in a conversation is checkpointed, so a follow-up turn
  prefills only the new message (2.7 to 8.3x faster than a grid-only cache on
  the same multi-turn replay).
- Sampling (`temperature`, `top_p`, `top_k`, `min_p`, `seed`) runs on the
  device inside the decode loop of `serve/engine.mojo`, speculation included;
  measured on the dense `qwen35` pack, the MoE, and all five `serve/spark.mojo`
  models (distribution, seed and HTTP gates, `exchange/lane-SAMPLE-report.md`).
- `tools` calls come back in the OpenAI `tool_calls` shape, and
  `chat_template_kwargs` reaches the chat template (for example
  `enable_thinking: false`). `response_format` with a JSON schema is enforced on
  the qwen35 dense and MoE engines by a device token mask: every output over
  the 32-schema corpus is valid JSON for its schema at T=0 and T=0.7, with
  reasoning models masked only after `</think>`. Those requests run without
  speculation or the megakernel (about 1.25x slower per token); the
  `serve/spark.mojo` families still refuse it with HTTP 400
  (`bench/grammar-protocol.md`).
- `/v1/fork` branches a conversation from its checkpoint: restore takes 2 to
  4 ms at any prefix length, 2.6x to 3.9x faster wall clock than re-prefilling
  at 1k to 32k tokens.
- One request decodes at a time. Batching is the next design round.

Contract between server and engine: [`serve/PROTOCOL.md`](serve/PROTOCOL.md).

## Building

Requires ROCm 7.2, CMake, Rust (for the server) and
[`uv`](https://docs.astral.sh/uv/). Nothing is installed machine-wide: `uv sync`
creates a repo-local `.venv` pinning `max[all]==26.5.0` (Mojo 1.0.0), and every
script calls `./.venv/bin/mojo`, never a system Mojo.

```sh
uv sync            # repo-local .venv with the pinned Mojo/MAX toolchain
./run-tests.sh     # builds the shim, runs the parity tests and the kernel census;
                   # without a model pack it skips the one test that needs one and exits 3
./bench/run.py     # correctness gate, then throughput
```

Serving a model, one command, any supported GGUF (self-describing `-BARO-*.gguf`
bakes, made from a Hugging Face GGUF or checkpoint by the import pipeline; the
model roster and the pipeline are in [docs/CAPABILITIES.md](docs/CAPABILITIES.md)):

```sh
tools/baro serve MODEL.gguf [--port 8080] [--chat-template-file PATH] [--rebuild]
```

Resolves a structural id from the GGUF's own header (`tools/model-id.py`:
architecture, every dimension the engine compiles in, layer pattern, expert
count -- never the weights), builds the engine binary and weight pack only on
a cache miss, and reuses them on every later `baro serve` of a same-shape
checkpoint. Cache: `~/.cache/baro/<id>/{engine,manifest.json,packs/<gguf-sha256>/}`
(override with `$BARO_CACHE`); packs are keyed by id + the checkpoint's own
sha256, since weights differ per checkpoint even at one shape. Eviction is
manual: `rm -rf ~/.cache/baro/<id>` drops an engine and every pack under it,
`rm -rf ~/.cache/baro/<id>/packs/<sha>` drops one pack. `--rebuild` forces a
fresh engine and pack even on a hit. Engine selection: qwen35/qwen35moe get
`serve/engine.mojo` with `-D BARO_MODEL=<arch>`; llama/qwen2/granite/spark2_5
get `serve/spark.mojo` with a profile read straight from the checkpoint. An
unsupported architecture or tokenizer refuses with the exact missing piece
before any build starts.

Manual build (no cache, one specific engine/pack pair, after packing a GGUF
with `tools/engine-pack.py`):

```sh
./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine
(cd serve && cargo build --release)
./serve/target/release/baro-serve --engine .work/engine --pack .work/engine-pack-q4 --port 8080
```

Pass `--chat-template-file PATH` to either command to override the pack's
`tokenizer-meta.json` chat template for the lifetime of that server. The file
contains raw Jinja and uses the same `messages`, `tools`,
`chat_template_kwargs`, `bos_token`, `eos_token`, and `add_generation_prompt`
context as the embedded template.

`tools/test_server.sh` is the end-to-end gate for that path.

## Layout

| | |
|---|---|
| `kernels/` | Mojo GPU kernels and their parity tests (`docs/KERNELS.md` lists all 100, generated) |
| `serve/` | the engines (`engine.mojo`, `spark.mojo`), tokenizer, prefix cache, and the Rust server |
| `bench/` | benchmark harnesses and the frozen protocols |
| `shim/` | C++ hipBLASLt shim behind a C ABI, the vendor reference arm |
| `tools/` | GGUF packer, numpy reference implementations, gates |
| `results/` | receipts |
| `docs/BASELINE.md` | current truth: every verified number and trap |

All boundaries are C ABI. The tokenizer is Mojo, bit-equal to llama.cpp on the
same GGUF ([`docs/TOKENIZER.md`](docs/TOKENIZER.md)).

## Portability

Everything here is measured on one card, and RDNA3 specifics are load-bearing:
warp size 32 (not 64 as on CDNA), 64 KB LDS per block, tile and CPT parameters
swept for this card's 96 compute units and 96 MB Infinity Cache. Other RDNA3
cards use the same ISA, so the kernels should build, but the tile-dispatch
thresholds were swept to fill this card. Whether they hold elsewhere is
unmeasured, which is exactly what [docs/amd-family.md](docs/amd-family.md) asks
for help with.

## On Android

[baro.apk](https://github.com/amarbaro/baro.apk) is the Android side: a chat client for
`baro-serve`, and a Mojo engine for arm64 phone CPUs that matches llama.cpp's greedy tokens on
two phones (not yet wired into the app). The GPU kernels in this repo do not run there.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) lists where help is wanted, with the check that
decides each item, and the template every change fills in: claim, kind, the check you
ran, the frozen prediction for speed work, the arm receipt, and what you did not verify.
Issue forms: hardware report, bug report, kernel or speed proposal. Hardware reports:
[docs/amd-family.md](docs/amd-family.md).

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 amarbaro.org /
amarbaro.com. Third-party material is named in the NOTICE: the vendored `toml/` reader
(DataBooth, Apache-2.0) and the pre-tokenizer regexes in `serve/pretok-table.json`
(from llama.cpp, MIT). llama.cpp is fetched at a pinned commit, never redistributed.
