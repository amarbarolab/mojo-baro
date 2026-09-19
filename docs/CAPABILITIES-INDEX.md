# Capability index

One line per capability, with its status word and where it is switched on.
The prose, the evidence and the caveats live in
[`docs/CAPABILITIES.md`](CAPABILITIES.md); this file is only the map. Status
words mean what they mean there: **WORKS** (a named check passed),
**PARTIAL** (works inside a stated limit), **DEFAULT-OFF** (implemented, not
on), **PRESENT BUT UNUSED** (nothing calls it), **CLAIMED** (no check),
**FAILS** / **PARKED** / **KILLED**.

Where a row and `CAPABILITIES.md` disagree, that file wins.

## Engine and decode

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

## Quantization and memory

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

## Serving APIs

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

## State, checkpoints and forking

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

## Routing and multi-engine

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

## Clients

| capability | status | notes |
|---|---|---|
| Android app as a remote chat client | WORKS | 80 requests byte-identical to curl |
| Android on-device llama.cpp mode | FAILS its gate | 4/5 exact, near-tie logprob divergence |
| Android voice input end to end | UNVERIFIED | only unit-tested against a mock |
| PWA client served by the server itself | WORKS | offline shell, service worker, 400px clean |
| PWA browser voice capture into transcription | WORKS | real whisper sidecar |
| ComfyUI node (chat and JSON-schema) | WORKS | time-sliced, engine stops around each call |
| aarch64: router and CPU tooling | WORKS | no decode engine there |

## Models and pipeline

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

## Training

| capability | status | notes |
|---|---|---|
| LoRA training and GGUF write-back | PASS on an amended bar | rank-16 on `blk.24..31` ffn_down |
| Self-distilled draft head | PARKED, no signal | +2.30 pp against a required +4 |

## Verification

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
