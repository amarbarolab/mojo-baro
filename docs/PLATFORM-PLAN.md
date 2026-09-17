# Platform plan: clients everywhere, engines where a GPU is, state moves instead of prompts

One rule orders all six asks of 2026-09-17 (vision and audio in, multi-GPU, training past the
draft-head smoke, running on Android and iPad, NVIDIA's Personal AI Router as a basic ability,
LatentOS as standard): a phone or an iPad is a client of a router, a GPU box is an engine node, and
what travels between nodes is conversation state in our LAT1 format, not prompts. Everything that
does not fit that rule (a CPU decode backend for phones, tensor parallel across cards, training
kernels in Mojo) is listed at the end as not planned, with the reason.

Format follows `docs/NEXT-PLAN.md`: each item names what exists, the design, LOC, the lane, the
gates, a kill line and its GPU budget. Numbers with a receipt cite it; nothing else is measured.
`docs/BASELINE.md` stays the truth for what runs today.

## PAIR studied: thirteen Go services whose whole scheduler is one sort

`~/Projects/imports/Personal-AI-Router`, 0.1.1, Apache-2.0, 460 Go files, an Electron desktop and a
TUI. A control plane for a LAN of peers running Ollama or LM Studio, with Ollama-compatible (`:11434`)
and OpenAI-compatible (`:1234`) proxies that route each independent request to one node. It does not
pool memory, shard a model, or split a request. What is worth taking, each verified in the source:

- **Hierarchy and eligibility.** Cluster > node > engine > models; a request names a model and only a
  node whose running engine holds it is eligible; membership symmetric, no primary (`docs/architecture.mdx`).
- **The rank rule** (`nvpair-job-scheduler/schedule.go`, `rankAt`): nodes sort by pending workloads
  plus GPU pressure, then pressure, then stable id; stale telemetry counts as pressure. That is the
  whole scheduler.
- **Adoption by port probe**: an engine already on its usual port is adopted, an identity-probe header
  (`X-NVPAIR-Engine-Identity-Probe`) tells a real engine from a foreign listener, a health loop follows
  the readiness probe (`nvpair-engine-manager/lifecycle.go`).
- **Discovery** by mDNS (`_nvpair-node._tcp`) plus manual nodes; **pairing** by EAP-NOOB (RFC 9140,
  `services/eap-noob`: ephemeral ECDH authenticated by one out-of-band code), then a mutual-TLS mesh.
- **Control and inference on separate paths**, and every proxy endpoint enumerated: `/api/chat`,
  `/api/generate`, `/api/tags`, `/api/ps`, `/api/version`, `/api/pull`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/models/{load,unload,pull,delete}`, `/v1/node-info`.
- **`scripts/inference-dispatcher`**: a stdlib-only client that enters the cluster as a third party
  would, picks a model from the live inventory, fans out N prompts. Our `bench/served-prompts.sh` is
  the same idea; the model auto-pick and parallel mode are worth copying.

What PAIR cannot do and we can: it knows nothing about conversation state, so a fork or a
continuation can land on a node that must re-prefill. Our router routes on state locality (which
node holds the prefix checkpoint or the latent) and moves the state instead of the prompt. Two ways
to be compatible, both in P0: speak Ollama's API so PAIR and every Ollama client adopt `baro-serve`
as an engine; and run our own router with the same node/engine/model shape, rank rule and proxy
surface, so a PAIR client or `inference-dispatcher` cannot tell the difference.

## Order

| # | item | size | lane | GPU | unblocks |
|---|---|---|---|---|---|
| P0a | Ollama-compatible API on `baro-serve` | S (300 LOC Rust) | sonnet | minutes | PAIR adoption, Open WebUI, phone clients |
| P1 | LatentOS standard: state export/import API, cross-node fork, reader/followers policy | M (450) | opus design, sonnet build | 30 min | routing on state, P6 clients |
| P0b | `baro-router`: discovery, pairing, inventory, rank, proxies, state locality | L (900 Rust) | sonnet | 30 min | P4, P6 |
| P4 | Multi-GPU: engine per device, router data parallel; harness on this box with the iGPU | S wiring, M gate (250) | sonnet | 1 h | |
| P5a | Draft head by self-distillation | S (120 Python) | sonnet | 20 min smoke, 3 h full | A4 unpark |
| P3a | Speech in: whisper.cpp sidecar behind `/v1/audio/transcriptions` | S (150 Rust) | sonnet | minutes | voice clients |
| P2 | Vision in: the `qwen3vl_merger` encoder in Mojo, mrope, image tokens | XL (1,300) | fable kernels, sonnet API | 4 h | images on Ornith, Qwythos-v2, RegesCore |
| P5b | Trunk fine-tune (LoRA in torch) with generalized GGUF write-back | M (350 Python) | sonnet | hours, preemptible | |
| P6 | Anywhere: Android and iPad clients, Android and Mac as llama.cpp engine nodes, aarch64 probe | M clients, S probes | sonnet | none | |
| P3b | Speech out: chatterbox sidecar behind `/v1/audio/speech` | S (120) | sonnet | minutes | optional |

Standing rules, unchanged from `NEXT-PLAN.md` plus today's ledger: protocol note frozen before the
timed run, every GPU job through `gpu-wait run --timeout`, `bench/preflight.sh` and `gate-dryrun`
(with a dry-run stop in the script) before a queue, identity gates against a reference arm, the lane's
own `./run-tests.sh` receipt in its report before a merge (`lane-merge BRANCH` checks it, the cited receipts and ci
in the lane's worktree), no kernel file comments, commits by pathspec. Before a lane that bakes: `disk-dupes` for
exclusive bytes, never apparent.

## P0a. Ollama-compatible API on `baro-serve` (S)

**Exists.** OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/fork`, `/v1/cancel`,
streaming, chat templates (minja), tool calls, JSON and grammars, per-request sampling (`serve/PROTOCOL.md`).

**Design.** A translation layer in `serve/src/`: `GET /api/tags` and `/api/ps` from the model registry;
`GET /api/version` (PAIR's identity probe reads the header and a 200); `POST /api/show`; `POST /api/chat`
and `/api/generate` on the existing request path with NDJSON streaming (`{"message":{...},"done":false}`
frames, a final frame with `done:true`, `eval_count`, `eval_duration`, `prompt_eval_count`);
`/api/embeddings` and `/v1/embeddings` returning the pooled last hidden state (the latent path already
exposes it). Options map: `num_predict`, `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`repeat_penalty`; `num_ctx` above `BARO_TMAX` is reported, not silently clamped. `--ollama-port 11434`
opts into PAIR's expected port. `/api/pull` answers 501 with the `model-import` command.

**Gates.** (1) `pair-dispatch --backend ollama --port <ours> --count 5 --mode
parallel` completes 5/5 with token counts equal to the same prompts through `/v1/completions` (same
seed, T=0). (2) The `ollama` Python client, `chat` and `generate`, streaming and not: byte-identical
text to `/v1/chat/completions` on 20 prompts. (3) PAIR built from source on this box adopts `baro-serve`
on 11434 as Ollama and routes a request to it; receipt is PAIR's workload log naming our node.
**Kill line:** any mismatch in gate 1 or 2. **GPU:** minutes.

## P1. LatentOS as a standard ability (M)

**Exists.** `serve/latent.mojo` mints and ingests KV pages, SSM checkpoints, hidden states and chain
slots (`mint_*`, `ingest_*`); the LAT1 header and kinds (`latentos/proto.mojo`: KV_PAGES, SSM_CKPT,
HIDDEN, LOGITS_TOPK, TEXT); a Unix-socket IPC sidecar; `BARO_STATE_LOAD` state files (f32 KV; int8 KV
refused); `/v1/fork` on the checkpoint chain. Receipts: E12 (KV/SSM handoff scores like text, receiver
35% cheaper), E12-long (8k to 32k), E12-ipc (HIP IPC handle), E14 (one reader, N followers: 1.68x
llama.cpp at N=3, 3.70x at N=10, identity 3/3 and 9/10), E15 (the handoff works through llama.cpp's
state API on three other models), B4-mini (payload vs recipe on a shaped link, `bench/b4-cross-host.sh`).
Design docs in `~/AMDHQ/docs/design/latent-os/`.

**Design.** Promote the experiment surface to a documented, gated API that the router moves around:

1. `POST /v1/state/export` `{request_id | prefix_hash, kinds:[kv_pages, ssm_ckpt, hidden], pos}` returns a
   LAT1 stream, or writes a `.baro` file when `path` is given; `POST /v1/state/import` ingests one and
   returns the request id it seeds; `GET /v1/state` lists resident checkpoints with prefix hashes and
   sizes. int8 KV in state files (C2-mini), a 32k prefix at a quarter of today's bytes.
2. `/v1/fork` gains `target: node_id`: export on the holder, import on the target, answer from there.
   Sticky routing: a request whose prefix hash is resident on a node goes there unless its rank is
   worse by more than one pending job, because a restore costs 2 to 4 ms and a 32k re-prefill 16 s (B5).
3. Reader and followers as a router policy: `POST /v1/fanout` `{prompt, followers:[...]}` runs E14's
   shape on demand: one prefill, N continuations where the rank puts them, one export per follower.
4. llama.cpp nodes take part through E15's bridge: LAT1 KV state to llama.cpp's slot file
   (`tools/llama-slot-to-state.mojo` is the reverse direction; the forward direction is 150 LOC), so a
   phone or a Mac running llama.cpp continues a conversation our engine started.
5. Identity on every move: the LAT1 header's model id, tokenizer hash and `check_identity` are enforced
   on import; a mismatch is a 409 naming both ids, never a silent restore.

**Gates.** (1) Export then import on one node reproduces E12's 20-prompt identity and the restore band
(2.2 to 4.3 ms). (2) Cross-node through the B4-mini veth rig at 100 Mbit, 1 Gbit, 10 Gbit: fork-on-target
ids equal single-node ids; the payload arm beats the recipe arm at 1 Gbit and above for a 32k prefix
(B4-mini's own prediction, re-measured). (3) E14 through the API: N=3 identity 3/3, N=10 at least 9/10
with the reduction-order discordance documented, tok/s within 5% of the E14 receipt. (4) The bridge:
E15's three models continue from our state with the first 32 tokens identical. **Kill line:** an identity
miss outside the documented E14 one, or a cross-node fork slower than re-prefill at 1 Gbit for 32k.
**GPU:** about 30 minutes.

## P0b. `baro-router` (L)

**Design.** One Rust binary in the `serve` crate (`src/bin/router.rs`): mDNS advertise and browse
(`_baro-node._tcp`, and answer `_nvpair-node._tcp` browses with our node-info so PAIR's scanner lists us);
`/v1/node-info` (GPUs, VRAM used, RAM, engines, models with sizes and state, pending count, resident
prefix hashes); engines per node (`baro-serve` instances, one per GPU per P4; Ollama, `llama-server` and
LM Studio adopted by port probe with PAIR's identity-probe convention); inventory refresh on a timer
and on push; rank exactly PAIR's `rankAt` plus the state-locality term from P1; proxies on both
surfaces with streaming pass-through; failover to the next ranked node on connection refusal before
the first byte, never after it; a workload catalog (queued, running, done, node, tokens, duration) at
`/v1/workloads` and an events stream. Pairing by a six-digit code or QR that seeds a PSK, mutual TLS
after it (EAP-NOOB is the right protocol; their Go library is not ours to embed, and a PSK bootstrap is
80 LOC that the P6 clients display as a QR). One TOML config; with none, `baro-router` fronts the local
engines alone, which is what the ComfyUI node and the phone clients talk to.

**Gates.** (1) `pair-dispatch --count 20 --mode parallel` against the router with two engines on
this box (P4's rig): 20/20 complete, placement follows the rank rule (catalog shows pending balanced
within one), every response identical to its single-engine run at T=0. (2) Kill one engine mid-run:
requests in flight on it fail loudly, new ones route to the other; receipt in the catalog. (3) A PAIR
built from source lists our node from the mDNS answer. **Kill line:** placement off the rule, or a
response that differs from its single-engine run. **GPU:** 30 minutes.

## P4. Multi-GPU: data parallel across engine processes (S wiring, M gate)

**Facts.** One RX 7900 XTX (gfx1100) and the Raphael iGPU (gfx1036, 2 CUs). `DeviceContext(device_id=...)`
compiles (`.work/xc/dev.mojo`). The MAX runtime holds about 22 GB per engine process regardless of pack
(COMFY receipt), so two engines cannot share the XTX. The iGPU runs our kernels under
`HSA_OVERRIDE_GFX_VERSION=10.3.0` with a build made under that override (vector add, 0 mismatches, OS
note 2026-09-17; `igpu-env` prints the env, `--probe` proves it, `--run CMD` applies it) but has no bf16 WMMA or bf16 dot, so only kernels without them run there: a functional
second device for the harness, never a performance arm.

**Design.** One `baro-serve` process per GPU, pinned by `ROCR_VISIBLE_DEVICES` (UUID for discrete cards,
index for the iGPU with `HIP_VISIBLE_DEVICES` unset), registered as separate engines of one node; the
router spreads requests and moves state with P1 when a fork lands elsewhere. If a second discrete AMD
card arrives, the wiring is the same plus one measurement: two engines against one, expected 2.0x minus
router overhead. Tensor or pipeline parallel across cards is not planned: no hardware to measure on,
and consumer PCIe peer-to-peer would make every layer boundary a 28 GB/s hop.

**Gates.** (1) Two engines on this box (XTX with the dense q4 pack, iGPU with Qwen2.5-0.5B under the
override on a WMMA-free kernel set), 20 prompts split by the router, each response identical to the
same engine alone. (2) State moved XTX to iGPU and back through P1 reproduces the ids. **Kill line:** any
identity miss. **GPU:** 1 hour; the iGPU jobs run outside the queue, the XTX ones inside it.

## P5. Training beyond the draft-head smoke

**Exists.** `tools/mtp_head.py` (extract, torch replica, parity, byte-verified write-back of `blk.32`),
`tools/mtp_train.py`, `bench/draft_dump.mojo` (real-text hidden-state dumps), the E13 trainer in AMDHQ,
`gguf_to_hf_qwen35.py`, the quality suite (`bench/quality-run.sh`), and two receipts: lr 2e-4 wrecked
the head (3.06%); the corrected recipe converged but read 63.67% against 67.78% untrained
(`exchange/lane-A4-report.md`).

**P5a, self-distillation (S).** Acceptance is agreement with the target's own greedy pick at each
step, and both smokes trained on human next tokens. Change the target: dump the trunk's logits (argmax
and top-8 with probabilities) at every position of the same held-out prompts through
`bench/draft_dump.mojo --mode dump`, train the head with a KL term against that distribution and a
cross-entropy term on the argmax, recipe otherwise as smoke 2. Frozen kill line unchanged: at least
+4 pp on the 5-prompt subset over the untrained head in the same stint, identity 20/20. Then the full
run: 2 GPU hours preemptible at priority 10, the 20-prompt acceptance and the k=2 tok/s as the claim.
**Kill line:** the smoke does not lift, and A4 parks for good with three receipts.

**P5b, trunk fine-tune with generalized write-back (M).** Generalize `--mode writeback` to any tensor
set (`tools/gguf-writeback.py`: names, shapes, dtypes from the source GGUF, byte ranges verified as
today); train a LoRA in torch (AMDHQ venv, the E13 loop) on a task corpus; merge into the named
tensors; write back; repack with the unmodified `engine-pack.py`; bake. Gates: the write-back receipt
(zero differing bytes outside the named ranges), `dense-run.sh` forced agreement against llama.cpp on
the patched file at the class bar, the quality table before and after, the 20-prompt identity harness
on unpatched prompts. Training stays in torch on this box; the engine's job is the receipts and the bake.

## P3. Audio: a sidecar today, native tokens when a model has an audio tower

**P3a, speech in (S).** `~/Models/whisper.cpp/build/bin/whisper-server` and `whisper-cli` exist with
`ggml-large-v3-turbo-q5_0.bin` and `ggml-base.bin`. Add `POST /v1/audio/transcriptions` (OpenAI shape:
multipart file, `model`, `language`, `response_format`) to `baro-serve`, proxied to a whisper-server the
engine lifecycle starts on demand and stops after idle, through `gpu-wait` like every resident; the
router registers it as an engine of kind `stt`. Gate: a fixed 20-clip set transcribed through our
endpoint equals `whisper-cli` output byte for byte on the same model and beam settings (same engine, so
the gate is the plumbing). **Kill line:** any difference. **GPU:** minutes.

**P3b, speech out (S, optional).** `~/Models/tts/chatterbox` behind `/v1/audio/speech`, same sidecar
pattern. Gate: fixed-seed audio identical across two runs (sha256).

**P3c, native audio tokens.** None of our mmproj files carries `clip.has_audio_encoder`. When a model in
`~/Models` does, it follows P2's design with the audio tower in place of the ViT.

## P2. Vision input: the encoder is 27 blocks of kernels we already have (XL)

**Facts.** Three of our models ship a projector: `mmproj-Ornith-1.5-9B-BF16.gguf`,
`mmproj-Qwythos-9B-v2-BF16.gguf`, `RegesCore-1.0-35/mmproj-F16.gguf`; `mmproj-info FILE` reads them in one line (same 27-block encoder in all three,
about 411M parameters, 576 image tokens per picture; RegesCore projects to 2048). The Ornith file:
architecture `clip`, `clip.projector_type = qwen3vl_merger`, image 768, patch 16, hidden 1152, FFN 4304,
27 blocks, 16 heads, GELU, spatial merge 2x2, projection dim 4096 (the LLM hidden), 363 tensors,
`is_deepstack_layers` flags per block. Reference arms: `llama-mtmd-cli` and `libmtmd` (built here) and
the HF Qwen3-VL vision tower in the AMDHQ venv.

**Design.**

1. Preprocessing: resize to the model's grid (multiples of 32 after the merge), normalize with the
   file's mean and std; a Python oracle (`tools/vision-ref.py`, the HF processor) and a Mojo port; the
   gate compares the patch tensor bit for bit in bf16.
2. Patch embedding: the 3D conv is a GEMM over unfolded patches; reuse the WMMA prefill GEMM.
3. 27 encoder blocks: layernorm, non-causal attention over all patches (the prefill WMMA attention gains
   a `causal=False` parameter, fable), GELU MLP, 2D rotary in the encoder (its own small kernel).
   Deepstack layers feed intermediate features to the merger as the flags say.
4. Merger: the 2x2 merge is a reshape, then the MLP to 4096; the output is a run of image tokens.
5. Injection: image tokens replace the `<|vision_start|>...<|vision_end|>` span through the hidden-ingest
   seam (`ingest_hidden_latent` already writes embeddings at positions); the LLM's rope becomes mrope
   with three position channels (t, h, w) for image tokens and (t, t, t) for text, a change to position
   handling in `kernels/attn.mojo` and `dattn.mojo` (fable), gated by the text identity harness first.
6. API: `image_url` content parts (base64 and local paths) in `/v1/chat/completions`, the template's
   vision tokens through minja, Ollama's `images` field in P0a.

**Parity ladder** (kernel-parity skill): preprocessed patches, patch embedding, block 0, block 27, merger,
each against the HF tower in float32 with the same bf16 rounding points, tolerances frozen before the
run; then the whole thing: first 64 generated tokens on 20 images with fixed prompts, teacher-forced
agreement against `llama-mtmd-cli` on the same GGUF and mmproj at the `dense-run.sh` class bar, with the
same prompts run text-only first to prove mrope moved nothing.

**Gates.** The ladder, the 20-image agreement, text-only identity 20/20, the kernel census, and a
throughput line (encoder ms per 768 image; decode tok/s within 2% with images in context). **Kill line:**
text identity broken by mrope, or agreement under the class bar after one repair round (the granite
precedent). **GPU:** 4 hours. **Lane shape:** fable for the two kernels and mrope, sonnet for preprocessing,
API and the oracle, dispatched after a builder conference (CLAUDE.md §10).

## P6. Running on anything: clients everywhere, engines where a GPU is

**Facts measured 2026-09-17.** Mojo 1.0 emits aarch64 code for `aarch64-unknown-linux-gnu` and
`aarch64-linux-android` (the archive is produced) but the host linker refuses it and the x86 package
ships no aarch64 runtime; Modular publishes a Linux aarch64 package, so an aarch64 host or container
builds natively. Mojo has no Windows, iOS or Android target. Our kernels are gfx1100; other AMD parts
need a per-arch retune (RDNA2 through the override, proven on the iGPU); Apple Metal exists as a Mojo
target for a future port only. The engine has no CPU path (about 10k LOC to build one, landing at
llama.cpp's CPU speed).

1. **Android client (M).** An app in `~/Android`'s style: mDNS discovery of the router, QR pairing (P0b),
   model list, streaming chat, voice through P3a, images through P2 when present, and "continue on this
   device", which exports the conversation state (P1) to a local llama.cpp when installed. Gate: the
   20-prompt identity through the phone equals `curl`, read from the router's catalog.
2. **iPad and iPhone client (M).** PWA first, served by the router at `/`, same features; a Swift wrapper
   only if the PWA limits bite. Gate as above.
3. **Android and Mac as engine nodes (S each).** llama.cpp arm64 (Vulkan on Android through Termux or a
   Linux chroot, Metal on a Mac) adopted by the router as kind `llamacpp`, taking part in LatentOS
   through the E15 bridge. Gate: E15's identity through the router, and a routed request completing on
   the phone with its node named in the catalog.
4. **mojo-baro tooling on aarch64 Linux (S probe).** Modular's aarch64 package in an aarch64 VM (the
   `labiso` QEMU rig): build `baro-serve`, the router and the CPU-only tools (`gguf_reader`, tokenizer,
   pack tools). Gate: `tools/ci-checks.sh` and the tokenizer parity on aarch64. The decode engine needs
   an AMD GPU or the unplanned CPU backend, so the aarch64 deliverable is the router and the tooling.
5. **Other AMD cards (M per arch).** gfx1030 through gfx1103: build under the matching target, run the
   parity ladder, retune the four hot kernels. Gated by hardware access.

## Not planned, and why

- Tensor parallel or sharding across GPUs or nodes: PAIR does not either, and there is no hardware to
  measure on.
- Backward kernels or training loops in Mojo: torch on this box does it, and the engine's value is the
  receipts and the bake.
- A CPU decode backend: XL for llama.cpp's CPU speed; it returns to the table only if an on-phone
  mojo-baro engine shows a payoff a llama.cpp node cannot give.
- Embedding PAIR's Go services: we speak its protocols instead.
- Windows engines (no Mojo there): Windows machines are clients or WSL2 llama.cpp nodes.

The first thing that would prove this plan wrong is P0a's third gate: if a PAIR built from source will
not adopt `baro-serve` as an engine, the compatibility half of the thesis is a claim about their code,
not ours, and P0b's own router becomes the only path.
