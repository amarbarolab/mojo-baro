# Chat-engine lane — single-file build plan (2026-09-08)

Self-contained. Everything a fresh agent needs is in this file or in the repo
at the paths named here. Nothing outside the repo checkout is required except
the machine (one RX 7900 XTX, ROCm, 64 GB RAM) and the model packs.

Read order: §1 state → §2 environment → §3 rules → §4 gates → §5 milestones
(M0 … M6, each a build tutorial) → §6 deliverable formats. Do not skip §3.

---

## 1. Where the lane stands (updated 2026-09-08 12:00)

| item | value |
|---|---|
| worktree | `~/Projects/mojo-baro-lanes/chat` (git worktree of `~/Projects/mojo-baro`) |
| branch | `lane-chat`, forked from `main` at `5ae2925` |
| HEAD | `48a48b1` M0d landed; tree clean |
| landed | M0a (`4027b6e`), M0b (`9d3b228`, f32 KV; f16/bf16 fail identity 17/20 → M5), M0c (`a86516b`, runtime `BARO_TMAX`, `exceed_context_size_error`, long prompt files p8192/p32768/p100000), M0d (`48a48b1`, split-K decode attention, `BARO_ATT_SPLIT_T`/`BARO_ATT_SPLIT`) |
| numbers | 20-prompt champion path 133.2 tok/s_gen unchanged, bit-exact. Long T decode: 8k 117.5, 32k 85.0, 100k 48.7. Prefill: 8k 8.3 s, 32k 50 s, 100k 299 s (llama.cpp 2.6 / 12.8 / 63.7 s) |
| status file | `.work/briefs/status-chat.md` (one open question on P-E1's count band, non-blocking) |
| protocol | `bench/chat-protocol.md` (Baseline, M0a, M0b, M0b-2, M0c, M0d all with Results) |
| binaries | `.work/engine-m0b`, `.work/engine-m0c`, `.work/engine-m0d` = `.work/engine` |

M0 is complete. **Start at §5.2 (M1).** Open levers recorded, not scheduled:
in-block attention parallelism (512 threads in flight, both halves of the
block) for another ~1.5x at 32k; prefill attention re-reads the whole KV
per 1024-row chunk (O(T²)), the reason prefill is 3–5x behind llama.cpp
on top of the bf16 GEMM.

## 2. Environment

### 2.1 Toolchain

```
cd ~/Projects/mojo-baro-lanes/chat
uv sync                      # once; creates .venv with the pinned Mojo/MAX toolchain
./.venv/bin/mojo --version   # always this binary, never a system mojo
```

The worktree needs three symlinks into the main checkout (already present;
recreate if missing):

```
ln -sfn ~/Projects/mojo-baro/.work/engine-pack-q4 .work/engine-pack-q4
ln -sfn ~/Projects/mojo-baro/.work/engine-pack-q8 .work/engine-pack-q8
ln -sfn ~/Projects/mojo-baro/.work/engine-pack    .work/engine-pack
```

Packs: `engine-pack-q4` (6.2 GB, default, the champion path),
`engine-pack-q8`, `engine-pack` (bf16 reference). Each pack dir carries
`prompt-tokens.txt` and `ref-tokens-64.txt` (the identity reference).

### 2.2 Build commands

```
./.venv/bin/mojo build serve/engine.mojo            -I kernels -o .work/engine
./.venv/bin/mojo build kernels/test_prefill.mojo    -I kernels -o .work/test_prefill
./.venv/bin/mojo build kernels/test_attn_block.mojo -I kernels -o .work/test_attn_block
./.venv/bin/mojo build kernels/test_mega_block.mojo -I kernels -o .work/test_mega_block
cd serve && cargo build --release                  # baro-serve (Rust HTTP front), binary serve/target/release/baro-serve
```

A/B arm binaries are named `.work/engine-<label>`; keep `.work/engine-m0a`
(built from `4027b6e`) as the reference arm for M0.

### 2.3 Running the engine

One-shot (every gate uses this path):

```
BARO_PACK=.work/engine-pack-q4 ./.work/engine                       # prompt from <pack>/prompt-tokens.txt, 64 tokens
BARO_PACK=.work/engine-pack-q4 BARO_PROMPT=bench/prefill-prompts/p0512.tokens ./.work/engine
```

Prints, and these lines ARE the parameter receipts (rule P1): `TMAX:`,
`kv dtype:`, `spec k:`, `prompt tokens:`, `prefill rows:`, `prefill_s`,
`tok/s_gen:`, `GENERATED: …`, `mega fail word: N`.

Server mode:

```
serve/target/release/baro-serve --engine .work/engine --pack .work/engine-pack-q4 --port 8080
```

Contract: `serve/PROTOCOL.md` (stdin/stdout JSON lines, `BARO_SERVE=1`,
strictly serial requests, EOF = clean shutdown).

Engine env knobs: `BARO_PACK`, `BARO_PROMPT`, `BARO_MEGA` (1 default),
`BARO_SPEC` / `BARO_SPEC_K` (MTP, k=2), `BARO_DRAFT_Q4`, `BARO_MEGA_WIN`,
`BARO_DOT` (0 default), `BARO_PREFILL_C`, `BARO_DUMP=path` (per-layer dump
for divergence localisation), `BARO_PROFILE`.

### 2.4 GPU queue (mandatory)

Every command that touches the GPU runs as

```
gpu-wait run [--priority N] [--vram GB] [--shared] -- <cmd>
```

Timed runs `--priority 90 --vram 23` (exclusive). Builds, parity tests,
identity one-shots `--priority 50 --vram 8 --shared`. Long runs are detached:
`setsid nohup … > .work/<name>.out 2>&1 &` and a done-marker line at the end
of the script; poll the file, never the terminal. `gpu-wait list` shows the
queue; another lane's `llama-server` (7 GB) may be resident.

### 2.5 Repo map (what each file owns)

| path | owns |
|---|---|
| `kernels/attn.mojo` | `attn_head_body` (shared decode attention body), `amar_attn_decode`, `amar_kv_append`, `amar_attn_prefill` (online-softmax, T-agnostic), KV constants `KVT KVPAGE=128 KVPSH=7 KVPAD KVHSTR TCAP`, `kv_off[NAT](t, att_i, kvh)` = the only KV address function |
| `kernels/mega.mojo` | megakernel `amar_mega_token` (32 layers + head, one persistent launch per token, grid barrier with bounded spin + fail word), `attn_phases`, `mega_body` |
| `kernels/ssm.mojo` | gated-delta-net SSM layers (24 of the 32) |
| `kernels/matmul_skinny*.mojo`, `kernels/matmul_mmq.mojo` | decode dot kernels (q4 b-form, q8 wave-per-row), prefill GEMM |
| `kernels/test_*.mojo` | parity tests vs numpy references (`tools/*-ref.py`) |
| `serve/registry.mojo` | shape constants (`H=4096 FFN=12288 VOCAB=248320 N_LAYERS=32 N_SSM=24 N_ATT=8 NKVH=4 NQH=16 HD=256 TMAX=1088 GEN_N=64 CP=1024`), kernel instantiations, pool sizes `TPAGES KVPOOL KVPOOL1` |
| `serve/window.mojo` | per-token / per-window forward: buffers, launches, MTP draft window (`blk32_forward`) |
| `serve/engine.mojo` | host loop: pack load, prompt, prefill chunks, decode loop, one-shot vs `BARO_SERVE` JSON loop, receipts |
| `serve/src/` (Rust) | `baro-serve`: HTTP → engine stdin JSON; today OpenAI chat completions only |
| `tools/merge-gate.sh` | the whole-repo gate (§4.1) |
| `tools/mega-gate.sh` | megakernel identity gate (`BARO_MEGA=1` vs `0`) |
| `tools/check-tokens.sh REF LOG` | 64-token identity check |
| `tools/model-ref.py` | numpy fp32 reference decode (the arbiter for non-bit-exact changes) |
| `tools/test_server.sh OUTDIR` | server suite, writes `SUMMARY.txt` |
| `tools/deerflow-smoke.py`, `tools/openai-tap.py` | harness smoke + wire tap |
| `tools/isa-receipt.py`, `~/iTools/bin/isa-loops` | ISA fingerprints (instruction count, vgpr, spills, dot-loop VOPD class) |
| `bench/ab-prompts.sh`, `bench/mtp-prompts.sh`, `bench/mtp-prompts/p*.tokens` | 20-prompt P4 A/B |
| `bench/prefill-prompts/p0512.tokens` | the T≈575 decode probe |
| `bench/PROTOCOL-RULES.md` | P1–P6, binding |
| `bench/chat-protocol.md` | this lane's preregistrations and results |
| `docs/KERNELS.md` | kernel census; `tools/ci-checks.sh` fails if a kernel or test user is missing from it |
| `.work/` | gitignored build/run dir; binaries, logs, A/B dirs |
| `.work/design/prs/` (main checkout) | 100+ incumbent PR bodies, `index.tsv`; read the named ones before M1/M2/M6 |

---

## 3. Rules (binding, no exceptions)

1. **Preregister before building.** Each milestone gets a section in
   `bench/chat-protocol.md` with Change / Not in this step / Predictions
   (numbered P-xN, each with rationale and a falsifier) / Verification before
   timing / Gate / Result (empty). Commit that text BEFORE the first edit; the
   commit hash is the receipt. Never edit a prediction after the run.
2. **P1: parameter read-back.** A timed arm exists only if every arm-defining
   value is read back from the run's own output (§2.3 lines) or from a
   rebuild in the same command. No receipt → the arm is VOID, rerun it.
3. **P4: decode numbers are the 20-prompt median** from `bench/ab-prompts.sh`.
   A single prompt is an instrument receipt, never a claim. Spread > 10 %
   voids the arm.
4. **Bit-exact vs identity.** A change that preserves expression form and
   summation order must be bit-exact (`GENERATED` byte-identical). Any other
   change is gated on identity: q4 64/64, q8 64/64 (`tools/check-tokens.sh`),
   mtp 20/20, A/B 20/20; disagreements are arbitrated by `tools/model-ref.py`.
5. **ISA fingerprint before any megakernel A/B.** After every build touching
   `kernels/mega.mojo` or anything it inlines: `tools/isa-receipt.py --hist`
   on the q4 `amar_mega_token` object (instructions 12727 ± 40, vgpr spill 0)
   and `isa-loops` fused dot loop `dual ≥ 105`, loops 2–4 within ±2 of
   80/80/60. A miss is a lost schedule lottery ticket: re-roll once by
   respelling only, second miss → stop and write the question to the status
   file. Never GPU-time a losing fingerprint.
6. **Kernel files carry zero comments and zero docstrings.** Rationale goes
   in commit messages, `bench/chat-protocol.md`, `docs/`.
7. **Gate fails twice → stop.** Write the question to
   `.work/briefs/status-chat.md` under `## QUESTIONS`; do not relax the gate.
8. **Commit per concern, conventional subject, why-body with the numbers.**
   No `Co-Authored-By` or any model attribution trailer. Never commit red.
   No git remote exists; never propose one.
9. **Deliverables are files.** Status `.work/briefs/status-chat.md`, report
   `exchange/lane-chat-report.md`, protocol `bench/chat-protocol.md`.
   Terminal output is status only.
10. **Search.** Local repo + `.work/design/` digests first; web only via
    firecrawl (`firecrawl_search` / `firecrawl_scrape`); never WebSearch /
    WebFetch.
11. **Not this lane.** 27B, Flash-Next experts, int8 prefill GEMM (lane
    `int8`), multi-tenant batching, the grammar PDA internals (lane
    `grammar` builds `grammar/`; M6 consumes it).
12. **Ownership.** This lane owns `kernels/attn.mojo`, `kernels/mega.mojo`,
    `kernels/ssm.mojo`, `serve/`. Other lanes never touch them; this lane
    never touches `kernels/matmul_mmq.mojo` or `grammar/`.

---

## 4. Gates

### 4.1 Whole-repo gate (every milestone)

```
gpu-wait run --priority 50 --vram 8 --shared -- tools/merge-gate.sh
cat .work/merge-gate.txt
```

Lines required at least as green as the frozen baseline (`bench/chat-protocol.md`
"Baseline"): engine build exit 0 · `run-tests.sh` exit 0, 0 orphans ·
`ci-checks.sh` all passed · test_prefill PASS · one-shot q4 `PASS: 64 tokens
match` · one-shot q8 PASS · prefill p0512 `prefill rows 511`, `mega fail word: 0`
· mtp k=2 identity 20/20 · `test_server.sh` ALL PASS.

Plus the three kernel parity binaries (`test_prefill`, `test_attn_block`,
`test_mega_block`) each exit 0 with PASS / `0 mismatches`.

### 4.2 20-prompt A/B (every milestone that touches decode)

```
AB_ENGINE_B=.work/engine gpu-wait run --priority 90 --vram 23 -- \
  bench/ab-prompts.sh .work/engine-<ref> .work/ab-<label> \
  "BARO_PACK=.work/engine-pack-q4" "BARO_PACK=.work/engine-pack-q4" ref new
```

Read `.work/ab-<label>/arm.txt` FIRST (binary hashes, power cap, voltage);
then `results.txt` → median, spread, identity fails. Band: within ±2 % of the
reference arm unless the preregistration says otherwise; identity 20/20.

### 4.3 Lane-ending gates (all must hold before `exchange/lane-chat-report.md` says DONE)

- §4.1 ALL PASS; identity at T ≤ 1088 unchanged (20/20, 64/64).
- Byte-exact restore test green (M1); determinism test green (M3).
- `NONINTERACTIVE=1 MODEL=baro-9b uv run python tools/deerflow-smoke.py`
  reaches write_file → read_file → answer, exit 0; the tap
  (`tools/openai-tap.py`) shows call 2 prefill < 200 tokens; TTFT at 8k warm
  < 60 ms.
- Claude Code and OpenClaw each complete one tool-using task on the Anthropic
  Messages endpoint; `event: tool_call_dispatch` observed before stream end.
- RULER subset: K8/V4 effective length equals bf16 at 8k–128k (M5).
- 20-prompt A/B vs `main`'s 133.9 tok/s_gen at temperature 0: no regression.

---

## 5. Milestones — build tutorials

Every milestone follows the same loop:

```
preregister (commit) → build → fingerprint (if mega touched) → parity tests →
merge-gate → A/B → fill Result → commit with numbers → status line
```

Sizes are LOC estimates; kernel-touching steps are marked (K).

### 5.1 M0c — runtime T end to end (~150 LOC, K) — DONE `a86516b`; M0d DONE `48a48b1` (see bench/chat-protocol.md)

**Goal.** `TMAX` stops being a compile-time capacity. A request may carry up
to 128k tokens; the engine allocates from an env/request capacity; prefill
past `CP=1024` is chunked; overflow returns llama.cpp's `exceed_context_size`
error shape instead of raising.

**Files.**
- `serve/registry.mojo`: `TMAX` → `TMAX_DEFAULT` (1088 stays the identity
  reference); `TPAGES`, `KVPOOL`, `KVPOOL1` computed from a runtime `tcap`
  passed to the allocator, not from a comptime.
- `serve/engine.mojo`: read `BARO_TMAX` (default 1088) at start; allocate
  K/V pools, token buffer from it; print `TMAX:` (already printed, keep as the
  P1 receipt); in `BARO_SERVE` mode accept a per-request `max_context`
  bounded by the process capacity; overflow → JSON
  `{"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error","n_prompt_tokens":N,"n_ctx":C}}`
  (this is llama.cpp's shape; DeerFlow's handling of it is a test, §5.5).
- `serve/window.mojo`: prefill loop over chunks of `CP` until the prompt is
  consumed; `amar_attn_prefill` already handles arbitrary T (online softmax,
  `PA_TK=16` tiles); check that `amar_kv_append` writes through `kv_off` for
  positions ≥ 1088.
- `serve/src/` (Rust): pass `max_context` through; map the error JSON to HTTP
  400 verbatim.
- `kernels/`: nothing should change. If a kernel needs a runtime T argument,
  it is already `Int32(att_i)`-style; do not reintroduce a comptime extent.

**Preregister.**
- P-D1 identity at `BARO_TMAX=1088` bit-exact vs the M0b binary (only
  allocation sites change).
- P-D2 20-prompt median within ±2 % of M0b.
- P-D3 TTFT table (prefill_s from the run) at 8k / 32k / 100k on
  `bench/prefill-prompts/` extended set (generate `p8192.tokens`,
  `p32768.tokens`, `p100000.tokens` with `tools/baro-tokenize` from a long
  text; commit them). Compare against llama.cpp on the same token files
  (`tools/llama-ref-run.sh`, `--pure Q4_0` gguf). Record the gap; the bf16
  prefill is known slower (int8 MMQ is lane `int8`'s job).
- P-D4 `mega fail word: 0`; decode at T=32k and T=100k runs to 64 tokens
  without a residency fault.
- Falsifier: identity at 1088 not bit-exact.

**Gate.** §4.1, §4.2 vs M0b binary, P-D3 table filled.

**Also in M0c (design §4, K, ~150 LOC, optional but preregistered
separately as M0d if attempted):** the GQA-grouped long-context decode kernel:
one KV head + its 4 Q heads per wave, K/V loaded once, 32 lanes per 256-dim
row, split-K over 64-page spans with (m, l, O) partials and one merge
workgroup per head, no LDS. Only worth building if the p0512-style probe at
T=32k shows decode falling below ~100 tok/s; measure first, write M0d only
if the number demands it.

### 5.2 M1 — checkpoints, radix pages, retention, salt, churn (~350 LOC)

**Read first.** `.work/design/prs/` bodies: TRT-LLM #18272 (branch-point
snapshots), vLLM #52789 (checkpoint inside one forward), ollama #17901 and
#14887 (restore points survive cancel; trie), llama.cpp #24176 / #25472
(message-boundary checkpoints, min-step eviction). Design §3 and §3a.

**Model facts.** 24 SSM layers (state 32 heads × 128×128 f32 + conv 4×6144 =
26.25 MiB per checkpoint) + 8 attention layers (KV pages). SSM state lives in
ring slots (`sl` slots, `rg` index) in `serve/window.mojo`; a checkpoint is a
slot copy, never a recompute.

**New file `serve/prefix.mojo`** (host side, no kernels):
- `PageHash`: canonical-bytes hash of (parent_hash, 128 token ids, salt) —
  never a language pickle; SHA-256 over a fixed byte layout.
- `RadixTree`: nodes keyed by page hash; each node holds page index into the
  KV pool, refcount, retention priority (1–100, default 35, anchors 100),
  last-use tick, and a list of SSM snapshot positions spanning its edge.
- `Checkpoint`: {position, prefix_hash(tokens[0:position]), abi_hash (pack
  header hash), generation, conv[24], delta[24], page-chain ref}; stored in
  pinned host memory (`DeviceContext` host alloc); restore REQUIRES the hash
  match, never position alone.
- `lookup(tokens, salt) → (matched_pages, usable_up_to)` where usable is the
  last snapshot boundary at or before the KV match (TRT-LLM hybrid rule).
- Boundaries where snapshots are taken: every role boundary in the rendered
  template, prompt end, branch points (decided at lookup time on the
  diverging request, aligned down to the page), periodic 1024 as safety net.
- Eviction: prioritised LRU, only leaf pages; never trim a diverged prefix,
  fall back to the nearest earlier snapshot; checkpoints within a min-step
  (256 tokens) of each other collapse to the later one.
- Cancel path: a cancelled prefill keeps every restore point it crossed.
- Receipt fields per response: `prefix_hit_tokens`, `checkpoint_hit`,
  `prefix_churn` (first differing token position vs the previous request from
  the same salt).

**Wire.** `serve/engine.mojo` `BARO_SERVE` loop: on request, tokenise,
lookup, restore (slot copy + page chain), replay from `usable_up_to`, chunked
prefill checkpoints mid-chunk inside one forward (split only the SSM kernel
at the offset, not the whole forward: vLLM #52789). `serve/src/` accepts
`cache_salt` and `retention_priority` in a `baro` request extension.

**New test `kernels/test_prefix.mojo`** (byte-exact restore CI):
cold A||B vs restore(A)+replay(B): conv, delta, KV pages, next-token logits
`memcmp`-equal at positions 0/1/1023/1024/1025/7900/8191/8192; mutations at
first token, checkpoint±1, last token must miss; a corrupted hash must fail
restoration. Register in `run-tests.sh` and `docs/KERNELS.md`.

**Preregister.**
- P-E1 restore test passes all listed positions and mutations.
- P-E2 DeerFlow smoke: call 2 prefill < 200 tokens (tap receipt); TTFT at
  the 7,914-token system prompt, warm, < 60 ms.
- P-E3 identity and 20-prompt median unchanged vs M0c (decode path
  untouched).
- P-E4 memory: 32 checkpoints per slot = 840 MiB pinned; engine prints
  `checkpoints:` capacity as the P1 receipt.
- Falsifier: any byte mismatch in P-E1.

### 5.3 M2 — control block, EOS/stop, limits, overlap host loop (~120 LOC, K-adjacent)

**Design §8.** Today the host syncs per token (device 7.36 ms, wall 7.66 ms).

**Files.**
- `serve/window.mojo`: 64-byte control block in pinned host memory
  {generation, cancel, state, produced}; the megakernel reads `cancel` once
  per token at the top (one `global_load` of a host-coherent pointer, before
  the first grid barrier) and exits the persistent loop early. Kernel edit is
  ~10 lines in `kernels/mega.mojo` → fingerprint rule 5 applies.
- `serve/engine.mojo`: overlap loop — launch step n, then do step n−1's CPU
  work (detokenise, lexer, grammar advance, receipts) while the GPU runs;
  EOS set {248046 `<|im_end|>`, 248044 `<|endoftext|>`}; stop strings matched
  across token boundaries on the host (keep a rolling decoded suffix);
  `max_tokens`, `max_context` limits; cancel on stdin request
  `{"cancel":true}` or client disconnect (server closes the request).
- `serve/src/`: propagate disconnect → cancel line; `stop`, `max_tokens`
  fields.

**Preregister.**
- P-F1 cancel mid-token: a request cancelled at token 10 stops within one
  token; the engine returns `produced: 10 or 11` and is ready for the next
  request; restore points survive (M1 test rerun after a cancel).
- P-F2 wall − device < 0.1 ms/token on the 20-prompt set (print both
  `gpu_total_s` and `host_enqueue_s`; they are already measured).
- P-F3 identity 20/20 and median within ±2 %; fingerprint in class.
- P-F4 stop-string test corpus (`tools/test_server.sh` new cases: stop
  inside a token, across two tokens, at EOS) PASS.

### 5.4 M3 — device sampler, exact speculation, SA/NGram drafters (~350 LOC, K)

**Design §4–§5.** Today: argmax only, MTP k=2 fixed.

**Files.**
- New `kernels/sample.mojo`: device sampler chain in llama.cpp's default
  order: penalties (repeat / frequency / presence over last-N) → DRY →
  top-n-sigma → top-k → typical-p → top-p → min-p → XTC → temperature →
  dist. Temperature 0 = existing argmax fast path (untouched, keeps bit-exact
  gates). Counter-based RNG (Philox-style from `seed`, request index, token
  index) so replays are deterministic. Vocab 248320: one workgroup per row,
  top-k via wave-level partial sorts; keep it out of the megakernel (own
  launch after the head phase) so the fingerprint stays clean.
- `serve/window.mojo`: rejection-sampled MTP (Leviathan): draft and target
  distributions with penalties applied identically; accept/resample on
  device; recurrent rollback = the ring slot for the last accepted draft
  position (llama.cpp #26623 invariant — write it as a test).
- Drafters (host side, `serve/draft.mojo`): suffix automaton over generated
  tokens (threshold 4) and NGram prompt lookup; exact under greedy; adaptive
  draft length controller (ollama #16791): pick the length maximising
  committed tokens/s from per-position acceptance EWMA and forward cost, back
  off to plain decode; one host sync per round.
- Defaults from the gguf generation config; Qwen thinking preset temp 0.7 /
  top_p 0.8 / top_k 20 / min_p 0. OpenAI names mapped: `frequency_penalty`,
  `presence_penalty`, `seed`, `logprobs`; `extra_body.repeat_penalty`
  (DeerFlow sends 1.15 with temperature 0).
- Test `kernels/test_sample.mojo`: sampler vs a numpy reference
  (`tools/sample-ref.py`, new) on fixed logits for each stage; determinism:
  same request + seed, spec on / off, 3 runs, byte-equal output.

**Preregister.**
- P-G1 temperature-0 path bit-exact vs M2 binary (fast path untouched).
- P-G2 determinism test green (3×, spec on/off).
- P-G3 20-prompt A/B at temp 0: no regression (±2 %); at temp 0.7 the spec
  acceptance receipt is recorded per prompt (not gated).
- P-G4 SA/NGram: on a prompt containing a 512-token file echoed back,
  committed tokens/s ≥ 1.5× plain decode (instrument receipt, one prompt,
  reported not claimed).

### 5.5 M4 — lexer, dispatch event, three APIs, receipts (~350 LOC, no kernels)

**Design §7. Reference parsers:** llama.cpp `common/chat.cpp` Qwen3/Hermes
(local checkout `~/llama.cpp`), Dynamo #13975 (emit tool-call deltas on every
parser delta), #6422 (reasoning state machine normal→reasoning→normal per
`<think>` block, handles `</think>text<think>` inside one chunk).

**Files.**
- `serve/src/lexer.rs`: state-machine lexer over the raw token-text stream:
  `<think>…</think>`, `<tool_call>…</tool_call>` (Qwen3 JSON body), plain
  text; one owner for reasoning parsing; interleaved segments keep order on
  replay; prior thinking behind tool calls is NOT truncated in agent turns.
- `serve/src/api_openai.rs` (chat completions, extend the existing one),
  `serve/src/api_responses.rs` (OpenAI Responses), `serve/src/api_anthropic.rs`
  (Messages: `message_start` with `input_tokens`, content blocks, tool_use
  blocks, `GET /v1/models` and `/v1/models/{id}` with slashed ids, tolerate
  `cache_control` annotations).
- Side channel: SSE `event: tool_call_dispatch` the moment the closing tag is
  parsed, carrying the parsed call, while decode continues.
- Request extension `baro`: `priority`, `expected_output_tokens`,
  `speculative_prefill`, `cache_salt`, `retention_priority`.
- Receipts in `usage.baro`: prefill_ms, cached_tokens, decode_ms,
  acceptance, checkpoint_hit, prefix_churn.
- Conformance corpus `tools/chat-corpus/` (new): parallel calls, optional
  params, nested JSON, calls after thinking, `</think>` mid-chunk; runner
  `tools/test_chat_corpus.py`; add to `tools/test_server.sh`.

**Preregister.**
- P-H1 corpus 100 % on all three APIs.
- P-H2 DeerFlow smoke exit 0, call 2 prefill < 200 tokens; the
  `exceed_context_size` 400 is exercised by one oversize request and
  DeerFlow's reaction recorded.
- P-H3 Claude Code (Anthropic endpoint, `ANTHROPIC_BASE_URL`) and OpenClaw
  each complete one write-file task; `tool_call_dispatch` timestamp precedes
  stream end in the tap.
- P-H4 engine identity and A/B unchanged (no engine-side change except
  receipts).

### 5.6 M5 — K8/V4 and K4/V4 KV, RULER gate (~120 LOC, K)

**Facts from M0b.** f16 and bf16 KV both broke identity (17/20). The
narrow formats here are for long context only; the bf16/f32 mode stays for
bit-exact gates. `KVT` and `kv_off` are the only seams.

**Files.**
- `kernels/attn.mojo`: `KVQ` comptime mode {f32, k8v4, k4v4}; K stored int8
  with per-(token, head) f16 scale, V int4 with per-(token, head, 64-group)
  scale; dequant at the point of use in `attn_head_body` and
  `amar_attn_prefill`; `amar_kv_append` quantises. 1664 B/token/layer at K8/V4.
- `serve/engine.mojo`: `BARO_KV=f32|k8v4|k4v4`, printed as `kv dtype:`.
- RULER subset generated locally: `tools/ruler-gen.py` (NIAH single / multi-
  key, variable tracking, common-words aggregation) at 8k / 32k / 64k /
  128k; runner `tools/ruler-run.py` through the OpenAI endpoint. Lane
  `ruler` already has a llama.cpp baseline harness under
  `~/Projects/mojo-baro-lanes/ruler/bench/ruler/`; reuse its task generator
  rather than writing a second one.

**Preregister.**
- P-I1 f32 mode bit-exact vs M4 binary; fingerprint in class.
- P-I2 effective length (largest T with score ≥ the 4k baseline threshold):
  K8/V4 = f32 at every tested length; K4/V4 recorded, ships only if equal.
- P-I3 decode at T=32k: K8/V4 faster than f32 by the KV byte ratio
  (report, band ±20 % of the byte-ratio prediction).
- Falsifier: K8/V4 effective length below f32 at any length → K8/V4 does not
  ship, lane reports f32 only.

### 5.7 M6 — guided decoding with grammar rollback (~60 LOC glue)

**Depends on** lane `grammar` landing `grammar/` (JSON-schema / regex → byte
PDA with mask / accept / snapshot / rollback; `docs/grammar.md`). If not
landed on `main` when M6 starts, write the question to the status file and
finish M5's report instead.

**Files.**
- `serve/engine.mojo` overlap loop (M2): compute the token mask for step n on
  the CPU while the GPU runs step n−1; upload as a bitset; `kernels/sample.mojo`
  applies it before the chain (masked logits = −inf).
- Speculation: advance grammar per draft token, snapshot before the window,
  roll back to the last accepted (design §6); `</think>` and the first
  post-marker tokens enter the FSM exactly once (vLLM #44993 — write that
  test).
- `tool_choice` forcing a tool → that tool's JSON schema is the grammar.

**Preregister.**
- P-J1 schema corpus (`grammar/corpus`) valid output 100 % with spec on and
  off; outputs identical between the two under temperature 0.
- P-J2 the `</think>` boundary test passes under spec on.
- P-J3 20-prompt A/B with grammar off unchanged.

---

## 6. Deliverable formats

### 6.1 `.work/briefs/status-chat.md` — append-only, one block per stint

```
## <date time> <milestone> <state: PLAN|BUILD|GATE|DONE|BLOCKED>
- what changed (files)
- receipts (numbers with their source file)
- next
## QUESTIONS   (only when blocked; the driver answers in this file)
```

### 6.2 `bench/chat-protocol.md` — one section per milestone

Change · Not in this step · Predictions (P-xN + rationale + falsifier) ·
Verification before timing (which printed lines are the receipts) · Gate ·
Result (table: receipt | ref | new | source; verdict per prediction;
unpredicted observations marked "observation, not promoted").

### 6.3 `exchange/lane-chat-report.md` — written once at lane end

Sections: Landed (commit per milestone with its headline numbers) · Gates
(§4.3 table, each with the file that proves it) · Not landed and why ·
Falsified along the way (so nothing is retried) · Open questions. Last line
`DONE` or `BLOCKED: <one line>`.

### 6.4 Commit message shape

```
<area>: <what> (<milestone>) -- <headline number>

<why, 2-6 lines: the prediction it satisfies, the receipt files,
anything falsified>
```

Areas: `kernels`, `serve`, `bench`, `tools`, `docs`. One concern per commit;
the preregistration is its own `bench:` commit before the build.
