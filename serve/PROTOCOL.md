# Engine <-> server protocol

`serve/engine.mojo` (qwen35, qwen35moe) and `serve/spark.mojo` (the dense
families: llama, qwen2, granite, spark2_5) are both single-request decoders
that speak this same line protocol under `BARO_SERVE=1`, sharing the
byte-scanner reader and request parser in `serve/serve_proto.mojo` rather
than each defining their own wire format. `serve/src` (`baro-serve`, Rust)
keeps one engine process alive (either binary, selected with `--engine`) and
turns HTTP requests into engine requests one at a time. This file is the
contract between the two, and the seam where the process boundary is later
replaced by a C ABI. Not every optional field is acted on by every engine:
spark has no draft head (`spec` is parsed and ignored, `spec_k` in its
`ready` line is always `0`) and no prefix-checkpoint chain (`ckpt` is parsed
and ignored); `stop` and the required id/prompt/n fields are honored by
both.

## Process model

```
client --HTTP--> baro-serve --stdin JSON lines--> engine (BARO_SERVE=1)
                            <--stdout JSON lines--
```

- The server spawns `.work/engine` (`--engine`) with `BARO_SERVE=1` and
  `BARO_PACK=<pack>` (`--pack`), inheriting its own environment, so every
  other `BARO_*` knob (`BARO_SPEC`, `BARO_SPEC_K`, `BARO_MEGA`,
  `BARO_DRAFT_Q4`, ...) still reaches the engine.
- The engine loads the pack and allocates every buffer **once**, prints the
  ready line, then blocks on stdin.
- Requests are strictly serial: the server's worker writes one request line,
  reads until that request's terminal line, then writes the next. HTTP
  requests wait in a bounded queue (64) and get `503` beyond it.
- Stdin EOF ends the engine: the request loop exits and the process
  returns 0. That is the clean shutdown; the server closes stdin on
  SIGINT and kills the child only if it has not exited after 30 s.
- `BARO_SERVE=0` (default) is the unchanged one-shot path: prompt from
  `BARO_PROMPT`/`<pack>/prompt-tokens.txt`, `GEN_N` tokens, `GENERATED:`
  line, draft receipt, exit. Every existing gate runs that path.

## Lines the engine prints (stdout)

One JSON object per line, no embedded newlines. Any line that does not
start with `{`, is not a JSON object, or fails the field checks below is a
**log line** and is forwarded to the server's stderr — never acted on.
The engine keeps printing its usual diagnostics (`pack loaded in`,
`tok/s_gen:`, `GENERATED:`, ...) between the JSON lines.

| line | when | fields |
|---|---|---|
| `{"ready":true,"tmax":128,"mrows":8,"kmax":8,"spec_k":2,"pack":"..."}` | once, after init | `tmax` = KV/token capacity: `len(prompt)+n <= tmax` or the request is rejected. `mrows` = prefill chunk, `kmax` = max draft width, `spec_k` = the draft width in effect (`spec-k.txt` / `BARO_SPEC_K`). |
| `{"id":ID,"tok":T}` | per generated token, in order | `T` is the greedy token id at the next position. In spec mode a verified window emits its accepted tokens plus the correction token together. |
| `{"id":ID,"done":true,"n":N,"prefill_s":..,"decode_s":..,"tok_s":..,"finish":"length"\|"stop"\|"cancelled"}` | after the last token | `n` = tokens actually generated (`<=` request `n`: less than requested when `finish` is `"stop"` or `"cancelled"`), `tok_s` = `(n-1)/decode_s` (0 when n<=1), matching the one-shot `tok/s_gen`. With spec: `"drafted"`, `"accepted"`, `"k"`. |
| `{"id":ID,"error":"..."}` | instead of tokens | request rejected before any GPU work: bad JSON, missing/negative fields, empty prompt, `n < 1`, `len(prompt)+n > tmax`. The loop continues. |

`id` is echoed from the request. The engine parses `id` first, so an error
for an unparseable line carries `"id":0`.

## Lines the server writes (stdin)

```
{"id":ID,"prompt":[INT,...],"n":N,"spec":BOOL,"stop":[[INT,...],...],"ckpt":[INT,...]}\n
{"cancel":ID}\n
```

- `id`: unsigned integer, assigned by the server (monotonic, starts at 1).
- `prompt`: non-empty list of non-negative token ids, in order.
- `n`: tokens to generate, `>= 1` (an upper bound: `stop` or `cancel` can end
  generation sooner, see below).
- `spec`: optional; `true`/`false` selects MTP speculative decode for this
  request. Absent => the engine's `BARO_SPEC` default.
- `stop`: optional, default `[]`. A list of token-id sequences; once the
  tail of the tokens generated so far equals any of them, the engine stops
  (that sequence's tokens are still emitted as `tok` lines and counted in
  `n`) and reports `"finish":"stop"`. Checked once per decode window (not
  mid-window), and only synced back from the GPU when `stop` is non-empty --
  a request with no `stop` pays nothing for this. `serve/src/main.rs` builds
  `stop` from the tokenizer's own EOS-like ids (each a length-1 sequence)
  plus any caller `stop` string, tokenized.
- `ckpt`: optional, default `[]` (M1b role-boundary checkpoints). Token
  positions where `serve/prefix.mojo` should also take a prefix checkpoint,
  in addition to the periodic 1024-token grid and the prompt-end point.
  `serve/src/main.rs` sends, for a chat request, the token length after
  each message when the conversation so far is rendered with no generation
  prompt (`Text::role_boundaries`) -- a later turn's full prompt starts
  with exactly those same bytes, since history does not change, so these
  positions are restore points for it. Index 0 (by convention the system
  prompt) is saved pinned against eviction; every hint is saved as a
  role-boundary checkpoint, evicted only after every periodic-grid one. A
  hint that misses a real tokenization boundary wastes a checkpoint slot,
  never corrupts a restore -- `lookup` still requires the stored SHA-256
  hash (salted per pack, so two packs never share a checkpoint) to match.
- `temperature`/`top_p`/`top_k`/`min_p`/`seed`/`presence_penalty`/
  `frequency_penalty`: optional (C3 control block). Parsed into
  `SampleParams` and **not yet acted on** -- the decode loop still always
  takes the greedy/MTP path regardless of these. `serve/sample_ref.mojo`
  is the host reference these will drive once wired (matched byte-for-byte
  against `kernels/sample.mojo`'s `amar_sample_row`/`amar_spec_accept`,
  lane-KSAMP); absent, the request line is identical to before this field
  existed.
- `{"cancel":ID}`: a second line shape, written to the same stdin at any
  point while `ID` is decoding (the request line for the *next* id is never
  written before this one's `done` line, so a stray line mid-request can
  only be this). The engine polls for it once per decode window (`poll(0,
  POLLIN, timeout=0)`, never blocking, never touching the GPU queue); a
  match breaks the loop and the `done` line reports `"finish":"cancelled"`
  with `n` = tokens actually generated. A cancel for a request that is not
  the one currently decoding (already finished, or still queued on the
  server side) is silently dropped -- cancelling a queued request is not
  supported.

The engine's parser is a byte scanner, not a JSON library: keys must be
double-quoted, values must be plain integers / `true` / `false` / arrays of
the shapes above, and no other structure is interpreted. Extra keys are
ignored.

## Per-request state

Before each request the engine zeroes the conv/SSM slot rings, both KV
caches (trunk and draft) and the megakernel counters, and resets the
position to 0, so the request runs exactly as the one-shot path would.
Verified: three back-to-back requests (no-spec, spec, no-spec) on the q4
pack each reproduce `ref-tokens-64.txt` (`tools/test_server.sh`,
`.work/lane/gate1.log`).

Streaming costs one host copy + synchronize per generated token on the
non-spec path (spec mode reuses the accept-window copies it already
makes): about 0.5% of tok/s_gen on the 5-token receipt prompt.

## HTTP surface (`baro-serve`)

| route | body | notes |
|---|---|---|
| `GET /health` | | `status`, `queue` (waiting+running), `tokenizer` (bool), `limits` (the ready line), `pack` |
| `GET /v1/models` | | one model, id = pack directory name |
| `POST /v1/completions` | `prompt` (string, or array of token ids), `max_tokens` (default 64), `stream`, `spec` (extension), `stop` (string or array of strings), `temperature`/`top_p`/`top_k`/`min_p`/`seed`/`presence_penalty`/`frequency_penalty`/`logprobs` (C3, parsed and carried, not yet acted on) | response adds `choices[0].tokens` (the generated ids) and `timings` (the done line, including `finish`). Token-id prompts need no tokenizer; `stop` needs one (silently `[]` without). |
| `POST /v1/chat/completions` | `messages`, `max_tokens`/`max_completion_tokens`, `stream`, `spec`, `stop`, the same C3 sampler fields | needs the tokenizer; template from `tokenizer-meta.json` (`chat_template`, Jinja via minijinja + pycompat) else ChatML |
| `POST /v1/cancel` | `{"id": "cmpl-7"}` / `{"id": "chatcmpl-7"}` (the response `id`, or the SSE `id` field of its first chunk -- read while the request is still streaming) | `{"cancelled": bool}`; `true` only if that request was the one actively decoding. A queued-but-not-started or already-finished id returns `false`. |
| `POST /tokenize` | `{"content": "...", "add_special": false}` | `{"tokens": [...]}` |
| `POST /detokenize` | `{"tokens": [...]}` | `{"content": "..."}` |

`stream: true` returns SSE: one `data:` chunk per token (`text` delta and
`tokens: [id]`), a final chunk with `finish_reason` + `usage` + `timings`
(+ the full `tokens` list on completions), then `data: [DONE]`.

Errors are `{"error": {"message", "type", "code"}}` with the HTTP status:
`400` bad request (empty prompt, over-length, bad JSON), `503` no
tokenizer / queue full / engine gone, `502` engine error mid-request.

### Tokenizer files (owned by the tokenizer lane)

`<pack>/tokenizer.json` (HF `tokenizers` format; loaded with the Rust
`tokenizers` crate, oniguruma regex backend) and, optionally,
`<pack>/tokenizer-meta.json` with `chat_template` (string),
`bos_token_id`/`eos_token_id` (or `bos_id`/`eos_id`), `bos_token`/
`eos_token` (strings), `add_bos` (bool). Stop ids = `eos_token_id` plus
`<|im_end|>` / `<|endoftext|>` when the vocabulary has them. Without
`tokenizer.json` the server starts anyway and the text endpoints return
`503`; `--tokenizer PATH` points at a file elsewhere.

## The C-ABI seam

Everything above the engine is written against three operations, and the
process boundary is the only thing the line protocol adds:

| stdin/stdout today | C ABI later |
|---|---|
| spawn + `ready` line | `baro_engine_t* baro_open(const char* pack)` returning the same limits struct |
| request line | `int baro_generate(baro_engine_t*, const uint32_t* prompt, size_t n_prompt, uint32_t n, bool spec, baro_token_cb cb, void* user)` — `cb(user, token)` per token, return = done stats or a negative error code |
| stdin EOF | `void baro_close(baro_engine_t*)` |

`serve/src/engine.rs` is the only file that knows about the child
process; `Engine::submit` / `Event` are the interface the handlers use and
would be re-implemented over the ABI (one worker task still serialises
calls, since the engine stays single-request). `serve/src/protocol.rs`
(the line parser and its tests) is the part that disappears. Rust must
bind Mojo's exported C symbols only — never the C++ hipBLASLt shim
(`docs/BASELINE.md`, "Layers").

## Verifying

```
tools/test_server.sh [OUTDIR]      # builds, then: clippy, cargo test, /health,
                                   # completion == ref-tokens-64, SSE == ref,
                                   # 2 queued == ref, 400 on over-length, SIGINT shutdown
```

Runs under `gpu-wait` by itself. Receipts land in `OUTDIR/SUMMARY.txt`
(default `.work/server-test/`).
