# Lane server — report (2026-09-06)

Worktree `$HOME/Projects/mojo-baro-lanes/server`, branch `lane-server`, base `751bc3c`.
Plan item **server** of `~/Brain/mojo-baro/briefs/2026-09-06-engine-lanes.md`. All three sub-steps landed.

## What landed

| commit | content |
|---|---|
| `d939a05` | `serve/engine.mojo`: `main()` = init once (pack load + every buffer) + request loop. `BARO_SERVE=0` (default) is the unchanged one-shot path. `BARO_SERVE=1`: prints a `{"ready":...}` line after init, then reads `{"id","prompt":[ids],"n","spec"}` JSON lines from stdin until EOF; per request zeroes conv/SSM rings, both KV caches, mega counters; streams `{"id","tok"}` per token, then `{"id","done":true,"n","prefill_s","decode_s","tok_s"[,"drafted","accepted","k"]}`; malformed requests get `{"id","error"}` and the loop continues. TMAX untouched. |
| `106b250` | `serve/Cargo.toml`, `serve/Cargo.lock`, `serve/src/{main,engine,protocol,text}.rs` (`baro-serve`: axum + tokio + tokenizers(onig) + minijinja/pycompat), `serve/PROTOCOL.md` (stdin/stdout contract, HTTP surface, tokenizer file contract, C-ABI seam), `tools/test_server.sh`. |

Server: spawns `.work/engine` with `BARO_SERVE=1` + env passthrough, one worker task drives stdin/stdout so requests queue (cap 64 → 503). Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions` (prompt = text or token-id array; response adds `choices[0].tokens` + `timings`), `POST /v1/chat/completions` (template from `tokenizer-meta.json` `chat_template` via minijinja+pycompat, ChatML fallback), SSE on `stream:true`, `POST /tokenize`, `POST /detokenize`. `--tokenizer PATH` override; no `tokenizer.json` → server runs, text endpoints 503. Engine stdout parsing (`protocol.rs`) type-checks every field; anything else is a stderr log line, never an event (9 unit tests incl. truncated / wrong-typed / out-of-range lines).

Usage: `BARO_PACK=.work/engine-pack-q4 serve/target/release/baro-serve --engine .work/engine --port 8080` (build: `cd serve && cargo build --release`).

## Gate

Command: `gpu-wait run --priority 60 --timeout 1800 -- .work/lane/full-gate.sh` → `.work/server-gate.txt` (worktree). Contents:

```
== gate 2026-09-06T10:19:51+02:00 commit 106b250
run-tests.sh exit 0 : GEMM OK — 4 x 3 @ 3 x 2 matches host reference;54 kernels, 26 in registry, 0 orphans;
ci-checks.sh exit 1 :   FAIL docs/KERNELS.md is stale; regenerate with tools/kernel-census.py;1 check(s) failed;
one-shot q4 default exit 0 : PASS: 64 tokens match tok/s_gen: 118.18090769427681
one-shot q8 exit 0 : PASS: 64 tokens match tok/s_gen: 80.03838559664379
mtp-prompts.sh k=2 exit 0 : identity PASS 20/20 FAIL 0
test_server.sh exit 0 :
PASS clippy: no warnings
PASS cargo-test: 9 passed
PASS build: engine + baro-serve
PASS start: http://127.0.0.1:44277 (pack loaded in 0.463876197 s)
PASS health: {"limits":{"kmax":8,"mrows":8,"spec_k":2,"tmax":128},"pack":".work/engine-pack-q4","queue":0,"status":"ok","tokenizer":true}
PASS models: engine-pack-q4
PASS completion: PASS: 64 tokens match; tok/s_gen 130.8 prefill_s 0.0354
PASS stream: PASS: 64 tokens match
PASS queue: 2 concurrent requests both match ref; /health during: queue 2
PASS tokenize: detokenize/tokenize roundtrip on the receipt prompt: 'The capital of France is'
PASS completion-text: PASS: 64 tokens match; text ' Paris.\nThe capital of France is Paris.\n'
PASS chat: length '1.  **Analyze the Request:** The user simply said "Say hello'
PASS chat-stream: 16 delta chunks; streamed text == non-streamed text
PASS reject: 400 prompt (5) + max_tokens (100000) exceeds the engine's context of 128 tokens
PASS shutdown: server exit 0, engine child exited
ALL PASS
== end 2026-09-06T10:21:29+02:00
```

Sub-step receipts (all in the worktree):
- Engine step (`.work/lane/gate1.log`, commit `d939a05`): one-shot q4 `GENERATED` == `ref-tokens-64.txt` for spec∈{0,1} × mega∈{0,1}; serve mode 3 back-to-back requests (no-spec, spec, no-spec) each == ref; 4 malformed requests each answered with an error line, engine kept running, clean exit 0 at stdin EOF.
- Server step (`.work/server-test/SUMMARY.txt`, `tools/test_server.sh`): clippy `-D warnings` clean, `cargo test` 9/9, `/health`, completion by ids == ref, SSE stream == ref (64 chunks + finish + `[DONE]`), 2 concurrent requests both == ref with `/health` reporting `queue 2` mid-run, 400 on prompt+max_tokens > TMAX, SIGINT → server exit 0 and engine child gone. With the tokenizer lane's `tokenizer.json`/`tokenizer-meta.json` (appeared in `.work/engine-pack-q4` at 10:10): `tokenize(detokenize(prompt)) == prompt` (`'The capital of France is'`), string-prompt completion == ref (text `' Paris.\nThe capital of France is Paris.\n'`), chat completion through the real Qwen Jinja template renders and streams (streamed text == non-streamed).

## Numbers (instrument receipts, single prompt = the 5-token receipt prompt; not P4 claims)

| what | value | command / file |
|---|---|---|
| one-shot q4 no-spec mega tok/s_gen | 131.4 | `BARO_PACK=.work/engine-pack-q4 ./.work/engine` → `.work/lane/gate1/q4.spec0.mega1.log` |
| serve-mode same request (per-token stream sync) | 130.8 / 130.6 | `.work/lane/gate1.log` request a; `tools/test_server.sh` completion step |
| serve-mode spec k=2 | 189.9 (drafted 43, accepted 42) | `.work/lane/gate1.log` request b |
| pack load (q4, page-cached) | 0.44 s | `.work/server-test/server.stderr` |
| engine ready → first HTTP request possible | ~0.7 s after spawn | `.work/lane/gate1.log` |

Streaming cost on the non-spec path: one host copy + synchronize per token, ≈0.5% tok/s_gen on this prompt. Spec mode reuses the accept-window copies, no extra sync.

Engine-side parameter read-back (P1): the `ready` line echoes `tmax/mrows/kmax/spec_k/pack` from the running binary; each request logs `prompt tokens: N  n: N  spec: bool`; the engine's usual `BARO_*:` lines stay on stdout (forwarded to server stderr).

Gate verdict: every check green except `ci-checks.sh`, whose single failure is the pre-existing stale `docs/KERNELS.md` (see notes). `bench/mtp-prompts.sh .work/engine .work/lane/mtp 2` → 20/20 spec==no-spec identity on the q4 pack (`.work/lane/mtp/results.txt`). The one-shot q4 tok/s_gen varied 118–131 across runs today while the prefill lane shared the card through the queue; it is an identity receipt here, not a throughput claim (P4).

## What is left / notes

- Engine generates exactly `n` tokens; no stop-token or cancel in the protocol. The server trims text at a stop id (`finish_reason: "stop"`) but the GPU still runs to `n` (documented in PROTOCOL.md; adding cancel = a `{"cancel":id}` line + a per-window check, not done in this lane).
- `docs/KERNELS.md` is stale on the base commit (`tools/ci-checks.sh` FAIL "docs/KERNELS.md is stale", diff = `test_mega_block.mojo` usage rows); pre-existing on `751bc3c`, outside this lane's ownership, not touched. Everything else in ci-checks passes.
- `.work/engine-pack` (q8 ref-tokens) was not symlinked into the worktree by setup; I added the symlink locally (read-only input) so the q8 one-shot identity could run.
- Chat template: the real `tokenizer-meta.json` template is a Qwen VL-style Jinja (namespace, `.strip()` etc.); it renders under minijinja+pycompat. Not compared byte-for-byte against llama.cpp's rendering (tokenizer lane's `--chat` CLI is the reference once it exists).
- `serve/target/` is gitignored; `Cargo.lock` committed.

## Questions

- None blocking. Meta-file key names were assumed before the tokenizer lane wrote them; they matched (`eos_token_id`, `bos_token_id: null`, `eos_token`, `add_bos`, `chat_template`).
