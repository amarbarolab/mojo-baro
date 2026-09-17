# P1 state API: design spec

State already moves between requests on one node; it cannot move between nodes, because the pack
salt is the hash of a directory path. P1 fixes that one identity rule, wraps the existing state file
in the LAT1 header, and puts three routes in front of it. It adds no kernel and does not touch
`serve/window.mojo`.

Status: DESIGN, frozen for the build 2026-09-17 (coordinator). Item P1 of `docs/PLATFORM-PLAN.md`.
Builders: team A (sonnet builds, codex gates). A change to anything marked CONTRACT goes through the
coordinator, because P0b's rank term, P4's gate 2 and P6's clients are written against it.

## What exists and is reused

- The engine wire carries per-request `state_save` and `state_load` paths (`serve/src/protocol.rs`,
  `serve/engine.mojo` `save_state` / `load_state`). A state file is `BAROST01` (f32) or `BAROST02`
  (int8 pages with per-block scales, `BARO_STATE_INT8=1`): header (pos, conv_n, ssm_n, kv_n), pack
  salt, the prompt tokens, conv and SSM slots, K and V pages below `pos`. It restores a prefix
  byte-exactly (C2-mini gate, `bench/checkpoint-api.sh`).
- The prefix checkpoint chain (`serve/prefix.mojo`, `Chain`): salted SHA-256 of `tokens[0:pos]`,
  `lookup`, `save`, `restore`, pinned and role-boundary retention.
- The LAT1 header (`latentos/proto.mojo`, 256 bytes): kind, dtype, `weights_uuid`, `tokenizer_sha`,
  `runtime`, `pos_lo/pos_hi`, `prefix_hash` (u64), `payload_len`, `payload_sha`, `hmac`.
- `POST /v1/fork` (branches on one node), the E12/E14/E15 harnesses, `bench/b4-cross-host.sh`.

## CONTRACT 1: the container (`.baro` state stream)

```
LAT1 header (256 B, kind = KV_PAGES, dtype = f32 or the new DTYPE_I8_BLOCK = 3)
BAROST01 or BAROST02 body, unchanged
```

- `pos_lo = 0`, `pos_hi = pos`, `prefix_hash` = first 8 bytes (little endian) of the UNSALTED
  SHA-256 of the int32 token bytes `tokens[0:pos]`, `payload_len` and `payload_sha` over the body,
  `weights_uuid` and `tokenizer_sha` from the bake's identity block, `runtime` = sha256 of the engine
  build string, `hmac` zero until P0b pairing supplies a key.
- One stream carries one prefix. The SSM checkpoint travels inside the BAROST body as today; `kinds`
  accepts only `kv_pages` and answers 400 for the rest, naming P1's scope.
- Media type `application/vnd.baro.state`. Streams are written and read in 8 MiB chunks; no route
  buffers a whole 32k state in the HTTP layer.

## CONTRACT 2: identity (amended 2026-09-17 after team A's finding)

The pack salt today is `sha256(packdir)`. Two nodes holding the same pack under different paths
refuse each other's state ("saved from a different pack"). The first draft of this spec keyed the
fix on `weights_uuid` and the vocab-and-merges `tokenizer_sha` of `serve/identity.mojo`. Team A
showed that nothing at serve time can produce either: the pack directory holds no GGUF, that module
is called only from a bench, and `serve/src/checkpoints.rs` hashes the tokenizer FILE, a different
algorithm. The identity is therefore the thing the engine actually runs:

- `pack_sha256` = sha256 of `pack.bin`, all 32 bytes. A q4 and a q8 pack of one GGUF differ here,
  which is correct: their KV is not interchangeable.
- `tokenizer_sha256` = sha256 of the pack's `tokenizer.json` file bytes, the algorithm
  `checkpoints.rs` already uses. `serve/identity.mojo` stays a bench module and is not used by P1.
- Both live in `<pack>/identity.json`: `{"pack_sha256","tokenizer_sha256","pack_bytes","pack_mtime_ns",
  "source":NAME,"general_uuid":STR or null}`. `tools/engine-pack.py` writes it at pack time, and
  `tools/engine-pack.py --identity PACKDIR` writes it for an existing pack (one 7 GB hash, reused
  while `pack_bytes` and `pack_mtime_ns` still match). One writer, in one language; the engine and
  `baro-serve` only READ the file, so no hash logic is duplicated.
- Salt = `sha256(pack_sha256 || tokenizer_sha256)` when `identity.json` is present and its size and
  mtime fields match `pack.bin`; else `sha256(packdir)` as today, and `GET /v1/state` reports
  `"portable":false`, and export answers 409 `{"error":"pack_identity_missing","fix":"engine-pack.py --identity"}`.
  State files written under the old salt are refused with a message that says so; they are caches.
- LAT1 header mapping: `role_sha` (32 B, "plain artifact sha256") = `pack_sha256`; `weights_uuid`
  (16 B) = its first 16 bytes; `tokenizer_sha` = `tokenizer_sha256`.
- `checkpoints.rs` `Identity`: `pack` becomes `pack_sha256` from the file (path hash when absent),
  `weights_uuid` stays null, `tokenizer_sha` unchanged. It sits beside the current field, it does
  not change its algorithm.
- Import checks, in order: LAT1 magic and version; `payload_sha`; `role_sha`; `tokenizer_sha`; slot
  sizes and `pos` against `BARO_TMAX`. Any miss is `409 {"error":"state_identity","field":...,
  "ours":...,"theirs":...}` and nothing is restored. A differing `runtime` is reported
  (`"runtime_differs":true`), not refused: the identity gate judges it.

## CONTRACT 3: routes on `baro-serve`

| route | request | response |
|---|---|---|
| `GET /v1/state` | none | `{"model":ID,"weights_uuid":HEX,"kv":"f32" or "int8","states":[{"prefix_hash":HEX16,"pos":N,"bytes":N,"pinned":B,"boundary":B,"age_s":N}]}` from the chain, no GPU work |
| `POST /v1/state/export` | `{"prompt" or "messages" or "tokens", "pos":N?, "format":"f32" or "int8" (default int8), "path":STR?}` | the stream, or `{"path","bytes","pos","prefix_hash"}` when `path` is given |
| `POST /v1/state/import` | the stream as body, or `{"path":STR}` | `{"prefix_hash","pos","restore_ms","runtime_differs"}` |
| `POST /v1/fork` | today's body plus optional `"target":"NODE_ID"` | with `target`, baro-serve answers 501 `{"error":"fork_target_needs_router"}`: cross-node fork is the router's job (P0b), built on export and import |
| `POST /v1/fanout` | `{"prompt" or "messages", "followers":[{max_tokens, sampler fields}, ...]}` | E14's shape on one node: one prefill, N continuations, an array in follower order |

- Export of a prefix that is not resident runs the prefill first (a request with `n = 0` and
  `state_save`), so export is always answerable. `pos` defaults to the last role boundary at or
  before the prompt end, else the prompt end minus one.
- Import lands the state as a chain checkpoint pinned for a TTL (the header's `ttl_s`, default
  600 s), so the next request whose tokens start with that prefix restores it through the ordinary
  `lookup` path. Import does not start a generation.
- An engine running `BARO_KVQ=int8` answers export and import with 501
  `{"error":"kvq_state_open","see":"A2 step 3"}`. `GET /v1/state` still works.
- Export and import are ordinary queued requests on the wire (`n = 0`), never a side channel into a
  running generation.

## CONTRACT 4: what the router reads (P0b)

`/v1/node-info` carries, per engine, the `GET /v1/state` object verbatim. The locality term: a
request whose unsalted `prefix_hash` at any role boundary matches a resident state on engine E gets
E's rank improved by the equivalent of one pending job. The router computes the hash from the
rendered prompt through `POST /tokenize` and the rule in CONTRACT 1. Nothing else about state is
visible to the router.

## Out of scope for P1

HIDDEN and LOGITS_TOPK streams over HTTP, the Unix-socket IPC sidecar, HIP IPC handles, delta
states, hmac enforcement, MoE packs (MoE export answers 501 until measured).

## Build order (each step lands with its gate, in the team's item template)

1. **Salt and container.** `identity.json` writer in `tools/engine-pack.py`, the salt read from it, LAT1 wrap and unwrap, the 409 paths. Gate:
   `bench/checkpoint-api.sh` still green; a state saved under pack path A loads under a symlinked
   path B (show it failing before the change); a flipped body byte is a 409.
2. **Routes.** `GET /v1/state`, export, import, in a NEW `serve/src/state.rs`; `main.rs` lines in one
   window. Plan gate 1: export then import on one node reproduces the 20-prompt identity and the
   restore band (2.2 to 4.3 ms), both formats, int8 bytes at most 0.30 of f32.
3. **Cross-node rig.** Two `baro-serve` processes cannot share the XTX (22 GB each), so the rig is
   sequential on one GPU: serve A exports to a file and exits, the file crosses the veth link of
   `bench/b4-cross-host.sh` at 100 Mbit, 1 Gbit and 10 Gbit, serve B (other pack path, same bake)
   imports and continues. Plan gate 2: ids equal the single-node ids; payload arm against recipe arm
   at each rate, prediction frozen in a NEW protocol note, `p1-state-protocol.md` in the bench folder, before the run.
4. **Fanout.** `POST /v1/fanout` on one node. Plan gate 3: N=3 identity 3/3, N=10 at least 9/10 with
   the E14 discordance named, tok/s within 5% of the E14 receipt.
5. **llama.cpp bridge, forward direction** (LAT1 KV to a llama.cpp slot file, Mojo, beside
   `tools/llama-slot-to-state.mojo`). Plan gate 4: E15's three models continue with the first 32
   tokens identical. May be split off as P1b if steps 1 to 4 fill the M budget; say so in the
   report, never drop it silently.

Kill line (plan): an identity miss outside the documented E14 one, or a cross-node move slower than
re-prefill at 1 Gbit for 32k. GPU: about 30 minutes, one resident engine per step, every job
through `gpu-wait run --timeout`.

## Check these first, not last

- The chain holds 8 checkpoints. Imports compete with the request's own grid checkpoints; the TTL
  pin must not starve them. Measure eviction on a 32k prompt after one import.
- `save_state` needs a committed checkpoint AT `pos` and raises without one. Export therefore passes
  `pos` as a `ckpt` hint on its prefill request.
- The unsalted `prefix_hash` is 64 bits and visible on the LAN. It is a routing hint only: the
  restore still requires the full salted SHA-256, so a collision costs a misroute, never a wrong
  restore.
