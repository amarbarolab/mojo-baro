# Team A P1 report

Date: 2026-09-17
Branch: `lane-team-a`. Status: build order steps 1-2 (`docs/P1-STATE-API.md`)
landed and live-smoked on one node; plan gate 1's full 20-prompt sweep across
both KV formats not yet run (see Kill lines and next action). Steps 3-5
(cross-node rig, fanout, llama.cpp bridge) not started.

## Landed

- `d4e1983` + `57afcbe` add `tools/engine-pack.py --identity`, writing
  `<pack>/identity.json` (`pack_sha256`, `tokenizer_sha256`, `pack_bytes`,
  `pack_mtime_ns`) per CONTRACT 2's amended shape.
- `a781915` wires the salt read: `serve/prefix.mojo::pack_identity_salt`
  reads `identity.json` when present and its size/mtime still match
  `pack.bin`, else falls back to `sha256(packdir)` as before (`GET /v1/state`
  reports `"portable":false` in the fallback case). `serve/src/checkpoints.rs`
  gained `Identity.portable` and `weights_uuid_hex()`.
- `6d0d27f` adds `exact_prefix_hash` (CONTRACT 1's unsalted `sha256(tokens[0:pos])`
  as little-endian u64), independently verified against a Python `hashlib`
  vector, not a fabricated one.
- `1d5648e` adds `GET /v1/state` (CONTRACT 3), read-only from the
  HTTP-facing `checkpoints::Registry` -- no GPU work.
- `067493e` adds `POST /v1/state/export`, `path` variant. `map_kvq_refusal`
  translates the engine's own `BARO_KVQ` state refusal into the documented
  501 `kvq_state_open`; this is shared by every later export/import call
  site, path or stream.
- `34c765b` adds `POST /v1/state/import`, `path` variant.
- `0df9230` fixes a real bug the round-trip smoke found: `Chain::lookup`
  (`serve/prefix.mojo:337`) only accepts a checkpoint at `pos <= n - 1` (it
  always reserves the request prompt's last token as the live decode seed,
  confirmed against `checkpoints::create`/`fork`, which always submit
  `pos + 1` tokens). Import's `path` form built its engine request from
  exactly the file's `pos` embedded tokens, so `n` was always `pos` and the
  just-loaded checkpoint could never clear the bar -- 502 on every file,
  regardless of match. Fixed by repeating the file's own last token once
  (`n = pos + 1`); `exact_prefix_hash` and the checkpoint's own registered
  hash both only read `tokens[0:pos]`, so this changes nothing that is
  checked. `bench/p1-state-roundtrip-smoke.sh` replaces the export-only
  smoke.
- `e7fd602` adds CONTRACT 1's LAT1 container (`serve/src/state.rs::lat1`,
  byte-for-byte against `latentos/proto.mojo::LatentHeader`'s own offsets)
  and wires it into both routes' "no path" form:
  - `export`'s request shape is identical with or without `path`; only the
    response differs, so `export()` now dispatches on `r.path`. The no-path
    form prefills to an internal scratch path, hashes the file in one
    streamed pass (`payload_sha`/`payload_len`), then streams header + file
    in a second pass, 8 MiB chunks, via a spawned task feeding an `mpsc`
    channel into `axum::body::Body::from_stream` (no new streaming
    dependency -- reuses `tokio-stream`'s `ReceiverStream`, already a
    dependency; `tokio`'s `fs` feature was added).
  - `import`'s request shape genuinely differs (JSON `{"path"}` vs the raw
    stream as body), so `import()` now takes the whole `Request` and
    branches on `Content-Type`. The stream form accumulates exactly 256
    bytes for the header regardless of HTTP chunk boundaries, writes the
    remainder to a scratch file while hashing incrementally, then runs
    CONTRACT 2's ordered identity checks (magic/version, kind, dtype,
    `payload_sha`, `role_sha`, `tokenizer_sha`, `pos` vs `BARO_TMAX`)
    *before* ever calling `state_load`. Any miss removes the scratch file
    and refuses via a new `ApiError::StateIdentity` 409
    `{"error":"state_identity","field","ours","theirs"}` -- CONTRACT 2's own
    shape, not this repo's usual `{"error":{"message",...}}` envelope. This
    touched the shared `main.rs` enum; the room's main.rs window was taken
    and released with the commit hash. `runtime_differs` is a real bool on
    the stream form (the header carries a `runtime` field to compare
    against, unlike the raw `path` form, which has nothing to compare and
    stays `null`). Both forms share `import_from_path`, so the `n = pos + 1`
    fix above applies to both.
  - `checkpoints.rs`'s `sha256` is refactored into an incremental `Sha256`
    (`new`/`update`/`finalize`) behind the same one-shot `sha256()`
    signature: CONTRACT 1 explicitly forbids buffering a whole state stream
    to hash it, and the prior implementation needed the entire input in one
    `Vec`. Verified behavior-preserving against 6 chunk sizes spanning the
    64-byte block and padding boundaries, plus the existing known-vector
    tests.
- `GET /v1/state` and the int8 501 needed no new work this session:
  `GET /v1/state` (`1d5648e`) already matches CONTRACT 3's shape exactly
  (`model`, `weights_uuid`, `portable`, `kv`, `states[]`); `map_kvq_refusal`
  already covers every `collect()` call site, path and stream alike, so the
  501 `kvq_state_open` applies to both LAT1-wrapped routes without change.

## Checks and receipts

```text
cargo check --bin baro-serve
Finished (clean)

cargo clippy --bin baro-serve -- -D warnings
Finished (clean)

cargo nextest run --bin baro-serve
68 tests run: 68 passed, 0 skipped

cargo build --release --bin baro-serve
Finished (clean)
```

Two live GPU smokes, both through `gpu-wait run --vram 24 --timeout 600`,
`GPUWR_SOCKET=/run/user/1000/gpu-waiting-room.sock`:

```text
bench/p1-state-roundtrip-smoke.sh   (path variant, both routes)
export OK: pos=4 bytes=61079640 prefix_hash=a3762c148f44e787
import OK: pos=4 prefix_hash=a3762c148f44e787 restore_ms=1.95
round trip: export and import agree on prefix_hash and pos for the same prompt
PASS p1-state-roundtrip-smoke

bench/p1-state-lat1-smoke.sh        (LAT1 stream variant, both routes)
content-type: application/vnd.baro.state
export stream OK: pos_hi=4 payload_len=61079640 prefix_hash=a3762c148f44e787
import stream OK: pos=4 prefix_hash=a3762c148f44e787 runtime_differs=False
corrupt-payload falsifier OK: payload_sha
bad-magic falsifier OK: magic
PASS p1-state-lat1-smoke
```

Both smokes recover the identical `prefix_hash` (`a3762c148f44e787`) for the
same prompt through their own independent mechanism (raw file vs LAT1
stream), which is the cross-check this pair of smokes exists to make.

## Gates

| gate | result | reason or receipt |
|---|---|---|
| Build order step 1 (`bench/checkpoint-api.sh` still green, path-under-symlink and flipped-byte falsifiers) | CARRIED, not re-run this session | Landed and gated in an earlier session (`a781915` and prior); this session's changes to `checkpoints.rs` are the `sha256` refactor only, proven behavior-identical by 68/68 unit tests including new chunked-hash equivalence tests, so not re-run live to conserve GPU time. |
| Build order step 2 routes, single-prompt smoke (this session's own bar) | PASS | Both smokes above, live. |
| Plan gate 1: 20-prompt identity + restore band (2.2-4.3 ms), both formats, int8 bytes <= 0.30 of f32 | NOT RUN | Every smoke this session used one prompt and the server's default f32 format (`BARO_STATE_INT8` unset). `restore_ms=1.95` on both smokes is *below* the cited 2.2 ms floor, but the cited band is presumably calibrated for a representative-sized restore; this session's file was pos=4, 61 MB, likely too small to compare against that band meaningfully either way. Needs its own dedicated GPU run: 20 prompts, `BARO_STATE_INT8=0` and `=1` server instances, and the int8/f32 byte-size ratio check. |

## Kill lines and next action

No identity miss outside the documented E14 one has been observed; every
falsifier that should refuse did (missing export input, missing export path,
export/import against a too-short or wrong-magic raw file, a corrupted LAT1
payload, a corrupted LAT1 magic). Nothing here is a kill-line candidate.

Next action: the 20-prompt, both-format plan gate 1 run (needs a
`BARO_STATE_INT8=1` server instance in addition to the default f32 one, and a
size-ratio assertion against the f32 export of the same prompt set) before
build order step 3 (the cross-node rig, `bench/b4-cross-host.sh` at 100
Mbit/1 Gbit/10 Gbit) can start, since step 3's gate depends on gate 1's
numbers as its baseline.
