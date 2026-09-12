# Lane CHAT report: sampler/M1b/concurrency control block (2026-09-11)

Branch `lane-CHAT` (worktree `$HOME/Projects/mojo/mojo-baro-lanes/CHAT`), plan item
CHAT of `~/Brain/mojo/mojo-baro/briefs/2026-09-11-chat-engine-next.md`. Four steps, C1-C4,
each preregistered in `bench/chat-protocol.md` before its build, gated, and committed in
order on `lane-CHAT`.

**Verdict: C1, C2, C3 PASS every frozen gate. C4's code (default pool) PASSES; the
2-engine concurrency claim is UNVERIFIED on this machine — one engine already uses
~23.5-24.8 GB of this card's 25.75 GB, so a second cannot load beside it. Not an
implementation defect; measured and diagnosed below.**

## Commits (ahead of `main` at `38a85c7`, no attribution trailers)

| commit | what |
|---|---|
| `d322904` | bench(chat): preregister C1 |
| `8f5f6f1` | serve, kernels: C1 -- control block: stop sequences, EOS-in-engine, cancel, finish reason |
| `b3b4244` | serve, kernels: C2 -- M1b role-boundary checkpoints, SHA-256, retention |
| `1f70e45` | bench(chat): preregister C3 |
| `8792c58` | serve, kernels: C3 -- sampler host reference, matched to KSAMP's semantics |
| `a3ba89a` | bench(chat): preregister C4 |
| `2154144` | serve, tools: C4 -- engine pool routing (BARO_POOL); pool>1 needs a 2nd GPU |

## C1 -- M2 control block: stop, cancel, finish reason

Gate: `tools/test_server.sh` ALL PASS (`.work/CHAT-server-test/SUMMARY.txt`), including two
new cases (`stop`, `cancel`+`cancel-recovery`); `run-tests.sh` exit 0, 82 kernels / 38 in
registry / 0 orphans (no kernel touched); 20-prompt A/B vs `main`: ratio 0.998, identity
20/20 (`.work/ab-c1-results`).

Wire additions: `stop` (token-id sequences, engine-side EOS/stop matching, replacing
today's client-side-only cut), `{"cancel":ID}` (polled once per decode window via
`poll(0, POLLIN, 0)`), `"finish":"length"|"stop"|"cancelled"` on the done line. `POST
/v1/cancel` added; SSE/JSON response `id` is now `cmpl-<internal id>` / `chatcmpl-<internal
id>` so a streaming caller can name its own request.

## C2 -- M1b role-boundary checkpoints

Fixes M1a's measured gap: the prompt-end checkpoint (after the rendered generation-prompt
marker) is never a prefix of the next turn's re-render. `Text::role_boundaries` (Rust)
renders each prefix of the conversation with no generation prompt and reports its token
length; the engine takes a checkpoint there too (`Checkpoint.pinned`/`.boundary`,
`Chain.save`'s eviction order: never pinned, periodic-grid before role-boundary, oldest
within a class). `Checkpoint.hash` becomes a 32-byte SHA-256 (own FIPS 180-4, checked
against the empty-string/"abc"/NIST-56-byte KATs) of a per-pack salt + tokens, replacing
FNV-1a 64.

Two bugs found and fixed inside the gate (not after): a hint inside one big prefill chunk
was silently skipped (`prefix.mojo::next_ckpt_stop` now caps the chunk); a chat template's
own `raise_exception` on a system-only prefix propagated into a 500 on every chat request
(`role_boundaries` now skips a prefix that fails to render, returns `Vec<u32>` not
`Result`).

Gate: `run-tests.sh` exit 0 (new hint-save/lookup/byte-exact-restore/retention cases in
`kernels/test_prefix.mojo`, every original M1a case still passing under SHA-256).
`tools/test_server.sh` ALL PASS. 20-prompt A/B: ratio 0.999, identity 20/20
(`.work/ab-c2-results`). DeerFlow tap replay (`tools/tap-replay.py`, same 3 rows M1a
measured): `cached` 5750-5765 vs M1a's 5120, `prefill_rows` 4-92 vs M1a's 634-737, wall
time **2.7x-8.3x faster** than M1a on the identical rows.

## C3 -- sampler host reference

`serve/sample_ref.mojo`: host-only (no GPU) reference matched byte-for-byte against
`kernels/sample.mojo`'s `amar_sample_row`/`amar_spec_accept` (lane-KSAMP -- merged to
`main` partway through this lane, `dc9e06b`; read via `git show lane-KSAMP:kernels/
sample.mojo` before that, confirmed byte-identical to the merged version afterward) --
same Philox4x32-10 (verified against Random123's zero/all-ones/pi KATs), same `rng4`/
`rng_word`/`unif`/`gumbel` and stream numbering, same cut order (top-k, top-p mass at
T=1, min-p, temperature last) via a full sort instead of the kernel's parallel radix
select, same tie-break (lower index). Presence/frequency penalties are host-only
preprocessing. `serve/src/main.rs` parses `temperature`/`top_p`/`top_k`/`min_p`/`seed`/
`presence_penalty`/`frequency_penalty`/`logprobs` and carries them in the control block
(`SampleParams`, `#[serde(skip_serializing_if)]` so an unset request is byte-identical to
before).

**Scope, flagged as a design-call deferral**: the live decode-loop hookup (copy logits to
host, sample, write the token back) is not built, mirroring KSAMP's own "no engine wiring"
scope for the device kernel -- the frozen gates (distribution, seed, temperature 0) are
unit properties of the sampler function, the same shape as KSAMP's own kernel gate.
`serve/engine.mojo` parses every sampler field into `SampleParams` and does not act on it.

Gate: `kernels/test_sample_ref.mojo` (new, host-only) PASS on the Philox KAT,
temperature-0 (4 synthetic cases + the real 248,320-logit draft-receipt row), distribution
chi-square (6 configs, all inside their Wilson-Hilferty critical value), same-seed
reproducibility (2000/2000), speculation (accept-or-resample matches the exact target
distribution, chi2 15.02 vs critical 42.44), penalties. `run-tests.sh` exit 0,
`tools/test_server.sh` ALL PASS, `cargo test` 16 passed, clippy clean.

## C4 -- engine pool

`EnginePool` (`serve/src/engine.rs`) owns `BARO_POOL` (default 1) `Engine` processes and
one id counter shared across them (moved off `Engine`, so ids stay unique across engines);
`submit` routes to whichever engine has the fewest requests waiting/running, ties by
lowest index. At the default pool size of 1 this is unconditionally engine 0 -- a receipt,
not a design bet. `/health` gains `"pool":[q0, q1, ...]`.

- **P-J1 (default pool = today): PASS.** `tools/test_server.sh` ALL PASS at `BARO_POOL`
  unset (`.work/CHAT-c4-server-test/SUMMARY.txt`), identical to every earlier gate in this
  lane; `/health`'s only visible addition is `"pool":[0]`.
- **P-J2/P-J3 (two engines): UNVERIFIED, diagnosed, not hand-waved.** Three attempts at
  `BARO_POOL=2` (`tools/test_pool.sh`) all hit `hipErrorOutOfMemory` loading the second
  engine's pack -- including one immediately after `gpu-waitd` restarted with the card
  otherwise idle. Measured directly (a lone `baro-serve`, `BARO_POOL` unset,
  `rocm-smi --showmeminfo vram` before/after, GPU idle both times, confirmed twice):
  **one engine alone takes this 25.75 GB card to ~23.5-24.8 GB used.** The q4 pack itself
  is 6.64 GB; the rest is the MAX runtime's own reserved device pool (`docs/BASELINE.md`'s
  "~22.3 GB free to MAX" read as a ceiling the allocator claims on init, not a hint about
  what the pack needs). Clean shutdown (`SIGINT`, not a force-kill) does release it,
  confirmed by a direct before/after read ~13 s later. **The design brief's own sizing
  ("weights ~5.2 GB q4 each plus KV and state") undercounted the real footprint by
  4-5x** -- concurrency option (a), engine-pool, is not viable on one 24 GB card at any
  pool size above 1, independent of `EnginePool`'s own correctness. Fixing this (a second
  GPU, or finding/using a MAX device-memory-limit knob -- none found in the time
  available) is outside this lane's scope; flagged for whoever picks up the concurrency
  design round the plan already calls for (option b, continuous batching).

## Test counts

`cargo test` (serve/): **9 -> 16** (main `38a85c7` had 7 in `protocol.rs` + 2 in
`text.rs`; CHAT adds 3 in C1, 1+2 in C2, 1 in C3 = +7, all passing).
`run-tests.sh` kernel census: **82 kernels, 38 in registry, 0 orphans, unchanged** (no
kernel added or touched by this lane -- KSAMP's kernel work is a separate lane, merged
to `main` independently).
`kernels/test_prefix.mojo`: extended in place with M1b's 4 new assertions (hint save,
constrained lookup, byte-exact restore, retention x2), all original M1a cases still PASS.
`kernels/test_sample_ref.mojo`: new, 6 chi-square configs + KAT + temperature-0 (5 cases,
one at real vocab size) + reproducibility + speculation + penalties, all PASS.

## Evidence

- `bench/chat-protocol.md`: C1/C2/C3/C4 sections, predictions frozen before each build,
  Results filled in after (C4's Result carries the VRAM diagnosis in full).
- `.work/CHAT-server-test/`, `.work/CHAT-c2-server-test/`, `.work/CHAT-c4-server-test/`:
  `tools/test_server.sh` SUMMARYs at each step.
- `.work/ab-c1-results/`, `.work/ab-c2-results/`: 20-prompt A/B vs `main`.
- `.work/tap-replay-server.std{out,err}`: the M1b DeerFlow tap-replay receipt.
- `.work/pool-test/`, `.work/CHAT-c4-gate-v{2,3,4}.txt`: the three failed `BARO_POOL=2`
  attempts; `.work/single2.std{out,err}` + the `rocm-smi` reads around it: the single-
  engine VRAM measurement.

## Before merging: `main` has moved, checked for real conflicts

Branched at `38a85c7`; `main` is now `7120ab9` (lane-KSAMP, lane-E/RULER and a
`latent-os` feature all merged in the meantime). Checked, not assumed:

- `git diff --stat 38a85c7 main -- <every file this lane touches>` is empty **except**
  `bench/chat-protocol.md` (+266 lines on `main` -- KSAMP's own KSAMP/KSAMP-b/KSAMP-c
  preregistrations, a different section of the same shared file). No other file this
  lane edited has moved on `main`: no textual merge conflict expected, but the
  chat-protocol.md merge should still be read once, not trusted blind.
- `serve/sha256.mojo` landed on `main` via the `latent-os` merge (`506e91d`), vendoring
  the **same** `max._core_mojo.sha256` source I found and chose not to import for C2
  (its own docstring gives the identical reason: `mojo run` cannot resolve
  `max._core_mojo`). `serve/prefix.mojo`'s inline SHA-256 (this lane) now duplicates it.
  **Action for whoever merges this lane**: after rebasing onto `main`, replace
  `prefix.mojo`'s `sha256_bytes`/`_sha_*` functions with `from sha256 import sha256` and
  re-run the empty/"abc"/56-byte KATs to confirm the swap is inert. Not done here --
  out of this lane's scope (a different feature's file, not a conflict to resolve to
  keep building) and not safe to do blind against a `main` still in motion.
- `kernels/sample.mojo` on `main` (post-merge) is **byte-identical** to what C3 read via
  `git show lane-KSAMP:kernels/sample.mojo` (`git diff` between the two: empty) -- the
  host reference in `serve/sample_ref.mojo` needs no rework for the merge having landed.

## What's left (not this lane's job, named for the next one)

- C3's live decode-loop hookup (host sampling actually driving real generation).
- C4's concurrency-viable design: a second GPU, or a MAX memory-limit knob, or the
  design brief's option (b) (continuous batching) that the plan already flagged as the
  real fix for throughput under multiple users.
- The `serve/sha256.mojo` de-duplication noted above.
- Brief's open items untouched by CHAT: M4 (tool-call lexer + endpoints), M5 (quantised
  KV + RULER), M6 (guided decoding), prefill speed, disk-persisted checkpoints.
