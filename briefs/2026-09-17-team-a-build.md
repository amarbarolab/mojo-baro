# Team A build: P0a now, then P1, then P0b

Read `briefs/2026-09-17-team-build-rules.md` first (in your worktree after `cd`), then the P0a, P1
and P0b sections of the corrected `docs/PLATFORM-PLAN.md`. The conference is over; this is the build.

## Item P0a: Ollama-compatible API on `baro-serve` (your own template, completed)

- **Files.** NEW `serve/src/ollama.rs` (request/response types, option map, NDJSON framing,
  translation onto the existing request path); `serve/src/main.rs` (route registration only, inside
  a MAIN.RS WINDOW); NEW `serve/src/embeddings.rs` plus whatever the engine wire needs for a pooled
  vector (announce the wire change in the room before making it; `serve/PROTOCOL.md` gets the new
  routes and the wire addition); NEW `bench/p0a-contract-gate.sh` (codex; this is the
  `serve-contract-gate` you asked for, written as the item's own gate with a `GATE_DRYRUN` stop).
  `serve/src/protocol.rs` is not touched for HTTP types.
- **Owner.** sonnet: `ollama.rs`, `embeddings.rs`, `main.rs` lines, engine wire. codex:
  `bench/p0a-contract-gate.sh`, the client matrix (pair-dispatch, the `ollama` Python client in a
  venv under `.work/`), PAIR node bring-up for gate 3, `serve/PROTOCOL.md` sections, the report.
- **Build.** `cd serve && CARGO_TARGET_DIR=... cargo build --release`; `~/iTools/bin/rust-verify -p serve`
  after route edits.
- **Preflight.** Gate 1 with `--count 1`, gate 2 on one prompt, `gate-dryrun` on the gate script.
  The engine behind `baro-serve` is a GPU resident: start it once through `gpu-wait run`, reuse it
  for every gate, stop it when done.
- **Gates** (verbatim from the plan): (1) `~/iTools/bin/pair-dispatch --backend ollama --port <ours>
  --count 5 --mode parallel` 5/5, token counts equal to `/v1/completions` same seed T=0. (2) `ollama`
  Python client, `chat` and `generate`, streaming and not, byte-identical text to
  `/v1/chat/completions` on 20 prompts. (3) PAIR's engine manager adopts `baro-serve` on 11434 as
  Ollama and routes a request; receipt is PAIR's workload log naming our node. (4, added)
  embeddings: `/api/embeddings` and `/v1/embeddings` return the same vector for the same input, unit
  norm, stable across two runs, and different inputs give cosine below 0.999.
- **Receipts.** `.work/team-A/<agent>/p0a/*.log`, report `exchange/lane-P0A-report.md`.
- **Kill line.** Any mismatch in gate 1 or 2. Gate 3 failure: report at once, keep building.
- **GPU budget.** Minutes; one resident engine, `--vram 24 --timeout 3600`, priority default.
- **Dependencies.** None.
- **Size.** 380 LOC Rust plus the gate script; over it, say why in the commit message.

## After P0a

Post the report, tell the coordinator, then: sonnet starts P1 from `docs/P1-STATE-API.md` (if it
has not landed, ask once and work on P0a leftovers); codex starts the P0b CPU skeleton (mDNS
advertise and browse, `/v1/node-info` without the locality term, port-probe adoption, proxies;
NEW `serve/src/bin/router.rs` and router-only modules) and P0b gate 3. P0b gates 1, 2 and 4 wait
for P1's contract: deferred, never stubbed. Write each next item in the template above in the room
before building it, so your partner can object.

Add your own cards for the sub-steps (`~/iTools/bin/room card+ A "..."`) and turn them as you go.
