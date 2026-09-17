# Build rules for teams A and B (both peers read this once, before the item brief)

These are your own rules from `exchange/2026-09-17-team-{A,B}-plan-shape.md`, merged, plus the
coordinator's answers. Where this file and your conference document differ, this file wins.

## Where you work

- Team A: worktree `~/Projects/mojo/mojo-baro-lanes/team-a`, branch `lane-team-a`, room A.
- Team B: worktree `~/Projects/mojo/mojo-baro-lanes/team-b`, branch `lane-team-b`, room B.
- `cd` into your worktree FIRST and stay there. Never edit or commit in the main checkout
  `~/Projects/mojo/mojo-baro`. Both worktrees are lane-prepped (`.venv`, packs linked) and sit at
  main `b30506e`, which holds the corrected `docs/PLATFORM-PLAN.md`: re-read your items there.
- Codex: the lanes folder, the main repo's `.git`, `exchange`, `.work` and the Brain rooms folder are
  in your `writable_roots`. If a write is refused, say so in the room and ask the coordinator; do
  not route around it.
- Room commands work from the worktree: `~/iTools/bin/room say|read|cards|card A|B <name> ...`.

## Coordinator's answers (the maintainer confirmed 2026-09-17)

- Order team A: P0a, then P1, then P0b; P0b's CPU skeleton may run parallel to P1. Team B: P3a and
  P3b first, then P5a, P4 wiring, P5b, P4 timed gate, P6.
- Embeddings: last-token pooling of the final hidden state, L2-normalized.
- The iGPU is NOT exempt from `gpu-wait`.
- P0a gate 3 (PAIR adoption) failing does not void P0a: report it to the coordinator the moment it
  is seen, as its own message, and keep building.
- The P1 design spec comes from the coordinator (in progress, lands as `docs/P1-STATE-API.md`
  before P0a is done). Do not start P1 without it.
- PAIR builds from source here: `~/Projects/imports/Personal-AI-Router/services/build/bin/` holds
  13 binaries (log `.work/pair/build.log` in the main checkout). The engine manager adopts whatever
  already answers on an engine's fixed port when `engine:start` runs (its README, section Adoption).
- C2-mini int8 state files ARE on main (`BAROST02`, `BARO_STATE_INT8=1`); only state save/load under
  `BARO_KVQ=int8` is open. The baton line saying otherwise was stale.
- Rust convention: `cd serve && cargo build --release`, each agent with its own
  `CARGO_TARGET_DIR=$PWD/../.work/team-<T>/<agent>/target`.
- `serve/src/main.rs` route registration is ONE edit window at a time ACROSS BOTH TEAMS: before
  touching it, tell the coordinator (`~/iTools/bin/herd tell w82:pC "MAIN.RS WINDOW: team X, item
  Y"`), keep the edit to the registration lines, commit, tell the coordinator `MAIN.RS RELEASED`.
  Everything else of your item lives in your own new files.
- The standing GPU rule, restated here because codex cannot read the global config: every GPU
  workload (engine, llama.cpp, whisper-server on GPU, torch, benchmarks) runs as
  `gpu-wait run [--priority N] --vram GB --timeout S -- <cmd>`, never bare. The full suite is
  wrapped whole: `gpu-wait run --vram 24 --timeout 3600 -- ./run-tests.sh`. The client is `$HOME/.local/bin/gpu-wait`; a job's PATH is minimal (gpu-wait drops the shell env), so scripts and code name it absolutely. Inside a job (`GPU_WAITING_ROOM_JOB` set) GPU work runs bare: never nest `gpu-wait run`. CPU preflight first:
  build every binary and run the gate once on its smallest input before any queue slot.
- No em dashes anywhere. No attribution lines in commits. `/tmp` is banned for artifacts: use
  `.work/team-<T>/<agent>/<item>/` in your worktree.

## Your protocol (yours, verbatim in substance)

- Default split: sonnet owns the wiring in source files; codex owns the protocol note, fixtures,
  gate script, parameter read-back and the receipt, and runs the gate. Each reviews the other.
  Trade ownership only by saying so in the room.
- Ownership by exact path. Announce in the room before editing a path outside your list or any
  shared doc. Never edit a path your partner is mid-edit on.
- Announce `RUNNING GPU` / `GPU DONE` / `RUNNING SUITE` / `SUITE DONE` in the room. One gate runner
  at a time.
- Commits: pathspec only, per landed sub-step. `index.lock` busy: wait 3 s, retry.
- Review: partner reviews the diff and touched-path list before commit for shared surfaces, at the
  next commit boundary for own new files. The reviewer checks that the gate can fail for the claimed
  reason, the receipt postdates the commit, scope stayed inside the item, no partner path leaked
  (`git show --name-only`), and re-runs the falsifier (`~/iTools/bin/claim-check`) or records why not.
- Turn your cards as you go; `done` needs the check and the log path.
- Never idle on a question you have not posted in the room. Blocked on the coordinator:
  `~/iTools/bin/herd tell w82:pC "QUESTION: ..."` and continue with what does not depend on it.

## Done means

One report per item, `exchange/lane-<ITEM>-report.md` on your branch: the gate commands and exit
codes, the receipt paths (they must exist), the `./run-tests.sh` receipt line (exit code and PASS
count, log path), the commit list, what is UNVERIFIED and why. Then
`~/iTools/bin/herd tell w82:pC "ITEM <X> report written to exchange/lane-<X>-report.md"` and move
to your next item without waiting. The coordinator merges with `lane-merge`; a report without the
suite receipt is NOT READY.
