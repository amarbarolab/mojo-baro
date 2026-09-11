# mojo-baro

- **Kernel files (`kernels/matmul*.mojo`, `kernels/elementwise.mojo`) carry
  ZERO comments and ZERO docstrings.** Rationale, sweep numbers, and design
  notes go in commit messages, `docs/`, and `~/Brain/mojo-baro/` — never in
  kernel source. Test files may keep docstrings.
- `docs/BASELINE.md` = current truth; read before kernel work.
- **Before ANY timed run: read every arm-defining parameter back from the running
  system and record it.** Passing a flag is not evidence it took effect; a
  silently-inert parameter produces clean numbers with tight spread and spread
  will not catch it. No receipt, no arm — see `bench/PROTOCOL-RULES.md` P1. **Decode numbers = 20-prompt median (P4); any m>1 kernel needs a row-scaling receipt (P5); harness before kernel (P6).**
- Perf claims require the preregistration flow in `bench/coldcache-protocol.md`
  style: freeze predictions by commit BEFORE running; single-buffer GEMM
  timings at W >= 96 MB are invalid (Infinity Cache contamination).
- **The reference arm IS an arm.** Cold-cache rotation, clock probe and parameter
  read-back apply to the vendor/reference side exactly as to ours. Ours streams 36
  layers and is cold by construction; theirs is not. (ggml `down` GEMV "1.40x
  slower" was 28 MB living in the 96 MB Infinity Cache at 1.1 TB/s — above HBM peak
  — and cost two kernel rounds chasing a gap that did not exist; same class as the
  4096^3 timings that invalidated 0.791 R and left D1 PROVISIONAL.) A reference
  implementation is not a reference: llama.cpp's own f16-KV config fails its own f32
  ref at 5/7 lengths, so identity gates are **teacher-forced agreement**
  (`BARO_FORCE`), never greedy 64-token equality past ~256 ids.
- **Read the receipt before diagnosing the number.** First action on any ratio is to
  confirm arm identity in the arm file (`engA=`/`engB=`); no physical cause — power
  cap, alternation order, env form — is proposed before that line is read. Two
  20-prompt A/Bs compared the champion with itself and three GPU jobs went to
  explaining a 1.00.
- **A gate the candidate can write is not a gate.** The stopwatch lives in
  `engine.mojo`, inside the artifact under test; a sync-deleting candidate passed
  perf at +63%. So: timing state, host syncs, file writes and fixture paths banned
  at scope; reference fixture snapshotted before any run; identity on EVERY timed
  run; **fail word read on every run** (a NOT-RESIDENT early exit is silent).
  Synthetic tests decide nothing — a bit-exact, faster, fewer-spills synthetic win
  lost 1.5% on the real pack.
- **Close the round or kill it; never pause it.** A frozen prediction with no numbers
  against it is the most perishable state in the repo. Before opening a lane: merge,
  close, or explicitly kill every lane branch ahead of `main`, and land its
  `exchange/` report.
- **`baton.md` is rewritten the same turn the champion number or in-flight set
  changes.** It auto-prints at SessionStart, so a stale baton misinforms every later
  session before it reads anything else; run `board-vs-reality` before building on
  any board claim.
- Mojo builds that touch `serve/tokenizer.mojo` / `serve/spark.mojo` need `-I ~/Projects/mojo/mojo-uregex/src -I ~/Projects/mojo/mojo-minja/src` (regex engine + chat-template renderer, own repos).
- Verify: `./run-tests.sh`, `./bench/run.py [bench-src.mojo]`, parity tests
  in `kernels/test_*.mojo` (build AOT into `.work/`, `-I kernels`).
