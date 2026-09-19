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
- Mojo builds that touch `serve/tokenizer.mojo` / `serve/spark.mojo` /
  `serve/engine.mojo` / `serve/latent.mojo` need `-I .` (regex engine, chat-template
  renderer, and the LatentOS sidecar, vendored at repo root as `uregex/` /
  `minja/` / `latentos/`; `tools/profile.mojo` likewise reads `toml/`, DataBooth/mojo-toml; upstreams `~/Projects/mojo/mojo-uregex` /
  `~/Projects/mojo/mojo-minja` / `~/AMDHQ/src/latentos`, synced by hand, drift
  checked by `tools/ci-checks.sh` when present).
- **Bit-exact means a byte test against the m=1 / decode kernel, not a token gate.**
  New multi-row kernels call the m=1 dot helpers with the same per-element order
  (`kernels/test_moe_rows.mojo`, `kernels/test_ssm_rows.mojo`). The dense chunk
  kernels `amar_attn_prefill_wmma` and `amar_ssm_delta_chunk_w` reorder sums (the scan
  differs from the decode step in 87% of outputs and flipped a greedy token at index
  4): never reuse them where replay identity is the gate. A kernel that fails greedy
  equality gets a teacher-forced agreement gate with a same-arm control and a bar set
  BEFORE the run; WMMA attention in MoE prefill read 97.92% against 99% and stayed
  opt-in.
- **`gpu-wait run` carries neither your environment nor your stdin.** Knobs go in
  as `env K=V` inside the queued command; a redirect goes inside it too (`bash -c`,
  or the script re-execs itself under the queue). An engine fed an empty stdin prints
  ready, exits 0 and the job is green with nothing served. `gpu-wait gpu` prints a
  10 KB dict: parse it, never print it.
- **Budget VRAM against MAX's pool, not the card.** MAX takes about 90% of the VRAM
  free at start. The resident qwen35moe pack (21 GB) leaves 0.08 to 0.4 GB and ran out
  of memory at an 8k f32 KV cache, then ran slower than tier mode at 1k. `pack-fit
  PACK --tmax N [--tier CAP] --used-gb 0.75` before any resident arm; tier mode
  (`BARO_TIER=64 BARO_TIER_PINNED=1 BARO_TIER_ZC=1`) is the MoE arm past 1k context.
- **Exploratory gate runs say so in their exit code.** A gate that cannot pass its
  `bench/preflight.sh --check` yet runs with `EXPLORE=1`: verdict capped at
  `UNVERIFIED`, exit 3, never PASS. A gate also counts how many of its inputs reach
  the feature's threshold and names the rest NOT EXERCISED (9 of the 20 `mtp-prompts`
  reach PF_MIN + 1 = 17 tokens; none crosses a prefill chunk).
- **`.work/` is not storage.** It is gitignored and was recreated empty on 2026-09-19
  03:07: `engine-pack-q4`, `m5`, `spark`, `refcache` and every receipt path cited by
  the board and by `exchange/` reports before that date are gone, the lane worktrees'
  fixture symlinks dangle, and `lane-merge` fails on receipts alone. Packs are rebuilt
  into `~/.cache/baro/<model>/packs/` by `tools/baro`; a receipt a report cites is
  copied under `exchange/receipts/<lane>/` (small text only) in the same commit.
  `.work/engine-pack-q4` is a link to the cached pack of the real Qwythos gguf
  (`sha-cache.tsv` names it; two other cached q4 packs are P5b patched models).
- **One primary checkout, on `main`; lanes live in worktrees** (`.work/lanes/<lane>`
  or `../mojo-baro-lanes/<lane>`, prepared with `lane-prep`). `lane-status .` before
  opening or merging a lane, `lane-merge <branch>` before the merge. A long-running
  shared branch checked out in the primary directory (lane-r63 carried 31 commits of
  serve, audio, router and kernel work for two days) hides what is and is not on main.
- Verify: `bench/preflight.sh` (CPU only: `tools/ci-checks.sh`, every test and both
  engines build; gate scripts call `--check`), `./run-tests.sh` under `gpu-wait`,
  `./bench/run.py [bench-src.mojo]`, parity tests in `kernels/test_*.mojo` (build AOT
  into `.work/`, `-I kernels`). Protocol rules P1 to P20: `bench/PROTOCOL-RULES.md`.
