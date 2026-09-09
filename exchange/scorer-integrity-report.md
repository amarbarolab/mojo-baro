# Scorer integrity audit — can a loop candidate still game the gate?

Date 2026-09-08. Scope: `tools/loop-gate.sh`, `tools/loop-propose.py`,
`tools/diff-normalise.py`, `bench/loop-protocol.md`, `bench/PROTOCOL-RULES.md`,
`docs/BASELINE.md`, receipts `.work/loop/001..005`. Line numbers cite the
gate at `c0cfb6f` (before this audit) unless marked *new*.

## Verdict

**Yes — before this audit, a candidate could write any `tok/s_gen` it liked with a
one-line edit and pass every stage that was reachable.** Four such candidates were
built and run through the real ladder on the GPU (`.work/loop/audit-scorer/`):

| cand | edit (engine.mojo, allowlisted, zero banned tokens) | identity | tok/s_gen runs 1-3 (real ≈ 67) | ladder verdict at `c0cfb6f` |
|---|---|---|---|---|
| 0 | `t_prefill_end += 500_000_000` after the prefill stamp | PASS | 148.5 / 114.2 / 132.6 | perf FAIL — **on spread only** (0.374) |
| 1 | delete the `ctx.synchronize()` before `var dt = …` | PASS | 108.6 / 108.9 / 110.8 | **perf PASS at +63 %**, spread 0.021 → died at stage 4 |
| 2 | delete `prefill_done = True` | PASS | 597 887 / 923 618 / 821 692 | perf FAIL — on spread only (0.545) |
| 3 | set token 1 to `0`, then rewrite `ref-tokens-64.txt` with its own output | **PASS** (64/64 against its own file) | 67.0 (no timing edit) | perf FAIL — did not claim speed |
| 4 | garbage into `shim/src/amarbaro_shim.cpp` | — | — | scope FAIL (see B5) |

cand-1 is a winner under the rule as written: the only thing that stopped it is
**stage 4, which no candidate can pass, because the champion's own sources fail it**
(B1 below). cand-3 shows the identity gate trusts a file the candidate's process
can write. The spread term caught cand-0 and cand-2 by accident of the arithmetic
(a tiny `decode_s` jitters proportionally more); a candidate that shifts the stamp
by 100 ms instead of 500 ms would have passed with a tidy +13 %.

P1 receipts for those runs: `prompt file: .work/engine-pack-q8/prompt-tokens.txt`,
`prompt tokens: 5`, `BARO_SPEC: False`, `spec k: 2`, `BARO_DOT: False`,
`BARO_DRAFT_Q4: False` read back from each run log; 8083 down (`ss -ltnp` empty);
GPU idle before the batch (13 W, 41 °C, 1.8 GB used, queue empty); champion
argument 66.81 = iteration 005's median. Cold-GPU jitter is visible in cand-0's
`gpu_total_s` (0.97–1.13 s); it does not change any verdict above. The reference
file was backed up before cand-3 ran and restored after (sha `5f07a4ec…` verified).

The original gate output is `.work/loop/audit-scorer/gate-run.log`; the per-run
logs were regenerated when the hardened gate was re-run on the same set (all four
now die at scope), so the per-run numbers above live in this report.

## The seven controls, here

| # | control | status | where / why |
|---|---|---|---|
| 1 | write allowlist | **present** | `loop-gate.sh:15-16` — touched basenames must be in `$dir/FILES` = gguf `baro.kernel.files`. Strong: every out-of-list file dies here. Weak in one way that matters: **`engine.mojo` is on the list and contains the stopwatch** (`t0`, `t_prefill_end`, `dt`, `decode_s`, the `tok/s_gen` print, `serve/engine.mojo:857-1427` at HEAD, `engine.mojo:515-934` in the ffa1808 gguf). The allowlist admits the instrument by construction. |
| 2 | independent scorer | **partial** | The supervisor (gate) runs the benchmark, the candidate never invokes it (`:41,:45`). But the benchmark *is* the candidate's binary and the score is its own `print` (`engine.mojo:934`), grepped back at `:45`. Independent in who presses the button, not in who reads the clock. |
| 3 | scoring artifacts outside the writable tree | **absent** → *partial (new)* | Timing code: inside the allowlisted file. Reference tokens `.work/engine-pack/ref-tokens-64.txt` and prompt `.work/engine-pack-q8/prompt-tokens.txt`: read by the gate **after** the candidate process (same uid, cwd = repo root) has run — cand-3 rewrote it. *New:* the gate snapshots the reference before any candidate runs, compares every run against the snapshot, and fails + restores if the on-disk file changed (`loop-gate.sh:9-18` new). Timing code is still inside the tree; only a source split can fix that (P-A below). |
| 4 | fresh checkout per score | **partial, and the lead overstates it** | `$work` is `rm -rf` + `cp -r "$dir/src"` per candidate (`:10,:27`); `$dir/src` is materialised from the gguf by `loop-propose.py:151-154`. There is no "candidate's commit" — the candidate is a diff that never touches git — so at the *source* level the copy is as clean as a checkout. What is not fresh is the **runtime**: cwd is the repo root, `.work/` is shared across candidates and across runs 0-3, env is inherited. That is the channel cand-3 used, and the channel a candidate would use to behave on run 0 (identity) and differently on runs 1-3 (timed). |
| 5 | diff policy gate before execution | **partial** | `:20` — a token denylist on `+`/`-` lines only. Proven reachable-around four ways above. *New:* pattern extended to every identifier the `tok/s_gen` arithmetic reads, `synchronize`, `open(`, fixture paths, `GEN_N`, `toks_h`, `generated` (`:48` new); none of these names occur in any kernel file (checked per file), so kernel edits are unaffected. Regression: 14/14 historical candidates keep their stage-0 verdict; 4/4 audit candidates now die at scope. It is still a denylist and still unbounded (residual R1, R6). No check for mode changes, renames or symlinks — GNU `patch` on a plain unified diff cannot create them, so this is covered by the input format, not by the gate. |
| 6 | correctness before performance | **present, thin** | Stage 2 (`:41-42`) precedes stage 3. Two gaps: identity was checked on **run 0 only** — the three timed runs were never compared (*fixed*, every run now); and "correct" means 64 greedy tokens on one 5-token prompt (`The capital of France is`) through the m=1 non-spec path. Nothing exercises m>1, `BARO_SPEC=1`, or a second prompt. |
| 7 | multiple hidden workloads | **absent** → *partial, behind a flag (new)* | One fixture. *New:* `LOOP_PROMPT2=<ids>` builds the iteration's pristine sources once, generates that prompt's reference from them, and requires every candidate to reproduce it (`:19-29, :71-75` new). `bench/loop-prompt2.txt` = 27 tokens. Still the same shape: `GEN_N = 64` is comptime (`registry.mojo:31`), `TMAX = 128` caps the prompt at 64. |

## The four leads

**1. Metric self-reported — confirmed.** `engine.mojo:934` prints
`tok/s_gen = (GEN_N-1)/decode_s`; `decode_s = dt - prefill_s` where every term is a
`var` in the candidate's file. The denylist bans the *calls* (`perf_counter_ns`)
and the *print*, not the *state*: cands 0-2 never touched a banned line.

Independent measurement, what it would cost here:

- *Gate wall clock around the process* (now in every receipt as `wall_s`, beside the
  engine's `gpu_total_s`). Pristine ffa1808 build, two runs: wall 2.109 / 1.989 s,
  pack load 0.845 / 0.798 s, decode 0.95 s. Decode is ~47 % of wall and the wall's
  own run-to-run jitter (6 %) exceeds the +2 % acceptance step. Usable as a
  **plausibility bound** (a claimed 63 % decode win must shave ≥0.35 s off wall;
  cand-1's wall would not have moved), not as the metric — unless generation gets
  long enough to dominate, which needs `GEN_N`/`TMAX` to become harness knobs
  (comptime today, and the candidate could edit them; both now denylisted).
- *GPU timestamps via `rocprofv3 --kernel-trace`* — the right instrument in
  principle (outside the binary, hardware clocks). Tried on the pristine build:
  captured 20 391 dispatches / 0.527 s span, then **the ROCr runtime asserted**
  (`stl_vector.h:1253 … AqlPacket … __n < this->size()`) and the engine hung;
  rocprofv3 1.1.0 against MAX 26.5's queue usage is not a working combination
  today. Cost: unknown engineering to make it not crash, then per-run overhead to
  measure. Not pursued further.
- *Two-length differential* (time at 64 vs 128 tokens, slope = decode): blocked by
  `TMAX = 128` and comptime `GEN_N`.

Recommendation: keep `tok/s_gen` as the metric, add the wall-clock plausibility
term to the rule (P-B), and split the stopwatch out of the candidate's file (P-A).

**2. One fixture — confirmed; cost of a second one measured.** Second workload
behind `LOOP_PROMPT2`: one extra `mojo build` of the pristine sources per gate
run (28 s), one champion run, one extra run per candidate (~2 s). It removes
"detect the 5-token prompt"; it does not remove "detect 64 generated tokens" or
"detect m=1 greedy", which are the shape of every fixture this engine can run at
`TMAX = 128`. The reference for prompt 2 is produced in-session by the iteration's
own sources — same trust root as the champion measurement.

**3. Copy, not checkout — partly wrong.** See control 4: the source copy is
pristine per candidate; the leak is runtime state, not stale sources. A clean
checkout of the candidate's commit would change nothing here (there is no commit;
the sources come from the gguf). A **fresh cwd per run** would close the
cross-run channel (`BARO_PACK`/`BARO_PROMPT` absolute, `mkdir run$k/.work` for the
engine's own `draft-*.bin` writes); not done — after the reference snapshot and
`open(` on the denylist, the remaining cross-run channel needs `external_call`
into libc, which is R6 below and is not closable by a denylist anyway.

**4. Denylist reachable around — confirmed, four ways, live.** Then tightened
(control 5). It stays a denylist: any new timing-adjacent identifier, any file API
reached through `external_call` (the engine already uses `external_call["pread"]`
for the pack), any string built by concatenation (`".work/engine-pack/ref-" +
"tokens-64.txt"` is exactly how cand-3 got past `ref-tokens`) walks through.
The allowlist half is the strong half **because it is structural**; the fix of the
same kind is to make the stopwatch a file the candidate cannot touch.

## Residual attacks this ladder does not cover, cheapest first

Ranked by lines of diff a candidate would need, given the hardened gate.

- **R1. Move work outside the timed region / precompute before `t0`** (~10 lines).
  Copy the decode loop in front of `var t0 = perf_counter_ns()` (context line,
  never scanned), then neuter the timed loop (`while pos < n_total - 1:` →
  `while False:` — no banned token). `toks_d` already holds the right ids, so
  identity passes on every run; `decode_s ≈ 0`. Today only the spread term stands
  in the way, and a candidate that leaves a cheap, steady per-token stub in the
  loop has tight spread and any `tok/s_gen` it wants. **There is no ceiling and no
  external cross-check in the rule.** This is the one to close first (P-B).
- **R2. Win the lottery** (0 lines). Champion spread 0.5–1.9 % in receipts, single
  runs 65.78–67.85 (3.1 %), threshold +2 %, champion measured before the batch
  and never interleaved. A no-op resubmitted every iteration will eventually clear
  +2 % on a warm afternoon. Not a defect of the gate's *integrity*, but it is the
  cheapest "speedup" available and every past receipt is exposed to it. Mitigation
  is protocol, not gate: ABAB interleave the champion binary with the candidate in
  the same session (the `LOOP_PROMPT2` path now leaves a `champion-engine` in
  `$dir`, so the binary is there).
- **R3. Special-case the fixture** (~5 lines). `len(prompt)` and the ids are
  runtime values the candidate's code can branch on; `prompt` is not and cannot be
  a banned word. `LOOP_PROMPT2` doubles the work; `GEN_N = 64` and m=1 greedy stay
  detectable in every fixture the engine can run.
- **R4. Behave on run 0, cheat on runs 1-3** (~5 lines). Identity now runs on every
  run, so the *output* must stay right; a run counter persisted through a file
  (`external_call` — R6) can still switch the *timing* trick on after run 0.
- **R5. Nondeterminism** (a few lines). Median-of-3 plus identity-on-every-run
  leaves little here; a race that is right 99 % of the time would need ~4 clean
  runs in a row and would then be embedded as a winner and fail in the field.
- **R6. Reach the fixtures through libc** (~6 lines). `external_call["open"]`,
  `["fopen"]`, `["pread"]` are not and cannot be denylisted (the loader uses them).
  With the reference snapshot the candidate can no longer *rewrite* the reference,
  but it can still **read** it and echo it — memoised output, the note's classic.
  Only running the candidate where the fixture is not readable closes this
  (different uid / `bwrap --ro-bind` with the ref masked, or `mv` the ref out of
  `.work` for the duration of the gate — not done, it would race any concurrent
  `mega-gate.sh`/`gguf-closure.sh` that reads the same file).
- **R7. Compiler flags → UB.** Not reachable: the build line is the gate's (`:39`),
  the candidate only supplies sources. UB inside the sources is covered by
  identity exactly as far as identity reaches (control 6).
- **R8. Consume an uncounted resource.** Nothing to consume: single process, single
  GPU, no wall-clock budget. Power/clock state is not candidate-controllable from
  Mojo.

## Bugs found

- **B1 — stage 4 is unpassable, so the loop could never have recorded a winner.**
  The champion sources in the ffa1808 gguf carry 9 kernel instantiations with
  scratch and spills (8× `amar_ssm_delta_step` SIMD variants, 81–603 VGPR spills;
  1× `amar_matmul_skinny`, 128 spills), read from the built binary's code objects
  with the gate's own extractor. `bench/mega-structural-protocol.md` already
  records spills as a known property of this engine. cand-1 is the first candidate
  ever to reach stage 4 and it died there for the champion's sins. **Not fixed**:
  the frozen rule says "no scratch and no spills in any embedded code object", and
  making it baseline-relative is a change to what counts as a win → P-C.
- **B2 — identity on run 0 only** (`:42`). Fixed: every timed run is checked, a
  failure names the run.
- **B3 — reference file read after the candidate ran.** Fixed: snapshot before any
  run, tamper check + restore.
- **B4 — denylist evasions** (cands 0-3). Fixed as far as a denylist can be.
- **B5 — allowlist compares basenames** (`:15`), so the nested entries of
  `baro.kernel.files` (`shim/CMakeLists.txt`, `shim/src/…`, `shim/include/…`) can
  never be edited although listed (cand-4). Harmless today — `engine.mojo` does not
  import `amarbaro`, the shim is not linked (`readelf -d`: only the Modular
  RUNPATH) and the gate never builds it — but the gguf's file list overstates the
  editable set and `loop-embed-winner.sh` would re-embed unbuilt shim sources as
  part of a "winner". Not changed; noted for the next embed.

## What was changed (commits on `main`)

Each was checked on a benign candidate (`.work/loop/audit-benign/`: swap two
adjacent `comptime` lines in `matmul_skinny.mojo`; semantically nothing) through
the hardened gate with `LOOP_PROMPT2=bench/loop-prompt2.txt` and champion set to
1 so that stage 3 passes on any real number. Result (`gate-run2.log`): scope PASS,
apply `-p1`, compile, identity PASS on the published fixture and on fixture 2
(27 prompt tokens, 64/64 identical to the champion's own output), identity PASS
on runs 1-3 (66.9 / 67.2 / 67.5 / 66.9 tok/s_gen, spread 0.009), perf PASS,
then **stage 4 FAIL — the same 9 spilling kernels as the champion** (B1). The
hardening touches nothing the champion's own sources would not also clear.
A first attempt of the same run had failed stage 3 on spread 0.057 (63.3 vs
66.9 on consecutive runs of an unchanged binary) — R2 in one screen.

1. `afd41f1` loop-gate: ban the timing state, host syncs, file writes and fixture paths at scope
2. `4062a4c` loop-gate: snapshot the reference before any candidate runs; identity on every timed run; wall_s in the receipt
3. `e8d44ac` loop-gate: optional second workload behind LOOP_PROMPT2, reference from the iteration's own sources (+ `bench/loop-prompt2.txt`)
4. `2211535` loop-protocol: the amendment recorded under the frozen rule, rule text untouched

Gotcha found on the way: `gpu-wait run` does not forward the caller's
environment, so `LOOP_PROMPT2=… gpu-wait run -- tools/loop-gate.sh` silently ran
without the second fixture (no "second fixture:" line in `gate-run.log`). Pass it
inside the job: `gpu-wait run -- env LOOP_PROMPT2=bench/loop-prompt2.txt
tools/loop-gate.sh ITER CHAMP`. Recorded in the protocol note and the script's
commit message.

None alters the frozen acceptance rule: a candidate that is bit-identical on every
run and does not touch the stopwatch sees the same verdict as before. Reading 2
strictly, "64/64 identical" now has to hold on all four runs instead of one —
flagged here as my reading of the rule, not a new term.

## Proposals — not applied, the maintainer's call

**P-A. Move the stopwatch out of the candidate's reach (structural; L).** Split
`serve/engine.mojo` so `main()` — pack load, prompt, `t0`/`t_prefill_end`/`dt`,
the prints, the draft receipt — lives in a file that is **not** in
`baro.kernel.files`, and the allowlisted `engine.mojo` exports the per-window
step. Then controls 1 and 3 become the same control and the denylist can be
deleted. Changes the embed (`gguf-embed.py` file list) and the proposer's region
slicing (`loop-propose.py` `MARK`/`slice_region`), so it is an interface change
and waits.

**P-B. Plausibility term in the acceptance rule** (`bench/loop-protocol.md`, frozen
2026-09-01). The receipt now carries `wall_s`; the rule could read:

```diff
 A candidate lands only if: touches only `baro.kernel.files`; compiles; 64/64
-greedy tokens identical; median of 3 `tok/s_gen` read back from the engine's
-own output >= champion + 2% with spread < 5%; no scratch and no spills in any
-embedded code object.
+greedy tokens identical on every run, and on the second fixture when one is
+set; median of 3 `tok/s_gen` read back from the engine's own output >= champion
++ 2% with spread < 5%; the claimed decode saving is visible in the gate's own
+wall clock: median(champion `wall_s`) - median(candidate `wall_s`) >=
+0.5 x (63/champion - 63/median `tok/s_gen`), champion `wall_s` measured by the
+same gate in the same session; no NEW scratch and no NEW spills relative to the
+champion build of the same sources.
```

The 0.5 factor is deliberately loose (wall jitter 6 % at this run length); it
exists to reject a 2× `tok/s_gen` whose process did not get faster, not to
adjudicate +2 %. Requires the gate to time the champion binary (the
`LOOP_PROMPT2` path already builds it; ~30 s + 3 runs).

**P-C. Baseline-relative ISA** (part of the diff above). Extract the same
`(.private_segment_fixed_size, spill_count)` per kernel name from
`$dir/champion-engine` and fail only on kernels whose numbers rose or that are
new and non-zero. Without this, stage 4 rejects every candidate, including a
genuine one, until someone removes the delta-step spills.

**P-D. Interleave.** Champion and candidate alternated, same session, same binary
pair, before any perf verdict; makes R2 a coin-flip per pair instead of a drift.

## Not done, and why

- P-A/P-B/P-C: acceptance or interface changes, per the brief.
- rocprofv3 instrument: crashes the MAX runtime today; recorded, not chased.
- Fresh cwd per run and read-protecting the fixture (R4, R6): small but they
  change where the engine writes its own `.work/draft-*.bin` and would race other
  tools reading the reference; wanted a decision on the sandbox shape first.
- B5 (basename allowlist): fix is one line but the honest version also drops the
  shim from `baro.kernel.files` at the next embed — an embed-format decision.
- 20-prompt P4 measurement for the loop: out of scope; the loop's metric is one
  prompt by its own frozen rule.
- The adversarial candidates' work dirs were overwritten by the re-run; verdicts
  are preserved in `gate-run.log`, per-run numbers in this report only.

## Update, 2026-09-08 (same night) — the maintainer said "do 1-4"; all four applied

The "Proposals" and "Not done" sections above are kept as written for the
audit; this is what changed after them.

**1. Stage 4 unblocked (P-C).** `tools/isa-spills.py` takes a census per kernel
family of `(.private_segment_fixed_size, spill_count)`; the gate builds the
iteration's own pristine sources once per run and fails only families whose
numbers rose or that are new and non-zero. Baseline at HEAD: 5 spilling families
of 29 (the delta-step variants plus one skinny matmul), all inherited.
Commits `e63dc77` (gate), `e9e3109` (rule amended in `bench/loop-protocol.md`,
iterations >= 006).

**2. Wall-clock plausibility (P-B).** Same commits. The gate times the champion
binary three times around the whole process; a candidate whose `tok/s_gen`
claims X seconds of decode saved must show >= 0.5·X in the gate's wall. One
correction found during the split dry run below: the term compared the
argument `CHAMPION_TOKPS` (another session's median) against the in-gate wall,
so a nonsense argument of 60 rejected a benign +0.15 % candidate for "0.567 s
saved". Fixed to the in-gate champion median, own commit (see "Commits").

**3. Iteration 006 with the placeholder skeleton** (`ddda3c4`, `75b7078`
`tools/loop-run.sh` = one GPU job). Result, full row in the protocol note:
example-echo 0/4 (was 8/14), real-line edits 3/4 (prediction >= 3/4). Of those
three, two edited profiling code and died at scope, one was a `# noqa` comment
that reached perf at +1.7 % against the closure median 65.91 — while the gate's
own build of the same sources measured 67.40 in-session. Survivors 0, six
iterations running. That +1.7 % no-op is R2 live and is the reason P-D below
stays open.

**4. Structural split (P-A).** `serve/engine.mojo` is now the harness only —
pack load, prompt, `t0`/`dt`, prints, draft receipt — and is **never
embedded**. The per-window body moved to `serve/window.mojo`, which is what the
proposer is shown and what the gguf carries; the embed list is the import
closure from `tools/embed-files.py`. The gate and `gguf-closure.sh` take the
harness from `git show <baro.kernel.commit>:serve/engine.mojo`, so a candidate
cannot supply it. Controls 1 and 3 of the ladder are now the same control; the
scope denylist for the split layout shrinks to profiling/print/file-write
tokens (`st.pos`, `cfg.e` are honest identifiers in `window.mojo`).
Commits `3a70336` `935d056` `0dbdc21` `3242573`. Verification:
- mega-gate on the split build: q8 / q8d / q4 × spec 0/1 identical, ref 64/64,
  launch 109.9 / mega 130.3 tok/s (`.work/mega-gate-split.log`).
- new gguf `…Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-3242573.gguf` (files attn,
  elementwise, matmul, matmul_prefill, matmul_skinny, mega, ssm, registry,
  window): `gguf-closure` rebuilds it from its own sources plus the git harness,
  PASS 64 tokens, 130.16 tok/s_gen (`.work/closure-split.log`).
- hardened gate on the split layout (`.work/loop/audit-split/gate-run.log`):
  a benign `window.mojo` view-swap passes scope, apply, compile, identity on
  both fixtures and all runs (130.4 vs 130.2 tok/s), and is rejected only at
  perf; `st.pos += 1000` in `window.mojo` compiles and fails identity at
  position 1 (expected 271, got 0). The stopwatch is no longer in any file the
  candidate can touch, so a timing-state edit has nothing to edit.

**Commits after the audit:** `e63dc77` `e9e3109` `ddda3c4` `75b7078` `3a70336`
`935d056` `0dbdc21` `3242573`, plus the plausibility-reference fix
(`git log -1 -- tools/loop-gate.sh`).

**Still open, the maintainer's call:**
- **P-D**, now the biggest hole: make the gate's in-session champion median the
  acceptance denominator (or interleave). Iteration 006 shows the closure
  median and the in-gate median differ by more than the +2 % step.
- Worth-it rule: 6 iterations, 0 survivors, no mechanism proposed yet. Note
  that with `BARO_MEGA=1` (the default the new gguf runs under) the launch-path
  regions the proposer edits are not on the executed path; iteration 007 needs
  a region/marker choice inside the megakernel path or `BARO_MEGA=0` in the
  gate. Design call.
- R4/R6 (fresh cwd, unreadable fixture) and B5 unchanged.
