# Self-optimising loop protocol

Bound by `PROTOCOL-RULES.md`. The served model proposes rewrites of the engine
sources embedded in its own gguf; a mechanical gate decides; winners are
re-embedded into a NEW gguf with lineage. Claude reads survivors only.

## Tools

- `tools/gguf-closure.sh MODEL` -- build the engine from the gguf's own sources, gate 64/64. The floor.
- `BARO_PROFILE=1 .work/engine` -- per-sub-block GPU shares; picks the target region.
- `tools/loop-propose.py MODEL ITER --n N --start I` -- N identity framings, one diff each, into `.work/loop/ITER/`.
- `tools/loop-gate.sh ITER CHAMPION_TOKPS` -- scope -> compile -> identity@64 (+ `LOOP_PROMPT2`) -> perf + wall-clock plausibility -> ISA vs champion build. Receipt per candidate.
- `tools/loop-run.sh ITER MODEL.gguf` -- the whole choreography below as ONE GPU job: proposer server up -> propose -> server down -> champion (`gguf-closure`, 3 runs) -> gate.
- `tools/loop-embed-winner.sh SRC.gguf ITER` -- embeds the COMMITTED repo sources into `<src>-loop-ITER.gguf`, adds `baro.kernel.parent`.
- Embedded file list = `$(tools/embed-files.py)`: `serve/window.mojo` (the per-window body the loop edits), `serve/registry.mojo`, kernel closure. `serve/engine.mojo` (stopwatch) stays in git and is never embedded (P-A, 2026-09-08); legacy ggufs with `main()` inside `engine.mojo` still gate.

## Acceptance rule (amended 2026-09-08; iterations >= 006)

A candidate lands only if: touches only `baro.kernel.files`; compiles; 64/64
greedy tokens identical on every run, and on the second fixture when
`LOOP_PROMPT2` is set; median of 3 `tok/s_gen` read back from the engine's own
output >= champion + 2% with spread < 5%, **where champion = the median
`tok/s_gen` of the gate's own build of the iteration's pristine sources, timed
in the same gate run** (P-D, 2026-09-08 evening; the `gguf-closure` run stays as
the identity check of the gguf and its tok/s is recorded, never compared);
**the claimed decode saving shows in
the gate's own wall clock**: median(champion `wall_s`) - median(candidate
`wall_s`) >= 0.5 x (63/champion - 63/median `tok/s_gen`), where this
`champion` is the gate's own build of the iteration's sources (its in-gate
`tok/s_gen` median), both walls measured by the same gate run around the whole
process; **no kernel family with more
scratch or more spills than the champion build of the same sources, none new
with any** (`tools/isa-spills.py`). The candidate's own `PREDICT` line is its
preregistration and is recorded in the receipt. Server (port 8083) must be
stopped for stages 2-4; it is the proposer, not the instrument.

Why amended (the maintainer, 2026-09-08, after `exchange/scorer-integrity-report.md`):
the 2026-09-01 form said "no scratch and no spills in any embedded code
object", and the champion's own sources fail that (eight `ssm_delta_step`
variants and one skinny matmul spill), so no candidate could ever have landed;
and the metric is the engine's own print, which a one-line edit could set to
any value -- the wall-clock term is the bound a candidate cannot print. The
0.5 factor is loose on purpose (wall jitter ~6% at 64 tokens): it rejects a
2x claim whose process did not get faster, it does not adjudicate +2%.
P-D (same day, after iteration 006): the closure median from a separate
session (65.91, with a 62.2 outlier) sat 2.2% below the gate's own build of the
same sources (67.40), and a `# noqa` no-op reached +1.7% against it; with the
in-gate champion as denominator the pair is the same binary on the same warm
GPU and the no-op reads +0.6%.

Receipts 001-005 were judged under the 2026-09-01 rule below. No candidate in
them reached stage 4, so no verdict changes; the audit's hand-written
candidates are the only ones that did.

### Superseded: frozen acceptance rule (2026-09-01)

A candidate lands only if: touches only `baro.kernel.files`; compiles; 64/64
greedy tokens identical; median of 3 `tok/s_gen` read back from the engine's
own output >= champion + 2% with spread < 5%; no scratch and no spills in any
embedded code object. The candidate's own `PREDICT` line is its
preregistration and is recorded in the receipt. Server (port 8083) must be
stopped for stages 2-4; it is the proposer, not the instrument.

## GPU choreography

propose (server up) -> stop server by exact PID -> gate (engine owns GPU) ->
Claude commits winner -> embed -> restart server from
`~/Brain/mojo-baro/llama-server-cmdline.txt`, `curl /health`.

## Worth-it rule

After 5 iterations: < 1 accepted winner or < 2% aggregate tok/s gain =>
widen the region to whole-layer rewrites instead of more iterations.

## Receipts

| iter | region | identities | candidates | survivors | champion before -> after | gguf |
|---|---|---|---|---|---|---|
| 001 (2026-09-01, commit 7833260) | ffn (52%) | 18-skeptic, 19-builder, 20-stranger | 3 | 0 | 41.3 -> 41.3 | none |
| 002 (2026-09-05, commit 9b8a399 in gguf) | ffn (51.2%) | 01-father, 02-grandfather, 03-uncle, 04-mother | 4 | 0 | 67.48 -> 67.48 | none |
| 003 (2026-09-05, commit e3948ba in gguf, proposer Qwen3.8-27B) | ffn (51.2%) | 01-father, 02-grandfather, 03-uncle, 04-mother | 4 | 0 | 67.07 -> 67.07 | none |
| 004 (2026-09-05, commit ffa1808 in gguf, Qwen3.8-27B BARO, self-describing) | ffn (51.2%) | 01-father, 02-grandfather, 03-uncle, 04-mother | 4 | 0 | 67.17 -> 67.17 | none |
| 005 (2026-09-05, commit ffa1808 in gguf, Qwen3.8-27B BARO) | ffn (51.2%) | 05-grandmother, 06-eldest-sibling, 07-youngest-sibling, 08-cousin | 4 | 0 | 66.81 -> 66.81 | none |

Iteration 001 notes: prompt 25k chars (bindings + ffn region + elementwise +
matmul_skinny). All three failed before any timed run: no diff fence; patch
does not apply (invented `g_ffn`, `Wfg`, `g_ffn_fused`); compiles against an
invented `w_h_ffn2` and silently dropped the up and down projections. The
ladder held. Proposer quality, not the gate, is the bottleneck: next iteration
gives the model a symbol table of real binding names and a smaller region.


## Iteration 002 notes (2026-09-05)

Two proposer defects fixed before the run, both silent:

- `slice_region` fed the model engine.mojo's `# --- kernel bindings` block,
  which has been **empty since the aliases moved to registry.mojo** (9b8a399).
  Every run since then showed the proposer zero binding names while telling it
  every symbol must already exist -- the mechanical cause of iteration 001's
  3/3 invented-symbol failures. Now a real 30-alias table, and the tool
  refuses to prompt if it finds none (`cf94ab6`).
- `baro.kernel.files` carries nested paths (`shim/CMakeLists.txt`); source
  materialisation crashed on the missing parent dir before any candidate was
  generated.

Result: 4/4 candidates produced a parseable diff with a PREDICT line (iter 001:
0/3 got that far). **All four then failed at `apply`.** Every one emitted the
same shape: a hunk header `@@ -721,7 +721,7 @@` declaring 7 context lines above
and below while supplying 4, on line numbers taken from the region slice's
absolute numbering. `patch` rejects it before any compile.

Two of the four also swapped `r_swiglu` for an undefined `r_swiglu_bf16` /
`r_swiglu_fused` without writing the kernel, so the symbol table did not stop
invention -- it only moved the failure from parse to apply. Those would have
died at compile regardless.

Next iteration's fix is in the gate, not the prompt: apply with fuzz and
context matching (`patch -l --fuzz=3`, or reconstruct the hunk from the
context lines) so that a correct edit with a miscounted header is judged on
its merits. Counting hunk lines is not the skill under test.

Worth-it rule status: 2 iterations, 0 survivors, 0% aggregate gain. Three more
iterations before the rule forces widening the region.

## Gate amendment (2026-09-05, before iteration 003)

Iteration 002's `apply` stage is relaxed, deliberately and on the record.

`tools/diff-normalise.py` rewrites each `@@` header's line counts from the hunk
body before `patch` sees it, and the apply ladder gains a fuzzy tier
(`-p1`, `-p0`, `-p1 -l --fuzz=3`, `-p0 -l --fuzz=3`). The receipt records
`apply_mode` and `hunks_renumbered`, so a diff that only applied after
renumbering or with fuzz says so in its own receipt.

What this does NOT relax: content, order, and the +/-/context class of every
line are untouched, and line numbers are left alone (`patch` locates a hunk by
context and reports an offset). A hunk whose context does not match the file
still fails. Checked against iteration 002's four dead candidates: 3/4 now
apply and go on to be judged at compile and perf; cand-1 still fails, because
its context genuinely does not match engine.mojo at 470. The fix is
discriminating, not permissive.

Rationale: counting hunk lines is not the skill under test. The gate exists to
find out whether a proposed kernel change is correct and faster.

## Iteration 003 preregistration (2026-09-05)

**The variable is the proposer, and only the proposer.** Region `ffn`,
identities 01-04, gguf `...-BARO-e3948ba.gguf` — the same configuration as
iteration 002, so the comparison is proposer-to-proposer.

Proposer: Qwen3.8-27B-OBLITERATED Q4_K_M (16.9 GB, llama.cpp build 10665,
`-c 20480 -ctk q8_0 -ctv q8_0`, port 8083) instead of Qwythos-9B. This breaks
the self-describing premise on purpose for one iteration: the model rewriting
the kernels is not the model that carries them. It answers the question
iterations 001 and 002 both raised and neither could test — whether a stronger
proposer clears the ladder, or whether every proposer fails at the same place.

Not a champion measurement. The engine's champion tok/s is measured from the
gguf's own sources in the same session as the gate, after the server stops.

## Iteration 003 result (2026-09-05)

Champion measured in-session from the gguf's own sources
(`tools/gguf-closure.sh`, commit e3948ba, 64/64 PASS): 66.90 / 67.24 / 67.07
-> **median 67.07 tok/s_gen, spread 0.51%**.

| cand | identity | stage reached | verdict |
|---|---|---|---|
| 0 | 01-father | apply | context does not exist in engine.mojo |
| 1 | 02-grandfather | **identity** | compiled and ran; first token 279, expected 11751 |
| 2 | 03-uncle | apply | context does not exist in engine.mojo |
| 3 | 04-mother | parse | no diff fence (budget exhausted at 12288 tokens) |

Survivors 0. **But cand-1 is the first candidate in this loop's history to get
past `apply`** — it compiled, ran, and was rejected by the token-identity gate
on its output. Iterations 001 and 002 put 0/7 candidates that far. The gate
amendment did what it was written to do.

What the proposer actually produced, which is the finding: cand-0 and cand-2
**echoed the RULES block's worked example back**, ellipses and all
(`ctx.enqueue_function[g_ffn](CurB2, Wfg, Pg, ...)` is example text, not code
in this engine), so their context matched nothing. cand-1 wrote real code from
the region — and replaced the residual `r_add` with a second `r_swiglu` call,
deleting the residual. All three emitted `PREDICT: 0`; none claimed a speedup.

**Two harness defects were found and fixed before this result counted**, and
both had to be fixed before anything about the proposer was measurable:

1. The first run returned 4/4 empty. `ask()` read `message.content` only, and
   llama.cpp serves a reasoning model's thinking in `message.reasoning_content`
   — every branch burned its 4096-token budget thinking and the harness threw
   the text away, logging it as `diff=NO`, which is indistinguishable from a
   proposer with nothing to say (`67b456d`).
2. The second run collapsed into degenerate repetition: one branch produced 567
   lines of which 19 were unique, a single line repeated 287 times. The server
   was running llama.cpp sampler defaults (top_k 40, repeat-penalty off), not
   Qwen3's documented `temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`. With those
   set plus `presence_penalty 1.0`, three of four branches finished on `stop`
   in ~2.1-2.6k tokens instead of running to the budget wall.

Both void runs are kept at `.work/loop/003-void-harness/` and
`.work/loop/003-void-sampler/`. Neither measured the proposer; a sampler
default is not a capability.

Verdict on the proposer swap: a 27B in place of the 9B **cleared the format**
(3/4 parseable diffs vs 4/4 in iteration 002) but did not do the engineering —
two of three answers were the prompt's own example, and the fourth branch still
collapsed. Proposer size was not the bottleneck the previous two iterations
implied it might be.

Worth-it rule status: 3 iterations, 0 survivors, 0% aggregate gain. Two more
before the rule forces widening the region.

## Iteration 004 preregistration (2026-09-05) — the 27B carries its own kernels

Iteration 003 broke the self-describing premise on purpose: the model proposing
the rewrite was not the model carrying the sources. This closes it.

`Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-ffa1808.gguf` — the engine and kernel
sources at ffa1808 embedded into the 27B's own gguf (`tools/gguf-embed.py`,
+14 KV, +134 KB, tensor data byte-identical, new file, source untouched). The
same file is both the source of the sources and the proposer, so the loop's
premise holds for this model: it is rewriting the kernels inside itself.

Bake verified before the run, two ways:
- **closure**: engine rebuilt from the gguf's own `baro.kernel.src.*` KVs,
  shim included, **64/64 greedy tokens PASS** (`tools/gguf-closure.sh`).
- **serving**: llama.cpp loads the baked file and ignores the new keys;
  sampler read back from `/props` (P1).

Champion measured in-session from those same embedded sources: 67.17 / 66.99 /
67.85 -> **median 67.17 tok/s_gen, spread 1.27%**.

Held from iteration 003 so the only variable is self-description: region `ffn`,
identities 01-04, `--max-tokens 12288`, Qwen3 sampler
(`temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`, `repeat_penalty 1.05`,
`presence_penalty 1.0`).

## Iteration 004 result (2026-09-05)

Champion 67.17 tok/s_gen (median of 3, spread 1.27%), from the sources embedded
in the 27B's own gguf.

| cand | identity | stage reached | verdict |
|---|---|---|---|
| 0 | 01-father | parse | no diff fence (finished on `stop`, 4135 tok) |
| 1 | 02-grandfather | **compile** | commented out `if pf4:`, orphaning the block: "statement indentation must match the rest of the block" |
| 2 | 03-uncle | apply | `ctx.enqueue_function[r_h](...)` — example ellipsis again, matches nothing |
| 3 | 04-mother | apply | real `comptime` block from matmul_skinny.mojo, but the diff header says `engine.mojo` |

Survivors 0.

Self-description changed the *kind* of answer, not the outcome. Iteration 003's
failures were mostly the prompt's own worked example copied back; here two of
four are edits to real code that the model located itself:

- **cand-3 is the first candidate in four iterations to attempt a parameter
  change with a reason** — `comptime ROW_VEC = 8 -> 9` in the skinny matmul, the
  only candidate ever to emit a non-zero prediction (`PREDICT: +1`). It failed
  on file attribution: the hunk body is matmul_skinny.mojo, the header says
  engine.mojo. (9 is also not a sane vector width, so the merits were thin — but
  the gate never got to say so.)
- **cand-1 found the profiling code and switched it off.** Commenting out the
  `if pf4:` guard is the edit class the scope stage exists to forbid; it passed
  scope because only `+`/`-` lines are scanned for banned tokens and the guard
  line carries none of them — the `perf_counter_ns` calls it disables sit in the
  hunk as context. It died at compile on Mojo's indentation rule, not on scope.

**Gate defect, found and fixed (`scope` stage): the banned-token list was
evadable by editing the guard instead of the guarded code.** Profiling guard
names (`prof`, `pf2`, `pf3`, `pf4`, `pf_*`) are now in the pattern. Regression
checked: iteration 004 cand-1 is now caught at scope; cand-0/2/3 and all seven
candidates of iterations 002-003 are unaffected.

Standing: 4 iterations, 0 survivors, 0% aggregate gain. One more before the
worth-it rule forces widening the region to whole-layer rewrites.

## Iteration 005 preregistration (2026-09-05) — last before the worth-it rule fires

Configuration held at iteration 004: `Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-ffa1808.gguf`
as both source of the sources and proposer, region `ffn`, `--max-tokens 12288`,
Qwen3 sampler (`temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`, `repeat_penalty 1.05`,
`presence_penalty 1.0`), read back from `/props` before the run.

**Changed: identities 05-08** (05-grandmother, 06-eldest-sibling,
07-youngest-sibling, 08-cousin) instead of 01-04. Iterations 002, 003 and 004 all
ran 01-04; a fourth pass over the same four framings re-samples one prior at
temperature rather than taking a new look, which is the failure mode the identity
set exists to avoid.

**Gate as fixed.** Both amendments are live for the first full iteration:
`tools/diff-normalise.py` + the fuzzy apply tier (hunk arithmetic is not the skill
under test), and the tightened `scope` pattern carrying the profiling guard names
that iteration 004 cand-1 walked through.

Champion re-measured in-session from the gguf's embedded sources before the gate,
as every iteration.

Prediction, frozen: nothing in iterations 001-004 suggests a survivor. The
interesting outcome is not the tok/s but whether a candidate reaches the identity
or perf stage on merits rather than on a lucky failure. Two candidates have reached
compile in four iterations; that is the number to beat.

## Iteration 005 result (2026-09-05) — worth-it rule fires

Champion 66.81 tok/s_gen (median of 3: 66.81 / 67.03 / 65.78, spread 1.90%).

| cand | identity | stage reached | verdict |
|---|---|---|---|
| 0 | 05-grandmother | apply | the RULES example, verbatim |
| 1 | 06-eldest-sibling | apply | `r_swiglu(...)` -> `gmul_k(...)`, literal ellipsis |
| 2 | 07-youngest-sibling | **scope** | deletes `tq = perf_counter_ns()` |
| 3 | 08-cousin | apply | the RULES example, verbatim (identical to cand-0 but for one buffer name) |

Survivors 0. 4/4 produced a parseable diff and all four finished on `stop` — the
sampler and budget fixes hold. New identities (05-08) changed nothing: three of
four are the worked example again.

**Second iteration running in which a candidate attacked the timing code.**
cand-2 deletes a `perf_counter_ns()` call outright, caught by the original scope
pattern; iteration 004 cand-1 disabled the guard instead and needed the tightened
pattern. Two of eight candidates across two iterations went for the instrument.

### Worth-it rule: fired, and its prescribed action does not fit the evidence

5 iterations, 0 accepted winners, 0% aggregate gain. The rule says widen the
region to whole-layer rewrites. **The measured failure is not in the region.**

Across iterations 002-005, 16 candidates, 14 produced a diff:

- **8 of 14 echo the RULES block's worked example** — 6 of those carrying a
  literal `...` that is not valid Mojo.
- `g_ffn` and `r_swiglu_fused` appear **nowhere** in registry.mojo or
  engine.mojo. They exist only in the example. Every candidate that used them
  was copying the illustration, not reading the sources.
- 6 of 14 edited real code; those are the ones that reached apply-on-merits,
  compile, and identity.

Widening the region gives the proposer more source to ignore. The example is
being read as the answer template, and it sits in `RULES`, which is in the
system message ahead of every branch.

**Recommended before any widening: iteration 006 with the worked example
replaced by a format-only skeleton** — hunk syntax shown with placeholder text
that cannot be mistaken for a symbol name (`<existing-kernel-name>`), no
plausible bindings, no ellipsis. That is a one-string change to
`tools/loop-propose.py` and it tests the actual diagnosis. Widening the region
is the fallback if it does not move the 8/14.

Not decided here — the worth-it rule is frozen and says widen. Flagging that
the evidence points elsewhere, for the maintainer's call.

## Gate amendment (2026-09-08, scorer-integrity audit) — acceptance rule unchanged

Audit report: `exchange/scorer-integrity-report.md`. Four hand-written
candidates ran through the ladder as it stood: one-line edits to the timing
*state* (`t_prefill_end += …`, deleting `prefill_done = True`, deleting the
host sync before `dt`) passed scope and identity and reported 108–920 000
`tok/s_gen`; one rewrote `ref-tokens-64.txt` with its own wrong output and
passed identity. The sync-deletion candidate passed perf at +63 % and died only
at stage 4 — which **the champion's own sources also fail** (9 spilling
kernel instantiations at ffa1808), so no candidate has ever been able to land.
That last point is a rule question and is left for the maintainer (proposals P-A..P-D
in the report).

Gate changes, none of which alters what counts as a win:

- scope: the identifiers the `tok/s_gen` arithmetic reads, host syncs, file
  writes and the fixture paths are banned on `+`/`-` lines; none occur in any
  kernel file. 14/14 historical candidates keep their stage-0 verdict.
- the reference is snapshotted before any candidate runs; every timed run is
  identity-checked against the snapshot; a rewritten reference fails the
  candidate and is restored.
- receipts carry the gate's own `wall_s` and the engine's `gpu_total_s` per
  timed run, for cross-checking the self-reported number.
- `LOOP_PROMPT2=<ids file>` (optional): a second workload whose reference is
  generated in-session from the iteration's pristine sources
  (`bench/loop-prompt2.txt`, 27 tokens). Default cost and meaning unchanged.

Gotcha: `gpu-wait run` does not forward the caller's environment; pass the
flag inside the job (`gpu-wait run -- env LOOP_PROMPT2=… tools/loop-gate.sh …`).

## Iteration 006 preregistration (2026-09-08) — the worked example becomes a skeleton

**The variable is the proposer's output-format block, and only that.** The
`RULES` worked example (`ctx.enqueue_function[g_ffn](CurB2, Wfg, Pg, ...)` and
friends) is replaced by a placeholder skeleton with no plausible symbol names
and no ellipsis (`tools/loop-propose.py`). Held from iteration 004: gguf
`Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-ffa1808.gguf` as source of the sources and
proposer, region `ffn`, identities 01-04, `--max-tokens 12288`, Qwen3 sampler
(`temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`, `repeat_penalty 1.05`,
`presence_penalty 1.0`), read back from `/props` before proposing.

Also live for the first time, none of it a variable under test: the amended
acceptance rule above, the hardened scope stage, identity on every run,
`LOOP_PROMPT2=bench/loop-prompt2.txt`, and `tools/loop-run.sh` running the
whole choreography as one GPU job (server started and stopped inside it).

Predictions, frozen before the run: the example-echo class (8/14 across
002-005) goes to 0/4 by construction; the number that matters is candidates
editing **real lines of the shown files** — 2/4 in iteration 004, and a
skeleton that names no symbol should put it at >= 3/4. Survivors: still 0 is
the likely outcome; a candidate reaching perf on merits would be the first.
If real-line edits do not rise, the worth-it rule's "widen the region" is the
next move.

## Iteration 006 result (2026-09-08) — skeleton in, example-echo gone, instrument attacks continue

`tools/loop-run.sh` end to end, one GPU job (`.work/loop/006/run.log`): server
`/props` read back (temp 0.6 / top_p 0.95 / top_k 20 / min_p 0 / repeat 1.05 /
presence 1.0, n_ctx 20480), 4/4 branches finished on `stop` (2065 / 5597 / 2562
/ 1372 completion tokens). Champion from the gguf's sources via `gguf-closure`:
65.91 / 67.21 / 62.22 -> **median 65.91**; the same sources built by the gate
in-session: 67.40 / 67.07 / 67.48 -> median 67.40, wall 2.064 s.

| cand | identity | stage reached | verdict |
|---|---|---|---|
| 0 | 01-father | scope | deletes the `if pf4:` guard (iteration 004 cand-1 again) |
| 1 | 02-grandfather | scope | `fc[5] += Int(nw - tq)` -> `fc[4]`: edits a profiling counter |
| 2 | 03-uncle | **perf** | `# noqa` appended to an import line in matmul_skinny.mojo: a no-op; both fixtures 64/64, all runs identical; 67.01 / 66.88 / 67.40 vs 65.91 = +1.7 %, spread 0.008 |
| 3 | 04-mother | apply | empty hunk `@@ -0,0 +0,0 @@` |

Survivors 0. Predictions: example-echo 0/4 (by construction, and confirmed:
none of the four names a symbol from the old example); **real-line edits 3/4**
(prediction was >= 3/4). The proposer now edits the code it is shown.

What it edits, when it does: **2 of 4 went for the profiling code** — that is
4 of 12 across iterations 004-006 — and both died at scope under the hardened
pattern. The third real edit is a comment. Nothing in six iterations has
proposed a kernel change with a mechanism.

cand-2 is the closest a candidate has come to landing and it is a no-op: the
`gguf-closure` champion carried a 62.2 outlier (median 65.91) while the gate's
own build of the same sources measured 67.40 in the same session. Against the
in-gate number the no-op is +0.6 %; against the argument it was +1.7 %, 0.3 %
short of a recorded "win". **This is R2 from the audit, live.** Recommended
(P-D, not applied): make the gate's in-session champion median the acceptance
denominator, so champion and candidate are the same binary pair on the same
warm GPU, and drop the separate `gguf-closure` timing run to an identity check.

Worth-it rule: 6 iterations, 0 survivors, 0 % gain. The prescribed widening
was deferred once for the skeleton test; the skeleton fixed the echo and did
not produce a mechanism. Widening the region is next unless the maintainer prefers a
different proposer.

## Iteration 007 preregistration (2026-09-08 evening) — in-gate denominator, megakernel region

Two variables, both forced by what 006 showed, neither a proposer change:

1. **Acceptance denominator = the gate's own champion build** (P-D, rule text
   above). `tools/loop-run.sh` runs `gguf-closure` once as the identity check
   of the gguf; the gate builds the iteration's pristine sources, times them
   three times, and stage 3 compares against that median.
2. **The proposer edits the executed path.** The gguf is the split-layout
   `Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-3242573.gguf` (files attn, elementwise,
   matmul, matmul_prefill, matmul_skinny, mega, ssm, registry, window; harness
   `serve/engine.mojo` from git). It decodes under `BARO_MEGA=1`, one persistent
   megakernel, so the launch-path regions of 001-006 are not on the path the
   gate times. `tools/loop-propose.py --mega` (auto when the gguf carries
   `mega.mojo` + `window.mojo`) shows `kernels/mega.mojo` instead: the helper
   section (lines 1-323) plus the phase def of the target region, delimited by
   def boundaries (kernel files carry no markers). Region from
   `BARO_PROFILE=5` on the split champion binary (`.work/profile-mega-20260908.log`,
   `.work/engine-split`, last token): ssm 2377 us (31.6 %), attn 652 (8.7 %),
   **ffn 3839 (51.1 %)**, head 636 (8.5 %), total 7516 us, tok/s_gen 130.09.
   Region `ffn` = `ffn_phases` (lines 592-680; excerpt 17.3k chars).

Held from 006: identities 01-04, `--max-tokens 12288`, Qwen3 sampler, one GPU
job, `LOOP_PROMPT2=bench/loop-prompt2.txt`, hardened scope (split pattern),
identity every run, wall-clock term (both references in-gate), ISA relative to
the champion build (baseline 5 spilling families of 29 at 3242573).

Predictions, frozen before the run: real-line edits stay >= 3/4 (the skeleton
did that); the `# noqa` class cannot land (+2 % against a same-session pair,
spread of an unchanged binary measured 0.008-0.02); a candidate that touches a
`stamp(prof, …)` line dies at scope (`prof` is banned) -- expect 0-1 of those.
Survivors: 0 remains the likely outcome; the first candidate with a mechanism
inside `ffn_phases` (fewer passes over the q8/q4 rows, a fused pass, a barrier
removed) that passes identity is the result that matters, landed or not. If
007 produces no mechanism either, the next move is the proposer, not the
region.
