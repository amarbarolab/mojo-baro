# Lane report: sampling on every served model

Brief `briefs/2026-09-16-sampling-all-models-lane.md`, sonnet `w82:p7`,
supervised by `w82:p1`. One section per item.

## Item 1: MoE verify

**Gap found before any check could run.** The dense draft-head logits dump
(`.work/draft-logits.bin`, what `kernels/test_sample_device.mojo`'s docstring
tells you to capture) is gated by `if not MEGA_ALLOWED: return` in
`serve/engine.mojo`, and `model_qwen35moe.mojo` sets `MEGA_ALLOWED = False`,
so that dump never fires for qwen35moe: every "capture" attempt just copied
a stale file. Fixed with a 20-line, env-gated addition, `bb636ef`
(`serve(engine): BARO_DUMP_LOGITS dumps the final target row on any
profile`): dumps `b.logits_d` row 0 (the exact buffer `argmax_k`/
`sample_row_k` read from in `serve/window.mojo`) when `BARO_DUMP_LOGITS=path`
is set, inert otherwise. `kernels/*.mojo` untouched. `./run-tests.sh` (100
kernels, 55 in registry, 0 orphans, exit 0) and `tools/ci-checks.sh` (all
non-GPU checks) both green after.

**Distribution test at MoE vocab, real data, non-peaked prompt.**
`kernels/test_sample_device.mojo` built unmodified with
`-D BARO_MODEL=qwen35moe` (VOCAB is 248320 for both profiles: same width,
real MoE-model values). Captured two real decode rows via the new dump
(`BARO_MEGA=0 BARO_SPEC=0 BARO_PACK=.work/moe-w1/pack`): `p17-summarize`
(raw T=1 top prob 0.939, 60-candidate tail, `rest 0.0428`) and `p20-dialog`
(top prob 0.993). p17 is the non-peaked one the brief asks for: its
untruncated gate2 chi-square has **df=27** (27 real candidate bins above the
expected-count floor), i.e. genuine spread across dozens of distinct tokens,
not the single-point distribution B5's seed-2 case hit. Independent oracle
via `tools/sample-nucleus-oracle.py` (numpy, touches neither
`kernels/sample.mojo` nor `serve/sample_ref.mojo`). Staged at the test's
hardcoded `.work/m5/logits-p01.bin`/`-p02.bin` paths (dense fixtures backed
up first, restored after, `.work/m5-dense-backup/`, verified byte-identical
on restore). Full run:

```
PASS mask, PASS greedy (T=0), PASS gate1 (device==host, 64/64 draws, every
config, both rows), PASS gate2 (chi2 under crit at p=0.001, both rows, every
config; p01 T1_k0_p1: chi2 21.877 df 27 crit 55.58), PASS gate3/gate4 near
+ far (spec accept/resample, both draft arms)
PASS: device sampler and speculative accept match serve/sample_ref.mojo
(C3-fixed) at real vocab
```

**Same seed reproduces, different seeds diverge.** Ran the built test twice
independently (`.work/sample-moe/test-run1.log`, `-run2.log`): every
chi-square, candidate count and accepted count is identical to 15
significant digits between runs (only `elapsed_s` differs). Gate1 fixes
seed=42 across 64 counters; gate2/gate4 use 20 distinct seeds (1000..1019)
per config. `./run-tests.sh`'s own host-side P-K4 check (VS=64 synthetic,
architecture-general, not MoE-specific) states the same property directly:
`PASS 2000 / 2000 same-seed reproduced; 1780 / 2000 changed under a
different seed`.

**Real HTTP request, temperature 0.7.** `.work/moe-w1/pack` ships no
`tokenizer.json` (RegesCore-35B shares the Qwen3.5 vocabulary with the dense
Qwythos pack: same VOCAB=248320 constant in both `model_qwen35.mojo` and
`model_qwen35moe.mojo`), so `baro-serve` was pointed at it with
`--tokenizer .work/engine-pack-q4/tokenizer.json` (the dense pack's file).
`/health` confirmed `"tokenizer":true`; the response text is fully coherent
English, which is the empirical proof the borrowed tokenizer is correctly
aligned to this pack's vocab (a misaligned vocab would not produce readable
language). Request/response:

```
POST /v1/chat/completions
{"messages":[{"role":"user","content":"Write one short sentence about the ocean."}],"max_tokens":40,"temperature":0.7,"seed":7}

{"choices":[{"finish_reason":"length","index":0,"message":{"content":"Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Topic:** The ocean\n   - **Format:** One short sentence\n   - **Constraint:** Keep","role":"assistant"}, ...}],
 "timings":{"decode_s":0.40469397,"finish":"length","prefill_rows":47,"prefill_s":0.460057353,"tok_s_gen":96.36911565546676},
 "usage":{"completion_tokens":40,"prompt_tokens":48,"total_tokens":88}}
```

**T=0 output identical to champion.** Built HEAD (`dd2479a`, before the
`BARO_DUMP_LOGITS` commit) and the champion commit `38ee0b7` (git worktree)
in the same stint, same pack, same prompt (`p03-story`), `BARO_MEGA=0
BARO_SPEC=0`: `GENERATED` tokens byte-identical 64/64 (`995 7157 780 4307
2199 383 279 2919 13 1216 557 264 855 314 2342 4105 ...` through `4600`),
`tok/s_gen` 110.36 (champ) vs 109.95 (HEAD), both consistent with the
recorded 111.89 20-prompt median (single-prompt, P4 instrument receipt not
the bar). `git diff --stat 38ee0b7..dd2479a -- kernels/ serve/` shows the
only launch-path-relevant change is a pure body-extraction refactor in
`kernels/moe.mojo` (`amar_moe_router_top8_sig` to `router_top8_sig_body`,
same call site, same signature); this run is the empirical proof it is
behavior-preserving, on top of R6.0/R6.0b/R6.1/R6.2's own already-recorded
20/20 identity receipts against the same commit (`docs/BASELINE.md`).

**End-to-end seed check, coordinator-requested addendum.** The seed
reproducibility claim above was on fixed logits inside the kernel test plus
the synthetic VS=64 P-K4 check, not through the full HTTP/engine path. Three
real `POST /v1/chat/completions` requests, same open-ended prompt (107
prompt tokens, well over 48), `temperature=0.9`, `max_tokens=30`, seeds
7, 7, 8:

```
prompt: "Write a short story that begins as follows and continue it in your
own words, describing what she finds: In the depths of an ancient forest,
where sunlight rarely touched the ground, a young explorer named Elara
discovered a hidden path that seemed to shimmer with an otherworldly light,
and she wondered where it might lead her next."

seed 7: [8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983,
         2570, 5396, 64700, 198, 256, 471, 2972, 36282, 64700, 10382, 3255,
         593, 25637, 198, 256, 471, 2972]
seed 7: [8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983,
         2570, 5396, 64700, 198, 256, 471, 2972, 36282, 64700, 10382, 3255,
         593, 25637, 198, 256, 471, 2972]
seed 8: [8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983,
         2570, 5396, 64700, 198, 256, 471, 2972, 52782, 64700, 9357, 264,
         2716, 3255, 6941, 440, 279, 3766]
```

The two seed=7 runs are byte-identical (30/30 tokens); seed=8 matches the
shared prefix through token 20, then diverges at token 21 (`36282` vs
`52782`) and every token after. Real end-to-end reproduce/diverge, not just
the kernel-level receipt.

**Item 1: all gates PASS. Named check for each sub-claim above; nothing
here is UNVERIFIED.**

## Item 2: Spark engine sampling (5 models)

**Wiring, `3d268d8` (`serve/spark.mojo` only).** `sample` was already parsed
by `serve_proto` (C3/A5) but never acted on. Hoisted its declaration outside
the serve/one-shot branch (default `temperature 0`, matching
`serve/engine.mojo`'s own one-shot convention, so one-shot stays exactly
the T=0 path it always was). At `temperature > 0`, `amar_sample_row`
(instantiated at the profile's own `VOCAB`, same pattern as
`registry.mojo`'s `sample_row_1`) replaces the `amar_argmax_part`/
`amar_argmax_final` pair; the chosen token lands via the existing
`amar_tok_copy` indirection. At `temperature <= 0` the pre-existing argmax
path, including the `BARO_FORCE` override, is untouched, byte for byte.
Also added `BARO_DUMP_LOGITS` (same convention as `bb636ef`) for the
distribution test below.

**T=0 forced identity, 20 prompts, `BARO_FORCE` (`bench/dense-run.sh`).**

| model | range | min agreement | matches 09-11 baseline |
|---|---|---|---|
| Llama-3.2-1B | 61-64/64 | 95.3% | yes (was 95.3-100%) |
| Qwen2.5-7B | 62-64/64 (one 14/14) | 96.9% | yes (was 96.9-100%) |
| granite-4.2-3b | 63-64/64 | 98.4% | yes (was 98.4-100%) |
| lily-7b | 62-64/64 (one 5/5) | 96.9% | yes (was 96.9-100%) |

Spark-X2.5-4B has no llama.cpp-comparable architecture, so this model's
T=0 identity is a self A/B instead: built the pre-change commit `44a1742`
and HEAD `3d268d8` in the same stint (`git worktree`), same pack, same
prompt. `GENERATED` bit-identical 64/64 (`614 44107 95 344 390 ...` through
`2908 7629 2692`), which is the direct proof the `temperature > 0` branch
never executes at `temperature 0`.

**Distribution test at each model's own vocab, real data.** New harness
`bench/sample-spark-device.mojo` (`4affcad`), same shape as
`kernels/test_sample_device.mojo`'s gate1/gate2 but parameterized on the
Spark profile module (lives in `bench/`, not `kernels/`, since
`kernels/*.mojo` is off limits to this lane; a real reason, not a
workaround: `test_sample_device.mojo` is hardwired to `registry`/`model`'s
qwen35 VOCAB, which does not exist in a Spark build). Real rows captured
via `BARO_DUMP_LOGITS` on `p17-summarize` (`p20-dialog` for Llama-3.2,
whose `p17` row was too peaked); independent oracle via
`tools/sample-nucleus-oracle.py`.

| model | VOCAB | gate1 (device==host) | gate2 max df / chi2 vs crit | greedy T=0 |
|---|---|---|---|---|
| Spark-X2.5-4B | 131072 | PASS all 3 configs | df=11, 10.58 < 31.43 | PASS |
| Llama-3.2-1B | 128256 | PASS all 3 configs | df=56, 57.93 < 94.54 | PASS |
| Qwen2.5-7B | 152064 | PASS all 3 configs | df=1, 0.016 < 11.16 | PASS |
| granite-4.2-3b | 100352 | PASS all 3 configs | df=32, 23.07 < 62.59 | PASS |
| lily-7b | 32000 | PASS all 3 configs | df=3, 0.34 < 16.55 | PASS |

All 5: `PASS: device sampler matches serve/sample_ref.mojo at real vocab`.

**Same seed reproduces, different seeds diverge.** Two checks, one honest
about its own limit: a single (seed=7, counter=3) draw repeated gave the
same token on all 5 models (reproduces), but the matching single
(seed=8, counter=3) draw also landed on the same token on all 5 (a real,
reported non-result, not evidence of anything, since one draw from a
peaked-enough config can coincide by chance). The real divergence evidence
is gate2 itself: 20 different seeds (1000..1019) per config, and the
resulting distribution matches the independent oracle (chi2 under the
p=0.001 critical value on every row above), which is only possible if the
draws are genuinely spread, not stuck on one seed's token. The end-to-end
HTTP seed check below is the decisive version of this same property.

**One real HTTP request per model, temperature 0.7, plus the
coordinator-requested end-to-end seed check (T=0.9, seeds 7/7/8, same
107-108 token open-ended prompt as item 1) on all five, not just MoE.**
Llama-3.2-1B is the one without a usable tokenizer (`meta-llama/Llama-3.2-1B-Instruct`
is HF-gated, confirmed: `hf download` returns "Access denied. This
repository requires approval."); served it via `/v1/completions` with the
prompt as a token-id array instead, which `serve/PROTOCOL.md` already
documents as needing no tokenizer. The other four fetched a real
`tokenizer.json` (Qwen2.5-7B-Instruct, granite-4.2-3b, and lily-7b's own
`segolilylabs/Lily-Cybersecurity-7B-v0.2` from HF; Spark-X2.5-4B already
had one on disk) and were served via `/v1/chat/completions`.

| model | route | T=0.7 response | seed 7/7 | seed 8 |
|---|---|---|---|---|
| Spark-X2.5-4B | chat | "The ocean is a vast and powerful force..." | identical 30/30 | diverges token 1 |
| Llama-3.2-1B | completions (ids) | 30 tokens, `finish:"length"` | identical 30/30 | diverges token 1 |
| Qwen2.5-7B | chat | "The ocean covers most of the Earth's..." | identical 30/30 | shares prefix through token 14, diverges token 15 |
| granite-4.2-3b | chat | "The ocean spreads endlessly..." | identical 30/30 | shares prefix through token 4, diverges token 5 |
| lily-7b | chat | "The ocean is a vast, mysterious body..." | identical 30/30 | shares prefix through token 1, diverges token 2 |

Every model: the two seed=7 requests are byte-for-byte identical token
lists, and seed=8 diverges (either immediately or after a shared prefix,
both are the expected shape depending on how much the position's own
distribution happens to favor one token).

`run-tests.sh`/`ci-checks.sh` green after every commit in this item.
`.work/sample-spark/*/pack` (29 GB, rebuildable from
`tools/engine-pack.py --dense` / `tools/spark-pack.py` + `tools/gen-profile.mojo`)
deleted after these gates were recorded, per the coordinator's disk note
(`/home` was at 95%).

**Item 2: all gates PASS on all 5 models. Named check for each sub-claim;
nothing here is UNVERIFIED.**

## Items 3-4: penalties and top-N logprobs, PASS (dense/MoE engine)

`presence_penalty`/`frequency_penalty` were already parsed into
`SampleParams` (C3/A5) but nothing read them; `serve/sample_ref.mojo`'s host
reference (`apply_penalties`) already existed and is tested. Device side is
fable's `amar_apply_penalties`/`amar_topn_probs` (`90353f8`, receipts
`exchange/lane-SAMPLE-kernels.md`); host wiring is `0e25b36` (+
`42c57b3`, an unrelated git-index-race fix, see below), commit message has
the full design. Summary:

- `serve/serve_proto.mojo` parses `top_logprobs` off the wire.
- `serve/window.mojo`, in the plain (non-spec) single-token decode step
  (`m == 1`, past the prompt): copies `toks_d[n_prompt:st.pos+1]` back to
  host, builds the same distinct-id/count list `sample_ref.apply_penalties`
  would, calls `amar_apply_penalties` on the logits row before the draw,
  then `amar_topn_probs` after (when `top_logprobs > 0`), and threads the
  chosen token's own probability (`amar_sample_row`'s `Prob` output,
  already penalty-correct since penalties ran first) plus the top-N list
  into the per-token line: `{"id":...,"tok":...,"logprob":...,"top_logprobs":[...]}`.
- `serve/harness.mojo`/`serve/registry.mojo` carry the new buffers and
  kernel instantiations.
- `serve/src/protocol.rs`/`engine.rs`/`main.rs`: `top_logprobs` on the wire
  `SampleParams`; `EngineMsg::Tok`/`Event::Tok` carry the optional logprob
  data; both `/v1/completions` and `/v1/chat/completions` (streaming and
  not) return an OpenAI-shaped `logprobs` object built from it, `null` when
  nothing was requested (a request with neither field gets byte-identical
  output to before this landed).

**Scope, stated plainly.** Only the plain non-spec single-token decode step
is wired: spec+sample+penalties (fable's per-row `Ids/Cnt` design supports
it, row `j` = shared history + drafts `0..j-1`) and `serve/spark.mojo` are
NOT wired. Named as open follow-ups, not silently skipped.

**Gates.**

1. Kernel-vs-`sample_ref`: fable's own (`kernels/test_sample_pen.mojo`, in
   `run-tests.sh`, 3 real 248320-vocab rows, both penalty kinds and spec
   rows with drafts): device bit-equal to the host reference, top-N ids
   equal to the host sort, probs within 4.2e-7 (bar 1e-5, brief asked for
   1e-4). `run-tests.sh` 102 kernels, 57 in registry, 0 orphans, exit 0.
2. T=0 unaffected: `tools/test_server.sh` ALL PASS (clippy clean, 25 cargo
   tests, `.work/pen-test-server/`: 64/64 token match on completion, SSE,
   queued, stop, cancel-recovery, reject, shutdown) plus a direct one-shot
   spot check, same GENERATED tokens as every earlier run this session.
3. **`frequency_penalty` visibly suppresses a repeated token, real HTTP.**
   `/v1/completions` against `baro-serve` + the MoE pack, same prompt/seed,
   `temperature=0.6`: unpenalized, token `2972` (a list-numbering marker)
   appears 3 times (positions 4, 13, 25) in 40 generated tokens; with
   `frequency_penalty=50, presence_penalty=50`, it appears once (position
   4) and never again, replaced by different tokens from position 13
   onward. (A moderate penalty, 1.5, produced no visible change on this
   model at this temperature: the MoE decode logits are extremely peaked
   here, as items 1-2 already found, so a small penalty doesn't clear the
   logit gap to the runner-up; 50 does. Recorded as a real finding, not
   hidden by picking a magnitude that "worked".) Body:
   ```
   POST /v1/completions {"prompt":[8160,579,264,7047,1817,25],"max_tokens":40,"temperature":0.6,"seed":5}
   -> tokens [...2972(pos4)...2972(pos13)...2972(pos25)...]
   POST /v1/completions {"prompt":[8160,579,264,7047,1817,25],"max_tokens":40,"temperature":0.6,"seed":5,"frequency_penalty":50.0,"presence_penalty":50.0}
   -> tokens [...2972(pos4)...561...1156...] (2972 never repeats)
   ```
4. **`top_logprobs`, real HTTP, chosen-token consistency.** `/v1/chat/completions`,
   `temperature=0.7`, `logprobs:true, top_logprobs:5`, 8 real generated
   tokens: every token's own `logprob` equals its `top_logprobs[0].logprob`
   with `top_logprobs[0].id` equal to the chosen token id (the invariant a
   wiring bug would break), values monotonically descending within each
   list, all `<= 0`. Full response body in the commit message; example
   entry: `{"id":8160,"logprob":-0.0535,"top_logprobs":[{"id":8160,"logprob":-0.0535},{"id":90700,"logprob":-2.956},...]}`.

**Items 3-4: PASS for the dense/MoE engine, scoped to the plain non-spec
decode step. Spec+sample+penalties and `serve/spark.mojo` are named
follow-ups, not done.**

## Coordinator review: NOT accepted, inert-parameter defect (P1) -- fixed, `02cab13`

The gates above were real, but they didn't prove the parameter wasn't
inert on a **default** request: my MoE-engine tests never hit the bug
because the MoE profile has no draft head (`spec` never engages) and no
megakernel head-fold (`comptime if not MEGA_ALLOWED: head_folded = False`
unconditionally), so both bypass paths below simply don't exist there.
Reading `window.mojo` again, on the **dense** engine (`MEGA_ALLOWED=True`,
where the champion actually runs): `BARO_SPEC` defaults to `1` and A1 lets
sampling compose with speculation, so a default request runs the `win_spec`
branch, which never reaches my penalty/logprob code at all; `BARO_MEGA`
also defaults to `1`, and `mega_req = mega and temperature <= 0` meant a
`T=0` request with penalties went through the megakernel, whose dispatch is
`if head_folded: pass` -- skips the entire win_spec/argmax/sample chain,
including mine. A client setting `frequency_penalty` on a default request
would have gotten it silently ignored: exactly the class `bench/PROTOCOL-RULES.md`
P1 exists to forbid (the decode-race `speculative.n_max` and hipBLASLt
precedents it cites are the same shape).

**Fix, `02cab13`.** `serve/engine.mojo`: a new `want_extra` flag
(`presence_penalty != 0 or frequency_penalty != 0 or top_logprobs > 0`)
forces `spec = False` and folds into `mega_req`/`mega_win_req`, so a
request carrying either now always lands on the plain launch path.
`serve/window.mojo`: the `T <= 0` branch now also checks `want_extra` and
routes into the sample path instead of `argmax_k` when set --
`amar_sample_row` is argmax-equivalent at `temperature <= 0` (P-K2), so
"penalize then draw" resolves to the penalized argmax rather than ignoring
the penalty. `serve/spark.mojo`: refuses `presence_penalty`/
`frequency_penalty`/`top_logprobs` with a named error (not wired there
yet) instead of silently accepting and dropping them. `serve/src/main.rs`:
fixed the error-to-HTTP-status mapping (no `Tok` received yet = 400, a
pre-GPU-work rejection per `serve/PROTOCOL.md`; after generation started =
502, a real mid-stream failure) so spark's new refusal is an actual 400,
not the 502 every earlier refusal in this codebase produced.

**Gates, all against the DENSE engine (`.work/engine-pack-q4`) with default
env (`BARO_SPEC`/`BARO_MEGA` both unset = `1`), the exact scenario that was
broken:**

- **Default spec + `frequency_penalty` visibly suppresses.** No-penalty
  request degenerates into a 6x repeat loop (`9338 13 198 760 6511 314`
  repeating); the identical request with `frequency_penalty=50,
  presence_penalty=50` has no repeat anywhere in 40 tokens, and its `done`
  line carries no `drafted`/`accepted`/`k` (spec correctly disengaged,
  confirmed from the field's absence, not inferred).
- **`T=0` + penalty behaves as chosen (penalize-then-argmax).** `T=0`
  without penalty vs `T=0` with `frequency_penalty=50,presence_penalty=50`
  on the same prompt/seed: identical through token 2, diverge from token 3
  onward.
- **`T=0`, no penalty: unaffected.** `tools/test_server.sh` ALL PASS
  (`.work/pen-test-server2/`), 64/64 token match on completion/SSE/queued/
  stop/cancel-recovery, identical to every earlier run this session.
- **spark refuses, real 400.** `/v1/completions` with `frequency_penalty`:
  `{"error":{"code":400,...}}`, `HTTP_STATUS:400`. `/v1/chat/completions`
  with `logprobs:true,top_logprobs:3`: same. A plain request on the same
  server: `HTTP_STATUS:200`.
- **Item 4's frozen gate, done properly: real host comparison, not
  internal consistency.** One live 25-token decode (`temperature=0.8,
  top_k=40,top_p=0.95,presence_penalty=0.6,frequency_penalty=0.4,
  top_logprobs=8`), each step's raw pre-penalty row + exact history dumped
  via a new `BARO_DUMP_LOGITS_DIR` hook in `window.mojo` (off by default).
  Independent numpy oracle (`.work/pen-verify/compare.py`, built the same
  way `tools/sample-nucleus-oracle.py` already is) applies penalties in
  `sample_ref.apply_penalties`'s own float form, computes the nucleus
  distribution, compares to the device's actual chosen-token logprob and
  top-8 list. **20/20 tokens PASS, max diff 5.89e-07 against the 1e-4
  bar.**

`run-tests.sh` 102 kernels, 57 in registry, 0 orphans, exit 0 (retried
under `gpu-wait` after an OOM collision with a concurrent lane's job, not a
regression); `ci-checks.sh` 0; `tools/test_server.sh` ALL PASS.

**Items 3-4: PASS, dense/MoE engine, default request shape included.**
Still scoped to the plain non-spec decode step (a request now correctly
FORCES that path rather than silently skipping penalties within it); a
future round could instead honor the caller's `spec:true` by filling
fable's per-row `Ids/Cnt` design for spec windows, named as the real next
step rather than done today. `serve/spark.mojo` refuses loudly until wired.

**Git-index race, `0e25b36`/`42c57b3`.** Committing this lane's files by
explicit pathspec still swept in another lane's in-progress `bench/`
deletion and edits (concurrent `git add` in the same checkout, not
protected by an explicit path list). Fixed same-turn, nothing lost, logged
`~/Brain/m.ledger/mojo-baro.md` 2026-09-16 and flagged to `w82:p1`.

## INTERFACE received from fable (`w82:p5`), items 3-4

Routed by the coordinator to fable, brief `briefs/2026-09-16-fable-sample-kernels.md`.
Delivered interface for both kernels:

- `amar_apply_penalties[XL,IL,CL,NL](X: [R,VOCAB] f32 in place, Ids: [R,CAP]
  i32, Cnt: [R,CAP] i32, Npen: [R] i32, n_vocab, presence, frequency)`,
  `grid_dim=R block_dim=256`. Sparse by design: the host keeps a per-row
  list of DISTINCT generated ids and their counts (no device `Counts`
  buffer, no bump kernel needed, simpler than what I proposed), uploads it
  fresh each call. `X[row,id] -= presence + frequency*Cnt[row,id]`, applied
  to raw logits before temperature/truncation, same float form as
  `sample_ref.apply_penalties`.
- `amar_topn_probs[XL,IL,PL,CAP=SAMP_CAP](X: [R,VOCAB], TopIds: [R,NMAX]
  i32, TopProbs: [R,NMAX] f32, n_vocab, nsel<=20, temperature, top_k,
  top_p, min_p)`, `grid_dim=R block_dim=SAMP_THREADS`. Returns the top-N of
  the truncated post-penalty distribution at the request temperature (the
  same distribution `amar_sample_row` draws from), sorted descending by
  logit, ties by lower id, unused slots `id=-1 prob=0`. At `temperature<=0`
  entry 0 is the argmax with an implicit prob of 1. Penalties are applied
  to `X` before either kernel runs.

Acknowledged to fable, with one scoping question sent: whether `Npen`/`Ids`
for a spec-window row `j > 0` needs the shared history merged with drafts
`0..j-1` by me, or whether row 0 (the plain non-spec path, which is what
M5's `temperature > 0` branch actually reaches until spec+sample+penalties
combine) is the only row this item needs to cover. **Holding the
`serve/window.mojo`/`serve/spark.mojo` call sites until the kernel symbols
exist in `kernels/sample.mojo`** so the host wiring compiles and can be
gated properly (P8: a change against a symbol that does not exist yet
cannot be verified, only guessed) rather than landing untested code against
an interface that may still move during implementation.

## Status at this point in the lane

**All four items landed and gated.** Items 1-2 cover all served models
(MoE + 5 Spark dense targets); items 3-4 cover the dense/MoE engine's plain
non-spec decode step, with spec-window composition and `serve/spark.mojo`
named as open follow-ups rather than silently skipped. Whiteboard ticked
per item as it landed (§3 LIVE RULE); `herd tell` sent for each landing
plus every coordinator/kernel exchange.
