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

## Item 3: penalties, UNVERIFIED, blocked on the coordinator

Host reference already exists and is tested (`serve/sample_ref.mojo`'s
`apply_penalties`, part of the standing `run-tests.sh` suite: "PASS token 9
penalized to 6.5 ; untouched token 8 stays 8.0"). `presence_penalty`/
`frequency_penalty` are already parsed into `SampleParams` (C3/A5) but
nothing downstream reads them yet. The device side is a kernel change per
the brief's own routing rule, so it went to `w82:p1` as a `KERNEL:` message
with a concrete interface proposal (confirmed delivered, coordinator status
`working`): `amar_apply_penalties(X: [R,VOCAB], Counts: [R,VOCAB] i32, n,
presence_penalty, frequency_penalty)` applied before truncation/softmax,
plus a small `amar_bump_count` to update `Counts` once per generated token,
both keeping the per-request history off the host (the same class of cost
B4 already measured as the expensive one). Not started beyond the message:
the host-side per-request `Counts` buffer allocation and the bump call site
depend on whatever the coordinator actually lands, and building against a
guessed layout risks landing the wrong thing.

**Item 3: UNVERIFIED. Gate (device-vs-sample_ref check with penalties on,
plus an HTTP request where frequency_penalty visibly suppresses a repeated
token) cannot run until the kernel lands.**

## Item 4: logprobs, PARTIALLY SCOPED, not started

Chosen-token logprob needs no kernel change: `amar_sample_row` already
returns the drawn token's own probability (`window.mojo`'s plain sampled
path already computes it into `b.hmax_d` scratch, `serve/spark.mojo`'s new
`Sprob` scratch from item 2 the same way) and nothing currently copies it
back or surfaces it. `top_logprobs N` does need a kernel (shipping the
whole VOCAB-width row to host every token to sort it there is the same
expensive-round-trip class item 3 avoids), so it rode the same `KERNEL:`
message as item 3, proposed as `amar_topn_probs` reusing `sample.mojo`'s
existing radix-select machinery.

**Item 4: UNVERIFIED, not started.** Wiring chosen-token logprob through
both engines' line protocol, `serve/serve_proto.mojo`, the Rust HTTP layer
(`serve/src/*.rs`) for both endpoints plus SSE, and the OpenAI response
shape is real multi-file surgery I have not attempted yet; reported
honestly as not done rather than claimed and left unverified.

## Status at this point in the lane

Items 1-2 landed and gated, all receipts above. Item 3 blocked on the
coordinator's kernel delivery (message sent, in progress). Item 4 scoped
but not started. Whiteboard ticked per item as it lands (§3 LIVE RULE);
`herd tell w82:p1` sent for each landing plus the KERNEL request.

## Item 3-4

Not started.
