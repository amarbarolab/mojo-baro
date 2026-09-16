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

**Item 1: all gates PASS. Named check for each sub-claim above; nothing
here is UNVERIFIED.**

## Item 2-4

Not started.
