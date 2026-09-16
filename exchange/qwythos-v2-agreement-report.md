# Qwythos-v2 vs llama.cpp agreement lane, report

Brief `briefs/2026-09-16-qwythos-v2-agreement.md`, sonnet `w82:p8`, supervised by `w82:p1`.
Preregistration `bench/qwythos-v2-protocol.md` (`8864701`), result committed `213e333`.

## Question

The model-library lane (2026-09-16) baked `Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.gguf` and flagged
it `UNVERIFIED-correctness`: no teacher-forced agreement receipt against llama.cpp existed for
the v2 variant anywhere in the repo. This lane produces that receipt, reusing
`bench/ornith-protocol.md`'s step 3a (agreement) and 3b (tok/s) exactly, scoped down per the
brief (G1-G3 and steps 3c/3d out of scope, already proven on this code path).

## Result

**Step 3a, teacher-forced agreement vs llama.cpp: median 99.2% (63.5/64), range 90.6-100%
(58/64 to 64/64)**, above the frozen 45-85% predicted band and past the 95% falsifier. My own
preregistered prediction was wrong: I widened the band below Ornith's 98.4% result on the theory
that CLAUDE.md's Qwythos f16-KV caveat ("llama.cpp's own f16-KV config fails its own f32 ref at
5/7 lengths") would carry into this check. It did not, this check uses q8_0/q8_0 KV on both
sides, not f16-KV, and the agreement number came in higher than Ornith's own, not lower. Stated
plainly per the protocol rather than rounded to fit.

**Step 3b, decode tok/s_gen (20-prompt median): ours 80.77 (range 80.38-80.96), llama.cpp 81.54
(range 80.36-82.54), ratio 1.0096x llama.cpp/ours.** `ours` matched the model-library bake's own
one-prompt receipt (80.82) to within 0.06%. `llama.cpp` landed just above my predicted 50-80 band
(inside the 35-100 falsifier). The prediction was built on a byte-width scaling argument: Q6_K
here carries about 1.36x the stream bytes of Ornith's Q4_K_M, so llama.cpp should decode markedly
slower on this bake than it did on Ornith. That is the same style of argument Ornith's own
protocol already flagged as unreliable (it predicted llama.cpp 110-150, got 88.8). Two rounds now
show llama.cpp's decode throughput on this 9B/33-layer shape landing within about 10% of `ours`'
q8 arm regardless of source quant format. Read as a fixed per-token cost dominating over
bits-per-weight bandwidth scaling at this model size; not traced further this lane.

Full read-back and reasoning in `bench/qwythos-v2-protocol.md`'s Result section.

## Verdict

Qwythos-9B-v2-MTP-Q6_K decodes correctly through the K-quant->q8 engine pack. The
UNVERIFIED-correctness flag is closed.

## Updated

- `bench/qwythos-v2-protocol.md` (preregistration + result), `bench/qwythos-v2-run.sh` (new,
  trimmed run harness), `README.md` model table (`213e333`, this repo).
- `~/Models/library/models/qwythos-9b-v2-mtp-q6_k__Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.json`:
  `correctness_verified: true`, `correctness_status` with the number (Models is not a git repo,
  plain file write).
- `~/Models/library/INDEX.md` correctness table row updated (same, plain file write).

## Notes

`serve/serve_proto.mojo` was uncommitted and modified in the checkout when I ran, another lane's
item 4 logprobs work in progress. Left untouched, committed by path only (`README.md`,
`bench/qwythos-v2-protocol.md`, `bench/qwythos-v2-run.sh`). `.work/qv2` (10 GB pack + run scratch)
deleted after the result was recorded. GPU queue was empty throughout, no contention with other
lanes; `/home` stayed at 150 GB free before and after.
