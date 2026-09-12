# Lane KSAMP report: device sampler kernels

Branch `lane-KSAMP` (worktree `$HOME/Projects/mojo/mojo-baro-lanes/KSAMP`), plan item
KSAMP of `~/Brain/mojo/mojo-baro/briefs/2026-09-11-chat-engine-next.md`. Preregistrations and
results: `bench/chat-protocol.md`, sections KSAMP, KSAMP-b, KSAMP-c (three build rounds, each
frozen by commit before its build, each closed with its numbers).

**Verdict: gate PASS, correctness proven on every preregistered check. Speed is recorded,
not claimed: one of the frozen time bands held (greedy). The realistic presets run at 69 us
per call, 7 % over their band. Two presets tripped the falsifier. The round is closed with
the levers listed.**

## Flags for the reader (read first)

1. **Interface addition, one kernel beyond the plan.** `amar_spec_accept` needs the target and
   draft probability rows, and nothing in the fixed interface produced them, so
   `amar_sample_probs(X, P, n, temperature, top_k, top_p, min_p)` writes the truncated,
   tempered distribution. It shares the sampler's cut code, so the rows the acceptance rule
   sees are exactly the distribution `amar_sample_row` draws from.
2. **Comptime parameter `CAP: Int = 2048`** on `amar_sample_row` and `amar_sample_probs`
   (LDS candidate capacity of the fast path). Callers binding `[XLayout, OLayout, PLayout]`
   are unaffected; the test sets `CAP = 16` to force the general path.
3. **`amar_spec_accept` draws its own uniform** from Philox (stream 1 of the same seed/counter
   convention) instead of taking one as an argument, so a (seed, counter) pair reproduces the
   accept decision and the resample.
4. **One test fixture corrected before its first passing run.** The third Philox KAT's last
   word was transcribed from memory as `24f7f839`; Random123's vector is `24126ea1`. The
   implementation matched the other three words exactly, which a wrong Philox cannot do, and
   the zero and all-ones vectors matched in full. No test was weakened.
5. **Timing harness changed mid-lane (P6).** Single 1000-launch means were not receipts for a
   one-workgroup kernel: the same kernel on the same row read 71 us back-to-back and 640 us
   synced in one process, clock pinned high. Every arm is now 11 blocks of 100 launches with
   min/median/max. The first two rounds' single-mean timings carry that caveat in the protocol.
6. the maintainer's mid-lane correction (MTP opt-in, `BARO_SPEC=0` default) changes nothing here: the
   acceptance kernel is built and tested for when `BARO_SPEC=1`. No engine wiring, greedy
   path untouched.

## Interface (as built, `kernels/sample.mojo`)

Rows are `X: [R, V] f32`, one 1024-thread block per row, the same shape as `amar_argmax_row`.

- `amar_sample_row[XLayout, OLayout, PLayout, CAP=2048](X, Out[R] i32, Prob[R] f32, n,
  temperature, top_k, top_p, min_p, seed: u64, counter: u64)`. The token, plus its probability
  under the distribution it was drawn from. `temperature <= 0` returns `amar_argmax_row`'s token
  with probability 1. A row with no finite logit returns token -1, probability 0.
- `amar_sample_probs[XLayout, PLayout, CAP=2048](X, P[R, V], n, temperature, top_k, top_p, min_p)`.
- `amar_spec_accept[PLayout, TLayout](Pt, Pd, Dtok, Out, Acc, n, seed, counter)`. It accepts the
  drafted token when `u * p_d(x) < p_t(x)`. Otherwise it emits a draw from normalised
  `max(0, p_t - p_d)`, falling back to `p_t` if that residual is all zero. `Acc` is 1 or 0.

Semantics, llama.cpp order: top-k, then top-p within the top-k set at T = 1, then min-p
(`p >= min_p * p_max`), then temperature. Ties are broken by lower index, so every cut is a
prefix of one total order. RNG: Philox4x32-10 with key = seed and counter = (counter lo/hi,
row, stream << 28 | element / 4). The draw is Gumbel-max keyed by element index. The token
therefore does not depend on thread count or on which cut path ran.

## Gate

Command (through the waiting room, hold line re-read first, `.work/KSAMP-gate.txt`):

```
gpu-wait run --vram 12 -- bash -c './run-tests.sh; echo "run-tests.sh exit $?";
  ./.venv/bin/mojo build kernels/test_sample.mojo -o .work/test_sample -I kernels;
  SKIP_S=0 bench/clock-probe.sh bash -c "./.work/test_sample; echo test_sample exit \$?"'
```

| receipt | value |
|---|---|
| `run-tests.sh` | exit 0 |
| `test_sample` | exit 0, `PASS: device sampler` |
| gpu-wait | exit 0, queue empty before the run |
| kernel census before (`main` 38a85c7, `.work/KSAMP-gate-before.txt`) | 82 kernels, 38 in registry, 0 orphans; 14 test files |
| kernel census after | **85 kernels** (+`amar_sample_row`, `amar_sample_probs`, `amar_spec_accept`), 38 in registry, 0 orphans; **15 test files** |
| clock during the timed arms | sclk 3305-3311 MHz, 107-121 W |
| ISA (`tools/isa-receipt.py .work/test_sample`) | 0 scratch, 0 spills; 102-103 VGPR (`amar_sample_row`/`amar_sample_probs`), 33 (`amar_spec_accept`) |

## Correctness (final kernel, `.work/KSAMP-gate.txt`)

- **Philox KAT:** zero, all-ones and pi vectors equal Random123's.
- **Temperature 0 vs `amar_argmax_row`:** 13/13 rows equal at V = 248320: 8 Gaussian, ties,
  +0/-0, mostly -inf, NaN every third element, all -inf. Sampled at T0.7/k20/p0.8 on the same
  rows, every token is finite and inside the top 20 (the tie row forces the general path at the
  real V). The all -inf row gives -1/0.
- **Distribution, 10,000 draws per config at V = 64, Pearson chi-square vs exact float64, bins
  pooled to expected >= 5, critical value at p = 0.001:**

| config | chi2 | df | critical | draws outside the truncated set | max prob error |
|---|---|---|---|---|---|
| T1, k off, p off | 83.79 | 53 | 90.65 | 0 | 4.2e-9 |
| T0.7, k20, p0.8 | 8.51 | 11 | 31.43 | 0 | 1.1e-8 |
| T1.3, k12, p0.9, min-p 0.05 | 10.12 | 9 | 28.06 | 0 | 3.8e-9 |
| T0.5, k off, p0.6 | 7.46 | 9 | 28.06 | 0 | 1.0e-8 |
| ties, T1, k12 (cut inside a tie group) | 11.80 | 11 | 31.43 | 0 | 5.0e-9 |
| ties, T0.9, p0.3 (mass cut inside ties) | 6.33 | 6 | 22.68 | 0 | 6.4e-9 |

  `amar_sample_probs` rows match exact within 1.1e-8 and sum to 1 within 6e-8.
- **Reproducibility:** same (seed, counter) gives 10,000/10,000 equal tokens (two configs). A
  different seed changes 9,518/10,000 (flat) and 8,946/10,000 (Qwen preset).
- **Speculation, mismatched draft** (different logits, T0.9/k24/p0.95, drafts drawn by
  `amar_sample_row` from `p_d`): acceptance 0.5394 against exact `sum min(p_t, p_d)` = 0.5373
  (sigma 0.0050, 0.4 sigma off). Accepted-or-resampled tokens vs `p_t`: chi2 29.15, df 21,
  critical 46.92, 0 outside. Direct `p_t` draws: chi2 20.17. Spec vs direct two-sample: chi2
  23.50, df 21, critical 46.92, so speculation on and off are indistinguishable.
- **Path identity:** fast path vs forced general path (`CAP = 16`) gave 10,000/10,000 identical
  tokens and bit-equal probabilities in all 8 comparisons.

## Time per token at V = 248320

One row (R = 1), row hot in L2 as in the engine (the lm-head writes it just before). Medians of
11 blocks x 100 back-to-back launches, min-max in brackets. The Gaussian row is sigma 2.5 with
five tokens boosted to 10-14. The peaked row, closer to a real LM row, has the same bulk with
the boosted five at 20-24.

| row | preset | us/call | frozen band |
|---|---|---|---|
| any | greedy (T = 0) | **9.2** (9.1-9.3) | 5-10, held |
| peaked | Qwen T0.7/k20/p0.8 | **69.4** (69.4-70.4) | 35-65, missed by 7 % |
| gaussian | Qwen T0.7/k20/p0.8 | **69.4** (69.2-70.8) | 35-65, missed by 7 % |
| gaussian | llama.cpp T0.8/k40/p0.95/min-p 0.05 | 120.3 (117.6-121.4) | 35-65, missed |
| peaked | llama.cpp T0.8/k40/p0.95/min-p 0.05 | 145.5 (143.5-148.5) | 35-65, falsifier (2.2x) |
| peaked | k off/p 0.95 | 179.5 (177.6-181.6) | 35-65, falsifier (2.8x) |
| gaussian | plain T1 (no truncation) | 129.1 (128.0-130.0) | 80-110, missed |
| gaussian | k off/p 0.95 (nucleus = the bulk, general path) | 490.1 (487.9-492.3) | none |
| gaussian | `amar_argmax_row`, reference | 129.3 (128.7-133.9) | - |

Against a decode token of about 7.4 ms (135 tok/s), the Qwen preset costs about 0.9 % per
sampled token. Where the time goes (phase stamps, `.work/ksamp-diag/phases-c2.txt`):

- pass A takes ~11-14 us and the 16K subsample ~8-10.
- The llama and k-off presets go over because they retry the compaction pass (2-3 tries at
  ~32 us each) when the subsample estimate lands under the window the exact check accepts.
- One compaction costs 2.5x what the same loop shape cost in KSAMP-b (unexplained, ISA diff
  not done).
- The final pass takes 22-37 us.

Levers not taken, in order:

1. ISA-diff the compaction loop.
2. Widen the estimated window one step for min-p/top-p so one compaction suffices.
3. Keep two loads in flight in the compaction and final passes.
4. The general path (flat rows with top-k off) still costs ~480 us.

## Evidence

- `.work/KSAMP-gate.txt`: final gate (run-tests, test_sample, clock probe, exits).
- `.work/KSAMP-gate-before.txt`: baseline census and run-tests on `main`.
- `.work/KSAMP-run2.txt`: first build, the P-K1..P-K6 receipts.
- `.work/KSAMP-b-run.txt`: KSAMP-b receipts.
- `.work/KSAMP-c-run.txt`: KSAMP-c correctness receipts.
- `.work/ksamp-diag/phases.txt`, `phases-c.txt`, `phases-c2.txt`: phase stamps and
  back-to-back vs synced timing.
- `.work/ksamp-diag/mkdiag.sh`: derives the stamped kernel copy (uncommitted by design).

## Commits on `lane-KSAMP` (ahead of `main`, no attribution trailers, no `.venv`/`.work` in the diff)

| commit | what |
|---|---|
| `40f476a` | bench(ksamp): preregister device sampler predictions and gate |
| `21fe9e4` | kernels(sample): device sampler, probs row and speculative acceptance |
| `ee45582` | bench(ksamp): record first-build result, preregister KSAMP-b |
| `921ce0e` | bench(ksamp): record KSAMP-b result against its bands |
| `9d74668` | bench(ksamp): phase-timer diagnosis, preregister KSAMP-c |
| `512294d` | kernels(sample): sampled window, exact compaction, warp reductions (KSAMP-c) |
| `6f84075` | test(sample): block-median timing harness; record KSAMP-c result |

Files touched: `kernels/sample.mojo`, `kernels/test_sample.mojo`, `bench/chat-protocol.md`,
`docs/KERNELS.md` (census output). `kernels/sample.mojo` carries no comments or docstrings.

Unasked finding: at this vocabulary `amar_sample_row` at T = 0 returns `amar_argmax_row`'s exact
token in 9.2 us against `amar_argmax_row`'s own 129.3 us. The difference is 1024 threads with
float4 loads against 256 scalar-strided ones. The greedy path is out of this lane's scope.
