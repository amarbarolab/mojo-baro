# Sampled speculation (A1)

Preregistered 2026-09-15, before the first line of wiring, at commit `d1ec7a7`
(`docs/NEXT-PLAN.md` A1). Binds `bench/PROTOCOL-RULES.md`. The C3 tail round
(`4209cc4`) landed first, which is what unblocks this: the 53-bit Gumbel key
removed the 2^-24 floor that made the untruncated distribution wrong, so the
residual draw `norm(max(0, p - q))` can be trusted at real vocab.

## The rule being implemented

Leviathan 2211.17192 and Chen 2302.01318, with truncation applied to both
distributions before the rule:

- accept draft token `x` with probability `min(1, p(x) / q(x))`,
- on the first rejection, emit one draw from `norm(max(0, p - q))`,
- after the last accepted draft, emit a bonus token from `p`.

`p` is the target's truncated, temperature-scaled distribution for that row;
`q` is the draft head's, truncated with the same parameters. Both already
exist as code: `kernels/sample.mojo::amar_spec_accept` does the accept and
the residual draw on device, `kernels/sample.mojo::amar_sample_probs` fills a
truncated probability row, and `serve/sample_ref.mojo::spec_accept_ref` is the
host reference. **This round calls those kernels, it does not edit them**
(kernel files belong to the coordinator; the accept kernel already uses the
C3 `gumbel2` key).

Today `serve/engine.mojo` forces `spec = False` whenever `temperature > 0`,
because accepting a draft on argmax equality under sampling silently changes
the output distribution. That line is what this round replaces.

## What changes

`serve/window.mojo`'s verify window, and the spec/window config in
`serve/engine.mojo`. The A5 lane owns `serve/src/*`, `serve/serve_proto.mojo`
and the request-validation block of `serve/engine.mojo`; this round owns
`serve/window.mojo` and the spec config, and commits by path.

The draft head's own logits must survive to verify time, because the residual
needs the whole `q` row and not just `q(x)`: `blk32_forward` currently reduces
each draft row to an argmax in `dtok_d`.

## P1 read-back, before any timed run

- engine sha256, built in the same command as the run.
- the run's own echo: `BARO_SPEC`, `spec k`, `temperature`, `top_p`, `top_k`,
  `min_p`, `seed`, `prompt tokens`, `tokens`.
- for spec rows, `drafted` / `accepted` / `k` from the done line, which is
  what proves the window actually speculated rather than falling back.
- power cap, vddgfx offset, sclk from `bench/clock-probe.sh`.

## Gates, frozen

1. **T=0 unchanged, byte for byte.** `bench/force-ab.sh` reference against a
   build of this round, 20 prompts, plus `ref-tokens-64`. Anything other than
   20/20 identical is a failure of this round, not a property of sampling:
   the T=0 path must not be touched.
2. **Device accept equals the host reference per draw.** Fixed seeds, real
   vocab, `kernels/test_sample_device.mojo` shape: for each of three real
   logit rows, every draw's accept/reject decision and emitted token from
   `amar_spec_accept` equals `spec_accept_ref` on the same seed and counter.
   Exact equality, not a distributional check.
3. **Distribution.** 20,000 draws per row, three real rows, chi-square of the
   emitted tokens against the target distribution `p`, at T=1 and at
   T=0.7 / top_p 0.9, threshold p = 0.001. The same test compares the
   speculative path against direct sampling from `p` (two-sample), because
   the point of the rule is that they are the same distribution.
4. **Throughput.** 20-prompt median (P4) at T=0.7 / top_p 0.9 with spec on,
   against the same build at T=0.7 with spec off, in one stint. Frozen
   prediction: the speculative gain at T=0.7 is **within 5% of the T=0 spec
   gain** measured in the same stint, and acceptance at T=0.7 is **between
   0.7x and 1.0x** of the T=0 acceptance on the same prompts. A gain below
   half the T=0 gain means the draft is being rejected by the rule far more
   than temperature alone explains, and is a defect to trace, not a number to
   publish.
5. **The refusal.** `serve/engine.mojo`'s `temperature > 0 => spec = False` is
   removed in the same commit that passes gates 1 to 3, and the untruncated
   shape's own refusal (C3) is already lifted at `4209cc4`; this round
   re-checks that a `top_p = 1, top_k = 0` request with spec on is served and
   not refused.

## Falsifiers

- Gate 1 failing at all: the T=0 path was touched; revert and re-do.
- Gate 2 failing while gate 3 passes: the kernel and the reference disagree on
  individual draws and agree in aggregate, which is the sampler defect class
  this repo has already shipped once (`exchange/2026-09-15-m5-sampler-diagnosis.md`).
  Aggregate agreement does not excuse it.
- Gate 4 below half the T=0 gain: acceptance under the rule is far worse than
  under argmax equality; report the acceptance numbers and stop, rather than
  tuning k until the number looks better.

## Result

Filled in when the gates run. Nothing here is a claim until then.
