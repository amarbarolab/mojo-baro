# 18-skeptic: MoE stage 3 prefetch wishlist

## 1. Plan item shape, shown on W1

One section per item, four fields, nothing else:

- **Files**: exact paths touched (source + the bench script that reads the
  receipt back).
- **Check**: the command that proves the item ran, and what "pass" prints.
- **Receipt**: the number(s) printed to a log under `.work/` or `exchange/`,
  not a claim in the report.
- **Done**: the specific number or inequality that closes the item; no
  adjectives ("faster", "better").

Applied to W1:

- **Files**: `serve/expert_tier.mojo` (`prepare()` gets `perf_counter_ns()`
  stamps around each phase: D2H id copy + `ctx.synchronize()`, the LRU
  `touch()` loop, the `pread()` calls, the H2D `enqueue_copy` calls); a new
  `bench/moe-tier-stamp.py` (or extend `bench/carryover-stamp.py`) to parse
  and sum; `bench/clock-probe.sh` sampled inside the loop, not once per run.
- **Check**: one 20-prompt run with `BARO_TIER_STAMP=1`, per-layer table for
  every layer of at least one prompt, not a token-wide total.
- **Receipt**: a printed table, `layer, sync_wait_ns, lru_ns, pread_ns,
  h2d_ns, sclk_before, sclk_after`, committed under `exchange/`.
- **Done**: the phase that is >50% of `fetch_ns` is named in the report by
  number, and the sum of the four phases is within 5% of the measured
  `fetch_ns` for that token. Otherwise the stamp is missing a phase and is
  not trustworthy evidence for W2.

## 2. What is unbuildable or under-specified

**W2(b) is not buildable as written.** "The current layer's router logits
for the next layer" is Pre-gated MoE's shape, and Pre-gated MoE trains an
explicit small predictor network per architecture to make that projection
cheap and accurate. Nothing in this repo has measured whether layer `l`'s
router logits correlate at all with layer `l+1`'s top-8 for qwen35moe.
`bench/moe-locality-protocol.md` measured same-layer, cross-token repeat
(39.3%): a different question. Building W2(b) without first extending the
offline replay (`bench/moe-locality.py`) to report cross-layer predictive
accuracy from the existing trace is building on an unmeasured premise,
which `bench/PROTOCOL-RULES.md`'s own preregistration spirit forbids: a GPU
round to find out a guess doesn't correlate is a round that should have
been a CPU replay first.

**W2(c) is blocked by W5 as written, and it is the candidate most likely to
matter (see §3).** "A device-side residency check so a HIT layer needs no
host sync at all" needs code that runs on the GPU comparing ids against a
resident set: a kernel change, or at minimum a new small kernel, under
W5's own definition ("any kernel edit... is a request back to the
coordinator"). As scoped, this lane cannot build the one candidate that
removes a synchronize call instead of just prefetching bytes behind one.
Either W5 gets a named exception for one small predicate kernel, or W2(c)
is dropped from this lane's scope and the report says so.

**W1 as scoped will not catch the failure mode this repo has already hit
twice.** Neither W1 nor W4 asks for clock/power state sampled *during* the
40 per-token round trips (only "for any timed arm" generally, per P1). See
§3 for why that is not optional here.

## 3. What the wishlist gets wrong

**It targets the wrong 0.33%.** The record already isolates transfer cost:
0.1656 GB/token at 28.5 GB/s is 5.81 ms. `fetch_s` is 1.74 s. Transfer is
5.81 / 1740 = 0.33% of the time the wishlist calls "the cost." W2 is framed
as "issue their H2D copies while layer l computes," hiding a copy. Hiding
100% of a 5.81 ms copy cannot explain, and cannot close, a 1.74 s gap. The
`moe-tier-protocol.md` report already names the real term correctly ("40
host round trips per token"), but W2's build description reverts to
transfer language anyway, which will send a host-only lane after the wrong
lever.

**A round trip is not a copy.** `serve/expert_tier.mojo:prepare()` does one
`ctx.synchronize()` per layer (line 319), and per the code's own comment at
window.mojo, that synchronize is also what makes the *previous* layer's
fetches safe to overwrite: it is the only sync point between two
`prepare()` calls, so it drains everything enqueued since the last one:
that layer's router kernel, the top-8 kernel, and, if the pipeline was
still catching up, trailing work from the layer before. Prefetching bytes
earlier does not remove this call or its wait; it only changes what the
call is waiting for. Only two things remove the wait itself: (a) never
issuing it (W2c, blocked by W5), or (b) making the thing it waits for
already true when it's issued, which requires the prediction to be right
and validated without another round trip, which is again a device-side
check.

**This repo has a documented history of exactly this failure shape and the
wishlist doesn't ask for the instrument that would catch it.** 40
enqueue-then-block-then-enqueue cycles per token, with real GPU-idle gaps
between bursts, is the same shape that produced [[single-wg-kernel-timing-noise]]
(same kernel measured 71 vs 640 us between arms) and the clock-ramp
inflation the dattn lane had to correct for in its 200-iteration receipts.
D1's own result was +3% throughput from a power-cap change alone (290 to
402 W) on this same box. If part of the 43.5 ms/layer average
(1.74 s / 40) is GPU clock deboost-then-ramp from 40 idle bubbles rather
than host CPU work or driver sync latency, no amount of host-side
prefetching touches it, because the bubble is inherent to stopping and
restarting GPU work 40 times a token. This is testable and cheap
(`bench/clock-probe.sh` sampled inside the loop per W1 above) and should
be run before anyone writes W2's overlap code, not after.

**W4's "honest hit rate" rule needs to be per-layer, not aggregate, for the
same reason.** A wrong guess under candidate (a) evicts something that may
have been about to be needed two layers later within the same LRU (cap 64
already evicts under pressure); an aggregate wasted-bytes number can hide a
predictor that is net negative on the layers where it matters (early-cold
layers, per the 19.5% cold-touch number in the locality trace) while
looking fine on average.

## 4. Prediction band for tok/s after W2, and the kill line

Arithmetic: transfer is already isolated at 5.81 ms of the 1740 ms
`fetch_s` (0.33%). W2, built host-only under W5, can at best remove the
transfer-wait portion of each round trip and let host bookkeeping
(`pread`, LRU `touch`) start slightly earlier relative to the previous
layer's trailing compute. It cannot remove the `ctx.synchronize()` call
itself (that needs W2c). So the honest ceiling for a host-only W2, with
candidate (a) only (candidate (b) is not buildable per §2), is: save at
most the 5.81 ms transfer plus a small, unmeasured fraction of host
bookkeeping overlap: call it another 1 to 3% of `fetch_s` optimistically,
not more, since bookkeeping is CPU work that still runs between two syncs
either way.

**Band: 39.06 to 41.5 tok/s (0% to +6%).** Not the 62 to 71 the
transfer-only model predicts, and not close to the 111.89 full-pack
number.

**Kill line I would freeze: measured tok/s after W2 below 41 tok/s (< +5%
over the 39.06 stage-2b baseline, re-measured fresh in the same session,
not reused from the stage-2b report per the reference-arm-identity rule).**
Below that line, the round trip, not the transfer, is confirmed as the
cost, and the next move is not a better predictor: it is taking W2(c) back
to the coordinator as a kernel-scope request, exactly as W5 says to.

## 5. Rules I insist on verbatim

- W1's receipt is per-layer, not per-token-total, and includes sclk/vddgfx
  sampled at least twice inside the round-trip loop (idle-before-sync,
  right after the next kernel dispatch) for at least one full prompt.
  Reason: this repo's own D1 and single-wg-kernel-timing-noise history.
- W2(b) (cross-layer router-logit prediction) does not get built until an
  offline replay extension shows cross-layer predictive accuracy on the
  existing trace; until then only W2(a) (same-layer, previous-token) is in
  scope.
- W2(c) (device-side residency check) is out of scope for this lane under
  W5. If W1's stamp shows the synchronize wait, not the transfer, dominates,
  the report says so explicitly and stops there rather than shipping a
  host-only change that cannot move the number, per the kill line in §4.
- The 39.06 tok/s baseline used to judge any W2 result is re-measured fresh
  in the same commit/session as the W2 arm, never reused from the stage-2b
  report, per "the reference arm IS an arm."
- W4's hit-rate honesty rule reports wasted bytes per layer, not only
  aggregated, so a predictor that is net negative on cold early layers
  cannot hide behind a good average.
- Every GPU job through `gpu-wait run --timeout`, `bench/preflight.sh`
  first, arm parameters read back (P1), 20-prompt medians (P4), no em
  dashes, commits by pathspec.

## Round 2

**Agree, and retracting a mistake.** My round-1 denominator was wrong: I
read `fetch_s` (1.74s of a 1.75s decode) as per-token cost. It is the whole
~68-token p01 run, not one token. The coordinator's correction is right:
tier-vs-no-tier is 39.06 vs 111.89 tok/s, an added 16.7ms/token, and
transfer (5.81ms) is roughly 35% of that 16.7ms, not 0.33% as I (and
19-builder, same error) wrote. That changes the weight of my §3 argument
but not its shape: something host-side still dominates the majority of the
gap. W0 (`BARO_TIER_PINNED=1`, zero code, skips every `_fetch_piece` pread)
is the right next move, ahead of building W1's finer buckets: the
coordinator's CPU-only measurement (217 preads, 16.0ms hot-cache) already
predicts where it lands, and it's one env var and one 20-prompt run to
confirm or kill that prediction tonight. I also now agree, with three of us
landing here independently, that W2(b) needs its own offline cross-layer
trace before any build, and W2(c) vs W5 is a real contradiction, not a
nuance; both are settled, not still open.

**Still reject.** A pinned W0 result that lands near 62-71 tok/s answers
"is pread the cost," not "is W2 as worded the fix." The wishlist's W2 text
("predict... and issue their H2D copies") still frames this as hiding a
copy; if pinned wins, the fix is skip-the-pread (pin the store, or stage
into the pinned buffer ahead of the sync), which is a different shape of
change than a byte predictor, and a positive W0 should not be read as
vindicating W2's original wording. Separately, 19-builder's cap-128
"structural, not per-miss" claim (§3 there) doesn't survive the same
arithmetic: 23% fewer misses producing +20% tok/s (39.06 to 46.75) scales
with miss count, which argues for a per-miss cost (pread) as coordinator
found, not a fixed per-layer sync/open-close cost as builder argued.

**Format change I insist on.** W0 gets its own item in the brief, Files
(none) / Check (20-prompt run, `BARO_TIER_PINNED=1` vs `=0`) / Receipt
(tok/s and the tier's own pinned/page-cache startup line) / Done (lands in
62-71 or stays near 39, each falsifying one hypothesis), placed before W1.
Right now it exists only in the coordinator's note, not the plan, and it is
the cheapest falsifier in this round: it should not be optional prose.

DONE
