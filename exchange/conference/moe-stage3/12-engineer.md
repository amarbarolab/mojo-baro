# Participant 12-engineer: MoE stage 3 prefetch

Framing: what breaks first, how would we know. I read `serve/expert_tier.mojo`,
`serve/window.mojo` (778-808), `bench/moe-tier-protocol.md`,
`bench/pcie-bandwidth.mojo`, and the upstream `max/gpu/host/device_context.mojo`
at `~/Projects/mojo/modular-ref/modular-main`. No brain-recall note matched
this exact question; nearest hits are generic MoE/paper-adjudication notes,
not this codebase.

## 1. Plan item shape, shown on W1

Each item gets: Files touched. The check (what makes it pass, run how).
The receipt (what gets printed or written, and where). The number that
means done (one number, with the arm it was measured on).

**W1: stamp timeline of one tier token**

- Files: `serve/expert_tier.mojo` (split the existing `fetch_ns` accumulator
  into named phases), `serve/window.mojo` around 786 (one more `perf_counter_ns`
  bracket around `b.tier.prepare`), no kernel file. Output through the
  existing `report()` print, extended, not a new tool.
- Check: sum of the per-layer phase times reproduces the already-printed
  `fetch_s` to within 1%, on the same 20-prompt run the B4 stage 2b receipt
  used (`.work/b4/g23main/`), same engine build.
- Receipt: one line per layer class (attn/moe layers differ; 34/38/39 are
  q6_k) printed by the engine itself, not a side file: readback-sync ns,
  pread ns, H2D-enqueue ns, LRU-touch ns, remainder. `gpu-wait run` log is
  the artifact.
- Number that means done: which single phase is over 50% of the measured
  25.6 ms/token round-trip cost (1.74 s over roughly 68 decode tokens on
  p01, not 1.74 s per token, see the correction below). That phase is what
  W2 must attack. If no phase clears 50%, say so and give the top two.

**Before writing this instrumentation: two phases already exist.**
`WindowBufs.pf_ffn` (`window.mojo:735,1429`) is a synced wall-clock stamp
around the whole FFN block including the tier call, and `ExpertTier.fetch_ns`
already isolates the tier's own cost inside that block. `pf_ffn - fetch_ns`
already answers "kernels vs tier" with zero new code. W1 should print that
delta on the first run before adding finer phases: if it already explains
most of the gap, the finer split may not be needed.

**Correction to the situation section, load-bearing for every later number.**
"`fetch_s` is 1.74 s of a 1.75 s decode" is the B4 report's phrasing for
prompt p01's whole decode run (roughly 68 tokens at 39.06 tok/s, about 1.74
s), not 1.74 s per token: a per-token fetch cost that large would be 25
tok/s from fetch time alone, and it isn't. Per token the round trip is about
25.6 ms across 40 layers, roughly 0.64 ms/layer average, of which the
frozen prediction's own transfer estimate is 5.81 ms/token (about 0.15
ms/layer). So roughly three-quarters of the per-layer round trip is not
PCIe bytes at 28.5 GB/s at all, it is host-side: `ctx.synchronize()`
latency, the Dict/List bookkeeping in `LayerLru.touch`, and the per-layer
`open()`/`close()` of `experts.bin` in `prepare()`
(`expert_tier.mojo:312,352`) that runs 40 times a token whether or not
there is a miss to fetch. That open/close is very likely cheap
(dentry-cached, low microseconds), but it is free to remove and currently
untested. W1 should measure it before ruling it in or out, not assume
either way.

## 2. What would make me fail

**W3 is under-specified in a way that changes the whole shape of W2, and I
can answer part of it from this repo without a GPU run.**
`bench/pcie-bandwidth.mojo`'s own docstring says `DeviceContext.enqueue_copy`
"has no stream parameter in this API": true, but incomplete. The upstream
Mojo GPU binding (`max/gpu/host/device_context.mojo` in
`~/Projects/mojo/modular-ref/modular-main`) does expose `DeviceStream`
(`ctx.create_stream()`), `DeviceStream.enqueue_function` (same
`_FunctionEnqueuer` trait `DeviceContext` uses, so kernel-launch call sites
can move to a stream with the same syntax), and `DeviceEvent`
(`record_event`/`enqueue_wait_for`) for cross-stream ordering. What does
not exist anywhere in that module: a stream parameter on any `enqueue_copy`
variant, or on `enqueue_copy_no_cross_stream_sync`. So a second stream
exists for compute, not for the H2D/D2H copies `_fetch_piece` and the
router-id readback use.

That means the buildable version of W2/W3 is the opposite of how the
wishlist states it: instead of moving the fetch onto a second stream, the
lane would move the trunk kernels of layer l onto a created stream and
leave the tier's copies on the context's default stream, then hand-place a
`DeviceEvent` at exactly two boundaries per layer: after the router kernel
(so the D2H readback of `hidx_d` doesn't start early), and after the H2D
prefetch lands (so the gate/up kernel of the next layer doesn't start
early). That is a real change to every `ctx.enqueue_function` call in the
decode path window.mojo owns, not a change local to `expert_tier.mojo`. It
is a bigger and more invasive edit than "host code only" suggests, and it
is shared machinery every other lane's identity gates depend on.

**This is the failure mode I'd flag loudest: a missing or misplaced
`enqueue_wait_for` does not crash and does not obviously fail identity.** It
reads a cache slot before its H2D copy has landed. Depending on scheduling
it may read the previous occupant's bytes (wrong but plausible-looking
logits, silent) or a torn write (also silent, GPU memory has no
read/write fence-violation trap here). The existing gate (W4, identity
20/20 at cap 64 and 128) only exercises whatever race window this exact
hardware and this exact 20-prompt schedule happens to hit; a cross-stream
ordering bug that doesn't fire on that specific run ships. I would insist
on a stress variant before trusting green here (see rules below).

**W3's fallback question ("a host thread that can enqueue copies
concurrently") is answerable but not for the reason asked.** Nothing in
`window.mojo` spawns threads; `prepare()`'s `pread` loop is single-threaded
host code that blocks the same thread that will next enqueue the gate/up
kernel. A host thread pool could issue the pageable `pread()`s ahead of the
GPU needing them, but only for a known set of expert ids: it does not solve
the "we don't know the ids until the router runs" problem any more than a
second stream does. It's a real option for pipelining known misses one
layer deep (fetch layer l's confirmed misses while layer l-1's kernels are
still running), not for the speculative same-layer overlap W2 describes.

**W1 as literally scoped ("Instrument before any change") is buildable
today, host-only, no blocker.** Flagged in Q1 above, not repeating here.

## 3. What the wishlist gets wrong

**W2(a), "previous token's picks for the same layer, 39.3% repeat," is not
a new prefetch source. It is already the cache's steady-state hit
mechanism.** If an expert stayed resident since the last token because
nothing evicted it, `LayerLru.touch` returns a hit with zero fetch, already
counted in the measured 0.7743 hit rate. There is no PCIe cost on that path
to hide by "prefetching" it; it's already free. Predictor (a) buys nothing
the existing LRU doesn't already buy. The real target for W2 is the roughly
22.6% (cap 64) or 17.3% (cap 128) of references that are genuine misses,
and those are unknown until that layer's own router runs, not the previous
token's, not the previous layer's.

**W2(b), predicting layer l+1's picks from layer l's router logits, needs
something that does not exist in this repo and is not free to invent.**
This model was not trained Pre-gated-MoE style (its own router at layer
l+1 only sees layer l+1's own hidden state, which is not available until
layer l's full FFN, both routed and shared, has run). Any "cheap host or
device projection" from layer l's logits to layer l+1's picks is a new,
unvalidated heuristic whose hit rate on this model is unmeasured. The brief
should not spend an engineering slot on stream plumbing before a cheap
offline check (replay the `1aa06d5` trace: how often does layer l's top-8
predict layer l+1's top-8, or a cheap linear readout of it) says whether
this is 60% accurate or 6%. If it's low, W2(b) is dead on arrival regardless
of how good the stream engineering is, and the offline check costs no GPU
time.

**W2(c), a device-side residency check, is the direction that doesn't need
prediction accuracy at all** (it just removes the host round trip for the
roughly 77% of references that are already hits), and per Q2 above it's the
one that most plainly needs a new kernel (a device-side tag lookup and slot
remap running where `Idx` currently gets rewritten by host code), which is
explicitly out of this lane's scope by W5. The wishlist offers three
candidates as parallel options; they are not equally weighted. One is
already subsumed by the existing cache, one is a research risk this repo
has no data on yet, and the most promising one is blocked by the lane's own
rule. Say this plainly in the brief rather than let the lane discover it
mid-build.

**P4 (moe-tier-protocol.md) result already puts a number on "kernels vs
tier" that W1 half-repeats.** The B4 report states 39.06 tok/s against a
62-71 tok/s transfer-only prediction and names the 40 host round trips as
the missing term. The report already knows the direction of the answer. W1
should confirm the split, not rediscover that a gap exists.

## 4. Prediction band for tok/s after W2, and the kill line

Two numbers bound this, both from measured receipts, not new arithmetic:

- **Floor: 39.06 tok/s** (current, cap 64). A W2 that doesn't actually
  overlap anything, or overlaps but adds its own new host-thread/stream
  bookkeeping cost, should not regress below this at cap 64. Any drop below
  39.06 is a straightforward failure, not a subtle one.
- **Ceiling: 111.89 tok/s** (full-pack champion, P14's own no-tier
  reference) is unreachable by construction. The tier still pays the true
  miss-path PCIe bytes (0.1656 GB/token at cap 64) that the full-pack
  engine never pays, and pays a genuine per-layer host readback that cannot
  be fully hidden because the router dependency is real: layer l+1's own
  hidden state must exist before its router runs, so layer l+1's fetch
  cannot start before layer l+1's router without a predictor, and Q3 above
  says the cheap predictors are either already exploited or unvalidated.

**My band, conditioned on which term W1 finds dominant:**

- If the roughly 19-20 ms/layer-block round trip is dominated by fixed
  per-layer host overhead (sync latency, Dict/List bookkeeping, open/close)
  rather than by bytes, removing it (batching the readback and bookkeeping
  cost, not necessarily true stream overlap) should recover most of the gap
  between 39.06 and the 62-71 tok/s that pure transfer cost predicted. Band:
  **58 to 68 tok/s** at cap 64 (I shade below the original 62-71 because the
  correction in Q1 shows the round trip has real components, sync and
  bookkeeping, that a naive "remove one sync" fix won't zero out either).
- If the round trip turns out to be genuinely PCIe/latency-bound (each of
  40 layers pays a real, unavoidable device round trip whose floor is set
  by ROCm/HIP dispatch latency, not by anything this lane can batch away
  without true concurrent streams), then only the harder W3 stream-plus-
  event rebuild reaches that band, and a softer fix (removing the
  open/close, tightening the Dict loop) buys single digits of tok/s,
  landing around **42 to 48 tok/s**.

**Kill line I would freeze: below 45 tok/s at cap 64 after W2, with the W1
stamp showing the round-trip time did not shrink by at least 30% from its
per-layer baseline.** Not a bare tok/s number alone: a tok/s gain with no
matching shrink in the stamped round-trip time means the gain came from
somewhere else (a different arm parameter, a build difference), and P1's
read-back should catch it before the number is trusted.

## 5. Rules I insist on verbatim

- **A stress identity run, not just the standard 20-prompt gate, before any
  W2/W3 change that touches stream ordering or cross-stream events is
  trusted.** Run the same 20 prompts at least 3 times at cap 64 (the only
  capacity where eviction, hence overwrite-in-flight, is possible) and
  require bit-identical `GENERATED` across all runs, not just against the
  full-pack reference once. A race that fires 1 run in 5 passes a single
  20/20 and ships.
- **W1's instrumentation is added to the existing `report()`/`pf_ffn` path,
  not a new side tool**, so the same P1 read-back that already covers the
  tier's startup print covers the new phase numbers for free.
- **Any speculative touch of the LRU (from a predictor, W2b) must not call
  `LayerLru.touch` on an id that was never actually used by a kernel.**
  `test_expert_tier`'s replay gate assumes every `touch()` call corresponds
  to a real reference from the `1aa06d5` trace. A speculative touch that
  turns out wrong changes recency ordering for real future references and
  silently invalidates what that gate is checking, even if the printed hit
  rate still looks plausible.
- **W2(b) does not get a stream/event engineering slot until an offline,
  no-GPU check of predictor (b)'s accuracy on the existing trace exists.**
  Cheap to falsify, and falsifying it first avoids building plumbing for a
  predictor that turns out to guess wrong two times in three.
- Standing rules from the brief (gpu-wait, preflight, P1/P4/P6, pathspec
  commits, no em dashes) apply unchanged; I have nothing to add to those.

## Round 2

**Agree.** Coordinator's denominator fix is right and it overturns my own
round-1 framing too: comparing 5.81 ms transfer against the full 1740 ms
`fetch_s` (what I did, and what 18/19 did) counts GPU compute that runs on
the full-pack engine anyway. The real gap is 16.7 ms/token (25.6 minus
8.94), and 5.81/16.7 is about 35%, not 0.33% or my own "23%." W0 (pinned vs
page-cache, zero code) is the right first move, cheaper than my round-1
suggestion of the same thing via `pf_ffn - fetch_ns`, and I withdraw that in
its favor. 19-builder's DeviceStream/DeviceEvent finding matches mine
exactly (`create_stream`, `wait_for_host_value` blocks the host the same
way `synchronize` does for a data-dependent read); that's now three
independent reads of the same upstream file agreeing, good enough to treat
as settled without a fourth check.

**Still reject 19-builder's §3 claim that the round-trip cost is
"structural... not proportional to misses."** Run coordinator's own numbers
through the two capacities: misses/token = 320*(1-hit_rate) gives 72.2 at
cap 64, 55.2 at cap 128, a 23.5% drop, matching the report's own "23%
fewer misses" line. Scaling the measured 16.0 ms page-cache-hot pread cost
by that ratio predicts 12.2 ms of pread at cap 128; add the unchanged 8.94
ms compute and the same small residual (16.7 - 16.0 = 0.7 ms) and the model
predicts 21.9 ms/token, 45.7 tok/s. Measured: 46.75. That is a
miss-proportional model fitting within 2% across both capacities, which is
the opposite of "fixed cost regardless of hit rate." Builder's own evidence
(cap 128 not closing the gap to 111.89) is real but doesn't distinguish the
two hypotheses; it's equally consistent with "pread cost that's still large
because 55 misses/token is still a lot," which the fit above shows it is.
This matters for scope: if pread dominates and scales with misses, W2(a)
(stage the previous token's predicted-repeat set into the pinned/staged
buffer before the round trip) is back in play as a real lever, not the dead
end 19-builder's §0 arithmetic implied, contingent on W0 first isolating
pread from sync/open-close.

**One format change I insist on:** every W1/W0 receipt reports its
bottleneck bucket normalized per miss (or per pread), in addition to
per-token ms and tok/s. cap 64 and cap 128 have different miss counts, so
two absolute ms/token numbers conflate "the mechanism got cheaper" with
"there was simply less work to do." A per-miss number (ms of pread per
missed expert-piece) is the one figure comparable across capacities and
across W0/W1/W2 arms, and it's what would have let round 2 settle the
structural-vs-proportional question from the existing B4 stage-2b table
alone, before any new run.

DONE

