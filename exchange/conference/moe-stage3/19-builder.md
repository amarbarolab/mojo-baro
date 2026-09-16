# 19-builder: what resists first

Framing: what resists first? Fears elegant, unbuildable plans.

## 0. The number the wishlist has not looked at

`fetch_s` is 1.74 s of a 1.75 s decode token. The PCIe transfer inside that
same token is 5.81 ms. **Transfer is 0.33% of the time the tier spends in
`prepare()`.** Every candidate in W2 (a, b, c) is a predictor for which
BYTES to move earlier. Even a perfect predictor, moving 100% of the bytes
off the critical path, removes at most 5.81 ms from a 1750 ms token: 39.06
tok/s to about 39.2. That is the arithmetic ceiling of "prefetch the bytes"
read literally, and it is nowhere near 62 to 71.

So before I build anything: W2 as phrased is solving a problem that is
0.33% of the measured cost. The other 99.67% is inside `prepare()`'s own
scaffolding, not inside PCIe. Candidate (c), "a device-side residency check
so a HIT layer needs no host sync at all," is the one item in W2 that
actually addresses the dominant cost, and W5 forbids exactly the kernel
change that would let a HIT be detected without a host round trip (the
slot lookup lives in `self.lru[layer].slot_of`, a host `Dict`; the kernels
only ever see slots, never ids, so residency can't be checked device-side
without giving a kernel the Dict or its device-side equivalent). That is a
contradiction inside the wishlist itself, not a detail: W2(c) and W5 cannot
both hold. Flagging this is the single most important thing in this file.

## 1. Plan item shape

One section per item, five fields, shown on W1:

```
### W1: stamp timeline of one tier token

Files: serve/expert_tier.mojo (prepare(), _fetch_piece()), bench/carryover-stamp.py
  (existing site/kernel registry, extend rather than reinvent), a new
  bench/moe-tier-stamp-protocol.md preregistering the buckets below.

Change: readback sync, file open/close, per-piece pread, per-piece enqueue_copy,
  LRU touch() (hit and miss separately), final id writeback -- one
  perf_counter_ns() pair per bucket per layer, accumulated same as fetch_ns
  is now, printed per-layer AND summed per-token. No GPU kernel touched (W5).

Check: sum of buckets equals fetch_ns to within 1%, on the SAME run
  (a leak into an unmeasured 9th bucket is a bug, not noise).

Receipt: per-layer bucket table for one 64-token request (40 layers x
  6 buckets), plus the token-level sum, printed to a file under
  .work/moe-tier-stamp/ and pasted into the report.

Number that means done: which single bucket is >=50% of fetch_ns. That
  number is what W2 and W3 get built against; without it they are aimed
  at whichever bucket the coordinator's prose guessed at (transfer), which
  section 0 above already shows is wrong.
```

Same shape for W2/W3/W4: files touched, the check that would catch the
item lying to itself, the receipt (a path, not a terminal paste per §4b),
the one number that means the item is done. I will not start an item whose
"number that means done" is missing from the brief; W1 has one now because I
just wrote it, W2 to W4 don't yet (see §2).

## 2. What would make me fail

- **W1 is buildable as written.** It reuses `bench/carryover-stamp.py`'s
  site/kernel registry pattern, which is already the repo's answer to
  "measure inside a stream with no host sync." `prepare()` is host code
  with explicit `perf_counter_ns()` calls already (`fetch_ns`), so this is
  finer buckets on an existing timer, not new instrumentation machinery.
  Low risk.

- **W2 is under-specified in the one way that matters: it names three
  predictors before W1 has said what's being predicted for.** If W1 shows
  the sync/bookkeeping bucket (not transfer) dominates, then "predict and
  prefetch" is the wrong shape of fix entirely -- the fix is "touch the LRU
  and resolve slots for fewer layers per host trip," which is an algorithm
  change to `prepare()`'s call structure, not a predictor. I cannot build
  W2 from the wishlist as written; I need W1's bucket table first, and then
  a REVISED W2 written against whichever bucket actually dominates. Building
  predictor (a) or (b) now, before that number exists, is exactly the
  "elegant, unbuildable" failure mode this framing line warns about: it
  would ship a prefetcher that measurably cannot move the number past
  ~39.2 tok/s and only find that out after the GPU proof, at which point the
  report has to explain why a correctly-built item failed its own kill line.

- **W2(b), "the current layer's router logits for the next layer," is not
  supported by any measurement in the repo.** `bench/moe-locality-protocol.md`
  measured ONE locality axis: same layer, consecutive tokens (39.3%). It
  never measured layer l's picks predicting layer l+1's picks in the SAME
  token, which is a different axis (cross-layer correlation of routing,
  not cross-token repetition), and the two need not be related: a router is
  free to route completely differently layer to layer while still being
  sticky token to token within a layer. If (b) is kept in the plan it needs
  its own locality trace and its own frozen prediction before any prefetch
  code is written against it, same as B4 stage 2 gated the whole tier idea
  on a locality measurement before building. Building (b) without that
  measurement is the second "elegant, unbuildable" trap in this wishlist.

- **W2(c) and W5 conflict**, per §0. Either W5 is relaxed to allow a small,
  reviewed kernel-visible residency check (a host-filled bitmap the kernel
  reads, still no gather logic changes), which is a kernel edit and routes
  to the coordinator per W5's own text, or W2(c) is dropped from this lane's
  scope and named explicitly as fable's item, not sonnet's.

- **W3's factual question has an answer, and it changes what's buildable:**
  `max.gpu.host.DeviceContext.create_stream(self, *, priority: Int = 0) ->
  DeviceStream` exists, and `DeviceEvent` / `DeviceStream.wait_for(event)` /
  `enqueue_wait_for` give cross-stream ordering without a full
  `ctx.synchronize()`. So yes, a second stream is real and host-code-only.
  But it only helps the bucket it can help: a second stream can overlap an
  H2D copy with the OTHER stream's kernels, it cannot avoid the host needing
  to read `hidx_d` back before it knows which experts to fetch -- that read
  is a genuine data dependency (the ids don't exist until the router runs),
  and `DeviceStream.wait_for_host_value` still means the host blocks on a
  flag the device writes, which is the same latency shape as
  `ctx.synchronize()` for that one copy, not a way to skip it. So: yes to
  "does a second stream exist" (buildable), no to "does it remove the
  per-layer host wait for the router's own output" (it does not, and W3's
  own text already says the readback can't be batched across layers for
  exactly this reason -- W3 is internally consistent about that, W2's
  opening line "remove the synchronous round trip" is not).

- **W4 is buildable and correctly specified.** "A prefetched expert counts
  as a hit only if its copy completed before use" is checkable against the
  existing `hits`/`refs` counters plus a completion flag per staged copy;
  no new machinery needed, just don't count `touch()` returning True as a
  hit if the fetch that filled that slot hasn't finished. I'd add one field
  W4 doesn't state: the kill line must be written in the SAME units W1
  measures in (a bucket-share number), not just a tok/s floor, or a lane
  that shrinks fetch_ns's minor buckets while missing the dominant one
  passes tok/s trivially-badly and still "did something."

## 3. What the wishlist gets wrong

`serve/expert_tier.mojo`'s own docstring says the miss-driven round trip
"is inherent to demand fetching and is what stage 3's prefetch exists to
remove" -- but reading `prepare()` (lines 301-353), the host round trip is
not gated on a miss. `ctx.synchronize()` at line 319 runs unconditionally,
before the loop even checks which of the 8 ids are hits. The `fh = open(...)`
at line 312 and `fh.close()` at line 352 run unconditionally too, once per
layer, 40 times a token, regardless of hit rate. So a tier at cap 128 with
80%+ hit rate (the B4 stage-2b result table, this brief's own §"situation")
still pays the full sync-plus-open-plus-close cost on every layer -- which
is visible in the record already: cap 128 measured 46.75 tok/s, not
anywhere near the no-tier champion's 111.89, despite hit rate near the
offline ceiling (0.8274, "nothing evicted"). If misses were the cost,
raising the hit rate from 77% to 83% and having almost no evictions left
should have closed most of the gap to 111.89. It didn't move throughput
past 47. **That is the receipt that the round trip's cost is structural
(one sync + one file-open/close pair per layer, always), not proportional
to misses**, and it is sitting in the B4 stage-2b report the coordinator
already has. W2's framing ("remove the SYNCHRONOUS round trip" by
predicting bytes) treats the round trip as a miss-fetch problem; the code
and the cap-128 number both say it's a per-layer-unconditionally problem.
`window.mojo:786`'s `b.tier.prepare(ctx, layer, b.hidx_d)` call is also
unconditional, once per layer -- nothing there depends on whether this
layer's picks were predictable, so no predictor plugged in at that call
site changes call FREQUENCY, only what's already resident when the call
runs. A predictor cannot reduce a per-call fixed cost that fires 40
times a token no matter what.

## 4. Prediction band for tok/s after W2, and the kill line

Arithmetic, arm by arm:

- **If W2 ships as literally scoped (predictors a/b/c hiding bytes, no
  change to `prepare()`'s per-layer sync/open/close shape):** removes at
  most the 5.81 ms transfer component of a 1750 ms token (§0). Predicted
  band: **39.0 to 39.5 tok/s.** Effectively unchanged from the cap-64
  baseline (39.06). This is the band I would freeze if the coordinator
  insists on building W2 exactly as written, and I would write it down
  BEFORE the run specifically so a null result reads as "prediction held,
  the diagnosis in §0/§3 was right" rather than as a failed lane.

- **If W1 runs first and the dominant bucket turns out to be
  sync-plus-open/close overhead (my expectation, from §3's cap-128
  evidence), and the fix becomes "resolve slots for all layers whose
  picks were correctly predicted in ONE combined host step per token,
  falling back to per-layer sync only on a mispredicted layer":** the
  ceiling is bounded by the 39.3% same-layer repeat rate from the
  locality trace -- roughly 4 in 10 layers could, in principle, skip
  their own sync if predicted-from-previous-token picks are verified
  correct without a round trip (which itself needs a device-side check,
  running back into the W2(c)/W5 conflict in §0). Being conservative
  about that circularity, I would not freeze a number above **50 to 58
  tok/s** for this arm without W1's actual bucket table in hand; anything
  higher is a guess dressed as a prediction.

**Kill line:** if, after W1, the sync-plus-open/close bucket is under 50%
of `fetch_ns`, my §3 diagnosis is wrong and W2 as literally scoped (bytes
prediction) is reinstated as the right-shaped fix; I would then freeze the
transfer-bound band (62-71, the ORIGINAL stage-2b prediction) instead of
mine. Either way, **no W2 code is written before W1's bucket table exists
and is read**, and the frozen band for whichever arm is actually built is
picked from the two above, not from the coordinator's original prose.

## 5. Rules I insist on verbatim

- W1 runs and its bucket table is READ before a line of W2 or W3 is
  written. Not "instrument before any change" as a sentiment (the wishlist
  already says this) -- literally: no commit touching `expert_tier.mojo`'s
  prefetch path lands without a receipt path under `.work/moe-tier-stamp/`
  in the same report.
- W2(c) is out of this lane's scope unless W5 is explicitly relaxed for it
  in writing by the coordinator; until then this lane's W2 is (a) and/or
  (b) only, and (b) does not start without its own locality trace and its
  own frozen prediction, gated the same way B4 stage 2 gated the tier
  itself.
- The kill line is stated in the SAME unit W1 measures (bucket share or
  ms), in addition to a tok/s floor, so a lane cannot pass by shrinking a
  minor bucket while the dominant one is untouched.
- Identity 20/20 at cap 64 AND 128 re-runs after every item that touches
  `prepare()`, not once at the end -- W2 changes which slots are resident
  when a kernel launches, which is exactly the class of change gate 2 in
  the B4 stage-2b protocol was written to catch.
- Every timed run: `gpu-wait run --timeout`, `bench/preflight.sh` first,
  engine sha256 and tier's own startup print (cap, cache bytes, pinned or
  page-cache) read back before the number is trusted, per P1. No em dashes.
  Report to `exchange/lane-MOE3-report.md`.

## Round 2

**Agree, with a correction against myself.** My round-1 "transfer is 0.33%
of fetch_s" used the wrong denominator (`fetch_s` is a whole run's wall
time, not per-token) -- coordinator caught it, 18-skeptic made the same
error. The right gap is 16.7 ms/token (25.6 ms tier vs 8.94 ms full-pack).
On that denominator my structural claim (sync + open/close, fixed cost,
not proportional to misses) is also wrong: cap 128 cuts misses ~23%
relative and gains ~20% tok/s, almost 1:1, which is a per-miss signature,
not a fixed one. Coordinator's CPU-only pread measurement (16.0 ms for
217 preads, against a 16.7 ms gap) lands on the same conclusion from a
different direction. I retract the round-1 diagnosis in favor of: the
blocking `pread()` inside `_fetch_piece`, not `ctx.synchronize()` or
open/close, is the leading suspect.

**Still reject: freezing any tok/s band before W0 runs.** Three different
bands got written in round 1 (mine, 12-engineer's, 18-skeptic's) off the
same wrong denominator or off arithmetic alone. W0 (`BARO_TIER_PINNED=1`
vs `0`, zero code) settles pread-vs-sync directly and costs one GPU run.
No band should be frozen ahead of it, including the ones already in this
file.

**Still reject: W0 "confirmed" quietly becoming "solved."** If pinned
lands in 62-71, that proves pread is the term, it does not ship anything:
`expert_tier.mojo`'s own comment rejected full pinning already (18 GB
held hostage on a box that "runs other work"). The real fix is then a
partial pin of the roughly 100/256 hot experts per layer the locality
trace already names, or moving `pread` off the blocking thread -- a new
W2 shape, not the current one. The report must say this explicitly rather
than let a green W0 read as done.

**Format change I insist on:** promote W0 to its own itemized section,
same four fields as every other item, run BEFORE W1's instrumentation
(it needs none):

```
### W0: pinned vs page-cache, cap 64, zero code
Files: none (BARO_TIER_PINNED=1 vs unset, existing engine).
Check: two 20-prompt median runs, same engine sha256, same prompts.
Receipt: tok/s, hit rate, bytes/token, pinned flag from the tier's own
  startup print, for both arms, in exchange/.
Done: pinned tok/s >= 55 -> pread confirmed dominant, W1's phase split
  targets pread specifically. Pinned tok/s < 45 -> pread is not it,
  W1 runs as originally scoped with no prior.
```

DONE
