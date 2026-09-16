# MoE stage 3: prefetch for the host-resident expert tier (conference, round 1)

You are one of three participants. Do NOT build, do NOT edit the repo. Read, think, write your file
(path in the header above), end it with the line `DONE`. Output is a file, never terminal content.
Model: you are sonnet. Search: `brain-ask`/`brain-recall` for prior notes, firecrawl only for the web.

## The situation (measured, 2026-09-15/16, all in the repo)

- The 35B MoE (qwen35moe, 40 layers, 256 experts, top-8) runs with its routed experts OUT of VRAM:
  `serve/expert_tier.mojo` (B4 stage 2b, `bench/moe-tier-protocol.md`, report
  `exchange/lane-B4-stage2b-report.md`). Pack 2.68 GB in VRAM, 18.3 GB of experts in host RAM (page
  cache), a per-layer LRU of `cap` experts (cap 64 = 4.5 GB, cap 128 = 9.1 GB). No kernel change: the
  expert kernels index `e * stride` from a base pointer, so the cache is a base pointer plus an index
  of slots.
- Identity 20/20 vs the full-pack engine at cap 64 and 128. Live hit rate 0.7743 (offline replay
  predicted 0.767). 0.1656 GB per token moved, 5.81 ms of PCIe at 28.5 GB/s.
- **Throughput 39.06 tok/s against 62 to 71 predicted from transfer cost alone.** The cost is not the
  bytes: `fetch_s` is 1.74 s of a 1.75 s decode. It is 40 host round trips per token: each layer's
  router picks (`hidx_d`) are read back to the host, missing experts fetched synchronously, then the
  layer runs (`serve/window.mojo:786`, `b.tier.prepare(ctx, layer, b.hidx_d)`).
- Locality trace (`bench/moe-locality-protocol.md`): 39.3% of a layer's picks repeat from the previous
  token; 19.5% of references are cold first touches even for a perfect within-request cache; only
  about 100 of 256 experts per layer are touched in a 64-token request.
- Full-pack MoE champion (experts in VRAM, launch path): 111.89 tok/s. So P14 says: the bar for a
  tier with prefetch must be a number some configuration has hit; the transfer-only prediction (62 to
  71) is the ceiling class, 111.89 is the no-tier reference.
- `docs/NEXT-PLAN.md` A0 says: a stamp timeline of one MoE token FIRST (`bench/carryover-stamp.py`),
  then prefetch. B4 says: prefetch from the current layer's router for the next layer (Pre-gated MoE
  shape), GPU proof = one 20-prompt run per stage.

## The wishlist (what the coordinator would build; tell us what is wrong with it)

W1. Stamp timeline of one tier token: where the 1.74 s goes (readback sync, page-cache read, H2D
    copy, LRU bookkeeping, kernel wait), per layer. Instrument before any change.
W2. Remove the synchronous round trip: predict layer l+1's experts and issue their H2D copies while
    layer l computes. Candidate predictors: (a) previous token's picks for the same layer (39.3%
    repeat), (b) the current layer's router logits for the next layer (needs a cheap host or device
    projection), (c) a device-side residency check so a HIT layer needs no host sync at all.
W3. Batch the readback: one sync per token for all 40 layers' picks is impossible (picks depend on
    the layer's own hidden state), but the readback + fetch can overlap with the trunk kernels of the
    same layer on a second stream. Say whether Mojo's `DeviceContext` gives us a second stream or a
    host thread that can enqueue copies concurrently; if not, what is the fallback.
W4. Gates: identity 20/20 vs full-pack engine at cap 64 and 128 (unchanged rule), hit rate counted
    HONESTLY (a prefetched expert counts as a hit only if its copy completed before use; a
    late prefetch is a miss with wasted bytes), bytes per token, and 20-prompt median tok/s with a
    frozen prediction band and a kill line.
W5. Host code only. Any kernel edit (`kernels/*.mojo`) is a request back to the coordinator (fable),
    not something this lane does.

## Questions to answer in your file

1. How do you want the plan items shaped so you can build them: one section per item with what
   fields (files, the check, the receipt, the number that means done)? Show the shape on W1.
2. What would make you fail: which of W1 to W5 is unbuildable or under-specified as written, and
   what do you need instead.
3. What does the wishlist get wrong? Argue from the code (`serve/expert_tier.mojo`,
   `serve/window.mojo` around line 786, `bench/moe-tier-protocol.md`) and the record, not from
   general MoE lore.
4. Your prediction band for tok/s after W2, with the arithmetic, and the kill line you would freeze.
5. Rules you insist on verbatim in the brief.

Standing rules the lane will inherit: every GPU job via `gpu-wait run --timeout`, `bench/preflight.sh`
first, arm parameters read back from the running engine (P1), 20-prompt medians (P4), harness before
kernel (P6), commits by pathspec, no em dashes anywhere, report to `exchange/lane-MOE3-report.md`.
