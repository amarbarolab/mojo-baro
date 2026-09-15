# B4 stage 2b: the host-resident expert tier

Preregistered 2026-09-15, before a line of the tier was written, at commit
`c52ed6b`. Binds `bench/PROTOCOL-RULES.md`. Decision to build: the maintainer, on the
numbers in `bench/moe-locality-protocol.md` (`1aa06d5`).

## What is being built

Routed expert weights leave VRAM and live in host RAM. The trunk (attention,
SSM, norms, router, shared expert, embeddings, head) stays in VRAM. A VRAM
cache of `cap` experts per layer, laid out exactly as the pack lays out its
256, serves the top-8 gather; an LRU per layer decides what stays; a miss is
fetched over PCIe before the gate/up launch.

**No kernel changes, by construction.** `moe_gate_up_q4k_pack` indexes expert
`e` at `e * FFN * row_bytes` from a base pointer it is handed, with the gate
to up distance passed as a runtime argument, and `amar_moe_down_q4k` indexes
`e * N * row_bytes` from its own base. So a cache of `cap` experts with the
same inner layout, a base pointer into it, a gate-to-up distance of
`cap * 589,824`, and an index array holding **slots instead of expert ids**
runs the existing kernels unchanged. `serve/window.mojo` is fable's this leg
(R6.0b), so the wiring is delivered as a patch under `.work/b4/` and applied
by fable.

Sizes, from the pack index rather than from arithmetic on paper: per expert
per layer, gate 589,824 B, up 589,824 B, down 589,824 B (q4_k) or 860,160 B
(q6_k, layers 34, 38 and 39). At `cap` 64 that is 113 MB of cache per layer
and **4.53 GB over 40 layers**; the expert store in host RAM is 18.1 GB.

## P1 read-back, before any timed run

- engine sha256, built in the same command as the run.
- the tier's own startup print: capacity, cache bytes, host store bytes, and
  whether the host store is pinned or read through the page cache.
- `pack loaded in ... bytes`: with the tier on, this number must be the
  **trunk only** (about 2.9 GB, not 21 GB). If it is not, the tier pack was
  not what the engine loaded and every later number is void.
- per-run counters printed by the engine: expert references, hits, misses,
  bytes fetched over PCIe.
- power cap, vddgfx, sclk from `bench/clock-probe.sh` for any timed arm.

## Frozen predictions

1. **Live hit rate at capacity 64 reproduces the offline replay within 2
   points.** The replay says 76.7% on these 20 prompts with an LRU reset per
   prompt (`bench/moe-locality-protocol.md`). Same policy, same prompts, so
   the live number is 74.7% to 78.7%. Outside that band, one of the two is
   wrong and the report says which, with the trace to prove it: the replay
   reads `.work/b1/experts.txt`, and the live tier can emit the same rows.
2. **Bytes over PCIe per token at capacity 64: about 0.18 GB.** 0.78 GB of
   routed plus shared plus router bytes per token, of which the routed 0.573
   GB is what the cache serves, times the 23.3% miss rate, is 0.134 GB;
   adding the q6_k layers' larger experts puts it near 0.14 to 0.18 GB. At
   stage 1's measured 28.6 GB/s that is **4.7 to 6.4 ms per token** of
   transfer.
3. **Throughput, stated but not gated.** The current MoE decode is 107.27
   tok/s (R6.0, `1e270b5`), i.e. 9.3 ms per token. Adding 4.7 to 6.4 ms of
   demand-fetch transfer that nothing overlaps yet predicts **62 to 71 tok/s**
   with the tier at capacity 64, and about 107 at capacity 256 (no misses
   after warm-up). This stage does not gate on that: demand fetching without
   prefetch is stage 3's problem, and a per-layer host round trip is inherent
   to demand fetching, since the top-8 is only known after the router runs.

## Kill line

**Live hit rate below 70% at capacity 64.** Not tok/s: this stage proves the
mechanism and the locality, and a slower engine with a correct tier is a
result, not a failure. Below 70% the replay was optimistic about live
behaviour and B4's build order is rewritten around prefetch instead.

## Gates

1. **`test_expert_tier` (CPU).** The tier module's own LRU, replayed over the
   `1aa06d5` trace rows, reproduces 76.7% at capacity 64 and the same hit
   rate at 256 that the Python replay reports. This is what makes the live
   number comparable: the same code decides residency in both.
2. **Identity, 20 prompts, twice (GPU).** `GENERATED` bit-identical against
   `.work/moe-perf/engine-r60` with the tier at capacity 256 and at capacity
   64. Two capacities because a gather that depended on residency would pass
   at 256 and fail at 64, and because at 256 nothing is ever evicted.
3. **Live counters, 20 prompts (GPU).** Hit rate and bytes fetched per token
   from the engine's own counters, against predictions 1 and 2.
4. **`./run-tests.sh`, `tools/ci-checks.sh`, the kernel census**, on the tree
   being committed.

## Falsifiers

- Gate 2 failing at 256 but passing at 64, or vice versa: the slot remapping
  is wrong in a way that residency hides; report which capacity and stop.
- Gate 3's hit rate matching the replay while gate 2 fails: the LRU is right
  and the fetch is wrong, which is the more likely defect of the two.
- The pack load printing 21 GB with the tier on: the engine loaded the full
  pack and the tier is decorative. Every number in the run is void.

## Result

Filled in when the gates run. Nothing here is a claim yet.
