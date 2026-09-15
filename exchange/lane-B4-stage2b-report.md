# B4 stage 2b: the host-resident expert tier

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-b4-stage2b-expert-tier.md`.
Protocol, frozen before a line of the tier existed: `bench/moe-tier-protocol.md`
(`2b295f6`). Built on `bench/moe-locality-protocol.md`'s numbers (`1aa06d5`).

## What landed

| piece | where | commit |
|---|---|---|
| the tier: host store, per-layer VRAM cache, LRU, fetch, counters | `serve/expert_tier.mojo` | `8f8c8c3` |
| gate 1, the LRU against the offline replay | `kernels/test_expert_tier.mojo` | `8f8c8c3` |
| tier-pack loader and its two refusals | `serve/harness.mojo` | `8f8c8c3` |
| the counter print per request | `serve/engine.mojo` | `8f8c8c3` |
| pack splitter (w82:p3) | `tools/pack-split-experts.py` | see its own report |
| the wiring, for fable to apply | `.work/b4/window-tier.patch` | not committed |

`serve/window.mojo` is fable's this leg (R6.0b), so the 49-line wiring is a
patch rather than a commit. It was developed and compiled against a full
scratch copy of the tree (`.work/b4/src`, engine `.work/b4/engine-tier-wired`),
so it is known to build before anyone applies it.

## The design fact that made this cheap

**No kernel change, by construction.** `moe_gate_up_q4k_pack` reads expert `e`
at `e * FFN * row_bytes` from a base pointer it is handed, and its gate-to-up
distance is a runtime argument; `amar_moe_down_q4k` reads `e * N * row_bytes`
from its own base. Neither kernel knows how many experts exist. So a VRAM
cache of `cap` experts in the same inner layout, a base pointer into it, an
`up_offset` of `cap * per_expert_bytes`, and an index array holding **cache
slots instead of expert ids** run both kernels untouched. `prepare()` reads
the router's top-8 back, fetches misses, and rewrites the index in place.

## Gates

### Gate 1: the tier's LRU is the replay's LRU (CPU)

`./.work/b4/test_expert_tier`, receipt `.work/b4/gate1.log`:

```
trace: 20 prompts, 51200 rows
   PASS cap 64 : hit rate 0.76733154296875  replay said 0.767
   PASS cap 256 : hit rate 0.80541015625  replay said 0.805
PASS: expert tier LRU reproduces bench/moe-locality.py
```

This is the gate that makes the live number mean anything: if the tier's
residency policy were not the policy the replay measured, gate 3 would be
comparing two different things.

### Gates 2 and 3: identity and live counters, 20 prompts (GPU)

Engine `.work/b4/engine-main`, sha256 `af50a7357db6a3f2`, built from a clean
`git archive` of **`f798ed4`**, which is the wiring on main rather than the
scratch copy this was developed in. Reference `.work/b1/engine-r4base` (full
pack). Pack `.work/moe-tier`. Receipts in `.work/b4/g23main/`.

The first read-back is the one that could have voided the rest:
`loading pack: 2678180352 bytes  tier tensors: 120`. The engine loaded the
trunk only, **2.68 GB against 21 GB**, so 18.3 GB of expert weights are no
longer in VRAM.

| capacity | cache VRAM | identity | hit rate | bytes/token | ms/token at 28.5 GB/s | median tok/s_gen |
|---|---|---|---|---|---|---|
| **64** | 5.22 GB | **20/20** | **0.7743** | 0.1656 GB | 5.81 | 39.06 |
| 128 | 10.4 GB | **20/20** | 0.8274 | 0.1266 GB | 4.44 | 46.75 |
| 256 | 20.9 GB | does not fit: `hipErrorOutOfMemory`, `oom log request=19.45GB` | | | | |

Against the frozen predictions:

- **Prediction 1 held.** 0.7743 live against the replay's 0.767, inside the
  0.747 to 0.787 band. The live number is slightly HIGHER because the tier
  also serves the prefill replay steps, which warm the cache before the 64
  decode steps the replay counted.
- **Prediction 2 held.** 0.1656 GB per token against "about 0.18", and 5.81 ms
  of transfer against the frozen 4.7 to 6.4 ms.
- **Prediction 3 (stated, not gated) missed, and the protocol had already
  named why.** 39.06 tok/s against the 62 to 71 predicted from transfer cost
  alone. `fetch_s` is 1.74 s of a 1.75 s decode on p01: the engine now waits
  on the tier's own path, which is 40 host round trips per token to read the
  router's top-8 back before any fetch can be issued. That is inherent to
  demand fetching and is exactly what stage 3's prefetch removes.
- **Kill line not reached.** It was below 70% live; measured 77.4%.
- **Capacity 128 is the consistency check on the whole thing.** Its hit rate,
  0.8274, sits just above the offline ceiling for an unlimited cache (0.805),
  which is what "nothing was evicted" should look like: the only misses left
  are cold first touches. Identity is 20/20 there too.

**The identity result is the one worth keeping.** `GENERATED` is bit-identical
to the full-pack engine on 20 of 20 prompts with a quarter of expert
references missing and being refetched mid-token. That is what proves the slot
remapping is right rather than masked by residency.

### Gate 4: repository gates

On `f798ed4`, both green:

- `tools/ci-checks.sh`: **all non-GPU checks passed** (`.work/b4/ci2.log`).
- `./run-tests.sh`: **exit 0, 104 PASS lines**, census `97 kernels, 52 in
  registry, 0 orphans` (`.work/b4/tests.log`).

## Deviations from the brief, each with its reason

1. **Gate 2's second capacity is 128, not 256.** At 256 the cache is 20.9 GB
   and the allocation fails: `hipErrorOutOfMemory`, `oom log request=19.45GB`,
   receipt `.work/b4/smoke256.log`. 128 is the no-eviction arm instead, and it
   is a real one: the trace behind `1aa06d5` shows only about 100 distinct
   experts per layer per 64-token request, so at 128 nothing is evicted and the
   hit rate should equal the unlimited-cache ceiling.
2. **The host store is read through the page cache, not pinned.** The brief
   says pinned host RAM. Pinning 18.3 GB on a 64 GB box that is running three
   agents is a real cost, and stage 1 measured pinned against pageable
   host-to-device at 28.78 and 28.60 GB/s, a 0.6% difference. `BARO_TIER_PINNED=1`
   pins the whole store instead, and the tier prints which mode it is in
   because that is an arm-defining parameter.
3. **The pack splitter writes `expert` in the index's fifth column**, where the
   protocol said `tier`. The loader accepts either. Not worth a round trip.


## What this stage settled, and what it did not

**Settled.** A 35B MoE runs with its routed experts out of VRAM, from a 2.68
GB trunk plus a 5.22 GB expert cache, and produces byte-identical output to
the full-pack engine on 20 of 20 prompts while a quarter of its expert
references miss and are refetched mid-token. The locality the offline replay
predicted is the locality the live tier sees: 0.7743 against 0.767.

**Not settled, and not claimed.** Speed. 39.06 tok/s against 107.27 without
the tier. The cost is not the PCIe transfer the protocol predicted (5.81 ms
per token of it) but the 40 host round trips per token that demand fetching
requires: the top-8 does not exist until the router has run, so the ids come
back to the host before any fetch can be issued. `fetch_s` is 1.74 s of a
1.75 s decode.

That is the number stage 3 exists to attack, and the trace already says how:
the 19.5% of references that a perfect within-request cache still misses are
cold first touches, and 39.3% of picks repeat from the previous token
(`bench/moe-locality-protocol.md`). A prefetch driven by the previous layer's
router, issued while the current layer computes, removes the round trip rather
than the bytes.

**Worth stating plainly for B4's headline claim.** This proves the mechanism on
a model whose experts fit in VRAM anyway. It does not prove a 100B-class model
is interactive on this card: that needs the locality of a model nobody has
traced, and stage 1's arithmetic says 1.64 GB of routed bytes per token before
any cache. What is now measured rather than assumed is that the cache works,
what it costs, and where the time actually goes.

## Two process notes

- I committed `8f8c8c3` with the tier module but without the `alloc_bufs` half
  that constructs it, which only existed in my scratch copy, so the tree did
  not build. fable caught it. That is the same defect I logged this morning
  (run the gates on the tree you are about to commit, not the one you started
  from) and it is now in `m.ledger/mojo-baro.md` twice.
- The first gate numbers came from the scratch engine rather than a build of
  main. fable asked for the re-run and was right to: a receipt has to name a
  binary someone else can rebuild (P7). The numbers came out identical to four
  decimals, which is reassuring but is not what makes them valid.
