# Lane MOE3: pinned store A/B, the tier's timeline, and a request-scoped hot store

Worktree `mojo-baro-lanes/MOE3`, branch `lane-MOE3`, from `docs/MOE-STAGE3-PLAN.md` item
`MOE3` (items 0, 1, 2 in order). Host code only (`serve/expert_tier.mojo`, `bench/`); no file
under `kernels/` touched. Protocol: `bench/moe-tier-protocol.md`, "Stage 3" section, frozen by
commit before each item's timed run.

## Gate

`./run-tests.sh` + `tools/ci-checks.sh`, captured to `.work/MOE3-gate.txt`:

- `run-tests.sh` exit 0, **47 PASS lines, 0 FAIL**, census `103 kernels, 58 in registry, 0
  orphans`.
- `tools/ci-checks.sh`: **all non-GPU checks passed**.

Test count before this lane (same gate, same tree pre-item-0): 47 PASS, 0 FAIL, same census,
unchanged. Item 0 is zero-code by design; items 1 and 2 add instrumentation and an opt-in
(default-off) fetch path, neither of which changes any existing test's behavior. Floor met:
no regression.

`gpu-wait stats --days 1`: this lane's jobs all recorded OK, 0 failed (item2's `clock-probe.sh
item2-full.sh` row: 371s wall, 21.8 GB VRAM, 0s queue wait). Every job ran with `--timeout`.

## Item 0: pinned store A/B, zero code

No file changed. The tier already prints `mode pinned|page-cache` at load and per-request
`hits`/`refs`/`bytes_fetched`/`bytes_per_token`: the item's own "add the echo if missing"
clause did not apply.

Arm A `BARO_TIER_PINNED=0`, arm B `=1`, cap 64, 20 prompts, one stint, `bench/clock-probe.sh`
around the pair (`.work/moe3/item0/`, `.work/moe3/logs/item0-run.log`).

| | median tok/s_gen | spread |
|---|---|---|
| arm A (page-cache) | 48.57 | 128.5% |
| arm B (pinned) | 67.12 | 59.1% |

Ratio 1.382. Identity 20/20 both arms against each other and against the full-pack reference
(`.work/b4/g23main/*.ref.log`); hit rate and bytes/token equal to full precision between arms
(same LRU, same requests). Clock probe: sclk 1236-3318 MHz (median 3281), power 67-291 W, no
throttling.

**Result: arm B 67.12 >= 55, the blocking pread confirmed as the dominant term**, per the
frozen two-way falsifier.

**Deviation, worth flagging:** the plan's own baseline for arm A is 39.06 tok/s (stage 2b,
commit `f798ed4`, before the R6.0/R6.0b launch-fusion round landed on the same trunk kernels the
tier calls). Re-measured in this stint per protocol rule ("the 39.06 baseline is re-measured in
the same stint as any arm it is compared with"), arm A is **48.57**, not 39.06: the trunk got
faster since stage 2b, so the stale number is superseded, not contradicted.

## Item 1: stamp timeline of one tier token

`serve/expert_tier.mojo`: `BARO_TIER_STAMP=1` accumulates six `perf_counter_ns` buckets per
layer inside `prepare()`/`_fetch_piece()` (open/close, readback+sync, LRU touch, pread, copy
enqueue, id writeback), printed by `report()` as a 40-row table plus a run sum.
`bench/moe-tier-stamp.py` (new) parses and sums a log, checking the buckets against `fetch_ns`.

One `rocprofv3` kernel trace of the same request (`.work/moe3/item1/trace/`, pattern from
`bench/moe-launch-count.sh`): 61,225 dispatches, 604.5 ms total kernel busy time over a 1840 ms
kernel-dispatch span (arm A, p01-water, prefill+decode).

Stamp receipts (`.work/moe3/logs/item1-run.log`, `.work/moe3/item1/p01.*.stamp.log`), single
prompt p01-water, both arms of item 0:

| bucket | arm A (page-cache) | arm B (pinned) |
|---|---|---|
| open | 11.8 ms | 8.2 ms |
| readback (enqueue+sync) | 710.6 ms | 1107.0 ms |
| lru | 3.3 ms | 1.9 ms |
| **pread** | **910.9 ms** | 0.0 ms |
| copy | 26.5 ms | 17.6 ms |
| writeback | 4.4 ms | 3.5 ms |
| sum vs `fetch_ns` | 1667.4 / 1669.9 ms (0.15% dev) | 1138.1 / 1139.7 ms (0.14% dev) |

Check passed: buckets sum to `fetch_ns` within 5% (0.14-0.17% observed) on both arms.

**Done: the dominant term is `pread`, 910.9 ms of 1667.4 ms (54.6%) in the unpinned arm.** The
per-layer sync bubble (the `readback` bucket: `enqueue_copy` of the router's ids plus
`ctx.synchronize()`) is not the small 50-150 us guess in the plan's arithmetic. Measured average
per layer per call (readback total divided by 40 layers, single run) is 17.8 ms, three orders
of magnitude above that guess, because `synchronize()` waits on whatever GPU work is still in
flight from prior layers, not on a small fixed bubble. It is co-dominant with pread (42.6% of
the unpinned arm's total, 97.2% of the pinned arm's, since pinning removes pread entirely and
readback absorbs the rest).

## Item 2(a): request-scoped hot store for the tier's misses

Built only against item 1's dominant term (pread), per the plan's ordering.

**Shape:** `serve/expert_tier.mojo`, `BARO_TIER_HOT=1` (default off), `BARO_TIER_HOTCAP=128`.
A per-layer, per-request, non-evicting host cache: the first time an expert is fetched this
request, its pread lands in a persistent per-request slot (instead of the rotating stage ring)
and the slot is remembered; a later re-reference to that same expert (evicted from the 64-slot
VRAM LRU, referenced again) copies directly from the persistent slot, no second pread. Never
calls `LayerLru.touch`; a miss in the hot lookup (first touch, or hot store full) falls straight
through to the existing pread path, so correctness never depends on the hot store being right.

**Frozen prediction**, from a deterministic replay of a fresh router trace over the same 20
prompts (`bench/moe-locality.py`'s method, `.work/moe3/item2/expert-trace.txt`, generated on the
full pack `.work/moe-w1/pack` and cross-checked 20/20 against the full-pack reference before
use): of the 23.27% of references that miss the 64-slot LRU, only **16.4%** (3.81% of all
references) are repeats of an expert already seen this request. 81.9% of misses are genuine
first touches no request-scoped cache can avoid. Predicted pread-bucket shrink about 16%, frozen
below the item's 50% bar before the timed run.

**Measured** (`.work/moe3/item2/ab64/`, `.work/moe3/logs/item2-full.log`, 20 prompts, cap 64,
`BARO_TIER_STAMP=1`):

| bucket (sum over 20 prompts) | hot0 | hot1 | change |
|---|---|---|---|
| pread | 19483.2 ms | 17990.9 ms | -7.66% |
| readback | 15481.7 ms | 17510.2 ms | +13.1% |
| copy | 564.8 ms | 532.1 ms | -5.8% |
| total (fetch_s, sum) | 36.006 s | 36.526 s | +1.4% |

tok/s_gen: hot0 median 48.04 (spread 66.2%), hot1 median 50.54 (spread 48.4%), ratio 1.052.

**The pread shrink measured even smaller than the already-below-bar prediction (7.66%, not
16.4%), and the readback bucket got worse by more than pread improved: net host round-trip time
went up 1.4%.** The 1.052x tok/s ratio is not distinguishable from this system's per-prompt
noise floor (item 0 measured 59-129% spread on the same 20 prompts).

**Correctness, fully verified:**
- Identity 20/20, cap 64, both `BARO_TIER_HOT=0` and `=1`, vs the full-pack reference.
- Identity 20/20, cap 128, `BARO_TIER_HOT=1`, vs the full-pack reference.
- Three repeated 20-prompt runs at cap 64, `BARO_TIER_HOT=1`: bit-identical to each other
  (`diff` empty, r1 vs r2 and r1 vs r3). No in-flight overwrite race.

**Result: item 2(a) does not meet the Done bar** (50% shrink of the dominant bucket). It is
correct, opt-in, harmless when off, and left in the tree as a documented negative result rather
than reverted, matching this repo's precedent for a below-bar but correct opt-in (e.g. R6.2's
`BARO_MEGA` gate). The kill line was about identity, which held throughout: this is a
below-the-bar result, not a correctness kill.

**Why, and what's left:** the request-level reuse pool is small (16.4% of misses) because MoE
routing spreads references thinly across roughly 100 of 256 experts per layer per request. A
request-scoped cache of reasonable size cannot manufacture locality that the router's own
choices do not have. The `readback`/`ctx.synchronize()` bucket is co-dominant with pread (42.6%
of the unpinned arm, per item 1) and untouched by any host-only shape: it is the GPU-compute
wait folded into the host round trip. **Recommend to the coordinator, per item 2(c)'s framing:**
the device-side residency check named in the plan's own preamble (no host sync on a hit layer)
is the lever neither host-only shape reaches, and item 2(a)'s staging-thread variant (real
multi-threaded host I/O, overlapping pread with GPU compute) was not attempted. It is a
materially larger and riskier change (the plan's own stated risk: an in-flight overwrite race
that fires one run in five) than fits a host-only, three-repair-attempt lane. Both items 1 and
2's numbers are attached above for sizing that follow-on work.

## Commits, `lane-MOE3`

- `864f8b4` bench(moe-tier): freeze item 0's pinned-store A/B before the run
- `b350cf6` serve(moe): item 1, stamp timeline of one tier token
- `79284a3` serve(moe): item 2(a), request-scoped hot store for the tier's misses

Plus the coordinator's `26e0977` (docs/MOE-STAGE3-PLAN.md ci-checks fix), merged `--ff-only`
before item 0's run.

## Receipts

`.work/moe3/item0/` (results.txt, arm.txt, per-prompt logs), `.work/moe3/item1/` (stamps,
rocprofv3 trace), `.work/moe3/item2/` (expert-trace.txt, quick/, ab64/, cap128/, repeat/),
`.work/moe3/logs/` (full stdout of every GPU job and gate run cited above, moved out of `/tmp`
per CLAUDE.md section 11), `.work/MOE3-gate.txt` (final `run-tests.sh` + `tools/ci-checks.sh`).
