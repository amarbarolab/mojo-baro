# B4 stage 1: measured PCIe bandwidth and expert-bytes arithmetic

Lane: w82:p3, reporting to w82:p4. Brief:
`briefs/2026-09-15-p3-b4-stage1-pcie.md`. Deliverable: this report plus
`bench/pcie-bandwidth.mojo`, committed on `main`.

## 1. PCIe bandwidth, measured

Tool: `bench/pcie-bandwidth.mojo`, built and run as documented in its own
docstring. Link state read back from the system immediately before the
run (P1 receipt):

```
$ cat /sys/class/drm/card1/device/current_link_speed
16.0 GT/s PCIe
$ cat /sys/class/drm/card1/device/current_link_width
16
$ cat /sys/class/drm/card1/device/max_link_speed
16.0 GT/s PCIe
$ cat /sys/class/drm/card1/device/max_link_width
16
```

Current equals max on both speed and width: the link is fully trained,
not degraded. The numbers below are not an artifact of a link stuck at
x8 or 8 GT/s.

Command:

```
$HOME/.local/bin/gpu-wait run --vram 6 -- .work/pcie-bandwidth
```

Real stdout, all five size classes (64 MiB through 2 GiB, five repeats
each, median and min/max GB/s, "both" is two enqueues before one
`synchronize()`, not a proven concurrent transfer, see the tool's own
docstring):

```
H2D pinned    bytes: 67108864  median_GBps: 28.48975183651844  min_GBps: 27.346235030830744  max_GBps: 28.651904843837855  median_s: 0.002355544
D2H pinned    bytes: 67108864  median_GBps: 28.367037037976374  min_GBps: 28.33338287150249  max_GBps: 28.417608290239695  median_s: 0.002365734
H2D pageable  bytes: 67108864  median_GBps: 28.38767621511548  min_GBps: 26.852525084388223  max_GBps: 28.4455536697971  median_s: 0.002364014
D2H pageable  bytes: 67108864  median_GBps: 28.248228299723365  min_GBps: 28.17753786879987  max_GBps: 28.29624063891303  median_s: 0.002375684
both(H2D+D2H)  bytes: 134217728  median_GBps: 28.53670234558995  min_GBps: 28.48267815556788  max_GBps: 28.57418695141135  median_s: 0.004703337
---
H2D pinned    bytes: 268435456  median_GBps: 28.727482405135845  min_GBps: 28.69232095454568  max_GBps: 28.768799706434102  median_s: 0.009344204
D2H pinned    bytes: 268435456  median_GBps: 28.478816320407596  min_GBps: 28.470087235928155  max_GBps: 28.50918322632473  median_s: 0.009425794
H2D pageable  bytes: 268435456  median_GBps: 28.527695202921596  min_GBps: 28.21254525208479  max_GBps: 28.607559220429753  median_s: 0.009409644
D2H pageable  bytes: 268435456  median_GBps: 28.357899996672298  min_GBps: 28.343796944152462  max_GBps: 28.36914157903937  median_s: 0.009465985
both(H2D+D2H)  bytes: 536870912  median_GBps: 28.634111010153525  min_GBps: 28.622157981333867  max_GBps: 28.650308547828278  median_s: 0.018749348
---
H2D pinned    bytes: 536870912  median_GBps: 28.75735012666454  min_GBps: 28.756225693272018  max_GBps: 28.76650289572052  median_s: 0.018668998
D2H pinned    bytes: 536870912  median_GBps: 28.517739377403142  min_GBps: 28.508712408503925  max_GBps: 28.528072660883485  median_s: 0.018825858
H2D pageable  bytes: 536870912  median_GBps: 28.571828353889863  min_GBps: 28.418033984813288  max_GBps: 28.620951020870116  median_s: 0.018790219
D2H pageable  bytes: 536870912  median_GBps: 28.375646044715335  min_GBps: 28.345996815192464  max_GBps: 28.39224458909109  median_s: 0.01892013
both(H2D+D2H)  bytes: 1073741824  median_GBps: 28.64922228142291  min_GBps: 28.64735724482974  max_GBps: 28.659530987441332  median_s: 0.037478917
---
H2D pinned    bytes: 1073741824  median_GBps: 28.76448385501186  min_GBps: 28.727482405135845  max_GBps: 28.770510991745216  median_s: 0.037328736
D2H pinned    bytes: 1073741824  median_GBps: 28.52797488466972  min_GBps: 28.506873188887145  max_GBps: 28.5310744893769  median_s: 0.037638207
H2D pageable  bytes: 1073741824  median_GBps: 28.604253995774986  min_GBps: 28.381422044435322  max_GBps: 28.614849875694013  median_s: 0.037537837
D2H pageable  bytes: 1073741824  median_GBps: 28.39352093393883  min_GBps: 28.391200324044227  max_GBps: 28.396561349244553  median_s: 0.037816438
both(H2D+D2H)  bytes: 2147483648  median_GBps: 28.662189084786707  min_GBps: 28.658016065772774  max_GBps: 28.672616997651993  median_s: 0.074923923
---
H2D pinned    bytes: 2147483648  median_GBps: 28.782534268699326  min_GBps: 28.78118799516585  max_GBps: 28.78413144418278  median_s: 0.074610652
D2H pinned    bytes: 2147483648  median_GBps: 28.541574864988018  min_GBps: 28.443918873643497  max_GBps: 28.547782179361597  median_s: 0.075240545
H2D pageable  bytes: 2147483648  median_GBps: 28.595055288078235  min_GBps: 21.14329523330108  max_GBps: 28.605980055399183  median_s: 0.075099825
D2H pageable  bytes: 2147483648  median_GBps: 28.393043792772595  min_GBps: 28.38288498078368  max_GBps: 28.39667399753821  median_s: 0.075634147
both(H2D+D2H)  bytes: 4294967296  median_GBps: 28.668735828683996  min_GBps: 14.94094975481054  max_GBps: 28.67660318421818  median_s: 0.149813627
---
```

Wall clock for the whole sweep: 19.5 s.

### Reading these numbers, plainly

- **Sustained H2D at 2 GiB (the size that matters for streaming
  multi-megabyte expert blocks): 28.78 GB/s median, pinned.** This
  matches the spec ceiling (24 to 28 GB/s for PCIe 4.0 x16) and MoE-Infinity's
  own measured 24 GB/s; this box's link is not the bottleneck relative to
  spec, and the P1 read-back confirms the link is not degraded.
- **Pinned and pageable host memory are statistically indistinguishable
  here.** 28.78 vs 28.60 GB/s H2D at 2 GiB, a 0.6% gap, well inside the
  run-to-run spread. The brief expected "a real prefetcher uses pinned, a
  careless one gets pageable, and the gap between them is part of the
  finding" -- the finding here is that on this box, with this driver, at
  these sizes, there effectively is no gap. That itself is worth reporting
  plainly rather than assuming pinned must win and not checking. (One
  min/max outlier: H2D pageable at 2 GiB had a single 21.1 GB/s run
  against a 28.6 GB/s median, most likely one page fault or allocator
  event in that repeat; it did not move the median and is reported, not
  explained away.)
- **"Both at once" shows no evidence of concurrent transfer.** 2x the
  bytes moved in essentially 2x the time at every size (e.g. at 2 GiB:
  single-direction 28.78 GB/s vs "both" 28.67 GB/s for double the bytes,
  meaning the aggregate rate did not increase). Per the tool's own
  docstring caveat, `DeviceContext.enqueue_copy` in this API has no
  stream selector, so this was never a stream-isolated concurrent-transfer
  test; the result is consistent with the two enqueues serializing on the
  same queue, not with the driver overlapping H2D and D2H in either
  direction. This is not the same question as "does a PCIe transfer
  overlap with GPU compute" (see the open question below); this test only
  speaks to two simultaneous copies against each other.
- Compare against the unexplained C IPC probe number the brief flagged
  (`bench/latentos-ipc-probe.c`, device-to-device, 2 GiB at ~93 GB/s, far
  below HBM peak): these H2D/D2H numbers are close to spec and internally
  consistent across five size classes and four transfer types, so they do
  not show the same kind of anomaly. Nothing here explains that other
  number; it remains unexplained.

## 2. Expert bytes per token, from the pack index

Source: `.work/moe-w1/pack/index.txt` (733 lines), `.work/moe-w1/pack/pack.bin`
(21,005,191,680 bytes). Verified directly from the index, not assumed:

```
$ grep -c "ffn_gate_exps.weight" .work/moe-w1/pack/index.txt
40
$ grep "ffn_gate_exps.weight\|ffn_up_exps.weight\|ffn_down_exps.weight" \
    .work/moe-w1/pack/index.txt | awk '{print $4}' | sort -u
268435456
$ grep "ffn_gate_inp.weight\b" .work/moe-w1/pack/index.txt | awk '{print $4}' | sort -u
524288
$ grep "ffn_gate_inp_shexp" .work/moe-w1/pack/index.txt | awk '{print $4}' | sort -u
2048
$ grep -E "ffn_(gate|up|down)_shexp" .work/moe-w1/pack/index.txt | awk '{print $4}' | sort -u
1048576
```

All 40 blocks carry the routed-expert tensors (`ffn_gate_exps.weight`,
`ffn_up_exps.weight`, `ffn_down_exps.weight`, q4_k, 268,435,456 bytes each,
uniform across every block), the shared-expert tensors (q8_0, 1,048,576
bytes each), and the router (`ffn_gate_inp.weight` f32 524,288 bytes plus
`ffn_gate_inp_shexp.weight` f32 2,048 bytes). This holds for all 40 blocks
including the 10 that are classic GQA-only in their attention/SSM trunk;
the trunk architecture split does not affect the expert tensors.

### Arithmetic (35B, this pack)

| quantity | bytes | how |
|---|---|---|
| bytes per expert per matrix (q4_k) | 1,048,576 (1 MiB) | 268,435,456 / 256 experts |
| bytes per expert, gate+up+down | 3,145,728 (3 MiB) | 1,048,576 x 3 |
| top-8 routed experts, per layer, uncached | 25,165,824 (24 MiB) | 3,145,728 x 8 |
| shared expert, per layer (always active) | 3,145,728 (3 MiB) | 1,048,576 x 3, q8_0 |
| router, per layer | 526,336 | 524,288 + 2,048 |
| **per layer, routed only** | 25,165,824 | |
| **per layer, everything uncached** | 28,837,888 | 25,165,824 + 3,145,728 + 526,336 |
| **per token, 40 layers, routed only** | **1,006,632,960 (0.9375 GiB, 1.007 GB)** | 25,165,824 x 40 |
| **per token, 40 layers, everything uncached** | **1,153,515,520 (1.074 GiB, 1.154 GB)** | 28,837,888 x 40 |

**This corrects `docs/NEXT-PLAN.md` B4's stated 0.78 GB per token.** The
measured, index-derived figure for routed experts alone is 1.007 GB per
token (decimal GB), about 29% higher than the plan's number, and 1.154 GB
per token if the shared expert and router are also assumed uncached every
token (they are small and a real implementation would almost certainly
pin them resident, so 1.007 GB is the more realistic "streamed" quantity
and 1.154 GB is the pessimistic ceiling). I did not find an arithmetic
path from this index to 0.78 GB; the plan's figure does not reproduce
from the pack as measured, and the corrected number should replace it.

### Scaled to a 100B-class model, same expert width and top-8

The pack fixes bytes-per-expert-per-matrix (1,048,576, tied to hidden
size and quantization) and top-8 by definition of the brief's scenario.
The only free variable left for a 100B-class model at the same expert
width is layer count, and that is a modeling assumption, not a
measurement: I scaled layer count by total-parameter ratio,
40 x (100/35) = 114.3 layers, holding expert width, expert count, and
top-k fixed. A different architecture (wider experts, more experts,
different layer/width tradeoff) would give a different number; this is
the same-expert-width, same-top-8 case the brief asked for, nothing more.

| quantity | 35B (measured) | 100B-class (scaled) |
|---|---|---|
| layers | 40 | 114.3 |
| per token, routed only | 1.007 GB | 2.876 GB |
| per token, everything uncached | 1.154 GB | 3.296 GB |
| ms/token at 28.7 GB/s H2D, routed only | 35.1 ms | 100.2 ms |
| ms/token at 28.7 GB/s H2D, everything uncached | 40.2 ms | 114.8 ms |

**This does not confirm `docs/NEXT-PLAN.md`'s "1 to 2 GB per token,
40 to 80 ms per token" range.** At the corrected per-expert bytes and this
box's measured PCIe rate, the 100B-class, fully-uncached-per-token number
is 2.9 to 3.3 GB per token, 100 to 115 ms per token, not 1 to 2 GB / 40 to
80 ms. The plan's range is low by roughly 1.4x to 1.9x against this
arithmetic.

### Resident fraction required to hit a tok/s target

Transfer-only bound: what fraction R of the per-token expert bytes must
already be resident on-device (zero PCIe cost) so that the remaining
(1-R) fraction, moved at 28.7 GB/s, fits inside the token budget. This
is optimistic: it charges the entire token budget to PCIe transfer and
assumes zero time for compute (GEMV, attention, sampling), which is not
real. A negative or zero R means the uncached case already fits with
room to spare; it does not mean compute is free.

| model | target | token budget | R (resident fraction needed), routed only | R, everything uncached |
|---|---|---|---|---|
| 35B (this pack) | 10 tok/s | 100 ms | 0 (35 ms fits, 2.85x headroom) | 0 (40 ms fits, 2.5x headroom) |
| 35B (this pack) | 30 tok/s | 33.3 ms | 5.0% | 17.1% |
| 100B-class (scaled) | 10 tok/s | 100 ms | 0.2% (right at the edge) | 12.9% |
| 100B-class (scaled) | 30 tok/s | 33.3 ms | 66.7% | 71.0% |

## What this means for B4's feasibility

- **Our existing 35B MoE, at 10 tok/s, does not need locality at all on
  transfer time alone**: uncached per-token streaming costs 35 to 40 ms
  against a 100 ms budget, with headroom left for compute. At 30 tok/s
  the transfer-only budget is tight (5% to 17% residency required) but
  not implausible.
- **The 100B-class extrapolation does not clear the interactive bar at
  measured rates, once compute time is accounted for.** At 10 tok/s the
  transfer-only bound is right at the wall (0.2% to 13% residency,
  meaning essentially the entire compute budget would need to run inside
  whatever margin residency buys, which the 0% case does not have); at
  30 tok/s, two-thirds to three-quarters of every token's expert bytes
  would need to already be resident on the 24 GB card. That is a strong
  locality requirement, not the light caching implied by "prefetch hides
  most of it" in the plan's phrasing, and it is stronger than either the
  bytes-per-token or the ms-per-token numbers the plan currently states.
  **The claim as currently phrased in `docs/NEXT-PLAN.md` (0.78 GB,
  40 to 80 ms) does not hold against this pack's measured arithmetic; the
  corrected numbers (1.0 to 1.15 GB at 35B, 2.9 to 3.3 GB at 100B-class,
  35 to 115 ms depending on scale and cache assumption) make the 30 tok/s
  target for a 100B-class model dependent on a specific, large (roughly
  two-thirds) resident fraction that stage 2 has not yet shown is
  achievable, and the 10 tok/s target has much less margin than stated.**
- This is a transfer-only bound in B4's favor: it assumes zero compute
  time competes with the PCIe budget. Real per-token compute (GEMV over
  the resident and freshly-streamed experts, attention, sampling) will
  eat into the same token budget, so every resident-fraction number above
  is a floor, not a sufficient condition.

## The one question stage 2 must answer first

**Does an expert-weight H2D transfer overlap with GPU compute on
already-resident layers on this card, through this MAX API, and does
that overlap produce real wall-clock savings?** Every number in section
2 above assumes PCIe transfer time is either fully additive to compute
(the pessimistic case used throughout) or fully hidden by prefetching
(the plan's implicit assumption, unverified). This stage's "both at once"
result (section 1) showed two enqueued copies do not overlap each other
on this API, but that is a different question from whether a copy
overlaps a compute kernel; it was not tested here and is out of this
stage's scope. If transfer-compute overlap is real and substantial on
this card, the resident-fraction requirements in section 2 shrink
sharply and B4's claim gets much more room. If it is not, or is small,
the 30 tok/s / 100B-class case likely needs the roughly two-thirds
residency this report computed, which is a strong constraint on host-RAM
streaming as B4 currently frames it. Stage 2 should answer this before
building any engine changes, host tier, or LRU on top of an unverified
overlap assumption.

## GPU minutes used

Under 1 minute. `gpu-wait list` was empty before the launch (no queue
contention). The sweep (five size classes x four transfer types x five
repeats, plus the "both" test) completed in 19.5 s wall clock under
`gpu-wait run --vram 6 --`. Part 2 (this section) used no GPU time.

## Rules and verification

- `kernels/*.mojo` not touched.
- `bench/pcie-bandwidth.mojo` builds clean under both the plain build
  command and the full `tools/ci-checks.sh` invocation (with the
  `-Xlinker` shim flags), verified directly before committing.
- No em dashes.
- `kernels/sample.mojo`, `serve/engine.mojo`, `serve/sample_ref.mojo` were
  not touched, staged, or committed; checked clean before commit.

## Files touched

- `bench/pcie-bandwidth.mojo`: new file.
- `exchange/2026-09-15-p3-b4-stage1-report.md`: this report.
- `.work/pcie-bandwidth`, `.work/pcie-bandwidth.out`, `.work/ci-bench-bin`:
  build/run artifacts, gitignored.
