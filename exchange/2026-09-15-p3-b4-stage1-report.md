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

**Correction (this section rewritten after review):** the first version of
this report read `index.txt` column 4 as a byte count. It is not.
`tools/engine-pack.py` documents its own index format in its header
comment: `name dtype offset_bytes n_elem`. Column 4 is the element count,
not bytes; treating 268,435,456 (the element count of a routed-expert
tensor, 256 experts x 1024x1024) as if it were a byte count overstated
every downstream number by roughly 1/0.5625, since q4_k packs 256
elements into 144 bytes, not one byte per element. The true byte size of
every tensor is recoverable two ways, and both were checked against each
other: the block-quant formula (q4_k = 144 B / 256 elem = 0.5625 B/elem,
q8_0 = 34 B / 32 elem = 1.0625 B/elem, q6_k = 210 B / 256 elem = 0.8203
B/elem, f32 = 4 B/elem), and directly from the index itself, since tensors
are laid out contiguously: the byte size of a tensor is the offset gap to
the next tensor. Both agree exactly, and the sum of every gap across the
whole index equals `pack.bin`'s size to the byte:

```
$ python3 - <<'EOF'
rows = []
for l in open('.work/moe-w1/pack/index.txt'):
    name, dt, off, n = l.split()
    rows.append([name, dt, int(off), int(n)])
rows.sort(key=lambda r: r[2])
pack_size = 21005191680
total = 0
for i, (name, dt, off, n) in enumerate(rows):
    nxt = rows[i+1][2] if i+1 < len(rows) else pack_size
    total += nxt - off
print(total, pack_size, total == pack_size)
EOF
21005191680 21005191680 True
```

Doing this per block also surfaces something the naive uniform read
missed: **`ffn_down_exps.weight` is not q4_k on every block.** 37 of 40
blocks are q4_k (150,994,944 bytes for the 256-expert tensor); blocks 34,
38, and 39 are q6_k (220,200,960 bytes), a higher-precision quant on that
one matrix only, consistent with this being a "UD" (dynamic-precision)
GGUF quant that keeps a handful of sensitive tensors wider. `ffn_gate_exps`
and `ffn_up_exps` are q4_k uniformly across all 40 blocks; the shared
expert (q8_0, 1,114,112 bytes per matrix) and the router (f32, 2,097,152 +
8,192 bytes) are uniform too. This affects the total by about 1%, not the
order of magnitude, but it is the real, non-uniform number, computed per
block rather than assumed constant.

### Arithmetic (35B, this pack)

| quantity | bytes | how |
|---|---|---|
| bytes per expert per matrix, q4_k (37/40 blocks' down, all gate/up) | 589,824 | 150,994,944 / 256 experts |
| bytes per expert, down matrix only, q6_k (blocks 34, 38, 39) | 860,160 | 220,200,960 / 256 experts |
| bytes per expert, gate+up+down, q4_k blocks | 1,769,472 | 589,824 x 3 |
| bytes per expert, gate+up+down, q6_k-down blocks | 2,039,808 | 589,824 x 2 + 860,160 |
| top-8 routed experts, per layer, q4_k blocks (37 of 40) | 14,155,776 | 1,769,472 x 8 |
| top-8 routed experts, per layer, q6_k-down blocks (3 of 40) | 16,318,464 | 2,039,808 x 8 |
| shared expert, per layer (always active, q8_0) | 3,342,336 | 1,114,112 x 3 |
| router, per layer (f32) | 2,105,344 | 2,097,152 + 8,192 |
| **per token, 40 layers, routed only** | **572,719,104 (0.573 GB)** | 37 x 14,155,776 + 3 x 16,318,464 |
| **per token, 40 layers, everything uncached** | **790,626,304 (0.791 GB)** | routed + 40 x (3,342,336 + 2,105,344) |

**This confirms, not corrects, `docs/NEXT-PLAN.md` B4's 0.78 GB per
token figure**, computed correctly (the plan's number was right; the
error was entirely in this report's first pass, not in the plan). The
precise, per-block figure including the shared expert and router is
0.791 GB per token, 1.4% above the plan's stated 0.78 GB, and the gap is
fully explained by the 3 q6_k-down blocks pushing the uniform-quant
estimate (0.784 GB) up slightly. Routed experts alone: 0.573 GB per
token.

### Scaled to a 100B-class model, same expert width and top-8

Same method as before: expert width, expert count, and top-k held fixed,
layer count scaled by total-parameter ratio, 40 x (100/35) = 114.3
layers. The corrected per-layer average (using the actual 35B pack's
mixed q4_k/q6_k mix, 572,719,104 / 40 = 14,317,978 bytes routed,
790,626,304 / 40 = 19,765,658 bytes full) is what gets scaled, since a
100B-class model's own precision mix is unknown and holding the measured
average fixed is the same "same expert width" assumption as before,
applied correctly this time.

| quantity | 35B (measured) | 100B-class (scaled) |
|---|---|---|
| layers | 40 | 114.3 |
| per token, routed only | 0.573 GB | 1.636 GB |
| per token, everything uncached | 0.791 GB | 2.259 GB |
| ms/token at 28.78 GB/s H2D, routed only | 19.9 ms | 56.9 ms |
| ms/token at 28.78 GB/s H2D, everything uncached | 27.5 ms | 78.5 ms |

**This confirms `docs/NEXT-PLAN.md`'s "1 to 2 GB per token, 40 to 80 ms
per token" range**, with one caveat worth stating plainly rather than
rounding away: the routed-only number (1.636 GB, 56.9 ms) sits
comfortably inside the plan's range; the everything-uncached number
(2.259 GB, 78.5 ms) sits just outside the top of the stated 1-2 GB
band (13% over) while its time figure (78.5 ms) still lands inside the
stated 40-80 ms. Which of these is the right one to cite depends on
whether the shared expert and router are assumed resident (a reasonable
assumption in a real implementation, since they are small, constant
every token, and the obvious first thing to pin) or streamed from host
RAM like the routed experts every token (the pessimistic case). With that
one caveat, the plan's range holds.

### Resident fraction required to hit a tok/s target

Transfer-only bound: what fraction R of the per-token expert bytes must
already be resident on-device (zero PCIe cost) so that the remaining
(1-R) fraction, moved at 28.78 GB/s (this box's measured, 2 GiB H2D
pinned median, section 1), fits inside the token budget. This is
optimistic: it charges the entire token budget to PCIe transfer and
assumes zero time for compute (GEMV, attention, sampling), which is not
real. A negative or zero R means the uncached case already fits with
room to spare; it does not mean compute is free.

| model | target | token budget | R (resident fraction needed), routed only | R, everything uncached |
|---|---|---|---|---|
| 35B (this pack) | 10 tok/s | 100 ms | 0 (19.9 ms fits, 5x headroom) | 0 (27.5 ms fits, 3.6x headroom) |
| 35B (this pack) | 30 tok/s | 33.3 ms | 0 (19.9 ms fits, 1.7x headroom) | 0 (27.5 ms fits, 1.2x headroom) |
| 100B-class (scaled) | 10 tok/s | 100 ms | 0 (56.9 ms fits, 1.8x headroom) | 0 (78.5 ms fits, 1.3x headroom) |
| 100B-class (scaled) | 30 tok/s | 33.3 ms | 41.4% | 57.5% |

## What this means for B4's feasibility

- **Our existing 35B MoE needs no residency on transfer time alone, at
  either 10 or 30 tok/s.** Uncached per-token streaming costs 19.9 to
  27.5 ms; even the 30 tok/s budget (33.3 ms) has headroom left over for
  compute (1.2x to 1.7x depending on whether the shared expert and router
  are assumed resident).
- **The 100B-class extrapolation clears 10 tok/s on transfer time alone
  (56.9 to 78.5 ms against a 100 ms budget), but 30 tok/s requires real
  locality.** At 30 tok/s the transfer-only bound needs 41% to 58% of
  every token's expert bytes to already be resident on the 24 GB card;
  that is the point at which B4's claim stops being "streaming with light
  prefetch" and starts depending on a specific, large cache-hit rate that
  stage 2 has not yet shown is achievable.
- **`docs/NEXT-PLAN.md`'s stated numbers (0.78 GB, 1 to 2 GB at 100B-class,
  40 to 80 ms) hold, with the one caveat above about whether the shared
  expert and router are counted as streamed or resident.** The claim that
  needed correcting was this report's first pass, not the plan.
- This is a transfer-only bound in B4's favor: it assumes zero compute
  time competes with the PCIe budget. Real per-token compute (GEMV over
  the resident and freshly-streamed experts, attention, sampling) will
  eat into the same token budget, so every resident-fraction number above
  is a floor, not a sufficient condition, and the 10 tok/s "0 residency
  needed" rows are transfer-only headroom, not a claim that compute is
  free.

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
the 30 tok/s / 100B-class case likely needs the 41% to 58% residency
this report computed, which is a real constraint on host-RAM streaming
as B4 currently frames it, even though the 10 tok/s case and the plan's
bytes-per-token range hold up without it. Stage 2 should answer this
before building any engine changes, host tier, or LRU on top of an
unverified overlap assumption.

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
