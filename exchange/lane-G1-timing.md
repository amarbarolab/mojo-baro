# Lane G1 timing round: is the R5 M330 worth G2/G3?

Date 2026-09-17, branch `lane-g0`. Protocol and predictions were frozen at `b10b467`
(`exchange/lane-G1-timing-protocol.md`) and the timed run was made from that commit. Full run log:
`exchange/lane-G1-timing-run.log` (every TIME line carries its read-back). Tables below are generated
from that log by `tools/spirv-probe/timing-report.py`.

## Verdict

**Not worth G2/G3 as a speed target. By the frozen decision rule the card fails test (a) clearly and
test (b) by a hair. The lowering is not the problem; the card is.**

| arm, Qwen2.5-0.5B pure Q4_0, one token at a time | tok/s |
|---|---|
| llama.cpp on the box's CPU (i5-6200U), all GPU devices off (`-dev none`) | **54.47** |
| llama.cpp `-ngl 0` on the Vulkan build (not a clean CPU arm, see receipts) | 38.25 |
| llama.cpp Vulkan on the R5 M330 (`-ngl 99`) | **29.75** |
| ours as lowered, ceiling from the measured kernels (padded widths, upper bound) | **14.8** |

- (a) Does the card add anything on that machine? No. llama.cpp's mature Vulkan backend on the card
  reaches 29.75 tok/s; the same model on the laptop's own two cores reaches 54.47. The card is
  0.55x the CPU.
- (b) Is ours within 2x of llama.cpp Vulkan as lowered? At the line: ceiling 14.8 tok/s against
  0.5 x 29.75 = 14.9. Our GEMV is a steady 1.8x to 2.4x slower than llama.cpp's at every shape.
- Even a G2 that matched llama.cpp Vulkan kernel for kernel would land near 30 tok/s, still 0.55x
  the CPU. The card moved our q4 weights at 4 to 7.6 GB/s and llama.cpp's at roughly twice that at
  best; a 265 MiB model read once per token cannot beat the CPU's memory on this box.
- What the round does establish: the SPIR-V route costs nothing measurable. rmsnorm with its 11
  barriers runs at llama.cpp's speed (0.97x at rows 1, 1.02x to 1.07x at rows 16 to 64), and the
  per-dispatch floor through rusticl (about 40 us batched) equals RADV's under ggml (about 33 to 41
  us). So the G0/G1 converter work transfers to any better OpenCL 3.0 or Vulkan-class device; this
  particular card is just too small to beat its host.

The formula behind "ceiling" is tight on this card: the same sum over llama.cpp's own op times gives
30.5 tok/s at the true widths against its measured 29.75 (97.5%), so attention, rope and the rest
are small at short context and 14.8 is a fair estimate of ours, not only a bound.

## Results

### rmsnorm, H = 4096 (us)

| rows | ours sync med (min to max) | ours batch | host-reduction sync med (min to max) | host-reduction / lowered, med | same, min | reference sync | reference batched | ours batch / reference batched |
|---|---|---|---|---|---|---|---|---|
| 1 | 147 (132 to 169) | 39.6 | 186 (182 to 192) | 1.26 | 1.38 | 88 | 41.0 | 0.97 |
| 2 | 250 (171 to 414) | 65.5 | 189 (184 to 204) | 0.75 | 1.08 | 80 | 41.5 | 1.58 |
| 4 | 180 (171 to 265) | 62.5 | 193 (188 to 212) | 1.07 | 1.10 | 90 | 42.4 | 1.47 |
| 8 | 235 (194 to 340) | 76.7 | 200 (199 to 209) | 0.85 | 1.03 | 132 | 49.6 | 1.55 |
| 16 | 397 (232 to 454) | 88.7 | 274 (266 to 288) | 0.69 | 1.15 | 140 | 83.5 | 1.06 |
| 32 | 249 (226 to 730) | 152.7 | 560 (523 to 875) | 2.25 | 2.32 | 201 | 149.0 | 1.02 |
| 64 | 569 (526 to 688) | 300.4 | 1450 (1342 to 1663) | 2.55 | 2.55 | 351 | 281.5 | 1.07 |

### softmax, one row of 151936 (us)

| ours sync | ours batch | reference sync | reference batched | ours batch / reference batched |
|---|---|---|---|---|
| 1893 | 1622 | 357 | 304 | 5.33 |

### q4 GEMV, one token (us)

| N x K | ours q4rowb sync | ours q4rowb batch | ours reduce batch | ours pair batch | reference sync | reference batched | pair / reference | q4rowb alone / reference | our weight GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 1024 x 4096 | 650 | 375.6 | 44.9 | 420.5 | 248 | 209.8 | 2.00 | 1.79 | 6.3 |
| 2048 x 1024 | 438 | 256.9 | 46.7 | 303.6 | 190 | 124.9 | 2.43 | 2.06 | 4.6 |
| 1024 x 1024 | 219 | 135.6 | 45.4 | 181.0 | 139 | 81.9 | 2.21 | 1.66 | 4.3 |
| 1024 x 2048 | 353 | 213.4 | 46.2 | 259.6 | 192 | 129.3 | 2.01 | 1.65 | 5.5 |
| 3072 x 1024 | 615 | 358.4 | 49.8 | 408.2 | 267 | 167.2 | 2.44 | 2.14 | 4.9 |
| 1024 x 3072 | 521 | 309.1 | 47.5 | 356.6 | 246 | 178.7 | 2.00 | 1.73 | 5.7 |
| 151936 x 1024 | 15224 | 14490.8 | 508.9 | 14999.7 | 6447 | 6344.0 | 2.36 | 2.28 | 6.0 |
| 896 x 1024 | 204 | 124.8 | 45.0 | 169.8 | 139 | 78.1 | 2.17 | 1.60 | 4.1 |
| 128 x 1024 | 97 | 40.0 | 30.5 | 70.5 | 88 | 32.6 | 2.16 | 1.23 | 1.8 |
| 4864 x 1024 | 770 | 528.6 | 48.7 | 577.3 | 320 | 242.0 | 2.39 | 2.18 | 5.3 |
| 896 x 5120 | 711 | 422.8 | 46.6 | 469.4 | 299 | 229.2 | 2.05 | 1.84 | 6.1 |
| 2048 x 2048 | 592 | 393.9 | 43.8 | 437.7 | 252 | 214.5 | 2.04 | 1.84 | 6.0 |
| 6144 x 2048 | 1321 | 1016.8 | 45.0 | 1061.8 | 665 | 551.1 | 1.93 | 1.85 | 7.0 |
| 2048 x 6144 | 1252 | 936.7 | 47.1 | 983.8 | 661 | 553.0 | 1.78 | 1.69 | 7.6 |
| 151936 x 2048 | 23667 | 23113.8 | 505.6 | 23619.4 | 12599 | 12497.2 | 1.89 | 1.85 | 7.6 |

Reference only, Qwen2.5-0.5B true widths (ours cannot run them): 896 x 896 85.5, 128 x 896 39.7, 4864 x 896 259.5, 896 x 4864 233.1, 151936 x 896 6684.2 us batched.

### tok/s ceiling: L x (2 rmsnorm + q + 2 kv + o + 2 gate_up + down) + rmsnorm + head, batched medians

| model | ours us/token | ours ceiling tok/s | head share of ours | reference us/token | reference op ceiling tok/s | ours / reference |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, ours at padded widths | 67450 | 14.8 | 22% | 30783 | 32.5 | 0.46 |
| Qwen2.5-0.5B, true widths (reference only) | n/a | n/a | n/a | 32753 | 30.5 | n/a |
| Qwen3-0.6B | 76006 | 13.2 | 20% | 34752 | 28.8 | 0.46 |
| Qwen3-1.7B | 151933 | 6.6 | 16% | 80433 | 12.4 | 0.53 |

Identity: all 30 timed configurations passed their parity check in the same process right after the
timing pass (run log, `PASS` lines; last line `PASS timing-run`).

## Replication

The whole round was run a second time, from the tree with the CPU-arm fix, 8 minutes after the
first (`exchange/lane-G1-timing-run2.log`, 30 of 30 parity passes, `PASS timing-run`):

| quantity | run 1 | run 2 |
|---|---|---|
| llama-bench tg64, Radeon Vulkan | 29.75 | 29.72 |
| llama-bench tg64, CPU `-dev none` | 54.47 (post hoc) | 54.39 (in script) |
| our ceiling, Qwen2.5-0.5B padded / Qwen3-0.6B / Qwen3-1.7B | 14.8 / 13.2 / 6.6 | 14.5 / 13.2 / 6.5 |
| reference op ceiling, same three | 32.5 / 28.8 / 12.4 | 32.5 / 28.8 / 12.4 |
| GEMV 1024 x 4096 pair, ours / reference | 420.5 / 209.8 us = 2.00 | 418.9 / 210.5 us = 1.99 |
| head 151936 x 1024 pair, ours / reference | 14999.7 / 6344.0 us | 14999.6 / 6341.9 us |
| rmsnorm rows 1 batch, ours / reference | 39.6 / 41.0 us | 42.7 / 41.0 us |
| rmsnorm rows 64: host-reduction / lowered, sync medians | 2.55 | 3.30 |

Batched numbers repeat to within 1% to 8%; single-dispatch numbers move more, as the min to max
columns already say. With run 2's 14.5 the ceiling is below the 0.5 x line (14.9) in both runs.

## Barrier cost

- Batched, where the GPU is what is being timed: rmsnorm with 11 barriers costs 39.6 us per dispatch
  at rows 1. A dispatch with no barrier at all (`amar_skinny_reduce`, a copy) costs 44 to 50 us, and
  the 128 x 1024 GEMV with 10 barriers costs 40.0 us. llama.cpp's rmsnorm, which uses no barrier
  chain, costs 41.0 us. **At these sizes the barriers are inside the noise of the per-dispatch floor.**
- The control arm (no barrier, CPU-side reduction, two dispatches and a blocking read) against the
  lowered kernel, single dispatch plus `clFinish`: by medians the control arm is 0.69x to 1.26x at
  rows 1 to 16 (it wins at rows 2, 8 and 16) and loses 2.3x to 2.6x at rows 32 and 64, where the
  read-back grows. By minimums the lowered kernel wins at every row count (1.03x to 2.55x). The
  lowered kernel's single-dispatch times are noisy (171 to 414 us at rows 2), the control arm's are
  tight (184 to 204 us); I did not find out why and do not claim a cause.
- Either way the answer for G1 is the same: a host-side reduction buys nothing, and it cannot be
  batched at all (its read is a sync), so it would cost 186 us where the lowered kernel costs 40 us
  inside a token. The barrier-count optimisations listed in the G0 report (ping-pong scratch, a
  `warp.sum` idiom) are not worth building for this card.

## Where ours loses

- **GEMV, 1.8x to 2.4x behind at every shape**, large and small, so it is not dispatch overhead. The
  pair includes `amar_skinny_reduce`, which at m = 1 with one split is a plain copy and costs a fixed
  44 to 50 us: 11% of the 1024 x 4096 pair and 43% of the 128 x 1024 pair. Dropping it (writing the
  GEMV result straight to the output at m = 1) takes the gap to 1.2x to 2.3x ("q4rowb alone"
  column). The rest is the kernel's shape: one row dealt to 32 threads with a 16-wide vector body
  that radeonsi has to scalarize, designed for the 7900 XTX. That is a hypothesis from reading the
  kernel, not a measurement; the kill test would be a row-per-thread scalar variant on this card.
- **Softmax over the vocabulary, 5.3x behind** (1622 us against 304 us). 256 threads each walk all
  151936 entries at stride 256, twice, around 23 barriers. Not diagnosed further. It is off the
  greedy decode path (argmax is used there) and is not in the ceiling.
- **The head**: 15.0 ms of our 67.5 ms token (22%), 6.3 ms of llama.cpp's 32.8 ms (20%). Same 2.4x.

## Predictions, scored

| # | prediction (frozen) | measured | score |
|---|---|---|---|
| 1 | GEMV 1024x4096 ours batch 5 ms (2 to 12) | 0.42 ms pair, 0.376 ms kernel | **missed, 12x too pessimistic** |
| 2 | reference 0.8 ms (0.3 to 2); ours / reference >= 3 | 0.21 ms; 2.00 | missed on both (reference faster, gap smaller) |
| 3 | rmsnorm rows 1: ours sync 400, batch 150, reference 60 us | 147, 39.6, 41.0 | missed, all faster; reference part was not preregistered (seen in verify) |
| 4 | rmsnorm batch rows 64 / rows 1 <= 8x | 7.6x | hit |
| 5 | host-reduction / lowered sync > 1 everywhere, >= 1.5 at rows 1 | medians 1.26, 0.75, 1.07, 0.85, 0.69, 2.25, 2.55 | **missed** (three row counts below 1, rows 1 below 1.5); by minimums all above 1 |
| 6 | softmax ours batch 3 ms (1 to 8); reference 1.5 ms | 1.62 ms; 0.30 ms | ours hit; reference missed and was not preregistered |
| 7 | head 151936x1024: ours >= 300 ms, reference about 60 ms | 15.0 ms; 6.3 ms | **missed, 20x and 10x too pessimistic** |
| 8 | llama-bench Radeon 15 tok/s (8 to 25) | 29.75 | missed, and not preregistered (tg4 seen in verify) |
| 9 | llama-bench CPU 28 tok/s (18 to 40), CPU beats the card | 54.47 clean, 38.25 with `-ngl 0` | direction hit, number missed (clean arm above the range) |
| 10 | our ceiling <= 5 tok/s and < 0.5 x Vulkan tg | 14.8; 0.497 x | first half missed 3x, second half hit by 0.1 tok/s |

My model of this card was wrong by an order of magnitude on absolute speed (I priced the barriers
and the thread count as expensive; they are nearly free) and right on the ranking. The predicted
verdict stands, for a different reason than the one I gave: not because the lowering is slow, but
because the card cannot outrun its host's memory.

## Receipts and caveats

- Devices: ours `AMD Radeon R5 M330 (radeonsi, hainan, ACO, DRM 3.64)`, driver 26.2.2, OpenCL 3.1;
  reference `AMD Radeon R5 M330 (RADV HAINAN)` as `Vulkan1` in test-backend-ops and as the only
  visible device in llama-bench (the box's Intel HD 520 is `Vulkan0`). llama.cpp `ca3d5a3e1` plus
  the shapes patch, Release, `GGML_VULKAN=ON`. sha256 prefixes of `host`, `test-backend-ops`,
  `llama-bench` and the model are in the log.
- Clock state, sampled every 0.5 s: ours 41 of 46 samples at power level 4 (sclk 750 MHz, mclk
  1000 MHz), reference ops 181 of 190, llama-bench on the Radeon 23 of 23. The few level 0 and 1
  samples fall between kernels, while the host computes fp64 references or builds programs.
- **The scripted CPU arm was not a clean CPU arm.** `-ngl 0` on a Vulkan build gave 38.25 tok/s.
  Rerun after the round with `-dev none` (`exchange/lane-G1-timing-cpu-devnone.log`): 54.47 tok/s
  with the Radeon at power level 0 for 13 of 13 samples. That rerun is post hoc and outside the
  frozen script. I then changed `time.sh` to `-dev none` and ran the whole round again (next
  section): 54.39. Why `-ngl 0` is 30% slower than `-dev none` is not established; the clock
  samples cannot tell, because the card lingers at level 4 for several seconds after the Radeon arm
  (9 of 13 samples during the second run's `-dev none` arm, at an unchanged 54.4 tok/s).
- The verify pass leaked three reference numbers before the freeze; the protocol discloses it and
  the affected predictions are marked above.
- Reference batched numbers are one graph of up to 8192 copies of the op (the tool's design), so
  "med" there is a single sample that is itself a mean over thousands of ops; ours is the median of
  11 batches of 32. Submission is amortised over 8192 ops on their side and 32 on ours, which
  favours the reference on the smallest ops by at most the per-dispatch floor.
- Ours runs bf16 activations and llama.cpp f32. Each side's own design, not normalised.
- Qwen2.5-0.5B: the lowered GEMV cannot run K = 896 or 4864 (K must be a multiple of 1024), so ours
  is timed at padded widths, 14% and 5% more columns. At equal padded shapes the reference sums to
  32.5 tok/s, at true shapes 30.5, so padding is not what separates the arms.
- Nothing here was run on the workstation GPU; no gpu-wait jobs.
- Not measured: attention, rope, swiglu, embed, argmax timing, anything at m > 1, prefill, and any
  end-to-end number of ours (there is no engine on that box).

## If the lab box stays a target anyway

The cheapest wins, in order, all measured above: drop the m = 1 reduce dispatch (up to 43% of a
small GEMV), then a row-per-thread GEMV variant for narrow SIMD hardware (the 2x), then the K % 1024
constraint (needed before any real 0.5B model runs at all). None of it gets past the CPU on that
machine.

## Reproduce

```
./tools/spirv-probe/time.sh verify   # parity and device selection at every shape
./tools/spirv-probe/time.sh run      # the timed round, about 4 minutes on the lab box
python3 tools/spirv-probe/timing-report.py .work/spirv-probe/timing/run-<utc>.log
```
