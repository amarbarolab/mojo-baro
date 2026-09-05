# Ternary GEMV protocol — frozen before first run

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in
> particular: every parameter defining an arm is read back from the running
> system and recorded BEFORE the timed run. No receipt, no arm.

Question: cold-cache wave-per-row GEMV throughput for the three ternary
packs (`q2b3` = Q2_B3/B3S, `tq1` = TQ1_0, `tq2` = TQ2_0) at the ffn_gate
shape, against the landed `q8row` kernel as control. Correctness gate
(`kernels/test_ternary_gemm.mojo`) already passes; this round is bytes and
decode cost only.

## Instrument

`bench/bench_coldcache_ternary.mojo`, sibling of `bench/bench_coldcache_q8row.mojo`:
M=1, K=4096, N=12288 (blk.0.ffn_gate.weight), NBUF=8 distinct device
buffers per pack, 1 s clock warm (same rotation), 200 timed launches/rep,
10 reps. A = row 0 of the existing `.work/gguf/blk_0_ffn_gate_weight.a.bin`.
Correctness: fp64 host reference. q8row's reference is an in-kernel fp64
dot (same as `bench_coldcache_q8row.mojo`); the ternary references are
precomputed by `bench/gen-ternary-ref.py` -- C-dequantized production pack
(`tools/ternary-ref.c` via `tools/b3s-check.py`'s ctypes binding) dotted in
fp64 against the same A row -- into
`.work/gguf/blk_0_ffn_gate_weight.<fam>.ref1.bin`.

Bytes per launch (payload + fp16 scales, read back from each pack's
`index.txt` and confirmed against the file sizes actually loaded):
q8 53,477,376; q2b3 11,010,048; tq1 10,616,832; tq2 12,976,128.

## P1 receipt: q8row control does not reproduce its 2026-09-04 number

`bench/q8-protocol.md` records `amar_matmul_skinny_q8row[UNROLL=4]` at
62.6 us / 855 GB/s, clock 3069 MHz median, 253-312 W, on this same ffn_gate
shape. Re-running the ORIGINAL, unmodified `bench/bench_coldcache_q8row.mojo`
tonight (both the pre-existing prebuilt binary and a fresh rebuild from
current `kernels/matmul_skinny.mojo`) under `bench/clock-probe.sh` gives:

| run | q8row us (min-max, 10 reps) | sclk med (min-max) | power | junction |
|---|---|---|---|---|
| prebuilt binary, 1st (contaminated, see below) | 67.7-69.4 | 3069 (2724-3318) | 149-324 W | 71 C |
| prebuilt binary, 2nd (clean) | 68.5-69.7 | 3064 (2710-3310) | 160-316 W | 71 C |
| fresh rebuild, current source | 69.9-70.8 | 3068 (2747-3316) | 161-319 W | 70 C |

The bf16 arms in the same binary (`m1c8`, `row8`) reproduce their historical
numbers exactly (121.0-121.2 us and 117.3-117.6 us respectively, matching
`bench/q8-protocol.md`'s 121.1-121.2/117.4-117.5 to 3 significant figures)
in all three runs. Only `q8row` reads ~68-71 us instead of 62.6 -- at a
matched median clock (3064-3069 MHz vs the historical 3069 MHz) and matched
junction temp (70-71 C vs the historical run's clocks). A driver note
midway through this round reported a runaway `llama-cli` from lane C held
the GPU for ~3 min around 22:49-22:52 and was cancelled; the first
contaminated run and this session's original `bench_coldcache_ternary` T0
run both fall partly in that window and were re-run clean afterward
(`gpu-wait gpu` confirmed 0% busy, no stray processes, before each clean
re-run). The clean re-run and the fresh-rebuild run agree with the
contaminated run to within their own spread, ruling out that contamination
as the explanation.

**Conclusion**: this is a real, reproducible, clock-matched ~10-12% q8row
regression against the `bench/q8-protocol.md` receipt, present in the
unmodified control kernel and therefore out of this lane's scope to fix
(`kernels/matmul_skinny.mojo` is read-only for lane A). Since the
UNMODIFIED original harness reproduces the SAME ~68-71 us this session's
new `bench_coldcache_ternary.mojo` measures for the identical kernel and
shape, the two harnesses cross-validate each other -- `bench_coldcache_ternary.mojo`
is not broken, it agrees with the pre-existing bench to within run-to-run
spread. This session's q8row control number is therefore **70.85 us
(median of 10, clean run, clocks below)**, not 62.6 us; every ternary
arm below is compared against that measured control, not the stale
protocol figure. Flagged for the driver / other lanes: `bench/q8-protocol.md`'s
"round closed" verdict may need revisiting outside this brief's scope.

## Frozen predictions

| arm | prediction | reasoning |
|---|---|---|
| q2b3row | 40-120 us; likely SLOWER than q8row despite 4.9x fewer bytes | 26 scalar loads/lane/block, no prefetch, ~64 div/mod per 32-trit chunk |
| tq1row | 30-80 us | 13x 4 B loads, mul-shift decode |
| tq2row | 20-45 us | 4x 16 B loads, shift-mask; closest to q8row's load pattern |

Round falsifier: if q2b3row as written lands <= 25 us, the decode is
already free -- record that, skip T2, write DONE.

## Claim rule

Report min/max over 10 repeats, spread, clocks (`rocm-smi` read-back via
`bench/clock-probe.sh`), grid/block/template params echoed by the binary.
Spread > 5% voids the arm.

## T1 baseline result

Clean run, GPU idle before start (`gpu-wait list` -> `(no jobs)`), through
`bench/clock-probe.sh` + `gpu-wait run --priority 90`.

`grid_dim=1536 block_dim=256 ROW_WAVES=8 MR=1` (read back from the binary's
own print, matching `ceildiv(N, ROW_WAVES)` and `ROW_THREADS` for N=12288).

Correctness (fp64 host reference, same run):

| arm | max_rel | abs/maxref | correct |
|---|---|---|---|
| q8row | 9.44e-05 | 1.15e-07 | true |
| q2b3row | 6.90e-05 | 7.28e-08 | true |
| tq1row | 6.71e-05 | 7.94e-08 | true |
| tq2row | 6.71e-05 | 7.94e-08 | true |

Timing (10 reps, us):

| arm | min | max | spread | GB/s (median) | prediction | land |
|---|---|---|---|---|---|---|
| q8row (control) | 71.32 | 72.39 | 1.5% | 746 | -- (see P1 receipt above) | -- |
| q2b3row | 44.02 | 44.30 | 0.6% | 250 | 40-120, likely slower than q8row | in range; **beats q8row**, reasoning note wrong |
| tq1row | 53.56 | 53.88 | 0.6% | 197 | 30-80 | in range |
| tq2row | 45.81 | 46.30 | 1.1% | 282 | 20-45 | **misses ceiling by 2.3%** (46.05 vs 45) |

Clocks (`bench/clock-probe.sh`, whole 4-arm run): sclk 2830 MHz median
(2781-3018 MHz range), power 269-293 W, junction max 85 C. Sustained
4-arm x 10-rep load runs the card warmer than an isolated q8row-only
run (3064-3069 MHz median in the P1 receipt above); this is read back
and disclosed per P1, and is the reason this run's q8row control
(71.68 us median) sits a bit above the isolated control runs
(69-70.85 us) rather than a regression from this run itself.

Falsifier check: q2b3row (44.10 us median) is well above the 25 us
falsifier threshold -- decode is NOT free, proceed to T2.

Bytes moved (read back from the binary, matches the index-derived byte
counts above): q8=53,477,376 q2b3=11,010,048 tq1=10,616,832 tq2=12,976,128.

## T2 kernel round

| lever | change | frozen prediction |
|---|---|---|
| L1 vector loads | q2b3: load the 26 payload bytes as 6x 4-byte + 1x 2-byte vector loads (stride 26 is 2-aligned, not 16-aligned -- pack unchanged); tq1/tq2 unchanged | q2b3 -30 to -50% |
| L2 UNROLL | prefetch UNROLL blocks of payload+scale before the FMA loop, as q8row's qs[u]/ds[u] | -10 to -25% on top |
| L3 chunk striding | q2b3: lanes own 32-trit chunks instead of whole blocks; measure both shapes | 0% gate / -5 to -15% down |
| L4 LUT decode | 256-entry table decode, port of ggml_cuda_b3lut_x | -20 to -40% of remaining time |
| L5 int8 dot | only if gfx1100 exposes a 4x int8 dot; else skip | uncertain; last |

### L1 receipt: ABLATED -- regressed 4%, not the predicted -30 to -50%

Implementation: `Q.ptr` (TileTensor's raw base pointer -- confirmed via a
throwaway probe kernel that `TileTensor.__getattr_param__` only supports
`ptr`, not `unsafe_ptr()`) plus `MutPointer.unsafe_load[width=N](byte_off)`
for absolute, possibly-2-byte-aligned-only byte offsets -- this DOES
compile and run correctly (unaligned 4-byte loads at odd-block offsets
gave bit-correct results on both K=4096 and K=12288 shapes, `test_ternary_gemm`
PASS). But it is slower, not faster:

| arm | T1 baseline median | L1 median | delta | spread |
|---|---|---|---|---|
| q2b3row | 44.10 us | 45.88 us | **+4.0%** | 2.5% |
| tq1row (unchanged, control) | 53.78 us | 53.75 us | -0.1% (noise) | 0.9% |
| tq2row (unchanged, control) | 46.05 us | 46.13 us | +0.2% (noise) | 1.5% |
| q8row (unchanged, control) | 71.68 us | 71.72 us | +0.1% (noise) | 1.5% |

The three untouched arms sit within their own run-to-run noise of the T1
baseline (clocks: sclk 2835 MHz median, 2530-3044 MHz range, power
100-317 W, junction max 84 C -- comparable thermal state to T1), so the
q2b3row regression is real and attributable to the code change, not
environment drift. Reading the raw pointer directly bypasses whatever
the TileTensor library's own scalar-index path (`Q[row, i]`) does for
byte-level global loads on this shape/hardware -- plausibly the unaligned
4-byte load decomposes into multiple sub-word memory ops on gfx1100
(RDNA3 requires natural alignment for wide global loads; misaligned
`global_load_b32` either faults or the compiler splits it), erasing any
transaction-count win and adding address-computation overhead on top.
Per the T2 ablation rule, reverted; `kernels/matmul_ternary.mojo`
q2b3row is back to its T1 form (26 scalar `Q[row, i]` loads).

### L2 receipt: ABLATED -- geometric no-op made worse by refactor overhead

q2b3's block/lane geometry defeats this lever before it can help: at
K=4096, `nb = K // B3_BLOCK = 32 == WARP_SIZE`, so each lane owns exactly
ONE block total (the per-lane `while b < nb: ... b += WARP_SIZE` loop body
runs once). There is nothing to prefetch AHEAD of -- a lane with one block
has no next block. Any `UNROLL >= 2` therefore routes every lane through
the tail (single-block) path, identical in behavior to the T1 kernel,
just now calling a factored-out `decode_block(...)` closure instead of
inlined code (verified: with `UNROLL=2`, `b + UNROLL*WARP_SIZE <= nb` is
`lane + 64 <= 32`, false for every lane 0-31).

Measured anyway (`UNROLL=2`, so the down shape's parity check in
`test_ternary_gemm` -- K=12288, nb=96 -- does exercise the unrolled branch
once per lane, proving the refactor itself is bit-correct on both shapes):

| arm | T1 baseline median | L2 median | delta |
|---|---|---|---|
| q2b3row | 44.10 us | 50.20 us | **+13.8%** |
| tq1row/tq2row/q8row (untouched) | -- | within 1% of T1 | noise |

Correctness held (`test_ternary_gemm` PASS, both shapes, both MR). The
regression is pure closure/function-call overhead on the dead-for-gate
`decode_block` factoring -- no prefetch ever fires for this shape, so
100% of the delta is overhead, not a real load-latency-hiding trade.
Reverted per the T2 ablation rule; `kernels/matmul_ternary.mojo`,
`kernels/test_ternary_gemm.mojo`, and `bench/bench_coldcache_ternary.mojo`
are all back to their T1 form (`git diff --stat` on the three files is
empty against `main`... against the L1-ablation commit, confirmed clean).

This is the same root cause L3 (chunk striding) exists to fix: L3's own
prediction is "0% gate" for exactly this reason (nb == WARP_SIZE exactly
means there is no room to spread work more finely without changing the
per-lane unit from whole-block to sub-block). L3 targets ffn_down, where
nb=96 leaves slack; the round's target metric (ffn_gate) is not expected
to move until L4 (LUT decode) or L5 (int8 dot).

## Fork

(placeholder for lane C)
