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

## Fork

(placeholder for lane C)
