# Lane dattn report: generalised decode attention (2026-09-11)

Protocol: `bench/dattn-protocol.md` (frozen `d8fbb44`). Branch `lane-dattn`, not merged.

**Verdict: BETWEEN, no land.** Confirmation run c4 (commit `7721fe0`, 10 repeats in
alternating arm order, spread <= 1.5 % on every verdict row, every gate PASS in the same
job): ours beats llama.cpp's `flash_attn_ext_vec` by **1.43x (S1), 1.70x (S2), 1.48x (S3)**.
The land rule needs >= 1.5x on two shapes; only S2 clears it (S3 misses by 0.02, S1 by
0.07). Close negative needs < 1.1x on two shapes; none. Per the protocol: no land, and one
more round only with a new frozen prediction.

The frozen ratio predictions all missed because R's receipts were timed at 200 iterations
and were clock-ramp inflated 1.2-1.6x; ours held its own predictions on S1 and S2 and
missed on S3. The kernel is correct (760/760 numerics, bit-identical at the shipped
instantiation) and not wired into the engine.

## Gates

| gate | check | result | where read |
|---|---|---|---|
| 1 P1 read-back | every O run echoes HD, NQH, NKVH, KVT, KV, path, nsplit, NLD, ROT, span, grid, block, combine dims, bytes/arm, arms, rotated MB; every R run prints op_bench's arm line; VGPR / scratch / LDS / workgroup / grid per kernel from the rocprofv3 trace | echoed on every run | `stdout.txt` and `trace_kernel_trace.csv` per target |
| 1b R rotation | op_bench arm line per R target after the 256-arm cap | every target 402.7 MB, "exceeds" (KV 512: 192 arms, 1024: 96, 4096: 24, 16384: 6; S2 / S3: 48) | `rep1/R/*/stdout.txt` in the confirmation dir |
| 2 numerics | fp64 numpy softmax(QK^T scale)V over the same f16 bytes, every shape (S0 shipped f32, S1-S3 f16) x KV {1, 127, 128, 129, 4096} x exact / split ns {1, 8, 64} x NLD {2, 4, 8} x ROT {0, 1} x q scale {1, 8} | **PASS 760/760**, worst 1.43e-5 relative to max abs out, bound 2e-3 | `tools/dattn-ref.py`, `.work/dattn/ref-full4.log` |
| 3a bit-identity | `amar_dattn_exact[256, 16, 4, f32]` vs `amar_attn_decode` on `kernels/test_attn_block.mojo`'s Q and KV cache | **PASS**, 0 of 4096 words differ (confirmation job, rebuilt in the same command) | `gate3.txt` in the confirmation dir |
| 3b engine gate | `tools/mega-gate.sh`: build, test_mega_block, `./run-tests.sh`, mega == launch identity on every pack, q8 / q4 vs reference tokens | **ALL PASS** in c4: build, kernel, tests, mega == launch identity q8 / q8d / q4 x spec 0/1, q8 and q4 vs ref-tokens-64, perf ratio 1.201 | `mega-gate/SUMMARY.txt` in the confirmation dir |

The engine does not call the new kernel: `git diff main -- serve/ kernels/attn.mojo kernels/mega.mojo`
is empty. Gate 3b therefore shows the lane leaves the shipped path intact; it cannot exercise
the new kernel, which only a later wiring round can.

## What was built

- `kernels/dattn.mojo`, comptime `HD`, `NQH`, `NKVH`, `KVT`, runtime scale, paged `kv_off` layout.
  - `amar_dattn_exact`: `attn_head_span` with the Qwythos constants lifted to parameters and
    nothing else changed. Kept for bit-identity; in stint 3 it was slower than the split path
    at every length measured (2.5x at KV 512, 3.9x at 1024, 8x at 4096 and 16384) and should
    not be the engine's decode path.
  - `amar_dattn_split`: one block per (KV head, split), 8 waves, every wave-wide load one
    contiguous 512 B. All G = NQH / NKVH query heads of a KV head share each K/V load (the
    GQA reuse `amar_attn_decode` does not have: it re-reads K/V once per query head).
    q in LDS pre-scaled, q.k partials reduced by recursive halving, online softmax per wave,
    waves merged in wave order through LDS (deterministic), partials `(m, l, o[HD])` per split.
  - `amar_dattn_combine`: block-parallel over splits.
- `kernels/dattn_harness.mojo` (shared launch code), `kernels/test_dattn.mojo` + `tools/dattn-ref.py` (gate 2),
  gate 3 added to `kernels/test_attn_block.mojo`.
- `bench/bench_dattn.mojo` (arm O, op_bench's rotation rule), `bench/dattn-run.sh` (sweep, both arms
  in one job), `bench/dattn-confirm.sh` (gates then 10 alternating repeats, fail-closed).

## Exploration history (sweeps, not results)

Best device us/iter at KV 4096 per stint. Each row is the best of a sweep, picked after
seeing the numbers, so none of them is a claim.

| stint | kernel cut | S1 best | S2 best | S3 best | R S1 / S2 / S3 |
|---|---|---|---|---|---|
| 1 | cut 1: per-wave loads, q in registers | 30.8 | 24.5 | 42.6 | 64.0 / 40.5 / 66.9 |
| 2 | cut 3: + prefetch knob, 16-wave knob, block-parallel combine | 27.3 (no prefetch) | 18.8 (no prefetch) | 30.6 (no prefetch) | 56.6 / 34.4 / 64.7 |
| 3 | cut 4: prefetch removed, ROT and NLD 2 added | 26.4 | 16.8 | 27.9 | 57.2 / 35.0 / 66.0 |

What the sweeps ruled out (receipts in `.work/dattn-run/stint{1,2,3}/summary.md`):
- Software prefetch through a second register array: 144 VGPR / 0 B scratch became 192 VGPR /
  48-632 B scratch, and S1 went 25.4 to 32.3 us. Removed in cut 4.
- 16 waves per block: slower than 8 on every shape (S1 34.5 to 41.2 us, S3 44.3 to 82.6 us).
- NLD 8 at HD 128 spills (284-436 B scratch) and loses to NLD 2 / 4.
- Split counts above 16 at KV 4096 add combine traffic without filling anything the
  first 16 did not.

## Confirmation (frozen configs, commit `ffc7684`)

Run c4, `.work/dattn-confirm/c4/confirm.md` + `reps.tsv`. Device us per iteration, mean of 10.

| shape | config (frozen) | R us (spread) | O us (spread) | R/O | ideal us | O / R % of roof |
|---|---|---|---|---|---|---|
| S1 16/4/256 | split ns8 NLD8 | 37.29 (0.9 %) | 26.16 (1.0 %) | **1.43** | 17.5 | 67 / 47 |
| S2 40/8/64 | split ns8 NLD4 ROT | 28.24 (0.3 %) | 16.63 (1.5 %) | **1.70** | 8.7 | 52 / 31 |
| S3 28/4/128 | split ns16 NLD2 | 41.32 (0.3 %) | 27.85 (0.6 %) | **1.48** | 8.7 | 31 / 21 |

S1 scaling: KV 512 R 36.91 / O 10.57 (3.49x); KV 1024 R VOID (spread 37.6 %) / O 11.45;
KV 16384 R 104.74 / O 84.75 (1.24x, O at 82 % of its 69.9 us ideal). The exact path
(28.63 / 44.87 us at KV 512 / 1024) never beats split; its KV 4096 row is VOID (21.6 %).

Prediction check: O S1 HELD (24-30, 26.16), O S2 HELD (13-18, 16.63), O S3 MISSED (13-18,
27.85). All three ratio predictions MISSED (1.9-2.3 / 1.9-2.7 / 3.7-5.2 against 1.43 / 1.70 /
1.48), and all three misses come from R being 1.2-1.6x faster than its 200-iteration receipt.

Read-back: 5020 main-kernel dispatches per target on both arms (trace); rotation 402.7 MB
every target (arm lines); O split kernels 176 / 192 / 192 VGPR with 0 / 0 / 316 B scratch
(trace); busy sclk median 3296 MHz, busy power median 157 W, cap 290 W (sampler).

## Harness fixes made on the way

- `bench/ggml-harness/op_bench.c` capped rotation at 64 arms, so R's KV 512 / 1024 rows
  rotated 134 / 268 MB, below 4 x 96 MB. Those rows in stints 1-3 are void. Cap raised to 256 (`5d6dd59`).
- The worktree lacked `.work/engine-pack` and `.work/gguf`; the first confirmation job aborted
  at the engine gate before any timing. Linked from main; `lane-up` fixed in iTools (`c5e9df1`);
  ledger entry in `~/Brain/m.ledger/mojo-baro.md`.
- `tools/kernel-census.py` counts `kernels/*_harness.mojo` as launch sites.
- `bench/dattn-confirm.sh` first ran with `[ test ] && one ...` as the loop body's last
  command; a false test returned 1 and `set -e` killed the job after rep 1's R rows.
  Replaced with `if` blocks (`838d9ee`). The frozen configs did not change.
- **Confirmation run c3 is VOID on spread** (gates all passed; `.work/dattn-confirm/c3`).
  At 200 iterations each target was 25-30 ms of GPU work, shorter than the clock ramp.
  R rows spread 15-37 % and O S1 KV4096 13 %, over the 10 % rule. The reps split into two
  clock states by rep order: R S1 KV4096 read 45-47 us on O-first reps and 52-65 us on
  R-first reps, O moved the same way (26.3 against 28.8-30.0). In the slow reps R's 20
  warmup dispatches averaged 92-171 us against 49-53 us in the fast ones, and the timed
  iterations stayed high too (45-50 against 42-43 us). Fix (`bench/dattn-confirm.sh`,
  committed before c4): 5000 iterations per target on both arms, device time counted from
  the 21st main-kernel dispatch and divided by the main-kernel count, both from the trace.
  c3's numbers are not cited anywhere in this report's verdict.

## Open questions

1. **Wiring.** Nothing calls the new kernel yet. A later round wires `amar_dattn_split` into
   `attn_phases` / the launch path; the megakernel's split-across-blocks shape (fixed
   `NSPLIT` per grid) differs from this standalone grid, so the win must be re-measured there.
2. **Split count needs a rule, not a constant.** The best split count moved with KV length
   (stint 3: 8 at 512-4096, 16 at 16384 for S1). An engine needs `ns(KV, NKVH)` chosen at
   runtime; the frozen configs here are per-shape constants.
3. **HD 64 and 128 sit further from the roof than HD 256** (see the table's ideal column).
   At HD 64 one wave-wide load covers four tokens, so the per-token softmax and shuffle work
   dominates the bytes. A tile that processes several spans per reduction is the lever left.
4. **bf16 and f32 KV are numerically gated but not timed.** The protocol's timed arm is f16
   on both sides; the engine's shipped KV is f32 (the gate-3 instantiation).
5. **Clock state.** rocm-smi reports sclk up to ~3.3 GHz under this load; recorded verbatim
   in `clocks.log`, not cross-checked against another instrument.
6. **`lane-attn`** is still ahead of `main` with the reverted page arithmetic; untouched here.

## Tools and skills built during the lane

- iTool `rocprof-kernels` (Mojo): device time per kernel from a rocprofv3 trace with the
  trace's resource read-back.
- Skill `kernel-arm-round`: sweep, freeze by commit, fail-closed confirmation; void-row
  checks; the traps above.
