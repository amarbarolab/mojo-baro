# Baseline — what is verified working

Current truth for `mojo-baro`. Every number here was measured on this machine and
is reproducible with the command given. If you are an agent picking up a kernel
task, this is your starting point: **do not re-derive it, and do not trust a
number that is not in this file or produced by `bench/run.py`.**

Last verified: 2026-09-15 for the qwen35moe decode row (`bench/moe-persist-protocol.md`,
main-tree re-gate), 2026-09-12 for the long-context prefill rows
(`docs/prefill-long-ctx-2026-09-11.md`), and 2026-09-02 for the fp16 WMMA rows
(`bench/wmma-fp16-protocol.md` Round 3). Rows not named here were not re-measured
on those dates: check the protocol each one cites before trusting it.

## Hardware and toolchain

| | |
|---|---|
| GPU | AMD RX 7900 XTX, **gfx1100 (RDNA3)** — 24 GB, ~22.3 GB free to MAX |
| Warp size | **32** (RDNA3, *not* 64 like CDNA) |
| LDS | 64 KB/block |
| ROCm | 7.2, hipBLASLt present at `/opt/rocm/lib/libhipblaslt.so` |
| Mojo | **1.0.0** — repo-local venv, `./.venv/bin/mojo` |
| MAX | 26.5.0 |
| Rust | 1.97.1, service layer SHIPPED: `serve/` crate `baro-serve` (axum + tokio), the OpenAI-compatible HTTP front for `serve/engine.mojo`. One engine per card: `BARO_POOL` > 1 does not fit on a single 24 GB GPU, because one engine's MAX runtime reserves about 23.5 to 24.8 GB |

Nothing is installed machine-wide. `uv sync` creates `.venv` from the tracked
`pyproject.toml`, which pins `max[all]==26.5.0` (Mojo 1.0.0) from PyPI. **Not the
nightly index** — an earlier version of this file said nightly, but 26.5.0 is a
stable release and is not carried there, so a nightly pin does not resolve.

## Layers

```
Python/app  →  Mojo (GPU kernels)  →  C++ shim (vendor SDK)  →  hipBLASLt
                      ↑
                    Rust (network/API shell): serve/ crate baro-serve, shipped
```

Rust talks only to Mojo's C-ABI surface. It must **not** bind the C++ shim
directly: two independent owners of one hipBLASLt context is a lifetime bug.

## What runs

| Component | Path | Verify with |
|---|---|---|
| C++ hipBLASLt shim | `shim/` | `./run-tests.sh` |
| Mojo FFI binding | `kernels/amarbaro.mojo` | `./run-tests.sh` |
| GEMM kernels | `kernels/matmul.mojo` | `./bench/run.py` |
| Bench + correctness engine | `bench/run.py` | `./bench/run.py` |
| Parameter sweep | `bench/sweep.py` | `./bench/sweep.py` |

Related work lives in `~/AMDHQ` — this box's AMD/ROCm evaluation lab (`lab` CLI,
`tools/`, `runs/` ledger, rocprofv3 captures). Its 2026-08-27 shortlist already
recorded verdicts worth knowing before reaching for ROCm ecosystem pieces:
**aiter/ATOM have no RDNA3 build path** (gfx942/gfx950 only), **hip-ep's autotune
LUT is gfx1151-only**, and MIGraphX EP is already installed system-wide.

`./run-tests.sh` builds the shim and checks an fp16 GEMM through the C ABI
against a host reference. `./bench/run.py` gates on correctness *before*
reporting throughput and exits non-zero if any variant is wrong.

## Engine (2026-09-04)

The default pack is **q4** (`BARO_PACK`, default `.work/engine-pack-q4`); the
q8 numbers in this section are the q8 arm, kept because they are what the
kernel work below was measured on. The q8 pack is built by
`tools/engine-pack.py MODEL.gguf OUTDIR --q8` and `tools/q8-check.py` proves it
bit-equal to `llama-quantize Q8_0`. On that arm, all 2D
weight GEMMs are `amar_matmul_skinny_q8row`: one wave per weight row over the
weight-native [out, in] int8 layout plus fp16 block-32 scales, 855 GB/s on
the ffn shape (62.6 us per 100 MB-equivalent stream). Decode: **68.8 tok/s_gen**,
64/64 greedy identity with llama.cpp Q8_0 and the bf16 reference; llama.cpp
Q8_0 no-spec bar 74.1. The bf16 pack path (41.7 tok/s) is gone from the
engine; its numbers stay in `bench/q8-protocol.md`.

**MTP speculative decode (`BARO_SPEC=1`; default k=2 via `BARO_SPEC_K` or
`spec-k.txt`).** Draft = `blk.32` head; rows verified in one m=k+1 trunk
window, SSM/conv state in a (k+1)-slot ring so rollback is free.

**Speculation composes with sampling since A1 (2026-09-15, `b3c0d90`,
`bench/spec-sample-protocol.md`).** The window runs the real rule (accept with
min(1, p/q) on the truncated distributions, residual draw on the first
rejection, bonus token from p) instead of refusing to speculate at
temperature > 0. 20-prompt medians, one stint, dense q4, k=2:

| arm | median tok/s_gen | acceptance |
|---|---|---|
| T=0.7 top_p 0.9, spec on | **147.15** | 0.691 |
| T=0.7 top_p 0.9, spec off | 109.19 | |
| T=0, spec on | 150.24 | 0.660 |
| T=0, spec off | 134.97 | |

Two things to read off that table rather than the headline. **Sampling itself
costs 19% of decode at this vocab** (109.19 against 134.97 with no speculation
on either side): the device sampler scans 248,320 logits per draw where the
greedy path does one reduction. And **sampled decoding with speculation is
faster than greedy decoding without it** (147.15 against 134.97). Falsifier:
`bench/spec-sample-ab.sh <engine> OUT 0.7 0.9` on this card; a T=0.7 spec
median below the T=0.7 no-spec median refutes the row.

Headline is the **20-prompt median**, per `bench/PROTOCOL-RULES.md` P4 —
a single-prompt speculative number is an instrument receipt, never a verdict:

| set | ours | llama.cpp Q8_0 | ratio |
|---|---|---|---|
| **20 real prompts (median, k=2)** | **100.7** | **123.5** | **0.78x** |
| 5-token race prompt (k=4) | 145.6 | 109.8 | 1.33x |

So: ahead on the preregistered race prompt, **behind on real text**. Both
engines' speculative output matches their own greedy output on the race
prompt; on the 20-prompt set ours matches on 20/20, llama.cpp's on 16/20.

The earlier headline of **127.96 tok/s_gen (1.89x over 67.77, acceptance
50/53)** was measured on the 5-token race prompt alone and is superseded —
that prompt's repetitive tail inflates acceptance to ~94%. Kept here only so
the number is recognisable when it turns up in older notes.
Protocol, k sweep and bug log: `bench/mtp-protocol.md` Result 2.

Weight-native wave-per-row is the fastest measured stream on this card
(qingming-gfx1100-gemv 917 GB/s fp32; ours 856 bf16 / 855 q8). The earlier
"wt-layout is coalescing-bound" verdict was about a thread-per-column kernel.

## Measured kernel performance

512×512×512, fp32, 200 iterations, 10 warmup. Run-to-run spread <0.5%.

| variant | GFLOP/s (512³) | notes |
|---|---|---|
| `hipblaslt` | ~5200 | **vendor baseline**, tuned (algo × splitK × wgm) |
| `amar_matmul_naive` | ~1250 | one thread per output element |
| `amar_matmul_tiled` | ~2270 | 16×16 shared-memory tiles |
| `amar_matmul_regtile` | ~5130 | BM32 BN32 BK8 TM2 TN2, swept |

**regtile and tuned hipBLASLt are a tie**, trading places by size:

| size | hipBLASLt | regtile | winner |
|---|---|---|---|
| 512³ | 5370 | 5103 | vendor +5.2% |
| 1024³ | 6905 | 6790 | vendor +1.7% |
| 2048³ | 7089 | 7349 | ours +3.7% |
| 4096³ | 6758 | 7236 | ours +7.1% |
| 8192³ | 6263 | 6290 | tie +0.4% |

Both peak around 2048–4096³ and decline after; neither scales past 4096³, which
points at L2/HBM traffic rather than compute. `naive` degrades monotonically
(1129 → 658) as the working set outgrows cache.

Both sustain ~7.1 TFLOPS at large sizes, roughly **23% of the ~30.7 TFLOPS
plain-FMA peak** (~61 TFLOPS is the dual-issue figure).

**Do not repeat the retracted claim.** An earlier version of this file recorded
regtile as ~2× faster than hipBLASLt. That was measuring an *untuned vendor
call*, not a fast kernel. Removing three defects in our own shim — per-call
workspace allocation, trusting the heuristic's ordering, and never setting
splitK/wgm — took hipBLASLt from 2497 to 5201 GFLOP/s at 512³ and erased the
lead entirely. Matching a tuned hipBLASLt is still a good result; it is a much
smaller claim than the one first recorded.

A vendor baseline that looks easy to beat is a bug in your harness until
proven otherwise.

## Hard-won facts — do not relearn these

**Mojo 1.0 API.** `fn`, `alias`, `let`, `inout`, `owned` are all gone; use `def`,
`comptime`, `var`, `mut`. Imports take a `std.` prefix.

**`DLHandle` no longer exists.** FFI goes through `external_call`, and the binary
must be linked against the shared library at build time:
`-Xlinker -L<dir> -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker <dir>`.
`mojo run`'s JIT will **not** resolve external symbols and `LD_PRELOAD` does not
reach it — build AOT.

**`Pointer` is non-null by design** and cannot hold a C handle that may come back
null. Carry opaque handles as `Int`, or use `Optional[Pointer[...]]`.

**Kernel scalar arguments must be fixed-width.** `Int`/`UInt` are not
`DevicePassable` — use `Int32` and convert inside the kernel.

**Accumulators must be SIMD values with comptime-unrolled indices.** Holding them
in `stack_allocation` gives you *scratch memory*, which on AMD lives in device
memory. The first register-tiled kernel did this and ran at 570 GFLOP/s —
**slower than naive**. Same algorithm, 6.4× apart.

**Benchmark at ≥200 iterations.** At 20, launch overhead dominated: both variants
reported roughly half their true throughput with ~10% run-to-run spread.

**CMake: link `hip::host`, not `hip::device`,** for a shim with no device code.
`hip::device` injects `--offload-arch` flags a non-clang host compiler rejects.

**t-strings reject format specifiers** — build JSON with String concatenation.

**hipBLASLt needs three things done right or it benchmarks as garbage.**
(1) Cache the workspace and selected algorithm on the context — a per-call
heuristic plus `hipMalloc`/`hipFree` costs more than the GEMM and read 1373
GFLOP/s, *below our naive kernel*. (2) The heuristic returns candidates in
predicted, not measured, order — time all of them. (3) **`splitK` and `wgm` are
only reachable through `hipblaslt_ext`**, not the C API, and are worth more than
algorithm choice: they took 3097 → 5201 GFLOP/s. Winning `splitK` decays with
size (4 at 512³, 1 at 2048³), consistent with small problems failing to fill the
GPU with workgroups.

**Never let the GPU idle inside a benchmark.** The card sits at ~28 MHz / 31 W
and ramps to ~3000 MHz in **~0.4 s of back-to-back work**, then holds. It drops
again the moment anything interrupts. Consequences, all measured:

- A fixed 10-iteration warmup (~10 ms at 512³) never leaves idle clocks.
- The bias tracks *measurement order*: whichever variant runs first is penalised
  most. Fixing it moved naive +33%, regtile +9%, hipblaslt +4%.
- **Each variant re-warms immediately before it is timed.** One warmup at the
  start is not enough — the host-side correctness check between variants is long
  enough to lose the clocks.
- Do not call `rocm-smi` inside a timing loop. Doing so idled the GPU between
  samples and produced a fake 11k↔27k GFLOP/s oscillation at half true speed;
  power read 31–54 W against a 290 W cap, which is the tell. Sample clocks from
  a separate process.

**MAX claims ~90% of VRAM on `DeviceContext()` creation.** It is a pool, not a
leak: it appears instantly, sits at ~22.7 GB whether the problem is 4096³
(192 MB of matrices) or 8192³ (768 MB), and stops at whatever is free. Cap it
with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` — the bench harness
defaults it to 10.

| cap | VRAM held | regtile GFLOP/s (1024³) |
|---|---|---|
| 100% | 23.06 GB | **segfault in `hipblasLtCreate`** |
| 10% | 2.72 GB | 7703 |
| 2% | 0.88 GB | 7591 |

At 100% there is nothing left for hipBLASLt to allocate its handle and the
process dies. Capping costs no throughput — it measured slightly *faster*.
This is the constraint on co-tenancy: uncapped, one MAX process owns the card.

**fp32 WMMA does not exist on gfx1100 — it is the ISA, not Mojo.** Verified
three ways: `llvm-mc -mcpu=gfx1100` accepts `v_wmma_f32_16x16x16_f16` and
rejects `..._f32`; `BuiltinsAMDGPU.def` has zero f32-input WMMA entries; and
Mojo's own constraint reads *"RDNA WMMA does not support FP32 inputs (only
FP16/BF16 -> FP32)"*. **Any fp16 TFLOPS figure (aiter's ~82–89) is therefore not
comparable to this fp32 benchmark** — they use different hardware inside the
same chip.

**fp16 WMMA kernel — measured 2026-09-02 (`bench/bench_fp16_pipe.mojo`, `kernels/matmul_wmma_pipe.mojo`).**
Pipelined rewrite: 4x2 warps, 2x4 wave tiles (128x128 block, 256 threads), K32,
two LDS buffers, one barrier per K-step, **two-deep global prefetch** (tile k+2
in registers while k+1 waits in the second set), XOR-swizzled A and transposed B,
no pads, default VGPR cap (188 VGPR, 0 spills, 32 KB LDS = 2 blocks per WGP).
Interleaved race medians of 5 (`bench/race-fp16.sh`), vendor = hipBLASLt fp16 via shim:

| size | ours (fp16 C) | hipBLASLt fp16 | old `wmma_lds` | verdict |
|---|---|---|---|---|
| 512³ | 21619 (128x128) / **31841** (32x64 tile, 1-deep) | 27203 | 10948 | needs size dispatch |
| 2048³ | **93841** | 81698 | 56593 | +14.9%, ranges disjoint |
| 4096³ | **97957** | 89716 | 66197 | +9.2%, ranges disjoint |

Size dispatch landed 2026-09-02 (`6a4e38e`): the kernel takes warps/wtile as
parameters; the bench picks the tile from the number of 128x128 blocks the
grid would launch: >= 96 -> 128x128 8-wave, >= 64 -> 64x128 8-wave (2x4/2x2),
else 64x64 4-wave (2x2/2x2). `bench/fp16-templates.sh` (10 s warm-up, one build
per size, receipts in JSON), ours vs hipBLASLt:

| size | 256³ | 512³ | 768³ | 1024³ | 1536³ | 2048³ | 2560³ | 3072³ | 3584³ | 4096³ |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 6372 | 30642 | 64204 | 74824 | 93632 | 91300 | 97974 | 99307 | 105786 | 90705 |
| hipBLASLt | 6288 | 26324 | 54924 | 63623 | 69332 | 80203 | 87147 | 97224 | 85671 | 82437 |
| ratio | 1.01 | 1.16 | 1.17 | 1.18 | 1.35 | 1.14 | 1.12 | 1.02 | 1.24 | 1.10 |

Warm-up matters: at the old 1 s warm-up the same binary read 66k at 4096³
(clocks not settled); benches now warm 10 s and log `warmup_s`.

Clock under WMMA load (`bench/clock-probe.sh`, 2026-09-02, 4096³ 128x128 8-wave,
two runs): sclk holds 2559–2709 MHz, median ~2.6 GHz, at 278–320 W package
power and 72–77 °C junction. Lighter kernels hold ~3000 MHz at the same power
(the 44k kernel of 2026-09-01 read 3005–3008 MHz at 291–307 W), so the matrix
path is power-bound and the peak to measure against is 96 CU × 512 FLOP/clk ×
2.6 GHz ≈ 128 TFLOP/s (512 FLOP/clk/CU = AMD's 122.8 TFLOPS spec ÷ 2.5 GHz
÷ 96 CU), not 123 at spec boost or 147 at 3 GHz. The shipped kernel runs at
70–76% of that at 4096³ (89.5k/91.8k in the probes, 97957 race median) and 82%
at 3584³. That 15–20% is the whole remaining kernel-side headroom and none of
the textbook items claims it: LDS staging, XOR swizzle and edge-branch removal
are already in; fp16 fragments are already register-packed (the 4x4 config at
252 VGPR with 0 spills cannot exist unpacked: 128 acc + 128 bfr + 64 staging
before addressing); a vectorized epilogue is ≤1% at 4096³ (64 scalar stores
per wave against ~8000 main-loop instructions), and the D-fragment layout (a
lane owns rows 2i+half of one column) cannot widen a store without a permute
or an LDS-staged transpose.

What moved it, in order (each measured alone at 4096³): pipelining +5%; a 128x128
tile only once LDS fits two blocks per 64 KB WGP (pads out, swizzles in) +11%;
conflict-free transposed B +3%; dropping edge-bounds branches +6%; two-deep
prefetch +11% (the 1-deep register prefetch of 2026-09-01 was flat because it
had one LDS buffer and two barriers; depth pays only on top of double-buffered
LDS). Transposed B with *padding* was -16% (16-way store conflicts), which is
why the 2026-09-01 "NT loses" verdict was wrong about the cause. Vendor kernel
identity (`TENSILE_DB=0x8000`): `MT96x96x32 WG 4 waves MIWT3_3 PGR2 PLR1 TLDS1`.

**Two compiler facts that gate every kernel here.** (1) LDS is 64 KB per WGP
(`sharedMemPerMultiprocessor`), so a 36 KB block runs alone; 32 KB runs two.
(2) Mojo defaults `max_flat_workgroup_size` to 1024, which caps VGPRs at 192;
`@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](NTHREADS))`
raises it to 256 (receipt in the code-object notes, `tools/isa-receipt.py`).
Per-config, not free: it lets 4x4 wave tiles compile spill-free (91k) but makes
the 8-wave 2x4 kernel 6% slower, and the 98k champion runs with it OFF (`LB=0`).

The 2026-09-01 numbers (66197 at 4096³, "occupancy-bound, five variants lose")
stand as history in the protocol file; the kernel `matmul_wmma_lds.mojo` is kept.
Step-by-step receipts: `bench/wmma-fp16-protocol.md`.

**fp16 WMMA in Mojo works, and the fragment shape is the trap.** RDNA3 wave32
wants **a/b = 16 wide, c/d = 8 wide** (matching
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` typed `V8fV16hV16hV8f`). The
8-wide fragments `TensorCore.load_a` builds are the CDNA/NVIDIA shape and fail
with *"no valid implementation of mma"*. Measured lane mapping:

```
A: lane L, elem i(0..15) -> A[L % 16][i]
B: lane L, elem i(0..15) -> B[i][L % 16]
D: lane L, elem i(0..7)  -> D[2*i + L // 16][L % 16]
```

D is not the obvious layout — consecutive `i` steps two rows and the half-waves
interleave. Verified 16×16×16 tile: 0 mismatches, max_err 5.96e-07. Working
probes in `.work/wmma/`. Also: `get_mma_shape` has **no working RDNA entry** for
any dtype/shape_id — pass `Index(16,16,16)` to `TensorCore` explicitly. And the
whole `TensorCore` surface takes `LayoutTensor` (built via `Layout.row_major`),
which does not unify with the `TileTensor` our kernels use; calling `mma`
directly with hand-built fragments avoids both problems.

**gfx1100 is thinly tuned in hipBLASLt.** 8 fp32 (`SS_SS`) Tensile libraries vs
134 for gfx942; 95 total gfx1100 files vs 1111. The fp32 heuristic offers only
**4** candidate algorithms at any size. This is the contribution opportunity.

**MAX `DeviceBuffer.unsafe_ptr()`** yields a raw device address that the shim's
hipBLASLt calls accept directly — no copy needed to compare against vendor.

## Tuning findings

Small tiles win on RDNA3, contradicting NVIDIA-derived tiling guidance. Every
top-5 swept configuration uses BM=32 or 64; hand-picked BM128/BN128/TM8/TN8
**regressed to 1651 GFLOP/s** (0.45×), almost certainly register pressure
collapsing occupancy. Occupancy is buying more than data reuse here.

Do not trust large-tile intuition on this card. Measure it.

## Rules for kernel work

1. **Correctness first.** A fast wrong kernel is a failure. `bench/run.py`
   checks before it times.
2. **One number, one commit.** Results go to the **AMDHQ experiment ledger**
   (`~/AMDHQ/runs/runs.jsonl` + `runs.sqlite`) under `role_key="mojo-baro-gemm"`,
   tagged with the commit that produced them. That ledger is this box's existing
   record of ROCm experiments — do not start a second one here. `bench/sweep.py`
   keeps its own `sweep.jsonl` because a 198-point parameter search is search
   output, not an experiment record.
3. **Numeric parameter search belongs in `bench/sweep.py`,** not in a human or
   an agent. It is exact, free, and already beat a hand-tuned config by 1.38×.
4. **New strategies go in their own file** (`kernels/matmul_<strategy>.mojo`) and
   register a variant in `bench/bench.mojo`. Never edit another agent's kernel.

### fp16 roofline on this card (2026-09-04, protocol Round 6-7)

WMMA-only microbenchmark under the 290 W cap: **R = 125.4 TFLOP/s**, 436
FLOP/clk/CU at 3.0 GHz (`bench/bench_wmma_peak.mojo`). The pipe kernel at
4096^3 is 89-91k GFLOP/s = 0.71 R at 2.6 GHz (0.82 R per clock);
hipBLASLt is 0.65 R at 2.9 GHz. The gap is 13% clock (energy per FLOP)
and 19% issue. Five levers were raced in Round 7 and none kept; details
and receipts in `bench/wmma-fp16-protocol.md`.

## Megakernel decode path (default since 2026-09-06, `BARO_MEGA=1`)

Decode at m=1 without spec runs as ONE persistent launch per token
(`kernels/mega.mojo::amar_mega_token`, G=96 x 512 threads, bounded grid
barriers): all 32 layers + final norm + head GEMM + argmax. Prefill (m>1) and
the MTP window keep the launch path; `BARO_MEGA=0` restores it everywhere
(parity reference). Gate: `tools/mega-gate.sh` (build, kernel parity test,
run-tests, identity on every runnable pack with spec on/off, ref tokens,
3-run perf). Receipt 2026-09-06: launch 67.13 -> mega 81.98 tok/s_gen
(+22%), spread < 0.5%, all identities equal; `BARO_PROFILE=5` prints the
per-sub-block device profile, `BARO_DUMP=path` dumps X per layer for both
arms. Round receipts: `bench/megakernel-protocol.md`.

**Sampled decode on the megakernel since 2026-09-16 (lane-A6, `54fb4bb`,
`bench/chat-protocol.md` A6): no-spec T=0.7/top_p 0.9 132.30 tok/s_gen
against no-spec T=0 134.65 in the same stint (98.3%), was 108.32.** The 19%
"sampler cost" found by A1 was the megakernel bypass at `temperature > 0`;
the sampler kernel itself costs nothing measurable against the launch path's
argmax. A sampling request now runs `mega_token_*` with `fold_head = 0` and
the launch-path head plus `amar_sample_row` after it; T<=0 is byte-identical
(force-ab 20/20). Penalties, top_logprobs and grammar requests take the same
route since `39f6a4c` (A6.3): penalized T=0 109.85 -> 133.38, top_logprobs
107.45 -> 129.80, identity 20/20 on every arm (`bench/pen-ab.sh`); grammar cost
1.253x -> 1.010x (`bench/grammar-cost.py`), 66/66 valid. Only a plain greedy
request folds the head. The speculative window stays on the launch path:
the q4 window megakernel (`BARO_MEGA_WIN=1`, now correctly instantiated for
the q4 pack) measured 0.761x of it on k=2 spec (115.39 vs 151.68, A6.4), worse
than the q8 record (W2/W3). Report `exchange/2026-09-16-A6-report.md`.

**KV cache paged through a block table since 2026-09-17 (lane-A2 step 1, `09a8cd9`,
`bench/a2-protocol.md`).** 128-token pages; every KV kernel reads the physical page from a device
table (`kvtab_d`, identity for a single request, `BARO_KVTAB=reverse` is the gate arm that proves
the indirection is live). Forced agreement 60/60 at 8k/16k/32k under both tables
(`bench/a2-gate.sh`), decode after 32k 100.7 vs 101.6 unpaged, short context 0.994x, state files
saved in logical page order. `serve/kvpage.mojo` holds the allocator A3 will use.

**q4 m=1 champion since 2026-09-11 (`3824e20`, merge of lane-dattn): 136.37
tok/s_gen no-spec, 20-prompt median** (was 133.9). Post-merge A/B against
pre-merge main (`.work/engine-premerge`, same pack and env): 127.98 -> 136.37
(1.066x), identity 20/20; receipt `.work/postmerge-dattn/ab20.log`, merged
spread 4.4 % from one prompt (p15 130.8), every other prompt 135.9-136.7. Not
an attention gain (the new decode attention runs only above T = 1088): the
megakernel recompile re-rolled the q4 dot loop into a faster schedule,
`isa-loops` dual 114/78/78/53 -> 124/79/79/59, 0 scratch. Any later
`kernels/mega.mojo` edit can re-roll it back; read the fingerprint first.
Long context from the same merge: decode after p8192 112.70 -> 124.48, after
p32768 85.22 -> 101.36 (`bench/dattn-wire-protocol.md`).

**q4 m=1 champion since 2026-09-08 (`9e6feaa`): 133.9 tok/s_gen no-spec,
20-prompt median** (was 130.6). The per-phase rmsnorm+quantise and its grid
barrier are folded into the consuming GEMV's LDS prologue (`stage_rms`, 65
barriers per token gone) and the m=1 q4 kernel runs the chunked delta
(`RELOAD=True`, 239 VGPRs, 0 scratch). Bit-identical: gate 13/13, identity
20/20, q4 vs model-ref 64/64. Per token (`BARO_PROFILE=5`): 7498 -> 7358 us;
delta phase 464 -> 269. Known pool left by this round: the shared q4 dot loop
lost dual-issue pairs (102 -> 74) and gained `s_delay_alu` (184 -> 454), ffn
gate/up +180 us; read `isa-loops` (delay/dual columns) on every megakernel
build -- a loop that changes there with unchanged source is the register
allocator, and only the real-pack A/B settles it. `rocdl.waves_per_eu`
cannot be set through `@__llvm_metadata` (six spellings rejected).


## qwen35moe decode

**Champion (2026-09-15, R6.0b `38ee0b7`, `bench/moe-persist-protocol.md`):
111.89 tok/s_gen, 20-prompt median** (spread 1.2%), launch path
(`BARO_MEGA=0`), no spec, 727 launches per token. Up from 107.28 (R6.0
`1e270b5`, 857 launches) and 94.79 (R4 `7145b71`, 1117 launches), all
bit-identical to the previous engine on 20/20 prompts, fail word 0. R6.0 and
R6.0b fold per-layer elementwise chains into fewer launches; R6.0b landed at
1.043x, below the preregistered +5% kill line, on the maintainer's recorded override.

**Persistent MoE token kernel (2026-09-16, R6, `kernels/mega_moe.mojo`,
opt-in):** `BARO_MEGA=1` on the MoE profile runs the 40 layers in one launch
(8 launches per token with the head), bit-exact with the launch path (dump
compare 64 tokens x 80 slots identical, teacher-forced 64/64 on 20/20 vs
`38ee0b7`). 20-prompt A/B 110.90 -> 113.94 tok/s_gen, ratio 1.027, below the
+5% line, so the default stays `BARO_MEGA=0` and the champion above stands.
Refuses `BARO_TIER` and `BARO_EXPERTS`. Receipt that it ran: the engine's
`mega barrier gen` line, 470 per token. Details and the remaining pools in
`bench/moe-persist-protocol.md` R6.2 and `exchange/lane-R6-report.md`.
llama.cpp on the same GGUF: **109.92** (20-prompt median, 109.49 to 110.01).
The same-stint ratio is only for R6.0 (106.95, 0.973x); 111.89 was measured in
a different stint, so the 1.018x it implies is not a same-stint receipt.
Receipts: `.work/moe-perf/lc-r60b-main.log`, `ab-r60b-main.log`.

R6.1 (`f900bbb`, 2026-09-16): a one-launch-per-token MoE kernel
(`kernels/mega_moe.mojo`, `BARO_MEGA=1`, default off) is at parity (per-layer
dump identical, forced identity 20/20 at 64/64 against `38ee0b7`). Its speed
is not measured yet (R6.2, `exchange/lane-R6-report.md`).

**History, round 1 (2026-09-15, `bench/moe-perf-protocol.md`).** RegesCore-35B, pack `.work/moe-w1/pack` (Q4_K experts, Q8_0 projections, Q6_K
head), launch path (`BARO_MEGA=0`), no spec. **93.46 tok/s_gen, 20-prompt
median** (was 42.88), three landed arms in one session: vectorized Q4_K
expert dot (`8130f65`), vectorized Q8_0 row dot (`d49bfc3`), one-wave router
top-8 (R3). llama.cpp 109.4 on the same GGUF. Teacher-forced agreement
53.15/64 mean (band 52.70..53.70 around the W3 bar). Every MoE weight path
used to dequantize one element per lane per load; the bytes-per-token floor
is about 3.2 ms (2.7 GB), so the remaining gap is launch-bound small kernels
in the SSM sub-block and expert time, not bytes.

**Served path (2026-09-15, `bench/moe-served-protocol.md`).** The same binary
behind `baro-serve`, one `POST /v1/completions` per prompt, prompts as token
ids, 64 tokens: **92.89 tok/s_gen 20-prompt median (range 90.56 to 93.07),
0.993x the one-shot 93.52 (range 86.97 to 94.11) measured in the same stint,
and 74.74 tok/s over the whole HTTP round trip** (range 49.81 to 83.92).
Served tokens are identical to the one-shot tokens on 20/20 prompts. Per
request the medians are prefill 159.3 ms, decode 0.678 s, wall 0.843 s, so
serving costs 5.5 ms (HTTP, JSON, the line protocol, queueing) and the rest
of the gap to the decode-only number is prefill. Quote 92.89 against another
engine's decode rate and 74.74 against another server's end-to-end latency;
they are not the same measurement. Falsifier, runnable on any RDNA3 card with
this pack: `bench/served-prompts.sh .work/moe-served/engine .work/moe-w1/pack
OUT "BARO_SPEC=0 BARO_MEGA=0"`; a served/one-shot ratio outside 0.98 to 1.02,
or any identity failure, refutes this row.

## Self-describing bakes 2026-09-15 (`8184f7d`, closure PASS from the file, nothing external)

- `Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-8184f7d.gguf` sha256
  `8a0d20fba26b34b0a...` (`.work/bake/shas.txt`): `tools/gguf-verify.sh` PASS
  64/64, rebuilt engine 118.9 tok/s_gen on p09 (spec on, one prompt).
- `RegesCore-1.0-35B-UD-Q4_K_S-BARO-8184f7d.gguf` sha256 `a3d719282d1c5aa89...`:
  PASS 64/64, rebuilt engine 93.7 tok/s_gen on p09.
- Both carry `baro.hw.*` (card, driver, ROCm, power cap, 20-prompt median,
  config, protocol), `baro.kernel.model`, and the vendored `latentos/`
  package in `baro.kernel.src.latentos/*`, so the closure rebuilds the
  harness with no path outside the file (`8184f7d` vendored LatentOS; the
  `aa3f147` bakes needed `~/AMDHQ/src` and are superseded).
  `tools/gguf-verify.sh MODEL.gguf` is the contributor's one-command check
  (`docs/amd-family.md`).

## Quality vs llama.cpp (2026-09-16, `bench/quality-protocol.md`)

Same GGUF both arms. Perplexity: WikiText-2 test, 8 x 512-token chunks, llama-perplexity's second-half
scoring window (2,040 tokens), one token stream from llama.cpp's tokenizer. Task: `bench/data/e8_tasks.json`
(100 GSM8K math + 20 JSON), thinking off, T=0, identical token ids to both arms, exact match. Ours = the
engine and pack `tools/baro serve` caches.

| model | engine | PPL ours / llama.cpp (ratio) | task ours / llama.cpp | delta | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 22/120 / 24/120 | -1.7 pp | task PASS |
| lily-cybersecurity-7b-v0.2-Q6_K | spark | pending (spark logprobs landed after this sweep) | 13/120 / 15/120 | -1.7 pp | task PASS |
| Qwen2.5-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 46/120 / 46/120 | +0.0 pp | task PASS |
| Qwen2.5-Coder-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 21/120 / 20/120 | +0.8 pp | task PASS |
| Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16 | dense | 9.301 / 8.410 (1.106) | 70/120 / 80/120 | -8.3 pp | PASS |
| RegesCore-1.0-35B-UD-Q4_K_S | moe | 6.233 / 6.242 (0.999) | 11/120 / 12/120 | -0.8 pp | PASS |

- The Qwythos row compares our q4 pack (`baro.run.pack.flags = --q4`, what the 136 tok/s champion runs)
  against llama.cpp on the BF16 file: the 1.106 PPL ratio and -8.3 pp are mostly 4-bit quantization, not
  engine parity (the q4-both-sides rows land at 0.999 and within 2 pp).
- Not run this round: Ornith-1.5-9B, Qwythos-9B-v2, Spark-X2.5-4B (cut for time), Granite-4.2-3B (harness
  script edited mid-run, row void). Spark-family perplexity waits on a rerun now that `serve/spark.mojo`
  has logprobs.
