# Lane G1 timing round: protocol and frozen predictions

Date 2026-09-17, branch `lane-g0`. Opened by the coordinator with the maintainer's approval. This file is
committed BEFORE the timed run (`bench/PROTOCOL-RULES.md` P2); results go to
`exchange/lane-G1-timing.md`. Driver: `tools/spirv-probe/time.sh verify|run`.

## Question

Is aihq-lab's Radeon R5 M330 worth G2/G3? Concretely: how fast are the kernels as lowered by
`air2spv.py` on that card, against llama.cpp's Vulkan backend on the same card at the same shapes,
and what tok/s ceiling does that imply for a 0.5B q4 model.

## Arms

| arm | what runs | clock |
|---|---|---|
| ours | the SPIR-V that `run.sh` builds from the Mojo IR, through rusticl (`RUSTICL_ENABLE=radeonsi`), driven by `host.c --time` | host `CLOCK_MONOTONIC` around enqueue + `clFinish` |
| ours, barrier control | rmsnorm with no barrier and no local memory: partial sums kernel, blocking read of 256 partials per row, CPU reduction, scale upload, apply kernel (`amar_rmsnorm_hostreduce`, OpenCL C in `host.c`) | same |
| reference, op level | llama.cpp `ca3d5a3e1` (the workstation checkout, which carries an unrelated local diff in `common/debug.cpp`), `-DGGML_VULKAN=ON`, Release, built on the lab box; `test-backend-ops perf -b Vulkan<radeon>` with `tools/spirv-probe/llama-perf-shapes.patch` (adds our shapes and a per-graph median, touches no kernel) | its own `ggml_time_us` around `ggml_backend_graph_compute` |
| reference, end to end | `llama-bench -ngl 99 -p 0 -n 64 -r 5` on `qwen2.5-0.5b-instruct-q4_0-pure.gguf` (every weight Q4_0, tied embedding, quantized from the repo's bf16 import with `llama-quantize --pure`), Radeon only (`GGML_VK_VISIBLE_DEVICES`) | llama-bench |
| context | the same llama-bench on the box's CPU (i5-6200U, `-ngl 0`) | llama-bench |

## Shapes

- rmsnorm: H = 4096, rows 1, 2, 4, 8, 16, 32, 64 (n = 4096). Reference: `RMS_NORM` F32 `{4096, rows}`.
- softmax: one row of 151936. Reference: `SOFT_MAX` F32 `{151936, 1}`, no mask.
- q4 GEMV (`amar_matmul_skinny_q4rowb[2, 1]` + `amar_skinny_reduce[.., 1]`, m = 1), N rows x K columns.
  Reference: `MUL_MAT` Q4_0 x F32, one token.
  - requested: 1024x4096
  - Qwen3-0.6B (28 layers): q 2048x1024, k/v 1024x1024, o 1024x2048, gate/up 3072x1024, down 1024x3072, head 151936x1024
  - Qwen3-1.7B (28 layers): q 2048x2048, k/v 1024x2048, o 2048x2048, gate/up 6144x2048, down 2048x6144, head 151936x2048
  - Qwen2.5-0.5B (24 layers, the model on the board): true shapes 896x896, 128x896, 4864x896,
    896x4864, 151936x896. **The lowered GEMV cannot run them**: each row's K/32 blocks are dealt to 32
    lanes, so K must be a multiple of 1024 (896 and 4864 are not). Ours is timed at K padded up to
    the next legal width (896x1024, 128x1024, 4864x1024, 896x5120, 151936x1024), which overstates
    our work by 14% (K = 896) and 5% (K = 4864); the reference is timed at both the true and the
    padded shapes. N must be a multiple of 8, which all of these are.

## Method

- Ours: per kernel and shape, 3 warmup dispatches, then 11 single dispatches each followed by
  `clFinish` (`sync_us`, the per-dispatch wall), then 11 batches of B enqueues and one `clFinish`,
  divided by B (`batch_us`; B = 32, or 4 when the sync median exceeds 20 ms). Median with min and
  max. Program build and buffer creation are outside the clock. The GEMV pair is the sum of its two
  kernels.
- Reference op level: the tool's own loop (one warmup graph, then graphs of n copies of the op
  until 1 s has passed). Batched = its default n; per-dispatch = `G0_NRUNS=1` (one op per graph,
  so one submit and one wait per op). The patch prints the median, min and max per op over the
  graphs. The iteration count differs from ours (time-bounded, not 11); it is printed.
- Identity on every timed run: after the timing pass the same process runs the kernel once more on
  fresh inputs and checks it against the fp64 reference (the G0 bars). A FAIL voids that row.
- Read-back (P1), printed by the run itself: device and driver strings, global and local sizes,
  the kernel's max work-group size, every scalar argument read back from the device, every buffer's
  byte size, the reference's backend name, device description and op parameters, llama-bench's
  model and backend columns, sha256 prefixes of the binaries and the model, the GPU power level
  sampled every 0.5 s from `amdgpu_pm_info` during each arm, the CPU governor and load average.
- `time.sh verify` runs every arm at every shape for parity and device selection only, no timing
  flag, reference on a shape outside the timed set. It passed before this file was committed (see
  the commit message for its last line).
- Neither arm rotates buffers. Both hammer one weight buffer per shape. The card has no large
  on-die cache (2 GB DDR3 behind a 64-bit bus), so this is symmetric and not the Infinity Cache
  problem of the 7900 XTX; it is stated here so nobody has to wonder.

## Derived number: tok/s ceiling

Per token, for a model with L layers: `L x (2 x rmsnorm(rows 1) + q + 2 x kv + o + 2 x gate_up +
down) + rmsnorm + head`, using `batch_us` medians, GEMV = q4rowb + reduce. Ceiling = 1e6 / that
sum. It omits attention, rope, swiglu, residual adds, sampling and all host work, so it is an upper
bound on ours. The same sum over the reference's batched op times, divided into llama-bench's
measured tg, says how much of a real decode the formula captures on this card.

## Predictions (frozen)

Basis: 320 shader cores, about 0.4 TFLOPS fp32, DDR3 at no more than 14.4 GB/s. Our GEMV deals one
row to 32 threads and pays 10 barriers plus 10 local-memory round trips per row; radeonsi will
scalarize the 16-wide vectors; rusticl's per-dispatch overhead is unknown.

| # | quantity | prediction |
|---|---|---|
| 1 | GEMV 1024x4096, batch_us, ours | 5 ms (range 2 to 12) |
| 2 | same, reference batched | 0.8 ms (0.3 to 2); ours / reference >= 3 |
| 3 | rmsnorm rows 1: ours sync, ours batch, reference batched | 400 us, 150 us, 60 us |
| 4 | rmsnorm row scaling, ours batch, rows 64 / rows 1 | <= 8x (dispatch-bound at the low end) |
| 5 | barrier control: host-reduction sync_us / lowered sync_us | > 1 at every row count, >= 1.5 at rows 1 (the barriers are cheaper than a PCIe round trip) |
| 6 | softmax 151936: ours batch, reference batched | 3 ms (1 to 8), 1.5 ms |
| 7 | head 151936x1024: ours batch, reference batched | >= 300 ms, about 60 ms |
| 8 | llama-bench tg64, Radeon Vulkan | 15 tok/s (8 to 25) |
| 9 | llama-bench tg64, CPU | 28 tok/s (18 to 40), so the CPU beats the card |
| 10 | our ceiling, Qwen2.5-0.5B at padded shapes | <= 5 tok/s and < 0.5 x prediction 8's measured value |

### Disclosure: three reference numbers were seen before this commit

The prediction table above was written at 10:00:57 local (file sha256 `d2a06434eadfbaa5` before this
section was added) and has not been edited since. The verify pass that followed (10:11, log
`verify-20260917T081105Z.log`) was meant to read no clocks, but the reference patch adds the rmsnorm
and softmax cases whenever `G0_SHAPES` is set, and the 4-token llama-bench smoke prints a rate. So
before the freeze I saw: reference rmsnorm 40.7 us at rows 1 and 281.8 us at rows 64, reference
softmax 303.7 us, and llama-bench tg4 = 29.93 tok/s on the Radeon. The reference halves of
predictions 3 and 6 and prediction 8 are therefore **not preregistered**; they stay in the table
exactly as drafted (all three were already wrong on sight) and will be scored as misses, not
quietly improved. Nothing of ours was timed before this commit, no reference GEMV at a timed shape
was seen, and the CPU arm has not run.

## Decision rule (proposed; the maintainer may overrule it)

The card is worth G2/G3 only if both hold: (a) llama.cpp Vulkan on the card is at least as fast as
llama.cpp on the box's own CPU, so the card adds something on that machine, and (b) our ceiling as
lowered is at least 0.5 x llama.cpp Vulkan's measured tg, or the gap is carried by one construct
for which this round also measured a cheaper lowering. Predicted verdict: **not worth it as
lowered** (predictions 9 and 10).

## Falsifiers

Any prediction above outside its stated range is reported as missed, with the number. A parity
FAIL on a timed row voids the row. A reference run on the wrong device (the box also has an Intel
HD 520 Vulkan device) voids the reference arm; the device description line is the receipt.
