# MOE lane builder conference answer

## 1. Where the plan is wrong

The W2 to W4 direction is sound, but W3 needs a sharper acceptance boundary. A
successful Q4_K kernel test is not evidence that the engine can load the packed
model, and a successful one-block engine test is not evidence that routed and
shared experts are both wired correctly. W3 should therefore have two explicit
gates: first, loader plus one real attention-block parity; second, host m=1
decode wiring plus the 20-prompt parity gate. This does not change the lane
order or add a public step, but it does change the W3 brief's acceptance shape.

W4 should remain a measurement step, not a target-driven optimization step. Its
first number must be a baseline receipt using the same model, prompt set,
context, sampling, quantization, and warmup policy on both engines. Do not turn
the predicted W2 timing range into a pass threshold before it is measured.

## 2. What would make this lane fail

The largest risk is a format boundary being treated as a naming detail. W1
copies raw ggml Q4_K blocks, while the existing `gemm_w` q4 path expects the
engine's own q4 layout. Q4_K has 256-value superblocks with 6-bit scales and
mins, so its byte offsets, superblock traversal, and expert-major row indexing
must be decoded deliberately. The W2 gate must prove one real packed row from
the GGUF bytes through dequantization and accumulation before W3 wiring.

The next likely failure is shared-expert accumulation. The W3 host path must
initialize the destination, run routed down projections, add the shared path,
and preserve the intended gate and residual ordering. The one-block test must
exercise both a routed expert and the shared expert; a routed-only result is
insufficient.

The W2 gate should fail on any expert-ID mismatch, either output error above
`5e-3`, missing raw-block receipt, or a nonzero test exit. W4 should report a
20-prompt median and min-max range, with the competitor configuration read back
in the receipt. A result without those receipts is an instrument observation,
not a lane pass.

## 3. How to shape the items

Keep W2 as one specialized implementation step. Share the routing, expert
indexing, launch geometry, and accumulation structure with the bf16 kernels,
but use a specialized Q4_K row-dot/dequant helper. Keep thin specialized
`gate_up_q4k` and `down_q4k` entry points if that preserves compiler-visible
wave and register behavior. The superblock decoder should not be hidden behind
a broad generic dtype abstraction that forces scalar fallback or obscures the
expert base offset.

W2 should hand W3 three durable artifacts:

- a raw Q4_K block decoder test vector containing source bytes, dimensions,
  expected dequantized values, and expected dot output;
- an exact tensor-to-pack offset map for every routed and shared expert tensor;
- a real `blk.0` or one attention-layer reference receipt with expert IDs and
  output error.

Keep the existing W2 preregistration frozen. Record rotating-cache timing as a
measurement, not as a correctness gate or a hard performance promise.

W3 should consume those artifacts in the two gates above. The first gate should
prove pack lookup and one real block. The second should wire only the smallest
decode path needed by the brief, then run the existing prompt parity gate. Keep
prefill and broader batching out of this lane.

W4 should begin with a baseline table: engine commit, llama.cpp commit/build,
GPU and runtime, model path, context, prompt set, warmups, repetitions, and
sampling parameters. Report median plus min-max for both sides. Optimize only
after that receipt identifies a bottleneck, and attach every later number to a
commit.

## 4. What is missing

W1's raw-copy gate does not yet produce the loader-facing Q4_K offset schema or
a device-load fixture. W2 must produce both. The fixture needs one complete
256-value superblock, including scales and mins, plus a reference dequant/dot
result. The offset schema must distinguish tensor name, expert ID, layer,
projection, row range, and byte offset so W3 cannot infer layout from lexical
tensor order.

The lane also needs one explicit record of the llama.cpp Qwen3-MoE baseline
invocation for W4. Without that, a speed comparison can silently compare
different context or sampling settings.

## LOC-cut decisions

**A: reject.** Requantising the experts into the engine's q4 layout would be a
lossy second quantization, weaken the exact GGUF-to-reference gate, and still
would not make the existing GEMV reusable without solving the expert-major
gather and tensor-offset layout. It moves work into pack conversion and makes
the W3 oracle less trustworthy. The raw Q4_K path is the smaller real change.

**B: accept with a bound.** Reuse the existing routing, gather, and wave-layout
logic, and factor only the weight-load/dequant operation. Do not make the whole
kernel a generic dtype implementation if that compromises Q4_K specialization.
This keeps the LOC cut real while preserving a direct Q4_K parity gate.

This answer changes W3's acceptance shape as described above. Stop here for
conference review before implementing W2 or W3.
