# Web research: llama.cpp Q4_K/Q6_K/Q8_0 GEMV, RDNA3 memory, Gumbel sampling

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-web-research.md`. All retrieval
via firecrawl (`firecrawl_search` / `firecrawl_scrape`, `onlyMainContent`).
Retrieval date for every source below: 2026-09-15.

## Q1: llama.cpp Q4_K / Q6_K / Q8_0 GEMV dot layouts

Source: `ggml/src/ggml-cuda/vecdotq.cuh`, `ggml/src/ggml-cuda/mmvq.cu`,
`ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`
(https://github.com/ggml-org/llama.cpp, `master` branch, retrieved
2026-09-15). The HIP/ROCm build compiles this same `.cu` source through
HIPIFY; no separate AMD path for the dot-product math.

### 1. Q4_K super-block layout (256 weights, 144 bytes)

`ggml-common.h`:

```c
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };
    uint8_t scales[K_SCALE_SIZE]; // 12 bytes, scales+mins packed 6-bit
    uint8_t qs[QK_K/2];           // 128 bytes, 4-bit quants
} block_q4_K;                    // 2+2+12+128 = 144 bytes total
```

Byte offsets: `d` at 0-1, `dmin` at 2-3, `scales[12]` at 4-15, `qs[128]` at
16-143.

Scale/min unpack, `ggml-quants.c:880` (`get_scale_min_k4`, used identically on
the CUDA/HIP side via the same bit layout):

```c
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

Sub-blocks 0-3 read `scales[j]` and `scales[j+4]` directly as 6-bit values
(top 2 bits unused). Sub-blocks 4-7 assemble a 6-bit value from the low
nibble of `scales[j+4]` plus the top 2 bits of `scales[j-4]` (for `d`) or
`scales[j-0]` (for `m`) shifted into bits 4-5: the top-2-bit halves that
sub-blocks 0-3 left unused get reused as the high bits for 4-7. This is how
12 bytes encode 8 pairs of 6-bit (scale, min) values (8*2*6 = 96 bits = 12
bytes exactly).

### 2. `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1` / `vec_dot_q8_0_q8_1`

All three live in `vecdotq.cuh`. Each MMVQ (matrix-vector) thread handles
`vdr` quant groups per call:

- `VDR_Q4_K_Q8_1_MMVQ = 2`, `VDR_Q6_K_Q8_1_MMVQ = 1`, `VDR_Q8_0_Q8_1_MMVQ = 2`.

**Q4_K** (`vec_dot_q4_K_q8_1`, line 918): loads two 32-bit words per thread
(`v[0]`, `v[1]`) via `get_int_b1`-style 4-byte reads from `bq4_K->qs` at a
`bq8_offset`-derived index: plain 32-bit (dword) loads of packed 4-bit
nibbles, not per-element byte loads. The impl (`vec_dot_q4_K_q8_1_impl_vmmq`,
line 508) then does, per `QR4_K=2` sub-iteration:

```c
const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;   // low/high nibble split, 4 lanes at once
const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
const int dot1 = dp4a(v1i, u[2*i+1], dp4a(v0i, u[2*i+0], 0));  // int8 SIMD dot vs q8_1 activation
const int dot2 = dp4a(0x01010101, u[2*i+1], dp4a(0x01010101, u[2*i+0], 0)); // sum of activation for the min term
sumf_d += d8[i] * (dot1 * sc[i]);
sumf_m += d8[i] * (dot2 * m[i]);
return dm4.x*sumf_d - dm4.y*sumf_m;   // scale*sum - min*activation_sum, per super-block
```

`ggml_cuda_dp4a` maps to `V_DOT4_I32_IU8` on RDNA (see Q3): a single
instruction does 4 signed-int8 multiply-adds. The min term is subtracted
once per block using the *sum* of q8_1 activation values, not re-multiplied
per weight: this is the standard k-quant "affine dequant folded into the
dot" trick: `x_i = d*q_i - dmin*m` is expanded so the `dmin*m` part factors
out of the dot product entirely.

**Q6_K** (`vec_dot_q6_K_q8_1_impl_mmvq`, line 627): no min term (Q6_K is
scale-only, no zero-point), 4-bit low + 2-bit high split reconstructed as
`vi = (vil | vih) - 32` (signed 6-bit centered range), then one `dp4a` per
`QR6_K=2` sub-block against a per-sub-block int8 `scales[4*i]`.

**Q8_0** (`vec_dot_q8_0_q8_1_impl`, line 246): simplest case, no k-quant
scale packing: just `sumi = dp4a(v[i], u[i], sumi)` for `vdr=2` groups, then
`d8_0 * d8_1 * sumi` (both weight and activation are plain per-32-block
scaled int8, no min).

Activation (`u[]`) values throughout come from `q8_1` blocks that were
already quantized before the kernel launched. See Q2.

### 3. `mul_mat_vec_q` launch shape and RDNA tuning

`mmvq.cu:585` (`mul_mat_vec_q`) and the parameter tables above it
(`calc_nwarps`, `calc_rows_per_block`, lines 436-581).

- **Warp width on gfx11**: `ggml_cuda_get_physical_warp_size()`
  (`common.cuh:390`) returns **32** unless `GGML_USE_HIP && (__GFX9__ ||
  __GFX8__)`, in which case it returns 64. gfx1100 (RDNA3/Navi 31) is
  neither GFX8 nor GFX9, so llama.cpp treats it as **wave32**, matching AMD's
  own guidance that RDNA (as opposed to GCN/CDNA) executes wave32
  (GPUOpen RDNA Performance Guide, retrieved 2026-09-15,
  https://gpuopen.com/learn/rdna-performance-guide/: "RDNA runs shader
  threads in groups of 32 known as wave32").

- **`rows_per_cuda_block`** (`calc_rows_per_block`, line 563): for the
  `MMVQ_PARAMETERS_RDNA3_0` table (RDNA3), any table not in
  `{GENERIC, GCN, TURING, GB10}` falls through to `return 1`: RDNA3 always
  computes **one output row per CUDA block** in the decode (`ncols_dst==1`)
  path, unlike some other tables which can share a block across `small_k`
  rows.

- **`nwarps` per type on RDNA3_0** (`calc_nwarps`, line 491, comment: "RDNA3
  (W7900): stricter whitelist than RDNA4. Q2_K / Q5_K / IQ4_XS regress in
  full quant sweeps"): for `ncols_dst==1` (single-token decode),
  - `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` -> **8 warps** (256 threads/block)
  - `Q6_K` -> **2 warps**
  - `IQ4_NL` -> 8 warps
  - **everything else, including Q4_K and Q5_K, falls to the `default: return
    1` branch -> 1 warp (32 threads) per block.**

  This is the one finding directly relevant to a from-scratch RDNA3 GEMV
  kernel: upstream's own tuning table treats Q4_K decode as *not* benefiting
  from the wide (8-warp) launch shape on RDNA3. RDNA4's table (line 467)
  explicitly whitelists Q4_K/Q5_K/Q6_K for 8 warps with a comment that
  "types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
  pressure and lookup table contention at higher thread counts", implying
  Q4_K's `get_scale_min_k4` unpacking + two-level (scale, min) dequant is
  the kind of "complex vec_dot" that regresses at high occupancy on RDNA3
  but was fixed/improved by RDNA4's launch-bounds tuning. If the m=1 decode
  bottleneck is Q4_K/Q6_K-shaped weights, matching upstream's RDNA3 choice
  of low warp count (1-2 warps) rather than assuming "more warps = faster"
  is the load-bearing number here, not a byte-load pattern.

- `K` loop stride: `blocks_per_iter = vdr * nwarps * warp_size / qi` (line
  610): each iteration of the K-loop advances by this many quant blocks;
  no explicit `GGML_CUDA_MMV_Y` / `MMVQ_NWARPS` macros exist anymore in the
  current source (those look like an older llama.cpp naming this brief
  inherited). The current mechanism is the `calc_nwarps`/`calc_rows_per_block`
  constexpr tables keyed by `mmvq_parameter_table_id` (`GENERIC`, `GCN`,
  `RDNA2`, `RDNA3_0`, `RDNA4`, `TURING`, `GB10`, ...), selected per-arch at
  compile time (`get_device_table_id()`, line ~107) or at runtime via
  `GGML_CUDA_CC_IS_RDNA3(cc)` etc. (line 125-132).

### 4. MoE expert GEMV dispatch (`mul_mat_vec_q_moe`, `mmvq.cu:837`)

llama.cpp has a **dedicated MoE kernel**, not a per-expert relaunch of the
plain `mul_mat_vec_q`. Key structure:

```c
const uint32_t token_idx  = threadIdx.y;              // one thread-row per token in the batch
const int      row0       = c_rows_per_block*blockIdx.x;
const uint32_t channel_dst = blockIdx.y;               // one CUDA block column per (expert-slot) channel
...
const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride]; // expert id looked up per (channel, token)
```

So it is **one launch over the whole batch**, gathering rows: `ids_ptr`
holds, per destination channel and per token, which expert's weight rows
(`channel_x`) to read: the kernel indexes into the weight tensor with that
gathered id rather than the caller launching once per expert or doing a
separate host-side gather pass. `blockIdx.y` iterates destination channels
(routed-expert slots), `threadIdx.y` iterates tokens sharing that slot. No
RDNA-specific notes were found in this kernel beyond the same
`ggml_cuda_get_physical_warp_size()` / `get_mmvq_mmid_max_batch_for_device<type>()`
macros used for the plain path (`mmvq.cu:416-433`, arch-gated by
`RDNA4`/`RDNA3`/`RDNA2`/`RDNA1`/`CDNA`/`GCN` preprocessor branches).

## Q2: ggml quantized-activation trick

Source: `ggml/src/ggml-cuda/mmvq.cu:1484-1490` (the `ggml_cuda_op_mul_mat_vec_q`
host-side launcher), `ggml-common.h:255-269` (block_q8_1 layout), retrieved
2026-09-15.

Yes: llama.cpp quantizes the decode activation vector to `q8_1` **every time**
before a quantized GEMV, via `quantize_row_q8_1_cuda(src1_d, ..., src0->type,
...)` called unconditionally at the top of the op (`mmvq.cu:1490`), padded
to `MATRIX_ROW_PADDING` first. `block_q8_1` is `{ half2 ds; int8_t
qs[QK8_1] }`: a per-32-element block holding one packed `(scale, sum)`
half2 plus 32 signed int8 values. This is what lets every `vec_dot_*_q8_1`
function above do a pure integer `dp4a` (4x int8 dot) against the weight's
own quant format instead of converting weights up to fp16/bf16 first. Both
operands are integer, so the GPU's int8 dot-product throughput (2x-4x the
fp32 FMA rate on both NVIDIA and RDNA, see Q3) is usable regardless of the
weight's specific k-quant packing. The precision cost is exactly one int8
quantization step (plus the per-block scale) applied to the activation on
top of whatever the weight's own quant format already lost, noticeably
coarser than keeping the activation in bf16/fp16, but this is llama.cpp's
default and only decode GEMV path for every k-quant and legacy quant type;
there is no bf16-activation alternative in this code path.

## Q3: RDNA3 (gfx1100 / Navi 31 / RX 7900 XTX) memory facts for a GEMV kernel

Primary sources: AMD "RDNA3" Instruction Set Architecture Reference Guide
(70650, GPUOpen/AMD docs portal, PDF blocked by the fetcher's anti-bot check
on 2026-09-15, not independently re-verified here) and AMD RDNA Performance
Guide (https://gpuopen.com/learn/rdna-performance-guide/, retrieved
2026-09-15, confirms wave32 execution on RDNA quoted above). Numeric specs
below are from a third-party consolidated hardware/ISA reference
(https://zolotukhin.ai/zinc/docs/amd-gpu-reference/, "AMD RDNA3/RDNA4 GPU
Reference for Inference", last updated 2026-05-18 per the page, retrieved
2026-09-15. This is not an AMD-authored page; it states it is "consolidated
from AMD product pages, ROCm hardware tables, AMD ISA manuals, GPUOpen
documentation, and profiling data" and cross-checks wave/cache numbers
against ROCm's hardware tables). Treat the specific cycle-count "worked
examples" on that page as the author's own benchmarking, not an AMD
citation; the specs table and cache-line sizes are the useful part here.

- **RX 7900 XTX (Navi 31, gfx1100)**: 96 CUs, 960 GB/s VRAM bandwidth (384-bit
  GDDR6), **96 MB Infinity Cache**, **6 MB L2**. (Matches the 96 MB IC figure
  already in this repo's `docs/BASELINE.md` / coldcache-protocol notes.)
- **Cache line sizes**: L0 vector cache 64 B/line, **L2 128 B/line**,
  Infinity Cache 64 B/line.
- **Per-lane load widths**: `GLOBAL_LOAD_DWORDX4` loads 16 B (128 bits) per
  lane, the widest single-instruction global load. A wave32 issuing
  dwordx4 from consecutive lanes moves 32*16 B = 512 B per instruction,
  i.e. exactly 4 full L2 cache lines (128 B) or 8 full L0/Infinity-Cache
  lines (64 B) when addresses are contiguous across lanes. Full-bandwidth
  coalescing needs the 16 B *stride per lane* (contiguous dwordx4 chunks),
  not a 64 B stride; 64 B is the resulting *cache-line granularity*, not
  the required lane stride. For plain 4 B (`dword`) loads specifically,
  the same source states: "A wave32 reading 32 consecutive 4-byte values
  (128 bytes) generates 2 cache line requests (64 bytes each)": i.e.
  narrower loads still coalesce fully as long as lanes are contiguous, they
  just span more discrete 64 B lines per instruction relative to the bytes
  moved.
- **dp4a / int8 dot**: RDNA3 has `V_DOT4_I32_IU8` (4-element int8 dot to
  int32) at the same throughput class as `V_FMA_F32` per the same source's
  instruction-throughput table (1/clk on wave32): this is the instruction
  `ggml_cuda_dp4a` compiles to, explaining why the q8_1-activation trick in
  Q2 is worth it on this hardware specifically.

## Q4: Gumbel-max sampling with limited-precision uniforms

Sources: JAX docs, `jax.random.gumbel`
(https://docs.jax.dev/en/latest/_autosummary/jax.random.gumbel.html,
retrieved 2026-09-15); vLLM source,
`vllm/v1/worker/gpu/sample/gumbel.py` via
https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/sample/gumbel/
(retrieved 2026-09-15).

**The precision problem is real and JAX quantifies it directly.** From the
JAX docs for `mode`: "optional, 'high' or 'low' for how many bits to use
when sampling... When drawing float32 samples, with mode='low' the uniform
resolution is such that the largest possible gumbel logit is ~16; with
mode='high' this is increased to ~32, at approximately double the
computational cost." A float32 uniform built from the naive
"random-24-bit-mantissa in [0,1)" construction has minimum nonzero value
~2^-24, so `-log(u)` saturates around `24*ln(2) ≈ 16.6`, matching JAX's
"~16" low-mode figure exactly. For a 248320-token vocab, the top-1 order
statistic of that many Gumbel draws needs roughly `ln(248320) ≈ 12.4` nats
of *dynamic range* to be distinguishable from the runner-up in the tail;
16.6 nats of headroom is not a large margin, and any additional precision
loss (e.g. computing `u` from a 24-bit float32 mantissa when 32+ bits of
entropy were available) eats directly into that margin: this is the
mechanism, not just a magnitude coincidence.

**Where the precision actually gets lost, and vLLM's fix.** The standard
formula `g = -log(-log(u))` loses precision differently depending on
whether `u` is near 0 or near 1: for high-probability/winning draws, `u`
tends to land close to 1, and computing `log(u)` there needs `log1p(u-1)`
(or `log1p(-(1-u))`) to avoid catastrophic cancellation from `1-u` directly.
vLLM's Triton kernel (`gumbel_noised_argmax`, `vllm/v1/worker/gpu/sample/
gumbel.py`) does exactly this, with the comment inline:

```python
u = tl_rand32(gumbel_seed, keys, includes_zero=False)
# log1p keeps the winning tail at u -> 0, where fp32 resolves it.
gumbel_noise = -tl.log(-tldevice.log1p(-u))
```

(their `u` convention is flipped relative to the paragraph above: comment
says the winning tail is at `u -> 0`, handled by `log1p(-u)` instead of
`log(u)`; the mechanism is the same cancellation-avoidance regardless of
which end the tail sits on). vLLM defaults to computing this whole thing in
**fp32** (`USE_FP64: tl.constexpr` is an opt-in path) with an explicit note
that fp64 is "~1/32-1/64x the throughput on H100/Ada/Blackwell and
empirically indistinguishable for Gumbel-max" once the `log1p` rewrite is
in place: the fix is the formula, not the bit width. For mojo-baro's
sampler, the direct, minimal-effort fix based on this evidence is: keep the
uniform draw at whatever bit width is convenient, but always compute the
Gumbel key via `-log(-log1p(-u))` (or `-log1p(-u)` composed correctly for
your `u` convention) rather than naive `-log(-log(u))`, before reaching for
more RNG bits or a 53-bit double.

**Exponential-race alternative.** `-log(u)/p` (Gumbel-Top-k / exponential
race, used in some sampling code as an alternative to explicit Gumbel
noise) has the identical precision profile: it is driven by the same
`-log(u)` term, so the same `log1p` correction applies if `u` is drawn
close to 1. PyTorch's `torch.multinomial`/categorical sampling paths and
`gumbel_softmax` (`torch.nn.functional.gumbel_softmax`) use the textbook
`-log(-log(u))` with `u ~ Uniform(0,1)` sampled from the standard fp32 RNG
(no special-cased `log1p` rewrite as of the version surfaced in search); no
separate confirmation of PyTorch's exact uniform bit-width was retrieved in
this pass, flagging as unconfirmed rather than asserting a number.

## Q5: llama.cpp top-p implementation detail

Sources: `src/llama-sampler.cpp:1549-1602` (`llama_sampler_top_p_apply`) and
`common/common.h:261-271` (`common_params_sampling::samplers` default order),
`common/sampling.cpp` (chain construction order),
https://github.com/ggml-org/llama.cpp, retrieved 2026-09-15. Note: the
function is `llama_sampler_top_p_apply` in current source, not
`llama_sampler_top_p_impl` as named in the brief. The file is
`src/llama-sampler.cpp`, not `llama-sampling.cpp`.

**The cutoff is exactly "smallest prefix with cumulative probability >= p",
no rounding**, quoted in full:

```c
static void llama_sampler_top_p_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_top_p *) smpl->ctx;
    if (ctx->p >= 1.0f) { return; }
    llama_sampler_softmax_impl(cur_p, false);   // recompute softmax over cur_p as it stands NOW in the chain
    ...
    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += pdata[i].p;
        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        if (cum_sum >= ctx->p && i + 1 >= ctx->min_keep) {
            last_idx = i + 1;
            break;
        }
        ...
    }
    cur_p->size = last_idx;
}
```

No rounding/epsilon anywhere in the comparison: a token is included the
instant the running sum reaches `>= p`, subject only to `min_keep`.

**Which distribution it sees**: `llama_sampler_softmax_impl(cur_p, false)`
recomputes softmax over whatever logits are current in `cur_p` at the point
top_p executes in the chain: it is not hardcoded to raw or temp-scaled,
it depends entirely on chain order. The **default** chain order,
`common/common.h:261-270`:

```
PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE
```

confirmed by `common/sampling.cpp`'s `common_sampler_init`, which iterates
`params.samplers` in this order and pushes each corresponding
`llama_sampler_init_*` onto the chain (temperature's
`llama_sampler_init_temp_ext` is last before the final `dist` sampler is
appended). **So by default, top_p runs on the raw (pre-temperature)
softmax: temperature is applied after top_p, not before.** This only holds
for the default `--samplers` order; llama.cpp lets a user reorder or drop
samplers via `common_sampler_types_from_names`/`_from_chars`, so a
differently-configured chain could apply top_p post-temperature.
