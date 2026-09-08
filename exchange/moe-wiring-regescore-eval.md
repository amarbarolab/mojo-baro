# RegesCore-35B writing its own engine wiring — evaluation (2026-09-08)

The model was given 30,077 tokens of context (the MoE spike report, `kernels/moe.mojo`,
`serve/registry.mojo`, `serve/engine.mojo`, `docs/qwen35-ssm-notes.md`) and asked for the
qwen35moe registry block, the host-side MoE call sequence, the extra VRAM, and an honest
list of what it could not determine. Draft: `exchange/moe-wiring-regescore-draft.md`.

**Verdict: same pattern as every previous ornith/regescore build here — a cheap draft with
a load-bearing verify gate. Two genuinely useful catches, one fabricated fact, eight bugs
that would not compile or would silently produce wrong tokens.**

## Serving mechanics

| arm | prompt tok | completion tok | reasoning chars | answer chars | finish |
|---|---|---|---|---|---|
| thinking on, `max_tokens` 12000 | 30,075 | 12,000 | 32,358 | **0** | `length` |
| thinking off, `max_tokens` 10000 | 30,077 | 5,578 | 0 | 15,953 | `stop` |

Prefill 2,007 tok/s, generation 59–81 tok/s, 119 s wall for the useful arm. The first arm
spent its entire budget reasoning and returned an empty string — on-task reasoning, cut off
at the moment it began writing. **`chat_template_kwargs: {"enable_thinking": false}` is the
fix**, and on a task this mechanical the no-think arm lost nothing.

Context ceiling measured separately: the model loads and serves at its full native
**262,144** context in 24.85 GB of 25.75 GB, weights included, with q8_0 KV — only 10 of 40
layers carry full attention, so the KV cache is small enough that context is not the
constraint on this card.

## What it got right, including one thing I would have missed

- **`QF` must stay 8192, not follow `2 * H`.** `registry.mojo` defines `QF = 2 * H`, which
  is 8192 only because the dense 9B has `H = 4096` and its conv dim is also 8192. At
  `H = 2048` the identity breaks: `QF` is the SSM qkv projection width, tied to `CONV`, not
  to `H`. Re-pointing `H` alone would have silently halved that buffer. This is a real,
  non-obvious catch.
- **`NKVH` 4 → 2** — correct.
- Recognised that expert weights live in `wbuf` and need no new allocation, and that the
  MoE scratch is trivial (13.4 KB total). Correct and correctly argued.

## What it got wrong

**Fabricated:** claims `HD = 128 -> 256`. `kernels/attn.mojo` has `comptime HD = 256`
already; there is no change. Stated as confidently as the true items.

**Dangerous:** claims `KV = NKVH * HD` was `4*128 = 512`, is now `2*256 = 512`, "same
value, keep it". Actual 9B value is `4*256 = 1024`. The constant really does halve, and the
draft explicitly instructs you not to touch it.

**Would not compile:**
1. `type_of(moe_probs_d)` etc. — passes `DeviceBuffer` types where `TensorLayout` types are
   required. Every kernel binding in the draft has this.
2. `amar_matmul_skinny_m1_row[4, 1, ...]` — first two params are `in_dtype: DType, UNROLL: Int`;
   it passed two integers.
3. Same call passes three runtime args (`MROWS, N_EXP, H`); the kernel takes `(n, k_dim)`.
4. Passes `wr_d`, a raw `DeviceBuffer`, where a `TileTensor` is required.

**Would run and be wrong:**
5. Uses `amar_moe_sig_gate` to produce per-expert gates for the eight routed experts. That
   kernel is the *shared* expert's sigmoid gate only; routed weights come from
   `amar_moe_router_top8`'s renormalised softmax. Straight semantic error.
6. Launches `amar_moe_router_top8` with `grid_dim = MOE_THREADS` (256 blocks). It must be
   `grid_dim=1`; 256 blocks each writing all of `idx` and `w` is a data race.
7. Launches `amar_moe_down` with `grid_dim=ceildiv(MROWS,1)=1, block_dim=1` — a single
   thread for a kernel that needs one warp per output column (2048 columns → grid 256,
   block 256).
8. Writes both the routed and the shared `amar_moe_down` into the same `moe_out_d`. The
   kernel assigns rather than accumulates, so the second call discards the routed path
   entirely — the model's own report text, in its context, says routed and shared are
   summed.
9. Builds the down-projection weight view from `moe_hidden_d` (the activation buffer) and
   the gate/up weight views from `moe_gate_d`. Wrong tensors passed as weights.

## The "undetermined" list is padded

30 items, and the useful ones are #10 (which `off` table indices hold the MoE weights) and
#6 (whether `wbuf` keeps the router f32 or requantises it). Both are real and both are the
right questions. But at least ten items — expert weight layout, shared-expert layout, router
layout, whether renormalisation happens in the kernel, the activation function, whether the
MoE output is a residual — are answered explicitly in `kernels/moe.mojo` and the report,
which were in its context. Asked to be honest about uncertainty, it inflated the list rather
than reading what it had been given.

## Usable output

The `QF` catch and the VRAM accounting. The call sequence needs rewriting, not patching —
the parity test `kernels/test_moe_block.mojo` already contains a correct, compiling,
verified version of exactly this sequence, and that is what the wiring should be lifted
from.
