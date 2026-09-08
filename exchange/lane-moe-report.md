# qwen35moe sparse-MoE block spike — report (2026-09-08)

Brief: `~/Brain/mojo-baro/briefs/2026-09-08-lane-moe-spike.md`. Scope was set to
**spike only**: prove the MoE block against a numpy oracle and answer whether
top-8-of-256 routing is viable at m=1 on gfx1100. Nothing under `serve/` or in
`kernels/mega.mojo` was touched — the engine still builds the dense 9B.

Commits: `2e050e7` (oracle), `0c31983` (kernels + parity test + census).

## What the model needs, read off the artifacts

`RegesCore-1.0-35B` is `qwen35moe`, not the `qwen35` the engine is built around:
40 layers (30 `ssm_*`, 10 full-attention, every 4th), `H = 2048`, `NKVH = 2`,
`HD = 256`. **Every** layer is MoE — `ffn_gate_exps [2048, 512, 256]` on blk.0
and blk.3 alike, no dense interleave. Experts are Q4_K in the UD-Q4_K_S build;
everything else is Q8_0 bar a Q6_K `output.weight` (117 Q4_K / 251 Q8_0 / 4 Q6_K
/ 361 F32, per gguf-py). The shared expert is separate and
sigmoid-gated through its own `ffn_gate_inp_shexp [2048]` vector.

The tokenizer is already exact: 248,320 tokens / 247,587 merges, sha-identical
to the Ornith-1.5-9B GGUF on this box, and `VOCAB = 248320` is already the
comptime in `serve/registry.mojo`.

## The routing math, and the trap in it

From `Qwen3_5MoeSparseMoeBlock` in the local transformers checkout, not from
memory:

```
probs  = softmax(x @ Wr.T)     over ALL 256 experts, f32, FIRST
w, idx = topk(probs, 8)        THEN top-8
w      = w / w.sum()           THEN renormalise
```

Topk-then-softmax is the more common convention and produces different weights
from identical logits. `tools/moe-ref.py` pins the correct order so a future
implementation cannot quietly drift onto the common one.

## Kernels (`kernels/moe.mojo`)

| kernel | shape | note |
|---|---|---|
| `amar_moe_router_top8` | 1 block, 256 threads | shared-memory max/sum reductions for the softmax, then eight argmax passes with lowest-index-wins ties to match numpy's stable descending sort |
| `amar_moe_sig_gate` | 1 warp | `sigmoid(x · w_shexp_gate)` |
| `amar_moe_gate_up[NSEL, FFN]` | one warp per `(j, r)` | resolves the expert id per warp, indexes `[N_EXP*FFN, H]` at `e*FFN + r`, fuses silu(gate)*up |
| `amar_moe_down[NSEL, FFN]` | one warp per output column | loops the NSEL experts inside the warp and folds the routing weights into one f32 accumulation — no atomics, no partial buffers |

`gate_up` and `down` are parameterised on `NSEL`, so the shared expert is the
same two kernels instantiated at `NSEL=1` with its sigmoid as the weight. That
is the whole shared-expert path — no third code path to keep in sync.

The dense `amar_matmul_skinny_m1_row` is reused unchanged for the router GEMV
(f32 weights, f32 activations, K=2048, N=256). Nothing was rewritten that
already existed.

## Parity — all gates pass

`./.work/test_moe_block` against `tools/moe-ref.py`:

| gate | synthetic weights | **real blk.0 weights** |
|---|---|---|
| expert ids | **exact**, 8/8 | **exact**, 8/8 |
| routing weights | 7.7e-7 | 1.9e-7 |
| routed | 1.9e-5 | 9.6e-8 |
| shared | 8.7e-7 | 1.3e-4 |
| y | 1.9e-5 | 1.3e-4 |

The real-weight column reads blk.0 straight out of
`RegesCore-1.0-35B-UD-Q4_K_S.gguf` and dequantises it (Q4_K experts, Q8_0 shared,
F32 routers) via `tools/moe-ref.py --gguf`. That closes the largest caveat this
report originally carried.

`routed` and `shared` are checked separately so a failure localises.

**One oracle change was needed and it is worth flagging.** The first run failed
`routed` at 7.7e-3 against a 5e-3 gate. That was not an indexing bug: the kernel
stores `h` as bf16 between the two GEMVs, as the engine does everywhere, while
the oracle kept it f32. `ssm-ref.py` already models exactly this rounding on
`gated` before the out GEMV, so the fix was to follow the house precedent rather
than loosen the gate — after which the error fell by 400x to 1.9e-5. A loosened
gate would have passed a genuinely wrong kernel just as happily.

## Does the gather dominate? No — it costs about a third

`BARO_MOE_BENCH=1`, 200 iterations, rotating expert sets. **Feasibility
measurement, not a preregistered perf claim** — `bench/PROTOCOL-RULES.md` governs
champion claims and this is not one.

| arms | working set | µs/token/layer | effective |
|---|---|---|---|
| 8 (disjoint expert sets) | 402 MB | **92.1 – 101.6** | **496 – 547 GB/s** |
| 1 (fixed expert set) | 50 MB | 41.0 | 1228 GB/s |

The one-arm number is **above HBM peak**, which is exactly how you know it is a
cache artifact and not a result: the routed path touches only 50.3 MB of bf16
expert weights per token per layer, and that fits inside the 96 MB Infinity
Cache. Rotating across eight disjoint expert sets walks 402 MB and defeats it.
The rotation is load-bearing; without it this report would have claimed a 1.5x
speedup over the dense kernel that does not exist.

Against that, the repo's own cold-cache dense record is ~832 GB/s
(`amar_matmul_skinny_m1` at 121 µs for the 100.7 MB M=1/K=4096/N=12288 shape).
So **the scattered top-8 gather sustains 66% of dense streaming rate.** The
gather costs about a third; it does not dominate, and there is no structural
reason this shape cannot be served on gfx1100.

Extrapolated: 92.1 µs × 40 layers = 3.7 ms/token = **272 tok/s ceiling from
routed expert traffic alone, at bf16**. Deployment is q4 (~4x fewer bytes), which
moves that ceiling to roughly 1000 tok/s and puts the binding constraint back on
attention and SSM — llama.cpp serves this model at ~100 tok/s today, so the
headroom is real rather than marginal.

## What this spike does NOT establish

- **m=1 only.** No prefill, no m>1 row scaling, so P5 does not apply and has not
  been earned.
- **bf16 only.** The v2 round of the skinny GEMM showed q4 dequant ALU can eat
  the byte win outright. The q4 expert path is unproven and is the single
  biggest risk to the 1000 tok/s figure above.
- **Three launches per layer.** 120 launches/token at 40 layers is precisely the
  overhead `mega.mojo` exists to remove, and MoE routing inside a persistent
  megakernel is the hard part of the full port — untouched here.
- **Nothing is wired.** `serve/registry.mojo` still hardcodes `H = 4096`,
  `N_LAYERS = 32`. Parameterising those is the first task of any follow-on.

## Pre-existing red, not caused here

`tools/kernel-census.py --check` exits 1 on three orphans in
`spark_kernels.mojo` (`amar_attn_decode_swa`, `amar_head_gate_mul_cast`,
`amar_skinny_reduce_gelu_par_bf16`). Verified against a clean stash: identical
before this work. `docs/KERNELS.md` is regenerated and includes the four new
kernels.
