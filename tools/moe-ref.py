#!/usr/bin/env python3
"""Numpy reference for one decode token through the qwen35moe sparse-MoE block.

Implements transformers `Qwen3_5MoeSparseMoeBlock` exactly:
  router_logits = x @ Wr.T                       Wr  [N_EXP, H], f32 in the GGUF
  probs         = softmax(router_logits, f32)    softmax over ALL experts first
  w, idx        = topk(probs, TOPK)              then take top-8
  w            /= w.sum()                        then renormalise the 8
  expert e      : silu(x @ Wg[e].T) * (x @ Wu[e].T) @ Wd[e].T, scaled by w
  shared        : silu(x @ Wgs.T) * (x @ Wus.T) @ Wds.T, scaled by sigmoid(x @ wsg)
  out           = sum(experts) + gated shared

Softmax-before-topk is the part worth getting right: topk-then-softmax is the
more common convention and gives different weights for the same logits.

Unlike tools/ssm-ref.py this invents fixed-seed weights instead of reading
blk.0 from .work/gguf/ -- the 35B bf16 GGUF (69 GB) is not on this box, and a
block-parity test only needs the math to agree, not these particular numbers.
Weights are rounded to bf16 so the GPU consumes exactly what numpy did.

Writes inputs + expected outputs for kernels/test_moe_block.mojo.
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / ".work/gguf"

H = 2048          # qwen35moe hidden size
N_EXP = 256       # num_experts
TOPK = 8          # num_experts_per_tok
E_FFN = 512       # moe_intermediate_size
SH_FFN = 512      # shared_expert_intermediate_size
SEED = 20260908


def to_bf16(x):
    """Round-to-nearest-even f32 -> bf16, returned as f32 (the pack's rounding)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def bf16_store(x):
    """The same value as the upper 16 bits, for writing a bf16 fixture."""
    return (np.ascontiguousarray(to_bf16(x), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    x = to_bf16(rng.standard_normal(H, dtype=np.float32) * 0.5)

    # Router and the shared-expert gate stay f32: that is how the GGUF stores
    # ffn_gate_inp.weight and ffn_gate_inp_shexp.weight.
    wr = (rng.standard_normal((N_EXP, H)) * 0.02).astype(np.float32)
    wsg = (rng.standard_normal(H) * 0.02).astype(np.float32)

    wg = to_bf16(rng.standard_normal((N_EXP, E_FFN, H), dtype=np.float32) * 0.02)
    wu = to_bf16(rng.standard_normal((N_EXP, E_FFN, H), dtype=np.float32) * 0.02)
    wd = to_bf16(rng.standard_normal((N_EXP, H, E_FFN), dtype=np.float32) * 0.02)

    wgs = to_bf16(rng.standard_normal((SH_FFN, H), dtype=np.float32) * 0.02)
    wus = to_bf16(rng.standard_normal((SH_FFN, H), dtype=np.float32) * 0.02)
    wds = to_bf16(rng.standard_normal((H, SH_FFN), dtype=np.float32) * 0.02)

    # --- router ---
    logits = (x @ wr.T).astype(np.float32)
    m = logits.max()
    probs = np.exp(logits - m)
    probs /= probs.sum()
    idx = np.argsort(-probs, kind="stable")[:TOPK].astype(np.int32)
    w = probs[idx]
    w = w / w.sum()

    # --- routed experts ---
    # h is rounded to bf16 before the down projection: the engine stores
    # activations bf16 between the two skinny GEMVs, and ssm-ref.py models the
    # same rounding on `gated`. Without it the reference measures a precision
    # choice rather than whether the kernel implements the block.
    out = np.zeros(H, dtype=np.float32)
    for j in range(TOPK):
        e = int(idx[j])
        h = to_bf16(silu(x @ wg[e].T) * (x @ wu[e].T))
        out += w[j] * (h @ wd[e].T)

    # --- shared expert, sigmoid-gated ---
    hs = to_bf16(silu(x @ wgs.T) * (x @ wus.T))
    shared = (hs @ wds.T) * sigmoid(np.float32(x @ wsg))
    y = out + shared

    x.tofile(D / "moe_x.bin")
    wr.tofile(D / "moe_wr.bin")
    wsg.tofile(D / "moe_wsg.bin")
    bf16_store(wg).tofile(D / "moe_wg.bin")
    bf16_store(wu).tofile(D / "moe_wu.bin")
    bf16_store(wd).tofile(D / "moe_wd.bin")
    bf16_store(wgs).tofile(D / "moe_wgs.bin")
    bf16_store(wus).tofile(D / "moe_wus.bin")
    bf16_store(wds).tofile(D / "moe_wds.bin")
    idx.tofile(D / "moe_idx_ref.bin")
    w.astype(np.float32).tofile(D / "moe_w_ref.bin")
    out.tofile(D / "moe_routed_ref.bin")
    shared.tofile(D / "moe_shared_ref.bin")
    y.tofile(D / "moe_y_ref.bin")

    print(json.dumps({
        "top1_expert": int(idx[0]),
        "top1_weight": float(w[0]),
        "topk_weight_sum": float(w.sum()),
        "routed_mean_abs": float(np.abs(out).mean()),
        "shared_mean_abs": float(np.abs(shared).mean()),
        "y_mean_abs": float(np.abs(y).mean()),
    }))


if __name__ == "__main__":
    main()
