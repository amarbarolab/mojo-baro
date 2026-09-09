#!/usr/bin/env python3
"""REALIGN oracle (round 2): numpy-only, end-to-end check of
e_dev == softmax(rmsnorm(x0, output_norm.weight) @ output.weight.T) @ W_emb.

Reads the raw f32 dumps kernels/test_realign.mojo writes to
.work/realign-dump/<prompt>-{x0,e}.f32 -- x0 is the pre-final-norm hidden
state (WindowBufs.x_d row 0) at the point realign_expected_embedding was
called, e is its computed output -- and recomputes logits and e entirely
independently off the pack's own weights (token_embd.weight bf16,
output_norm.weight f32, output.weight q4_0). Never touches the GPU, never
reads a Mojo-side logits/hn dump (there isn't one: see serve/realign.mojo's
docstring for why b.logits_d/b.hn_d are not usable as of round 2).

Usage: tools/realign_oracle.py [--pack DIR] [--dump DIR] [PROMPT ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

VOCAB = 248320
H = 4096
EPS = 1e-6

DEFAULT_PROMPTS = ["p01-water", "p02-python-fib", "p03-story", "p04-list-planets", "p05-math"]


def read_index(packdir: Path) -> dict[str, tuple[str, int, int]]:
    idx = {}
    with open(packdir / "index.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            idx[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
    return idx


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_embed_table(packdir: Path, idx: dict) -> np.ndarray:
    dt, off, n = idx["token_embd.weight"]
    assert dt == "bf16", dt
    assert n == VOCAB * H, n
    raw = np.memmap(packdir / "pack.bin", dtype=np.uint16, mode="r", offset=off, shape=(VOCAB, H))
    return bf16_to_f32(raw)


def load_norm(packdir: Path, idx: dict, name: str, n: int) -> np.ndarray:
    dt, off, ne = idx[name]
    assert dt == "f32", dt
    assert ne == n, (name, ne, n)
    return np.memmap(packdir / "pack.bin", dtype=np.float32, mode="r", offset=off, shape=(n,)).copy()


def dequant_q4_0(packdir: Path, idx: dict, name: str, n_out: int, k: int) -> np.ndarray:
    """ggml Q4_0: per 32-block, 16 packed-nibble bytes + fp16 scale d.
    x[j] = d*(lo[j]-8) for j in 0..15, x[j+16] = d*(hi[j]-8) for j in 0..15
    (tools/engine-pack.py quantize_q4_0, inverted)."""
    dt, off, ne = idx[name]
    assert dt == "q4", dt
    assert ne == n_out * k, (name, ne, n_out * k)
    nb = k // 32
    nqs = ne // 2
    pack = np.memmap(packdir / "pack.bin", dtype=np.uint8, mode="r")
    q = np.frombuffer(pack[off : off + nqs], dtype=np.uint8).reshape(n_out, nb, 16)
    d = np.frombuffer(pack[off + nqs : off + nqs + nb * n_out * 2], dtype=np.float16).reshape(n_out, nb, 1).astype(np.float32)
    lo = (q & 0x0F).astype(np.float32) - 8.0
    hi = (q >> 4).astype(np.float32) - 8.0
    x = np.empty((n_out, nb, 32), dtype=np.float32)
    x[:, :, 0:16] = lo * d
    x[:, :, 16:32] = hi * d
    return x.reshape(n_out, k)


def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return (x * scale * g).astype(np.float32)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16, kept widened to f32 (same value the
    GPU's cast[DType.bfloat16]() then implicit widen-on-read produces) --
    rmsc_h2 casts the normalized activation to bf16 before the head GEMM
    reads it, so the oracle must lose the same 16 mantissa bits or its
    "reference" is a higher-precision computation the GPU never performed."""
    u32 = x.astype(np.float32).view(np.uint32)
    rounded = (u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def nearest_embedding(e: np.ndarray, table: np.ndarray) -> tuple[int, float]:
    en = e / (np.linalg.norm(e) + 1e-30)
    tn = np.linalg.norm(table, axis=1) + 1e-30
    cos = (table @ en) / tn
    tok = int(np.argmax(cos))
    return tok, float(cos[tok])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=".work/engine-pack-q4")
    ap.add_argument("--dump", default=".work/realign-dump")
    ap.add_argument("prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = ap.parse_args()

    packdir = Path(args.pack)
    dumpdir = Path(args.dump)
    idx = read_index(packdir)

    emb_table = load_embed_table(packdir, idx)
    norm_w = load_norm(packdir, idx, "output_norm.weight", H)
    head_w = dequant_q4_0(packdir, idx, "output.weight", VOCAB, H)

    print("prompt | argmax_tok | nearest_emb_tok(cos) | max|e_diff|/max|e| | verdict")
    print("-------+------------+----------------------+---------------------+--------")

    worst = 0.0
    for pname in args.prompts:
        x0 = np.fromfile(dumpdir / f"{pname}-x0.f32", dtype=np.float32)
        e_got = np.fromfile(dumpdir / f"{pname}-e.f32", dtype=np.float32)
        if x0.shape[0] != H or e_got.shape[0] != H:
            raise ValueError(f"{pname}: bad dump shapes x0={x0.shape} e={e_got.shape}")

        curb = to_bf16(rmsnorm(x0, norm_w, EPS))
        logits = head_w @ curb
        probs = softmax(logits.astype(np.float64)).astype(np.float32)
        e_ref = probs @ emb_table

        max_abs_e = float(np.max(np.abs(e_ref))) or 1.0
        diff = float(np.max(np.abs(e_got - e_ref))) / max_abs_e
        worst = max(worst, diff)

        argmax_tok = int(np.argmax(logits))
        nn_tok, nn_cos = nearest_embedding(e_got, emb_table)

        verdict = "PASS" if diff <= 1e-3 else "FAIL"
        print(f"{pname} | {argmax_tok} | {nn_tok} ({nn_cos:.4f}) | {diff:.6e} | {verdict}")

    print("-------------------------------------------------------------------------")
    print(f"worst max|e_diff|/max|e| = {worst:.6e} (threshold 1e-3)")
    return 0 if worst <= 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
