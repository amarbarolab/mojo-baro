#!/usr/bin/env python3
"""REALIGN oracle: numpy-only check that e_dev == softmax(logits) @ W_emb.

Reads the raw f32 dumps kernels/test_realign.mojo writes to
.work/realign-dump/<prompt>-{logits,e}.f32 and the token-embedding table
straight out of the pack (token_embd.weight, bf16, offset 0). Never
touches the GPU.

Usage: tools/realign_oracle.py [--pack DIR] [--dump DIR] [PROMPT ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

VOCAB = 248320
H = 4096

DEFAULT_PROMPTS = ["p01-water", "p02-python-fib", "p03-story", "p04-list-planets", "p05-math"]


def load_embed_table(packdir: Path) -> np.ndarray:
    off = None
    n = None
    with open(packdir / "index.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            if parts[0] == "token_embd.weight":
                if parts[1] != "bf16":
                    raise ValueError(f"token_embd.weight dtype {parts[1]!r}, expected bf16")
                off, n = int(parts[2]), int(parts[3])
                break
    if off is None:
        raise ValueError("token_embd.weight not found in index.txt")
    if n != VOCAB * H:
        raise ValueError(f"token_embd.weight n={n}, expected VOCAB*H={VOCAB * H}")

    raw = np.memmap(packdir / "pack.bin", dtype=np.uint16, mode="r", offset=off, shape=(VOCAB, H))
    # bf16 -> f32: same bits, upper 16 of a 32-bit float.
    return (raw.astype(np.uint32) << 16).view(np.float32)


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
    table = load_embed_table(packdir)

    print("prompt | argmax_tok | nearest_emb_tok(cos) | max|e_diff|/max|e| | verdict")
    print("-------+------------+----------------------+---------------------+--------")

    worst = 0.0
    for pname in args.prompts:
        logits = np.fromfile(dumpdir / f"{pname}-logits.f32", dtype=np.float32)
        e_got = np.fromfile(dumpdir / f"{pname}-e.f32", dtype=np.float32)
        if logits.shape[0] != VOCAB or e_got.shape[0] != H:
            raise ValueError(f"{pname}: bad dump shapes logits={logits.shape} e={e_got.shape}")

        probs = softmax(logits.astype(np.float64)).astype(np.float32)
        e_ref = probs @ table

        max_abs_e = float(np.max(np.abs(e_ref))) or 1.0
        diff = float(np.max(np.abs(e_got - e_ref))) / max_abs_e
        worst = max(worst, diff)

        argmax_tok = int(np.argmax(logits))
        nn_tok, nn_cos = nearest_embedding(e_got, table)

        verdict = "PASS" if diff <= 1e-3 else "FAIL"
        print(f"{pname} | {argmax_tok} | {nn_tok} ({nn_cos:.4f}) | {diff:.6e} | {verdict}")

    print("-------------------------------------------------------------------------")
    print(f"worst max|e_diff|/max|e| = {worst:.6e} (threshold 1e-3)")
    return 0 if worst <= 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
