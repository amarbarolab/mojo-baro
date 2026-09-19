#!/usr/bin/env python3
"""tools/gguf-msb-truncate.py: oracle for a shared-weight low-bit drafter.
Copies a GGUF and, in every routed-expert Q4_K tensor, keeps only the top
BITS bits of each 4-bit code, scales untouched. The dropped bits are filled
with mid-1 and mid on alternating codes, so the fill averages the true midpoint:
a constant fill shifts every weight the same way and that DC bias alone turns
the model to noise (measured 2026-09-19: 0% agreement, PPL 447k at 3 bits).
The result still loads as Q4_K in llama.cpp, so bench/draft-agreement.sh can
measure how often a drafter that reads only the top bit-plane of the SAME
weights agrees with the full model. No speed meaning: it is an accuracy probe.

Usage: tools/gguf-msb-truncate.py SRC.gguf DST.gguf BITS   (BITS = 1, 2 or 3)
"""
import shutil, sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T


def main():
    src, dst, bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if bits not in (1, 2, 3):
        raise RuntimeError("BITS must be 1, 2 or 3")
    drop = 4 - bits
    keep, mid = (0xF << drop) & 0xF, 1 << (drop - 1)
    byte_lut = np.array([((b & 15) & keep) | (mid - 1) | ((((b >> 4) & keep) | mid) << 4)
                         for b in range(256)], dtype=np.uint8)
    shutil.copyfile(src, dst)
    reader = GGUFReader(dst, "r+")
    done, skipped = 0, {}
    for t in reader.tensors:
        if "_exps" not in t.name:
            continue
        if t.tensor_type != T.Q4_K:
            skipped[t.tensor_type.name] = skipped.get(t.tensor_type.name, 0) + 1
            continue
        blocks = t.data.reshape(-1, 144)          # d, dmin, 12 scale bytes, 128 code bytes
        blocks[:, 16:] = byte_lut[blocks[:, 16:]]
        done += 1
    reader.data.flush()
    print(f"truncated {done} expert tensors to {bits} bits; left untouched: {skipped}")
    if done == 0:
        raise RuntimeError("no Q4_K expert tensors found")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL gguf-msb-truncate: {error}", file=sys.stderr)
        sys.exit(1)
