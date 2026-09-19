#!/usr/bin/env python3
"""tools/mtp-head/export_frozen.py: dequantize the frozen token embedding and
output head from a GGUF to OUT/embed.npy and OUT/lm_head.npy (f16, [V, D]).
Usage: export_frozen.py MODEL.gguf OUT"""
import sys
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

try:
    reader, out = GGUFReader(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    found = 0
    for t in reader.tensors:
        name = {"token_embd.weight": "embed", "output.weight": "lm_head"}.get(t.name)
        if name:
            w = dequantize(t.data, t.tensor_type).astype(np.float16)
            np.save(out / f"{name}.npy", w)
            print(name, w.shape)
            found += 1
    if found != 2:
        raise RuntimeError(f"found {found} of 2 tensors")
except Exception as error:
    print(f"FAIL export_frozen: {error}", file=sys.stderr)
    sys.exit(1)
