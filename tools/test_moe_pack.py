#!/usr/bin/env python3
"""Verify raw qwen35moe pack copies and the Q6_K output head requantisation."""
import sys
from pathlib import Path
import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize as gguf_dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
s = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(s)
s.loader.exec_module(ge)


def main():
    model, packdir = map(Path, sys.argv[1:3])
    f, infos, data_start, kv = ge.parse(model)
    assert kv["general.architecture"] == "qwen35moe"
    lines = (packdir / "index.txt").read_text().splitlines()
    assert len(lines) == len(infos), (len(lines), len(infos))
    pack = np.memmap(packdir / "pack.bin", dtype=np.uint8, mode="r")
    exact = 0
    head_bad = 0
    for line in lines:
        name, dt, off, n_elem = line.split()
        off, n_elem = int(off), int(n_elem)
        dims, ttype, toff = infos[name]
        if name == "output.weight":
            qtype = GGMLQuantizationType(ttype)
            block, tsize = GGML_QUANT_SIZES[qtype]
            f.seek(data_start + toff)
            src = np.frombuffer(f.read((n_elem // block) * tsize), dtype=np.uint8)
            source = gguf_dequantize(src, qtype).astype(np.float32)
            shape = list(reversed(dims))
            q = pack[off:off + n_elem].view(np.int8).reshape(shape[0], shape[1] // 32, 32)
            d = pack[off + n_elem:off + n_elem + n_elem // 32 * 2].view(np.float16).reshape(shape[0], shape[1] // 32, 1)
            got = (q.astype(np.float32) * d.astype(np.float32)).reshape(-1)
            tol = np.broadcast_to(d.astype(np.float32) * 0.5, q.shape).reshape(-1) + np.abs(source) * 2.0 ** -7 + 1e-8
            head_bad += int((np.abs(got - source) > tol).sum())
            continue
        if ttype in ge.GGML_BYTES:
            size = n_elem * ge.GGML_BYTES[ttype][1]
        else:
            block, size = GGML_QUANT_SIZES[GGMLQuantizationType(ttype)]
            size = n_elem // block * size
        f.seek(data_start + toff)
        source = f.read(size)
        assert bytes(pack[off:off + size]) == source, name
        exact += 1
    print(f"raw copies: {exact}/{len(lines) - 1}; output q6k over-bound: {head_bad}")
    ok = exact == len(lines) - 1 and head_bad == 0 and pack.size <= 21.5 * 2**30
    print("MoE pack gate:", "PASS" if ok else "FAIL", "bytes", pack.size)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
