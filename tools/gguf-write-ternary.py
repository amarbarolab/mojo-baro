#!/usr/bin/env python3
"""Write a Q2_B3 / TQ1_0 / TQ2_0 GGUF from a bf16 source model.

Usage:
  tools/gguf-write-ternary.py MODEL.gguf OUT.gguf --q2b3|--tq1|--tq2
  tools/gguf-write-ternary.py --verify PACKDIR OUT.gguf name [name ...]

Tensor selection matches tools/engine-pack.py --q8: every 2D bf16 weight
except token_embd.weight is quantized (functions imported from engine-pack,
not duplicated); everything else -- 1D tensors, token_embd, all KV metadata,
tokenizer -- is copied verbatim via the source file's own gguf-py GGUFReader
(NOT tools/gguf-extract.py's parser for the metadata path: that reader
truncates KV arrays over 4096 elements to a placeholder string, which would
silently drop the tokenizer vocab; tools/gguf-extract.py's parser is still
used, exactly as engine-pack.py uses it, to select and read the raw bf16
tensor bytes that get quantized).

engine-pack.py's --q2b3/--tq1/--tq2 packers store payload and scales split
(all payload bytes for a row, then all fp16 scales); a ggml block on disk is
interleaved per block instead:
  q2b3: d (fp16, 2B) FIRST, then 26 payload bytes  = 28 B/block
  tq1:  48 qs + 4 qh (52 B) then d LAST            = 54 B/block
  tq2:  64 qs then d LAST                          = 66 B/block
interleave()/deinterleave() do this with numpy views, no Python loops.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from importlib.util import spec_from_file_location, module_from_spec


def _load(name):
    spec = spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ge = _load("gguf-extract")
ep = _load("engine-pack")

GGUF_PY_CANDIDATES = [
    os.path.expanduser("~/llama.cpp-b3s/gguf-py"),
    os.path.expanduser("~/llama.cpp/gguf-py"),
]

# family -> (block weights, payload bytes, block bytes, scale-first, ggml type, LLAMA_FTYPE_MOSTLY_*)
FAMILIES = {
    "q2b3": dict(quant=ep.quantize_q2_b3, block=128, nbytes=26, bsize=28, scale_first=True, gtype=43, ftype=42),
    "tq1": dict(quant=ep.quantize_tq1_0, block=256, nbytes=52, bsize=54, scale_first=False, gtype=34, ftype=36),
    "tq2": dict(quant=ep.quantize_tq2_0, block=256, nbytes=64, bsize=66, scale_first=False, gtype=35, ftype=37),
}


def _import_gguf_bypass(gguf_py_dir):
    """`import gguf` runs gguf/__init__.py top to bottom, which pulls in
    tensor_mapping.py; that module's TensorNameMap class body references
    MODEL_TENSOR.DSPARK_MARKOV_W1, which this fork's constants.py does not
    define (a pre-existing gguf-py bug, unrelated to Q2_B3 and outside
    tools/gguf-write-ternary.py's ownership) so the plain import crashes
    before GGUFWriter/GGUFReader/GGMLQuantizationType are even reached.
    Load just the four submodules this tool needs, in their dependency
    order, without ever executing tensor_mapping/vocab/utility/metadata."""
    pkg_path = Path(gguf_py_dir) / "gguf"
    pkg = types.ModuleType("gguf")
    pkg.__path__ = [str(pkg_path)]
    sys.modules["gguf"] = pkg
    for modname in ("constants", "lazy", "quants", "gguf_reader", "gguf_writer"):
        spec = spec_from_file_location(f"gguf.{modname}", pkg_path / f"{modname}.py")
        mod = module_from_spec(spec)
        sys.modules[f"gguf.{modname}"] = mod
        spec.loader.exec_module(mod)
        for k, v in vars(mod).items():
            if not k.startswith("_"):
                setattr(pkg, k, v)
    return pkg


def _import_gguf():
    for c in GGUF_PY_CANDIDATES:
        if Path(c).is_dir():
            gguf = _import_gguf_bypass(c)
            assert gguf.GGMLQuantizationType.Q2_B3 == 43, gguf.GGMLQuantizationType.Q2_B3
            assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q2_B3] == (128, 28), \
                gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q2_B3]
            return gguf, c
    raise RuntimeError(f"no gguf-py with Q2_B3 found in {GGUF_PY_CANDIDATES}")


def interleave(payload, scales, fam):
    f = FAMILIES[fam]
    n, nb = scales.shape
    q3 = payload.reshape(n, nb, f["nbytes"])
    d3 = np.ascontiguousarray(scales).view(np.uint8).reshape(n, nb, 2)
    parts = (d3, q3) if f["scale_first"] else (q3, d3)
    block = np.concatenate(parts, axis=2)
    assert block.shape[2] == f["bsize"], (block.shape, f["bsize"])
    return np.ascontiguousarray(block).reshape(n, nb * f["bsize"])


def deinterleave(raw, n, nb, fam):
    f = FAMILIES[fam]
    b = raw.reshape(n, nb, f["bsize"])
    if f["scale_first"]:
        d, q = b[:, :, :2], b[:, :, 2:]
    else:
        q, d = b[:, :, :f["nbytes"]], b[:, :, f["nbytes"]:]
    return (np.ascontiguousarray(q).reshape(n, -1),
            np.ascontiguousarray(d).reshape(n, -1).view(np.float16))


def write_gguf(model_path, out_path, fam, gguf_mod):
    f, infos, data_start, _kv = ge.parse(model_path)
    fspec = FAMILIES[fam]

    selected = {}
    for name, (dims, ttype, _toff) in infos.items():
        tname, _esize = ge.GGML_BYTES.get(ttype, (None, None))
        shape = list(reversed(dims))
        if tname == "bf16" and len(shape) == 2 and name != "token_embd.weight":
            selected[name] = shape  # [out, in]

    def quantized_bytes(name, shape):
        dims, _ttype, toff = infos[name]
        n_elem = int(np.prod(dims))
        f.seek(data_start + toff)
        w16 = np.frombuffer(f.read(n_elem * 2), dtype=np.uint16).reshape(shape)
        payload, scales = fspec["quant"](w16)
        return interleave(payload, scales, fam)

    reader = gguf_mod.GGUFReader(str(model_path), "r")
    Keys = gguf_mod.Keys
    arch_field = reader.get_field(Keys.General.ARCHITECTURE)
    arch = arch_field.contents()
    writer = gguf_mod.GGUFWriter(str(out_path), arch=arch, endianess=reader.endianess)

    align_field = reader.get_field(Keys.General.ALIGNMENT)
    if align_field is not None:
        writer.data_alignment = align_field.contents()

    for field in reader.fields.values():
        if field.name in (Keys.General.ARCHITECTURE, Keys.General.FILE_TYPE):
            continue
        if field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf_mod.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_uint32(Keys.General.FILE_TYPE, fspec["ftype"])

    quant_cache = {}
    ggml_type = gguf_mod.GGMLQuantizationType(fspec["gtype"])
    for t in reader.tensors:
        if t.name in selected:
            qb = quantized_bytes(t.name, selected[t.name])
            quant_cache[t.name] = qb
            writer.add_tensor_info(t.name, qb.shape, qb.dtype, qb.nbytes, raw_dtype=ggml_type)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        if t.name in quant_cache:
            writer.write_tensor_data(quant_cache[t.name])
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    return len(selected)


def verify(packdir, out_path, names):
    idx = {}
    for line in (packdir / "index.txt").read_text().splitlines():
        n, dt, off, ne = line.split()
        idx[n] = (dt, int(off), int(ne))
    pack = np.memmap(packdir / "pack.bin", dtype=np.uint8, mode="r")
    f, infos, data_start, _kv = ge.parse(out_path)
    ok = True
    for name in names:
        dt, off, _ne = idx[name]
        fspec = FAMILIES[dt]
        block, nbytes = fspec["block"], fspec["nbytes"]
        dims, _ttype, toff = infos[name]
        in_f, out_f = dims[0], dims[1]
        nb = in_f // block
        row_bytes = nb * fspec["bsize"]
        f.seek(data_start + toff)
        raw = np.frombuffer(f.read(out_f * row_bytes), dtype=np.uint8).reshape(out_f, row_bytes)
        gq, gd = deinterleave(raw, out_f, nb, dt)
        pq = np.frombuffer(pack[off: off + out_f * nb * nbytes], dtype=np.uint8).reshape(out_f, -1)
        pd = np.frombuffer(pack[off + out_f * nb * nbytes: off + out_f * nb * nbytes + out_f * nb * 2],
                            dtype=np.float16).reshape(out_f, nb)
        dq = int((gq != pq).sum())
        dd = int((gd.view(np.uint16) != pd.view(np.uint16)).sum())
        print(f"{name} [{dt}]: payload mismatch {dq}/{pq.size}  scales mismatch {dd}/{pd.size}")
        ok = ok and dq == 0 and dd == 0
    print("verify:", "PASS" if ok else "FAIL")
    return ok


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--verify"]:
        packdir, out_path, names = Path(argv[1]), Path(argv[2]), argv[3:]
        sys.exit(0 if verify(packdir, out_path, names) else 1)

    fam = None
    for flag in FAMILIES:
        if f"--{flag}" in argv:
            argv.remove(f"--{flag}")
            fam = flag
    assert fam, "need --q2b3 | --tq1 | --tq2"
    model, out = Path(argv[0]), Path(argv[1])

    gguf_mod, gguf_py_path = _import_gguf()
    print(f"using gguf-py: {gguf_py_path}")
    n = write_gguf(model, out, fam, gguf_mod)
    print(f"wrote {out}: {n} tensors quantized to {fam}")


if __name__ == "__main__":
    main()
