#!/usr/bin/env python3
"""Build the engine weight pack from a GGUF: one flat binary + text index.

2D bf16 weights are stored TRANSPOSED (B-layout [in, out]) for the skinny
GEMM; f32 tensors (norms, conv, ssm scalars) as-is. Emits, per line:
  name dtype offset_bytes n_elem
in a fixed, engine-known order. The blk.32 NextN draft head is appended
after output.weight, so every trunk offset is unchanged by its presence.

Usage: tools/engine-pack.py MODEL.gguf OUTDIR [--q8|--q4|--q2b3|--tq1|--tq2] [--q4-draft]

A source GGUF may hold its 2D weights as K-quants (Q4_K, Q6_K, ...) instead of
bf16: each such tensor is dequantised to f32 with gguf-py
(`gguf.quants.dequantize`), rounded to bf16 (round-to-nearest-even), then fed
through the same bf16 paths below (quantize/transpose) as a native-bf16
source. `token_embd.weight` stays row-major, untransposed, unquantized either
way.

--q8: every 2D bf16 weight except token_embd is stored int8 in weight-native
[out, in] layout followed by fp16 block scales [out, in/32], ggml q8_0
rounding (d = amax/127 in fp32, q = roundf(x * (1/d)), d stored as fp16).
Index dtype is "q8"; n_elem counts weights, the scales follow at
offset + n_elem bytes.

--q4-draft: after the existing order (trunk untouched, keeps whatever the
--q8 flag gave it), append output.weight a second time as ggml-exact Q4_0
(int4 nibbles [N, K/2], element order per quantize_row_q4_0_ref: qs[j] low
nibble = x[j], high nibble = x[j+16], j in 0..15 per 32-block; fp16 block-32
scale d = max/-8 where max is the SIGNED value at the largest-|x| position)
under index name "output.weight.q4draft", dtype "q4". This is the MTP draft
head's weight only; the draft path opts in via BARO_DRAFT_Q4, the trunk
lookup (name "output.weight") is unaffected since it matches the first,
unchanged, entry.

--q2b3 / --tq1 / --tq2: same tensor selection as --q8, ternary block codecs
instead. Each stores the block payload bytes verbatim in weight-native
[out, in/BLOCK * PAYLOAD] order followed by fp16 block scales [out, in/BLOCK]
(scale = fp16(amax), weights are exactly {-d, 0, +d}). Index dtypes and
geometry:
  q2b3  Q2_B3 (B3S fork, ggml type 43): 128 weights, 26 payload bytes,
        base-3 five trits per byte in the v2 chunk-aligned layout
  tq1   TQ1_0 (ggml type 34): 256 weights, 52 payload bytes (qs[48] + qh[4])
  tq2   TQ2_0 (ggml type 35): 256 weights, 64 payload bytes, 2 bits each
Quantizers are quantize_row_*_ref from ggml, vectorized; tools/b3s-check.py
proves them bit-equal to the C in tools/ternary-ref.c.
"""
import sys
from pathlib import Path

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize as gguf_dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)

N_LAYERS = 32


def is_attn(i):
    return (i + 1) % 4 == 0


def mtp_names():
    base = "blk.32."
    core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
            "attn_output.weight", "post_attention_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "nextn.eh_proj.weight", "nextn.enorm.weight",
            "nextn.hnorm.weight", "nextn.shared_head_norm.weight"]
    return [base + n for n in core]


def layer_names(i):
    base = f"blk.{i}."
    if is_attn(i):
        core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
                "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                "attn_output.weight"]
    else:
        core = ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"]
    ffn = ["post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight",
           "ffn_down.weight"]
    return [base + n for n in core + ffn]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x32):
    """Round-to-nearest-even, matching ggml's ggml_fp32_to_bf16."""
    u = x32.astype(np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def dequantize_kquant(f, data_start, toff, ttype, shape):
    """A Q4_K/Q6_K (or any other gguf-py-known non-bf16/f32/f16) tensor,
    dequantised to bf16 raw bytes via gguf-py so it can pass through the
    same bf16 quantize/store paths below."""
    qtype = GGMLQuantizationType(ttype)
    block, tsize = GGML_QUANT_SIZES[qtype]
    n_elem = int(np.prod(shape))
    assert n_elem % block == 0, (shape, block)
    f.seek(data_start + toff)
    raw_q = np.frombuffer(f.read(n_elem // block * tsize), dtype=np.uint8)
    w32 = gguf_dequantize(raw_q, qtype).reshape(shape).astype(np.float32)
    return f32_to_bf16(w32).tobytes()


def quantize_q8_0(w16):
    x = bf16_to_f32(w16).reshape(w16.shape[0], -1, 32)
    amax = np.abs(x).max(axis=2)
    d = (amax / 127.0).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0)).astype(np.float32)
    x0 = (x * idv[:, :, None]).astype(np.float32)
    q = np.sign(x0) * np.floor(np.abs(x0) + np.float32(0.5))
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(w16.shape[0], -1), d.astype(np.float16)


def quantize_q4_0(w16):
    """ggml quantize_row_q4_0_ref, vectorized. w16: [N, K] bf16, K % 32 == 0."""
    n_out, k = w16.shape
    x = bf16_to_f32(w16).reshape(n_out, k // 32, 32)
    amax_idx = np.abs(x).argmax(axis=2)
    signed_max = np.take_along_axis(x, amax_idx[:, :, None], axis=2)[:, :, 0]
    d = (signed_max / np.float32(-8.0)).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0)).astype(np.float32)
    x0 = x[:, :, 0:16] * idv[:, :, None]
    x1 = x[:, :, 16:32] * idv[:, :, None]
    xi0 = np.clip(np.trunc(x0 + np.float32(8.5)), 0, 15).astype(np.uint8)
    xi1 = np.clip(np.trunc(x1 + np.float32(8.5)), 0, 15).astype(np.uint8)
    qs = (xi0 | (xi1 << np.uint8(4))).astype(np.uint8)
    return qs.reshape(n_out, -1), d.astype(np.float16)


def _ternary_codes(w16, block):
    """ggml ternary rounding shared by q2_b3 / tq1_0 / tq2_0: per block of
    `block` weights, d = amax, id = 1/amax (fp32), code = roundf(x*id) + 1 in
    {0,1,2} (half away from zero, as roundf/lroundf). Returns codes
    [N, nb, block] uint8 and scales [N, nb] fp16."""
    n_out, k = w16.shape
    assert k % block == 0, (k, block)
    x = bf16_to_f32(w16).reshape(n_out, k // block, block)
    amax = np.abs(x).max(axis=2).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(amax > 0, np.float32(1.0) / amax, np.float32(0)).astype(np.float32)
    p = (x * idv[:, :, None]).astype(np.float32)
    q = np.rint(p)
    tie = np.abs(p) == np.float32(0.5)
    q[tie] = np.sign(p[tie])
    q = np.clip(q.astype(np.int32) + 1, 0, 2).astype(np.uint8)
    return q, amax.astype(np.float16)


def _q2b3_pack_matrix():
    P = np.zeros((128, 26), dtype=np.int32)
    pw3 = [1, 3, 9, 27, 81]
    for j in range(128):
        c, t = j >> 5, j & 31
        byte = 6 * c + t // 5 if t < 30 else 24 + (c >> 1)
        digit = t % 5 if t < 30 else 2 * (c & 1) + (t - 30)
        P[j, byte] = pw3[digit]
    return P


def quantize_q2_b3(w16):
    """B3S quantize_row_q2_b3_ref, vectorized. w16: [N, K] bf16, K % 128 == 0.
    Returns payload [N, K/128 * 26] uint8 and scales [N, K/128] fp16."""
    q, d = _ternary_codes(w16, 128)
    qs = (q.astype(np.int32) @ _q2b3_pack_matrix()).astype(np.uint8)
    return qs.reshape(w16.shape[0], -1), d


def _tq1_pack_matrix():
    P = np.zeros((256, 52), dtype=np.int32)
    for m in range(32):
        for n in range(5):
            P[m + n * 32, m] = 3 ** (4 - n)
    for m in range(16):
        for n in range(5):
            P[160 + m + n * 16, 32 + m] = 3 ** (4 - n)
    for j in range(4):
        for m in range(4):
            P[240 + j + m * 4, 48 + j] = 3 ** (4 - m)
    return P


def quantize_tq1_0(w16):
    """ggml quantize_row_tq1_0_ref, vectorized. w16: [N, K] bf16, K % 256 == 0.
    Returns payload [N, K/256 * 52] uint8 (qs[48] then qh[4] per block, the
    ceiling-scaled base-3 bytes) and scales [N, K/256] fp16."""
    q, d = _ternary_codes(w16, 256)
    v = q.astype(np.int32) @ _tq1_pack_matrix()
    qs = ((v * 256 + 242) // 243).astype(np.uint8)
    return qs.reshape(w16.shape[0], -1), d


def _tq2_pack_matrix():
    P = np.zeros((256, 64), dtype=np.int32)
    for j in (0, 32):
        for l in range(4):
            for m in range(32):
                P[j * 4 + l * 32 + m, j + m] = 1 << (2 * l)
    return P


def quantize_tq2_0(w16):
    """ggml quantize_row_tq2_0_ref, vectorized. w16: [N, K] bf16, K % 256 == 0.
    Returns payload [N, K/256 * 64] uint8 (2 bits per weight) and scales
    [N, K/256] fp16."""
    q, d = _ternary_codes(w16, 256)
    qs = (q.astype(np.int32) @ _tq2_pack_matrix()).astype(np.uint8)
    return qs.reshape(w16.shape[0], -1), d


TERNARY = {"q2b3": quantize_q2_b3, "tq1": quantize_tq1_0, "tq2": quantize_tq2_0}


def main():
    q8 = "--q8" in sys.argv
    if q8:
        sys.argv.remove("--q8")
    q4 = "--q4" in sys.argv
    if q4:
        sys.argv.remove("--q4")
    assert not (q8 and q4), "--q8 and --q4 are exclusive"
    ternary = None
    for flag in TERNARY:
        if "--" + flag in sys.argv:
            sys.argv.remove("--" + flag)
            ternary = flag
    assert not ((q8 or q4) and ternary), "--q8/--q4 and a ternary flag are exclusive"
    q4_draft = "--q4-draft" in sys.argv
    if q4_draft:
        sys.argv.remove("--q4-draft")
    model, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    f, infos, data_start, kv = ge.parse(model)

    order = ["token_embd.weight"]
    for i in range(N_LAYERS):
        order += layer_names(i)
    order += ["output_norm.weight", "output.weight"]
    order += mtp_names()

    idx_lines = []
    off = 0
    with open(outdir / "pack.bin", "wb") as out:
        for name in order:
            dims, ttype, toff = infos[name]
            n_elem = int(np.prod(dims))
            shape = list(reversed(dims))
            if ttype in ge.GGML_BYTES:
                tname, esize = ge.GGML_BYTES[ttype]
                f.seek(data_start + toff)
                raw = f.read(n_elem * esize)
            else:
                tname = "bf16"
                raw = dequantize_kquant(f, data_start, toff, ttype, shape)
            if tname == "bf16" and len(shape) == 2 and name != "token_embd.weight":
                w = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                if q8:
                    q, d = quantize_q8_0(w)
                    raw = q.tobytes() + d.tobytes()
                    tname = "q8"
                elif q4:
                    q, d = quantize_q4_0(w)
                    raw = q.tobytes() + d.tobytes()
                    tname = "q4"
                elif ternary:
                    q, d = TERNARY[ternary](w)
                    raw = q.tobytes() + d.tobytes()
                    tname = ternary
                else:
                    raw = np.ascontiguousarray(w.T).tobytes()
            out.write(raw)
            idx_lines.append(f"{name} {tname} {off} {n_elem}")
            off += len(raw)
        if q4_draft:
            dims, ttype, toff = infos["output.weight"]
            tname, esize = ge.GGML_BYTES[ttype]
            assert tname == "bf16", tname
            n_elem = int(np.prod(dims))
            f.seek(data_start + toff)
            w = np.frombuffer(f.read(n_elem * esize), dtype=np.uint16).reshape(
                list(reversed(dims))
            )
            q, d = quantize_q4_0(w)
            raw = q.tobytes() + d.tobytes()
            out.write(raw)
            idx_lines.append(f"output.weight.q4draft q4 {off} {n_elem}")
            off += len(raw)
    (outdir / "index.txt").write_text("\n".join(idx_lines) + "\n")
    print(f"packed {len(order)} tensors, {off/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
