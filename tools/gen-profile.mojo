"""Emit a comptime Mojo profile module from a GGUF's own metadata.

Every dense-family GGUF (llama, qwen2, granite, spark2_5) stores its recipe
under uniform `<arch>.*` keys (llama.cpp convention), so the dims and norm
eps are read directly; VOCAB is read off `token_embd.weight`'s own shape
(the tensor bytes are authority, not a metadata count that can drift).
Rope type (norm/neox) and activation are not stored in the GGUF -- llama.cpp
picks them per architecture in `llm_arch_rope_type` / the model's graph
builder, so they are looked up from a small per-arch table below, same as
any other recipe fact read out of llama.cpp source as spec.

Usage: tools/gen-profile.mojo MODEL.gguf OUT.mojo [--tmax N]
Build:  ./.venv/bin/mojo build tools/gen-profile.mojo -I serve \
          -I ~/Projects/mojo/mojo-uregex/src -o .work/gen-profile
"""
from std.collections import Dict
from std.math import sqrt
from std.memory import bitcast
from std.sys import argv
from tokenizer import Reader

comptime HEADER_MAX = 96 << 20


def scalar_f32(mut r: Reader) raises -> Float64:
    r.need(4)
    var bits: UInt32 = 0
    for i in range(4):
        bits |= UInt32(r.buf[r.pos + i]) << UInt32(8 * i)
    r.pos += 4
    return Float64(bitcast[DType.float32, 1](bits)[0])


def scalar_f64(mut r: Reader) raises -> Float64:
    r.need(8)
    var bits: UInt64 = 0
    for i in range(8):
        bits |= UInt64(r.buf[r.pos + i]) << UInt64(8 * i)
    r.pos += 8
    return bitcast[DType.float64, 1](bits)[0]


def is_rope_neox_arch(arch: String) -> Bool:
    return arch == "spark2_5" or arch == "qwen2"


def is_rope_norm_arch(arch: String) -> Bool:
    # llama.cpp's llm_arch_rope_type (llama-model.cpp): LLM_ARCH_GRANITE is
    # grouped with LLM_ARCH_LLAMA under "normal RoPE, pairs of consecutive
    # head values" -- NOT with LLM_ARCH_QWEN2's NeoX-style half-offset group.
    # Measured: granite-4.2-3b scored ~40/64 forced agreement vs llama.cpp
    # with NEOX assumed, ~62/64+ after moving it here (bench/dense-protocol.md).
    return arch == "llama" or arch == "granite"


def is_gelu_arch(arch: String) -> Bool:
    return arch == "spark2_5"


def is_silu_arch(arch: String) -> Bool:
    return arch == "llama" or arch == "qwen2" or arch == "granite"


def find_swa_period(pattern: List[Int]) raises -> Tuple[Int, Int]:
    """pattern[i] == 1 means layer i is windowed, 0 means full attention.
    Returns (period, full_phase) with swa(i) == (i % period != full_phase),
    or raises if no period in [1, 8] reproduces the array exactly."""
    var n = len(pattern)
    for period in range(1, 9):
        for full_phase in range(period):
            var ok = True
            for i in range(n):
                if (pattern[i] == 1) != ((i % period) != full_phase):
                    ok = False
                    break
            if ok:
                return (period, full_phase)
    raise Error("sliding_window_pattern is not periodic in [1,8]; gen-profile only supports a period/phase SWA rule")


def mojo_bool(b: Bool) -> String:
    return "True" if b else "False"


struct Model:
    var arch: String
    var ints: Dict[String, Int]
    var floats: Dict[String, Float64]
    var int_arrays: Dict[String, List[Int]]
    var tensor_dims: Dict[String, List[Int]]

    def __init__(out self, path: String) raises:
        self.arch = String("")
        self.ints = Dict[String, Int]()
        self.floats = Dict[String, Float64]()
        self.int_arrays = Dict[String, List[Int]]()
        self.tensor_dims = Dict[String, List[Int]]()
        var buf: List[UInt8]
        with open(path, "r") as f:
            buf = f.read_bytes(HEADER_MAX)
        var r = Reader(buf^)
        var magic = r.u32()
        if magic != 0x46554747:
            raise Error("gguf: bad magic")
        var version = r.u32()
        if version != 3:
            raise Error("gguf: unsupported version " + String(version))
        var n_tensors = r.u64()
        var n_kv = r.u64()
        for _ in range(n_kv):
            var key = r.string()
            var vtype = r.u32()
            if vtype == 8:
                var s = r.string()
                if key == "general.architecture":
                    self.arch = s
            elif vtype == 9:
                self._array(r, key)
            elif vtype == 6:
                self.floats[key] = scalar_f32(r)
            elif vtype == 12:
                self.floats[key] = scalar_f64(r)
            else:
                self.ints[key] = r.scalar_int(vtype)
        if self.arch == "":
            raise Error("gguf: no general.architecture key")
        for _ in range(n_tensors):
            var name = r.string()
            var n_dims = r.u32()
            var dims = List[Int]()
            for _ in range(n_dims):
                dims.append(r.u64())
            _ = r.u32()  # tensor type, unused
            _ = r.u64()  # data offset, unused
            self.tensor_dims[name] = dims^

    def _array(mut self, mut r: Reader, key: String) raises:
        var etype = r.u32()
        var n = r.u64()
        if etype == 8:
            for _ in range(n):
                _ = r.string()
            return
        if etype == 9:
            raise Error("gguf: nested arrays unsupported")
        var small_int = etype == 0 or etype == 1 or etype == 2 or etype == 3 or etype == 4 or etype == 5 or etype == 7 or etype == 10 or etype == 11
        if small_int and n <= 512:
            var vals = List[Int]()
            for _ in range(n):
                vals.append(r.scalar_int(etype))
            self.int_arrays[key] = vals^
            return
        for _ in range(n):
            if etype == 6:
                _ = scalar_f32(r)
            elif etype == 12:
                _ = scalar_f64(r)
            else:
                r.skip_scalar(etype)

    def akv_int(self, suffix: String, default: Int) -> Int:
        return self.ints.get(self.arch + "." + suffix, default)

    def akv_float(self, suffix: String, default: Float64) -> Float64:
        return self.floats.get(self.arch + "." + suffix, default)

    def tensor_present(self, name: String) -> Bool:
        return len(self.tensor_dims.get(name, List[Int]())) > 0


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: gen-profile MODEL.gguf OUT.mojo [--tmax N]")
        return
    var model_path = String(args[1])
    var out_path = String(args[2])
    var tmax = 4096
    for i in range(3, len(args)):
        if String(args[i]) == "--tmax" and i + 1 < len(args):
            tmax = atol(String(args[i + 1]))

    var m = Model(model_path)
    var arch = m.arch

    var h = m.akv_int("embedding_length", 0)
    var ffn = m.akv_int("feed_forward_length", 0)
    var n_layers = m.akv_int("block_count", 0)
    var nqh = m.akv_int("attention.head_count", 0)
    var nkvh = m.akv_int("attention.head_count_kv", 0)
    var hd = m.akv_int("attention.key_length", h // nqh)
    var norm_eps = m.akv_float("attention.layer_norm_rms_epsilon", 1e-6)
    var nrot_full = m.akv_int("rope.dimension_count", hd)
    var base_full = m.akv_float("rope.freq_base", 10000.0)

    var embd_dims = m.tensor_dims.get("token_embd.weight", List[Int]())
    if len(embd_dims) != 2:
        raise Error("token_embd.weight: expected 2 dims, got " + String(len(embd_dims)))
    var vocab = embd_dims[1]
    var tie_embed = not m.tensor_present("output.weight")
    var qkv_bias = m.tensor_present("blk.0.attn_q.bias")
    var has_gate = m.tensor_present("blk.0.attn_gate.weight")

    var swa_key = arch + ".attention.sliding_window_pattern"
    var swa_win: Int
    var nrot_swa: Int
    var base_swa: Float64
    var swa_period: Int
    var swa_full_phase: Int
    if len(m.int_arrays.get(swa_key, List[Int]())) > 0:
        var pattern = m.int_arrays[swa_key].copy()
        swa_win = m.akv_int("attention.sliding_window", 0)
        nrot_swa = m.akv_int("rope.dimension_count_swa", nrot_full)
        base_swa = m.akv_float("rope.freq_base_swa", base_full)
        var pf = find_swa_period(pattern)
        swa_period = pf[0]
        swa_full_phase = pf[1]
    else:
        swa_win = 0
        nrot_swa = nrot_full
        base_swa = base_full
        swa_period = 1
        swa_full_phase = 0

    var rope_neox: Bool
    if is_rope_neox_arch(arch):
        rope_neox = True
    elif is_rope_norm_arch(arch):
        rope_neox = False
    else:
        raise Error("unknown architecture " + arch + ": add it to is_rope_neox_arch or is_rope_norm_arch in tools/gen-profile.mojo after reading llama.cpp's llm_arch_rope_type for it")

    var activation_gelu: Bool
    if is_gelu_arch(arch):
        activation_gelu = True
    elif is_silu_arch(arch):
        activation_gelu = False
    else:
        raise Error("unknown architecture " + arch + ": add it to is_gelu_arch or is_silu_arch in tools/gen-profile.mojo after reading the model's graph builder in llama.cpp")

    var granite_mult = arch == "granite"
    var default_attn_scale = 1.0 / sqrt(Float64(hd))
    var attn_scale = m.akv_float("attention.scale", default_attn_scale) if granite_mult else default_attn_scale
    var embed_scale = m.akv_float("embedding_scale", 1.0)
    var residual_scale = m.akv_float("residual_scale", 1.0)
    var logit_scale = m.akv_float("logit_scale", 1.0)

    var ctx_len = m.akv_int("context_length", tmax)
    if ctx_len < tmax:
        print("gen-profile: warning: TMAX", tmax, "exceeds", arch + ".context_length", ctx_len)

    var out = String("# generated by tools/gen-profile.mojo from ") + model_path + " (arch=" + arch + ") -- do not hand-edit\n"
    out += "comptime H = " + String(h) + "\n"
    out += "comptime FFN = " + String(ffn) + "\n"
    out += "comptime VOCAB = " + String(vocab) + "\n"
    out += "comptime N_LAYERS = " + String(n_layers) + "\n"
    out += "comptime NQH = " + String(nqh) + "\n"
    out += "comptime NKVH = " + String(nkvh) + "\n"
    out += "comptime HD = " + String(hd) + "\n"
    out += "comptime NORM_EPS = Float32(" + String(norm_eps) + ")\n"
    out += "comptime NROT_FULL = " + String(nrot_full) + "\n"
    out += "comptime BASE_FULL = Float32(" + String(base_full) + ")\n"
    out += "comptime NROT_SWA = " + String(nrot_swa) + "\n"
    out += "comptime BASE_SWA = Float32(" + String(base_swa) + ")\n"
    out += "comptime SWA_WIN = " + String(swa_win) + "\n"
    out += "comptime SWA_PERIOD = " + String(swa_period) + "\n"
    out += "comptime SWA_FULL_PHASE = " + String(swa_full_phase) + "\n"
    out += "comptime ROPE_NEOX = " + mojo_bool(rope_neox) + "\n"
    out += "comptime QKV_BIAS = " + mojo_bool(qkv_bias) + "\n"
    out += "comptime HAS_GATE = " + mojo_bool(has_gate) + "\n"
    out += "comptime TIE_EMBED = " + mojo_bool(tie_embed) + "\n"
    out += "comptime ACTIVATION_GELU = " + mojo_bool(activation_gelu) + "\n"
    out += "comptime GRANITE_MULT = " + mojo_bool(granite_mult) + "\n"
    out += "comptime ATTN_SCALE = Float32(" + String(attn_scale) + ")\n"
    out += "comptime EMBED_SCALE = Float32(" + String(embed_scale) + ")\n"
    out += "comptime RESIDUAL_SCALE = Float32(" + String(residual_scale) + ")\n"
    out += "comptime LOGIT_SCALE = Float32(" + String(logit_scale) + ")\n"
    out += "comptime TMAX = " + String(tmax) + "\n"
    with open(out_path, "w") as f:
        f.write(out)
    print("wrote", out_path, "(arch=" + arch, "H=" + String(h), "FFN=" + String(ffn),
          "N_LAYERS=" + String(n_layers), "NQH=" + String(nqh), "NKVH=" + String(nkvh),
          "HD=" + String(hd), "VOCAB=" + String(vocab) + ")")
