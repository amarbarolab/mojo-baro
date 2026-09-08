from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from attn import KVT, HD, NQH, NKVH, KVPAGE, KVHSTR, amar_kv_append
from elementwise import amar_rmsnorm_cast, amar_argmax_pos
from matmul_skinny import amar_matmul_skinny_q8row, amar_skinny_reduce, amar_skinny_reduce_add, ROW_WAVES, ROW_THREADS
from spark_kernels import (
    amar_embed_lookup_f32, amar_rope_plain, amar_attn_decode_swa,
    amar_head_gate_mul_cast, amar_skinny_reduce_gelu_par_bf16,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime H = 2560
comptime FFN = 10240
comptime VOCAB = 131072
comptime N_LAYERS = 36
comptime QDIM = NQH * HD
comptime KVDIM = NKVH * HD
comptime QKV = QDIM + 2 * KVDIM
comptime SWA_WIN = 512
comptime NROT_FULL = 64
comptime NROT_SWA = 256
comptime BASE_FULL = Float32(5e6)
comptime BASE_SWA = Float32(1e4)
comptime TMAX = 4096
comptime TPAGES = TMAX // KVPAGE
comptime KVPOOL = TPAGES * N_LAYERS * NKVH * KVHSTR

comptime x_l = row_major[1, H]()
comptime xb_l = row_major[1, H]()
comptime h_l = row_major[H]()
comptime toks_l = row_major[TMAX]()
comptime emb_l = row_major[VOCAB, H]()
comptime p_qkv_l = row_major[1, 1, QKV]()
comptime qkv_l = row_major[1, QKV]()
comptime q_l = row_major[NQH, HD]()
comptime kv_l = row_major[NKVH, HD]()
comptime cache_l = row_major[KVPOOL]()
comptime aoflat_l = row_major[QDIM]()
comptime aob_l = row_major[1, QDIM]()
comptime p_gate_l = row_major[1, 1, NQH]()
comptime gate2_l = row_major[1, NQH]()
comptime gate_l = row_major[NQH]()
comptime p_h_l = row_major[1, 1, H]()
comptime p_ffn_l = row_major[1, 1, FFN]()
comptime fgb_l = row_major[1, FFN]()
comptime p_v_l = row_major[1, 1, VOCAB]()
comptime logits_l = row_major[1, VOCAB]()

comptime q_qkv = row_major[QKV, H]()
comptime s_qkv = row_major[QKV, H // 32]()
comptime q_gate = row_major[NQH, H]()
comptime s_gate = row_major[NQH, H // 32]()
comptime q_o = row_major[H, QDIM]()
comptime s_o = row_major[H, QDIM // 32]()
comptime q_ffn = row_major[FFN, H]()
comptime s_ffn = row_major[FFN, H // 32]()
comptime q_down = row_major[H, FFN]()
comptime s_down = row_major[H, FFN // 32]()
comptime q_out = row_major[VOCAB, H]()
comptime s_out = row_major[VOCAB, H // 32]()

comptime k_emb = amar_embed_lookup_f32[type_of(emb_l), type_of(x_l), type_of(toks_l)]
comptime k_rms = amar_rmsnorm_cast[type_of(x_l), type_of(h_l), type_of(xb_l)]
comptime k_qkv = amar_matmul_skinny_q8row[4, 1, type_of(xb_l), type_of(q_qkv), type_of(s_qkv), type_of(p_qkv_l)]
comptime k_r_qkv = amar_skinny_reduce[type_of(p_qkv_l), type_of(qkv_l), 1]
comptime k_rope_full = amar_rope_plain[NROT_FULL, type_of(q_l)]
comptime k_rope_swa = amar_rope_plain[NROT_SWA, type_of(q_l)]
comptime k_rope_full_k = amar_rope_plain[NROT_FULL, type_of(kv_l)]
comptime k_rope_swa_k = amar_rope_plain[NROT_SWA, type_of(kv_l)]
comptime k_append = amar_kv_append[type_of(cache_l), type_of(kv_l), N_LAYERS]
comptime k_att = amar_attn_decode_swa[type_of(q_l), type_of(cache_l), type_of(q_l), N_LAYERS]
comptime k_gate = amar_matmul_skinny_q8row[4, 1, type_of(xb_l), type_of(q_gate), type_of(s_gate), type_of(p_gate_l)]
comptime k_r_gate = amar_skinny_reduce[type_of(p_gate_l), type_of(gate2_l), 1]
comptime k_hgate = amar_head_gate_mul_cast[type_of(aoflat_l), type_of(gate_l), type_of(aoflat_l)]
comptime k_o = amar_matmul_skinny_q8row[4, 1, type_of(aob_l), type_of(q_o), type_of(s_o), type_of(p_h_l)]
comptime k_r_add = amar_skinny_reduce_add[type_of(p_h_l), type_of(x_l), 1]
comptime k_ffn = amar_matmul_skinny_q8row[4, 1, type_of(xb_l), type_of(q_ffn), type_of(s_ffn), type_of(p_ffn_l)]
comptime k_r_gelu = amar_skinny_reduce_gelu_par_bf16[type_of(p_ffn_l), type_of(fgb_l), 1]
comptime k_down = amar_matmul_skinny_q8row[4, 1, type_of(fgb_l), type_of(q_down), type_of(s_down), type_of(p_h_l)]
comptime k_head = amar_matmul_skinny_q8row[4, 1, type_of(xb_l), type_of(q_out), type_of(s_out), type_of(p_v_l)]
comptime k_r_head = amar_skinny_reduce[type_of(p_v_l), type_of(logits_l), 1]
comptime k_argmax = amar_argmax_pos[type_of(logits_l), type_of(toks_l)]


def wq[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[DType.int8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.int8](ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.int8]](), n, owning=False)
    return rebind[TileTensor[DType.int8, LT, MutAnyOrigin]](TileTensor(b, lt))


def ws[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](ctx, (wbuf.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](TileTensor(b, lt))


def wf[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[f32]](), n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(b, lt))


def sub_f32[LT: TensorLayout](ctx: DeviceContext, b: DeviceBuffer[f32], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var v = DeviceBuffer[f32](ctx, b.unsafe_ptr() + o, n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(v, lt))


def load_pack(ctx: DeviceContext, packdir: String, mut off: List[Int]) raises -> DeviceBuffer[DType.uint8]:
    var total = 0
    with open(packdir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var n = Int(parts[3])
            var dt = String(parts[1])
            off.append(Int(parts[2]))
            if dt == "f32":
                total += n * 4
            elif dt == "q8":
                total += n + (n // 32) * 2
            else:
                raise Error("bad dtype " + dt)
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    comptime RSPLIT = 4
    var stage0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var stage1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t_load = perf_counter_ns()
    with open(packdir + "/pack.bin", "r") as f:
        var fd = f._get_raw_fd()
        var done = 0
        var flip = False
        while done < total:
            var want = min(CHUNK, total - done)
            var stage = stage1 if flip else stage0
            var sptr = stage.unsafe_ptr()
            var rerr = List[Int64](unsafe_uninit_length=RSPLIT)
            var rerr_ptr = rerr.unsafe_ptr()
            def rchunk(t: Int) {imm fd, imm want, imm done, imm sptr, imm rerr_ptr}:
                var lo = want * t // RSPLIT
                var hi = want * (t + 1) // RSPLIT
                var got = lo
                while got < hi:
                    var n = external_call["pread", c_ssize_t](fd, sptr.unsafe_offset(got), hi - got, Int64(done + got))
                    if n <= 0:
                        rerr_ptr[t] = 1
                        return
                    got += Int(n)
                rerr_ptr[t] = 0
            parallelize(rchunk, RSPLIT)
            for t in range(RSPLIT):
                if rerr[t] != 0:
                    raise Error("short read")
            ctx.synchronize()
            var dslice = DeviceBuffer[DType.uint8](ctx, wbuf.unsafe_ptr() + done, want, owning=False)
            var hslice = stage.create_sub_buffer[DType.uint8](0, want) if want != CHUNK else stage
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            done += want
            flip = not flip
    ctx.synchronize()
    print("pack loaded", total, "bytes in", Float64(perf_counter_ns() - t_load) / 1e9, "s")
    return wbuf^


def read_prompt(path: String) raises -> List[Int]:
    var prompt = List[Int]()
    with open(path, "r") as f:
        var data = f.read_bytes()
        var val = 0
        var have = False
        for i in range(len(data)):
            var b = Int(data[i])
            if b >= 48 and b <= 57:
                val = val * 10 + (b - 48)
                have = True
            else:
                if have:
                    prompt.append(val)
                val = 0
                have = False
        if have:
            prompt.append(val)
    return prompt^


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/spark/pack-q8")
    var gen_n = atol(getenv("BARO_GEN", "64"))
    var prompt = read_prompt(getenv("BARO_PROMPT", packdir + "/prompt-tokens.txt"))
    var n_prompt = len(prompt)
    var n_total = n_prompt + gen_n
    if n_total > TMAX:
        raise Error("prompt + gen exceeds TMAX")
    print("prompt tokens:", n_prompt, "gen:", gen_n)

    var off = List[Int]()
    var wbuf = load_pack(ctx, packdir, off)

    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    for i in range(TMAX):
        toks_h[i] = Int32(prompt[i]) if i < n_prompt else Int32(0)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var xb_d = ctx.enqueue_create_buffer[bf16](H)
    var p_qkv_d = ctx.enqueue_create_buffer[f32](QKV)
    var qkv_d = ctx.enqueue_create_buffer[f32](QKV)
    var kc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var vc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var ao_d = ctx.enqueue_create_buffer[f32](QDIM)
    var aob_d = ctx.enqueue_create_buffer[bf16](QDIM)
    var p_gate_d = ctx.enqueue_create_buffer[f32](NQH)
    var gate_d = ctx.enqueue_create_buffer[f32](NQH)
    var p_h_d = ctx.enqueue_create_buffer[f32](H)
    var p_g_d = ctx.enqueue_create_buffer[f32](FFN)
    var p_u_d = ctx.enqueue_create_buffer[f32](FFN)
    var fgb_d = ctx.enqueue_create_buffer[bf16](FFN)
    var p_v_d = ctx.enqueue_create_buffer[f32](VOCAB)
    var logits_d = ctx.enqueue_create_buffer[f32](VOCAB)
    ctx.synchronize()

    var Emb = wf(ctx, wbuf, off[0], VOCAB * H, emb_l)
    var X = TileTensor(x_d, x_l)
    var Xb = TileTensor(xb_d, xb_l)
    var Toks = TileTensor(toks_d, toks_l)
    var Pqkv = TileTensor(p_qkv_d, p_qkv_l)
    var Qkv = TileTensor(qkv_d, qkv_l)
    var Q = sub_f32(ctx, qkv_d, 0, QDIM, q_l)
    var K = sub_f32(ctx, qkv_d, QDIM, KVDIM, kv_l)
    var V = sub_f32(ctx, qkv_d, QDIM + KVDIM, KVDIM, kv_l)
    var Kc = TileTensor(kc_d, cache_l)
    var Vc = TileTensor(vc_d, cache_l)
    var Ao = TileTensor(ao_d, q_l)
    var Aoflat = TileTensor(ao_d, aoflat_l)
    var AoBflat = TileTensor(aob_d, aoflat_l)
    var AoB = TileTensor(aob_d, aob_l)
    var Pgate = TileTensor(p_gate_d, p_gate_l)
    var Gate2 = TileTensor(gate_d, gate2_l)
    var Gate = TileTensor(gate_d, gate_l)
    var Ph = TileTensor(p_h_d, p_h_l)
    var Pg = TileTensor(p_g_d, p_ffn_l)
    var Pu = TileTensor(p_u_d, p_ffn_l)
    var Fgb = TileTensor(fgb_d, fgb_l)
    var Pv = TileTensor(p_v_d, p_v_l)
    var Logits = TileTensor(logits_d, logits_l)
    var OutNorm = wf(ctx, wbuf, off[1 + 8 * N_LAYERS], H, h_l)
    var out_off = off[2 + 8 * N_LAYERS]
    var Woq = wq(ctx, wbuf, out_off, VOCAB * H, q_out)
    var Wos = ws(ctx, wbuf, out_off, VOCAB * H, s_out)

    var t_gen_start: Int = 0
    for pos in range(n_total - 1):
        if pos == n_prompt - 1:
            ctx.synchronize()
            t_gen_start = perf_counter_ns()
        ctx.enqueue_function[k_emb](Emb, X, Toks, Int32(pos), Int32(H), grid_dim=(ceildiv(H, 256), 1), block_dim=256)
        for i in range(N_LAYERS):
            var e = 1 + 8 * i
            var swa = (i % 4) != 3
            var AttnNorm = wf(ctx, wbuf, off[e], H, h_l)
            var FfnNorm = wf(ctx, wbuf, off[e + 4], H, h_l)
            ctx.enqueue_function[k_rms](X, AttnNorm, Xb, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
            ctx.enqueue_function[k_qkv](Xb, wq(ctx, wbuf, off[e + 1], QKV * H, q_qkv), ws(ctx, wbuf, off[e + 1], QKV * H, s_qkv), Pqkv, Int32(1), Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_r_qkv](Pqkv, Qkv, Int32(1), Int32(QKV), grid_dim=ceildiv(QKV, 256), block_dim=256)
            if swa:
                ctx.enqueue_function[k_rope_swa](Q, Int32(pos), Int32(NQH), BASE_SWA, grid_dim=(NQH, 1), block_dim=NROT_SWA // 2)
                ctx.enqueue_function[k_rope_swa_k](K, Int32(pos), Int32(NKVH), BASE_SWA, grid_dim=(NKVH, 1), block_dim=NROT_SWA // 2)
            else:
                ctx.enqueue_function[k_rope_full](Q, Int32(pos), Int32(NQH), BASE_FULL, grid_dim=(NQH, 1), block_dim=NROT_FULL // 2)
                ctx.enqueue_function[k_rope_full_k](K, Int32(pos), Int32(NKVH), BASE_FULL, grid_dim=(NKVH, 1), block_dim=NROT_FULL // 2)
            ctx.enqueue_function[k_append](Kc, K, Int32(pos), Int32(i), grid_dim=(NKVH, 1), block_dim=HD)
            ctx.enqueue_function[k_append](Vc, V, Int32(pos), Int32(i), grid_dim=(NKVH, 1), block_dim=HD)
            ctx.enqueue_function[k_att](Q, Kc, Vc, Ao, Int32(pos + 1), Int32(SWA_WIN if swa else 0), Float32(0.0625), Int32(i), grid_dim=(NQH, 1), block_dim=HD)
            ctx.enqueue_function[k_gate](Xb, wq(ctx, wbuf, off[e + 2], NQH * H, q_gate), ws(ctx, wbuf, off[e + 2], NQH * H, s_gate), Pgate, Int32(1), Int32(NQH), Int32(H), grid_dim=ceildiv(NQH, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_r_gate](Pgate, Gate2, Int32(1), Int32(NQH), grid_dim=1, block_dim=256)
            ctx.enqueue_function[k_hgate](Aoflat, Gate, AoBflat, Int32(QDIM), grid_dim=ceildiv(QDIM, 256), block_dim=256)
            ctx.enqueue_function[k_o](AoB, wq(ctx, wbuf, off[e + 3], H * QDIM, q_o), ws(ctx, wbuf, off[e + 3], H * QDIM, s_o), Ph, Int32(1), Int32(H), Int32(QDIM), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_r_add](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
            ctx.enqueue_function[k_rms](X, FfnNorm, Xb, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
            ctx.enqueue_function[k_ffn](Xb, wq(ctx, wbuf, off[e + 5], FFN * H, q_ffn), ws(ctx, wbuf, off[e + 5], FFN * H, s_ffn), Pg, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_ffn](Xb, wq(ctx, wbuf, off[e + 6], FFN * H, q_ffn), ws(ctx, wbuf, off[e + 6], FFN * H, s_ffn), Pu, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_r_gelu](Pg, Pu, Fgb, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
            ctx.enqueue_function[k_down](Fgb, wq(ctx, wbuf, off[e + 7], H * FFN, q_down), ws(ctx, wbuf, off[e + 7], H * FFN, s_down), Ph, Int32(1), Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_r_add](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        if pos >= n_prompt - 1:
            ctx.enqueue_function[k_rms](X, OutNorm, Xb, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
            ctx.enqueue_function[k_head](Xb, Woq, Wos, Pv, Int32(1), Int32(VOCAB), Int32(H), grid_dim=ceildiv(VOCAB, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_r_head](Pv, Logits, Int32(1), Int32(VOCAB), grid_dim=ceildiv(VOCAB, 256), block_dim=256)
            ctx.enqueue_function[k_argmax](Logits, Toks, Int32(VOCAB), Int32(pos + 1), grid_dim=1, block_dim=256)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t_gen_start) / 1e9
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
    ctx.synchronize()
    var s = String("")
    for i in range(n_prompt, n_total):
        s += String(Int(toks_h[i])) + " "
    print("generated:", s)
    print("tok/s_gen:", Float64(gen_n) / dt, "(", gen_n, "steps,", dt, "s )")
