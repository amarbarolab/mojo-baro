"""Stage 1+2 gate of bench/megakernel-protocol.md.

A synthetic 4-layer pack (ssm, ssm, ssm, attn; ffn after each) in one
device byte buffer with an engine-style offset table. The launch path runs
the engine's per-layer kernel sequence; amar_mega_token runs the same four
layers in one persistent launch at G=96. Gates: residual X, conv windows,
ssm states and the KV cache bit-identical; then per-token time of both.
"""
from std.math import ceildiv
from std.memory import bitcast
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import amar_rmsnorm_cast
from matmul_skinny import amar_matmul_skinny_q8row, amar_skinny_reduce, amar_skinny_reduce_add, amar_skinny_reduce_swiglu_bf16, ROW_WAVES, ROW_THREADS, SM, SPLITK
from ssm import amar_ssm_reduce_gates, amar_ssm_conv, amar_ssm_qk_l2norm, amar_ssm_delta_step, amar_ssm_gated_out_bf16, CONV, NH_V, SSTATE
from attn import amar_head_rmsnorm, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split, amar_rope_yarn, amar_kv_append, HD, NQH, NKVH
from mega import amar_mega_token, MEGA_G, H, FFN, QF, KV

comptime NL = 4
comptime N_SSM_T = 3
comptime N_ATT_T = 1
comptime SLOTS = 2
comptime TM = 64
comptime POS = 9
comptime ITERS = 100
comptime u8 = DType.uint8
comptime u32 = DType.uint32
comptime i64 = DType.int64
comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8

comptime xm_layout = row_major[1, H]()
comptime h_layout = row_major[H]()
comptime hd_layout = row_major[HD]()
comptime qfm_layout = row_major[1, QF]()
comptime convm_layout = row_major[1, CONV]()
comptime g32_layout = row_major[NH_V]()
comptime g32m_layout = row_major[1, NH_V]()
comptime om_layout = row_major[1, NH_V, SSTATE]()
comptime cw_layout = row_major[CONV, 4]()
comptime n128_layout = row_major[SSTATE]()
comptime csall_layout = row_major[SLOTS, N_SSM_T, 3, CONV]()
comptime ssall_layout = row_major[SLOTS, N_SSM_T, NH_V, SSTATE, SSTATE]()
comptime kvm_flat = row_major[1, KV]()
comptime kvm_layout = row_major[NKVH, HD]()
comptime qm_layout = row_major[NQH, HD]()
comptime xflat_layout = row_major[H]()
comptime cache_layout = row_major[NKVH, TM, HD]()
comptime ffnm_layout = row_major[1, FFN]()
comptime ffn1_layout = row_major[1, FFN]()
comptime ctr_layout = row_major[3]()
comptime q_h_qf = row_major[QF, H]()
comptime s_h_qf = row_major[QF, H // 32]()
comptime q_h_h = row_major[H, H]()
comptime s_h_h = row_major[H, H // 32]()
comptime q_h_kv = row_major[KV, H]()
comptime s_h_kv = row_major[KV, H // 32]()
comptime q_h_32 = row_major[NH_V, H]()
comptime s_h_32 = row_major[NH_V, H // 32]()
comptime q_conv_h = row_major[CONV, H]()
comptime s_conv_h = row_major[CONV, H // 32]()
comptime q_h_ffn = row_major[FFN, H]()
comptime s_h_ffn = row_major[FFN, H // 32]()
comptime q_ffn_h = row_major[H, FFN]()
comptime s_ffn_h = row_major[H, FFN // 32]()
comptime p_qf = row_major[SPLITK, SM, QF]()
comptime p_kv = row_major[SPLITK, SM, KV]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime p_32 = row_major[SPLITK, SM, NH_V]()
comptime p_ffn = row_major[SPLITK, SM, FFN]()

comptime XL = type_of(xm_layout)
comptime rmsc_k = amar_rmsnorm_cast[XL, type_of(h_layout), XL]
comptime g_conv = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_conv_h), type_of(s_conv_h), type_of(p_qf)]
comptime g_qf = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_qf), type_of(s_h_qf), type_of(p_qf)]
comptime g_kv = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_kv), type_of(s_h_kv), type_of(p_kv)]
comptime g_h = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_h), type_of(s_h_h), type_of(p_h)]
comptime g_ab = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_32), type_of(s_h_32), type_of(p_32)]
comptime g_ffn = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_ffn), type_of(s_h_ffn), type_of(p_ffn)]
comptime g_down = amar_matmul_skinny_q8row[4, 1, type_of(ffnm_layout), type_of(q_ffn_h), type_of(s_ffn_h), type_of(p_h)]
comptime r_conv = amar_skinny_reduce[type_of(p_qf), type_of(convm_layout), 1]
comptime r_qf = amar_skinny_reduce[type_of(p_qf), type_of(qfm_layout), 1]
comptime r_kv = amar_skinny_reduce[type_of(p_kv), type_of(kvm_flat), 1]
comptime r_h = amar_skinny_reduce[type_of(p_h), XL, 1]
comptime r_add = amar_skinny_reduce_add[type_of(p_h), XL, 1]
comptime r_swiglu = amar_skinny_reduce_swiglu_bf16[type_of(p_ffn), type_of(ffnm_layout), 1]
comptime rgates_k = amar_ssm_reduce_gates[type_of(p_32), type_of(g32m_layout), type_of(g32_layout)]
comptime conv_k = amar_ssm_conv[type_of(convm_layout), type_of(csall_layout), type_of(cw_layout), type_of(convm_layout)]
comptime l2_k = amar_ssm_qk_l2norm[type_of(convm_layout)]
comptime delta_k = amar_ssm_delta_step[1, type_of(ssall_layout), type_of(convm_layout), type_of(g32m_layout), type_of(om_layout)]
comptime gated_k = amar_ssm_gated_out_bf16[type_of(om_layout), XL, type_of(n128_layout), XL]
comptime split_k = amar_qgate_split[type_of(qfm_layout), type_of(qm_layout), type_of(xflat_layout)]
comptime hrms_q = amar_head_rmsnorm[type_of(qm_layout), type_of(hd_layout)]
comptime hrms_kv = amar_head_rmsnorm[type_of(kvm_layout), type_of(hd_layout)]
comptime rope_q = amar_rope_yarn[type_of(qm_layout)]
comptime rope_k = amar_rope_yarn[type_of(kvm_layout)]
comptime append_k = amar_kv_append[type_of(cache_layout), type_of(kvm_layout)]
comptime att_k = amar_attn_decode[type_of(qm_layout), type_of(cache_layout), type_of(qm_layout)]
comptime gmul_k = amar_gate_mul_cast[type_of(xflat_layout), type_of(xflat_layout), type_of(xflat_layout)]
comptime mega_k = amar_mega_token[
    XL, XL,
    type_of(convm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout),
    type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(xflat_layout),
    type_of(ffn1_layout), type_of(ffnm_layout), type_of(row_major[64]()), type_of(ctr_layout),
    TM, NL,
]


def hash01(i: Int, salt: Int) -> Float32:
    var x = UInt32(i * 2654435761 + salt * 40503 + 12345)
    x ^= x >> 13
    x *= UInt32(0x5bd1e995)
    x ^= x >> 15
    return Float32(Float64(x & 0xFFFFFF) / 16777216.0)


def q8_bytes(n: Int, k: Int) -> Int:
    return n * k + (n * k // 32) * 2


def put_q8(h: HostBuffer[u8], o: Int, n: Int, k: Int, salt: Int):
    for i in range(n * k):
        h[o + i] = UInt8(Int(hash01(i, salt) * 255.0) & 0xFF)
    var so = o + n * k
    for i in range(n * k // 32):
        var v = Scalar[f16](0.002 + 0.004 * hash01(i, salt + 7))
        var b = bitcast[DType.uint16, 1](SIMD[f16, 1](v))[0]
        h[so + 2 * i] = UInt8(b & 0xFF)
        h[so + 2 * i + 1] = UInt8((b >> 8) & 0xFF)


def put_f32(h: HostBuffer[u8], o: Int, n: Int, salt: Int, lo: Float32, hi: Float32):
    for i in range(n):
        var v = lo + (hi - lo) * hash01(i, salt)
        var b = bitcast[DType.uint32, 1](SIMD[f32, 1](v))[0]
        h[o + 4 * i] = UInt8(b & 0xFF)
        h[o + 4 * i + 1] = UInt8((b >> 8) & 0xFF)
        h[o + 4 * i + 2] = UInt8((b >> 16) & 0xFF)
        h[o + 4 * i + 3] = UInt8((b >> 24) & 0xFF)


def fill_f32(ctx: DeviceContext, n: Int, salt: Int, lo: Float32, hi: Float32) raises -> DeviceBuffer[f32]:
    var h = ctx.enqueue_create_host_buffer[f32](n)
    ctx.synchronize()
    for i in range(n):
        h[i] = lo + (hi - lo) * hash01(i, salt)
    var d = ctx.enqueue_create_buffer[f32](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def clone(ctx: DeviceContext, src: DeviceBuffer[f32], n: Int) raises -> DeviceBuffer[f32]:
    var d = ctx.enqueue_create_buffer[f32](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=src)
    return d^


def diff(ctx: DeviceContext, name: String, a: DeviceBuffer[f32], b: DeviceBuffer[f32], n: Int) raises -> Int:
    var ha = ctx.enqueue_create_host_buffer[f32](n)
    var hb = ctx.enqueue_create_host_buffer[f32](n)
    ctx.enqueue_copy(dst_buf=ha, src_buf=a)
    ctx.enqueue_copy(dst_buf=hb, src_buf=b)
    ctx.synchronize()
    var bad = 0
    var maxd: Float32 = 0
    for i in range(n):
        if ha[i] != hb[i]:
            bad += 1
            var d = abs(ha[i] - hb[i])
            if d > maxd:
                maxd = d
    print(name, " mismatches:", bad, "/", n, " max|d|=", maxd, " sample:", ha[0], ha[1], ha[n - 1])
    return bad


def tq[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[i8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[i8](ctx, (w.unsafe_ptr() + o).unsafe_bitcast[Scalar[i8]](), n, owning=False)
    return rebind[TileTensor[i8, LT, MutAnyOrigin]](TileTensor(b, lt))


def ts[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[f16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f16](ctx, (w.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[f16]](), n // 32, owning=False)
    return rebind[TileTensor[f16, LT, MutAnyOrigin]](TileTensor(b, lt))


def tf[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, (w.unsafe_ptr() + o).unsafe_bitcast[Scalar[f32]](), n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(b, lt))


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var off = List[Int]()
    var kinds = List[Int]()
    var nrows = List[Int]()
    var ncols = List[Int]()
    var cursor = 0
    off.append(0)
    kinds.append(-1)
    nrows.append(0)
    ncols.append(0)

    @parameter
    def add(kind: Int, n: Int, k: Int):
        off.append(cursor)
        kinds.append(kind)
        nrows.append(n)
        ncols.append(k)
        if kind == 1:
            cursor += q8_bytes(n, k)
        else:
            cursor += n * 4

    for layer in range(NL):
        add(0, H, 0)
        if (layer + 1) % 4 == 0:
            add(1, QF, H)
            add(1, KV, H)
            add(1, KV, H)
            add(0, HD, 0)
            add(0, HD, 0)
            add(1, H, H)
        else:
            add(1, CONV, H)
            add(1, H, H)
            add(1, NH_V, H)
            add(1, NH_V, H)
            add(0, CONV * 4, 0)
            add(0, NH_V, 0)
            add(0, NH_V, 0)
            add(0, SSTATE, 0)
            add(1, H, H)
        add(0, H, 0)
        add(1, FFN, H)
        add(1, FFN, H)
        add(1, H, FFN)
    print("synthetic pack:", cursor, "bytes,", len(off), "entries")

    var wh = ctx.enqueue_create_host_buffer[u8](cursor)
    var offh = ctx.enqueue_create_host_buffer[i64](64)
    ctx.synchronize()
    for i in range(64):
        offh[i] = 0
    for e in range(len(off)):
        offh[e] = Int64(off[e])
        if kinds[e] == 1:
            put_q8(wh, off[e], nrows[e], ncols[e], 100 + e)
        elif kinds[e] == 0:
            var n = nrows[e]
            if n == CONV * 4:
                put_f32(wh, off[e], n, 100 + e, -0.5, 0.5)
            elif n == NH_V and e % 2 == 0:
                put_f32(wh, off[e], n, 100 + e, -2.0, -0.1)
            elif n == NH_V:
                put_f32(wh, off[e], n, 100 + e, -1.0, 1.0)
            else:
                put_f32(wh, off[e], n, 100 + e, 0.5, 1.5)
    var wbuf = ctx.enqueue_create_buffer[u8](cursor)
    var offd = ctx.enqueue_create_buffer[i64](64)
    ctx.enqueue_copy(dst_buf=wbuf, src_buf=wh)
    ctx.enqueue_copy(dst_buf=offd, src_buf=offh)
    ctx.synchronize()

    var x0 = fill_f32(ctx, H, 1, -1.0, 1.0)
    var cs0 = fill_f32(ctx, SLOTS * N_SSM_T * 3 * CONV, 12, -0.3, 0.3)
    var ss0 = fill_f32(ctx, SLOTS * N_SSM_T * NH_V * SSTATE * SSTATE, 13, -0.1, 0.1)
    var kc0 = fill_f32(ctx, N_ATT_T * NKVH * TM * HD, 14, -1.0, 1.0)
    var vc0 = fill_f32(ctx, N_ATT_T * NKVH * TM * HD, 15, -1.0, 1.0)
    comptime NCS = SLOTS * N_SSM_T * 3 * CONV
    comptime NSS = SLOTS * N_SSM_T * NH_V * SSTATE * SSTATE
    comptime NKC = N_ATT_T * NKVH * TM * HD
    var xL = clone(ctx, x0, H)
    var xM = clone(ctx, x0, H)
    var csL = clone(ctx, cs0, NCS)
    var csM = clone(ctx, cs0, NCS)
    var ssL = clone(ctx, ss0, NSS)
    var ssM = clone(ctx, ss0, NSS)
    var kcL = clone(ctx, kc0, NKC)
    var kcM = clone(ctx, kc0, NKC)
    var vcL = clone(ctx, vc0, NKC)
    var vcM = clone(ctx, vc0, NKC)

    var curb_d = ctx.enqueue_create_buffer[bf16](H)
    var resb_d = ctx.enqueue_create_buffer[bf16](H)
    var qkv_d = ctx.enqueue_create_buffer[f32](CONV)
    var z_d = ctx.enqueue_create_buffer[f32](H)
    var araw_d = ctx.enqueue_create_buffer[f32](NH_V)
    var braw_d = ctx.enqueue_create_buffer[f32](NH_V)
    var eg_d = ctx.enqueue_create_buffer[f32](NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](CONV)
    var so_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE)
    var qf_d = ctx.enqueue_create_buffer[f32](QF)
    var k_d = ctx.enqueue_create_buffer[f32](KV)
    var v_d = ctx.enqueue_create_buffer[f32](KV)
    var q_d = ctx.enqueue_create_buffer[f32](H)
    var gate_d = ctx.enqueue_create_buffer[f32](H)
    var ao_d = ctx.enqueue_create_buffer[f32](H)
    var fgb_d = ctx.enqueue_create_buffer[bf16](FFN)
    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_ffn_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_ffn2_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var ctr_d = ctx.enqueue_create_buffer[u32](3)
    var prof_d = ctx.enqueue_create_buffer[i64](16 * NL + 4)
    ctx.enqueue_memset(prof_d, 0)
    ctx.enqueue_memset(p_qf_d, 0)
    ctx.enqueue_memset(p_kv_d, 0)
    ctx.enqueue_memset(p_h_d, 0)
    ctx.enqueue_memset(p_32_d, 0)
    ctx.enqueue_memset(p_32b_d, 0)
    ctx.enqueue_memset(p_ffn_d, 0)
    ctx.enqueue_memset(p_ffn2_d, 0)
    ctx.enqueue_memset(ctr_d, 0)
    ctx.synchronize()

    var CurB = TileTensor(curb_d, xm_layout)
    var ResB = TileTensor(resb_d, xm_layout)
    var Qkvm = TileTensor(qkv_d, convm_layout)
    var Zm = TileTensor(z_d, xm_layout)
    var Araw = TileTensor(araw_d, g32m_layout)
    var Braw = TileTensor(braw_d, g32m_layout)
    var Eg = TileTensor(eg_d, g32m_layout)
    var Beta = TileTensor(beta_d, g32m_layout)
    var Conv = TileTensor(conv_d, convm_layout)
    var So = TileTensor(so_d, om_layout)
    var Qfm = TileTensor(qf_d, qfm_layout)
    var Kflat = TileTensor(k_d, kvm_flat)
    var Khd = TileTensor(k_d, kvm_layout)
    var Vflat = TileTensor(v_d, kvm_flat)
    var Vhd = TileTensor(v_d, kvm_layout)
    var Q = TileTensor(q_d, qm_layout)
    var Gate = TileTensor(gate_d, xflat_layout)
    var Ao = TileTensor(ao_d, qm_layout)
    var Aoflat = TileTensor(ao_d, xflat_layout)
    var AoBflat = TileTensor(resb_d, xflat_layout)
    var FgB = TileTensor(fgb_d, ffnm_layout)
    var Pq = TileTensor(p_qf_d, p_qf)
    var Pkv = TileTensor(p_kv_d, p_kv)
    var Ph = TileTensor(p_h_d, p_h)
    var Pab = TileTensor(p_32_d, p_32)
    var Pab2 = TileTensor(p_32b_d, p_32)
    var Pg = TileTensor(p_ffn_d, p_ffn)
    var Pu = TileTensor(p_ffn2_d, p_ffn)
    var Pg1 = TileTensor(p_ffn_d, ffn1_layout)
    var Pu1 = TileTensor(p_ffn2_d, ffn1_layout)
    var Ctr = TileTensor(ctr_d, ctr_layout)
    var Off = TileTensor(offd, row_major[64]())
    var XL_ = TileTensor(xL, xm_layout)
    var XM_ = TileTensor(xM, xm_layout)
    var CsL = TileTensor(csL, csall_layout)
    var CsM = TileTensor(csM, csall_layout)
    var SsL = TileTensor(ssL, ssall_layout)
    var SsM = TileTensor(ssM, ssall_layout)
    var KcL = TileTensor(kcL, cache_layout)
    var VcL = TileTensor(vcL, cache_layout)

    @parameter
    def launch_path(ring: Int) raises:
        var w = 1
        var ssm_i = 0
        for layer in range(NL):
            ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[w], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
            if (layer + 1) % 4 == 0:
                ctx.enqueue_function[g_qf](CurB, tq(ctx, wbuf, off[w + 1], QF * H, q_h_qf), ts(ctx, wbuf, off[w + 1], QF * H, s_h_qf), Pq, Int32(1), Int32(QF), Int32(H), grid_dim=ceildiv(QF, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_qf](Pq, Qfm, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
                ctx.enqueue_function[g_kv](CurB, tq(ctx, wbuf, off[w + 2], KV * H, q_h_kv), ts(ctx, wbuf, off[w + 2], KV * H, s_h_kv), Pkv, Int32(1), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
                ctx.enqueue_function[g_kv](CurB, tq(ctx, wbuf, off[w + 3], KV * H, q_h_kv), ts(ctx, wbuf, off[w + 3], KV * H, s_h_kv), Pkv, Int32(1), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_kv](Pkv, Vflat, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
                ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, 1), block_dim=HD)
                ctx.enqueue_function[hrms_q](Q, tf(ctx, wbuf, off[w + 4], HD, hd_layout), Float32(1e-6), grid_dim=NQH, block_dim=HD)
                ctx.enqueue_function[hrms_kv](Khd, tf(ctx, wbuf, off[w + 5], HD, hd_layout), Float32(1e-6), grid_dim=NKVH, block_dim=HD)
                ctx.enqueue_function[rope_q](Q, Int32(POS), Int32(NQH), grid_dim=(NQH, 1), block_dim=32)
                ctx.enqueue_function[rope_k](Khd, Int32(POS), Int32(NKVH), grid_dim=(NKVH, 1), block_dim=32)
                ctx.enqueue_function[append_k](KcL, Khd, Int32(POS), grid_dim=(NKVH, 1), block_dim=HD)
                ctx.enqueue_function[append_k](VcL, Vhd, Int32(POS), grid_dim=(NKVH, 1), block_dim=HD)
                ctx.enqueue_function[att_k](Q, KcL, VcL, Ao, Int32(POS + 1), Float32(0.0625), grid_dim=(NQH, 1), block_dim=HD)
                ctx.enqueue_function[gmul_k](Aoflat, Gate, AoBflat, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ctx.enqueue_function[g_h](ResB, tq(ctx, wbuf, off[w + 6], H * H, q_h_h), ts(ctx, wbuf, off[w + 6], H * H, s_h_h), Ph, Int32(1), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_add](Ph, XL_, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                w += 7
            else:
                ctx.enqueue_function[g_conv](CurB, tq(ctx, wbuf, off[w + 1], CONV * H, q_conv_h), ts(ctx, wbuf, off[w + 1], CONV * H, s_conv_h), Pq, Int32(1), Int32(CONV), Int32(H), grid_dim=ceildiv(CONV, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[g_h](CurB, tq(ctx, wbuf, off[w + 2], H * H, q_h_h), ts(ctx, wbuf, off[w + 2], H * H, s_h_h), Ph, Int32(1), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[g_ab](CurB, tq(ctx, wbuf, off[w + 3], NH_V * H, q_h_32), ts(ctx, wbuf, off[w + 3], NH_V * H, s_h_32), Pab, Int32(1), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[g_ab](CurB, tq(ctx, wbuf, off[w + 4], NH_V * H, q_h_32), ts(ctx, wbuf, off[w + 4], NH_V * H, s_h_32), Pab2, Int32(1), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_conv](Pq, Qkvm, Int32(1), Int32(CONV), grid_dim=ceildiv(CONV, 256), block_dim=256)
                ctx.enqueue_function[r_h](Ph, Zm, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, tf(ctx, wbuf, off[w + 6], NH_V, g32_layout), tf(ctx, wbuf, off[w + 7], NH_V, g32_layout), Int32(1), grid_dim=1, block_dim=NH_V)
                ctx.enqueue_function[conv_k](Qkvm, CsL, tf(ctx, wbuf, off[w + 5], CONV * 4, cw_layout), Conv, Int32(ring), Int32(ssm_i), Int32(SLOTS), Int32(1), grid_dim=ceildiv(CONV, 256), block_dim=256)
                ctx.enqueue_function[l2_k](Conv, Int32(1), grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[delta_k](SsL, Conv, Eg, Beta, So, Int32(ring), Int32(ssm_i), Int32(SLOTS), grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[gated_k](So, Zm, tf(ctx, wbuf, off[w + 8], SSTATE, n128_layout), ResB, Int32(1), grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[g_h](ResB, tq(ctx, wbuf, off[w + 9], H * H, q_h_h), ts(ctx, wbuf, off[w + 9], H * H, s_h_h), Ph, Int32(1), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_add](Ph, XL_, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ssm_i += 1
                w += 10
            ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[w], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
            ctx.enqueue_function[g_ffn](CurB, tq(ctx, wbuf, off[w + 1], FFN * H, q_h_ffn), ts(ctx, wbuf, off[w + 1], FFN * H, s_h_ffn), Pg, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[g_ffn](CurB, tq(ctx, wbuf, off[w + 2], FFN * H, q_h_ffn), ts(ctx, wbuf, off[w + 2], FFN * H, s_h_ffn), Pu, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[r_swiglu](Pg, Pu, FgB, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
            ctx.enqueue_function[g_down](FgB, tq(ctx, wbuf, off[w + 3], H * FFN, q_ffn_h), ts(ctx, wbuf, off[w + 3], H * FFN, s_ffn_h), Ph, Int32(1), Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[r_add](Ph, XL_, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
            w += 4

    @parameter
    def mega_path(ring: Int) raises:
        ctx.enqueue_function[mega_k](
            wbuf.unsafe_ptr(), Off, XM_, CurB, ResB, Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, CsM, SsM,
            Qfm, Kflat, Vflat, Q, Gate, Ao, kcM.unsafe_ptr(), vcM.unsafe_ptr(), Pg1, Pu1, FgB, Ctr, prof_d.unsafe_ptr(), prof_d.unsafe_ptr().unsafe_bitcast[Scalar[f32]](),
            Int32(ring), Int32(SLOTS), Int32(POS), Int32(0), grid_dim=MEGA_G, block_dim=ROW_THREADS,
        )

    launch_path(0)
    ctx.synchronize()
    mega_path(0)
    ctx.synchronize()
    var flag = ctx.enqueue_create_host_buffer[u32](3)
    ctx.enqueue_copy(dst_buf=flag, src_buf=ctr_d)
    ctx.synchronize()
    if flag[2] != 0:
        print("FAIL: megakernel barrier NOT-RESIDENT at G=", MEGA_G)
        return
    var bad = diff(ctx, "residual X", xL, xM, H)
    bad += diff(ctx, "conv windows", csL, csM, NCS)
    bad += diff(ctx, "ssm states", ssL, ssM, NSS)
    bad += diff(ctx, "K cache", kcL, kcM, NKC)
    bad += diff(ctx, "V cache", vcL, vcM, NKC)
    if bad != 0:
        print("FAIL: megakernel token differs from launch path")
        return
    print("PASS: per-token megakernel (4 layers: ssm,ssm,ssm,attn + ffn) bit-identical to the launch path (m=1)")

    for it in range(10):
        launch_path(it % SLOTS)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for it in range(ITERS):
        launch_path(it % SLOTS)
    ctx.synchronize()
    var us_launch = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
    for it in range(10):
        mega_path(it % SLOTS)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for it in range(ITERS):
        mega_path(it % SLOTS)
    ctx.synchronize()
    var us_mega = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=flag, src_buf=ctr_d)
    ctx.synchronize()
    var ph = ctx.enqueue_create_host_buffer[i64](16 * NL + 4)
    ctx.enqueue_copy(dst_buf=ph, src_buf=prof_d)
    ctx.synchronize()
    for layer in range(NL):
        var line = String("  layer ") + String(layer) + (" attn" if (layer + 1) % 4 == 0 else " ssm ") + " phases us:"
        var nph = 5 if (layer + 1) % 4 == 0 else 6
        for k in range(1, nph + 1):
            line += " " + String(Float64(ph[16 * layer + k] - ph[16 * layer + k - 1]) / 100.0)
        line += " | tail " + String(Float64(ph[16 * layer + 7] - ph[16 * layer + nph]) / 100.0)
        line += " | ffn:"
        for k in range(8, 11):
            line += " " + String(Float64(ph[16 * layer + k] - ph[16 * layer + k - 1]) / 100.0)
        line += " tail " + String(Float64(ph[16 * layer + 11] - ph[16 * layer + 10]) / 100.0)
        if (layer + 1) % 4 != 0:
            line += " | pre-rmsc " + String(Float64(ph[16 * layer + 12] - ph[16 * layer]) / 100.0)
        line += " | sub " + String(Float64(ph[16 * layer + 7] - ph[16 * layer]) / 100.0) + " ffn " + String(Float64(ph[16 * layer + 11] - ph[16 * layer + 7]) / 100.0)
        print(line)
    print("4-layer us/token: launch=", us_launch, " mega(G=", MEGA_G, ")=", us_mega, " ratio=", us_mega / us_launch, " fail=", flag[2])
