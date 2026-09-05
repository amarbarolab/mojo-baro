"""Stage 1 gate of bench/megakernel-protocol.md: one qwen35 ssm layer as a
persistent 7-phase kernel (kernels/mega.mojo, G=96) vs the engine's
14-launch sequence built from the same kernel bodies.

Gates: residual X, new conv window and new recurrent state bit-identical to
the launch path; then per-layer time of both, device-synchronised, 200 iters.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import amar_rmsnorm_cast
from matmul_skinny import amar_matmul_skinny_q8row, amar_skinny_reduce, amar_skinny_reduce_add, ROW_WAVES, ROW_THREADS, SM, SPLITK
from ssm import amar_ssm_reduce_gates, amar_ssm_conv, amar_ssm_qk_l2norm, amar_ssm_delta_step, amar_ssm_gated_out_bf16, CONV, NH_V, SSTATE
from mega import amar_mega_ssm_layer, MEGA_G

comptime H = 4096
comptime SLOTS = 2
comptime ITERS = 200
comptime u32 = DType.uint32
comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8

comptime xm_layout = row_major[1, H]()
comptime h_layout = row_major[H]()
comptime qfm_layout = row_major[1, CONV]()
comptime g32_layout = row_major[NH_V]()
comptime g32m_layout = row_major[1, NH_V]()
comptime convm_layout = row_major[1, CONV]()
comptime om_layout = row_major[1, NH_V, SSTATE]()
comptime cw_layout = row_major[CONV, 4]()
comptime n128_layout = row_major[SSTATE]()
comptime csall_layout = row_major[SLOTS, 1, 3, CONV]()
comptime ssall_layout = row_major[SLOTS, 1, NH_V, SSTATE, SSTATE]()
comptime ctr_layout = row_major[3]()
comptime q_h_qf = row_major[CONV, H]()
comptime s_h_qf = row_major[CONV, H // 32]()
comptime q_h_h = row_major[H, H]()
comptime s_h_h = row_major[H, H // 32]()
comptime q_h_32 = row_major[NH_V, H]()
comptime s_h_32 = row_major[NH_V, H // 32]()
comptime p_qf = row_major[SPLITK, SM, CONV]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime p_32 = row_major[SPLITK, SM, NH_V]()

comptime XL = type_of(xm_layout)
comptime rmsc_k = amar_rmsnorm_cast[XL, type_of(h_layout), XL]
comptime g_qf = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_qf), type_of(s_h_qf), type_of(p_qf)]
comptime g_h = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_h), type_of(s_h_h), type_of(p_h)]
comptime g_ab = amar_matmul_skinny_q8row[4, 1, XL, type_of(q_h_32), type_of(s_h_32), type_of(p_32)]
comptime r_qf = amar_skinny_reduce[type_of(p_qf), type_of(qfm_layout), 1]
comptime r_h = amar_skinny_reduce[type_of(p_h), XL, 1]
comptime r_add = amar_skinny_reduce_add[type_of(p_h), XL, 1]
comptime rgates_k = amar_ssm_reduce_gates[type_of(p_32), type_of(g32m_layout), type_of(g32_layout)]
comptime conv_k = amar_ssm_conv[type_of(qfm_layout), type_of(csall_layout), type_of(cw_layout), type_of(convm_layout)]
comptime l2_k = amar_ssm_qk_l2norm[type_of(convm_layout)]
comptime delta_k = amar_ssm_delta_step[1, type_of(ssall_layout), type_of(convm_layout), type_of(g32m_layout), type_of(om_layout)]
comptime gated_k = amar_ssm_gated_out_bf16[type_of(om_layout), XL, type_of(n128_layout), XL]
comptime mega_k = amar_mega_ssm_layer[
    XL, type_of(h_layout), XL,
    type_of(q_h_qf), type_of(s_h_qf), type_of(q_h_h), type_of(s_h_h), type_of(q_h_32), type_of(s_h_32),
    type_of(cw_layout), type_of(g32_layout), type_of(n128_layout),
    type_of(qfm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout), type_of(ctr_layout),
]


def hash01(i: Int, salt: Int) -> Float32:
    var x = UInt32(i * 2654435761 + salt * 40503 + 12345)
    x ^= x >> 13
    x *= UInt32(0x5bd1e995)
    x ^= x >> 15
    return Float32(Float64(x & 0xFFFFFF) / 16777216.0)


def fill_f32(ctx: DeviceContext, n: Int, salt: Int, lo: Float32, hi: Float32) raises -> DeviceBuffer[f32]:
    var h = ctx.enqueue_create_host_buffer[f32](n)
    ctx.synchronize()
    for i in range(n):
        h[i] = lo + (hi - lo) * hash01(i, salt)
    var d = ctx.enqueue_create_buffer[f32](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def fill_q8q(ctx: DeviceContext, n: Int, k: Int, salt: Int) raises -> DeviceBuffer[i8]:
    var q = ctx.enqueue_create_host_buffer[i8](n * k)
    ctx.synchronize()
    for i in range(n * k):
        q[i] = Scalar[i8](Int(hash01(i, salt) * 255.0) - 127)
    var qd = ctx.enqueue_create_buffer[i8](n * k)
    ctx.enqueue_copy(dst_buf=qd, src_buf=q)
    ctx.synchronize()
    return qd^


def fill_q8s(ctx: DeviceContext, n: Int, k: Int, salt: Int) raises -> DeviceBuffer[f16]:
    var s = ctx.enqueue_create_host_buffer[f16](n * k // 32)
    ctx.synchronize()
    for i in range(n * k // 32):
        s[i] = Scalar[f16](0.002 + 0.004 * hash01(i, salt + 7))
    var sd = ctx.enqueue_create_buffer[f16](n * k // 32)
    ctx.enqueue_copy(dst_buf=sd, src_buf=s)
    ctx.synchronize()
    return sd^


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


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var x0 = fill_f32(ctx, H, 1, -1.0, 1.0)
    var gn = fill_f32(ctx, H, 2, 0.5, 1.5)
    var wqkvq = fill_q8q(ctx, CONV, H, 3)
    var wqkvs = fill_q8s(ctx, CONV, H, 3)
    var wzq = fill_q8q(ctx, H, H, 4)
    var wzs = fill_q8s(ctx, H, H, 4)
    var waq = fill_q8q(ctx, NH_V, H, 5)
    var was = fill_q8s(ctx, NH_V, H, 5)
    var wbq = fill_q8q(ctx, NH_V, H, 6)
    var wbs = fill_q8s(ctx, NH_V, H, 6)
    var wsoq = fill_q8q(ctx, H, H, 7)
    var wsos = fill_q8s(ctx, H, H, 7)
    var cw = fill_f32(ctx, CONV * 4, 8, -0.5, 0.5)
    var ssma = fill_f32(ctx, NH_V, 9, -2.0, -0.1)
    var dtb = fill_f32(ctx, NH_V, 10, -1.0, 1.0)
    var nw = fill_f32(ctx, SSTATE, 11, 0.5, 1.5)
    var cs0 = fill_f32(ctx, SLOTS * 3 * CONV, 12, -0.3, 0.3)
    var ss0 = fill_f32(ctx, SLOTS * NH_V * SSTATE * SSTATE, 13, -0.1, 0.1)

    var xL = clone(ctx, x0, H)
    var xM = clone(ctx, x0, H)
    var csL = clone(ctx, cs0, SLOTS * 3 * CONV)
    var csM = clone(ctx, cs0, SLOTS * 3 * CONV)
    var ssL = clone(ctx, ss0, SLOTS * NH_V * SSTATE * SSTATE)
    var ssM = clone(ctx, ss0, SLOTS * NH_V * SSTATE * SSTATE)

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
    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * CONV)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var ctr_d = ctx.enqueue_create_buffer[u32](3)
    ctx.enqueue_memset(p_qf_d, 0)
    ctx.enqueue_memset(p_h_d, 0)
    ctx.enqueue_memset(p_32_d, 0)
    ctx.enqueue_memset(p_32b_d, 0)
    ctx.enqueue_memset(ctr_d, 0)
    ctx.synchronize()

    var Gn = TileTensor(gn, h_layout)
    var CurB = TileTensor(curb_d, xm_layout)
    var ResB = TileTensor(resb_d, xm_layout)
    var Wqkvq = TileTensor(wqkvq, q_h_qf)
    var Wqkvs = TileTensor(wqkvs, s_h_qf)
    var Wzq = TileTensor(wzq, q_h_h)
    var Wzs = TileTensor(wzs, s_h_h)
    var Waq = TileTensor(waq, q_h_32)
    var Was = TileTensor(was, s_h_32)
    var Wbq = TileTensor(wbq, q_h_32)
    var Wbs = TileTensor(wbs, s_h_32)
    var Wsoq = TileTensor(wsoq, q_h_h)
    var Wsos = TileTensor(wsos, s_h_h)
    var Cw = TileTensor(cw, cw_layout)
    var SsmA = TileTensor(ssma, g32_layout)
    var DtB = TileTensor(dtb, g32_layout)
    var Nw = TileTensor(nw, n128_layout)
    var Qkvm = TileTensor(qkv_d, qfm_layout)
    var Zm = TileTensor(z_d, xm_layout)
    var Araw = TileTensor(araw_d, g32m_layout)
    var Braw = TileTensor(braw_d, g32m_layout)
    var Eg = TileTensor(eg_d, g32m_layout)
    var Beta = TileTensor(beta_d, g32m_layout)
    var Conv = TileTensor(conv_d, convm_layout)
    var So = TileTensor(so_d, om_layout)
    var Pq = TileTensor(p_qf_d, p_qf)
    var Ph = TileTensor(p_h_d, p_h)
    var Pab = TileTensor(p_32_d, p_32)
    var Pab2 = TileTensor(p_32b_d, p_32)
    var Ctr = TileTensor(ctr_d, ctr_layout)
    var XL_ = TileTensor(xL, xm_layout)
    var XM_ = TileTensor(xM, xm_layout)
    var CsL = TileTensor(csL, csall_layout)
    var CsM = TileTensor(csM, csall_layout)
    var SsL = TileTensor(ssL, ssall_layout)
    var SsM = TileTensor(ssM, ssall_layout)

    @parameter
    def launch_path(ring: Int) raises:
        ctx.enqueue_function[rmsc_k](XL_, Gn, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        ctx.enqueue_function[g_qf](CurB, Wqkvq, Wqkvs, Pq, Int32(1), Int32(CONV), Int32(H), grid_dim=ceildiv(CONV, ROW_WAVES), block_dim=ROW_THREADS)
        ctx.enqueue_function[g_h](CurB, Wzq, Wzs, Ph, Int32(1), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
        ctx.enqueue_function[g_ab](CurB, Waq, Was, Pab, Int32(1), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
        ctx.enqueue_function[g_ab](CurB, Wbq, Wbs, Pab2, Int32(1), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
        ctx.enqueue_function[r_qf](Pq, Qkvm, Int32(1), Int32(CONV), grid_dim=ceildiv(CONV, 256), block_dim=256)
        ctx.enqueue_function[r_h](Ph, Zm, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, SsmA, DtB, Int32(1), grid_dim=1, block_dim=NH_V)
        ctx.enqueue_function[conv_k](Qkvm, CsL, Cw, Conv, Int32(ring), Int32(0), Int32(SLOTS), Int32(1), grid_dim=ceildiv(CONV, 256), block_dim=256)
        ctx.enqueue_function[l2_k](Conv, Int32(1), grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[delta_k](SsL, Conv, Eg, Beta, So, Int32(ring), Int32(0), Int32(SLOTS), grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[gated_k](So, Zm, Nw, ResB, Int32(1), grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[g_h](ResB, Wsoq, Wsos, Ph, Int32(1), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
        ctx.enqueue_function[r_add](Ph, XL_, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)

    @parameter
    def mega_path(ring: Int) raises:
        ctx.enqueue_function[mega_k](
            XM_, Gn, CurB, Wqkvq, Wqkvs, Wzq, Wzs, Waq, Was, Wbq, Wbs, Cw, SsmA, DtB, Nw, Wsoq, Wsos,
            Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, ResB, CsM, SsM, Ctr,
            Int32(ring), Int32(0), Int32(SLOTS), grid_dim=MEGA_G, block_dim=ROW_THREADS,
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
    bad += diff(ctx, "conv window", csL, csM, SLOTS * 3 * CONV)
    bad += diff(ctx, "ssm state", ssL, ssM, SLOTS * NH_V * SSTATE * SSTATE)
    if bad != 0:
        print("FAIL: megakernel ssm layer differs from launch path")
        return
    print("PASS: megakernel ssm layer bit-identical to the 14-launch path (m=1)")

    for it in range(20):
        launch_path(it % SLOTS)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for it in range(ITERS):
        launch_path(it % SLOTS)
    ctx.synchronize()
    var us_launch = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
    for it in range(20):
        mega_path(it % SLOTS)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for it in range(ITERS):
        mega_path(it % SLOTS)
    ctx.synchronize()
    var us_mega = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=flag, src_buf=ctr_d)
    ctx.synchronize()
    print("ssm layer us/iter: launch(14)=", us_launch, " mega(G=", MEGA_G, ")=", us_mega, " ratio=", us_mega / us_launch, " fail=", flag[2])
