"""baro window step: everything the engine does per decode window -- prefill
chunk or MTP draft, embed, the 32-block hybrid stack (launch path or the
persistent megakernel), head, verify -- over buffers the harness owns.

This file is embedded in BARO ggufs (`baro.kernel.files`) and is what the
self-optimising loop may edit. The stopwatch, the prints and the fixtures
live in serve/engine.mojo, which is not embedded: a candidate cannot reach
t0, t_prefill_end or dt from here (exchange/scorer-integrity-report.md, P-A).
"""
from std.math import ceildiv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *


def is_attn(i: Int) -> Bool:
    return (i + 1) % 4 == 0


def push(mut off: List[Int], mut cursor: Int, n_bytes: Int):
    off.append(cursor)
    cursor += n_bytes


def wbf16(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int
) -> DeviceBuffer[DType.bfloat16]:
    return DeviceBuffer[DType.bfloat16](
        ctx,
        (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.bfloat16]](),
        n, owning=False,
    )


def wf32(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int
) -> DeviceBuffer[DType.float32]:
    return DeviceBuffer[DType.float32](
        ctx,
        (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.float32]](),
        n, owning=False,
    )


def tens_bf16[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.bfloat16, LT, MutAnyOrigin]:
    var b = wbf16(ctx, wbuf, o, n)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.bfloat16, LT, MutAnyOrigin]](t)


def tens_q8q[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.int8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.int8](
        ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.int8]](), n, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.int8, LT, MutAnyOrigin]](t)


def tens_q8s[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](
        ctx, (wbuf.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](t)


def tens_q4q[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.uint8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.uint8](
        ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.uint8]](), n // 2, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.uint8, LT, MutAnyOrigin]](t)


def tens_q4s[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](
        ctx, (wbuf.unsafe_ptr() + o + n // 2).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](t)


def tens_f32[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float32, LT, MutAnyOrigin]:
    var b = wf32(ctx, wbuf, o, n)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float32, LT, MutAnyOrigin]](t)


def row_f32[
    LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[f32], o: Int, n: Int, lt: LT
) -> TileTensor[f32, LT, MutAnyOrigin]:
    var s = DeviceBuffer[f32](ctx, b.unsafe_ptr() + o, n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](t)


def row_bf16[
    LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[bf16], o: Int, n: Int, lt: LT
) -> TileTensor[bf16, LT, MutAnyOrigin]:
    var s = DeviceBuffer[bf16](ctx, b.unsafe_ptr() + o, n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[bf16, LT, MutAnyOrigin]](t)


def blk32_forward(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], e: Int,
    m: Int, pos: Int, tok_pos: Int, do_head: Bool,
    hsrc: DeviceBuffer[f32], mut x_d: DeviceBuffer[f32], mut curb_d: DeviceBuffer[bf16],
    mut qf_d: DeviceBuffer[f32], mut q_d: DeviceBuffer[f32], mut k_d: DeviceBuffer[f32],
    mut v_d: DeviceBuffer[f32], mut gate_d: DeviceBuffer[f32], mut ao_d: DeviceBuffer[f32],
    mut resb_d: DeviceBuffer[bf16], mut fgb_d: DeviceBuffer[bf16],
    mut p_qf_d: DeviceBuffer[f32], mut p_kv_d: DeviceBuffer[f32], mut p_h_d: DeviceBuffer[f32],
    mut p_ffn_d: DeviceBuffer[f32], mut p_ffn2_d: DeviceBuffer[f32], mut p_v_d: DeviceBuffer[f32],
    mut logits_d: DeviceBuffer[f32], mut cc_d: DeviceBuffer[bf16], mut de_d: DeviceBuffer[f32],
    mut hd_d: DeviceBuffer[f32], mut kc32_d: DeviceBuffer[f32], mut vc32_d: DeviceBuffer[f32],
    mut toks_d: DeviceBuffer[DType.int32], mut dtok_d: DeviceBuffer[DType.int32],
    prof3: Bool, mut p3: List[Int],
    draft_q4: Bool, q4_off: Int, pack_q4: Bool,
) raises:
    # blk.32 (NextN) draft head over m rows: row r is token Toks[tok_pos + r]
    # at sequence position pos + r, paired with hidden row r of hsrc
    # (docs/mtp-notes.md: h from BEFORE that token). Writes the draft's own
    # post-shared-head-norm hidden into hd_d rows and, if do_head, the
    # argmax of every row into dtok_d[r].
    var Embd = tens_bf16(ctx, wbuf, off[0], VOCAB * H, emb_layout)
    var Toks = TileTensor(toks_d, toks_layout)
    var Dtok = TileTensor(dtok_d, dtok_layout)
    var Xm = TileTensor(x_d, xm_layout)
    var CurBm = TileTensor(curb_d, xm_layout)
    var Logitsm = TileTensor(logits_d, vm_layout)
    var DeM = TileTensor(de_d, xm_layout)
    var t3 = 0
    if prof3:
        ctx.synchronize()
        t3 = perf_counter_ns()
    ctx.enqueue_function[embed_k](
        Embd, DeM, Toks, Int32(tok_pos), Int32(H),
        grid_dim=(ceildiv(H, 256), m), block_dim=256,
    )
    var Enorm = tens_f32(ctx, wbuf, off[e + 12], H, h_layout)
    var Hnorm = tens_f32(ctx, wbuf, off[e + 13], H, h_layout)
    for r in range(m):
        var TokEmb = row_f32(ctx, de_d, r * H, H, h2_layout)
        var HnRow = row_f32(ctx, hsrc, r * H, H, h2_layout)
        var CcEmbed = row_bf16(ctx, cc_d, r * QF, H, h2_layout)
        var CcH = row_bf16(ctx, cc_d, r * QF + H, H, h2_layout)
        ctx.enqueue_function[rmsc_h2](TokEmb, Enorm, CcEmbed, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        ctx.enqueue_function[rmsc_h2](HnRow, Hnorm, CcH, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    var CcM = TileTensor(cc_d, qfm_layout)
    var Wehq = tens_q8q(ctx, wbuf, off[e + 11], QF * H, q_qf_h)
    var Wehs = tens_q8s(ctx, wbuf, off[e + 11], QF * H, s_qf_h)
    var PhEh = TileTensor(p_h_d, p_h)
    gemm_w[H, QF](ctx, CcM, wbuf, off[e + 11], pack_q4, PhEh, m)
    ctx.enqueue_function[r_h](PhEh, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

    var AttNorm32 = tens_f32(ctx, wbuf, off[e + 0], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, AttNorm32, CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    var Wqq = tens_q8q(ctx, wbuf, off[e + 1], H * QF, q_h_qf)
    var Wqs = tens_q8s(ctx, wbuf, off[e + 1], H * QF, s_h_qf)
    var Wkq = tens_q8q(ctx, wbuf, off[e + 2], H * KV, q_h_kv)
    var Wks = tens_q8s(ctx, wbuf, off[e + 2], H * KV, s_h_kv)
    var Wvq = tens_q8q(ctx, wbuf, off[e + 3], H * KV, q_h_kv)
    var Wvs = tens_q8s(ctx, wbuf, off[e + 3], H * KV, s_h_kv)
    var Qn = tens_f32(ctx, wbuf, off[e + 4], HD, hd_layout)
    var Kn = tens_f32(ctx, wbuf, off[e + 5], HD, hd_layout)
    var Woq = tens_q8q(ctx, wbuf, off[e + 6], H * H, q_h_h)
    var Wos = tens_q8s(ctx, wbuf, off[e + 6], H * H, s_h_h)
    var Pqf = TileTensor(p_qf_d, p_qf)
    var Pkv = TileTensor(p_kv_d, p_kv)
    var Ph = TileTensor(p_h_d, p_h)
    var Qfm = TileTensor(qf_d, qfm_layout)
    var Q = TileTensor(q_d, qm_layout)
    var Gate = TileTensor(gate_d, xflat_layout)
    var Kflat = TileTensor(k_d, kvm_flat)
    var Khd = TileTensor(k_d, kvm_layout)
    var Vflat = TileTensor(v_d, kvm_flat)
    var Vhd = TileTensor(v_d, kvm_layout)
    var Kc = TileTensor(kc32_d, cache_layout)
    var Vc = TileTensor(vc32_d, cache_layout)
    var Ao = TileTensor(ao_d, qm_layout)
    var Aoflat = TileTensor(ao_d, xflat_layout)
    var AoB = TileTensor(resb_d, xflat_layout)
    var AoBm = TileTensor(resb_d, xm_layout)
    gemm_w[QF, H](ctx, CurBm, wbuf, off[e + 1], pack_q4, Pqf, m)
    ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
    gemm_w[KV, H](ctx, CurBm, wbuf, off[e + 2], pack_q4, Pkv, m)
    ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
    gemm_w[KV, H](ctx, CurBm, wbuf, off[e + 3], pack_q4, Pkv, m)
    ctx.enqueue_function[r_kv](Pkv, Vflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
    ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, m), block_dim=HD)
    ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
    ctx.enqueue_function[hrms_kv](Khd, Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
    ctx.enqueue_function[rope_q](Q, Int32(pos), Int32(NQH), grid_dim=(NQH, m), block_dim=32)
    ctx.enqueue_function[rope_k](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, m), block_dim=32)
    ctx.enqueue_function[append_k](Kc, Khd, Int32(pos), grid_dim=(NKVH, m), block_dim=HD)
    ctx.enqueue_function[append_k](Vc, Vhd, Int32(pos), grid_dim=(NKVH, m), block_dim=HD)
    ctx.enqueue_function[att_k](Q, Kc, Vc, Ao, Int32(pos + 1), Float32(0.0625), grid_dim=(NQH, m), block_dim=HD)
    ctx.enqueue_function[gmul_k](Aoflat, Gate, AoB, Int32(m * H), grid_dim=ceildiv(m * H, 256), block_dim=256)
    gemm_w[H, H](ctx, AoBm, wbuf, off[e + 6], pack_q4, Ph, m)
    ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

    var PostAttnNorm = tens_f32(ctx, wbuf, off[e + 7], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, PostAttnNorm, CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    var Wfgq = tens_q8q(ctx, wbuf, off[e + 8], H * FFN, q_h_ffn)
    var Wfgs = tens_q8s(ctx, wbuf, off[e + 8], H * FFN, s_h_ffn)
    var Wfuq = tens_q8q(ctx, wbuf, off[e + 9], H * FFN, q_h_ffn)
    var Wfus = tens_q8s(ctx, wbuf, off[e + 9], H * FFN, s_h_ffn)
    var Wfdq = tens_q8q(ctx, wbuf, off[e + 10], FFN * H, q_ffn_h)
    var Wfds = tens_q8s(ctx, wbuf, off[e + 10], FFN * H, s_ffn_h)
    var Pg = TileTensor(p_ffn_d, p_ffn)
    var Pu = TileTensor(p_ffn2_d, p_ffn)
    var Ph2 = TileTensor(p_h_d, p_h)
    var FgBm = TileTensor(fgb_d, ffnm_layout)
    gemm_w[FFN, H](ctx, CurBm, wbuf, off[e + 8], pack_q4, Pg, m)
    gemm_w[FFN, H](ctx, CurBm, wbuf, off[e + 9], pack_q4, Pu, m)
    ctx.enqueue_function[r_swiglu](Pg, Pu, FgBm, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
    gemm_w[H, FFN](ctx, FgBm, wbuf, off[e + 10], pack_q4, Ph2, m)
    ctx.enqueue_function[r_add](Ph2, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

    var SharedHeadNorm = tens_f32(ctx, wbuf, off[e + 14], H, h_layout)
    var HdM = TileTensor(hd_d, xm_layout)
    ctx.enqueue_function[rms_m](Xm, SharedHeadNorm, HdM, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    if prof3:
        ctx.synchronize()
        var now3 = perf_counter_ns()
        p3[0] += Int(now3 - t3)
        t3 = now3
    if do_head:
        ctx.enqueue_function[rmsc_k](Xm, SharedHeadNorm, CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        var Pv = TileTensor(p_v_d, p_v)
        if draft_q4:
            var Wheadq4 = tens_q4q(ctx, wbuf, q4_off, H * VOCAB, q4_h_v)
            var Wheads4 = tens_q4s(ctx, wbuf, q4_off, H * VOCAB, s_h_v)
            gemm_q4(ctx, CurBm, Wheadq4, Wheads4, Pv, m, VOCAB, H)
        else:
            var Wheadq = tens_q8q(ctx, wbuf, off[e - 1], H * VOCAB, q_h_v)
            var Wheads = tens_q8s(ctx, wbuf, off[e - 1], H * VOCAB, s_h_v)
            gemm_w[VOCAB, H](ctx, CurBm, wbuf, off[e - 1], pack_q4, Pv, m)
        ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
        if prof3:
            ctx.synchronize()
            var now4 = perf_counter_ns()
            p3[1] += Int(now4 - t3)
            t3 = now4
        ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(VOCAB), Int32(0), grid_dim=m, block_dim=256)
        if prof3:
            ctx.synchronize()
            p3[2] += Int(perf_counter_ns() - t3)



def gemm_w[
    N: Int, K: Int, AL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    wbuf: DeviceBuffer[DType.uint8], o: Int, q4: Bool,
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int,
) raises:
    if q4:
        gemm_q4(ctx, A, tens_q4q(ctx, wbuf, o, N * K, row_major[N, K // 2]()), tens_q4s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), P, m, N, K)
    else:
        gemm_q8(ctx, A, tens_q8q(ctx, wbuf, o, N * K, row_major[N, K]()), tens_q8s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), P, m, N, K)



def tok_line(id: Int, tok: Int) -> String:
    return String("{\"id\":") + String(id) + ",\"tok\":" + String(tok) + "}"


def err_line(id: Int, msg: String) -> String:
    return String("{\"id\":") + String(id) + ",\"error\":\"" + msg + "\"}"

def gemm_pw[
    N: Int, K: Int, ACC: Bool, AL: TensorLayout, CL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    wbuf: DeviceBuffer[DType.uint8], o: Int, q4: Bool,
    C: TileTensor[f32, CL, MutAnyOrigin],
    m: Int,
) raises:
    if q4:
        gemm_prefill_q4[ACC](ctx, A, tens_q4q(ctx, wbuf, o, N * K, row_major[N, K // 2]()), tens_q4s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), C, m, N, K)
    else:
        gemm_prefill_q8[ACC](ctx, A, tens_q8q(ctx, wbuf, o, N * K, row_major[N, K]()), tens_q8s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), C, m, N, K)


def prefill_forward(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], pack_q4: Bool,
    m: Int, pos: Int, ring: Int,
    mut toks_d: DeviceBuffer[DType.int32], mut convstate_d: DeviceBuffer[f32], mut sstate_d: DeviceBuffer[f32],
    mut kc_d: DeviceBuffer[f32], mut vc_d: DeviceBuffer[f32],
    mut xp_d: DeviceBuffer[f32], mut curbp_d: DeviceBuffer[bf16], mut qkvp_d: DeviceBuffer[f32], mut zp_d: DeviceBuffer[f32],
    mut arp_d: DeviceBuffer[f32], mut brp_d: DeviceBuffer[f32], mut egp_d: DeviceBuffer[f32], mut betap_d: DeviceBuffer[f32],
    mut convp_d: DeviceBuffer[f32], mut sop_d: DeviceBuffer[f32], mut resbp_d: DeviceBuffer[bf16], mut qfp_d: DeviceBuffer[f32],
    mut qp_d: DeviceBuffer[f32], mut gatep_d: DeviceBuffer[f32], mut kp_d: DeviceBuffer[f32], mut vp_d: DeviceBuffer[f32],
    mut aop_d: DeviceBuffer[f32], mut gp_d: DeviceBuffer[f32], mut up_d: DeviceBuffer[f32], mut fgbp_d: DeviceBuffer[bf16],
) raises:
    # Prompt rows pos .. pos+m-1 through the 32-block trunk in one chunk
    # (bench/prefill-protocol.md): WMMA GEMMs over the weight-native q4/q8
    # layout, causal chunk attention, SSM conv + delta recurrence batched per
    # chunk. Same per-layer op order as the decode window path; state ring
    # slot (ring + m) receives the chunk's final conv window / delta state.
    var Embd = tens_bf16(ctx, wbuf, off[0], VOCAB * H, emb_layout)
    var Toks = TileTensor(toks_d, toks_layout)
    var ConvStateAll = TileTensor(convstate_d, csall_layout)
    var SStateAll = TileTensor(sstate_d, ssall_layout)
    var Xp = TileTensor(xp_d, xp_layout)
    var CurBp = TileTensor(curbp_d, xp_layout)
    ctx.enqueue_function[embed_p](Embd, Xp, Toks, Int32(pos), Int32(H), grid_dim=(ceildiv(H, 256), m), block_dim=256)
    var w = 1
    var ssm_i = 0
    var att_i = 0
    for layer in range(N_LAYERS):
        ctx.enqueue_function[rmsc_p](Xp, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBp, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        if is_attn(layer):
            var Qn = tens_f32(ctx, wbuf, off[w + 4], HD, hd_layout)
            var Kn = tens_f32(ctx, wbuf, off[w + 5], HD, hd_layout)
            var Qfp = TileTensor(qfp_d, qfp_layout)
            var Qp = TileTensor(qp_d, qp_layout)
            var Gatep = TileTensor(gatep_d, xpflat_layout)
            var Kflat = TileTensor(kp_d, kvp_flat)
            var Khd = TileTensor(kp_d, kvp_layout)
            var Vflat = TileTensor(vp_d, kvp_flat)
            var Vhd = TileTensor(vp_d, kvp_layout)
            var kcb = DeviceBuffer[f32](ctx, kc_d.unsafe_ptr() + att_i * NKVH * TMAX * HD, NKVH * TMAX * HD, owning=False)
            var vcb = DeviceBuffer[f32](ctx, vc_d.unsafe_ptr() + att_i * NKVH * TMAX * HD, NKVH * TMAX * HD, owning=False)
            var Kc = TileTensor(kcb, cache_layout)
            var Vc = TileTensor(vcb, cache_layout)
            var Aop = TileTensor(aop_d, qp_layout)
            var Aopflat = TileTensor(aop_d, xpflat_layout)
            var AoBp = TileTensor(resbp_d, xpflat_layout)
            var AoBpm = TileTensor(resbp_d, xp_layout)
            gemm_pw[QF, H, False](ctx, CurBp, wbuf, off[w + 1], pack_q4, Qfp, m)
            gemm_pw[KV, H, False](ctx, CurBp, wbuf, off[w + 2], pack_q4, Kflat, m)
            gemm_pw[KV, H, False](ctx, CurBp, wbuf, off[w + 3], pack_q4, Vflat, m)
            ctx.enqueue_function[split_p](Qfp, Qp, Gatep, grid_dim=(NQH, m), block_dim=HD)
            ctx.enqueue_function[hrms_qp](Qp, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
            ctx.enqueue_function[hrms_kvp](Khd, Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
            ctx.enqueue_function[rope_qp](Qp, Int32(pos), Int32(NQH), grid_dim=(NQH, m), block_dim=32)
            ctx.enqueue_function[rope_kp](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, m), block_dim=32)
            ctx.enqueue_function[append_p](Kc, Khd, Int32(pos), grid_dim=(NKVH, m), block_dim=HD)
            ctx.enqueue_function[append_p](Vc, Vhd, Int32(pos), grid_dim=(NKVH, m), block_dim=HD)
            ctx.enqueue_function[attp_k](Qp, Kc, Vc, Aop, Int32(pos), Int32(m), Float32(0.0625), grid_dim=(NKVH, ceildiv(m, PA_ROWS)), block_dim=256)
            ctx.enqueue_function[gmul_p](Aopflat, Gatep, AoBp, Int32(m * H), grid_dim=ceildiv(m * H, 256), block_dim=256)
            gemm_pw[H, H, True](ctx, AoBpm, wbuf, off[w + 6], pack_q4, Xp, m)
            att_i += 1
            w += 7
        else:
            var Cw = tens_f32(ctx, wbuf, off[w + 5], CONV * 4, cw_layout)
            var SsmA = tens_f32(ctx, wbuf, off[w + 6], NH_V, g32_layout)
            var DtB = tens_f32(ctx, wbuf, off[w + 7], NH_V, g32_layout)
            var Nw = tens_f32(ctx, wbuf, off[w + 8], SSTATE, n128_layout)
            var Qkvp = TileTensor(qkvp_d, qfp_layout)
            var Zp = TileTensor(zp_d, xp_layout)
            var Arp = TileTensor(arp_d, g32p_layout)
            var Brp = TileTensor(brp_d, g32p_layout)
            var Egp = TileTensor(egp_d, g32p_layout)
            var Betap = TileTensor(betap_d, g32p_layout)
            var Convp = TileTensor(convp_d, convp_layout)
            var Sop = TileTensor(sop_d, op_layout)
            var ResBp = TileTensor(resbp_d, xp_layout)
            gemm_pw[CONV, H, False](ctx, CurBp, wbuf, off[w + 1], pack_q4, Qkvp, m)
            gemm_pw[H, H, False](ctx, CurBp, wbuf, off[w + 2], pack_q4, Zp, m)
            gemm_pw[NH_V, H, False](ctx, CurBp, wbuf, off[w + 3], pack_q4, Arp, m)
            gemm_pw[NH_V, H, False](ctx, CurBp, wbuf, off[w + 4], pack_q4, Brp, m)
            ctx.enqueue_function[gates_p](Arp, Brp, Egp, Betap, SsmA, DtB, Int32(m), grid_dim=ceildiv(m * NH_V, 256), block_dim=256)
            ctx.enqueue_function[conv_p](Qkvp, ConvStateAll, Cw, Convp, Int32(ring), Int32(ssm_i), Int32(SLOTS), Int32(m), grid_dim=ceildiv(CONV, 256), block_dim=256)
            ctx.enqueue_function[l2_p](Convp, Int32(m), grid_dim=(NH_V, m), block_dim=SSTATE)
            ctx.enqueue_function[delta_p](SStateAll, Convp, Egp, Betap, Sop, Int32(ring), Int32(ssm_i), Int32(SLOTS), Int32(m), grid_dim=NH_V, block_dim=SSTATE)
            ctx.enqueue_function[gated_p](Sop, Zp, Nw, ResBp, grid_dim=(NH_V, m), block_dim=SSTATE)
            gemm_pw[H, H, True](ctx, ResBp, wbuf, off[w + 9], pack_q4, Xp, m)
            ssm_i += 1
            w += 10
        ctx.enqueue_function[rmsc_p](Xp, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBp, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        var Gp = TileTensor(gp_d, ffnp_layout)
        var Up = TileTensor(up_d, ffnp_layout)
        var FgBp = TileTensor(fgbp_d, ffnp_layout)
        gemm_pw[FFN, H, False](ctx, CurBp, wbuf, off[w + 1], pack_q4, Gp, m)
        gemm_pw[FFN, H, False](ctx, CurBp, wbuf, off[w + 2], pack_q4, Up, m)
        ctx.enqueue_function[swiglu_p](Gp, Up, FgBp, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
        gemm_pw[H, FFN, True](ctx, FgBp, wbuf, off[w + 3], pack_q4, Xp, m)
        w += 4



@fieldwise_init
struct WindowBufs(Copyable, Movable):
    var wbuf: DeviceBuffer[DType.uint8]
    var off: List[Int]
    var dtok_h: HostBuffer[DType.int32]
    var win_h: HostBuffer[DType.int32]
    var x_d: DeviceBuffer[f32]
    var curb_d: DeviceBuffer[bf16]
    var qkv_d: DeviceBuffer[f32]
    var z_d: DeviceBuffer[f32]
    var eg_d: DeviceBuffer[f32]
    var beta_d: DeviceBuffer[f32]
    var conv_d: DeviceBuffer[f32]
    var so_d: DeviceBuffer[f32]
    var resb_d: DeviceBuffer[bf16]
    var qf_d: DeviceBuffer[f32]
    var q_d: DeviceBuffer[f32]
    var gate_d: DeviceBuffer[f32]
    var k_d: DeviceBuffer[f32]
    var v_d: DeviceBuffer[f32]
    var ao_d: DeviceBuffer[f32]
    var fgb_d: DeviceBuffer[bf16]
    var aq_d: DeviceBuffer[DType.int8]
    var asc_d: DeviceBuffer[DType.float16]
    var logits_d: DeviceBuffer[f32]
    var toks_d: DeviceBuffer[DType.int32]
    var hn_d: DeviceBuffer[f32]
    var de_d: DeviceBuffer[f32]
    var hd_d: DeviceBuffer[f32]
    var cc_d: DeviceBuffer[bf16]
    var dtok_d: DeviceBuffer[DType.int32]
    var p_qf_d: DeviceBuffer[f32]
    var p_h_d: DeviceBuffer[f32]
    var p_kv_d: DeviceBuffer[f32]
    var p_32_d: DeviceBuffer[f32]
    var p_32b_d: DeviceBuffer[f32]
    var p_ffn_d: DeviceBuffer[f32]
    var p_ffn2_d: DeviceBuffer[f32]
    var p_v_d: DeviceBuffer[f32]
    var xp_d: DeviceBuffer[f32]
    var curbp_d: DeviceBuffer[bf16]
    var qkvp_d: DeviceBuffer[f32]
    var zp_d: DeviceBuffer[f32]
    var arp_d: DeviceBuffer[f32]
    var brp_d: DeviceBuffer[f32]
    var egp_d: DeviceBuffer[f32]
    var betap_d: DeviceBuffer[f32]
    var convp_d: DeviceBuffer[f32]
    var sop_d: DeviceBuffer[f32]
    var resbp_d: DeviceBuffer[bf16]
    var qfp_d: DeviceBuffer[f32]
    var qp_d: DeviceBuffer[f32]
    var gatep_d: DeviceBuffer[f32]
    var kp_d: DeviceBuffer[f32]
    var vp_d: DeviceBuffer[f32]
    var aop_d: DeviceBuffer[f32]
    var gp_d: DeviceBuffer[f32]
    var up_d: DeviceBuffer[f32]
    var fgbp_d: DeviceBuffer[bf16]
    var convstate_d: DeviceBuffer[f32]
    var sstate_d: DeviceBuffer[f32]
    var kc_d: DeviceBuffer[f32]
    var vc_d: DeviceBuffer[f32]
    var kc32_d: DeviceBuffer[f32]
    var vc32_d: DeviceBuffer[f32]
    var off_d: DeviceBuffer[DType.int64]
    var araw_d: DeviceBuffer[f32]
    var braw_d: DeviceBuffer[f32]
    var ctr_d: DeviceBuffer[DType.uint32]
    var prof_d: DeviceBuffer[DType.int64]
    var dbg_d: DeviceBuffer[f32]
    var hmax_d: DeviceBuffer[f32]
    var hidx_d: DeviceBuffer[DType.int32]
    var dump_h: HostBuffer[f32]
    var stream_h: HostBuffer[DType.int32]


@fieldwise_init
struct WindowCfg(Copyable, Movable):
    var pack_q4: Bool
    var draft_q4: Bool
    var q4_off: Int
    var e: Int
    var kcfg: Int
    var spec: Bool
    var spec_dbg: Bool
    var serve: Bool
    var req_id: Int
    var prof: Bool
    var pf2: Bool
    var pf3: Bool
    var pf4: Bool
    var dump: Bool
    var mega: Bool
    var mega_win: Bool
    var dot3: Bool
    var pf_chunk: Int
    var pf_rows: Int
    var pf_tail: Int
    var n_total: Int
    var n_prompt: Int


@fieldwise_init
struct WindowState(Copyable, Movable):
    var pos: Int
    var pos_prev: Int
    var ring: Int
    var n_drafted: Int
    var n_accepted: Int
    var n_spec_windows: Int
    var n_dumped: Int
    var tp: Int
    var tq: Int
    var pf_att: Int
    var pf_ssm: Int
    var pf_ffn: Int
    var pf_head: Int
    var pf_proc: Int
    var pf_draft: Int
    var fc: List[Int]
    var pc: List[Int]
    var p3: List[Int]

    def reset(mut self, t0: Int):
        # per request; n_dumped spans requests (BARO_DUMP)
        self.pos = 0
        self.pos_prev = 0
        self.ring = 0
        self.n_drafted = 0
        self.n_accepted = 0
        self.n_spec_windows = 0
        self.tp = t0
        self.tq = t0
        self.pf_att = 0
        self.pf_ssm = 0
        self.pf_ffn = 0
        self.pf_head = 0
        self.pf_proc = 0
        self.pf_draft = 0
        self.fc = [0, 0, 0, 0, 0, 0]
        self.pc = [0, 0, 0, 0, 0, 0, 0, 0]
        self.p3 = [0, 0, 0, 0]


def step_window(ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState) raises:
    var Xm = TileTensor(b.x_d, xm_layout)
    var CurBm = TileTensor(b.curb_d, xm_layout)
    var Logitsm = TileTensor(b.logits_d, vm_layout)
    var Toks = TileTensor(b.toks_d, toks_layout)
    var Pv = TileTensor(b.p_v_d, p_v)
    var Embd = tens_bf16(ctx, b.wbuf, b.off[0], VOCAB * H, emb_layout)
    var ConvStateAll = TileTensor(b.convstate_d, csall_layout)
    var SStateAll = TileTensor(b.sstate_d, ssall_layout)
    if st.pos < cfg.pf_rows:
        var mc = min(cfg.pf_chunk, cfg.pf_rows - st.pos)
        prefill_forward(ctx, b.wbuf, b.off, cfg.pack_q4, mc, st.pos, st.ring, b.toks_d, b.convstate_d, b.sstate_d, b.kc_d, b.vc_d,
            b.xp_d, b.curbp_d, b.qkvp_d, b.zp_d, b.arp_d, b.brp_d, b.egp_d, b.betap_d, b.convp_d, b.sop_d, b.resbp_d, b.qfp_d,
            b.qp_d, b.gatep_d, b.kp_d, b.vp_d, b.aop_d, b.gp_d, b.up_d, b.fgbp_d)
        st.ring = (st.ring + mc) % SLOTS
        st.pos_prev = st.pos
        st.pos += mc
        if st.pos == cfg.pf_rows:
            var Xtail = row_f32(ctx, b.xp_d, (mc - cfg.pf_tail) * H, MROWS * H, xm_layout)
            ctx.enqueue_function[rms_m](
                Xtail, tens_f32(ctx, b.wbuf, b.off[1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4], H, h_layout),
                TileTensor(b.hn_d, xm_layout), Int32(H), Float32(1e-6), grid_dim=cfg.pf_tail, block_dim=256,
            )
            st.pos_prev = st.pos - cfg.pf_tail
    else:
        var m = 1
        var win_spec = False
        if st.pos + 1 < cfg.n_prompt:
            m = min(MROWS, cfg.n_prompt - 1 - st.pos)
        elif cfg.spec:
            # process: tokens st.pos_prev+1..pos through the draft head with the
            # trunk's h rows (row r = h(st.pos_prev + r)); last row = draft step 0.
            var nproc = st.pos - st.pos_prev
            if cfg.prof:
                ctx.synchronize()
                st.tp = perf_counter_ns()
            var hn_rows = DeviceBuffer[f32](ctx, b.hn_d.unsafe_ptr(), nproc * H, owning=False)
            blk32_forward(ctx, b.wbuf, b.off, cfg.e, nproc, st.pos_prev + 1, st.pos_prev + 1, True, hn_rows,
                b.x_d, b.curb_d, b.qf_d, b.q_d, b.k_d, b.v_d, b.gate_d, b.ao_d, b.resb_d, b.fgb_d, b.p_qf_d, b.p_kv_d, b.p_h_d,
                b.p_ffn_d, b.p_ffn2_d, b.p_v_d, b.logits_d, b.cc_d, b.de_d, b.hd_d, b.kc32_d, b.vc32_d, b.toks_d, b.dtok_d,
                cfg.pf3, st.p3, cfg.draft_q4, cfg.q4_off, cfg.pack_q4)
            var Dtok = TileTensor(b.dtok_d, dtok_layout)
            ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(nproc - 1), Int32(st.pos + 1), Int32(1), grid_dim=1, block_dim=32)
            m = min(cfg.kcfg + 1, cfg.n_total - 1 - st.pos)
            if cfg.prof:
                ctx.synchronize()
                st.pf_proc += Int(perf_counter_ns() - st.tp)
                st.tp = perf_counter_ns()
            var hrow = nproc - 1
            for j in range(1, m - 1):
                var hd_row = DeviceBuffer[f32](ctx, b.hd_d.unsafe_ptr() + hrow * H, H, owning=False)
                blk32_forward(ctx, b.wbuf, b.off, cfg.e, 1, st.pos + j, st.pos + j, True, hd_row,
                    b.x_d, b.curb_d, b.qf_d, b.q_d, b.k_d, b.v_d, b.gate_d, b.ao_d, b.resb_d, b.fgb_d, b.p_qf_d, b.p_kv_d, b.p_h_d,
                    b.p_ffn_d, b.p_ffn2_d, b.p_v_d, b.logits_d, b.cc_d, b.de_d, b.hd_d, b.kc32_d, b.vc32_d, b.toks_d, b.dtok_d,
                    cfg.pf3, st.p3, cfg.draft_q4, cfg.q4_off, cfg.pack_q4)
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(0), Int32(st.pos + j + 1), Int32(1), grid_dim=1, block_dim=32)
                hrow = 0
            st.n_drafted += m - 1
            win_spec = True
            st.n_spec_windows += 1
            if cfg.prof:
                ctx.synchronize()
                st.pf_draft += Int(perf_counter_ns() - st.tp)

        ctx.enqueue_function[embed_k](
            Embd, Xm, Toks, Int32(st.pos), Int32(H),
            grid_dim=(ceildiv(H, 256), m), block_dim=256,
        )

        var w = 1
        var ssm_i = 0
        var att_i = 0
        var use_mega = cfg.mega and m == 1 and not win_spec and st.pos + 1 >= cfg.n_prompt
        var use_mega_win = cfg.mega_win and win_spec and m == MEGA_MR
        if use_mega or use_mega_win:
            var Hnm0 = TileTensor(b.hn_d, xm_layout)
            var Dtok0 = TileTensor(b.dtok_d, dtok_layout)
            if use_mega and cfg.pack_q4:
                ctx.enqueue_function[mega_token_q4_k](
                    b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
                    TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
                    TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
                    TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
                    TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
                    TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
                    TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
                    b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(),
                    TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                    TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                    Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                    Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(1 if cfg.dump else 0), Int32(1), grid_dim=MEGA_G, block_dim=ROW_THREADS,
                )
            elif use_mega:
                ctx.enqueue_function[mega_token_k](
                    b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
                    TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
                    TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
                    TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
                    TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
                    TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
                    TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
                    b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(),
                    TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                    TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                    Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                    Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(1 if cfg.dump else 0), Int32(1), grid_dim=MEGA_G, block_dim=ROW_THREADS,
                )
            else:
                ctx.enqueue_function[mega_win_k](
                    b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
                    TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
                    TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
                    TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
                    TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
                    TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
                    TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
                    b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(),
                    TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                    TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                    Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                    Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(m), Int32(0), Int32(0), grid_dim=MEGA_G_WIN, block_dim=ROW_THREADS,
                )
        for layer in range(0 if (use_mega or use_mega_win) else N_LAYERS):
            if cfg.prof:
                ctx.synchronize()
                st.tp = perf_counter_ns()
                st.tq = st.tp
            # -- attention / ssm sub-block --
            ctx.enqueue_function[rmsc_k](
                Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )

            if is_attn(layer):
                var Wqq = tens_q8q(ctx, b.wbuf, b.off[w + 1], H * QF, q_h_qf)
                var Wqs = tens_q8s(ctx, b.wbuf, b.off[w + 1], H * QF, s_h_qf)
                var Wkq = tens_q8q(ctx, b.wbuf, b.off[w + 2], H * KV, q_h_kv)
                var Wks = tens_q8s(ctx, b.wbuf, b.off[w + 2], H * KV, s_h_kv)
                var Wvq = tens_q8q(ctx, b.wbuf, b.off[w + 3], H * KV, q_h_kv)
                var Wvs = tens_q8s(ctx, b.wbuf, b.off[w + 3], H * KV, s_h_kv)
                var Qn = tens_f32(ctx, b.wbuf, b.off[w + 4], HD, hd_layout)
                var Kn = tens_f32(ctx, b.wbuf, b.off[w + 5], HD, hd_layout)
                var Woq = tens_q8q(ctx, b.wbuf, b.off[w + 6], H * H, q_h_h)
                var Wos = tens_q8s(ctx, b.wbuf, b.off[w + 6], H * H, s_h_h)
                var Pqf = TileTensor(b.p_qf_d, p_qf)
                var Pkv = TileTensor(b.p_kv_d, p_kv)
                var Ph = TileTensor(b.p_h_d, p_h)
                var Qfm = TileTensor(b.qf_d, qfm_layout)
                var Q = TileTensor(b.q_d, qm_layout)
                var Gate = TileTensor(b.gate_d, xflat_layout)
                var Kflat = TileTensor(b.k_d, kvm_flat)
                var Khd = TileTensor(b.k_d, kvm_layout)
                var Vflat = TileTensor(b.v_d, kvm_flat)
                var Vhd = TileTensor(b.v_d, kvm_layout)
                var kcb = DeviceBuffer[f32](
                    ctx, b.kc_d.unsafe_ptr() + att_i * NKVH * TMAX * HD,
                    NKVH * TMAX * HD, owning=False,
                )
                var vcb = DeviceBuffer[f32](
                    ctx, b.vc_d.unsafe_ptr() + att_i * NKVH * TMAX * HD,
                    NKVH * TMAX * HD, owning=False,
                )
                var Kc = TileTensor(kcb, cache_layout)
                var Vc = TileTensor(vcb, cache_layout)
                var Ao = TileTensor(b.ao_d, qm_layout)
                var Aoflat = TileTensor(b.ao_d, xflat_layout)
                var AoB = TileTensor(b.resb_d, xflat_layout)
                var AoBm = TileTensor(b.resb_d, xm_layout)

                gemm_w[QF, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pqf, m)
                ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
                gemm_w[KV, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Pkv, m)
                ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                gemm_w[KV, H](ctx, CurBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Pkv, m)
                ctx.enqueue_function[r_kv](Pkv, Vflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, m), block_dim=HD)
                ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
                ctx.enqueue_function[hrms_kv](Khd, Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
                ctx.enqueue_function[rope_q](Q, Int32(st.pos), Int32(NQH), grid_dim=(NQH, m), block_dim=32)
                ctx.enqueue_function[rope_k](Khd, Int32(st.pos), Int32(NKVH), grid_dim=(NKVH, m), block_dim=32)
                ctx.enqueue_function[append_k](Kc, Khd, Int32(st.pos), grid_dim=(NKVH, m), block_dim=HD)
                ctx.enqueue_function[append_k](Vc, Vhd, Int32(st.pos), grid_dim=(NKVH, m), block_dim=HD)
                ctx.enqueue_function[att_k](Q, Kc, Vc, Ao, Int32(st.pos + 1), Float32(0.0625), grid_dim=(NQH, m), block_dim=HD)
                ctx.enqueue_function[gmul_k](Aoflat, Gate, AoB, Int32(m * H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                gemm_w[H, H](ctx, AoBm, b.wbuf, b.off[w + 6], cfg.pack_q4, Ph, m)
                ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                att_i += 1
                w += 7
            else:
                var Wqkvq = tens_q8q(ctx, b.wbuf, b.off[w + 1], H * CONV, q_h_qf)
                var Wqkvs = tens_q8s(ctx, b.wbuf, b.off[w + 1], H * CONV, s_h_qf)
                var Wzq = tens_q8q(ctx, b.wbuf, b.off[w + 2], H * H, q_h_h)
                var Wzs = tens_q8s(ctx, b.wbuf, b.off[w + 2], H * H, s_h_h)
                var Waq = tens_q8q(ctx, b.wbuf, b.off[w + 3], H * NH_V, q_h_32)
                var Was = tens_q8s(ctx, b.wbuf, b.off[w + 3], H * NH_V, s_h_32)
                var Wbq = tens_q8q(ctx, b.wbuf, b.off[w + 4], H * NH_V, q_h_32)
                var Wbs = tens_q8s(ctx, b.wbuf, b.off[w + 4], H * NH_V, s_h_32)
                var Cw = tens_f32(ctx, b.wbuf, b.off[w + 5], CONV * 4, cw_layout)
                var SsmA = tens_f32(ctx, b.wbuf, b.off[w + 6], NH_V, g32_layout)
                var DtB = tens_f32(ctx, b.wbuf, b.off[w + 7], NH_V, g32_layout)
                var Nw = tens_f32(ctx, b.wbuf, b.off[w + 8], SSTATE, n128_layout)
                var Wsoutq = tens_q8q(ctx, b.wbuf, b.off[w + 9], H * H, q_h_h)
                var Wsouts = tens_q8s(ctx, b.wbuf, b.off[w + 9], H * H, s_h_h)
                var Pq = TileTensor(b.p_qf_d, p_qf)
                var Ph = TileTensor(b.p_h_d, p_h)
                var Pab = TileTensor(b.p_32_d, p_32)
                var Pab2 = TileTensor(b.p_32b_d, p_32)
                var Qkvm = TileTensor(b.qkv_d, qfm_layout)
                var Zm = TileTensor(b.z_d, xm_layout)
                var Eg = TileTensor(b.eg_d, g32m_layout)
                var Beta = TileTensor(b.beta_d, g32m_layout)
                var Conv = TileTensor(b.conv_d, convm_layout)
                var So = TileTensor(b.so_d, om_layout)
                var ResBm = TileTensor(b.resb_d, xm_layout)

                gemm_w[CONV, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pq, m)
                gemm_w[H, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Ph, m)
                gemm_w[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Pab, m)
                gemm_w[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 4], cfg.pack_q4, Pab2, m)
                ctx.enqueue_function[r_qf](Pq, Qkvm, Int32(m), Int32(CONV), grid_dim=ceildiv(m * CONV, 256), block_dim=256)
                ctx.enqueue_function[r_h](Ph, Zm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[0] += Int(nw - st.tq)
                    st.tq = nw
                ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, SsmA, DtB, Int32(m), grid_dim=1, block_dim=NH_V)
                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[1] += Int(nw - st.tq)
                    st.tq = nw
                ctx.enqueue_function[conv_k](Qkvm, ConvStateAll, Cw, Conv, Int32(st.ring), Int32(ssm_i), Int32(SLOTS), Int32(m), grid_dim=ceildiv(CONV, 256), block_dim=256)
                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[2] += Int(nw - st.tq)
                    st.tq = nw
                ctx.enqueue_function[l2_k](Conv, Int32(m), grid_dim=NH_V, block_dim=SSTATE)
                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[3] += Int(nw - st.tq)
                    st.tq = nw
                delta_dispatch(ctx, SStateAll, Conv, Eg, Beta, So, Int32(st.ring), Int32(ssm_i), Int32(SLOTS), m)
                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[4] += Int(nw - st.tq)
                    st.tq = nw
                ctx.enqueue_function[gated_k](So, Zm, Nw, ResBm, Int32(m), grid_dim=NH_V, block_dim=SSTATE)

                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[5] += Int(nw - st.tq)
                    st.tq = nw
                gemm_w[H, H](ctx, ResBm, b.wbuf, b.off[w + 9], cfg.pack_q4, Ph, m)
                ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                ssm_i += 1
                w += 10

            if cfg.prof:
                ctx.synchronize()
                var now = perf_counter_ns()
                if is_attn(layer):
                    st.pf_att += Int(now - st.tp)
                else:
                    st.pf_ssm += Int(now - st.tp)
                    if cfg.pf2:
                        st.pc[6] += Int(now - st.tq)
                st.tp = now
                st.tq = now
            if cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt:
                ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr() + (2 * layer) * H, H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False))
            # -- ffn sub-block --
            if cfg.pf4:
                ctx.synchronize()
                st.tq = perf_counter_ns()
            ctx.enqueue_function[rmsc_k](
                Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[0] += Int(nw - st.tq)
                st.tq = nw
            var Wfgq = tens_q8q(ctx, b.wbuf, b.off[w + 1], H * FFN, q_h_ffn)
            var Wfgs = tens_q8s(ctx, b.wbuf, b.off[w + 1], H * FFN, s_h_ffn)
            var Wfuq = tens_q8q(ctx, b.wbuf, b.off[w + 2], H * FFN, q_h_ffn)
            var Wfus = tens_q8s(ctx, b.wbuf, b.off[w + 2], H * FFN, s_h_ffn)
            var Wfdq = tens_q8q(ctx, b.wbuf, b.off[w + 3], FFN * H, q_ffn_h)
            var Wfds = tens_q8s(ctx, b.wbuf, b.off[w + 3], FFN * H, s_ffn_h)
            var Pg = TileTensor(b.p_ffn_d, p_ffn)
            var Pu = TileTensor(b.p_ffn2_d, p_ffn)
            var Ph2 = TileTensor(b.p_h_d, p_h)
            var FgBm = TileTensor(b.fgb_d, ffnm_layout)
            var AqH = TileTensor(b.aq_d, aqm_h)
            var AsH = TileTensor(b.asc_d, asm_h)
            var use_dot = cfg.dot3 and m >= 3
            if use_dot:
                quant_rows(ctx, CurBm, AqH, AsH, m, H)
                gemm_q8dot(ctx, AqH, AsH, Wfgq, Wfgs, Pg, m, FFN, H)
            else:
                gemm_w[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pg, m)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[1] += Int(nw - st.tq)
                st.tq = nw
            if use_dot:
                gemm_q8dot(ctx, AqH, AsH, Wfuq, Wfus, Pu, m, FFN, H)
            else:
                gemm_w[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Pu, m)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[2] += Int(nw - st.tq)
                st.tq = nw
            ctx.enqueue_function[r_swiglu](Pg, Pu, FgBm, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[3] += Int(nw - st.tq)
                st.tq = nw
            if use_dot:
                var AqF = TileTensor(b.aq_d, aqm_ffn)
                var AsF = TileTensor(b.asc_d, asm_ffn)
                quant_rows(ctx, FgBm, AqF, AsF, m, FFN)
                gemm_q8dot(ctx, AqF, AsF, Wfdq, Wfds, Ph2, m, H, FFN)
            else:
                gemm_w[H, FFN](ctx, FgBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Ph2, m)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[4] += Int(nw - st.tq)
                st.tq = nw
            ctx.enqueue_function[r_add](Ph2, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[5] += Int(nw - st.tq)
                st.tq = nw
            w += 4
            if cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt:
                ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr() + (2 * layer + 1) * H, H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False))
            if cfg.prof:
                ctx.synchronize()
                var now = perf_counter_ns()
                st.pf_ffn += Int(now - st.tp)
                st.tp = now

        if use_mega or use_mega_win:
            w = 1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4
        if cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt and st.n_dumped < GEN_N:
            ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dump_h.unsafe_ptr() + st.n_dumped * 2 * N_LAYERS * H, 2 * N_LAYERS * H, owning=False), src_buf=b.dbg_d)
            st.n_dumped += 1
        # -- head --
        if cfg.prof:
            ctx.synchronize()
            st.tp = perf_counter_ns()
        # f32 copy of the post-final-norm hidden state (pre-LM-head): row r is
        # h(st.pos + r), what the MTP draft head pairs with token st.pos + r + 1.
        var Hnm = TileTensor(b.hn_d, xm_layout)
        if not use_mega:
            ctx.enqueue_function[rms_m](
                Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), Hnm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
        if st.pos + m >= cfg.n_prompt:
            if not use_mega:
                ctx.enqueue_function[rmsc_k](
                    Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm,
                    Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
                )
                var Wheadq = tens_q8q(ctx, b.wbuf, b.off[w + 1], H * VOCAB, q_h_v)
                var Wheads = tens_q8s(ctx, b.wbuf, b.off[w + 1], H * VOCAB, s_h_v)
                gemm_w[VOCAB, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pv, m)
                ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
            if use_mega:
                pass
            elif win_spec:
                var Dtok = TileTensor(b.dtok_d, dtok_layout)
                ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(VOCAB), Int32(0), grid_dim=m, block_dim=256)
                var t_acc = 0
                if cfg.pf3:
                    ctx.synchronize()
                    t_acc = perf_counter_ns()
                ctx.enqueue_copy(dst_buf=b.dtok_h, src_buf=DeviceBuffer[DType.int32](ctx, b.dtok_d.unsafe_ptr(), KMAX + 1, owning=False))
                ctx.enqueue_copy(dst_buf=b.win_h, src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr() + st.pos + 1, KMAX + 1, owning=False))
                ctx.synchronize()
                var n_acc = 0
                while n_acc < m - 1 and b.dtok_h[n_acc] == b.win_h[n_acc]:
                    n_acc += 1
                st.n_accepted += n_acc
                if cfg.spec_dbg:
                    var line = String("win pos=") + String(st.pos) + " m=" + String(m) + " n_acc=" + String(n_acc) + " toks:"
                    for i in range(m):
                        line += " " + String(b.win_h[i]) + "/" + String(b.dtok_h[i])
                    print(line)
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(n_acc), Int32(st.pos + n_acc + 1), Int32(1), grid_dim=1, block_dim=32)
                if cfg.pf3:
                    ctx.synchronize()
                    st.p3[3] += Int(perf_counter_ns() - t_acc)
                m = n_acc + 1
            else:
                ctx.enqueue_function[argmax_k](Logitsm, Toks, Int32(VOCAB), Int32(st.pos + 1), grid_dim=m, block_dim=256)
            if cfg.serve:
                if win_spec:
                    for i in range(m - 1):
                        print(tok_line(cfg.req_id, Int(b.win_h[i])))
                    print(tok_line(cfg.req_id, Int(b.dtok_h[m - 1])))
                else:
                    ctx.enqueue_copy(dst_buf=b.stream_h.create_sub_buffer[DType.int32](0, m), src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr() + st.pos + 1, m, owning=False))
                    ctx.synchronize()
                    for i in range(m):
                        print(tok_line(cfg.req_id, Int(b.stream_h[i])))
            if cfg.prof:
                ctx.synchronize()
                st.pf_head += Int(perf_counter_ns() - st.tp)

        # advance by the window width; once verify lands this becomes the
        # accepted-token count, which is what makes rollback free.
        st.ring = (st.ring + m) % SLOTS
        st.pos_prev = st.pos
        st.pos += m
