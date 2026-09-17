"""baro window step: everything the engine does per decode window -- prefill
chunk or MTP draft, embed, the 32-block hybrid stack (launch path or the
persistent megakernel), head, verify -- over buffers the harness owns.

This file is embedded in BARO ggufs (`baro.kernel.files`) and is what the
self-optimising loop may edit. The stopwatch, the prints and the fixtures
live in serve/engine.mojo, which is not embedded: a candidate cannot reach
t0, t_prefill_end or dt from here (exchange/scorer-integrity-report.md, P-A).
"""
from std.collections import Dict
from std.math import ceildiv, log
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *
from expert_tier import ExpertTier
from serve_proto import SampleParams
from moe import (
    moe_embed_q8_0_pos, moe_matmul_q8_0_m1, moe_matmul_q8_0_m1_add,
    amar_moe_router_top8_sig, moe_gate_up_q4k_pack,
    amar_moe_down_q4k, amar_moe_down_q6k, moe_gate_up_q8_0, moe_down_q8_0_res,
    moe_gate_up_q4k_zc, amar_moe_down_q4k_zc,
    MOE_WAVES, MOE_THREADS, N_EXP, TOPK, E_FFN, SH_FFN,
)
from matmul_skinny import amar_matmul_skinny_m1_row, amar_matmul_skinny_m1_row2
from ssm import amar_widen_bf16, amar_ssm_gated_out_bf16
from elementwise import amar_rmsnorm_cast2
from grammar.automaton import Bitset
from grammar.matcher import Matcher
from grammar_rt import reasoning_boundary_observe

comptime rmsc2_k = amar_rmsnorm_cast2[type_of(xm_layout), type_of(h_layout), type_of(xm_layout), type_of(h2_layout)]
from dattn import dattn_nsplit
from mega import DATT_NLD


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
        (wbuf.unsafe_ptr().unsafe_offset(o)).unsafe_bitcast[Scalar[DType.bfloat16]](),
        n, owning=False,
    )


def wf32(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int
) -> DeviceBuffer[DType.float32]:
    return DeviceBuffer[DType.float32](
        ctx,
        (wbuf.unsafe_ptr().unsafe_offset(o)).unsafe_bitcast[Scalar[DType.float32]](),
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
        ctx, (wbuf.unsafe_ptr().unsafe_offset(o)).unsafe_bitcast[Scalar[DType.int8]](), n, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.int8, LT, MutAnyOrigin]](t)


def tens_q8s[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](
        ctx, (wbuf.unsafe_ptr().unsafe_offset(o + n)).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](t)


def tens_q4q[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.uint8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.uint8](
        ctx, (wbuf.unsafe_ptr().unsafe_offset(o)).unsafe_bitcast[Scalar[DType.uint8]](), n // 2, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.uint8, LT, MutAnyOrigin]](t)


def tens_i32[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.int32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.int32](
        ctx, (wbuf.unsafe_ptr().unsafe_offset(o)).unsafe_bitcast[Scalar[DType.int32]](), n, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.int32, LT, MutAnyOrigin]](t)


def tens_q4s[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](
        ctx, (wbuf.unsafe_ptr().unsafe_offset(o + n // 2)).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False
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
    var s = DeviceBuffer[f32](ctx, b.unsafe_ptr().unsafe_offset(o), n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](t)


def row_bf16[
    LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[bf16], o: Int, n: Int, lt: LT
) -> TileTensor[bf16, LT, MutAnyOrigin]:
    var s = DeviceBuffer[bf16](ctx, b.unsafe_ptr().unsafe_offset(o), n, owning=False)
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
    mut hd_d: DeviceBuffer[f32], mut kc32_d: DeviceBuffer[KVT], mut vc32_d: DeviceBuffer[KVT],
    mut kvtab_d: DeviceBuffer[DType.int32],
    mut toks_d: DeviceBuffer[DType.int32], mut dtok_d: DeviceBuffer[DType.int32],
    prof3: Bool, mut p3: List[Int],
    draft_q4: Bool, q4_off: Int, pack_q4: Bool, fr_k: Int = 0, fr_off: Int = 0, fr_ids_off: Int = 0,
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
    var Gate = TileTensor(gate_d, attflat_layout)
    var Kflat = TileTensor(k_d, kvm_flat)
    var Khd = TileTensor(k_d, kvm_layout)
    var Vflat = TileTensor(v_d, kvm_flat)
    var Vhd = TileTensor(v_d, kvm_layout)
    var Kc = TileTensor(kc32_d, cache1_layout)
    var Vc = TileTensor(vc32_d, cache1_layout)
    var Ao = TileTensor(ao_d, qm_layout)
    var Aoflat = TileTensor(ao_d, attflat_layout)
    var AoB = TileTensor(resb_d, attflat_layout)
    var AoBm = TileTensor(resb_d, attm_layout)
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
    ctx.enqueue_function[append_1](Kc, Khd, kvtab_d.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, m), block_dim=HD)
    ctx.enqueue_function[append_1](Vc, Vhd, kvtab_d.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, m), block_dim=HD)
    ctx.enqueue_function[att_1](Q, Kc, Vc, Ao, kvtab_d.unsafe_ptr(), Int32(pos + 1), Float32(0.0625), Int32(0), grid_dim=(NQH, m), block_dim=HD)
    ctx.enqueue_function[gmul_k](Aoflat, Gate, AoB, Int32(m * ATT), grid_dim=ceildiv(m * ATT, 256), block_dim=256)
    gemm_w[H, ATT](ctx, AoBm, wbuf, off[e + 6], pack_q4, Ph, m)
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
        var nv = VOCAB
        if fr_k > 0:
            nv = fr_k
            var Wfrq = tens_q4q(ctx, wbuf, fr_off, H * fr_k, q4_h_v)
            var Wfrs = tens_q4s(ctx, wbuf, fr_off, H * fr_k, s_h_v)
            gemm_q4(ctx, CurBm, Wfrq, Wfrs, Pv, m, fr_k, H)
        elif draft_q4:
            var Wheadq4 = tens_q4q(ctx, wbuf, q4_off, H * VOCAB, q4_h_v)
            var Wheads4 = tens_q4s(ctx, wbuf, q4_off, H * VOCAB, s_h_v)
            gemm_q4(ctx, CurBm, Wheadq4, Wheads4, Pv, m, VOCAB, H)
        else:
            var Wheadq = tens_q8q(ctx, wbuf, off[e - 1], H * VOCAB, q_h_v)
            var Wheads = tens_q8s(ctx, wbuf, off[e - 1], H * VOCAB, s_h_v)
            gemm_w[VOCAB, H](ctx, CurBm, wbuf, off[e - 1], pack_q4, Pv, m)
        ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(nv), grid_dim=ceildiv(m * nv, 256), block_dim=256)
        if prof3:
            ctx.synchronize()
            var now4 = perf_counter_ns()
            p3[1] += Int(now4 - t3)
            t3 = now4
        ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(nv), Int32(0), grid_dim=m, block_dim=256)
        if fr_k > 0:
            ctx.enqueue_function[remap_d](tens_i32(ctx, wbuf, fr_ids_off, fr_k, frmap_layout), Dtok, Int32(m), grid_dim=1, block_dim=32)
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


def tok_line_lp(id: Int, tok: Int, logprob: Float64, top_ids: List[Int], top_lp: List[Float64]) -> String:
    var s = String("{\"id\":") + String(id) + ",\"tok\":" + String(tok) + ",\"logprob\":" + String(logprob) + ",\"top_logprobs\":["
    for i in range(len(top_ids)):
        if i > 0:
            s += ","
        s += "{\"id\":" + String(top_ids[i]) + ",\"logprob\":" + String(top_lp[i]) + "}"
    s += "]}"
    return s


def err_line(id: Int, msg: String) -> String:
    return String("{\"id\":") + String(id) + ",\"error\":\"" + msg + "\"}"

def gemm_pw[
    N: Int, K: Int, ACC: Bool, AL: TensorLayout, CL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    wbuf: DeviceBuffer[DType.uint8], o: Int, q4: Bool,
    C: TileTensor[f32, CL, MutAnyOrigin],
    m: Int, prof: Bool, mut gemm_ns: Int,
) raises:
    var t0 = 0
    if prof:
        ctx.synchronize()
        t0 = perf_counter_ns()
    if q4:
        gemm_prefill_q4[ACC](ctx, A, tens_q4q(ctx, wbuf, o, N * K, row_major[N, K // 2]()), tens_q4s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), C, m, N, K)
    else:
        gemm_prefill_q8[ACC](ctx, A, tens_q8q(ctx, wbuf, o, N * K, row_major[N, K]()), tens_q8s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), C, m, N, K)
    if prof:
        ctx.synchronize()
        gemm_ns += Int(perf_counter_ns() - t0)


def prefill_forward(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], pack_q4: Bool,
    m: Int, pos: Int, ring: Int, prof: Bool,
    mut toks_d: DeviceBuffer[DType.int32], mut convstate_d: DeviceBuffer[f32], mut sstate_d: DeviceBuffer[f32],
    mut kc_d: DeviceBuffer[KVT], mut vc_d: DeviceBuffer[KVT], mut kvtab_d: DeviceBuffer[DType.int32],
    mut xp_d: DeviceBuffer[f32], mut curbp_d: DeviceBuffer[bf16], mut qkvp_d: DeviceBuffer[f32], mut zp_d: DeviceBuffer[f32],
    mut arp_d: DeviceBuffer[f32], mut brp_d: DeviceBuffer[f32], mut egp_d: DeviceBuffer[f32], mut betap_d: DeviceBuffer[f32],
    mut convp_d: DeviceBuffer[f32], mut sop_d: DeviceBuffer[f32], mut resbp_d: DeviceBuffer[bf16], mut qfp_d: DeviceBuffer[f32],
    mut qp_d: DeviceBuffer[f32], mut gatep_d: DeviceBuffer[f32], mut kp_d: DeviceBuffer[f32], mut vp_d: DeviceBuffer[f32],
    mut aop_d: DeviceBuffer[f32], mut gp_d: DeviceBuffer[f32], mut up_d: DeviceBuffer[f32], mut fgbp_d: DeviceBuffer[bf16],
    mut pfx: List[Int],
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
    # BARO_PROFILE != 0: synchronize around every prefill GEMM and print the
    # chunk's GEMM share (bench/pfgemm-protocol.md). Serialized, so chunk_s
    # exceeds the unsynchronized chunk time; read the share, not the sum.
    var gemm_ns = 0
    var t_chunk = 0
    if prof:
        ctx.synchronize()
        t_chunk = perf_counter_ns()
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
            var Gatep = TileTensor(gatep_d, attpflat_layout)
            var Kflat = TileTensor(kp_d, kvp_flat)
            var Khd = TileTensor(kp_d, kvp_layout)
            var Vflat = TileTensor(vp_d, kvp_flat)
            var Vhd = TileTensor(vp_d, kvp_layout)
            var Kc = TileTensor(kc_d, cache_layout)
            var Vc = TileTensor(vc_d, cache_layout)
            var Aop = TileTensor(aop_d, qp_layout)
            var Aopflat = TileTensor(aop_d, attpflat_layout)
            var AoBp = TileTensor(resbp_d, attpflat_layout)
            var AoBpm = TileTensor(resbp_d, attp_layout)
            gemm_pw[QF, H, False](ctx, CurBp, wbuf, off[w + 1], pack_q4, Qfp, m, prof, gemm_ns)
            gemm_pw[KV, H, False](ctx, CurBp, wbuf, off[w + 2], pack_q4, Kflat, m, prof, gemm_ns)
            gemm_pw[KV, H, False](ctx, CurBp, wbuf, off[w + 3], pack_q4, Vflat, m, prof, gemm_ns)
            ctx.enqueue_function[split_p](Qfp, Qp, Gatep, grid_dim=(NQH, m), block_dim=HD)
            ctx.enqueue_function[hrms_qp](Qp, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
            ctx.enqueue_function[hrms_kvp](Khd, Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
            ctx.enqueue_function[rope_qp](Qp, Int32(pos), Int32(NQH), grid_dim=(NQH, m), block_dim=32)
            ctx.enqueue_function[rope_kp](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, m), block_dim=32)
            ctx.enqueue_function[append_p](Kc, Khd, kvtab_d.unsafe_ptr(), Int32(pos), Int32(att_i), grid_dim=(NKVH, m), block_dim=HD)
            ctx.enqueue_function[append_p](Vc, Vhd, kvtab_d.unsafe_ptr(), Int32(pos), Int32(att_i), grid_dim=(NKVH, m), block_dim=HD)
            var ta = 0
            if prof:
                ctx.synchronize()
                ta = perf_counter_ns()
            ctx.enqueue_function[attpw_k](Qp, Kc, Vc, Aop, kvtab_d.unsafe_ptr(), Int32(pos), Int32(m), Float32(0.0625), Int32(att_i), grid_dim=(NKVH, ceildiv(m, PW_ROWS)), block_dim=PW_THREADS)
            if prof:
                ctx.synchronize()
                pfx[0] += Int(perf_counter_ns() - ta)
            ctx.enqueue_function[gmul_p](Aopflat, Gatep, AoBp, Int32(m * ATT), grid_dim=ceildiv(m * ATT, 256), block_dim=256)
            gemm_pw[H, ATT, True](ctx, AoBpm, wbuf, off[w + 6], pack_q4, Xp, m, prof, gemm_ns)
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
            gemm_pw[CONV, H, False](ctx, CurBp, wbuf, off[w + 1], pack_q4, Qkvp, m, prof, gemm_ns)
            gemm_pw[H, H, False](ctx, CurBp, wbuf, off[w + 2], pack_q4, Zp, m, prof, gemm_ns)
            gemm_pw[NH_V, H, False](ctx, CurBp, wbuf, off[w + 3], pack_q4, Arp, m, prof, gemm_ns)
            gemm_pw[NH_V, H, False](ctx, CurBp, wbuf, off[w + 4], pack_q4, Brp, m, prof, gemm_ns)
            var ts = 0
            if prof:
                ctx.synchronize()
                ts = perf_counter_ns()
            ctx.enqueue_function[gates_p](Arp, Brp, Egp, Betap, SsmA, DtB, Int32(m), grid_dim=ceildiv(m * NH_V, 256), block_dim=256)
            ctx.enqueue_function[conv_p](Qkvp, ConvStateAll, Cw, Convp, Int32(ring), Int32(ssm_i), Int32(SLOTS), Int32(m), grid_dim=ceildiv(CONV, 256), block_dim=256)
            ctx.enqueue_function[l2_p](Convp, Int32(m), grid_dim=(NH_V, m), block_dim=SSTATE)
            ctx.enqueue_function[deltaw_p](SStateAll, Convp, Egp, Betap, Sop, Int32(ring), Int32(ssm_i), Int32(SLOTS), Int32(m), grid_dim=DC_BLOCKS, block_dim=32)
            ctx.enqueue_function[gated_p](Sop, Zp, Nw, ResBp, grid_dim=(NH_V, m), block_dim=SSTATE)
            if prof:
                ctx.synchronize()
                pfx[1] += Int(perf_counter_ns() - ts)
            gemm_pw[H, H, True](ctx, ResBp, wbuf, off[w + 9], pack_q4, Xp, m, prof, gemm_ns)
            ssm_i += 1
            w += 10
        ctx.enqueue_function[rmsc_p](Xp, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBp, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        var Gp = TileTensor(gp_d, ffnp_layout)
        var Up = TileTensor(up_d, ffnp_layout)
        var FgBp = TileTensor(fgbp_d, ffnp_layout)
        gemm_pw[FFN, H, False](ctx, CurBp, wbuf, off[w + 1], pack_q4, Gp, m, prof, gemm_ns)
        gemm_pw[FFN, H, False](ctx, CurBp, wbuf, off[w + 2], pack_q4, Up, m, prof, gemm_ns)
        ctx.enqueue_function[swiglu_p](Gp, Up, FgBp, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
        gemm_pw[H, FFN, True](ctx, FgBp, wbuf, off[w + 3], pack_q4, Xp, m, prof, gemm_ns)
        w += 4
    if prof:
        ctx.synchronize()
        var chunk_ns = Int(perf_counter_ns() - t_chunk)
        pfx[2] += gemm_ns
        pfx[3] += chunk_ns
        print("prefill profile: rows", m, " gemm_s", Float64(gemm_ns) / 1e9, " chunk_s", Float64(chunk_ns) / 1e9, " share", Float64(gemm_ns) / Float64(chunk_ns))



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
    var kvpool: Int
    var kvtab_d: DeviceBuffer[DType.int32]
    var kvtab_h: HostBuffer[DType.int32]
    var tpages: Int
    var kc_d: DeviceBuffer[KVT]
    var vc_d: DeviceBuffer[KVT]
    var kc32_d: DeviceBuffer[KVT]
    var vc32_d: DeviceBuffer[KVT]
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
    # A1: truncated probability rows for the speculative rule, plus the
    # per-row emitted token and accept flag it writes. pt_d holds the target's
    # rows for this window, pd_d the draft's, one row per drafted position.
    var pt_d: DeviceBuffer[f32]
    var pd_d: DeviceBuffer[f32]
    # R4 (bench/moe-persist-protocol.md): one int32 zero, allocated and zeroed
    # once. moe_ffn used to make a host buffer and copy a 0 into hidx_d[0] on
    # every layer, 40 host buffers and 40 copies per token, and that copy also
    # clobbered the router's own idx[0] after the routed path had consumed it.
    # B4 stage 2: TRACE_TOK tokens x N_LAYERS x TOPK expert ids, device side.
    # B4 stage 2b (bench/moe-tier-protocol.md): routed experts in host RAM,
    # an LRU of resident experts in this cache. Empty when BARO_TIER is unset.
    var tier: ExpertTier
    var etrace_d: DeviceBuffer[DType.int32]
    var etrace_h: HostBuffer[DType.int32]
    var zidx_d: DeviceBuffer[DType.int32]
    var dids_d: DeviceBuffer[DType.int32]
    var sout_d: DeviceBuffer[DType.int32]
    var sacc_d: DeviceBuffer[DType.int32]
    var sout_h: HostBuffer[DType.int32]
    var sacc_h: HostBuffer[DType.int32]
    # Item 3-4, briefs/2026-09-16-sampling-all-models-lane.md: sparse penalty
    # lists and the top-N probability row (kernels/sample.mojo). pen_hist_h
    # mirrors this request's generated tokens (toks_d[n_prompt:st.pos+1])
    # back to host once per applicable window, small (bounded by tmax, not
    # VOCAB), so the distinct-id/count list can be built the same way
    # serve/sample_ref.mojo's apply_penalties already does.
    var pen_hist_h: HostBuffer[DType.int32]
    var pen_ids_h: HostBuffer[DType.int32]
    var pen_ids_d: DeviceBuffer[DType.int32]
    var pen_cnt_h: HostBuffer[DType.int32]
    var pen_cnt_d: DeviceBuffer[DType.int32]
    var pen_npen_h: HostBuffer[DType.int32]
    var pen_npen_d: DeviceBuffer[DType.int32]
    var topn_ids_d: DeviceBuffer[DType.int32]
    var topn_ids_h: HostBuffer[DType.int32]
    var topn_probs_d: DeviceBuffer[f32]
    var topn_probs_h: HostBuffer[f32]
    # The chosen token's own probability (amar_sample_row's Prob output,
    # already penalty-correct since apply_penalties ran first): the right
    # source for the response's chosen-token logprob, since the drawn token
    # is not always top-1 of the top-N list.
    var samp_prob_h: HostBuffer[f32]
    # JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
    # one grammar mask row, VOCAB bits. m == 1 only this round, so one row
    # is enough while grammar requests run with spec off.
    var gmask_h: HostBuffer[DType.uint64]
    var gmask_d: DeviceBuffer[DType.uint64]
    # Item 4 verification staging (coordinator review, 2026-09-16): the raw
    # pre-penalty row, VOCAB-wide, written by an enqueue_copy in
    # window.mojo when cfg.dump_pen; read and written to disk only by
    # serve/engine.mojo, the harness.
    var dump_row_h: HostBuffer[f32]


@fieldwise_init
struct WindowCfg(Copyable, Movable):
    var pack_q4: Bool
    var draft_q4: Bool
    var q4_off: Int
    var fr_k: Int
    var fr_off: Int
    var fr_ids_off: Int
    var e: Int
    var kcfg: Int
    var spec: Bool
    var spec_dbg: Bool
    # B4 stage 2 (bench/moe-locality-protocol.md): capture the router's top-8
    # per layer per token into a device plane, copied out once per request.
    # Off by default; a traced run is an instrumentation run, never a timed one.
    var expert_trace: Bool
    var serve: Bool
    var req_id: Int
    var prof: Bool
    var pf2: Bool
    var pf3: Bool
    var pf4: Bool
    var dump: Bool
    var dump4: Bool
    # BARO_DUMP_LAYER: which layer the dump4 sub-block captures come
    # from. Was hardcoded to 0; the first all-layer tdiff run put the
    # divergence at layer 31, which layer-0-only captures cannot reach.
    var dump_layer: Int
    var mega: Bool
    var att_split: Int
    var mega_win: Bool
    var dot3: Bool
    var pf_chunk: Int
    var pf_rows: Int
    var pf_tail: Int
    var n_total: Int
    var n_prompt: Int
    var sample: SampleParams
    # Item 4 verification (coordinator review, 2026-09-16): stage the raw
    # pre-penalty row into b.dump_row_h when set; serve/engine.mojo (the
    # harness) does the sync and file write, never this file.
    var dump_pen: Bool


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
    var pfx: List[Int]
    # JSON-enforcement item 1/2 (briefs/2026-09-16-json-enforcement-lane.md):
    # set explicitly per request by engine.mojo (like st.pos), not by
    # reset() -- building a Matcher needs the request's schema, which reset()
    # does not have. grammar is None for every request without
    # response_format, the overwhelming majority, so those pay nothing.
    var grammar: Optional[Matcher]
    var grammar_mask: Bitset
    # True while still scanning for "</think>" (item 2); grammar is not
    # consulted for fill_mask/accept until this goes False. Always False
    # when grammar is None.
    var grammar_pending_think: Bool
    var grammar_think_buf: List[UInt8]
    # True once the matcher reached an accepting state with no further
    # legal continuation possible in a well-formed document -- the decode
    # loop stops the request on this the same way it stops on a stop
    # sequence.
    var grammar_stop: Bool
    # Gate 4 receipt (bench/grammar-protocol.md): masked_draws is
    # incremented at every masked kernel launch, accepted at every
    # matcher.accept() call past the boundary. Equal at request end proves
    # the two sides of the host/device boundary stayed in sync -- a NOT-
    # RESIDENT-style silent skip on either side would desync them.
    var grammar_masked_draws: Int
    var grammar_accepted: Int

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
        self.pfx = [0, 0, 0, 0]


comptime moe_router_layout = row_major[N_EXP, H]()
comptime moe_one_layout = row_major[1]()
comptime moe_vec_layout = row_major[H]()
comptime moe_logits_layout = row_major[N_EXP]()
comptime moe_idx_layout = row_major[TOPK]()
comptime moe_expert_layout = row_major[TOPK, E_FFN]()
comptime moe_shared_layout = row_major[1, SH_FFN]()
comptime moe_expert_f_layout = row_major[TOPK * E_FFN]()
comptime moe_shared_f_layout = row_major[SH_FFN]()
comptime ssm_gate_w_layout = row_major[NH_V, H]()
comptime ssm_gate_o_layout = row_major[NH_V]()
comptime moe_expert_flat_layout = row_major[TOPK * E_FFN]()
comptime moe_shared_flat_layout = row_major[SH_FFN]()
comptime moe_inner_layout = row_major[NH_V * SSTATE]()
comptime moe_inner_m_layout = row_major[1, NH_V * SSTATE]()
comptime moe_om_layout = row_major[1, NH_V, SSTATE]()
def moe_ffn(ctx: DeviceContext, mut b: WindowBufs, Xm: TileTensor[f32, type_of(xm_layout), MutAnyOrigin], CurBm: TileTensor[bf16, type_of(xm_layout), MutAnyOrigin], w: Int, layer: Int, routed_base: Int, extra_base: Int, trace_slot: Int = -1) raises:
    var Xres = row_f32(ctx, b.x_d, 0, H, moe_vec_layout)
    var X = row_f32(ctx, b.p_h_d, 0, H, moe_vec_layout)
    var X2 = row_f32(ctx, b.p_h_d, 0, H, h2_layout)
    var sigmoid = row_f32(ctx, b.p_v_d, 0, 1, moe_one_layout)
    var Router = tens_f32(ctx, b.wbuf, b.off[extra_base], N_EXP * H, moe_router_layout)
    var Logits = row_f32(ctx, b.logits_d, 0, N_EXP, moe_logits_layout)
    var Idx = TileTensor[DType.int32, type_of(moe_idx_layout), MutAnyOrigin](b.hidx_d, moe_idx_layout)
    var Wt = TileTensor[f32, type_of(moe_idx_layout), MutAnyOrigin](b.hmax_d, moe_idx_layout)
    comptime k_router = amar_matmul_skinny_m1_row[f32, 2, type_of(h2_layout), type_of(moe_router_layout), type_of(moe_logits_layout)]
    ctx.enqueue_function[k_router](X2, Router, Logits, Int32(N_EXP), Int32(H), grid_dim=ceildiv(N_EXP, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[amar_moe_router_top8_sig[type_of(moe_logits_layout), type_of(moe_idx_layout), type_of(moe_idx_layout), type_of(moe_vec_layout), type_of(h_layout), type_of(moe_one_layout)]](
        Logits, Idx, Wt, X, tens_f32(ctx, b.wbuf, b.off[extra_base + 4], H, h_layout), sigmoid, Int32(H), grid_dim=1, block_dim=MOE_THREADS // MOE_WAVES)
    if trace_slot >= 0 and trace_slot < TRACE_TOK:
        # Device to device, no sync: the ids are read out once per request.
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.int32](ctx, b.etrace_d.unsafe_ptr().unsafe_offset((trace_slot * N_LAYERS + layer) * TOPK), TOPK, owning=False),
            src_buf=DeviceBuffer[DType.int32](ctx, b.hidx_d.unsafe_ptr(), TOPK, owning=False),
        )
    var routed = row_f32(ctx, b.p_qf_d, 0, H, moe_vec_layout)
    var routed_h_flat = TileTensor[bf16, type_of(moe_expert_flat_layout), MutAnyOrigin](b.fgb_d, moe_expert_flat_layout)
    # B4 stage 2b: with the tier on, the gather reads the VRAM expert cache
    # instead of the pack, and Idx holds cache SLOTS instead of expert ids
    # (prepare rewrites it in place, after fetching any miss). The kernels are
    # untouched: their expert stride is a runtime argument.
    var He = TileTensor[DType.int32, type_of(moe_idx_layout), MutAnyOrigin](b.tier.hoste_d, moe_idx_layout)
    if b.tier.active and b.tier.zc:
        # Zero-copy misses (bench/moe-tier-protocol.md): rows of a missed
        # expert come straight from the pinned store and the wave fills the
        # cache slot with what it read; hits are the same bytes as before.
        b.tier.prepare(ctx, layer, b.hidx_d)
        ctx.enqueue_function[moe_gate_up_q4k_zc[TOPK, E_FFN, type_of(xm_layout), type_of(moe_idx_layout), type_of(moe_expert_flat_layout), bf16]](
            CurBm, b.tier.cache.unsafe_ptr().unsafe_offset(b.tier.gate_base(layer)), Idx, He,
            b.tier.store.unsafe_ptr().unsafe_offset(b.tier.geom[layer].gate_off), routed_h_flat, Int32(H),
            Int32(b.tier.up_offset(layer)), Int32(b.tier.geom[layer].up_off - b.tier.geom[layer].gate_off),
            grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    elif b.tier.active:
        b.tier.prepare(ctx, layer, b.hidx_d)
        ctx.enqueue_function[moe_gate_up_q4k_pack[TOPK, E_FFN, type_of(xm_layout), type_of(moe_idx_layout), type_of(moe_expert_flat_layout), bf16]](
            CurBm, b.tier.cache.unsafe_ptr().unsafe_offset(b.tier.gate_base(layer)), Idx, routed_h_flat, Int32(H), Int32(b.tier.up_offset(layer)), grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    else:
        ctx.enqueue_function[moe_gate_up_q4k_pack[TOPK, E_FFN, type_of(xm_layout), type_of(moe_idx_layout), type_of(moe_expert_flat_layout), bf16]](
            CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[routed_base]), Idx, routed_h_flat, Int32(H), Int32(b.off[routed_base + 1] - b.off[routed_base]), grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    var routed_h = TileTensor[bf16, type_of(moe_expert_layout), MutAnyOrigin](b.fgb_d, moe_expert_layout)
    if layer == 34 or layer == 38 or layer == 39:
        ctx.enqueue_function[amar_moe_down_q6k[TOPK, E_FFN, type_of(moe_expert_layout), type_of(moe_idx_layout), type_of(moe_idx_layout), type_of(moe_vec_layout)]](
            routed_h, (b.tier.cache.unsafe_ptr().unsafe_offset(b.tier.down_base(layer)) if b.tier.active else b.wbuf.unsafe_ptr().unsafe_offset(b.off[routed_base + 2])), Idx, Wt, routed, Int32(H), grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
    elif b.tier.active and b.tier.zc:
        ctx.enqueue_function[amar_moe_down_q4k_zc[TOPK, E_FFN, type_of(moe_expert_layout), type_of(moe_idx_layout), type_of(moe_idx_layout), type_of(moe_vec_layout)]](
            routed_h, b.tier.cache.unsafe_ptr().unsafe_offset(b.tier.down_base(layer)), Idx, He,
            b.tier.store.unsafe_ptr().unsafe_offset(b.tier.geom[layer].down_off), Wt, routed, Int32(H),
            grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
    else:
        ctx.enqueue_function[amar_moe_down_q4k[TOPK, E_FFN, type_of(moe_expert_layout), type_of(moe_idx_layout), type_of(moe_idx_layout), type_of(moe_vec_layout)]](
            routed_h, (b.tier.cache.unsafe_ptr().unsafe_offset(b.tier.down_base(layer)) if b.tier.active else b.wbuf.unsafe_ptr().unsafe_offset(b.off[routed_base + 2])), Idx, Wt, routed, Int32(H), grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
    var idx0 = TileTensor[DType.int32, type_of(moe_one_layout), MutAnyOrigin](b.zidx_d, moe_one_layout)
    var shared_h_flat = TileTensor[bf16, type_of(moe_shared_flat_layout), MutAnyOrigin](b.fgbp_d, moe_shared_flat_layout)
    var shared_gate = b.wbuf.unsafe_ptr().unsafe_offset(b.off[extra_base + 1])
    var shared_up = b.wbuf.unsafe_ptr().unsafe_offset(b.off[extra_base + 2])
    ctx.enqueue_function[moe_gate_up_q8_0[1, SH_FFN, type_of(xm_layout), type_of(moe_one_layout), type_of(moe_shared_flat_layout), bf16]](
        CurBm, shared_gate, idx0, shared_h_flat, Int32(H), Int32((H // 32) * 34), Int32(shared_up - shared_gate), grid_dim=ceildiv(SH_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    var shared_h = TileTensor[bf16, type_of(moe_shared_layout), MutAnyOrigin](b.fgbp_d, moe_shared_layout)
    ctx.enqueue_function[moe_down_q8_0_res[1, SH_FFN, type_of(moe_shared_layout), type_of(moe_one_layout), type_of(moe_one_layout), type_of(moe_vec_layout), type_of(moe_vec_layout)]](
        shared_h, b.wbuf.unsafe_ptr().unsafe_offset(b.off[extra_base + 3]), idx0, sigmoid, routed, Xres, Int32(H), Int32((SH_FFN // 32) * 34), grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)


# A1 (bench/spec-sample-protocol.md): the draft head's own draw, under
# sampling. Two launches on one logits row: the truncated q row into the draft
# plane's row j, and one token drawn from that same truncated distribution.
#
# The draft's seed is mixed, because amar_sample_row uses the same Philox
# streams for every caller: an unmixed draft at (seed, counter = pos + j) would
# share a Gumbel key with the target row drawn at the same counter, which
# correlates q and p exactly where the accept rule assumes they are
# independent. The constant is the golden-ratio mix, no significance beyond
# being a fixed odd word.
comptime DRAFT_SEED_MIX = UInt64(0x9E3779B97F4A7C15)


def draft_draw(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, pos: Int, j: Int, logits_row: Int
) raises:
    var Xrow = row_f32(ctx, b.logits_d, logits_row * VOCAB, VOCAB, vrow_layout)
    var Qrow = row_f32(ctx, b.pd_d, j * VOCAB, VOCAB, vrow_layout)
    var Dtok = TileTensor(b.dtok_d, dtok_layout)
    var Prob = TileTensor(b.hmax_d, dtok_layout)
    ctx.enqueue_function[sample_probs_1](
        Xrow, Qrow, Int32(VOCAB), Float32(cfg.sample.temperature), Int32(cfg.sample.top_k),
        Float32(cfg.sample.top_p), Float32(cfg.sample.min_p), grid_dim=1, block_dim=SAMP_THREADS,
    )
    ctx.enqueue_function[sample_row_1](
        Xrow, Dtok, Prob, Int32(VOCAB), Float32(cfg.sample.temperature), Int32(cfg.sample.top_k),
        Float32(cfg.sample.top_p), Float32(cfg.sample.min_p),
        cfg.sample.seed ^ DRAFT_SEED_MIX, UInt64(pos + j), grid_dim=1, block_dim=SAMP_THREADS,
    )


def step_window(ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState) raises:
    var Xm = TileTensor(b.x_d, xm_layout)
    var CurBm = TileTensor(b.curb_d, xm_layout)
    var moe_z_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE)
    var moe_res_d = ctx.enqueue_create_buffer[bf16](NH_V * SSTATE)
    var Logitsm = TileTensor(b.logits_d, vm_layout)
    var Toks = TileTensor(b.toks_d, toks_layout)
    var Pv = TileTensor(b.p_v_d, p_v)
    var Embd = tens_bf16(ctx, b.wbuf, b.off[0], VOCAB * H, emb_layout)
    var ConvStateAll = TileTensor(b.convstate_d, csall_layout)
    var SStateAll = TileTensor(b.sstate_d, ssall_layout)
    if st.pos < cfg.pf_rows:
        var mc = min(cfg.pf_chunk, cfg.pf_rows - st.pos)
        prefill_forward(ctx, b.wbuf, b.off, cfg.pack_q4, mc, st.pos, st.ring, cfg.prof, b.toks_d, b.convstate_d, b.sstate_d, b.kc_d, b.vc_d, b.kvtab_d,
            b.xp_d, b.curbp_d, b.qkvp_d, b.zp_d, b.arp_d, b.brp_d, b.egp_d, b.betap_d, b.convp_d, b.sop_d, b.resbp_d, b.qfp_d,
            b.qp_d, b.gatep_d, b.kp_d, b.vp_d, b.aop_d, b.gp_d, b.up_d, b.fgbp_d, st.pfx)
        st.ring = (st.ring + mc) % SLOTS
        st.pos_prev = st.pos
        st.pos += mc
        if st.pos == cfg.pf_rows:
            if cfg.prof:
                var other = st.pfx[3] - st.pfx[0] - st.pfx[1] - st.pfx[2]
                print("prefill split s: attn", Float64(st.pfx[0]) / 1e9, " ssm_scan", Float64(st.pfx[1]) / 1e9, " gemm", Float64(st.pfx[2]) / 1e9, " other", Float64(other) / 1e9, " total", Float64(st.pfx[3]) / 1e9)
            var Xtail = row_f32(ctx, b.xp_d, (mc - cfg.pf_tail) * H, MROWS * H, xm_layout)
            ctx.enqueue_function[rms_m](
                Xtail, tens_f32(ctx, b.wbuf, b.off[1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4], H, h_layout),
                TileTensor(b.hn_d, xm_layout), Int32(H), Float32(1e-6), grid_dim=cfg.pf_tail, block_dim=256,
            )
            st.pos_prev = st.pos - cfg.pf_tail
    else:
        var m = 1
        var win_spec = False
        # Item 3-4, briefs/2026-09-16-sampling-all-models-lane.md: set when
        # the plain (non-spec) sampled branch below actually ran, so the
        # print block downstream knows a valid Prob/top-N sits in
        # b.hmax_d[0]/b.topn_*_h[0, :].
        var did_sample_lp = False
        comptime if MEGA_ALLOWED:
            if st.pos + 1 < cfg.n_prompt:
                m = min(MROWS, cfg.n_prompt - 1 - st.pos)
        else:
            # Gate 2 is deliberately the smallest decode path: one row per
            # call, including prompt replay.
            m = 1
        if cfg.spec and not (st.pos + 1 < cfg.n_prompt):
            # process: tokens st.pos_prev+1..pos through the draft head with the
            # trunk's h rows (row r = h(st.pos_prev + r)); last row = draft step 0.
            var nproc = st.pos - st.pos_prev
            if cfg.prof:
                ctx.synchronize()
                st.tp = perf_counter_ns()
            var hn_rows = DeviceBuffer[f32](ctx, b.hn_d.unsafe_ptr(), nproc * H, owning=False)
            blk32_forward(ctx, b.wbuf, b.off, cfg.e, nproc, st.pos_prev + 1, st.pos_prev + 1, True, hn_rows,
                b.x_d, b.curb_d, b.qf_d, b.q_d, b.k_d, b.v_d, b.gate_d, b.ao_d, b.resb_d, b.fgb_d, b.p_qf_d, b.p_kv_d, b.p_h_d,
                b.p_ffn_d, b.p_ffn2_d, b.p_v_d, b.logits_d, b.cc_d, b.de_d, b.hd_d, b.kc32_d, b.vc32_d, b.kvtab_d, b.toks_d, b.dtok_d,
                cfg.pf3, st.p3, cfg.draft_q4, cfg.q4_off, cfg.pack_q4, cfg.fr_k, cfg.fr_off, cfg.fr_ids_off)
            var Dtok = TileTensor(b.dtok_d, dtok_layout)
            # A1 (bench/spec-sample-protocol.md). Under sampling the draft is a
            # DRAW from q, not q's argmax, and the whole q row has to survive to
            # verify time: the residual norm(max(0, p - q)) needs every entry,
            # not just q(x). blk32_forward's own argmax is ignored on this path.
            # The draft stream uses a different seed from the target's so the
            # two never share a Gumbel key at the same (counter, row).
            var samp_spec = cfg.sample.temperature > 0
            if samp_spec:
                draft_draw(ctx, b, cfg, st.pos, 0, nproc - 1)
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(0), Int32(st.pos + 1), Int32(1), grid_dim=1, block_dim=32)
            else:
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(nproc - 1), Int32(st.pos + 1), Int32(1), grid_dim=1, block_dim=32)
            m = min(cfg.kcfg + 1, cfg.n_total - 1 - st.pos)
            if cfg.prof:
                ctx.synchronize()
                st.pf_proc += Int(perf_counter_ns() - st.tp)
                st.tp = perf_counter_ns()
            var hrow = nproc - 1
            for j in range(1, m - 1):
                var hd_row = DeviceBuffer[f32](ctx, b.hd_d.unsafe_ptr().unsafe_offset(hrow * H), H, owning=False)
                blk32_forward(ctx, b.wbuf, b.off, cfg.e, 1, st.pos + j, st.pos + j, True, hd_row,
                    b.x_d, b.curb_d, b.qf_d, b.q_d, b.k_d, b.v_d, b.gate_d, b.ao_d, b.resb_d, b.fgb_d, b.p_qf_d, b.p_kv_d, b.p_h_d,
                    b.p_ffn_d, b.p_ffn2_d, b.p_v_d, b.logits_d, b.cc_d, b.de_d, b.hd_d, b.kc32_d, b.vc32_d, b.kvtab_d, b.toks_d, b.dtok_d,
                    cfg.pf3, st.p3, cfg.draft_q4, cfg.q4_off, cfg.pack_q4, cfg.fr_k, cfg.fr_off, cfg.fr_ids_off)
                if samp_spec:
                    draft_draw(ctx, b, cfg, st.pos, j, 0)
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(0), Int32(st.pos + j + 1), Int32(1), grid_dim=1, block_dim=32)
                hrow = 0
            st.n_drafted += m - 1
            win_spec = True
            st.n_spec_windows += 1
            if cfg.prof:
                ctx.synchronize()
                st.pf_draft += Int(perf_counter_ns() - st.tp)

        comptime if MEGA_ALLOWED:
            ctx.enqueue_function[embed_k](
                Embd, Xm, Toks, Int32(st.pos), Int32(H),
                grid_dim=(ceildiv(H, 256), m), block_dim=256,
            )
        else:
            ctx.enqueue_function[moe_embed_q8_0_pos[type_of(xm_layout), type_of(toks_layout)]](
                b.wbuf.unsafe_ptr().unsafe_offset(b.off[0]), Xm, Toks, Int32(st.pos), Int32(H),
                Int32((H // 32) * 34), grid_dim=(ceildiv(H, 256), m), block_dim=256,
            )

        var w = 1
        var moe_w = 1
        var ssm_i = 0
        var att_i = 0
        var plain_head = cfg.sample.temperature <= 0 and cfg.sample.presence_penalty == 0 and cfg.sample.frequency_penalty == 0 and cfg.sample.top_logprobs <= 0 and not st.grammar.__bool__()
        var use_mega = cfg.mega and m == 1 and not win_spec and st.pos + 1 >= cfg.n_prompt
        var use_mega_win = cfg.mega_win and win_spec and m == MEGA_MR
        comptime if not MEGA_ALLOWED:
            use_mega_win = False
            if use_mega:
                ctx.enqueue_function[mega_moe_k](
                    b.wbuf.unsafe_ptr(), b.off_d.unsafe_ptr(), b.x_d.unsafe_ptr(), b.curb_d.unsafe_ptr(),
                    b.logits_d.unsafe_ptr(), b.p_h_d.unsafe_ptr(), b.qkv_d.unsafe_ptr(), moe_z_d.unsafe_ptr(),
                    b.araw_d.unsafe_ptr(), b.braw_d.unsafe_ptr(), b.eg_d.unsafe_ptr(), b.beta_d.unsafe_ptr(),
                    b.conv_d.unsafe_ptr(), b.so_d.unsafe_ptr(), moe_res_d.unsafe_ptr(), ConvStateAll, SStateAll,
                    b.qf_d.unsafe_ptr(), b.k_d.unsafe_ptr(), b.v_d.unsafe_ptr(), b.q_d.unsafe_ptr(), b.gate_d.unsafe_ptr(),
                    b.ao_d.unsafe_ptr(), b.resb_d.unsafe_ptr(), b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(), b.p_ffn_d.unsafe_ptr(),
                    b.hidx_d.unsafe_ptr(), b.hmax_d.unsafe_ptr(), b.p_v_d.unsafe_ptr(), b.fgb_d.unsafe_ptr(), b.fgbp_d.unsafe_ptr(),
                    b.ctr_d.unsafe_ptr(), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                    Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1 if cfg.dump else 0), Int32(cfg.att_split),
                    grid_dim=MOE_G, block_dim=ROW_THREADS,
                )
        else:
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
                        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(),
                        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(1 if cfg.dump else 0), Int32(1 if plain_head else 0), Int32(cfg.att_split), grid_dim=MEGA_G, block_dim=ROW_THREADS,
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
                        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(),
                        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(1 if cfg.dump else 0), Int32(1 if plain_head else 0), Int32(cfg.att_split), grid_dim=MEGA_G, block_dim=ROW_THREADS,
                    )
                elif cfg.pack_q4:
                    ctx.enqueue_function[mega_win_q4_k](
                        b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
                        TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
                        TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
                        TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
                        TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
                        TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
                        TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
                        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(),
                        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(m), Int32(0), Int32(0), Int32(cfg.att_split), grid_dim=MEGA_G_WIN, block_dim=ROW_THREADS,
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
                        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(),
                        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
                        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
                        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
                        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(m), Int32(0), Int32(0), Int32(cfg.att_split), grid_dim=MEGA_G_WIN, block_dim=ROW_THREADS,
                    )
        for layer in range(0 if (use_mega or use_mega_win) else N_LAYERS):
            if cfg.prof:
                ctx.synchronize()
                st.tp = perf_counter_ns()
                st.tq = st.tp
            # -- attention / ssm sub-block --
            var moe_base = moe_w
            var decode_base = w
            comptime if not MEGA_ALLOWED:
                decode_base = moe_base
            comptime if MEGA_ALLOWED:
                ctx.enqueue_function[rmsc_k](
                    Xm, tens_f32(ctx, b.wbuf, b.off[decode_base], H, h_layout), CurBm,
                    Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
                )
            else:
                ctx.enqueue_function[rmsc2_k](
                    Xm, tens_f32(ctx, b.wbuf, b.off[decode_base], H, h_layout), CurBm,
                    row_f32(ctx, b.logits_d, 0, H, h2_layout),
                    Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
                )

            if is_attn(layer):
                var Wqq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 1], H * QF, q_h_qf)
                var Wqs = tens_q8s(ctx, b.wbuf, b.off[decode_base + 1], H * QF, s_h_qf)
                var Wkq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 2], H * KV, q_h_kv)
                var Wks = tens_q8s(ctx, b.wbuf, b.off[decode_base + 2], H * KV, s_h_kv)
                var Wvq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 3], H * KV, q_h_kv)
                var Wvs = tens_q8s(ctx, b.wbuf, b.off[decode_base + 3], H * KV, s_h_kv)
                var Qn = tens_f32(ctx, b.wbuf, b.off[decode_base + 4], HD, hd_layout)
                var Kn = tens_f32(ctx, b.wbuf, b.off[decode_base + 5], HD, hd_layout)
                var Woq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 6], H * H, q_h_h)
                var Wos = tens_q8s(ctx, b.wbuf, b.off[decode_base + 6], H * H, s_h_h)
                var Pqf = TileTensor(b.p_qf_d, p_qf)
                var Pkv = TileTensor(b.p_kv_d, p_kv)
                var Ph = TileTensor(b.p_h_d, p_h)
                var Qfm = TileTensor(b.qf_d, qfm_layout)
                var Q = TileTensor(b.q_d, qm_layout)
                var Gate = TileTensor(b.gate_d, attflat_layout)
                var Kflat = TileTensor(b.k_d, kvm_flat)
                var Khd = TileTensor(b.k_d, kvm_layout)
                var Vflat = TileTensor(b.v_d, kvm_flat)
                var Vhd = TileTensor(b.v_d, kvm_layout)
                var kcb = DeviceBuffer[KVT](
                    ctx, b.kc_d.unsafe_ptr(),
                    b.kvpool, owning=False,
                )
                var vcb = DeviceBuffer[KVT](
                    ctx, b.vc_d.unsafe_ptr(),
                    b.kvpool, owning=False,
                )
                var Kc = TileTensor(kcb, cache_layout)
                var Vc = TileTensor(vcb, cache_layout)
                var Ao = TileTensor(b.ao_d, qm_layout)
                var Aoflat = TileTensor(b.ao_d, attflat_layout)
                var AoB = TileTensor(b.resb_d, attflat_layout)
                var AoBm = TileTensor(b.resb_d, attm_layout)

                comptime if MEGA_ALLOWED:
                    gemm_w[QF, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pqf, m)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1[type_of(h_layout), type_of(xm_layout)]](
                        CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 1]), row_f32(ctx, b.qf_d, 0, QF, h_layout), Int32(QF), Int32(H), Int32((H // 32) * 34),
                        grid_dim=ceildiv(QF, 8), block_dim=256)
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slots 8-11: llama Qcur_full-N, the fused q+gate projection
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(8 * H), QF, owning=False), src_buf=DeviceBuffer[f32](ctx, b.qf_d.unsafe_ptr(), QF, owning=False))
                comptime if MEGA_ALLOWED:
                    gemm_w[KV, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Pkv, m)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1[type_of(h_layout), type_of(xm_layout)]](
                        CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 2]), row_f32(ctx, b.k_d, 0, KV, h_layout), Int32(KV), Int32(H), Int32((H // 32) * 34),
                        grid_dim=ceildiv(KV, 8), block_dim=256)
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                comptime if MEGA_ALLOWED:
                    gemm_w[KV, H](ctx, CurBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Pkv, m)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1[type_of(h_layout), type_of(xm_layout)]](
                        CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 3]), row_f32(ctx, b.v_d, 0, KV, h_layout), Int32(KV), Int32(H), Int32((H // 32) * 34),
                        grid_dim=ceildiv(KV, 8), block_dim=256)
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[r_kv](Pkv, Vflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, m), block_dim=HD)
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
                    ctx.enqueue_function[hrms_kv](Khd, Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
                else:
                    ctx.enqueue_function[hrr_q](Q, Qn, Float32(1e-6), Int32(st.pos), Int32(NQH), grid_dim=m * NQH, block_dim=HD)
                    ctx.enqueue_function[hrr_kv](Khd, Kn, Float32(1e-6), Int32(st.pos), Int32(NKVH), grid_dim=m * NKVH, block_dim=HD)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slots 12-13: llama Qcur_normed-N, after the per-head q norm
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(12 * H), ATT, owning=False), src_buf=DeviceBuffer[f32](ctx, b.q_d.unsafe_ptr(), ATT, owning=False))
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slot 16: llama Kcur_normed-N, after the per-head k norm
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(16 * H), KV, owning=False), src_buf=DeviceBuffer[f32](ctx, b.k_d.unsafe_ptr(), KV, owning=False))
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[rope_q](Q, Int32(st.pos), Int32(NQH), grid_dim=(NQH, m), block_dim=32)
                    ctx.enqueue_function[rope_k](Khd, Int32(st.pos), Int32(NKVH), grid_dim=(NKVH, m), block_dim=32)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slots 14-15: llama Qcur-N, q after rope
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(14 * H), ATT, owning=False), src_buf=DeviceBuffer[f32](ctx, b.q_d.unsafe_ptr(), ATT, owning=False))
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slot 17: llama Kcur-N, k after rope
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(17 * H), KV, owning=False), src_buf=DeviceBuffer[f32](ctx, b.k_d.unsafe_ptr(), KV, owning=False))
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[append_k](Kc, Khd, b.kvtab_d.unsafe_ptr(), Int32(st.pos), Int32(att_i), grid_dim=(NKVH, m), block_dim=HD)
                    ctx.enqueue_function[append_k](Vc, Vhd, b.kvtab_d.unsafe_ptr(), Int32(st.pos), Int32(att_i), grid_dim=(NKVH, m), block_dim=HD)
                else:
                    ctx.enqueue_function[append2_k](Kc, Vc, Khd, Vhd, b.kvtab_d.unsafe_ptr(), Int32(st.pos), Int32(att_i), grid_dim=(NKVH, m, 2), block_dim=HD)
                if st.pos + 1 > cfg.att_split:
                    var dns = dattn_nsplit[HD, DATT_NLD, NKVH](st.pos + 1, m, MEGA_G)
                    var Pa = TileTensor(b.p_ffn_d, p_att_layout)
                    ctx.enqueue_function[datt_k](Q, Kc, Vc, Ao, Pa, b.kvtab_d.unsafe_ptr(), Int32(st.pos + 1), Int32(dns), Float32(0.0625), Int32(att_i), grid_dim=(NKVH, dns, m), block_dim=ROW_THREADS)
                    if dns > 1:
                        ctx.enqueue_function[dcomb_k](Pa, Ao, Int32(dns), grid_dim=m * NQH, block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[att_k](Q, Kc, Vc, Ao, b.kvtab_d.unsafe_ptr(), Int32(st.pos + 1), Float32(0.0625), Int32(att_i), grid_dim=(NQH, m), block_dim=HD)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slots 4-5: attention output before the gate and the output
                    # projection, ATT=4096 floats, i.e. llama's attn_output-N.
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(4 * H), ATT, owning=False), src_buf=DeviceBuffer[f32](ctx, b.ao_d.unsafe_ptr(), ATT, owning=False))
                ctx.enqueue_function[gmul_k](Aoflat, Gate, AoB, Int32(m * ATT), grid_dim=ceildiv(m * ATT, 256), block_dim=256)
                comptime if MEGA_ALLOWED:
                    gemm_w[H, ATT](ctx, AoBm, b.wbuf, b.off[w + 6], cfg.pack_q4, Ph, m)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1_add[type_of(h_layout), type_of(attm_layout), type_of(h_layout)]](
                        AoBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 6]), row_f32(ctx, b.p_h_d, 0, H, h_layout), row_f32(ctx, b.x_d, 0, H, h_layout), Int32(H), Int32(ATT), Int32((ATT // 32) * 34),
                        grid_dim=ceildiv(H, 8), block_dim=256)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slot 6: the output projection result, the value added to
                    # the residual. Splits "attention is wrong" from "o_proj is
                    # wrong" in one run.
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(6 * H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.p_h_d.unsafe_ptr(), H, owning=False))
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                att_i += 1
                w += 7
            else:
                var Wqkvq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 1], H * CONV, q_h_qf)
                var Wqkvs = tens_q8s(ctx, b.wbuf, b.off[decode_base + 1], H * CONV, s_h_qf)
                var Wzq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 2], H * H, q_h_h)
                var Wzs = tens_q8s(ctx, b.wbuf, b.off[decode_base + 2], H * H, s_h_h)
                var Waq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 3], H * NH_V, q_h_32)
                var Was = tens_q8s(ctx, b.wbuf, b.off[decode_base + 3], H * NH_V, s_h_32)
                var Wbq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 4], H * NH_V, q_h_32)
                var Wbs = tens_q8s(ctx, b.wbuf, b.off[decode_base + 4], H * NH_V, s_h_32)
                var Cw = tens_f32(ctx, b.wbuf, b.off[decode_base + 5], CONV * 4, cw_layout)
                var SsmA = tens_f32(ctx, b.wbuf, b.off[decode_base + 6], NH_V, g32_layout)
                var DtB = tens_f32(ctx, b.wbuf, b.off[decode_base + 7], NH_V, g32_layout)
                var Nw = tens_f32(ctx, b.wbuf, b.off[decode_base + 8], SSTATE, n128_layout)
                var Wsoutq = tens_q8q(ctx, b.wbuf, b.off[decode_base + 9], H * H, q_h_h)
                var Wsouts = tens_q8s(ctx, b.wbuf, b.off[decode_base + 9], H * H, s_h_h)
                var Pq = TileTensor(b.p_qf_d, p_qf)
                var Ph = TileTensor(b.p_h_d, p_h)
                var Pab = TileTensor(b.p_32_d, p_32)
                var Pab2 = TileTensor(b.p_32b_d, p_32)
                var Qkvm = TileTensor(b.qkv_d, qfm_layout)
                var Zm = TileTensor[f32, type_of(moe_inner_m_layout), MutAnyOrigin](moe_z_d, moe_inner_m_layout)
                var Zflat = row_f32(ctx, moe_z_d, 0, NH_V * SSTATE, moe_inner_layout)
                var ZmOld = TileTensor(b.z_d, xm_layout)
                var Eg = TileTensor(b.eg_d, g32m_layout)
                var Beta = TileTensor(b.beta_d, g32m_layout)
                var Conv = TileTensor(b.conv_d, convm_layout)
                var So = TileTensor(b.so_d, om_layout)
                var ResBm = TileTensor[bf16, type_of(moe_inner_m_layout), MutAnyOrigin](moe_res_d, moe_inner_m_layout)
                var ResBmOld = TileTensor(b.resb_d, xm_layout)

                comptime if MEGA_ALLOWED:
                    gemm_w[CONV, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pq, m)
                    gemm_w[H, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Ph, m)
                    gemm_w[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Pab, m)
                    gemm_w[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 4], cfg.pack_q4, Pab2, m)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1[type_of(h_layout), type_of(xm_layout)]](
                        CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 1]), row_f32(ctx, b.qkv_d, 0, CONV, h_layout), Int32(CONV), Int32(H), Int32((H // 32) * 34),
                        grid_dim=ceildiv(CONV, 8), block_dim=256)
                    ctx.enqueue_function[moe_matmul_q8_0_m1[type_of(h_layout), type_of(xm_layout)]](
                        CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 2]), row_f32(ctx, b.p_h_d, 0, H, h_layout), Int32(H), Int32(H), Int32((H // 32) * 34),
                        grid_dim=ceildiv(H, 8), block_dim=256)
                    # ssm_alpha/ssm_beta are F32 in the MoE pack; the dense
                    # pack stores them q4. Reading f32 bytes through the q8_0
                    # kernel made every beta sigmoid saturate to exactly
                    # 0.0/1.0 (W3 gate 2 bug 4, 2026-09-12). f32 weights need
                    # the skinny f32 matmul and an f32 copy of the activation.
                    # R4: these two used to be zeroed here, 60 memsets per
                    # token. amar_ssm_reduce_gates sums SPLITK partials, and on
                    # this profile only partial 0 row 0 is ever written (by the
                    # two skinny f32 matmuls below, which WRITE their output
                    # rather than accumulate into it), so the rest of the plane
                    # has to be zero but never stops being zero. It is zeroed
                    # once at allocation instead. The dense profile's gemm_w
                    # partial writer is not compiled into this build
                    # (MEGA_ALLOWED is False for the MoE), which is what makes
                    # that safe; if a future path writes these planes per
                    # layer, the zeroing has to come back with it.
                    var Xg = row_f32(ctx, b.logits_d, 0, H, h2_layout)
                    ctx.enqueue_function[amar_matmul_skinny_m1_row2[f32, 2, type_of(h2_layout), type_of(ssm_gate_w_layout), type_of(ssm_gate_o_layout)]](
                        Xg, tens_f32(ctx, b.wbuf, b.off[moe_base + 3], NH_V * H, ssm_gate_w_layout),
                        tens_f32(ctx, b.wbuf, b.off[moe_base + 4], NH_V * H, ssm_gate_w_layout),
                        row_f32(ctx, b.p_32_d, 0, NH_V, ssm_gate_o_layout), row_f32(ctx, b.p_32b_d, 0, NH_V, ssm_gate_o_layout),
                        Int32(NH_V), Int32(H), grid_dim=ceildiv(2 * NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[r_qf](Pq, Qkvm, Int32(m), Int32(CONV), grid_dim=ceildiv(m * CONV, 256), block_dim=256)
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[r_h](Ph, ZmOld, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1[type_of(moe_inner_layout), type_of(xm_layout)]](
                        CurBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[moe_base + 2]), Zflat, Int32(NH_V * SSTATE), Int32(H), Int32((H // 32) * 34),
                        grid_dim=ceildiv(NH_V * SSTATE, 8), block_dim=256)

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
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slot 4: conv output after l2norm (llama conv_output_silu /
                    # Qcur_normed). slot 5: the two gate vectors, Eg then Beta,
                    # NH_V each (llama a_softplus / beta_sigmoid).
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(8 * H), CONV, owning=False), src_buf=DeviceBuffer[f32](ctx, b.conv_d.unsafe_ptr(), CONV, owning=False))
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(5 * H), NH_V, owning=False), src_buf=DeviceBuffer[f32](ctx, b.eg_d.unsafe_ptr(), NH_V, owning=False))
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(5 * H + NH_V), NH_V, owning=False), src_buf=DeviceBuffer[f32](ctx, b.beta_d.unsafe_ptr(), NH_V, owning=False))
                delta_dispatch(ctx, SStateAll, Conv, Eg, Beta, So, Int32(st.ring), Int32(ssm_i), Int32(SLOTS), m)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slot 6: delta-scan output, before the output gate.
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(6 * H), NH_V * SSTATE, owning=False), src_buf=DeviceBuffer[f32](ctx, b.so_d.unsafe_ptr(), NH_V * SSTATE, owning=False))
                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[4] += Int(nw - st.tq)
                    st.tq = nw
                comptime if MEGA_ALLOWED:
                    ctx.enqueue_function[gated_k](So, ZmOld, Nw, ResBmOld, Int32(m), grid_dim=NH_V, block_dim=SSTATE)
                else:
                    ctx.enqueue_function[amar_ssm_gated_out_bf16[type_of(om_layout), type_of(moe_inner_m_layout), type_of(n128_layout), type_of(moe_inner_m_layout)]](
                        So, Zm, Nw, ResBm, Int32(m), grid_dim=NH_V, block_dim=SSTATE)

                if cfg.pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    st.pc[5] += Int(nw - st.tq)
                    st.tq = nw
                comptime if MEGA_ALLOWED:
                    gemm_w[H, H](ctx, ResBmOld, b.wbuf, b.off[w + 9], cfg.pack_q4, Ph, m)
                else:
                    ctx.enqueue_function[moe_matmul_q8_0_m1_add[type_of(h_layout), type_of(moe_inner_m_layout), type_of(h_layout)]](
                        ResBm, b.wbuf.unsafe_ptr().unsafe_offset(b.off[decode_base + 9]), row_f32(ctx, b.p_h_d, 0, H, h_layout), row_f32(ctx, b.x_d, 0, H, h_layout), Int32(H), Int32(NH_V * SSTATE), Int32((NH_V * SSTATE // 32) * 34),
                        grid_dim=ceildiv(H, 8), block_dim=256)
                if cfg.dump4 and cfg.dump and m == 1 and layer == cfg.dump_layer and st.pos + 1 >= cfg.n_prompt:
                    # slot 7: the ssm_out projection result, i.e. llama's
                    # linear_attn_out, the last value before the residual add.
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(7 * H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.p_h_d.unsafe_ptr(), H, owning=False))
                comptime if MEGA_ALLOWED:
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
                if not cfg.dump4:
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset((2 * layer) * H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False))
                if cfg.dump4 and layer == cfg.dump_layer:
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr(), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False))
            # -- ffn sub-block --
            if cfg.pf4:
                ctx.synchronize()
                st.tq = perf_counter_ns()
            var ffn_norm_base = w
            comptime if not MEGA_ALLOWED:
                ffn_norm_base = moe_base + (7 if is_attn(layer) else 10)
            comptime if MEGA_ALLOWED:
                ctx.enqueue_function[rmsc_k](
                    Xm, tens_f32(ctx, b.wbuf, b.off[ffn_norm_base], H, h_layout), CurBm,
                    Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
                )
            else:
                ctx.enqueue_function[rmsc2_k](
                    Xm, tens_f32(ctx, b.wbuf, b.off[ffn_norm_base], H, h_layout), CurBm,
                    row_f32(ctx, b.p_h_d, 0, H, h2_layout),
                    Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
                )
            if cfg.dump4 and cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt and layer == cfg.dump_layer:
                var DbgNorm = row_f32(ctx, b.p_h_d, 0, H, moe_vec_layout)
                ctx.enqueue_function[amar_widen_bf16[type_of(moe_vec_layout), type_of(moe_vec_layout)]](row_bf16(ctx, b.curb_d, 0, H, moe_vec_layout), DbgNorm, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.p_h_d.unsafe_ptr(), H, owning=False))
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[0] += Int(nw - st.tq)
                st.tq = nw
            comptime if not MEGA_ALLOWED:
                moe_ffn(ctx, b, Xm, CurBm, moe_base, layer, moe_base + (8 if is_attn(layer) else 11), moe_base + (11 if is_attn(layer) else 14),
                        st.pos if cfg.expert_trace else -1)
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
                comptime if MEGA_ALLOWED:
                    gemm_w[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pg, m)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[1] += Int(nw - st.tq)
                st.tq = nw
            if use_dot:
                gemm_q8dot(ctx, AqH, AsH, Wfuq, Wfus, Pu, m, FFN, H)
            else:
                comptime if MEGA_ALLOWED:
                    gemm_w[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Pu, m)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[2] += Int(nw - st.tq)
                st.tq = nw
            comptime if MEGA_ALLOWED:
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
                comptime if MEGA_ALLOWED:
                    gemm_w[H, FFN](ctx, FgBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Ph2, m)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[4] += Int(nw - st.tq)
                st.tq = nw
            comptime if MEGA_ALLOWED:
                ctx.enqueue_function[r_add](Ph2, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
            if cfg.pf4:
                ctx.synchronize()
                var nw = perf_counter_ns()
                st.fc[5] += Int(nw - st.tq)
                st.tq = nw
            w += 4
            if is_attn(layer):
                moe_w += 16
            else:
                moe_w += 19
            if cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt:
                if not cfg.dump4:
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset((2 * layer + 1) * H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False))
                if cfg.dump4 and layer == cfg.dump_layer:
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(2 * H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False))
            if cfg.prof:
                ctx.synchronize()
                var now = perf_counter_ns()
                st.pf_ffn += Int(now - st.tp)
                st.tp = now

        var head_folded = use_mega and plain_head
        comptime if not MEGA_ALLOWED:
            head_folded = False
            w = moe_w
            if use_mega:
                w = 1 + N_SSM * MOE_W_SSM + N_ATT * MOE_W_ATT
        else:
            if use_mega or use_mega_win:
                w = 1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4
        if cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt and st.n_dumped < GEN_N:
            ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dump_h.unsafe_ptr().unsafe_offset(st.n_dumped * 2 * N_LAYERS * H), 2 * N_LAYERS * H, owning=False), src_buf=b.dbg_d)
            st.n_dumped += 1
        # -- head --
        if cfg.prof:
            ctx.synchronize()
            st.tp = perf_counter_ns()
        # f32 copy of the post-final-norm hidden state (pre-LM-head): row r is
        # h(st.pos + r), what the MTP draft head pairs with token st.pos + r + 1.
        var Hnm = TileTensor(b.hn_d, xm_layout)
        if not head_folded:
            ctx.enqueue_function[rms_m](
                Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), Hnm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
        if cfg.dump4 and cfg.dump and m == 1 and st.pos + 1 >= cfg.n_prompt:
            ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, b.dbg_d.unsafe_ptr().unsafe_offset(3 * H), H, owning=False), src_buf=DeviceBuffer[f32](ctx, b.hn_d.unsafe_ptr(), H, owning=False))
        if st.pos + m >= cfg.n_prompt:
            if not head_folded:
                ctx.enqueue_function[rmsc_k](
                    Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm,
                    Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
                )
                var Wheadq = tens_q8q(ctx, b.wbuf, b.off[w + 1], H * VOCAB, q_h_v)
                var Wheads = tens_q8s(ctx, b.wbuf, b.off[w + 1], H * VOCAB, s_h_v)
                gemm_w[VOCAB, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pv, m)
                ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
            if head_folded:
                pass
            elif win_spec and cfg.sample.temperature > 0:
                # A1: speculative SAMPLING. Truncate the target's rows into p,
                # draw one token per row from p (row m-1 is the bonus token if
                # every draft is accepted), and run the accept plus residual
                # rule against the drafts already sitting in toks_d.
                var Pt = TileTensor(b.pt_d, vm_layout)
                var Pd = TileTensor(b.pd_d, vm_layout)
                var SampTok = TileTensor(b.dtok_d, dtok_layout)
                var SampProb = TileTensor(b.hmax_d, dtok_layout)
                ctx.enqueue_function[sample_probs_k](
                    Logitsm, Pt, Int32(VOCAB), Float32(cfg.sample.temperature), Int32(cfg.sample.top_k),
                    Float32(cfg.sample.top_p), Float32(cfg.sample.min_p), grid_dim=m, block_dim=SAMP_THREADS,
                )
                ctx.enqueue_function[sample_row_k](
                    Logitsm, SampTok, SampProb, Int32(VOCAB),
                    Float32(cfg.sample.temperature), Int32(cfg.sample.top_k),
                    Float32(cfg.sample.top_p), Float32(cfg.sample.min_p),
                    cfg.sample.seed, UInt64(st.pos), grid_dim=m, block_dim=SAMP_THREADS,
                )
                if m == 1:
                    # No draft to verify: the window collapsed to one row (the
                    # generation limit, or k = 0). The target's own draw is the
                    # token, which is what the greedy path does here too.
                    ctx.enqueue_function[tokcp_k](SampTok, Toks, Int32(0), Int32(st.pos + 1), Int32(1), grid_dim=1, block_dim=32)
                else:
                  # The drafted ids are already in toks_d; they are copied into
                  # their own plane because a view over toks_d is immutable here
                  # and a kernel argument has to be mutable.
                  ctx.enqueue_copy(
                      dst_buf=b.dids_d,
                      src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(st.pos + 1), KMAX + 1, owning=False),
                  )
                  var DraftIds = TileTensor(b.dids_d, dtok_layout)
                  var Sout = TileTensor(b.sout_d, dtok_layout)
                  var Sacc = TileTensor(b.sacc_d, dtok_layout)
                  ctx.enqueue_function[spec_accept_k](
                      Pt, Pd, DraftIds, Sout, Sacc, Int32(VOCAB),
                      cfg.sample.seed, UInt64(st.pos), grid_dim=m - 1, block_dim=SAMP_THREADS,
                  )
                  ctx.enqueue_copy(dst_buf=b.sout_h, src_buf=b.sout_d)
                  ctx.enqueue_copy(dst_buf=b.sacc_h, src_buf=b.sacc_d)
                  ctx.synchronize()
                  var n_acc = 0
                  while n_acc < m - 1 and b.sacc_h[n_acc] == 1:
                      n_acc += 1
                  st.n_accepted += n_acc
                  if n_acc < m - 1:
                      # First rejection: the residual draw replaces that draft.
                      ctx.enqueue_function[tokcp_k](Sout, Toks, Int32(n_acc), Int32(st.pos + n_acc + 1), Int32(1), grid_dim=1, block_dim=32)
                  else:
                      # Every draft accepted: the bonus token comes from p.
                      ctx.enqueue_function[tokcp_k](SampTok, Toks, Int32(m - 1), Int32(st.pos + m), Int32(1), grid_dim=1, block_dim=32)
                  if cfg.spec_dbg:
                      var sline = String("specsample pos=") + String(st.pos) + " m=" + String(m) + " n_acc=" + String(n_acc) + " acc:"
                      for i in range(m - 1):
                          sline += " " + String(b.sacc_h[i]) + "/" + String(b.sout_h[i])
                      print(sline)
                  m = n_acc + 1
            elif win_spec:
                var Dtok = TileTensor(b.dtok_d, dtok_layout)
                ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(VOCAB), Int32(0), grid_dim=m, block_dim=256)
                var t_acc = 0
                if cfg.pf3:
                    ctx.synchronize()
                    t_acc = perf_counter_ns()
                ctx.enqueue_copy(dst_buf=b.dtok_h, src_buf=DeviceBuffer[DType.int32](ctx, b.dtok_d.unsafe_ptr(), KMAX + 1, owning=False))
                ctx.enqueue_copy(dst_buf=b.win_h, src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(st.pos + 1), KMAX + 1, owning=False))
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
            elif plain_head:
                ctx.enqueue_function[argmax_k](Logitsm, Toks, Int32(VOCAB), Int32(st.pos + 1), grid_dim=m, block_dim=256)
            else:
                # Sampling (M5). engine.mojo forces spec and the megakernel
                # off whenever temperature > 0, OR temperature <= 0 with
                # penalties/top_logprobs requested (items 3-4,
                # briefs/2026-09-16-sampling-all-models-lane.md) -- the
                # latter case still lands here rather than argmax_k because
                # amar_sample_row is argmax-equivalent at temperature <= 0
                # (P-K2), which is what lets a penalized row still resolve
                # to the penalized argmax. dtok_d and hmax_d are free scratch
                # here -- sample into them, 0-based, then reuse the same
                # tokcp_k the spec path already uses to place the result at
                # the real position.
                var SampTok = TileTensor(b.dtok_d, dtok_layout)
                var SampProb = TileTensor(b.hmax_d, dtok_layout)
                # Item 3-4, briefs/2026-09-16-sampling-all-models-lane.md:
                # penalties (applied to Logitsm in place, before the draw)
                # and top-N logprobs (of the same penalized/truncated row
                # amar_sample_row draws from). Scoped to m == 1 past the
                # prompt: every call here except MEGA_ALLOWED's batched
                # prompt-tail replay (m > 1, not real generation, where
                # penalties/logprobs are skipped rather than guessed).
                var want_pen = cfg.sample.presence_penalty != 0 or cfg.sample.frequency_penalty != 0
                var want_lp = cfg.sample.top_logprobs > 0
                if m == 1 and st.pos + 1 >= cfg.n_prompt and (want_pen or want_lp):
                    var hn = st.pos + 1 - cfg.n_prompt
                    if hn > 0:
                        ctx.enqueue_copy(
                            dst_buf=b.pen_hist_h.create_sub_buffer[DType.int32](0, hn),
                            src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(cfg.n_prompt), hn, owning=False),
                        )
                        ctx.synchronize()
                    # Item 4 verification staging only (coordinator review,
                    # 2026-09-16): the sync and file write for this stay in
                    # serve/engine.mojo (the harness, never embedded) --
                    # window.mojo is the self-optimising loop's candidate
                    # file, where host syncs and file writes are banned at
                    # scope even behind an env gate. This is one more
                    # enqueue_copy into a pre-allocated buffer, same class as
                    # the pen_hist_h copy just above, nothing blocking and
                    # nothing touching disk.
                    if cfg.dump_pen:
                        ctx.enqueue_copy(
                            dst_buf=b.dump_row_h,
                            src_buf=DeviceBuffer[f32](ctx, b.logits_d.unsafe_ptr(), VOCAB, owning=False),
                        )
                    var npen = 0
                    if want_pen:
                        var seen = Dict[Int, Int]()
                        for i in range(hn):
                            var t = Int(b.pen_hist_h[i])
                            seen[t] = seen.get(t, 0) + 1
                        for entry in seen.items():
                            if npen >= SAMP_CAP:
                                break
                            b.pen_ids_h[npen] = Int32(entry.key)
                            b.pen_cnt_h[npen] = Int32(entry.value)
                            npen += 1
                        b.pen_npen_h[0] = Int32(npen)
                        ctx.enqueue_copy(dst_buf=b.pen_npen_d.create_sub_buffer[DType.int32](0, 1), src_buf=b.pen_npen_h.create_sub_buffer[DType.int32](0, 1))
                        if npen > 0:
                            ctx.enqueue_copy(dst_buf=b.pen_ids_d.create_sub_buffer[DType.int32](0, npen), src_buf=b.pen_ids_h.create_sub_buffer[DType.int32](0, npen))
                            ctx.enqueue_copy(dst_buf=b.pen_cnt_d.create_sub_buffer[DType.int32](0, npen), src_buf=b.pen_cnt_h.create_sub_buffer[DType.int32](0, npen))
                            ctx.enqueue_function[apply_penalties_k](
                                Logitsm, TileTensor(b.pen_ids_d, pen_ids_layout), TileTensor(b.pen_cnt_d, pen_ids_layout), TileTensor(b.pen_npen_d, pen_npen_layout),
                                Int32(VOCAB), Float32(cfg.sample.presence_penalty), Float32(cfg.sample.frequency_penalty), grid_dim=1, block_dim=256,
                            )
                    if want_lp:
                        ctx.enqueue_function[topn_probs_k](
                            Logitsm, TileTensor(b.topn_ids_d, topn_layout), TileTensor(b.topn_probs_d, topn_layout),
                            Int32(VOCAB), Int32(min(cfg.sample.top_logprobs, NTOPLP)),
                            Float32(cfg.sample.temperature), Int32(cfg.sample.top_k), Float32(cfg.sample.top_p), Float32(cfg.sample.min_p),
                            grid_dim=1, block_dim=SAMP_THREADS,
                        )
                        ctx.enqueue_copy(dst_buf=b.topn_ids_h.create_sub_buffer[DType.int32](0, NTOPLP), src_buf=DeviceBuffer[DType.int32](ctx, b.topn_ids_d.unsafe_ptr(), NTOPLP, owning=False))
                        ctx.enqueue_copy(dst_buf=b.topn_probs_h.create_sub_buffer[f32](0, NTOPLP), src_buf=DeviceBuffer[f32](ctx, b.topn_probs_d.unsafe_ptr(), NTOPLP, owning=False))
                        ctx.synchronize()
                    did_sample_lp = want_lp
                # JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
                # m == 1, matching the penalties/logprobs scope above -- a
                # grammar governs one generated token at a time, never the
                # MEGA_ALLOWED batched prompt-tail replay. At temperature <= 0
                # the masked kernel is a masked argmax. engine.mojo forces
                # spec and the megakernel off and top_p=1/top_k=0/min_p=0 for
                # a grammar request: amar_sample_row_masked applies the mask
                # AFTER truncation, so a truncated draw could land outside
                # the allowed set and return -1.
                var grammar_here = m == 1 and st.pos + 1 >= cfg.n_prompt and st.grammar.__bool__() and not st.grammar_pending_think
                if grammar_here:
                    st.grammar_masked_draws += 1
                    ref mm = st.grammar.value()
                    mm.fill_mask(st.grammar_mask)
                    for i in range(len(st.grammar_mask.words)):
                        b.gmask_h[i] = st.grammar_mask.words[i]
                    ctx.enqueue_copy(dst_buf=b.gmask_d, src_buf=b.gmask_h)
                    ctx.enqueue_function[sample_row_masked_k](
                        Logitsm, SampTok, SampProb, Int32(VOCAB),
                        Float32(cfg.sample.temperature), Int32(cfg.sample.top_k),
                        Float32(cfg.sample.top_p), Float32(cfg.sample.min_p),
                        cfg.sample.seed, UInt64(st.pos),
                        b.gmask_d.unsafe_ptr(), Int32((VOCAB + 63) // 64),
                        grid_dim=m, block_dim=SAMP_THREADS,
                    )
                else:
                    ctx.enqueue_function[sample_row_k](
                        Logitsm, SampTok, SampProb, Int32(VOCAB),
                        Float32(cfg.sample.temperature), Int32(cfg.sample.top_k),
                        Float32(cfg.sample.top_p), Float32(cfg.sample.min_p),
                        cfg.sample.seed, UInt64(st.pos), grid_dim=m, block_dim=SAMP_THREADS,
                    )
                ctx.enqueue_function[tokcp_k](SampTok, Toks, Int32(0), Int32(st.pos + 1), Int32(m), grid_dim=1, block_dim=32)
            if cfg.serve:
                if win_spec and cfg.sample.temperature > 0:
                    # Sampled spec writes every emitted token into toks_d (the
                    # accepted drafts were written at draft time, the last one
                    # just now), so the plain read-back path is the right one.
                    ctx.enqueue_copy(dst_buf=b.stream_h.create_sub_buffer[DType.int32](0, m), src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(st.pos + 1), m, owning=False))
                    ctx.synchronize()
                    for i in range(m):
                        print(tok_line(cfg.req_id, Int(b.stream_h[i])))
                elif win_spec:
                    for i in range(m - 1):
                        print(tok_line(cfg.req_id, Int(b.win_h[i])))
                    print(tok_line(cfg.req_id, Int(b.dtok_h[m - 1])))
                else:
                    ctx.enqueue_copy(dst_buf=b.stream_h.create_sub_buffer[DType.int32](0, m), src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(st.pos + 1), m, owning=False))
                    if did_sample_lp:
                        ctx.enqueue_copy(
                            dst_buf=b.samp_prob_h.create_sub_buffer[f32](0, 1),
                            src_buf=DeviceBuffer[f32](ctx, b.hmax_d.unsafe_ptr(), 1, owning=False),
                        )
                    ctx.synchronize()
                    if m == 1 and st.pos + 1 >= cfg.n_prompt and st.grammar.__bool__():
                        # Item 1/2: advance the reasoning-boundary scan while
                        # waiting for "</think>", or the matcher once past
                        # it. Reasoning tokens are never accept()ed -- the
                        # matcher starts life already positioned at the
                        # boundary (grammar/test_reasoning_boundary.mojo).
                        var chosen_tok = Int(b.stream_h[0])
                        ref mm2 = st.grammar.value()
                        if chosen_tok < 0:
                            st.grammar_stop = True
                        elif st.grammar_pending_think:
                            if reasoning_boundary_observe(st.grammar_think_buf, mm2.vocab[].token_bytes[chosen_tok]):
                                st.grammar_pending_think = False
                        else:
                            st.grammar_accepted += 1
                            _ = mm2.accept(chosen_tok)
                            if mm2.is_terminated():
                                st.grammar_stop = True
                    if did_sample_lp:
                        var chosen = Int(b.stream_h[0])
                        var lp = log(Float64(b.samp_prob_h[0])) if b.samp_prob_h[0] > 0 else -1e30
                        var top_ids = List[Int]()
                        var top_lp = List[Float64]()
                        for i in range(min(Int(cfg.sample.top_logprobs), NTOPLP)):
                            var tid = Int(b.topn_ids_h[i])
                            if tid < 0:
                                break
                            top_ids.append(tid)
                            var p = b.topn_probs_h[i]
                            top_lp.append(log(Float64(p)) if p > 0 else -1e30)
                        print(tok_line_lp(cfg.req_id, chosen, lp, top_ids, top_lp))
                    else:
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
