from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import exp, fma
from std.memory import bitcast
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from matmul_skinny import ROW_WAVES, ROW_VEC
from moe import (
    N_EXP, TOPK, MOE_WAVES, MOE_THREADS, Q4K, Q4K_BYTES, Q6K, Q6K_BYTES,
    q8d_row_dot, q8_0_row_dot, q4k_dot_blocks, q6k_row_dot, q6k_value,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime i8 = DType.int8
comptime f16 = DType.float16


def moe_matmul_q8d_rows[
    OLayout: TensorLayout, ALayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 2
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var t = Int(block_idx.y)
    var k = Int(k_dim)
    var dot = q8d_row_dot(A, W, t, row * k, Int(s_off) + row * (k // 32) * 2, k)
    if lane_id() == 0:
        O[t, row] = rebind[O.ElementType](dot)


def moe_matmul_q8d_rows_add[
    ALayout: TensorLayout, XLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
):
    comptime assert A.flat_rank == 2 and X.flat_rank == 2
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var t = Int(block_idx.y)
    var k = Int(k_dim)
    var dot = q8d_row_dot(A, W, t, row * k, Int(s_off) + row * (k // 32) * 2, k)
    if lane_id() == 0:
        X[t, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[t, row]) + dot)


def moe_router_top8_sig_rows[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    W: TileTensor[f32, WLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert L.flat_rank == 2 and IDX.flat_rank == 2 and W.flat_rank == 2
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 1
    comptime PER = N_EXP // WARP_SIZE
    var t = Int(block_idx.x)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var v = InlineArray[Scalar[f32], PER](uninitialized=True)
    var lm = Scalar[f32](-3.4028234663852886e38)
    comptime for m in range(PER):
        v[m] = rebind[Scalar[f32]](L[t, lane + m * WARP_SIZE])
        if v[m] > lm:
            lm = v[m]
    var mx = warp.max(lm)
    var ls = Scalar[f32](0)
    comptime for m in range(PER):
        v[m] = exp(v[m] - mx)
        ls += v[m]
    var tot = warp.sum(ls)
    comptime for m in range(PER):
        v[m] = v[m] / tot
    var wsum = Scalar[f32](0)
    for j in range(TOPK):
        var best = Scalar[f32](-1)
        var bi = N_EXP
        comptime for m in range(PER):
            if v[m] > best:
                best = v[m]
                bi = lane + m * WARP_SIZE
        var gbest = warp.max(best)
        var cand = bi if best == gbest else N_EXP
        var gi = -warp.max(-cand)
        if gi == bi and best == gbest:
            comptime for m in range(PER):
                if lane + m * WARP_SIZE == gi:
                    v[m] = Scalar[f32](-1)
        if lane == 0:
            IDX[t, j] = rebind[IDX.ElementType](Int32(gi))
            W[t, j] = rebind[W.ElementType](gbest)
        wsum += gbest
    if lane == 0:
        for j in range(TOPK):
            W[t, j] = rebind[W.ElementType](rebind[Scalar[f32]](W[t, j]) / wsum)
    var Xv = X.vectorize[1, 8]()
    var Gv = G.vectorize[8]()
    var acc = SIMD[f32, 8](0)
    var i = lane
    while i < K // 8:
        acc = fma(rebind[SIMD[f32, 8]](Xv[t, i]), rebind[SIMD[f32, 8]](Gv[i]), acc)
        i += WARP_SIZE
    var s = warp.sum(acc.reduce_add())
    if lane == 0:
        O[t] = rebind[O.ElementType](Scalar[f32](1) / (Scalar[f32](1) + exp(-s)))


def moe_gate_up_q4k_rows[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout,
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[bf16, HLayout, MutAnyOrigin],
    k_dim: Int32,
    up_offset: Int32,
):
    comptime assert Xb.flat_rank == 2 and IDX.flat_rank == 2 and HO.flat_rank == 2
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var t = Int(block_idx.y)
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[t, j]))
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var g = q4k_dot_blocks(Xb, W, t, row_base, Int(k_dim))
    var u = q4k_dot_blocks(Xb, W, t, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[t * NSEL + j, r] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[bf16]())


def moe_down_q4k_rows[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 2 and WT.flat_rank == 2 and O.flat_rank == 2
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var t = Int(block_idx.y)
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q4K) * Q4K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[t, j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var dot = q4k_dot_blocks(Hb, WD, t * NSEL + j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[t, j]) * dot
    if lane_id() == 0:
        O[t, c] = rebind[O.ElementType](out)


def moe_down_q6k_rows[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 2 and WT.flat_rank == 2 and O.flat_rank == 2
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var t = Int(block_idx.y)
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q6K) * Q6K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[t, j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var dot = q6k_row_dot(Hb, WD, t * NSEL + j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[t, j]) * dot
    if lane_id() == 0:
        O[t, c] = rebind[O.ElementType](out)


def moe_shared_gate_up_q8_0_rows[
    FFN: Int,
    XLayout: TensorLayout, HLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    HO: TileTensor[bf16, HLayout, MutAnyOrigin],
    k_dim: Int32,
    row_bytes: Int32,
    up_offset: Int32,
):
    comptime assert X.flat_rank == 2 and HO.flat_rank == 2
    var r = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if r >= FFN:
        return
    var t = Int(block_idx.y)
    var row_base = r * Int(row_bytes)
    var g = q8_0_row_dot(X, W, t, row_base, Int(k_dim))
    var u = q8_0_row_dot(X, W, t, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[t, r] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[bf16]())


def moe_shared_down_q8_0_res_rows[
    FFN: Int,
    HLayout: TensorLayout, SLayout: TensorLayout,
    ALayout: TensorLayout, XLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    SG: TileTensor[f32, SLayout, MutAnyOrigin],
    A: TileTensor[f32, ALayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
    row_bytes: Int32,
):
    comptime assert Hb.flat_rank == 2 and SG.flat_rank == 1 and A.flat_rank == 2 and X.flat_rank == 2
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if c >= Int(n):
        return
    var t = Int(block_idx.y)
    var out = Scalar[f32](0)
    out += rebind[Scalar[f32]](SG[t]) * q8_0_row_dot(Hb, WD, t, c * Int(row_bytes), FFN)
    if lane_id() == 0:
        X[t, c] = rebind[X.ElementType](
            rebind[Scalar[f32]](X[t, c]) + rebind[Scalar[f32]](A[t, c]) + out
        )


def moe_skinny_f32_rows[
    UNROLL: Int,
    ALayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout
](
    A: TileTensor[f32, ALayout, MutAnyOrigin],
    W: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and W.flat_rank == 2 and O.flat_rank == 2
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return
    var t = Int(block_idx.y)
    var Wv = W.vectorize[1, ROW_VEC]()
    var Av = A.vectorize[1, ROW_VEC]()
    comptime STEP = WARP_SIZE * ROW_VEC
    var acc = SIMD[f32, ROW_VEC](0)
    var kk = 0
    while kk + UNROLL * STEP <= K:
        var ws = InlineArray[SIMD[f32, ROW_VEC], UNROLL](uninitialized=True)

        comptime for u in range(UNROLL):
            ws[u] = rebind[SIMD[f32, ROW_VEC]](Wv[row, (kk + u * STEP) // ROW_VEC + lane])

        comptime for u in range(UNROLL):
            var a = rebind[SIMD[f32, ROW_VEC]](Av[t, (kk + u * STEP) // ROW_VEC + lane]).cast[f32]()
            acc += ws[u].cast[f32]() * a
        kk += UNROLL * STEP
    while kk < K:
        var w = rebind[SIMD[f32, ROW_VEC]](Wv[row, kk // ROW_VEC + lane]).cast[f32]()
        var a = rebind[SIMD[f32, ROW_VEC]](Av[t, kk // ROW_VEC + lane]).cast[f32]()
        acc += w * a
        kk += STEP

    var total = warp.sum(acc.reduce_add())
    if lane == 0:
        O[t, row] = rebind[O.ElementType](total)


def moe_matmul_q8d_rows_shared[
    MR: Int, ADD: Bool,
    OLayout: TensorLayout, ALayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
    m: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 2
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var t0 = Int(block_idx.y) * MR
    var M = Int(m)
    var k = Int(k_dim)
    var q_base = row * k
    var s_base = Int(s_off) + row * (k // 32) * 2
    var lane = Int(lane_id())
    var Av = A.vectorize[1, 16]()
    var nb = k // 32
    var acc = InlineArray[SIMD[f32, 16], MR](fill=SIMD[f32, 16](0))
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var raw = W.unsafe_offset(s_base + b * 2).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(q_base + b * 32 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        comptime for u in range(MR):
            var a = rebind[SIMD[bf16, 16]](Av[min(t0 + u, M - 1), 2 * b + h]).cast[f32]()
            acc[u] = fma(v, a, acc[u])
        b += 16
    comptime for u in range(MR):
        var dot = warp.sum(acc[u].reduce_add())
        if lane == 0 and t0 + u < M:
            comptime if ADD:
                O[t0 + u, row] = rebind[O.ElementType](rebind[Scalar[f32]](O[t0 + u, row]) + dot)
            else:
                O[t0 + u, row] = rebind[O.ElementType](dot)


def moe_pairs_group[
    MR: Int, ILayout: TensorLayout, GLayout: TensorLayout, PLayout: TensorLayout, SLayout: TensorLayout
](
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    GE: TileTensor[i32, GLayout, MutAnyOrigin],
    GP: TileTensor[i32, PLayout, MutAnyOrigin],
    S: TileTensor[i32, SLayout, MutAnyOrigin],
    m: Int32,
    ng: Int32,
):
    comptime assert IDX.flat_rank == 2 and GE.flat_rank == 1 and GP.flat_rank == 1 and S.flat_rank == 1
    if lane_id() != 0 or global_idx.x != 0:
        return
    var M = Int(m)
    var NG = Int(ng)
    for e in range(N_EXP):
        S[e] = rebind[S.ElementType](Int32(0))
    for t in range(M):
        for j in range(TOPK):
            var e = Int(rebind[Scalar[i32]](IDX[t, j]))
            S[e] = rebind[S.ElementType](rebind[Scalar[i32]](S[e]) + 1)
    var g = 0
    for e in range(N_EXP):
        var c = Int(rebind[Scalar[i32]](S[e]))
        var n = (c + MR - 1) // MR
        S[N_EXP + e] = rebind[S.ElementType](Int32(g * MR))
        for q in range(n):
            GE[g + q] = rebind[GE.ElementType](Int32(e))
        g += n
    for q in range(g, NG):
        GE[q] = rebind[GE.ElementType](Int32(-1))
    for q in range(NG * MR):
        GP[q] = rebind[GP.ElementType](Int32(-1))
    for t in range(M):
        for j in range(TOPK):
            var e = Int(rebind[Scalar[i32]](IDX[t, j]))
            var slot = Int(rebind[Scalar[i32]](S[N_EXP + e]))
            GP[slot] = rebind[GP.ElementType](Int32(t * TOPK + j))
            S[N_EXP + e] = rebind[S.ElementType](Int32(slot + 1))


@always_inline
def q4k_dot_blocks_shared[
    MR: Int, XLayout: TensorLayout, PLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    GP: TileTensor[i32, PLayout, MutAnyOrigin],
    g: Int,
    row_div: Int,
    row_base: Int,
    k_dim: Int,
) -> InlineArray[Scalar[f32], MR]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // Q4K
    var rows = InlineArray[Int, MR](fill=0)
    var p0 = Int(rebind[Scalar[i32]](GP[g * MR]))
    comptime for u in range(MR):
        var p = Int(rebind[Scalar[i32]](GP[g * MR + u]))
        rows[u] = (p if p >= 0 else p0) // row_div
    var acc = InlineArray[SIMD[f32, 16], MR](fill=SIMD[f32, 16](0))
    var b = lane // 8
    var s = lane % 8
    var pair = s // 2
    var half = (s % 2) * 16
    while b < nb:
        var base = row_base + b * Q4K_BYTES
        var hdr = W.unsafe_offset(base).load[width=16]()
        var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
        var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
        var g0 = 2 * pair
        var g1 = g0 + 1
        var sc0: Int
        var mn0: Int
        var sc1: Int
        var mn1: Int
        if g0 < 4:
            sc0 = Int(hdr[4 + g0]) & 0x3F
            mn0 = Int(hdr[8 + g0]) & 0x3F
            sc1 = Int(hdr[4 + g1]) & 0x3F
            mn1 = Int(hdr[8 + g1]) & 0x3F
        else:
            var md0 = Int(hdr[12 + g0 - 4])
            var md1 = Int(hdr[12 + g1 - 4])
            sc0 = (md0 & 0x0F) | ((Int(hdr[4 + g0 % 4]) >> 2) & 0x30)
            mn0 = (md0 >> 4) | ((Int(hdr[8 + g0 % 4]) >> 2) & 0x30)
            sc1 = (md1 & 0x0F) | ((Int(hdr[4 + g1 % 4]) >> 2) & 0x30)
            mn1 = (md1 >> 4) | ((Int(hdr[8 + g1 % 4]) >> 2) & 0x30)
        var qb = W.unsafe_offset(base + 16 + pair * 32 + half).load[width=16]()
        var lo = (qb & 0x0F).cast[f32]()
        var hi = (qb >> 4).cast[f32]()
        var v0 = (SIMD[f32, 16](d * Scalar[f32](sc0)) * lo - SIMD[f32, 16](dm * Scalar[f32](mn0))).cast[bf16]().cast[f32]()
        var v1 = (SIMD[f32, 16](d * Scalar[f32](sc1)) * hi - SIMD[f32, 16](dm * Scalar[f32](mn1))).cast[bf16]().cast[f32]()
        var k0 = b * Q4K + g0 * 32 + half
        comptime for u in range(MR):
            var a0 = rebind[SIMD[bf16, 16]](Xv[rows[u], k0 // 16]).cast[f32]()
            var a1 = rebind[SIMD[bf16, 16]](Xv[rows[u], (k0 + 32) // 16]).cast[f32]()
            acc[u] = fma(v0, a0, fma(v1, a1, acc[u]))
        b += 4
    var out = InlineArray[Scalar[f32], MR](fill=0)
    comptime for u in range(MR):
        out[u] = warp.sum(acc[u].reduce_add())
    return out^


def moe_gate_up_q4k_grouped[
    MR: Int, FFN: Int,
    XLayout: TensorLayout, GLayout: TensorLayout, PLayout: TensorLayout, HLayout: TensorLayout,
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    GE: TileTensor[i32, GLayout, MutAnyOrigin],
    GP: TileTensor[i32, PLayout, MutAnyOrigin],
    HO: TileTensor[bf16, HLayout, MutAnyOrigin],
    k_dim: Int32,
    up_offset: Int32,
):
    comptime assert Xb.flat_rank == 2 and GE.flat_rank == 1 and GP.flat_rank == 1 and HO.flat_rank == 2
    var r = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if r >= FFN:
        return
    var g = Int(block_idx.y)
    var e = Int(rebind[Scalar[i32]](GE[g]))
    if e < 0:
        return
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var gd = q4k_dot_blocks_shared[MR](Xb, W, GP, g, TOPK, row_base, Int(k_dim))
    var ud = q4k_dot_blocks_shared[MR](Xb, W, GP, g, TOPK, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        comptime for u in range(MR):
            var p = Int(rebind[Scalar[i32]](GP[g * MR + u]))
            if p >= 0:
                HO[p, r] = rebind[HO.ElementType]((gd[u] / (Scalar[f32](1) + exp(-gd[u])) * ud[u]).cast[bf16]())


def moe_down_q4k_grouped[
    MR: Int, FFN: Int,
    HLayout: TensorLayout, GLayout: TensorLayout, PLayout: TensorLayout, DLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    GE: TileTensor[i32, GLayout, MutAnyOrigin],
    GP: TileTensor[i32, PLayout, MutAnyOrigin],
    D: TileTensor[f32, DLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and GE.flat_rank == 1 and GP.flat_rank == 1 and D.flat_rank == 2
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var g = Int(block_idx.y)
    var e = Int(rebind[Scalar[i32]](GE[g]))
    if e < 0:
        return
    var row_bytes = (FFN // Q4K) * Q4K_BYTES
    var row_base = e * N * row_bytes + c * row_bytes
    var dd = q4k_dot_blocks_shared[MR](Hb, WD, GP, g, 1, row_base, FFN)
    if lane_id() == 0:
        comptime for u in range(MR):
            var p = Int(rebind[Scalar[i32]](GP[g * MR + u]))
            if p >= 0:
                D[p, c] = rebind[D.ElementType](dd[u])


def moe_down_combine[
    NSEL: Int, DLayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout,
](
    D: TileTensor[f32, DLayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert D.flat_rank == 2 and WT.flat_rank == 2 and O.flat_rank == 2
    var c = Int(global_idx.x)
    if c >= Int(n):
        return
    var t = Int(block_idx.y)
    var out = Scalar[f32](0)
    for j in range(NSEL):
        var dot = rebind[Scalar[f32]](D[t * NSEL + j, c])
        out += rebind[Scalar[f32]](WT[t, j]) * dot
    O[t, c] = rebind[O.ElementType](out)


def moe_down_q6k_grouped[
    MR: Int, FFN: Int,
    HLayout: TensorLayout, GLayout: TensorLayout, PLayout: TensorLayout, DLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    GE: TileTensor[i32, GLayout, MutAnyOrigin],
    GP: TileTensor[i32, PLayout, MutAnyOrigin],
    D: TileTensor[f32, DLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and GE.flat_rank == 1 and GP.flat_rank == 1 and D.flat_rank == 2
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var g = Int(block_idx.y)
    var e = Int(rebind[Scalar[i32]](GE[g]))
    if e < 0:
        return
    var row_bytes = (FFN // Q6K) * Q6K_BYTES
    var row_base = e * N * row_bytes + c * row_bytes
    var rows = InlineArray[Int, MR](fill=0)
    var p0 = Int(rebind[Scalar[i32]](GP[g * MR]))
    comptime for u in range(MR):
        var p = Int(rebind[Scalar[i32]](GP[g * MR + u]))
        rows[u] = p if p >= 0 else p0
    var acc = InlineArray[Scalar[f32], MR](fill=0)
    var k = Int(lane_id())
    while k < FFN:
        var v = q6k_value(WD, row_base, k)
        comptime for u in range(MR):
            acc[u] += rebind[Scalar[bf16]](Hb[rows[u], k]).cast[f32]() * v
        k += WARP_SIZE
    comptime for u in range(MR):
        var dot = warp.sum(acc[u])
        var p = Int(rebind[Scalar[i32]](GP[g * MR + u]))
        if lane_id() == 0 and p >= 0:
            D[p, c] = rebind[D.ElementType](dot)
