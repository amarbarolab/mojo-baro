from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import exp, fma
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from matmul_skinny import ROW_WAVES, ROW_VEC
from moe import (
    N_EXP, TOPK, MOE_WAVES, MOE_THREADS, Q4K, Q4K_BYTES, Q6K, Q6K_BYTES,
    q8d_row_dot, q8_0_row_dot, q4k_dot_blocks, q6k_row_dot,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8


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
