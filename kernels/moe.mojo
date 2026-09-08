from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import exp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32

comptime N_EXP = 256
comptime TOPK = 8
comptime E_FFN = 512
comptime SH_FFN = 512
comptime MOE_H = 2048

comptime MOE_WAVES = 8
comptime MOE_THREADS = MOE_WAVES * WARP_SIZE
comptime MOE_VEC = 8


def amar_moe_router_top8[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    W: TileTensor[f32, WLayout, MutAnyOrigin],
):
    comptime assert L.flat_rank == 1 and IDX.flat_rank == 1 and W.flat_rank == 1

    var p = stack_allocation[f32, address_space=AddressSpace.SHARED](
        row_major[N_EXP]()
    )
    var red = stack_allocation[f32, address_space=AddressSpace.SHARED](
        row_major[N_EXP]()
    )

    var tid = Int(thread_idx.x)
    var v = rebind[Scalar[f32]](L[tid])
    red[tid] = rebind[red.ElementType](v)
    barrier()

    var s = N_EXP // 2
    while s > 0:
        if tid < s:
            var a = rebind[Scalar[f32]](red[tid])
            var b = rebind[Scalar[f32]](red[tid + s])
            red[tid] = rebind[red.ElementType](a if a > b else b)
        barrier()
        s //= 2
    var mx = rebind[Scalar[f32]](red[0])
    barrier()

    var ev = exp(v - mx)
    p[tid] = rebind[p.ElementType](ev)
    red[tid] = rebind[red.ElementType](ev)
    barrier()

    s = N_EXP // 2
    while s > 0:
        if tid < s:
            red[tid] = rebind[red.ElementType](
                rebind[Scalar[f32]](red[tid]) + rebind[Scalar[f32]](red[tid + s])
            )
        barrier()
        s //= 2
    var tot = rebind[Scalar[f32]](red[0])
    barrier()

    p[tid] = rebind[p.ElementType](rebind[Scalar[f32]](p[tid]) / tot)
    barrier()

    if tid == 0:
        var wsum = Scalar[f32](0)
        for j in range(TOPK):
            var best = Scalar[f32](-1)
            var bi = 0
            for i in range(N_EXP):
                var pv = rebind[Scalar[f32]](p[i])
                if pv > best:
                    best = pv
                    bi = i
            IDX[j] = rebind[IDX.ElementType](Int32(bi))
            W[j] = rebind[W.ElementType](best)
            wsum += best
            p[bi] = rebind[p.ElementType](Scalar[f32](-1))
        for j in range(TOPK):
            W[j] = rebind[W.ElementType](rebind[Scalar[f32]](W[j]) / wsum)


def amar_moe_sig_gate[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert X.flat_rank == 1 and G.flat_rank == 1 and O.flat_rank == 1

    var lane = Int(lane_id())
    var K = Int(k_dim)
    var acc = Scalar[f32](0)
    var i = lane
    while i < K:
        acc += rebind[Scalar[f32]](X[i]) * rebind[Scalar[f32]](G[i])
        i += WARP_SIZE
    var t = warp.sum(acc)
    if lane == 0:
        O[0] = rebind[O.ElementType](Scalar[f32](1) / (Scalar[f32](1) + exp(-t)))


def amar_moe_gate_up[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, GLayout: TensorLayout, ULayout: TensorLayout,
    ILayout: TensorLayout, HLayout: TensorLayout
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    WG: TileTensor[bf16, GLayout, MutAnyOrigin],
    WU: TileTensor[bf16, ULayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[f32, HLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert Xb.flat_rank == 2 and WG.flat_rank == 2 and WU.flat_rank == 2
    comptime assert IDX.flat_rank == 1 and HO.flat_rank == 1

    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return

    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row = e * FFN + r
    var lane = Int(lane_id())
    var K = Int(k_dim)

    var Xv = Xb.vectorize[1, MOE_VEC]()
    var Gv = WG.vectorize[1, MOE_VEC]()
    var Uv = WU.vectorize[1, MOE_VEC]()
    comptime STEP = WARP_SIZE * MOE_VEC

    var ag = SIMD[f32, MOE_VEC](0)
    var au = SIMD[f32, MOE_VEC](0)
    var kk = 0
    while kk < K:
        var a = rebind[SIMD[bf16, MOE_VEC]](
            Xv[0, kk // MOE_VEC + lane]
        ).cast[f32]()
        ag += rebind[SIMD[bf16, MOE_VEC]](
            Gv[row, kk // MOE_VEC + lane]
        ).cast[f32]() * a
        au += rebind[SIMD[bf16, MOE_VEC]](
            Uv[row, kk // MOE_VEC + lane]
        ).cast[f32]() * a
        kk += STEP

    var g = warp.sum(ag.reduce_add())
    var u = warp.sum(au.reduce_add())
    if lane == 0:
        HO[wid] = rebind[HO.ElementType](
            g / (Scalar[f32](1) + exp(-g)) * u
        )


def amar_moe_down[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, DLayout: TensorLayout, ILayout: TensorLayout,
    WLayout: TensorLayout, OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: TileTensor[bf16, DLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and WD.flat_rank == 2
    comptime assert IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1

    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return

    var lane = Int(lane_id())
    var Hv = Hb.vectorize[1, MOE_VEC]()
    var Dv = WD.vectorize[1, MOE_VEC]()
    comptime STEP = WARP_SIZE * MOE_VEC

    var out = Scalar[f32](0)
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row = e * N + c
        var acc = SIMD[f32, MOE_VEC](0)
        var kk = 0
        while kk < FFN:
            var h = rebind[SIMD[bf16, MOE_VEC]](
                Hv[j, kk // MOE_VEC + lane]
            ).cast[f32]()
            acc += rebind[SIMD[bf16, MOE_VEC]](
                Dv[row, kk // MOE_VEC + lane]
            ).cast[f32]() * h
            kk += STEP
        out += rebind[Scalar[f32]](WT[j]) * warp.sum(acc.reduce_add())

    if lane == 0:
        O[c] = rebind[O.ElementType](out)
