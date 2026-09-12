from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.memory import bitcast
from std.math import exp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime i8 = DType.int8

comptime N_EXP = 256
comptime TOPK = 8
comptime E_FFN = 512
comptime SH_FFN = 512
comptime MOE_H = 2048

comptime MOE_WAVES = 8
comptime MOE_THREADS = MOE_WAVES * WARP_SIZE
comptime MOE_VEC = 8

comptime Q4K = 256
comptime Q4K_BYTES = 144
comptime Q6K = 256
comptime Q6K_BYTES = 210


@always_inline
def q4k_scale_min(
    w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, j: Int
) -> Tuple[Int, Int]:
    var d = Int(w[unsafe_offset=base + 4 + j % 4])
    var m = Int(w[unsafe_offset=base + 8 + j % 4])
    if j < 4:
        return (d & 0x3F, m & 0x3F)
    var md = Int(w[unsafe_offset=base + 12 + j - 4])
    return ((md & 0x0F) | ((d >> 2) & 0x30), (md >> 4) | ((m >> 2) & 0x30))


@always_inline
def q4k_value(
    w: MutPointer[Scalar[u8], MutAnyOrigin], row_base: Int, k: Int
) -> Scalar[f32]:
    var block = k // Q4K
    var within = k % Q4K
    var group = within // 32
    var in_group = within % 32
    var base = row_base + block * Q4K_BYTES
    var dbits = Int(w[unsafe_offset=base]) | (Int(w[unsafe_offset=base + 1]) << 8)
    var mbits = Int(w[unsafe_offset=base + 2]) | (Int(w[unsafe_offset=base + 3]) << 8)
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(dbits)))[0].cast[f32]()
    var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(mbits)))[0].cast[f32]()
    var sc, mn = q4k_scale_min(w, base, group)
    var qbyte = Int(w[unsafe_offset=base + 16 + (group // 2) * 32 + in_group])
    var q = qbyte & 0x0F if group % 2 == 0 else qbyte >> 4
    return (d * Scalar[f32](sc * q) - dm * Scalar[f32](mn)).cast[bf16]().cast[f32]()


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


@always_inline
def q4k_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var acc = Scalar[f32](0)
    var k = lane
    while k < k_dim:
        acc += rebind[Scalar[bf16]](X[x_row, k]).cast[f32]() * q4k_value(W, row_base, k)
        k += WARP_SIZE
    return warp.sum(acc)


def q4k_decode_vector[
    OLayout: TensorLayout,
](
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
):
    comptime assert O.flat_rank == 1
    var k = global_idx.x
    if k < Q4K:
        O[k] = rebind[O.ElementType](q4k_value(W, 0, k))


@always_inline
def q6k_value(
    w: MutPointer[Scalar[u8], MutAnyOrigin], row_base: Int, k: Int
) -> Scalar[f32]:
    var block = k // Q6K
    var within = k % Q6K
    var half = within // 128
    var group = (within % 128) // 32
    var lane = within % 32
    var base = row_base + block * Q6K_BYTES
    var ql_index = lane if group == 0 or group == 2 else lane + 32
    var ql = Int(w[unsafe_offset=base + half * 64 + ql_index])
    var qh = Int(w[unsafe_offset=base + 128 + half * 32 + lane])
    var shift = 0 if group == 0 else 2 if group == 1 else 4 if group == 2 else 6
    var q = ((ql >> (0 if group < 2 else 4)) & 0x0F) | (((qh >> shift) & 0x03) << 4)
    var scale_index = half * 8 + (lane // 16) + group * 2
    var sc = bitcast[i8, 1](SIMD[u8, 1](w[unsafe_offset=base + 192 + scale_index]))[0]
    var dbits = Int(w[unsafe_offset=base + 208]) | (Int(w[unsafe_offset=base + 209]) << 8)
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(dbits)))[0].cast[f32]()
    return d * Scalar[f32](Int(sc)) * Scalar[f32](q - 32)


@always_inline
def q6k_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var acc = Scalar[f32](0)
    var k = Int(lane_id())
    while k < k_dim:
        acc += rebind[Scalar[bf16]](X[x_row, k]).cast[f32]() * q6k_value(W, row_base, k)
        k += WARP_SIZE
    return warp.sum(acc)


@always_inline
def q8_0_value(w: MutPointer[Scalar[u8], MutAnyOrigin], row_base: Int, k: Int) -> Scalar[f32]:
    var block = k // 32
    var in_block = k % 32
    var p = w + row_base + block * 34
    var raw = p.unsafe_bitcast[Scalar[u16]]()[]
    var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
    var q = p.unsafe_offset(2 + in_block).unsafe_bitcast[Scalar[i8]]()[]
    return (d * Scalar[f32](q)).cast[bf16]().cast[f32]()


@always_inline
def q8_0_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var acc = Scalar[f32](0)
    var k = lane
    while k < k_dim:
        acc += rebind[Scalar[bf16]](X[x_row, k]).cast[f32]() * q8_0_value(W, row_base, k)
        k += WARP_SIZE
    return warp.sum(acc)


def moe_embed_q8_0_pos[
    OLayout: TensorLayout, KLayout: TensorLayout,
](
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Toks: TileTensor[i32, KLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
    row_bytes: Int32,
):
    comptime assert O.flat_rank == 2 and Toks.flat_rank == 1
    var idx = global_idx.x
    if idx >= Int(n):
        return
    var token = Int(rebind[Scalar[i32]](Toks[Int(pos) + Int(block_idx.y)]))
    var row_base = token * Int(row_bytes)
    O[block_idx.y, idx] = rebind[O.ElementType](q8_0_value(W, row_base, idx))


def moe_matmul_q8_0_m1[
    OLayout: TensorLayout, ALayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    row_bytes: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 1
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var dot = q8_0_row_dot(A, W, 0, row * Int(row_bytes), Int(k_dim))
    if lane_id() == 0:
        O[row] = rebind[O.ElementType](dot)


def moe_add3[
    ALayout: TensorLayout, BLayout: TensorLayout, XLayout: TensorLayout,
](
    A: TileTensor[f32, ALayout, MutAnyOrigin],
    B: TileTensor[f32, BLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert A.flat_rank == 1 and B.flat_rank == 1 and X.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        X[i] = rebind[X.ElementType](
            rebind[Scalar[f32]](X[i])
            + rebind[Scalar[f32]](A[i])
            + rebind[Scalar[f32]](B[i])
        )


def moe_sig_gate_q8_0[
    XLayout: TensorLayout, OLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert X.flat_rank == 2 and O.flat_rank == 1
    var dot = q8_0_row_dot(X, W, 0, 0, Int(k_dim))
    if lane_id() == 0:
        O[0] = rebind[O.ElementType](Scalar[f32](1) / (Scalar[f32](1) + exp(-dot)))


def moe_gate_up_q8_0[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[f32, HLayout, MutAnyOrigin],
    k_dim: Int32,
    row_bytes: Int32,
    up_offset: Int32,
):
    comptime assert X.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_base = (e * FFN + r) * Int(row_bytes)
    var g = q8_0_row_dot(X, W, 0, row_base, Int(k_dim))
    var u = q8_0_row_dot(X, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType](g / (Scalar[f32](1) + exp(-g)) * u)


def moe_down_q8_0[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    row_bytes: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if c >= Int(n):
        return
    var out = Scalar[f32](0)
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = (e * Int(n) + c) * Int(row_bytes)
        out += rebind[Scalar[f32]](WT[j]) * q8_0_row_dot(Hb, WD, j, row_base, FFN)
    if lane_id() == 0:
        O[c] = rebind[O.ElementType](out)


def amar_moe_gate_up_q4k[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    WG: MutPointer[Scalar[u8], MutAnyOrigin],
    WU: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[f32, HLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert Xb.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var g = q4k_row_dot(Xb, WG, 0, row_base, Int(k_dim))
    var u = q4k_row_dot(Xb, WU, 0, row_base, Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType](g / (Scalar[f32](1) + exp(-g)) * u)


def moe_gate_up_q4k_pack[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[f32, HLayout, MutAnyOrigin],
    k_dim: Int32,
    up_offset: Int32,
):
    comptime assert Xb.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var g = q4k_row_dot(Xb, W, 0, row_base, Int(k_dim))
    var u = q4k_row_dot(Xb, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType](g / (Scalar[f32](1) + exp(-g)) * u)


def amar_moe_down_q4k[
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
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var lane = Int(lane_id())
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q4K) * Q4K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var dot = q4k_row_dot(Hb, WD, j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[j]) * dot
    if lane == 0:
        O[c] = rebind[O.ElementType](out)


def amar_moe_down_q6k[
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
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q6K) * Q6K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var dot = q6k_row_dot(Hb, WD, j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[j]) * dot
    if lane_id() == 0:
        O[c] = rebind[O.ElementType](out)
