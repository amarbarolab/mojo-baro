from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.memory import bitcast
from std.math import exp, fma
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
    comptime PER = N_EXP // WARP_SIZE
    var lane = Int(lane_id())
    var v = InlineArray[Scalar[f32], PER](uninitialized=True)
    var lm = Scalar[f32](-3.4028234663852886e38)
    comptime for m in range(PER):
        v[m] = rebind[Scalar[f32]](L[lane + m * WARP_SIZE])
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
            IDX[j] = rebind[IDX.ElementType](Int32(gi))
            W[j] = rebind[W.ElementType](gbest)
        wsum += gbest
    if lane == 0:
        for j in range(TOPK):
            W[j] = rebind[W.ElementType](rebind[Scalar[f32]](W[j]) / wsum)


@always_inline
def router_top8_sig_body[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    mut IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    mut W: TileTensor[f32, WLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    mut O: TileTensor[f32, OLayout, MutAnyOrigin],
    K: Int,
):
    comptime assert L.flat_rank == 1 and IDX.flat_rank == 1 and W.flat_rank == 1
    comptime assert X.flat_rank == 1 and G.flat_rank == 1 and O.flat_rank == 1
    comptime PER = N_EXP // WARP_SIZE
    var lane = Int(lane_id())
    var v = InlineArray[Scalar[f32], PER](uninitialized=True)
    var lm = Scalar[f32](-3.4028234663852886e38)
    comptime for m in range(PER):
        v[m] = rebind[Scalar[f32]](L[lane + m * WARP_SIZE])
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
            IDX[j] = rebind[IDX.ElementType](Int32(gi))
            W[j] = rebind[W.ElementType](gbest)
        wsum += gbest
    if lane == 0:
        for j in range(TOPK):
            W[j] = rebind[W.ElementType](rebind[Scalar[f32]](W[j]) / wsum)
    var Xv = X.vectorize[8]()
    var Gv = G.vectorize[8]()
    var acc = SIMD[f32, 8](0)
    var i = lane
    while i < K // 8:
        acc = fma(rebind[SIMD[f32, 8]](Xv[i]), rebind[SIMD[f32, 8]](Gv[i]), acc)
        i += WARP_SIZE
    var t = warp.sum(acc.reduce_add())
    if lane == 0:
        O[0] = rebind[O.ElementType](Scalar[f32](1) / (Scalar[f32](1) + exp(-t)))


def amar_moe_router_top8_sig[
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
    var IDX_ = IDX
    var W_ = W
    var O_ = O
    router_top8_sig_body(L, IDX_, W_, X, G, O_, Int(k_dim))


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
    var Xv = X.vectorize[8]()
    var Gv = G.vectorize[8]()
    var acc = SIMD[f32, 8](0)
    var i = lane
    while i < K // 8:
        acc = fma(rebind[SIMD[f32, 8]](Xv[i]), rebind[SIMD[f32, 8]](Gv[i]), acc)
        i += WARP_SIZE
    var t = warp.sum(acc.reduce_add())
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


@always_inline
def q4k_dot_blocks[
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
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // Q4K
    var acc = SIMD[f32, 16](0)
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
        var a0 = rebind[SIMD[bf16, 16]](Xv[x_row, k0 // 16]).cast[f32]()
        var a1 = rebind[SIMD[bf16, 16]](Xv[x_row, (k0 + 32) // 16]).cast[f32]()
        acc = fma(v0, a0, fma(v1, a1, acc))
        b += 4
    return warp.sum(acc.reduce_add())


@always_inline
def q4k_dot_blocks_fill[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    F: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    fill_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // Q4K
    var acc = SIMD[f32, 16](0)
    var b = lane // 8
    var s = lane % 8
    var pair = s // 2
    var half = (s % 2) * 16
    while b < nb:
        var base = row_base + b * Q4K_BYTES
        var hdr = W.unsafe_offset(base).load[width=16]()
        var fbase = fill_base + b * Q4K_BYTES
        if s == 0:
            F.unsafe_offset(fbase).store(hdr)
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
        F.unsafe_offset(fbase + 16 + pair * 32 + half).store(qb)
        var lo = (qb & 0x0F).cast[f32]()
        var hi = (qb >> 4).cast[f32]()
        var v0 = (SIMD[f32, 16](d * Scalar[f32](sc0)) * lo - SIMD[f32, 16](dm * Scalar[f32](mn0))).cast[bf16]().cast[f32]()
        var v1 = (SIMD[f32, 16](d * Scalar[f32](sc1)) * hi - SIMD[f32, 16](dm * Scalar[f32](mn1))).cast[bf16]().cast[f32]()
        var k0 = b * Q4K + g0 * 32 + half
        var a0 = rebind[SIMD[bf16, 16]](Xv[x_row, k0 // 16]).cast[f32]()
        var a1 = rebind[SIMD[bf16, 16]](Xv[x_row, (k0 + 32) // 16]).cast[f32]()
        acc = fma(v0, a0, fma(v1, a1, acc))
        b += 4
    return warp.sum(acc.reduce_add())


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
    var p = w.unsafe_offset(row_base + block * 34)
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
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = SIMD[f32, 16](0)
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var base = row_base + b * 34
        var raw = W.unsafe_offset(base).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(base + 2 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


@always_inline
def q8d_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    q_base: Int,
    s_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = SIMD[f32, 16](0)
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var raw = W.unsafe_offset(s_base + b * 2).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(q_base + b * 32 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


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


def moe_matmul_q8d_m1[
    OLayout: TensorLayout, ALayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 1
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var k = Int(k_dim)
    var dot = q8d_row_dot(A, W, 0, row * k, Int(s_off) + row * (k // 32) * 2, k)
    if lane_id() == 0:
        O[row] = rebind[O.ElementType](dot)


def moe_matmul_q8d_m1_add[
    OLayout: TensorLayout, ALayout: TensorLayout, XLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 1 and X.flat_rank == 1
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var k = Int(k_dim)
    var dot = q8d_row_dot(A, W, 0, row * k, Int(s_off) + row * (k // 32) * 2, k)
    if lane_id() == 0:
        O[row] = rebind[O.ElementType](dot)
        X[row] = rebind[X.ElementType](rebind[Scalar[f32]](X[row]) + dot)


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
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout, out_dt: DType = f32,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[out_dt, HLayout, MutAnyOrigin],
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
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[out_dt]())


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


def moe_down_q8_0_res[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    ALayout: TensorLayout, XLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    A: TileTensor[f32, ALayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
    row_bytes: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and A.flat_rank == 1 and X.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if c >= Int(n):
        return
    var out = Scalar[f32](0)
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = (e * Int(n) + c) * Int(row_bytes)
        out += rebind[Scalar[f32]](WT[j]) * q8_0_row_dot(Hb, WD, j, row_base, FFN)
    if lane_id() == 0:
        X[c] = rebind[X.ElementType](
            rebind[Scalar[f32]](X[c]) + rebind[Scalar[f32]](A[c]) + out
        )


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
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout, out_dt: DType = f32
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[out_dt, HLayout, MutAnyOrigin],
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
    var g = q4k_dot_blocks(Xb, W, 0, row_base, Int(k_dim))
    var u = q4k_dot_blocks(Xb, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[out_dt]())


def moe_gate_up_q4k_zc[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout, out_dt: DType = f32
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HE: TileTensor[i32, ILayout, MutAnyOrigin],
    HG: MutPointer[Scalar[u8], MutAnyOrigin],
    HO: TileTensor[out_dt, HLayout, MutAnyOrigin],
    k_dim: Int32,
    up_offset: Int32,
    h_up_offset: Int32,
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
    var he = Int(rebind[Scalar[i32]](HE[j]))
    var g: Scalar[f32]
    var u: Scalar[f32]
    if he >= 0:
        var hbase = he * FFN * row_bytes + r * row_bytes
        g = q4k_dot_blocks_fill(Xb, HG, W, 0, hbase, row_base, Int(k_dim))
        u = q4k_dot_blocks_fill(Xb, HG, W, 0, hbase + Int(h_up_offset), row_base + Int(up_offset), Int(k_dim))
    else:
        g = q4k_dot_blocks(Xb, W, 0, row_base, Int(k_dim))
        u = q4k_dot_blocks(Xb, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[out_dt]())


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
        var dot = q4k_dot_blocks(Hb, WD, j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[j]) * dot
    if lane == 0:
        O[c] = rebind[O.ElementType](out)


def amar_moe_down_q4k_zc[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HE: TileTensor[i32, ILayout, MutAnyOrigin],
    HD: MutPointer[Scalar[u8], MutAnyOrigin],
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
        var he = Int(rebind[Scalar[i32]](HE[j]))
        var dot: Scalar[f32]
        if he >= 0:
            dot = q4k_dot_blocks_fill(Hb, HD, WD, j, he * N * row_bytes + c * row_bytes, row_base, FFN)
        else:
            dot = q4k_dot_blocks(Hb, WD, j, row_base, FFN)
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
