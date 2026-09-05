from std.atomic import Atomic, Ordering
from std.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import exp, log1p, rsqrt, sqrt
from std.sys import llvm_intrinsic
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from elementwise import EW_THREADS
from matmul_skinny import ROW_WAVES, ROW_THREADS
from ssm import CONV, KDIM, NH_K, NH_V, SSTATE, SSM_EPS

comptime u32 = DType.uint32
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime f16 = DType.float16
comptime MEGA_G = 96
comptime SPIN_LIMIT = 1 << 22
comptime QV = 16
comptime UNROLL = 4


@always_inline
def grid_barrier(
    ctr: MutPointer[Scalar[u32], MutAnyOrigin],
    gen: MutPointer[Scalar[u32], MutAnyOrigin],
    fail: MutPointer[Scalar[u32], MutAnyOrigin],
) -> Bool:
    barrier()
    if thread_idx.x == 0:
        var nb = UInt32(grid_dim.x)
        var g = Atomic[u32, scope="agent"].load[ordering=Ordering.ACQUIRE](gen)
        if Atomic[u32, scope="agent"].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](ctr, 1) == nb - 1:
            Atomic[u32, scope="agent"].store[ordering=Ordering.RELAXED](ctr, 0)
            Atomic[u32, scope="agent"].store[ordering=Ordering.RELEASE](gen, g + 1)
        else:
            var spins = 0
            while Atomic[u32, scope="agent"].load[ordering=Ordering.ACQUIRE](gen) == g:
                llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
                spins += 1
                if spins > SPIN_LIMIT or Atomic[u32, scope="agent"].load[ordering=Ordering.RELAXED](fail) != 0:
                    Atomic[u32, scope="agent"].store[ordering=Ordering.RELEASE](fail, 1)
                    break
    barrier()
    return Atomic[u32, scope="agent"].load[ordering=Ordering.ACQUIRE](fail) == 0


@always_inline
def q8_row_dot[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[i8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    row: Int, lane: Int, K: Int,
) -> Float32:
    comptime STEP = WARP_SIZE * QV
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = SIMD[f32, QV](0)
    var kk = 0
    while kk + UNROLL * STEP <= K:
        var qs = InlineArray[SIMD[i8, QV], UNROLL](uninitialized=True)
        var ds = InlineArray[Scalar[f16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            qs[u] = rebind[SIMD[i8, QV]](Qv[row, kb // QV + lane])
            ds[u] = rebind[Scalar[f16]](S[row, (kb + lane * QV) // 32])
        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            var w = qs[u].cast[f32]() * ds[u].cast[f32]()
            var a = rebind[SIMD[bf16, QV]](Av[0, kb // QV + lane]).cast[f32]()
            acc += w * a
        kk += UNROLL * STEP
    while kk < K:
        var q = rebind[SIMD[i8, QV]](Qv[row, kk // QV + lane]).cast[f32]()
        var d = rebind[Scalar[f16]](S[row, (kk + lane * QV) // 32]).cast[f32]()
        var a = rebind[SIMD[bf16, QV]](Av[0, kk // QV + lane]).cast[f32]()
        acc += (q * d) * a
        kk += STEP
    return warp.sum(acc.reduce_add())


def amar_mega_ssm_layer[
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout,
    QqL: TensorLayout, QsL: TensorLayout,
    HqL: TensorLayout, HsL: TensorLayout,
    AqL: TensorLayout, AsL: TensorLayout,
    CwL: TensorLayout, G32L: TensorLayout, NwL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout, CtrL: TensorLayout,
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    Wqkvq: TileTensor[i8, QqL, MutAnyOrigin], Wqkvs: TileTensor[f16, QsL, MutAnyOrigin],
    Wzq: TileTensor[i8, HqL, MutAnyOrigin], Wzs: TileTensor[f16, HsL, MutAnyOrigin],
    Waq: TileTensor[i8, AqL, MutAnyOrigin], Was: TileTensor[f16, AsL, MutAnyOrigin],
    Wbq: TileTensor[i8, AqL, MutAnyOrigin], Wbs: TileTensor[f16, AsL, MutAnyOrigin],
    Cw: TileTensor[f32, CwL, MutAnyOrigin],
    SsmA: TileTensor[f32, G32L, MutAnyOrigin],
    DtB: TileTensor[f32, G32L, MutAnyOrigin],
    Nw: TileTensor[f32, NwL, MutAnyOrigin],
    Wsoutq: TileTensor[i8, HqL, MutAnyOrigin], Wsouts: TileTensor[f16, HsL, MutAnyOrigin],
    Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    Zm: TileTensor[f32, XL, MutAnyOrigin],
    Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    So: TileTensor[f32, OmL, MutAnyOrigin],
    ResB: TileTensor[bf16, CBL, MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32,
):
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Qkvm.flat_rank == 2
    comptime assert Conv.flat_rank == 2 and So.flat_rank == 3 and Eg.flat_rank == 2
    comptime assert ConvState.flat_rank == 4 and SAll.flat_rank == 5
    comptime H = 4096
    comptime G_QKV = CONV // ROW_WAVES
    comptime G_Z = H // ROW_WAVES
    comptime G_AB = NH_V // ROW_WAVES
    comptime G_ALL = G_QKV + G_Z + 2 * G_AB
    comptime G_OUT = H // ROW_WAVES
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)
    var rs = rg % sl
    var ws = (rg + 1) % sl
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2, SSTATE]()
    )
    if Atomic[u32, scope="agent"].load[ordering=Ordering.ACQUIRE](fail) != 0:
        return

    if bid == 0:
        var partial: Float32 = 0
        if tid < EW_THREADS:
            var i = tid
            while i < H:
                var v = rebind[Scalar[f32]](X[0, i])
                partial += v * v
                i += EW_THREADS
            var wsum = warp.sum(partial)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](wsum)
        barrier()
        if tid < EW_THREADS:
            var total: Float32 = 0
            comptime for w in range(EW_THREADS // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            var scale = rsqrt(total / Float32(H) + Float32(1e-6))
            var i = tid
            while i < H:
                CurB[0, i] = rebind[CurB.ElementType](
                    (rebind[Scalar[f32]](X[0, i]) * scale * rebind[Scalar[f32]](Gn[i])).cast[bf16]()
                )
                i += EW_THREADS
    if not grid_barrier(ctr, gen, fail):
        return

    var g = bid
    while g < G_ALL:
        if g < G_QKV:
            var row = g * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wqkvq, Wqkvs, row, lane, H)
            if lane == 0:
                Qkvm[0, row] = rebind[Qkvm.ElementType](t)
        elif g < G_QKV + G_Z:
            var row = (g - G_QKV) * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wzq, Wzs, row, lane, H)
            if lane == 0:
                Zm[0, row] = rebind[Zm.ElementType](t)
        elif g < G_QKV + G_Z + G_AB:
            var row = (g - G_QKV - G_Z) * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Waq, Was, row, lane, H)
            if lane == 0:
                Araw[0, row] = rebind[Araw.ElementType](t)
        else:
            var row = (g - G_QKV - G_Z - G_AB) * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wbq, Wbs, row, lane, H)
            if lane == 0:
                Braw[0, row] = rebind[Braw.ElementType](t)
        g += nblk
    if not grid_barrier(ctr, gen, fail):
        return

    if bid == 0 and tid < NH_V:
        var h = tid
        var braw = rebind[Scalar[f32]](Braw[0, h])
        Beta[0, h] = rebind[Beta.ElementType](1 / (1 + exp(-braw)))
        var asum = rebind[Scalar[f32]](Araw[0, h]) + rebind[Scalar[f32]](DtB[h])
        var sp = log1p(exp(asum))
        Eg[0, h] = rebind[Eg.ElementType](exp(sp * rebind[Scalar[f32]](SsmA[h])))
    var c = bid * ROW_THREADS + tid
    while c < CONV:
        var cw0 = rebind[Scalar[f32]](Cw[c, 0])
        var cw1 = rebind[Scalar[f32]](Cw[c, 1])
        var cw2 = rebind[Scalar[f32]](Cw[c, 2])
        var cw3 = rebind[Scalar[f32]](Cw[c, 3])
        var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
        var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
        var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
        var w3 = rebind[Scalar[f32]](Qkvm[0, c])
        var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
        Conv[0, c] = rebind[Conv.ElementType](acc / (1 + exp(-acc)))
        ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
        ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
        ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)
        c += nblk * ROW_THREADS
    if not grid_barrier(ctr, gen, fail):
        return

    if bid < NH_V:
        var head = bid
        var base = head * SSTATE
        var v: Float32 = 0
        if tid < SSTATE:
            v = rebind[Scalar[f32]](Conv[0, base + tid])
            var ssq = warp.sum(v * v)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](ssq)
        barrier()
        if tid < SSTATE:
            var total: Float32 = 0
            comptime for w in range(SSTATE // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            var inv = rsqrt(total + SSM_EPS)
            if head < NH_K:
                inv = inv / sqrt(Float32(SSTATE))
            Conv[0, base + tid] = rebind[Conv.ElementType](v * inv)
    if not grid_barrier(ctr, gen, fail):
        return

    if bid < NH_V:
        var h = bid
        var j = tid
        var kh = h % NH_K
        if j < SSTATE:
            kq[0, j] = rebind[kq.ElementType](Conv[0, kh * SSTATE + j])
            kq[1, j] = rebind[kq.ElementType](Conv[0, KDIM + kh * SSTATE + j])
        barrier()
        if j < SSTATE:
            var eg = rebind[Scalar[f32]](Eg[0, h])
            var beta = rebind[Scalar[f32]](Beta[0, h])
            var vj = rebind[Scalar[f32]](Conv[0, 2 * KDIM + h * SSTATE + j])
            var col = SIMD[f32, SSTATE]()
            comptime for i in range(SSTATE):
                col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i, j])
            var sk: Float32 = 0
            comptime for i in range(SSTATE):
                sk += col[i] * eg * rebind[Scalar[f32]](kq[1, i])
            var d = (vj - sk) * beta
            var o: Float32 = 0
            comptime for i in range(SSTATE):
                var s = col[i] * eg + rebind[Scalar[f32]](kq[1, i]) * d
                SAll[ws, si, h, i, j] = rebind[SAll.ElementType](s)
                o += s * rebind[Scalar[f32]](kq[0, i])
            So[0, h, j] = rebind[So.ElementType](o)
    if not grid_barrier(ctr, gen, fail):
        return

    if bid < NH_V:
        var h = bid
        var j = tid
        var v: Float32 = 0
        if j < SSTATE:
            v = rebind[Scalar[f32]](So[0, h, j])
            var ssq = warp.sum(v * v)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](ssq)
        barrier()
        if j < SSTATE:
            var total: Float32 = 0
            comptime for w in range(SSTATE // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
            var z = rebind[Scalar[f32]](Zm[0, h * SSTATE + j])
            ResB[0, h * SSTATE + j] = rebind[ResB.ElementType](
                (v * scale * rebind[Scalar[f32]](Nw[j]) * (z / (1 + exp(-z)))).cast[bf16]()
            )
    if not grid_barrier(ctr, gen, fail):
        return

    g = bid
    while g < G_OUT:
        var row = g * ROW_WAVES + wave
        var t = q8_row_dot(ResB, Wsoutq, Wsouts, row, lane, H)
        if lane == 0:
            X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t)
        g += nblk
