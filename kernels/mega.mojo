from std.atomic import Atomic, Ordering
from std.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import cos, exp, log, log1p, rsqrt, sin
from std.math import sqrt
from std.sys import llvm_intrinsic
from std.utils import StaticTuple
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from elementwise import EW_THREADS
from matmul_skinny import ROW_WAVES, ROW_THREADS
from ssm import CONV, KDIM, NH_K, NH_V, SSTATE, SSM_EPS
from attn import HD, NQH, NKVH, MAX_T, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE

comptime u32 = DType.uint32
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime f16 = DType.float16
comptime MEGA_G = 96
comptime SPIN_LIMIT = 1 << 22
comptime QV = 16
comptime UNROLL = 4
comptime u8 = DType.uint8
comptime i64 = DType.int64
comptime H = 4096
comptime FFN = 12288
comptime QF = 2 * H
comptime KV = NKVH * HD
comptime N_LAYERS = 32
comptime VOCAB = 248320
comptime i32 = DType.int32
comptime ATT_SCALE = Float32(0.0625)
comptime RMS_EPS = Float32(1e-6)
comptime q_h_qf = row_major[QF, H]()
comptime s_h_qf = row_major[QF, H // 32]()
comptime q_h_h = row_major[H, H]()
comptime s_h_h = row_major[H, H // 32]()
comptime q_h_kv = row_major[KV, H]()
comptime s_h_kv = row_major[KV, H // 32]()
comptime q_h_32 = row_major[NH_V, H]()
comptime s_h_32 = row_major[NH_V, H // 32]()
comptime q_h_ffn = row_major[FFN, H]()
comptime s_h_ffn = row_major[FFN, H // 32]()
comptime q_ffn_h = row_major[H, FFN]()
comptime s_ffn_h = row_major[H, FFN // 32]()
comptime q_conv_h = row_major[CONV, H]()
comptime s_conv_h = row_major[CONV, H // 32]()
comptime h_layout = row_major[H]()
comptime hd_layout = row_major[HD]()
comptime cw_layout = row_major[CONV, 4]()
comptime g32_layout = row_major[NH_V]()
comptime n128_layout = row_major[SSTATE]()


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
def stamp(prof: MutPointer[Scalar[i64], MutAnyOrigin], idx: Int):
    if block_idx.x == 0 and thread_idx.x == 0:
        prof[idx] = llvm_intrinsic["llvm.amdgcn.s.sendmsg.rtn", Int64](Int32(131))


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


@always_inline
def ssm_phases[
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout,
    QqL: TensorLayout, QsL: TensorLayout,
    HqL: TensorLayout, HsL: TensorLayout,
    AqL: TensorLayout, AsL: TensorLayout,
    CwL: TensorLayout, G32L: TensorLayout, NwL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout, CtrL: TensorLayout,
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    Wqkvq: TileTensor[i8, QqL, MutAnyOrigin], Wqkvs: TileTensor[f16, QsL, MutAnyOrigin],
    Wzq: TileTensor[i8, HqL, MutAnyOrigin], Wzs: TileTensor[f16, HsL, MutAnyOrigin],
    Waq: TileTensor[i8, AqL, MutAnyOrigin], Was: TileTensor[f16, AsL, MutAnyOrigin],
    Wbq: TileTensor[i8, AqL, MutAnyOrigin], Wbs: TileTensor[f16, AsL, MutAnyOrigin],
    Cw: TileTensor[f32, CwL, MutAnyOrigin],
    SsmA: TileTensor[f32, G32L, MutAnyOrigin],
    DtB: TileTensor[f32, G32L, MutAnyOrigin],
    Nw: TileTensor[f32, NwL, MutAnyOrigin],
    Wsoutq: TileTensor[i8, HqL, MutAnyOrigin], Wsouts: TileTensor[f16, HsL, MutAnyOrigin],
    mut Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    mut Zm: TileTensor[f32, XL, MutAnyOrigin],
    mut Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    mut So: TileTensor[f32, OmL, MutAnyOrigin],
    mut ResB: TileTensor[bf16, CBL, MutAnyOrigin],
    mut ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    mut SAll: TileTensor[f32, SsL, MutAnyOrigin],
    mut Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32,
    prof: MutPointer[Scalar[i64], MutAnyOrigin], pbase: Int,
) -> Bool:
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Qkvm.flat_rank == 2
    comptime assert Conv.flat_rank == 2 and So.flat_rank == 3 and Eg.flat_rank == 2
    comptime assert ConvState.flat_rank == 4 and SAll.flat_rank == 5
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

    stamp(prof, pbase + 12)
    rmsc_phase(X, Gn, CurB)
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 1)

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
        return False
    stamp(prof, pbase + 2)

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
        return False
    stamp(prof, pbase + 3)

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
        return False
    stamp(prof, pbase + 4)

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
        return False
    stamp(prof, pbase + 5)

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
        return False
    stamp(prof, pbase + 6)

    g = bid
    while g < G_OUT:
        var row = g * ROW_WAVES + wave
        var t = q8_row_dot(ResB, Wsoutq, Wsouts, row, lane, H)
        if lane == 0:
            X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t)
        g += nblk
    return True


@always_inline
def rmsc_phase[
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
):
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Gn.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
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
        var scale = rsqrt(total / Float32(H) + RMS_EPS)
        var i = Int(block_idx.x) * EW_THREADS + tid
        while i < H:
            CurB[0, i] = rebind[CurB.ElementType](
                (rebind[Scalar[f32]](X[0, i]) * scale * rebind[Scalar[f32]](Gn[i])).cast[bf16]()
            )
            i += Int(grid_dim.x) * EW_THREADS


@always_inline
def ffn_phases[
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout,
    GqL: TensorLayout, GsL: TensorLayout, DqL: TensorLayout, DsL: TensorLayout,
    PL: TensorLayout, FBL: TensorLayout, CtrL: TensorLayout,
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    Wgq: TileTensor[i8, GqL, MutAnyOrigin], Wgs: TileTensor[f16, GsL, MutAnyOrigin],
    Wuq: TileTensor[i8, GqL, MutAnyOrigin], Wus: TileTensor[f16, GsL, MutAnyOrigin],
    Wdq: TileTensor[i8, DqL, MutAnyOrigin], Wds: TileTensor[f16, DsL, MutAnyOrigin],
    mut Pg: TileTensor[f32, PL, MutAnyOrigin],
    mut Pu: TileTensor[f32, PL, MutAnyOrigin],
    mut FgB: TileTensor[bf16, FBL, MutAnyOrigin],
    mut Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin], pbase: Int,
) -> Bool:
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Gn.flat_rank == 1
    comptime assert Pg.flat_rank == 2 and Pu.flat_rank == 2 and FgB.flat_rank == 2
    comptime G_F = FFN // ROW_WAVES
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)

    rmsc_phase(X, Gn, CurB)
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 1)

    var g = bid
    while g < 2 * G_F:
        if g < G_F:
            var row = g * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wgq, Wgs, row, lane, H)
            if lane == 0:
                Pg[0, row] = rebind[Pg.ElementType](t)
        else:
            var row = (g - G_F) * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wuq, Wus, row, lane, H)
            if lane == 0:
                Pu[0, row] = rebind[Pu.ElementType](t)
        g += nblk
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 2)

    var c = bid * ROW_THREADS + tid
    while c < FFN:
        var gg = rebind[Scalar[f32]](Pg[0, c])
        var u = rebind[Scalar[f32]](Pu[0, c])
        var silu = gg / (1 + exp(-gg))
        FgB[0, c] = rebind[FgB.ElementType]((silu * u).cast[bf16]())
        c += nblk * ROW_THREADS
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 3)

    g = bid
    while g < H // ROW_WAVES:
        var row = g * ROW_WAVES + wave
        var t = q8_row_dot(FgB, Wdq, Wds, row, lane, FFN)
        if lane == 0:
            X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t)
        g += nblk
    return True


@always_inline
def rope_cs(j: Int, pos: Int) -> SIMD[f32, 2]:
    var theta_ex = Float32(pos) * exp(Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE))
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    return SIMD[f32, 2](cos(theta) * MSCALE, sin(theta) * MSCALE)


@always_inline
def attn_phases[
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout,
    QqL: TensorLayout, QsL: TensorLayout, KqL: TensorLayout, KsL: TensorLayout,
    HqL: TensorLayout, HsL: TensorLayout, HdL: TensorLayout,
    QfL: TensorLayout, KvfL: TensorLayout, QmL: TensorLayout,
    GfL: TensorLayout, CacheL: TensorLayout, CtrL: TensorLayout,
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    Wqq: TileTensor[i8, QqL, MutAnyOrigin], Wqs: TileTensor[f16, QsL, MutAnyOrigin],
    Wkq: TileTensor[i8, KqL, MutAnyOrigin], Wks: TileTensor[f16, KsL, MutAnyOrigin],
    Wvq: TileTensor[i8, KqL, MutAnyOrigin], Wvs: TileTensor[f16, KsL, MutAnyOrigin],
    Qn: TileTensor[f32, HdL, MutAnyOrigin],
    Kn: TileTensor[f32, HdL, MutAnyOrigin],
    Woq: TileTensor[i8, HqL, MutAnyOrigin], Wos: TileTensor[f16, HsL, MutAnyOrigin],
    mut Qfm: TileTensor[f32, QfL, MutAnyOrigin],
    mut Kflat: TileTensor[f32, KvfL, MutAnyOrigin],
    mut Vflat: TileTensor[f32, KvfL, MutAnyOrigin],
    mut Q: TileTensor[f32, QmL, MutAnyOrigin],
    mut Gate: TileTensor[f32, GfL, MutAnyOrigin],
    mut Ao: TileTensor[f32, QmL, MutAnyOrigin],
    mut AoB: TileTensor[bf16, CBL, MutAnyOrigin],
    mut Kc: TileTensor[f32, CacheL, MutAnyOrigin],
    mut Vc: TileTensor[f32, CacheL, MutAnyOrigin],
    mut Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    pos: Int,
    prof: MutPointer[Scalar[i64], MutAnyOrigin], pbase: Int,
) -> Bool:
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Gn.flat_rank == 1
    comptime assert Qfm.flat_rank == 2 and Kflat.flat_rank == 2 and Vflat.flat_rank == 2
    comptime assert Q.flat_rank == 2 and Ao.flat_rank == 2 and Gate.flat_rank == 1 and AoB.flat_rank == 2
    comptime assert Kc.flat_rank == 3 and Vc.flat_rank == 3 and Qn.flat_rank == 1 and Kn.flat_rank == 1
    comptime G_Q = QF // ROW_WAVES
    comptime G_KV = KV // ROW_WAVES
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD]()
    )
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[MAX_T]()
    )

    rmsc_phase(X, Gn, CurB)
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 1)

    var g = bid
    while g < G_Q + 2 * G_KV:
        if g < G_Q:
            var row = g * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wqq, Wqs, row, lane, H)
            if lane == 0:
                Qfm[0, row] = rebind[Qfm.ElementType](t)
        elif g < G_Q + G_KV:
            var row = (g - G_Q) * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wkq, Wks, row, lane, H)
            if lane == 0:
                Kflat[0, row] = rebind[Kflat.ElementType](t)
        else:
            var row = (g - G_Q - G_KV) * ROW_WAVES + wave
            var t = q8_row_dot(CurB, Wvq, Wvs, row, lane, H)
            if lane == 0:
                Vflat[0, row] = rebind[Vflat.ElementType](t)
        g += nblk
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 2)

    if bid < NQH:
        var h = bid
        var v: Float32 = 0
        if tid < HD:
            v = rebind[Scalar[f32]](Qfm[0, h * 2 * HD + tid])
            Gate[h * HD + tid] = rebind[Gate.ElementType](Qfm[0, h * 2 * HD + HD + tid])
            var ssq = warp.sum(v * v)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](ssq)
        barrier()
        if tid < HD:
            var total: Float32 = 0
            comptime for w in range(HD // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            Q[h, tid] = rebind[Q.ElementType](
                v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Qn[tid])
            )
        barrier()
        if tid < NROT // 2:
            var cs = rope_cs(tid, pos)
            var x0 = rebind[Scalar[f32]](Q[h, tid])
            var x1 = rebind[Scalar[f32]](Q[h, tid + NROT // 2])
            Q[h, tid] = rebind[Q.ElementType](x0 * cs[0] - x1 * cs[1])
            Q[h, tid + NROT // 2] = rebind[Q.ElementType](x0 * cs[1] + x1 * cs[0])
    elif bid < NQH + NKVH:
        var h = bid - NQH
        var v: Float32 = 0
        if tid < HD:
            v = rebind[Scalar[f32]](Kflat[0, h * HD + tid])
            var ssq = warp.sum(v * v)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](ssq)
        barrier()
        if tid < HD:
            var total: Float32 = 0
            comptime for w in range(HD // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            Kflat[0, h * HD + tid] = rebind[Kflat.ElementType](
                v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Kn[tid])
            )
        barrier()
        if tid < NROT // 2:
            var cs = rope_cs(tid, pos)
            var x0 = rebind[Scalar[f32]](Kflat[0, h * HD + tid])
            var x1 = rebind[Scalar[f32]](Kflat[0, h * HD + tid + NROT // 2])
            Kflat[0, h * HD + tid] = rebind[Kflat.ElementType](x0 * cs[0] - x1 * cs[1])
            Kflat[0, h * HD + tid + NROT // 2] = rebind[Kflat.ElementType](x0 * cs[1] + x1 * cs[0])
        barrier()
        if tid < HD:
            Kc[h, pos, tid] = rebind[Kc.ElementType](Kflat[0, h * HD + tid])
            Vc[h, pos, tid] = rebind[Vc.ElementType](Vflat[0, h * HD + tid])
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 3)

    if bid < NQH:
        var h = bid
        var kvh = h // (NQH // NKVH)
        var T = pos + 1
        if tid < HD:
            qs[tid] = rebind[qs.ElementType](Q[h, tid])
        barrier()
        var local_max = Float32(-3.4e38)
        if tid < HD:
            var t = tid
            while t < T:
                var acc: Float32 = 0
                for d in range(HD):
                    acc += rebind[Scalar[f32]](qs[d]) * rebind[Scalar[f32]](Kc[kvh, t, d])
                scores[t] = rebind[scores.ElementType](acc * ATT_SCALE)
                t += HD
        barrier()
        if tid < HD:
            var t = tid
            while t < T:
                var sc = rebind[Scalar[f32]](scores[t])
                if sc > local_max:
                    local_max = sc
                t += HD
            var wmax = warp.max(local_max)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](wmax)
        barrier()
        var row_max = Float32(-3.4e38)
        if tid < HD:
            comptime for w in range(HD // WARP_SIZE):
                var sc = rebind[Scalar[f32]](sums[w])
                if sc > row_max:
                    row_max = sc
        barrier()
        if tid < HD:
            var partial: Float32 = 0
            var t = tid
            while t < T:
                var e = exp(rebind[Scalar[f32]](scores[t]) - row_max)
                scores[t] = rebind[scores.ElementType](e)
                partial += e
                t += HD
            var wsum = warp.sum(partial)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](wsum)
        barrier()
        var inv: Float32 = 0
        if tid < HD:
            var total: Float32 = 0
            comptime for w in range(HD // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            inv = 1 / total
        barrier()
        if tid < HD:
            var o: Float32 = 0
            for tt in range(T):
                o += rebind[Scalar[f32]](scores[tt]) * rebind[Scalar[f32]](Vc[kvh, tt, tid])
            Ao[h, tid] = rebind[Ao.ElementType](o * inv)
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 4)

    var i = bid * ROW_THREADS + tid
    while i < H:
        var gg = rebind[Scalar[f32]](Gate[i])
        AoB[0, i] = rebind[AoB.ElementType](
            (rebind[Scalar[f32]](Ao[i // HD, i % HD]) * (1 / (1 + exp(-gg)))).cast[bf16]()
        )
        i += nblk * ROW_THREADS
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 5)

    g = bid
    while g < H // ROW_WAVES:
        var row = g * ROW_WAVES + wave
        var t = q8_row_dot(AoB, Woq, Wos, row, lane, H)
        if lane == 0:
            X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t)
        g += nblk
    return True


@always_inline
def wq[N: Int, K: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[i8, type_of(row_major[N, K]()), MutAnyOrigin]:
    return TileTensor((wbuf + o).unsafe_bitcast[Scalar[i8]](), row_major[N, K]())


@always_inline
def ws[N: Int, K: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f16, type_of(row_major[N, K // 32]()), MutAnyOrigin]:
    return TileTensor((wbuf + o + N * K).unsafe_bitcast[Scalar[f16]](), row_major[N, K // 32]())


@always_inline
def wf[N: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f32, type_of(row_major[N]()), MutAnyOrigin]:
    return TileTensor((wbuf + o).unsafe_bitcast[Scalar[f32]](), row_major[N]())


@always_inline
def wf2[N: Int, M: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f32, type_of(row_major[N, M]()), MutAnyOrigin]:
    return TileTensor((wbuf + o).unsafe_bitcast[Scalar[f32]](), row_major[N, M]())


@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_mega_token[
    XL: TensorLayout, CBL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout,
    QfL: TensorLayout, KvfL: TensorLayout, QmL: TensorLayout, GfL: TensorLayout,
    PfL: TensorLayout, FbL: TensorLayout, OffL: TensorLayout, CtrL: TensorLayout, TkL: TensorLayout,
    TM: Int, NL: Int,
](
    wbuf: MutPointer[Scalar[u8], MutAnyOrigin],
    off: TileTensor[i64, OffL, MutAnyOrigin],
    X: TileTensor[f32, XL, MutAnyOrigin],
    CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    ResB: TileTensor[bf16, CBL, MutAnyOrigin],
    Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    Zm: TileTensor[f32, XL, MutAnyOrigin],
    Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    So: TileTensor[f32, OmL, MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    Qfm: TileTensor[f32, QfL, MutAnyOrigin],
    Kflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Vflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Q: TileTensor[f32, QmL, MutAnyOrigin],
    Gate: TileTensor[f32, GfL, MutAnyOrigin],
    Ao: TileTensor[f32, QmL, MutAnyOrigin],
    kc: MutPointer[Scalar[f32], MutAnyOrigin],
    vc: MutPointer[Scalar[f32], MutAnyOrigin],
    Pg: TileTensor[f32, PfL, MutAnyOrigin],
    Pu: TileTensor[f32, PfL, MutAnyOrigin],
    FgB: TileTensor[bf16, FbL, MutAnyOrigin],
    Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin],
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    Toks: TileTensor[i32, TkL, MutAnyOrigin],
    hmax: MutPointer[Scalar[f32], MutAnyOrigin],
    hidx: MutPointer[Scalar[i32], MutAnyOrigin],
    ring: Int32, slots: Int32, pos: Int32, dump: Int32, fold_head: Int32,
):
    comptime assert off.flat_rank == 1 and Toks.flat_rank == 1
    var X_ = X
    var CurB_ = CurB
    var ResB_ = ResB
    var Qkvm_ = Qkvm
    var Zm_ = Zm
    var Araw_ = Araw
    var Braw_ = Braw
    var Eg_ = Eg
    var Beta_ = Beta
    var Conv_ = Conv
    var So_ = So
    var ConvState_ = ConvState
    var SAll_ = SAll
    var Qfm_ = Qfm
    var Kflat_ = Kflat
    var Vflat_ = Vflat
    var Q_ = Q
    var Gate_ = Gate
    var Ao_ = Ao
    var Pg_ = Pg
    var Pu_ = Pu
    var FgB_ = FgB
    var Ctr_ = Ctr
    comptime cache_layout = row_major[NKVH, TM, HD]()
    comptime ATT32 = NKVH * TM * HD
    var fail = Ctr_.ptr.unsafe_offset(2)
    if Atomic[u32, scope="agent"].load[ordering=Ordering.ACQUIRE](fail) != 0:
        return
    var w = 1
    var ssm_i = 0
    var att_i = 0
    var p = Int(pos)
    for layer in range(NL):
        stamp(prof, 16 * layer)
        var o0 = Int(rebind[Scalar[i64]](off[w]))
        if (layer + 1) % 4 == 0:
            var o1 = Int(rebind[Scalar[i64]](off[w + 1]))
            var o2 = Int(rebind[Scalar[i64]](off[w + 2]))
            var o3 = Int(rebind[Scalar[i64]](off[w + 3]))
            var o4 = Int(rebind[Scalar[i64]](off[w + 4]))
            var o5 = Int(rebind[Scalar[i64]](off[w + 5]))
            var o6 = Int(rebind[Scalar[i64]](off[w + 6]))
            var Kc = TileTensor(kc + att_i * ATT32, cache_layout)
            var Vc = TileTensor(vc + att_i * ATT32, cache_layout)
            if not attn_phases(
                X_, wf[H](wbuf, o0), CurB_,
                wq[QF, H](wbuf, o1), ws[QF, H](wbuf, o1),
                wq[KV, H](wbuf, o2), ws[KV, H](wbuf, o2),
                wq[KV, H](wbuf, o3), ws[KV, H](wbuf, o3),
                wf[HD](wbuf, o4), wf[HD](wbuf, o5),
                wq[H, H](wbuf, o6), ws[H, H](wbuf, o6),
                Qfm_, Kflat_, Vflat_, Q_, Gate_, Ao_, ResB_, Kc, Vc, Ctr_, p, prof, 16 * layer,
            ):
                return
            att_i += 1
            w += 7
        else:
            var o1 = Int(rebind[Scalar[i64]](off[w + 1]))
            var o2 = Int(rebind[Scalar[i64]](off[w + 2]))
            var o3 = Int(rebind[Scalar[i64]](off[w + 3]))
            var o4 = Int(rebind[Scalar[i64]](off[w + 4]))
            var o5 = Int(rebind[Scalar[i64]](off[w + 5]))
            var o6 = Int(rebind[Scalar[i64]](off[w + 6]))
            var o7 = Int(rebind[Scalar[i64]](off[w + 7]))
            var o8 = Int(rebind[Scalar[i64]](off[w + 8]))
            var o9 = Int(rebind[Scalar[i64]](off[w + 9]))
            if not ssm_phases(
                X_, wf[H](wbuf, o0), CurB_,
                wq[CONV, H](wbuf, o1), ws[CONV, H](wbuf, o1),
                wq[H, H](wbuf, o2), ws[H, H](wbuf, o2),
                wq[NH_V, H](wbuf, o3), ws[NH_V, H](wbuf, o3),
                wq[NH_V, H](wbuf, o4), ws[NH_V, H](wbuf, o4),
                wf2[CONV, 4](wbuf, o5), wf[NH_V](wbuf, o6), wf[NH_V](wbuf, o7), wf[SSTATE](wbuf, o8),
                wq[H, H](wbuf, o9), ws[H, H](wbuf, o9),
                Qkvm_, Zm_, Araw_, Braw_, Eg_, Beta_, Conv_, So_, ResB_, ConvState_, SAll_, Ctr_,
                ring, Int32(ssm_i), slots, prof, 16 * layer,
            ):
                return
            ssm_i += 1
            w += 10
        var ctr = Ctr_.ptr
        var gen = Ctr_.ptr.unsafe_offset(1)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, 16 * layer + 7)
        if dump != 0 and block_idx.x == 0:
            var i = Int(thread_idx.x)
            while i < H:
                dbg[(2 * layer) * H + i] = rebind[Scalar[f32]](X_[0, i])
                i += ROW_THREADS
        var f0 = Int(rebind[Scalar[i64]](off[w]))
        var f1 = Int(rebind[Scalar[i64]](off[w + 1]))
        var f2 = Int(rebind[Scalar[i64]](off[w + 2]))
        var f3 = Int(rebind[Scalar[i64]](off[w + 3]))
        if not ffn_phases(
            X_, wf[H](wbuf, f0), CurB_,
            wq[FFN, H](wbuf, f1), ws[FFN, H](wbuf, f1),
            wq[FFN, H](wbuf, f2), ws[FFN, H](wbuf, f2),
            wq[H, FFN](wbuf, f3), ws[H, FFN](wbuf, f3),
            Pg_, Pu_, FgB_, Ctr_, prof, 16 * layer + 7,
        ):
            return
        w += 4
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, 16 * layer + 11)
        if dump != 0 and block_idx.x == 0:
            var i = Int(thread_idx.x)
            while i < H:
                dbg[(2 * layer + 1) * H + i] = rebind[Scalar[f32]](X_[0, i])
                i += ROW_THREADS
    if fold_head == 0:
        return
    stamp(prof, 16 * NL)
    var Toks_ = Toks
    var ctrh = Ctr_.ptr
    var genh = Ctr_.ptr.unsafe_offset(1)
    var ho0 = Int(rebind[Scalar[i64]](off[w]))
    var ho1 = Int(rebind[Scalar[i64]](off[w + 1]))
    rmsc_phase(X_, wf[H](wbuf, ho0), CurB_)
    if not grid_barrier(ctrh, genh, fail):
        return
    stamp(prof, 16 * NL + 1)
    var Whq = wq[VOCAB, H](wbuf, ho1)
    var Whs = ws[VOCAB, H](wbuf, ho1)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var bv = Float32(-3.4e38)
    var bi: Int32 = 0
    var g = bid
    while g < VOCAB // ROW_WAVES:
        var row = g * ROW_WAVES + wave
        var t = q8_row_dot(CurB_, Whq, Whs, row, lane, H)
        if t > bv or (t == bv and Int32(row) < bi):
            bv = t
            bi = Int32(row)
        g += nblk
    var wv = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[ROW_WAVES]())
    var wi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[ROW_WAVES]())
    if lane == 0:
        wv[wave] = rebind[wv.ElementType](bv)
        wi[wave] = rebind[wi.ElementType](bi)
    barrier()
    if tid == 0:
        var pv = Float32(-3.4e38)
        var pi: Int32 = 0
        comptime for k in range(ROW_WAVES):
            var v = rebind[Scalar[f32]](wv[k])
            var ix = rebind[Scalar[i32]](wi[k])
            if v > pv or (v == pv and ix < pi):
                pv = v
                pi = ix
        hmax[bid] = pv
        hidx[bid] = pi
    if not grid_barrier(ctrh, genh, fail):
        return
    stamp(prof, 16 * NL + 2)
    if bid == 0 and tid == 0:
        var fv = Float32(-3.4e38)
        var fi: Int32 = 0
        for k in range(nblk):
            var v = hmax[k]
            var ix = hidx[k]
            if v > fv or (v == fv and ix < fi):
                fv = v
                fi = ix
        Toks_[p + 1] = rebind[Toks_.ElementType](fi)
    stamp(prof, 16 * NL + 3)
