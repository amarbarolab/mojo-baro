from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import cos, exp, fma, log, rsqrt, sin
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime f32 = DType.float32
comptime HD = 256
comptime NQH = 16
comptime NKVH = 4
comptime MAX_T = 1088


def amar_head_rmsnorm[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1
    var h = block_idx.x
    var d = thread_idx.x
    var v = rebind[Scalar[f32]](X[h, d])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[d // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(HD // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    X[h, d] = rebind[X.ElementType](
        v * rsqrt(total / Float32(HD) + eps) * rebind[Scalar[f32]](G[d])
    )


@always_inline
def attn_head_body[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout,
    QsL: TensorLayout, ScL: TensorLayout, RdL: TensorLayout
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[f32, KLayout, MutAnyOrigin],
    Vc: TileTensor[f32, KLayout, MutAnyOrigin],
    mut O: TileTensor[f32, OLayout, MutAnyOrigin],
    mut qs: TileTensor[f32, QsL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut scores: TileTensor[f32, ScL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut red: TileTensor[f32, RdL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    qrow: Int, kvh: Int, T: Int, tid: Int, lane: Int, scale: Float32,
):
    # One decode head (q row qrow against kv head kvh over T positions), shared
    # by amar_attn_decode and the megakernel's attention phase so both paths
    # compute the same bits (bench/attn-latency-protocol.md). Threads >= HD
    # (a 512-thread megakernel block) only take part in the barriers.
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 3 and Vc.flat_rank == 3 and O.flat_rank == 2
    comptime assert qs.flat_rank == 1 and scores.flat_rank == 1 and red.flat_rank == 1
    var wave = tid // WARP_SIZE
    if tid < HD:
        qs[tid] = rebind[qs.ElementType](Q[qrow, tid])
    barrier()
    var local_max = Float32(-3.4e38)
    var Kv = Kc.vectorize[1, 1, 8]()
    var qv = qs.vectorize[8]()
    if tid < HD:
        var t = tid
        while t < T:
            # 8-wide loads of K and q, scalar accumulation in the same order
            # as the element loop (bench/attn-latency-protocol.md A1)
            var acc: Float32 = 0
            for d8 in range(HD // 8):
                var k8 = rebind[SIMD[f32, 8]](Kv[kvh, t, d8])
                var q8 = rebind[SIMD[f32, 8]](qv[d8])
                comptime for j in range(8):
                    acc += q8[j] * k8[j]
            scores[t] = rebind[scores.ElementType](acc * scale)
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
            red[wave] = rebind[red.ElementType](wmax)
    barrier()
    var row_max = Float32(-3.4e38)
    if tid < HD:
        comptime for w in range(HD // WARP_SIZE):
            var sc = rebind[Scalar[f32]](red[w])
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
            red[wave] = rebind[red.ElementType](wsum)
    barrier()
    var inv: Float32 = 0
    if tid < HD:
        var total: Float32 = 0
        comptime for w in range(HD // WARP_SIZE):
            total += rebind[Scalar[f32]](red[w])
        inv = 1 / total
    barrier()
    if tid < HD:
        var o: Float32 = 0
        var tt = 0
        while tt + 8 <= T:
            var v = InlineArray[Float32, 8](uninitialized=True)
            var sc = InlineArray[Float32, 8](uninitialized=True)
            comptime for j in range(8):
                v[j] = rebind[Scalar[f32]](Vc[kvh, tt + j, tid])
                sc[j] = rebind[Scalar[f32]](scores[tt + j])
            comptime for j in range(8):
                o += sc[j] * v[j]
            tt += 8
        while tt < T:
            o += rebind[Scalar[f32]](scores[tt]) * rebind[Scalar[f32]](Vc[kvh, tt, tid])
            tt += 1
        O[qrow, tid] = rebind[O.ElementType](o * inv)


def amar_attn_decode[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[f32, KLayout, MutAnyOrigin],
    Vc: TileTensor[f32, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    t_len: Int32,
    scale: Float32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 3 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var kvh = h // (NQH // NKVH)
    var T = Int(t_len) + r
    var qrow = r * NQH + h
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[MAX_T]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var O_ = O
    attn_head_body(Q, Kc, Vc, O_, qs, scores, red, qrow, kvh, T, tid, Int(lane_id()), scale)


def amar_gate_mul[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Gate.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    var g = rebind[Scalar[f32]](Gate[i])
    X[i] = rebind[X.ElementType](
        rebind[Scalar[f32]](X[i]) * (1 / (1 + exp(-g)))
    )


def amar_gate_mul_cast[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Gate.flat_rank == 1 and O.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    var g = rebind[Scalar[f32]](Gate[i])
    O[i] = rebind[O.ElementType](
        (rebind[Scalar[f32]](X[i]) * (1 / (1 + exp(-g)))).cast[
            DType.bfloat16
        ]()
    )


comptime NROT = 64
comptime YARN_LOW = Float32(14.0)
comptime YARN_HIGH = Float32(22.0)
comptime FREQ_BASE = Float32(1e7)
comptime FREQ_SCALE = Float32(0.25)
comptime MSCALE = Float32(1.1386294361119891)


def amar_qgate_split[
    FLayout: TensorLayout, QLayout: TensorLayout, GLayout: TensorLayout
](
    Qfull: TileTensor[f32, FLayout, MutAnyOrigin],
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
):
    comptime assert Qfull.flat_rank == 2 and Q.flat_rank == 2 and Gate.flat_rank == 1
    var h = block_idx.x
    var r = Int(block_idx.y)
    var d = thread_idx.x
    Q[r * NQH + h, d] = rebind[Q.ElementType](Qfull[r, h * 2 * HD + d])
    Gate[r * NQH * HD + h * HD + d] = rebind[Gate.ElementType](
        Qfull[r, h * 2 * HD + HD + d]
    )


def amar_rope_yarn[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    pos: Int32,
    nh: Int32,
):
    comptime assert X.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var j = thread_idx.x
    var theta_ex = Float32(Int(pos) + r) * exp(
        Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE)
    )
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    var c = cos(theta) * MSCALE
    var s = sin(theta) * MSCALE
    var row = r * Int(nh) + h
    var x0 = rebind[Scalar[f32]](X[row, j])
    var x1 = rebind[Scalar[f32]](X[row, j + NROT // 2])
    X[row, j] = rebind[X.ElementType](x0 * c - x1 * s)
    X[row, j + NROT // 2] = rebind[X.ElementType](x0 * s + x1 * c)


def amar_kv_append[
    CLayout: TensorLayout, NLayout: TensorLayout
](
    Cache: TileTensor[f32, CLayout, MutAnyOrigin],
    New: TileTensor[f32, NLayout, MutAnyOrigin],
    t_idx: Int32,
):
    comptime assert Cache.flat_rank == 3 and New.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var d = thread_idx.x
    Cache[h, Int(t_idx) + r, d] = rebind[Cache.ElementType](New[r * NKVH + h, d])


comptime PA_TK = 16
comptime PA_ROWS = 2
comptime PA_KS = HD + 4


def amar_attn_prefill[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[f32, KLayout, MutAnyOrigin],
    Vc: TileTensor[f32, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    pos: Int32,
    m: Int32,
    scale: Float32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 3 and Vc.flat_rank == 3 and O.flat_rank == 2
    var kvh = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var M = Int(m)
    var P = Int(pos)
    var head = kvh * (NQH // NKVH) + wave % (NQH // NKVH)
    var r = tile * PA_ROWS + wave // (NQH // NKVH)
    var valid = r < M
    var qrow = r * NQH + head
    var T = P + r + 1
    var t_blk = P + min(tile * PA_ROWS + PA_ROWS, M)
    var j = lane % PA_TK
    var dpart = lane // PA_TK

    var ks = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[PA_TK, PA_KS]())
    var vs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[PA_TK, HD]())
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[PA_ROWS * (NQH // NKVH), HD]())
    var ksv = ks.vectorize[1, 8]()
    var vsv = vs.vectorize[1, 8]()
    var qsv = qs.vectorize[1, 8]()
    var Qv = Q.vectorize[1, 8]()
    var Kv = Kc.vectorize[1, 1, 8]()
    var Vv = Vc.vectorize[1, 1, 8]()
    var O_ = O

    if valid:
        qsv[wave, lane] = rebind[qsv.ElementType](Qv[qrow, lane])
    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var o = SIMD[f32, 8](0)
    var lp = tid // 16
    var lc = (tid % 16) * 2
    var t0 = 0
    while t0 < t_blk:
        barrier()
        var p = t0 + lp
        if p < t_blk:
            ksv[lp, lc] = rebind[ksv.ElementType](Kv[kvh, p, lc])
            ksv[lp, lc + 1] = rebind[ksv.ElementType](Kv[kvh, p, lc + 1])
            vsv[lp, lc] = rebind[vsv.ElementType](Vv[kvh, p, lc])
            vsv[lp, lc + 1] = rebind[vsv.ElementType](Vv[kvh, p, lc + 1])
        else:
            ksv[lp, lc] = rebind[ksv.ElementType](SIMD[f32, 8](0))
            ksv[lp, lc + 1] = rebind[ksv.ElementType](SIMD[f32, 8](0))
            vsv[lp, lc] = rebind[vsv.ElementType](SIMD[f32, 8](0))
            vsv[lp, lc + 1] = rebind[vsv.ElementType](SIMD[f32, 8](0))
        barrier()
        var part: Float32 = 0
        comptime for d8 in range(HD // 16):
            var q8 = rebind[SIMD[f32, 8]](qsv[wave, dpart * (HD // 16) + d8])
            var k8 = rebind[SIMD[f32, 8]](ksv[j, dpart * (HD // 16) + d8])
            comptime for i in range(8):
                part += q8[i] * k8[i]
        part += warp.shuffle_xor(part, UInt32(PA_TK))
        var s = Float32(-3.4e38)
        if valid and t0 + j < T:
            s = part * scale
        var tmax = warp.max(s)
        if tmax > Float32(-3.4e38):
            var m_new = max(m_run, tmax)
            var alpha = exp(m_run - m_new)
            var pj = exp(s - m_new)
            var psum = pj if dpart == 0 else Float32(0)
            l_run = l_run * alpha + warp.sum(psum)
            o = o * alpha
            comptime for jj in range(PA_TK):
                var pb = warp.shuffle_idx(pj, UInt32(jj))
                var v8 = rebind[SIMD[f32, 8]](vsv[jj, lane])
                o = fma(v8, SIMD[f32, 8](pb), o)
            m_run = m_new
        t0 += PA_TK
    if valid:
        var inv = 1 / l_run
        comptime for i in range(8):
            O_[qrow, lane * 8 + i] = rebind[O_.ElementType](o[i] * inv)
