from std.gpu import block_dim, block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from max.gpu.sync import barrier
from std.math import cos, exp, log, sin, tanh
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from attn import KVT, HD, NQH, NKVH, attn_head_span, kv_off, kv_tab_off
from matmul_skinny import dtype, ROW_WAVES

comptime f32 = DType.float32


def amar_embed_lookup_f32[
    TLayout: TensorLayout, OLayout: TensorLayout, KLayout: TensorLayout
](
    Table: TileTensor[f32, TLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Toks: TileTensor[DType.int32, KLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
):
    comptime assert Table.flat_rank == 2 and O.flat_rank == 2 and Toks.flat_rank == 1
    var idx = global_idx.x
    if idx >= Int(n):
        return
    var token = Int(rebind[Scalar[DType.int32]](Toks[Int(pos) + Int(block_idx.y)]))
    O[block_idx.y, idx] = rebind[O.ElementType](rebind[Scalar[f32]](Table[token, idx]))


def amar_rope_plain[
    NROT_: Int, XLayout: TensorLayout, NEOX: Bool = True
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    pos: Int32,
    nh: Int32,
    freq_base: Float32,
):
    comptime assert X.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var j = thread_idx.x
    var theta = Float32(Int(pos) + r) * exp(
        Float32(-2 * j) / Float32(NROT_) * log(freq_base)
    )
    var c = cos(theta)
    var s = sin(theta)
    var row = r * Int(nh) + h
    comptime if NEOX:
        var x0 = rebind[Scalar[f32]](X[row, j])
        var x1 = rebind[Scalar[f32]](X[row, j + NROT_ // 2])
        X[row, j] = rebind[X.ElementType](x0 * c - x1 * s)
        X[row, j + NROT_ // 2] = rebind[X.ElementType](x0 * s + x1 * c)
    else:
        var x0 = rebind[Scalar[f32]](X[row, 2 * j])
        var x1 = rebind[Scalar[f32]](X[row, 2 * j + 1])
        X[row, 2 * j] = rebind[X.ElementType](x0 * c - x1 * s)
        X[row, 2 * j + 1] = rebind[X.ElementType](x0 * s + x1 * c)


def amar_gemv_q8[
    EPI: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    CLayout: TensorLayout, BLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    Ob: TileTensor[DType.bfloat16, BLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2
    comptime assert C.flat_rank == 1 and Ob.flat_rank == 1
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return
    comptime QV = 16
    comptime STEP = WARP_SIZE * QV
    comptime UNROLL = 4
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = SIMD[f32, QV](0)
    var kk = 0
    while kk + UNROLL * STEP <= K:
        var qs = InlineArray[SIMD[DType.int8, QV], UNROLL](uninitialized=True)
        var ds = InlineArray[Scalar[DType.float16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            qs[u] = rebind[SIMD[DType.int8, QV]](Qv[row, kb // QV + lane])
            ds[u] = rebind[Scalar[DType.float16]](S[row, (kb + lane * QV) // 32])
        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            var w = qs[u].cast[f32]() * ds[u].cast[f32]()
            var a = rebind[SIMD[DType.bfloat16, QV]](Av[0, kb // QV + lane]).cast[f32]()
            acc += w * a
        kk += UNROLL * STEP
    while kk < K:
        var q = rebind[SIMD[DType.int8, QV]](Qv[row, kk // QV + lane]).cast[f32]()
        var d = rebind[Scalar[DType.float16]](S[row, (kk + lane * QV) // 32]).cast[f32]()
        var a = rebind[SIMD[DType.bfloat16, QV]](Av[0, kk // QV + lane]).cast[f32]()
        acc += q * d * a
        kk += STEP
    var total = warp.sum(acc.reduce_add())
    if lane == 0:
        comptime if EPI == 0:
            C[row] = rebind[C.ElementType](total)
        elif EPI == 1:
            C[row] = rebind[C.ElementType](rebind[Scalar[f32]](C[row]) + total)
        elif EPI == 2:
            var g = rebind[Scalar[f32]](C[row])
            var gelu = 0.5 * g * (1 + tanh(0.7978845608028654 * g * (1 + 0.044715 * g * g)))
            Ob[row] = rebind[Ob.ElementType]((gelu * total).cast[DType.bfloat16]())
        else:
            var g = rebind[Scalar[f32]](C[row])
            var silu = g / (1 + exp(-g))
            Ob[row] = rebind[Ob.ElementType]((silu * total).cast[DType.bfloat16]())


def amar_rope_kv_append[
    NROT_: Int, NAT: Int, CLayout: TensorLayout, NLayout: TensorLayout, HD_: Int = HD, NKVH_: Int = NKVH, NEOX: Bool = True
](
    Kc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, CLayout, MutAnyOrigin],
    K: TileTensor[f32, NLayout, MutAnyOrigin],
    V: TileTensor[f32, NLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    pos: Int32,
    freq_base: Float32,
    att_i: Int32,
):
    comptime assert Kc.flat_rank == 1 and K.flat_rank == 2
    var h = block_idx.x
    var which = Int(block_idx.y)
    var d = Int(thread_idx.x)
    var cb = kv_tab_off[NAT, HD_, NKVH_](tab, Int(pos), Int(att_i), Int(h)) + d
    if which == 1:
        Vc.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](rebind[Scalar[f32]](V[h, d]).cast[KVT]())
        return
    var val = rebind[Scalar[f32]](K[h, d])
    if d < NROT_:
        comptime if NEOX:
            var half = NROT_ // 2
            var j = d if d < half else d - half
            var theta = Float32(Int(pos)) * exp(
                Float32(-2 * j) / Float32(NROT_) * log(freq_base)
            )
            var c = cos(theta)
            var s = sin(theta)
            var x0 = rebind[Scalar[f32]](K[h, j])
            var x1 = rebind[Scalar[f32]](K[h, j + half])
            val = x0 * c - x1 * s if d < half else x0 * s + x1 * c
        else:
            var j = d // 2
            var theta = Float32(Int(pos)) * exp(
                Float32(-2 * j) / Float32(NROT_) * log(freq_base)
            )
            var c = cos(theta)
            var s = sin(theta)
            var x0 = rebind[Scalar[f32]](K[h, 2 * j])
            var x1 = rebind[Scalar[f32]](K[h, 2 * j + 1])
            val = x0 * c - x1 * s if d % 2 == 0 else x0 * s + x1 * c
    Kc.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](val.cast[KVT]())


def amar_bias_add[
    XLayout: TensorLayout, BLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Bias: TileTensor[f32, BLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Bias.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    X[i] = rebind[X.ElementType](rebind[Scalar[f32]](X[i]) + rebind[Scalar[f32]](Bias[i]))


def amar_attn_decode_swa_gated[
    QLayout: TensorLayout, KLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout, NAT: Int, HD_: Int = HD, NQH_: Int = NQH, NKVH_: Int = NKVH, HAS_GATE: Bool = True
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    win: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Gate.flat_rank == 1 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var kvh = h // (NQH_ // NKVH_)
    var T = Int(t_len) + r
    var t_lo = 0
    if Int(win) > 0 and T > Int(win):
        t_lo = T - Int(win)
    var qrow = r * NQH_ + h
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_ // WARP_SIZE]())
    var res = attn_head_span[NAT=NAT, HD_=HD_, NKVH_=NKVH_](Q, Kc, Vc, tab, qs, scores, red, qrow, kvh, t_lo, T, tid, Int(lane_id()), scale, Int(att_i))
    if tid < HD_:
        var inv = 1 / res[1]
        var o = res[2] * inv
        comptime if HAS_GATE:
            var g = rebind[Scalar[f32]](Gate[h])
            O[qrow, tid] = rebind[O.ElementType]((o * (1 / (1 + exp(-g)))).cast[DType.bfloat16]())
        else:
            O[qrow, tid] = rebind[O.ElementType](o.cast[DType.bfloat16]())


def amar_argmax_part[
    NB: Int, XLayout: TensorLayout, VLayout: TensorLayout, ILayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Pv: TileTensor[f32, VLayout, MutAnyOrigin],
    Pi: TileTensor[DType.int32, ILayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Pv.flat_rank == 1 and Pi.flat_rank == 1
    comptime T = 256
    comptime V = 8
    var N = Int(n)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = (b * T + tid) * V
    while i < N:
        var v = X.ptr.load[width=V](i)
        comptime for j in range(V):
            if v[j] > best_v:
                best_v = v[j]
                best_i = Int32(i + j)
        i += NB * T * V
    var vals = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[T]())
    var idxs = stack_allocation[DType.int32, address_space = AddressSpace.SHARED](row_major[T]())
    vals[tid] = rebind[vals.ElementType](best_v)
    idxs[tid] = rebind[idxs.ElementType](best_i)
    barrier()
    var active = T
    comptime for _ in range(8):
        active >>= 1
        if tid < active:
            var v2 = rebind[Scalar[f32]](vals[tid + active])
            var i2 = rebind[Scalar[DType.int32]](idxs[tid + active])
            var v1 = rebind[Scalar[f32]](vals[tid])
            var i1 = rebind[Scalar[DType.int32]](idxs[tid])
            if v2 > v1 or (v2 == v1 and i2 < i1):
                vals[tid] = rebind[vals.ElementType](v2)
                idxs[tid] = rebind[idxs.ElementType](i2)
        barrier()
    if tid == 0:
        Pv[b] = rebind[Pv.ElementType](rebind[Scalar[f32]](vals[0]))
        Pi[b] = rebind[Pi.ElementType](rebind[Scalar[DType.int32]](idxs[0]))


def amar_argmax_final[
    NB: Int, VLayout: TensorLayout, ILayout: TensorLayout, OLayout: TensorLayout
](
    Pv: TileTensor[f32, VLayout, MutAnyOrigin],
    Pi: TileTensor[DType.int32, ILayout, MutAnyOrigin],
    Out: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    Pred: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    wpos: Int32,
    forced: Int32,
):
    comptime assert Pv.flat_rank == 1 and Pi.flat_rank == 1 and Out.flat_rank == 1
    if thread_idx.x != 0:
        return
    var bv = Float32(-3.4e38)
    var bi: Int32 = 0
    comptime for t in range(NB):
        var v = rebind[Scalar[f32]](Pv[t])
        var ix = rebind[Scalar[DType.int32]](Pi[t])
        if v > bv or (v == bv and ix < bi):
            bv = v
            bi = ix
    Pred[Int(wpos)] = rebind[Pred.ElementType](bi)
    Out[Int(wpos)] = rebind[Out.ElementType](forced if forced >= 0 else bi)
