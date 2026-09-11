from std.bit import log2_floor
from std.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives import warp
from std.math import exp, fma
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime f32 = DType.float32
comptime KVPAGE = 128
comptime KVPSH = 7
comptime DTHREADS = 256
comptime DWAVES = DTHREADS // WARP_SIZE
comptime NEG = Float32(-3.4e38)


@always_inline
def dkv_off[HD: Int, NKVH: Int, NAT: Int](t: Int, att_i: Int, kvh: Int) -> Int:
    return (((t >> KVPSH) * NAT + att_i) * NKVH + kvh) * (KVPAGE * HD) + (t & (KVPAGE - 1)) * HD


@always_inline
def dspan[HD: Int, NLD: Int]() -> Int:
    return NLD * (WARP_SIZE // (HD // 8))


@always_inline
def dattn_span[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int,
    QLayout: TensorLayout, KLayout: TensorLayout,
    QsL: TensorLayout, ScL: TensorLayout, RdL: TensorLayout,
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    mut qs: TileTensor[f32, QsL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut scores: TileTensor[f32, ScL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut red: TileTensor[f32, RdL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    qrow: Int, kvh: Int, t_lo: Int, t_hi: Int, tid: Int, lane: Int, scale: Float32, att_i: Int,
) -> SIMD[f32, 4]:
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1
    comptime assert qs.flat_rank == 1 and scores.flat_rank == 1 and red.flat_rank == 1
    var wave = tid // WARP_SIZE
    if tid < HD:
        qs[tid] = rebind[qs.ElementType](Q[qrow, tid])
    barrier()
    var qv = qs.vectorize[8]()
    var kp = Kc.ptr
    var vp = Vc.ptr
    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var o = Float32(0)
    var t0 = t_lo
    while t0 < t_hi:
        barrier()
        var s = Float32(-3.4e38)
        if tid < HD:
            var t = t0 + tid
            if t < t_hi:
                var kb = dkv_off[HD, NKVH, NAT](t, att_i, kvh)
                var Kr = TileTensor(kp.unsafe_offset(kb), row_major[HD]()).vectorize[8]()
                var acc: Float32 = 0
                for d8 in range(HD // 8):
                    var k8 = rebind[SIMD[KVT, 8]](Kr[d8]).cast[f32]()
                    var q8 = rebind[SIMD[f32, 8]](qv[d8])
                    comptime for j in range(8):
                        acc += q8[j] * k8[j]
                s = acc * scale
            scores[tid] = rebind[scores.ElementType](s)
            var wmax = warp.max(s)
            if lane == 0:
                red[wave] = rebind[red.ElementType](wmax)
        barrier()
        var cmax = Float32(-3.4e38)
        comptime for w in range(HD // WARP_SIZE):
            var sc = rebind[Scalar[f32]](red[w])
            if sc > cmax:
                cmax = sc
        var m_new = max(m_run, cmax)
        var alpha = exp(m_run - m_new)
        barrier()
        if tid < HD:
            var e = exp(s - m_new)
            scores[tid] = rebind[scores.ElementType](e)
            var wsum = warp.sum(e)
            if lane == 0:
                red[wave] = rebind[red.ElementType](wsum)
        barrier()
        var csum = Float32(0)
        comptime for w in range(HD // WARP_SIZE):
            csum += rebind[Scalar[f32]](red[w])
        l_run = l_run * alpha + csum
        m_run = m_new
        o = o * alpha
        if tid < HD:
            var n = t_hi - t0
            if n > HD:
                n = HD
            var tt = 0
            while tt + 8 <= n:
                var v = InlineArray[Float32, 8](uninitialized=True)
                var sc = InlineArray[Float32, 8](uninitialized=True)
                comptime for j in range(8):
                    v[j] = vp[unsafe_offset=dkv_off[HD, NKVH, NAT](t0 + tt + j, att_i, kvh) + tid].cast[f32]()
                    sc[j] = rebind[Scalar[f32]](scores[tt + j])
                comptime for j in range(8):
                    o += sc[j] * v[j]
                tt += 8
            while tt < n:
                o += rebind[Scalar[f32]](scores[tt]) * vp[unsafe_offset=dkv_off[HD, NKVH, NAT](t0 + tt, att_i, kvh) + tid].cast[f32]()
                tt += 1
        t0 += HD
    return SIMD[f32, 4](m_run, l_run, o, 0)


def amar_dattn_exact[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int,
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout,
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    t_len: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var kvh = h // (NQH // NKVH)
    var T = Int(t_len) + r
    var qrow = r * NQH + h
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var O_ = O
    var res = dattn_span[HD, NQH, NKVH, KVT, NAT](Q, Kc, Vc, qs, scores, red, qrow, kvh, 0, T, tid, Int(lane_id()), scale, Int(att_i))
    if tid < HD:
        var inv = 1 / res[1]
        O_[qrow, tid] = rebind[O_.ElementType](res[2] * inv)


def amar_dattn_split[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int, NLD: Int,
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout,
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    t_len: Int32,
    nsplit: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1 and O.flat_rank == 2 and Pg.flat_rank == 1
    comptime G = NQH // NKVH
    comptime LPR = HD // 8
    comptime RPL = WARP_SIZE // LPR
    comptime SPAN = NLD * RPL
    comptime DUP = LPR // NLD
    comptime LOG2_LPR = log2_floor(LPR)
    comptime PSTR = HD + 2
    comptime N8 = G * HD // 8
    comptime assert NQH == G * NKVH and LPR * RPL == WARP_SIZE and DUP * NLD == LPR and N8 <= DTHREADS
    var kvh = Int(block_idx.x)
    var sp = Int(block_idx.y)
    var ns = Int(nsplit)
    var T = Int(t_len)
    var ai = Int(att_i)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var lr = lane // LPR
    var lc = (lane % LPR) * 8
    var jtok = (lane % LPR) // DUP
    var NS = (T + SPAN - 1) // SPAN
    var s_lo = sp * NS // ns
    var s_hi = (sp + 1) * NS // ns
    var q = InlineArray[SIMD[f32, 8], G](uninitialized=True)
    var o = InlineArray[SIMD[f32, 8], G](fill=SIMD[f32, 8](0))
    var m = InlineArray[Float32, G](fill=NEG)
    var l = InlineArray[Float32, G](fill=Float32(0))
    comptime for g in range(G):
        q[g] = Q.ptr.unsafe_load[width=8]((kvh * G + g) * HD + lc)
    var si = s_lo + wave
    while si < s_hi:
        var t0 = si * SPAN
        var kr = InlineArray[SIMD[KVT, 8], NLD](uninitialized=True)
        var vr = InlineArray[SIMD[KVT, 8], NLD](uninitialized=True)
        comptime for i in range(NLD):
            var t = t0 + i * RPL + lr
            if t < T:
                var off = dkv_off[HD, NKVH, NAT](t, ai, kvh) + lc
                kr[i] = Kc.ptr.unsafe_load[width=8](off)
                vr[i] = Vc.ptr.unsafe_load[width=8](off)
            else:
                kr[i] = SIMD[KVT, 8](0)
                vr[i] = SIMD[KVT, 8](0)
        var part = InlineArray[Float32, NLD * G](uninitialized=True)
        comptime for i in range(NLD):
            var k8 = kr[i].cast[f32]()
            comptime for g in range(G):
                var acc = Float32(0)
                comptime for e in range(8):
                    acc = fma(q[g][e], k8[e], acc)
                part[g * NLD + i] = acc
        var v32 = InlineArray[SIMD[f32, 8], NLD](uninitialized=True)
        comptime for i in range(NLD):
            v32[i] = vr[i].cast[f32]()
        var t_me = t0 + jtok * RPL + lr
        comptime for g in range(G):
            comptime for r in range(LOG2_LPR):
                comptime off = LPR >> (r + 1)
                comptime n = NLD >> r
                comptime if n > 1:
                    comptime half = n // 2
                    var hi = (lane & off) != 0
                    comptime for j in range(half):
                        var a = part[g * NLD + j]
                        var b = part[g * NLD + j + half]
                        var send = a if hi else b
                        var keep = b if hi else a
                        part[g * NLD + j] = keep + warp.shuffle_xor(send, UInt32(off))
                else:
                    part[g * NLD] += warp.shuffle_xor(part[g * NLD], UInt32(off))
            var s = NEG
            if t_me < T:
                s = part[g * NLD] * scale
            var tmax = warp.max(s)
            var m_new = max(m[g], tmax)
            var alpha = exp(m[g] - m_new)
            var p = exp(s - m_new)
            var psum = p if (lane % DUP) == 0 else Float32(0)
            l[g] = l[g] * alpha + warp.sum(psum)
            o[g] = o[g] * alpha
            comptime for i in range(NLD):
                var pb = warp.shuffle_idx(p, UInt32(lr * LPR + i * DUP))
                o[g] = fma(v32[i], SIMD[f32, 8](pb), o[g])
            m[g] = m_new
        si += DWAVES
    comptime for g in range(G):
        comptime for r in range(log2_floor(RPL)):
            comptime off = LPR << r
            comptime for e in range(8):
                o[g][e] += warp.shuffle_xor(o[g][e], UInt32(off))
    var wm = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[DWAVES * G]())
    var wl = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[DWAVES * G]())
    var acc = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G * HD]())
    if lane == 0:
        comptime for g in range(G):
            wm[wave * G + g] = rebind[wm.ElementType](m[g])
            wl[wave * G + g] = rebind[wl.ElementType](l[g])
    barrier()
    var mb = InlineArray[Float32, G](uninitialized=True)
    var lb = InlineArray[Float32, G](uninitialized=True)
    comptime for g in range(G):
        var mm = NEG
        comptime for w in range(DWAVES):
            mm = max(mm, rebind[Scalar[f32]](wm[w * G + g]))
        var ll = Float32(0)
        comptime for w in range(DWAVES):
            ll += exp(rebind[Scalar[f32]](wm[w * G + g]) - mm) * rebind[Scalar[f32]](wl[w * G + g])
        mb[g] = mm
        lb[g] = ll
    var accv = acc.vectorize[8]()
    comptime for w in range(DWAVES):
        if wave == w and lr == 0:
            comptime for g in range(G):
                var wgt = exp(m[g] - mb[g])
                var idx = (g * HD + lc) // 8
                comptime if w == 0:
                    accv[idx] = rebind[accv.ElementType](o[g] * wgt)
                else:
                    accv[idx] = rebind[accv.ElementType](rebind[SIMD[f32, 8]](accv[idx]) + o[g] * wgt)
        barrier()
    if tid < N8:
        var g = tid // (HD // 8)
        var c = (tid % (HD // 8)) * 8
        var v = rebind[SIMD[f32, 8]](accv[tid])
        var h = kvh * G + g
        if ns == 1:
            var inv = 1 / lb[g]
            var O_ = O
            comptime for e in range(8):
                O_[h, c + e] = rebind[O_.ElementType](v[e] * inv)
        else:
            var pp = Pg.ptr.unsafe_offset((h * ns + sp) * PSTR)
            if c == 0:
                pp[unsafe_offset=0] = mb[g]
                pp[unsafe_offset=1] = lb[g]
            comptime for e in range(8):
                pp[unsafe_offset=2 + c + e] = v[e]


def amar_dattn_combine[
    HD: Int, PLayout: TensorLayout, OLayout: TensorLayout,
](
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    nsplit: Int32,
):
    comptime assert Pg.flat_rank == 1 and O.flat_rank == 2
    comptime PSTR = HD + 2
    var h = Int(block_idx.x)
    var d = Int(thread_idx.x)
    var ns = Int(nsplit)
    var pp = Pg.ptr.unsafe_offset(h * ns * PSTR)
    var mmax = NEG
    for sp in range(ns):
        mmax = max(mmax, pp[unsafe_offset=sp * PSTR])
    var l = Float32(0)
    var o = Float32(0)
    for sp in range(ns):
        var wgt = exp(pp[unsafe_offset=sp * PSTR] - mmax)
        l += wgt * pp[unsafe_offset=sp * PSTR + 1]
        o += wgt * pp[unsafe_offset=sp * PSTR + 2 + d]
    var O_ = O
    O_[h, d] = rebind[O_.ElementType](o * (1 / l))
