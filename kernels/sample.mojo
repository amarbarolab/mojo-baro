from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.math import ceil, exp, log
from std.memory import bitcast
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime SAMP_THREADS = 1024
comptime NBAND = 257
comptime f32 = DType.float32
comptime u32 = DType.uint32
comptime u64 = DType.uint64
comptime i32 = DType.int32
comptime FMAX = Float32(3.4028234663852886e38)
comptime MASS_ONE = Float32(1099511627776.0)
comptime NO_IDX = Int32(2147483647)


@always_inline
def philox4x32(c: SIMD[u32, 4], k0: UInt32, k1: UInt32) -> SIMD[u32, 4]:
    var x = c
    var a = k0
    var b = k1
    comptime for _ in range(10):
        var p0 = UInt64(0xD2511F53) * x[0].cast[u64]()
        var p1 = UInt64(0xCD9E8D57) * x[2].cast[u64]()
        x = SIMD[u32, 4](
            (p1 >> 32).cast[u32]() ^ x[1] ^ a,
            p1.cast[u32](),
            (p0 >> 32).cast[u32]() ^ x[3] ^ b,
            p0.cast[u32](),
        )
        a += 0x9E3779B9
        b += 0xBB67AE85
    return x


@always_inline
def rng_word(seed: UInt64, counter: UInt64, row: Int, stream: Int, i: Int) -> UInt32:
    var c = SIMD[u32, 4](
        counter.cast[u32](),
        (counter >> 32).cast[u32](),
        UInt32(row),
        (UInt32(stream) << 28) | UInt32(i >> 2),
    )
    var w = philox4x32(c, seed.cast[u32](), (seed >> 32).cast[u32]())
    return w[i & 3]


@always_inline
def unif(w: UInt32) -> Float32:
    return ((w >> 8).cast[f32]() + 0.5) * Float32(5.9604644775390625e-08)


@always_inline
def gumbel(w: UInt32) -> Float32:
    return -log(-log(unif(w)))


@always_inline
def okey(v: Float32) -> UInt32:
    var b = bitcast[u32, 1](v)
    if (b >> 31) != 0:
        return ~b
    return b | 0x80000000


@always_inline
def okey_inv(k: UInt32) -> Float32:
    if (k >> 31) != 0:
        return bitcast[f32, 1](k & 0x7FFFFFFF)
    return bitcast[f32, 1](~k)


@always_inline
def is_valid(v: Float32) -> Bool:
    return v >= -FMAX and v <= FMAX


@always_inline
def band_of(v: Float32, lmax: Float32) -> Int:
    var t = (lmax - v) * 8.0
    if t < 256.0:
        return Int(t)
    return NBAND - 1


@always_inline
def fixed_mass(v: Float32, lmax: Float32) -> UInt64:
    return (exp(v - lmax) * MASS_ONE).cast[u64]()


@always_inline
def in_cut(k: UInt32, i: Int, ck: UInt32, ci: Int) -> Bool:
    return k > ck or (k == ck and i <= ci)


@always_inline
def bsum_u64[
    RL: TensorLayout
](
    mut red: TileTensor[u64, RL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: UInt64,
) -> UInt64:
    comptime assert red.flat_rank == 1
    red[tid] = rebind[red.ElementType](v)
    barrier()
    comptime for s in range(10):
        comptime h = 512 >> s
        if tid < h:
            red[tid] = red[tid] + red[tid + h]
        barrier()
    var r = rebind[Scalar[u64]](red[0])
    barrier()
    return r


@always_inline
def bsum_f32[
    RL: TensorLayout
](
    mut red: TileTensor[f32, RL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: Float32,
) -> Float32:
    comptime assert red.flat_rank == 1
    red[tid] = rebind[red.ElementType](v)
    barrier()
    comptime for s in range(10):
        comptime h = 512 >> s
        if tid < h:
            red[tid] = red[tid] + red[tid + h]
        barrier()
    var r = rebind[Scalar[f32]](red[0])
    barrier()
    return r


@always_inline
def bmax_f32[
    RL: TensorLayout
](
    mut red: TileTensor[f32, RL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: Float32,
) -> Float32:
    comptime assert red.flat_rank == 1
    red[tid] = rebind[red.ElementType](v)
    barrier()
    comptime for s in range(10):
        comptime h = 512 >> s
        if tid < h:
            var a = rebind[Scalar[f32]](red[tid])
            var b = rebind[Scalar[f32]](red[tid + h])
            if b > a:
                red[tid] = rebind[red.ElementType](b)
        barrier()
    var r = rebind[Scalar[f32]](red[0])
    barrier()
    return r


@always_inline
def bargmax[
    FL: TensorLayout, IL: TensorLayout
](
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redi: TileTensor[i32, IL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: Float32,
    ix: Int32,
) -> Int32:
    comptime assert redf.flat_rank == 1 and redi.flat_rank == 1
    redf[tid] = rebind[redf.ElementType](v)
    redi[tid] = rebind[redi.ElementType](ix)
    barrier()
    comptime for s in range(10):
        comptime h = 512 >> s
        if tid < h:
            var a = rebind[Scalar[f32]](redf[tid])
            var b = rebind[Scalar[f32]](redf[tid + h])
            var ia = rebind[Scalar[i32]](redi[tid])
            var ib = rebind[Scalar[i32]](redi[tid + h])
            if b > a or (b == a and ib < ia):
                redf[tid] = rebind[redf.ElementType](b)
                redi[tid] = rebind[redi.ElementType](ib)
        barrier()
    var r = rebind[Scalar[i32]](redi[0])
    barrier()
    return r


@always_inline
def greedy_tok[
    XL: TensorLayout, FL: TensorLayout, IL: TensorLayout
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redi: TileTensor[i32, IL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    row: Int,
    N: Int,
    tid: Int,
) -> Int32:
    comptime assert X.flat_rank == 2
    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > best_v:
            best_v = v
            best_i = Int32(i)
        i += SAMP_THREADS
    return bargmax(redf, redi, tid, best_v, best_i)


def refine[
    XL: TensorLayout, HL: TensorLayout, SL: TensorLayout
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut hist: TileTensor[u64, HL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut st: TileTensor[u64, SL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    row: Int,
    N: Int,
    tid: Int,
    lmax: Float32,
    b: Int,
    ck: UInt32,
    ci: Int,
    use_mass: Bool,
    W: UInt64,
) -> Tuple[UInt32, Int]:
    comptime assert X.flat_rank == 2 and hist.flat_rank == 1 and st.flat_rank == 1
    var prefix: UInt32 = 0
    var pmask: UInt32 = 0
    var need = W
    var tiew: UInt64 = 0
    for d in range(4):
        var sh = UInt32(24 - 8 * d)
        if tid < 256:
            hist[tid] = 0
        barrier()
        var i = tid
        while i < N:
            var v = rebind[Scalar[f32]](X[row, i])
            if is_valid(v) and band_of(v, lmax) == b:
                var k = okey(v)
                if (k & pmask) == prefix and in_cut(k, i, ck, ci):
                    var w: UInt64 = 1
                    if use_mass:
                        w = fixed_mass(v, lmax)
                    _ = Atomic.fetch_add(hist.ptr.unsafe_offset(Int((k >> sh) & 255)), w)
            i += SAMP_THREADS
        barrier()
        if tid == 0:
            var acc: UInt64 = 0
            var sel = 0
            for bb in range(255, -1, -1):
                var h = rebind[Scalar[u64]](hist[bb])
                if acc + h >= need:
                    sel = bb
                    break
                acc += h
            st[0] = rebind[st.ElementType]((prefix | (UInt32(sel) << sh)).cast[u64]())
            st[1] = rebind[st.ElementType](need - acc)
            st[2] = hist[sel]
        barrier()
        prefix = rebind[Scalar[u64]](st[0]).cast[u32]()
        need = rebind[Scalar[u64]](st[1])
        tiew = rebind[Scalar[u64]](st[2])
        pmask |= UInt32(255) << sh
    barrier()

    var w0: UInt64 = 1
    if use_mass:
        w0 = fixed_mass(okey_inv(prefix), lmax)
        if w0 == 0:
            w0 = 1
    var ties = tiew // w0
    var r = (need + w0 - 1) // w0
    if r >= ties:
        if prefix == ck:
            return (prefix, ci)
        return (prefix, N)

    var nbits = 0
    while (1 << nbits) < N:
        nbits += 1
    var ipre = 0
    var imask = 0
    var rn = r
    for d in range((nbits + 7) // 8 - 1, -1, -1):
        var sh = 8 * d
        if tid < 256:
            hist[tid] = 0
        barrier()
        var i = tid
        while i < N:
            var v = rebind[Scalar[f32]](X[row, i])
            if is_valid(v) and band_of(v, lmax) == b:
                var k = okey(v)
                if k == prefix and in_cut(k, i, ck, ci) and (i & imask) == ipre:
                    _ = Atomic.fetch_add(hist.ptr.unsafe_offset((i >> sh) & 255), UInt64(1))
            i += SAMP_THREADS
        barrier()
        if tid == 0:
            var acc: UInt64 = 0
            var sel = 255
            for bb in range(256):
                var h = rebind[Scalar[u64]](hist[bb])
                if acc + h >= rn:
                    sel = bb
                    break
                acc += h
            st[0] = rebind[st.ElementType](UInt64(ipre | (sel << sh)))
            st[1] = rebind[st.ElementType](rn - acc)
        barrier()
        ipre = Int(rebind[Scalar[u64]](st[0]))
        rn = rebind[Scalar[u64]](st[1])
        imask |= 255 << sh
    barrier()
    return (prefix, ipre)


def sample_cut[
    XL: TensorLayout, HL: TensorLayout, BL: TensorLayout, SL: TensorLayout,
    UL: TensorLayout, FL: TensorLayout
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut hist: TileTensor[u64, HL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut hc: TileTensor[u64, BL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut hm: TileTensor[u64, BL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut st: TileTensor[u64, SL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redu: TileTensor[u64, UL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    row: Int,
    N: Int,
    tid: Int,
    top_k: Int32,
    top_p: Float32,
) -> Tuple[Float32, UInt64, UInt32, Int]:
    comptime assert X.flat_rank == 2 and hc.flat_rank == 1 and hm.flat_rank == 1 and st.flat_rank == 1
    var lm = -FMAX
    var cnt: UInt64 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if is_valid(v):
            if v > lm:
                lm = v
            cnt += 1
        i += SAMP_THREADS
    var lmax = bmax_f32(redf, tid, lm)
    var nvalid = bsum_u64(redu, tid, cnt)
    var ck: UInt32 = 0
    var ci = N
    if nvalid == 0:
        return (lmax, nvalid, ck, ci)

    var k_on = top_k > 0 and UInt64(Int(top_k)) < nvalid
    var p_on = top_p < 1.0
    if not k_on and not p_on:
        return (lmax, nvalid, ck, ci)

    if tid < NBAND:
        hc[tid] = 0
        hm[tid] = 0
    barrier()
    i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if is_valid(v):
            var bi = band_of(v, lmax)
            _ = Atomic.fetch_add(hc.ptr.unsafe_offset(bi), UInt64(1))
            _ = Atomic.fetch_add(hm.ptr.unsafe_offset(bi), fixed_mass(v, lmax))
        i += SAMP_THREADS
    barrier()

    var bk = NBAND
    if k_on:
        if tid == 0:
            var kk = UInt64(Int(top_k))
            var cum: UInt64 = 0
            var sel = NBAND - 1
            for bb in range(NBAND):
                var h = rebind[Scalar[u64]](hc[bb])
                if cum + h >= kk:
                    sel = bb
                    break
                cum += h
            st[0] = rebind[st.ElementType](UInt64(sel))
            st[1] = rebind[st.ElementType](kk - cum)
        barrier()
        bk = Int(rebind[Scalar[u64]](st[0]))
        var needk = rebind[Scalar[u64]](st[1])
        barrier()
        var cut = refine(X, hist, st, row, N, tid, lmax, bk, ck, ci, False, needk)
        ck = cut[0]
        ci = cut[1]

    if p_on:
        var z: UInt64 = 0
        if k_on:
            i = tid
            while i < N:
                var v = rebind[Scalar[f32]](X[row, i])
                if is_valid(v) and in_cut(okey(v), i, ck, ci):
                    z += fixed_mass(v, lmax)
                i += SAMP_THREADS
            z = bsum_u64(redu, tid, z)
        else:
            var hv: UInt64 = 0
            if tid < NBAND:
                hv = rebind[Scalar[u64]](hm[tid])
            z = bsum_u64(redu, tid, hv)
        var W = ceil(Float64(top_p) * z.cast[DType.float64]()).cast[u64]()
        if W < 1:
            W = 1
        if W > z:
            W = z
        if tid == 0:
            var cum: UInt64 = 0
            var sel = bk
            var lim = bk if bk < NBAND else NBAND
            for bb in range(lim):
                var h = rebind[Scalar[u64]](hm[bb])
                if cum + h >= W:
                    sel = bb
                    break
                cum += h
            st[0] = rebind[st.ElementType](UInt64(sel))
            st[1] = rebind[st.ElementType](W - cum)
        barrier()
        var bp = Int(rebind[Scalar[u64]](st[0]))
        var needp = rebind[Scalar[u64]](st[1])
        barrier()
        var cut = refine(X, hist, st, row, N, tid, lmax, bp, ck, ci, True, needp)
        ck = cut[0]
        ci = cut[1]
    return (lmax, nvalid, ck, ci)


def amar_sample_row[
    XLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[i32, OLayout, MutAnyOrigin],
    Prob: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    seed: UInt64,
    counter: UInt64,
):
    comptime assert X.flat_rank == 2 and Out.flat_rank == 1 and Prob.flat_rank == 1
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var hist = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[256]())
    var hc = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var hm = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var st = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[4]())
    var redu = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    if temperature <= 0:
        var g = greedy_tok(X, redf, redi, row, N, tid)
        if tid == 0:
            Out[row] = rebind[Out.ElementType](g)
            Prob[row] = rebind[Prob.ElementType](Float32(1))
        return

    var cut = sample_cut(X, hist, hc, hm, st, redu, redf, row, N, tid, top_k, top_p)
    var lmax = cut[0]
    var ck = cut[2]
    var ci = cut[3]
    if cut[1] == 0:
        if tid == 0:
            Out[row] = rebind[Out.ElementType](Int32(-1))
            Prob[row] = rebind[Prob.ElementType](Float32(0))
        return

    var min_on = min_p > 0
    var bs = -FMAX
    var bi = NO_IDX
    var zt: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if is_valid(v) and in_cut(okey(v), i, ck, ci) and (not min_on or exp(v - lmax) >= min_p):
            var e = (v - lmax) / temperature
            var s = e + gumbel(rng_word(seed, counter, row, 0, i))
            if s > bs:
                bs = s
                bi = Int32(i)
            zt += exp(e)
        i += SAMP_THREADS
    var tok = bargmax(redf, redi, tid, bs, bi)
    var z = bsum_f32(redf, tid, zt)
    if tid == 0:
        var lt = rebind[Scalar[f32]](X[row, Int(tok)])
        Out[row] = rebind[Out.ElementType](tok)
        Prob[row] = rebind[Prob.ElementType](exp((lt - lmax) / temperature) / z)


def amar_sample_probs[
    XLayout: TensorLayout, PLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    P: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
):
    comptime assert X.flat_rank == 2 and P.flat_rank == 2
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var hist = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[256]())
    var hc = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var hm = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var st = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[4]())
    var redu = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    if temperature <= 0:
        var g = Int(greedy_tok(X, redf, redi, row, N, tid))
        var i = tid
        while i < N:
            P[row, i] = rebind[P.ElementType](Float32(1) if i == g else Float32(0))
            i += SAMP_THREADS
        return

    var cut = sample_cut(X, hist, hc, hm, st, redu, redf, row, N, tid, top_k, top_p)
    var lmax = cut[0]
    var ck = cut[2]
    var ci = cut[3]
    var none = cut[1] == 0
    var min_on = min_p > 0
    var zt: Float32 = 0
    var i = tid
    while i < N and not none:
        var v = rebind[Scalar[f32]](X[row, i])
        if is_valid(v) and in_cut(okey(v), i, ck, ci) and (not min_on or exp(v - lmax) >= min_p):
            zt += exp((v - lmax) / temperature)
        i += SAMP_THREADS
    var z = bsum_f32(redf, tid, zt)
    i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        var p: Float32 = 0
        if not none and is_valid(v) and in_cut(okey(v), i, ck, ci) and (not min_on or exp(v - lmax) >= min_p):
            p = exp((v - lmax) / temperature) / z
        P[row, i] = rebind[P.ElementType](p)
        i += SAMP_THREADS


def amar_spec_accept[
    PLayout: TensorLayout, TLayout: TensorLayout
](
    Pt: TileTensor[f32, PLayout, MutAnyOrigin],
    Pd: TileTensor[f32, PLayout, MutAnyOrigin],
    Dtok: TileTensor[i32, TLayout, MutAnyOrigin],
    Out: TileTensor[i32, TLayout, MutAnyOrigin],
    Acc: TileTensor[i32, TLayout, MutAnyOrigin],
    n: Int32,
    seed: UInt64,
    counter: UInt64,
):
    comptime assert Pt.flat_rank == 2 and Pd.flat_rank == 2 and Dtok.flat_rank == 1
    comptime assert Out.flat_rank == 1 and Acc.flat_rank == 1
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    var x = Int(rebind[Scalar[i32]](Dtok[row]))
    var accept = False
    if x >= 0 and x < N:
        var pt = rebind[Scalar[f32]](Pt[row, x])
        var pd = rebind[Scalar[f32]](Pd[row, x])
        accept = unif(rng_word(seed, counter, row, 1, 0)) * pd < pt
    if accept:
        if tid == 0:
            Out[row] = rebind[Out.ElementType](Int32(x))
            Acc[row] = rebind[Acc.ElementType](Int32(1))
        return

    var bs = -FMAX
    var bi = NO_IDX
    var i = tid
    while i < N:
        var r = rebind[Scalar[f32]](Pt[row, i]) - rebind[Scalar[f32]](Pd[row, i])
        if r > 0:
            var s = log(r) + gumbel(rng_word(seed, counter, row, 2, i))
            if s > bs:
                bs = s
                bi = Int32(i)
        i += SAMP_THREADS
    var tok = bargmax(redf, redi, tid, bs, bi)
    if tok == NO_IDX:
        bs = -FMAX
        bi = NO_IDX
        i = tid
        while i < N:
            var r = rebind[Scalar[f32]](Pt[row, i])
            if r > 0:
                var s = log(r) + gumbel(rng_word(seed, counter, row, 3, i))
                if s > bs:
                    bs = s
                    bi = Int32(i)
            i += SAMP_THREADS
        tok = bargmax(redf, redi, tid, bs, bi)
    if tid == 0:
        Out[row] = rebind[Out.ElementType](Int32(-1) if tok == NO_IDX else tok)
        Acc[row] = rebind[Acc.ElementType](Int32(0))
