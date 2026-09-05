from std.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from layout import TileTensor, TensorLayout

from matmul import dtype
from matmul_skinny import ROW_WAVES

comptime B3_BLOCK = 128
comptime B3_BYTES = 26
comptime TQ_BLOCK = 256
comptime TQ1_BYTES = 52
comptime TQ2_BYTES = 64
comptime TV = 16


@always_inline
def tq1_trits[W: Int](q: SIMD[DType.uint8, W]) -> SIMD[dtype, W]:
    return ((q.cast[DType.uint16]() * UInt16(3)) >> UInt16(8)).cast[dtype]() - 1


@always_inline
def tq2_trits[W: Int](q: SIMD[DType.uint8, W], shift: Int) -> SIMD[dtype, W]:
    return ((q >> UInt8(shift)) & UInt8(3)).cast[dtype]() - 1


def amar_matmul_skinny_q2b3row[
    MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Av = A.vectorize[1, TV]()
    var acc = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))

    var nb = K // B3_BLOCK
    var b = lane
    while b < nb:
        var qs = InlineArray[UInt8, B3_BYTES](uninitialized=True)
        comptime for i in range(B3_BYTES):
            qs[i] = rebind[Scalar[DType.uint8]](Q[row, b * B3_BYTES + i])
        var d = rebind[Scalar[DType.float16]](S[row, b]).cast[dtype]()
        var blk = InlineArray[SIMD[dtype, TV], MR](fill=SIMD[dtype, TV](0))

        comptime for c in range(4):
            comptime for h in range(2):
                var w = SIMD[dtype, TV](0)
                comptime for i in range(TV):
                    comptime t = h * TV + i
                    comptime byte = 6 * c + t // 5 if t < 30 else 24 + (c >> 1)
                    comptime digit = t % 5 if t < 30 else 2 * (c & 1) + (t - 30)
                    comptime p = 3 ** digit
                    w[i] = ((qs[byte] // UInt8(p)) % UInt8(3)).cast[dtype]() - 1
                var e = b * B3_BLOCK + c * 32 + h * TV
                comptime for r in range(MR):
                    if r < M:
                        var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                        blk[r] += w * a

        comptime for r in range(MR):
            if r < M:
                acc[r] += blk[r].reduce_add() * d
        b += WARP_SIZE

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


def amar_matmul_skinny_tq1row[
    MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Av = A.vectorize[1, TV]()
    var Qv = Q.vectorize[1, 4]()
    var acc = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))

    var nb = K // TQ_BLOCK
    var b = lane
    while b < nb:
        var raw = InlineArray[SIMD[DType.uint8, 4], TQ1_BYTES // 4](uninitialized=True)
        comptime for i in range(TQ1_BYTES // 4):
            raw[i] = rebind[SIMD[DType.uint8, 4]](Qv[row, (b * TQ1_BYTES) // 4 + i])
        var qlo = raw[0].join(raw[1]).join(raw[2].join(raw[3]))
        var qhi = raw[4].join(raw[5]).join(raw[6].join(raw[7]))
        var q2 = raw[8].join(raw[9]).join(raw[10].join(raw[11]))
        var qh = raw[12]
        var d = rebind[Scalar[DType.float16]](S[row, b]).cast[dtype]()
        var blk = InlineArray[SIMD[dtype, TV], MR](fill=SIMD[dtype, TV](0))
        var e0 = b * TQ_BLOCK

        comptime for n_ in range(5):
            comptime p = 3 ** n_
            var wlo = tq1_trits[TV](qlo * UInt8(p))
            var whi = tq1_trits[TV](qhi * UInt8(p))
            var e = e0 + n_ * 32
            comptime for r in range(MR):
                if r < M:
                    var alo = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                    var ahi = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV + 1]).cast[dtype]()
                    blk[r] += wlo * alo + whi * ahi

        comptime for n_ in range(5):
            comptime p = 3 ** n_
            var w = tq1_trits[TV](q2 * UInt8(p))
            var e = e0 + 160 + n_ * TV
            comptime for r in range(MR):
                if r < M:
                    var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                    blk[r] += w * a

        var wq = SIMD[dtype, TV](0)
        comptime for n_ in range(4):
            comptime p = 3 ** n_
            var v = tq1_trits[4](qh * UInt8(p))
            comptime for j in range(4):
                wq[n_ * 4 + j] = v[j]
        comptime for r in range(MR):
            if r < M:
                var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, (e0 + 240) // TV]).cast[dtype]()
                blk[r] += wq * a

        comptime for r in range(MR):
            if r < M:
                acc[r] += blk[r].reduce_add() * d
        b += WARP_SIZE

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


def amar_matmul_skinny_tq2row[
    MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Av = A.vectorize[1, TV]()
    var Qv = Q.vectorize[1, TV]()
    var acc = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))

    var nb = K // TQ_BLOCK
    var b = lane
    while b < nb:
        var raw = InlineArray[SIMD[DType.uint8, TV], TQ2_BYTES // TV](uninitialized=True)
        comptime for i in range(TQ2_BYTES // TV):
            raw[i] = rebind[SIMD[DType.uint8, TV]](Qv[row, (b * TQ2_BYTES) // TV + i])
        var d = rebind[Scalar[DType.float16]](S[row, b]).cast[dtype]()
        var blk = InlineArray[SIMD[dtype, TV], MR](fill=SIMD[dtype, TV](0))
        var e0 = b * TQ_BLOCK

        comptime for jj in range(2):
            comptime for l in range(4):
                comptime for h in range(2):
                    var w = tq2_trits[TV](raw[jj * 2 + h], 2 * l)
                    var e = e0 + jj * 128 + l * 32 + h * TV
                    comptime for r in range(MR):
                        if r < M:
                            var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                            blk[r] += w * a

        comptime for r in range(MR):
            if r < M:
                acc[r] += blk[r].reduce_add() * d
        b += WARP_SIZE

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)
