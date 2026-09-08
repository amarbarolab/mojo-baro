from std.gpu import block_dim, block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.math import cos, exp, log, sin, tanh
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from attn import KVT, HD, NQH, NKVH, attn_head_span
from matmul_skinny import dtype, SPLITK

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
    NROT_: Int, XLayout: TensorLayout
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
    var x0 = rebind[Scalar[f32]](X[row, j])
    var x1 = rebind[Scalar[f32]](X[row, j + NROT_ // 2])
    X[row, j] = rebind[X.ElementType](x0 * c - x1 * s)
    X[row, j + NROT_ // 2] = rebind[X.ElementType](x0 * s + x1 * c)


def amar_attn_decode_swa[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, NAT: Int
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    t_len: Int32,
    win: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var kvh = h // (NQH // NKVH)
    var T = Int(t_len) + r
    var t_lo = 0
    if Int(win) > 0 and T > Int(win):
        t_lo = T - Int(win)
    var qrow = r * NQH + h
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var res = attn_head_span[NAT=NAT](Q, Kc, Vc, qs, scores, red, qrow, kvh, t_lo, T, tid, Int(lane_id()), scale, Int(att_i))
    if tid < HD:
        var inv = 1 / res[1]
        O[qrow, tid] = rebind[O.ElementType](res[2] * inv)


def amar_head_gate_mul_cast[
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
    var g = rebind[Scalar[f32]](Gate[i // HD])
    O[i] = rebind[O.ElementType](
        (rebind[Scalar[f32]](X[i]) * (1 / (1 + exp(-g)))).cast[DType.bfloat16]()
    )


def amar_skinny_reduce_gelu_par_bf16[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Gp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Up: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[DType.bfloat16, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Gp.flat_rank == 3 and Up.flat_rank == 3 and C.flat_rank == 2
    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N
    var g: Scalar[dtype] = 0
    var u: Scalar[dtype] = 0
    comptime for s in range(NSPLIT):
        g += rebind[Scalar[dtype]](Gp[s, r, c])
        u += rebind[Scalar[dtype]](Up[s, r, c])
    var gelu = 0.5 * g * (1 + tanh(0.7978845608028654 * g * (1 + 0.044715 * g * g)))
    C[r, c] = rebind[C.ElementType]((gelu * u).cast[DType.bfloat16]())
