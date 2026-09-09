"""ABLATION RECEIPT (bench/prefill-protocol.md R5-abl), not an arm: prefill GEMM on the ffn shape (N=12288, K=4096, Q4_0 from the q4
pack), bench/prefill-protocol.md kernel table: bf16-WMMA prefill kernel vs
the decode-path wave-per-row kernel looped in 8-row windows (its register
ceiling), n = 16 .. 1024 activation rows, plus (R5) the LDS-pipelined
schedule instantiated for bf16 (bf16-lds) and for int8 activations (mmq,
with the q8 quantiser timed separately as `quant`). NBUF weight copies
rotate so the weight stream is cold (8 x 28 MB > 96 MB Infinity Cache).
Correctness gates before any timing is printed: WMMA vs row loop rel <
1e-4, bf16-lds vs row loop rel < 1e-4, mmq vs bf16-lds relative Frobenius
< 1e-2 and max|a-b|/rms(b) < 5e-2.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from matmul_skinny import amar_matmul_skinny_q4rowb, amar_skinny_reduce, SM, ROW_WAVES, ROW_THREADS
from matmul_prefill import amar_matmul_prefill_q4, PF_THREADS
from matmul_mmq import amar_quant_q8, amar_matmul_lds_q4, MMQ_THREADS, QT_THREADS, ABL

comptime K = 4096
comptime N = 12288
comptime MMAX = 1024
comptime NBUF = 8
comptime ITERS = 16
comptime REPEATS = 5
comptime Q4BYTES = N * K // 2 + N * (K // 32) * 2

comptime a_layout = row_major[MMAX, K]()
comptime a8_layout = row_major[SM, K]()
comptime c_layout = row_major[MMAX, N]()
comptime c8_layout = row_major[SM, N]()
comptime q4_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()
comptime p_layout = row_major[1, SM, N]()
comptime d_layout = row_major[K // 32, MMAX]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime i8 = DType.int8
comptime i32 = DType.int32


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int, skip: Int
) raises:
    with open(path, "r") as f:
        _ = f.seek(skip)
        var data = f.read_bytes(size)
        if len(data) < size:
            raise Error("size mismatch for " + path)
        for i in range(size):
            dst[unsafe_offset=i] = data[i]


def pack_offset(name: String) raises -> Int:
    with open(".work/engine-pack-q4/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if parts[0] == name:
                if String(parts[1]) != "q4":
                    raise Error("not q4: " + name)
                return Int(parts[2])
    raise Error("missing " + name)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int((st >> 40) & 0xFFFF)) / 32768.0 - 1.0


def view[
    dt: DType, LT: TensorLayout
](ctx: DeviceContext, buf: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[dt, LT, MutAnyOrigin]:
    var b = DeviceBuffer[dt](ctx, buf.unsafe_ptr().unsafe_offset(o).unsafe_bitcast[Scalar[dt]](), n, owning=False)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[dt, LT, MutAnyOrigin]](t)


def view_f32[
    LT: TensorLayout
](ctx: DeviceContext, buf: DeviceBuffer[f32], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, buf.unsafe_ptr().unsafe_offset(o), n, owning=False)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](t)


def view_bf16[
    LT: TensorLayout
](ctx: DeviceContext, buf: DeviceBuffer[bf16], o: Int, n: Int, lt: LT) -> TileTensor[bf16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[bf16](ctx, buf.unsafe_ptr().unsafe_offset(o), n, owning=False)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[bf16, LT, MutAnyOrigin]](t)


def q_tensors(
    ctx: DeviceContext, q_dev: DeviceBuffer[u8], b: Int
) raises -> Tuple[TileTensor[u8, type_of(q4_layout), MutAnyOrigin], TileTensor[f16, type_of(s_layout), MutAnyOrigin]]:
    return (view[u8](ctx, q_dev, b * Q4BYTES, N * K // 2, q4_layout), view[f16](ctx, q_dev, b * Q4BYTES + N * K // 2, N * (K // 32), s_layout))


def wmma_launch[WTM: Int, WTN: Int, WAVES_M: Int](
    ctx: DeviceContext, A: TileTensor[bf16, type_of(a_layout), MutAnyOrigin],
    Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
    C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int,
) raises:
    comptime BM = WAVES_M * WTM * 16
    comptime BN = (8 // WAVES_M) * WTN * 16
    ctx.enqueue_function[amar_matmul_prefill_q4[WTM, WTN, WAVES_M, False, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Q, S, C, Int32(m), Int32(N), Int32(K), grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=PF_THREADS,
    )


def wmma_cfg_name(cfg: Int) -> String:
    if cfg == 0:
        return "WTM1xWTN2 wavesM1 tile16x256"
    if cfg == 1:
        return "WTM2xWTN2 wavesM1 tile32x256"
    if cfg == 2:
        return "WTM2xWTN2 wavesM2 tile64x128"
    return "WTM4xWTN2 wavesM2 tile128x128"


def wmma_grid(cfg: Int, m: Int) -> String:
    var bm = 16 if cfg == 0 else (32 if cfg == 1 else (64 if cfg == 2 else 128))
    var bn = 256 if cfg <= 1 else 128
    return "grid=(" + String(ceildiv(N, bn)) + "," + String(ceildiv(m, bm)) + ") block=" + String(PF_THREADS)


def wmma_dispatch(cfg: Int, ctx: DeviceContext, A: TileTensor[bf16, type_of(a_layout), MutAnyOrigin],
                  Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
                  C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int) raises:
    if cfg == 0:
        wmma_launch[1, 2, 1](ctx, A, Q, S, C, m)
    elif cfg == 1:
        wmma_launch[2, 2, 1](ctx, A, Q, S, C, m)
    elif cfg == 2:
        wmma_launch[2, 2, 2](ctx, A, Q, S, C, m)
    else:
        wmma_launch[4, 2, 2](ctx, A, Q, S, C, m)


def lds_launch[ADT: DType, WM: Int, WN: Int, TM: Int, TN: Int](
    ctx: DeviceContext, A: TileTensor[ADT, type_of(a_layout), MutAnyOrigin],
    Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin], An: TileTensor[i32, type_of(d_layout), MutAnyOrigin],
    Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
    C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int,
) raises:
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    ctx.enqueue_function[amar_matmul_lds_q4[ADT, WM, WN, TM, TN, False, type_of(a_layout), type_of(d_layout), type_of(d_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Ad, An, Q, S, C, Int32(m), Int32(N), Int32(K), grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=MMQ_THREADS,
    )


def lds_dispatch[ADT: DType](cfg: Int, ctx: DeviceContext, A: TileTensor[ADT, type_of(a_layout), MutAnyOrigin],
                 Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin], An: TileTensor[i32, type_of(d_layout), MutAnyOrigin],
                 Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
                 C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int) raises:
    if cfg == 0:
        lds_launch[ADT, 2, 4, 2, 2](ctx, A, Ad, An, Q, S, C, m)
    else:
        lds_launch[ADT, 4, 2, 2, 4](ctx, A, Ad, An, Q, S, C, m)


def lds_cfg_name(cfg: Int) -> String:
    if cfg == 0:
        return "WM2xWN4 TM2xTN2 tile64x128 BLK_K32 lds2buf"
    return "WM4xWN2 TM2xTN4 tile128x128 BLK_K32 lds2buf"


def lds_grid(cfg: Int, m: Int) -> String:
    var bm = 64 if cfg == 0 else 128
    return "grid=(" + String(ceildiv(N, 128)) + "," + String(ceildiv(m, bm)) + ") block=" + String(MMQ_THREADS)


def lds_cfg_for(m: Int) -> Int:
    return 0 if m <= 64 else 1


def quant_launch(ctx: DeviceContext, A: TileTensor[bf16, type_of(a_layout), MutAnyOrigin],
                 Aq: TileTensor[i8, type_of(a_layout), MutAnyOrigin], Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin],
                 An: TileTensor[i32, type_of(d_layout), MutAnyOrigin], m: Int, mpad: Int) raises:
    ctx.enqueue_function[amar_quant_q8[type_of(a_layout), type_of(a_layout), type_of(d_layout), type_of(d_layout)]](
        A, Aq, Ad, An, Int32(m), Int32(K), Int32(mpad), grid_dim=ceildiv(mpad * (K // 32), QT_THREADS), block_dim=QT_THREADS,
    )


def frob_gate(name: String, got: HostBuffer[f32], want: HostBuffer[f32], n: Int) raises:
    var se = Float64(0)
    var sw = Float64(0)
    var worst = Float64(0)
    for i in range(n):
        var b = Float64(want[i])
        var e = abs(Float64(got[i]) - b)
        se += e * e
        sw += b * b
        worst = max(worst, e)
    var rms = (sw / Float64(n)) ** 0.5
    var rf = (se / sw) ** 0.5
    print(name, "rel_frobenius", rf, "max_abs/rms", worst / rms)
    if rf > 1e-2 or worst / rms > 5e-2:
        raise Error("mmq vs bf16-lds mismatch: " + name)


def floored_gate(name: String, got: HostBuffer[f32], want: HostBuffer[f32], n: Int, gate: Float64) raises:
    var worst = Float64(0)
    for i in range(n):
        var e = abs(Float64(got[i]) - Float64(want[i])) / (abs(Float64(want[i])) + 1e-2)
        if e > worst:
            worst = e
    if worst > gate:
        raise Error(name + " mismatch: " + String(worst))


def timed(ctx: DeviceContext, m: Int, arm: String, cfg: String, launch: String, meds: List[Float64]) raises:
    var med = median5(meds)
    var mn = meds[0]
    var mx = meds[0]
    for i in range(len(meds)):
        mn = min(mn, meds[i])
        mx = max(mx, meds[i])
    print(m, arm, cfg, launch, med, mn, mx, Float64(2 * m) * Float64(N) * Float64(K) / med / 1e6)


def cfg_for(m: Int) -> Int:
    if m <= 16:
        return 0
    if m <= 32:
        return 1
    if m <= 64:
        return 2
    return 3


def rowloop(ctx: DeviceContext, a_d: DeviceBuffer[bf16], Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin],
            S: TileTensor[f16, type_of(s_layout), MutAnyOrigin], c_d: DeviceBuffer[f32], p_t: TileTensor[f32, type_of(p_layout), MutAnyOrigin], m: Int) raises:
    var r0 = 0
    while r0 < m:
        var mr = min(SM, m - r0)
        var Aw = view_bf16(ctx, a_d, r0 * K, SM * K, a8_layout)
        var Cw = view_f32(ctx, c_d, r0 * N, SM * N, c8_layout)
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, SM, type_of(a8_layout), type_of(q4_layout), type_of(s_layout), type_of(p_layout)]](
            Aw, Q, S, p_t, Int32(mr), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS,
        )
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_layout), type_of(c8_layout), 1]](
            p_t, Cw, Int32(mr), Int32(N), grid_dim=ceildiv(mr * N, 256), block_dim=256,
        )
        r0 += SM


def median5(v: List[Float64]) -> Float64:
    var s = v.copy()
    sort(s)
    return s[len(s) // 2]


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var q_h = ctx.enqueue_create_host_buffer[u8](Q4BYTES)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    ctx.synchronize()
    load_into(".work/engine-pack-q4/pack.bin", q_h.unsafe_ptr(), Q4BYTES, pack_offset("blk.0.ffn_gate.weight"))
    var st = UInt64(99)
    for i in range(MMAX * K):
        a_h[i] = Scalar[bf16](lcg(st))
    var q_d = ctx.enqueue_create_buffer[u8](NBUF * Q4BYTES)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var c_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    var aq_d = ctx.enqueue_create_buffer[i8](MMAX * K)
    var ad_d = ctx.enqueue_create_buffer[f32](MMAX * (K // 32))
    var an_d = ctx.enqueue_create_buffer[i32](MMAX * (K // 32))
    for b in range(NBUF):
        var dst = DeviceBuffer[u8](ctx, q_d.unsafe_ptr().unsafe_offset(b * Q4BYTES), Q4BYTES, owning=False)
        ctx.enqueue_copy(dst_buf=dst, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.synchronize()
    var A = TileTensor(a_d, a_layout)
    var C = TileTensor(c_d, c_layout)
    var Aq = TileTensor(aq_d, a_layout)
    var Ad = TileTensor(ad_d, d_layout)
    var An = TileTensor(an_d, d_layout)
    quant_launch(ctx, A, Aq, Ad, An, MMAX, MMAX)
    ctx.synchronize()
    print("ABLATION ABL", ABL, "(0 none, 1 no B global load, 2 no A global load, 3 no per-block epilogue, 4 no barrier, 5 no d8/nu loads) shape N", N, "K", K, "NBUF", NBUF, "ITERS", ITERS, "REPEATS", REPEATS)
    var ns: List[Int] = [256, 512, 1024]
    for ni in range(len(ns)):
        var m = ns[ni]
        var meds3 = List[Float64]()
        for _ in range(REPEATS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for it in range(ITERS):
                var qb = q_tensors(ctx, q_d, it % NBUF)
                lds_dispatch[bf16](1, ctx, A, Ad, An, qb[0], qb[1], C, m)
            ctx.synchronize()
            meds3.append(Float64(perf_counter_ns() - t0) / 1e3 / ITERS)
        timed(ctx, m, "bf16-lds", lds_cfg_name(1), lds_grid(1, m), meds3)
        var meds4 = List[Float64]()
        for _ in range(REPEATS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for it in range(ITERS):
                var qb = q_tensors(ctx, q_d, it % NBUF)
                lds_dispatch[i8](1, ctx, Aq, Ad, An, qb[0], qb[1], C, m)
            ctx.synchronize()
            meds4.append(Float64(perf_counter_ns() - t0) / 1e3 / ITERS)
        timed(ctx, m, "mmq", lds_cfg_name(1), lds_grid(1, m), meds4)
