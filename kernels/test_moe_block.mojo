"""Parity: one decode token through the qwen35moe sparse-MoE block on GPU vs
the numpy reference (tools/moe-ref.py implementing transformers
Qwen3_5MoeSparseMoeBlock).

Gates: the eight selected expert ids EXACT, and routing weights, routed output,
shared-expert output and the block output y all rel < 5e-3. routed and shared
are checked separately so a failure localises to the routed path, the shared
path, or the router itself.

Run tools/moe-ref.py first to write the fixtures into .work/gguf/.
"""
from std.math import ceildiv
from std.memory import alloc, unsafe_memcpy
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from matmul_skinny import amar_matmul_skinny_m1_row, ROW_WAVES, ROW_THREADS
from ssm import amar_cast_bf16
from moe import (
    amar_moe_router_top8, amar_moe_sig_gate, amar_moe_gate_up, amar_moe_down,
    N_EXP, TOPK, E_FFN, SH_FFN, MOE_H, MOE_WAVES, MOE_THREADS,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32

comptime x_1 = row_major[MOE_H]()
comptime x_2 = row_major[1, MOE_H]()
comptime wr_2 = row_major[N_EXP, MOE_H]()
comptime lg_1 = row_major[N_EXP]()
comptime idx_1 = row_major[TOPK]()
comptime wt_1 = row_major[TOPK]()
comptime one_1 = row_major[1]()

comptime wg_2 = row_major[N_EXP * E_FFN, MOE_H]()
comptime wd_2 = row_major[N_EXP * MOE_H, E_FFN]()
comptime h_1 = row_major[TOPK * E_FFN]()
comptime h_2 = row_major[TOPK, E_FFN]()

comptime wgs_2 = row_major[SH_FFN, MOE_H]()
comptime wds_2 = row_major[MOE_H, SH_FFN]()
comptime hs_1 = row_major[SH_FFN]()
comptime hs_2 = row_major[1, SH_FFN]()

comptime o_1 = row_major[MOE_H]()


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int
) raises:
    with open(path, "r") as f:
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("size mismatch for " + path)
        unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=size)


def check(name: String, got: MutPointer[Float32, MutUntrackedOrigin],
          want: MutPointer[Float32, MutUntrackedOrigin], n: Int,
          gate: Float64 = 5e-3) raises:
    var worst = Float64(0)
    var wi = 0
    for i in range(n):
        var e = abs(Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i]))
        var rel = e / (abs(Float64(want[unsafe_offset=i])) + 1e-2)
        if rel > worst:
            worst = rel
            wi = i
    print(
        name, "max_rel:", worst, "at", wi,
        "got", got[unsafe_offset=wi], "want", want[unsafe_offset=wi],
    )
    if worst > gate:
        raise Error("parity failure: " + name)


def check_i32(name: String, got: MutPointer[Int32, MutUntrackedOrigin],
              want: MutPointer[Int32, MutUntrackedOrigin], n: Int) raises:
    for i in range(n):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            print(name, "mismatch at", i,
                  "got", got[unsafe_offset=i], "want", want[unsafe_offset=i])
            raise Error("parity failure: " + name)
    print(name, "exact over", n)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime D = ".work/gguf/"

    comptime N_EW = N_EXP * E_FFN * MOE_H
    comptime N_DW = N_EXP * MOE_H * E_FFN

    var x_h = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var wr_h = ctx.enqueue_create_host_buffer[f32](N_EXP * MOE_H)
    var wsg_h = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var wg_h = ctx.enqueue_create_host_buffer[bf16](N_EW)
    var wu_h = ctx.enqueue_create_host_buffer[bf16](N_EW)
    var wd_h = ctx.enqueue_create_host_buffer[bf16](N_DW)
    var wgs_h = ctx.enqueue_create_host_buffer[bf16](SH_FFN * MOE_H)
    var wus_h = ctx.enqueue_create_host_buffer[bf16](SH_FFN * MOE_H)
    var wds_h = ctx.enqueue_create_host_buffer[bf16](MOE_H * SH_FFN)
    var idx0_h = ctx.enqueue_create_host_buffer[i32](1)
    ctx.synchronize()

    load_into(D + "moe_x.bin", x_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * 4)
    load_into(D + "moe_wr.bin", wr_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_EXP * MOE_H * 4)
    load_into(D + "moe_wsg.bin", wsg_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * 4)
    load_into(D + "moe_wg.bin", wg_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_EW * 2)
    load_into(D + "moe_wu.bin", wu_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_EW * 2)
    load_into(D + "moe_wd.bin", wd_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_DW * 2)
    load_into(D + "moe_wgs.bin", wgs_h.unsafe_ptr().unsafe_bitcast[UInt8](), SH_FFN * MOE_H * 2)
    load_into(D + "moe_wus.bin", wus_h.unsafe_ptr().unsafe_bitcast[UInt8](), SH_FFN * MOE_H * 2)
    load_into(D + "moe_wds.bin", wds_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * SH_FFN * 2)
    idx0_h[0] = 0

    var x_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var xb_d = ctx.enqueue_create_buffer[bf16](MOE_H)
    var wr_d = ctx.enqueue_create_buffer[f32](N_EXP * MOE_H)
    var wsg_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var lg_d = ctx.enqueue_create_buffer[f32](N_EXP)
    var idx_d = ctx.enqueue_create_buffer[i32](TOPK)
    var wt_d = ctx.enqueue_create_buffer[f32](TOPK)
    var sg_d = ctx.enqueue_create_buffer[f32](1)
    var idx0_d = ctx.enqueue_create_buffer[i32](1)
    var wg_d = ctx.enqueue_create_buffer[bf16](N_EW)
    var wu_d = ctx.enqueue_create_buffer[bf16](N_EW)
    var wd_d = ctx.enqueue_create_buffer[bf16](N_DW)
    var wgs_d = ctx.enqueue_create_buffer[bf16](SH_FFN * MOE_H)
    var wus_d = ctx.enqueue_create_buffer[bf16](SH_FFN * MOE_H)
    var wds_d = ctx.enqueue_create_buffer[bf16](MOE_H * SH_FFN)
    var h_d = ctx.enqueue_create_buffer[f32](TOPK * E_FFN)
    var hb_d = ctx.enqueue_create_buffer[bf16](TOPK * E_FFN)
    var hs_d = ctx.enqueue_create_buffer[f32](SH_FFN)
    var hsb_d = ctx.enqueue_create_buffer[bf16](SH_FFN)
    var routed_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var shared_d = ctx.enqueue_create_buffer[f32](MOE_H)

    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.enqueue_copy(dst_buf=wr_d, src_buf=wr_h)
    ctx.enqueue_copy(dst_buf=wsg_d, src_buf=wsg_h)
    ctx.enqueue_copy(dst_buf=wg_d, src_buf=wg_h)
    ctx.enqueue_copy(dst_buf=wu_d, src_buf=wu_h)
    ctx.enqueue_copy(dst_buf=wd_d, src_buf=wd_h)
    ctx.enqueue_copy(dst_buf=wgs_d, src_buf=wgs_h)
    ctx.enqueue_copy(dst_buf=wus_d, src_buf=wus_h)
    ctx.enqueue_copy(dst_buf=wds_d, src_buf=wds_h)
    ctx.enqueue_copy(dst_buf=idx0_d, src_buf=idx0_h)
    ctx.synchronize()

    var x1 = TileTensor(x_d, x_1)
    var x2 = TileTensor(x_d, x_2)
    var xb1 = TileTensor(xb_d, x_1)
    var xb2 = TileTensor(xb_d, x_2)
    var wr = TileTensor(wr_d, wr_2)
    var wsg = TileTensor(wsg_d, x_1)
    var lg = TileTensor(lg_d, lg_1)
    var idx = TileTensor(idx_d, idx_1)
    var wt = TileTensor(wt_d, wt_1)
    var sg = TileTensor(sg_d, one_1)
    var idx0 = TileTensor(idx0_d, one_1)
    var wg = TileTensor(wg_d, wg_2)
    var wu = TileTensor(wu_d, wg_2)
    var wd = TileTensor(wd_d, wd_2)
    var wgs = TileTensor(wgs_d, wgs_2)
    var wus = TileTensor(wus_d, wgs_2)
    var wds = TileTensor(wds_d, wds_2)
    var h1 = TileTensor(h_d, h_1)
    var hb2 = TileTensor(hb_d, h_2)
    var hs1 = TileTensor(hs_d, hs_1)
    var hsb2 = TileTensor(hsb_d, hs_2)
    var routed = TileTensor(routed_d, o_1)
    var shared = TileTensor(shared_d, o_1)

    comptime k_cast = amar_cast_bf16[type_of(x_1), type_of(x_1)]
    ctx.enqueue_function[k_cast](
        x1, xb1, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, 256), block_dim=256,
    )

    comptime k_router = amar_matmul_skinny_m1_row[
        f32, 2, type_of(x_2), type_of(wr_2), type_of(lg_1)
    ]
    ctx.enqueue_function[k_router](
        x2, wr, lg, Int32(N_EXP), Int32(MOE_H),
        grid_dim=ceildiv(N_EXP, ROW_WAVES), block_dim=ROW_THREADS,
    )

    comptime k_top8 = amar_moe_router_top8[
        type_of(lg_1), type_of(idx_1), type_of(wt_1)
    ]
    ctx.enqueue_function[k_top8](lg, idx, wt, grid_dim=1, block_dim=N_EXP)

    comptime k_sig = amar_moe_sig_gate[
        type_of(x_1), type_of(x_1), type_of(one_1)
    ]
    ctx.enqueue_function[k_sig](x1, wsg, sg, Int32(MOE_H), grid_dim=1, block_dim=32)

    comptime k_gu = amar_moe_gate_up[
        TOPK, E_FFN, type_of(x_2), type_of(wg_2), type_of(wg_2),
        type_of(idx_1), type_of(h_1),
    ]
    ctx.enqueue_function[k_gu](
        xb2, wg, wu, idx, h1, Int32(MOE_H),
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )

    comptime k_cast_h = amar_cast_bf16[type_of(h_1), type_of(h_1)]
    ctx.enqueue_function[k_cast_h](
        TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
        grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
    )

    comptime k_down = amar_moe_down[
        TOPK, E_FFN, type_of(h_2), type_of(wd_2), type_of(idx_1),
        type_of(wt_1), type_of(o_1),
    ]
    ctx.enqueue_function[k_down](
        hb2, wd, idx, wt, routed, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )

    comptime k_gu_s = amar_moe_gate_up[
        1, SH_FFN, type_of(x_2), type_of(wgs_2), type_of(wgs_2),
        type_of(one_1), type_of(hs_1),
    ]
    ctx.enqueue_function[k_gu_s](
        xb2, wgs, wus, idx0, hs1, Int32(MOE_H),
        grid_dim=ceildiv(SH_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )

    comptime k_cast_hs = amar_cast_bf16[type_of(hs_1), type_of(hs_1)]
    ctx.enqueue_function[k_cast_hs](
        TileTensor(hs_d, hs_1), TileTensor(hsb_d, hs_1), Int32(SH_FFN),
        grid_dim=ceildiv(SH_FFN, 256), block_dim=256,
    )

    comptime k_down_s = amar_moe_down[
        1, SH_FFN, type_of(hs_2), type_of(wds_2), type_of(one_1),
        type_of(one_1), type_of(o_1),
    ]
    ctx.enqueue_function[k_down_s](
        hsb2, wds, idx0, sg, shared, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()

    var idx_got = ctx.enqueue_create_host_buffer[i32](TOPK)
    var wt_got = ctx.enqueue_create_host_buffer[f32](TOPK)
    var routed_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var shared_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    ctx.enqueue_copy(dst_buf=idx_got, src_buf=idx_d)
    ctx.enqueue_copy(dst_buf=wt_got, src_buf=wt_d)
    ctx.enqueue_copy(dst_buf=routed_got, src_buf=routed_d)
    ctx.enqueue_copy(dst_buf=shared_got, src_buf=shared_d)
    ctx.synchronize()

    var idx_ref = alloc[Int32](TOPK)
    var wt_ref = alloc[Float32](TOPK)
    var routed_ref = alloc[Float32](MOE_H)
    var shared_ref = alloc[Float32](MOE_H)
    var y_ref = alloc[Float32](MOE_H)
    load_into(D + "moe_idx_ref.bin", idx_ref.unsafe_bitcast[UInt8](), TOPK * 4)
    load_into(D + "moe_w_ref.bin", wt_ref.unsafe_bitcast[UInt8](), TOPK * 4)
    load_into(D + "moe_routed_ref.bin", routed_ref.unsafe_bitcast[UInt8](), MOE_H * 4)
    load_into(D + "moe_shared_ref.bin", shared_ref.unsafe_bitcast[UInt8](), MOE_H * 4)
    load_into(D + "moe_y_ref.bin", y_ref.unsafe_bitcast[UInt8](), MOE_H * 4)

    var y_got = alloc[Float32](MOE_H)
    for i in range(MOE_H):
        y_got[unsafe_offset=i] = (
            routed_got.unsafe_ptr()[unsafe_offset=i]
            + shared_got.unsafe_ptr()[unsafe_offset=i]
        )

    check_i32("expert_ids", idx_got.unsafe_ptr(), idx_ref, TOPK)
    check("routing_weights", wt_got.unsafe_ptr(), wt_ref, TOPK, 1e-4)
    check("routed", routed_got.unsafe_ptr(), routed_ref, MOE_H)
    check("shared", shared_got.unsafe_ptr(), shared_ref, MOE_H)
    check("y", y_got, y_ref, MOE_H)
    print("PASS: qwen35moe sparse-MoE block matches numpy reference (m=1)")

    # Feasibility measurement, NOT a preregistered perf claim (CLAUDE.md /
    # bench/PROTOCOL-RULES.md apply to champion claims). The question is whether
    # the top-8-of-256 gather dominates the routed FFN at m=1.
    #
    # The routed path touches 8 x (gate 2.10 + up 2.10 + down 2.10 MB) = 50.3 MB
    # of bf16 expert weights per token per layer. That is under the 96 MB
    # Infinity Cache, so a fixed expert set would sit resident and report a
    # fantasy number. ROTATION defeats that: ARMS disjoint expert sets walk
    # 8 x 50.3 = 402 MB, well past the cache.
    if getenv("BARO_MOE_BENCH", "0") == "1":
        comptime ARMS = 8
        comptime ITERS = 200
        var arm_h = ctx.enqueue_create_host_buffer[i32](ARMS * TOPK)
        for a in range(ARMS):
            for j in range(TOPK):
                arm_h[a * TOPK + j] = Int32(a * TOPK + j)
        var arm_d = ctx.enqueue_create_buffer[i32](ARMS * TOPK)
        ctx.enqueue_copy(dst_buf=arm_d, src_buf=arm_h)
        ctx.synchronize()

        for a in range(ARMS):
            var sel = DeviceBuffer[i32](
                ctx, arm_d.unsafe_ptr() + a * TOPK, TOPK, owning=False
            )
            var seltt = rebind[TileTensor[i32, type_of(idx_1), MutAnyOrigin]](
                TileTensor(sel, idx_1)
            )
            ctx.enqueue_function[k_gu](
                xb2, wg, wu, seltt, h1, Int32(MOE_H),
                grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
            )
        ctx.synchronize()

        var arms_rt = Int(getenv("BARO_MOE_ARMS", "8"))
        var t0 = perf_counter_ns()
        for it in range(ITERS):
            var a = it % arms_rt
            var sel = DeviceBuffer[i32](
                ctx, arm_d.unsafe_ptr() + a * TOPK, TOPK, owning=False
            )
            var seltt = rebind[TileTensor[i32, type_of(idx_1), MutAnyOrigin]](
                TileTensor(sel, idx_1)
            )
            ctx.enqueue_function[k_gu](
                xb2, wg, wu, seltt, h1, Int32(MOE_H),
                grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
            )
            ctx.enqueue_function[k_cast_h](
                TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
                grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
            )
            ctx.enqueue_function[k_down](
                hb2, wd, seltt, wt, routed, Int32(MOE_H),
                grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()

        var us = Float64(t1 - t0) / 1000.0 / Float64(ITERS)
        var mb = Float64(TOPK) * 3.0 * Float64(E_FFN) * Float64(MOE_H) * 2.0 / 1e6
        print("moe routed block: arms", arms_rt, ":", us, "us/token/layer,", mb, "MB bf16 touched,",
              mb / us * 1000.0, "GB/s")
        print("  40 layers ->", us * 40.0 / 1000.0, "ms/token =",
              1000.0 / (us * 40.0 / 1000.0), "tok/s ceiling from expert traffic alone")
