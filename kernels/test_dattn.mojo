"""Numerics probe for the generic decode attention (bench/dattn-protocol.md gate 2).
deterministic inputs (op_bench's fill formula, V with a seed offset), one instantiation
per shape, output dumped for tools/dattn-ref.py's fp64 reference.
usage: test_dattn SHAPE T PATH NS NLD QSCALE OUT
  SHAPE S0 (shipped 256/16/4 f32) | S1 (256/16/4 f16) | S2 (64/40/8 f16) | S3 (128/28/4 f16)
  PATH exact | split; NS requested split count (clamped to the span count); NLD 4 | 8
"""
from std.sys import argv, has_accelerator

from max.gpu.host import DeviceContext, HostBuffer

from dattn import f32, dspan
from dattn_harness import fill_q, fill_kv, kv_pool_elems, eff_nsplit, launch_dattn

comptime f16 = DType.float16


def run[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NLD: Int
](T: Int, exact: Bool, ns_req: Int, qscale: Int, out_path: String) raises:
    var ctx = DeviceContext()
    var ns = 1 if exact else eff_nsplit[HD, NLD](T, ns_req)
    var pool = kv_pool_elems[HD, NKVH](T)
    var q_h = ctx.enqueue_create_host_buffer[f32](NQH * HD)
    var k_h = ctx.enqueue_create_host_buffer[KVT](pool)
    var v_h = ctx.enqueue_create_host_buffer[KVT](pool)
    var o_h = ctx.enqueue_create_host_buffer[f32](NQH * HD)
    ctx.synchronize()
    fill_q(q_h, NQH * HD, qscale)
    fill_kv[HD, NKVH, KVT](k_h, T, 0)
    fill_kv[HD, NKVH, KVT](v_h, T, 1000003)
    var q_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var k_d = ctx.enqueue_create_buffer[KVT](pool)
    var v_d = ctx.enqueue_create_buffer[KVT](pool)
    var o_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var p_d = ctx.enqueue_create_buffer[f32](NQH * ns * (HD + 2))
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=k_d, src_buf=k_h)
    ctx.enqueue_copy(dst_buf=v_d, src_buf=v_h)
    o_d.enqueue_fill(0)
    launch_dattn[HD, NQH, NKVH, KVT, NLD](ctx, q_d, k_d, v_d, o_d, p_d, T, ns, exact)
    ctx.enqueue_copy(dst_buf=o_h, src_buf=o_d)
    ctx.synchronize()
    print(
        "echo HD", HD, "NQH", NQH, "NKVH", NKVH, "KVT", "f16" if KVT == f16 else "f32",
        "T", T, "path", "exact" if exact else "split", "nsplit", ns, "nld", NLD,
        "span", dspan[HD, NLD](), "qscale", qscale,
    )
    with open(out_path, "w") as f:
        var p = o_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=NQH * HD * 4))


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var a = argv()
    if len(a) != 8:
        raise Error("usage: test_dattn SHAPE T PATH NS NLD QSCALE OUT")
    var shape = String(a[1])
    var T = Int(String(a[2]))
    var exact = String(a[3]) == "exact"
    var ns = Int(String(a[4]))
    var nld = Int(String(a[5]))
    var qscale = Int(String(a[6]))
    var out_path = String(a[7])
    if shape == "S0":
        if nld == 4:
            run[256, 16, 4, f32, 4](T, exact, ns, qscale, out_path)
        else:
            run[256, 16, 4, f32, 8](T, exact, ns, qscale, out_path)
    elif shape == "S1":
        if nld == 4:
            run[256, 16, 4, f16, 4](T, exact, ns, qscale, out_path)
        else:
            run[256, 16, 4, f16, 8](T, exact, ns, qscale, out_path)
    elif shape == "S2":
        if nld == 4:
            run[64, 40, 8, f16, 4](T, exact, ns, qscale, out_path)
        else:
            run[64, 40, 8, f16, 8](T, exact, ns, qscale, out_path)
    elif shape == "S3":
        if nld == 4:
            run[128, 28, 4, f16, 4](T, exact, ns, qscale, out_path)
        else:
            run[128, 28, 4, f16, 8](T, exact, ns, qscale, out_path)
    else:
        raise Error("unknown shape " + shape)
