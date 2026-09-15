"""Device sampler vs host reference, on the real fixed logits row (M5,
briefs/2026-09-15-wiring-lane.md, gate 2: "device sampler matches
serve/sample_ref.mojo per token on fixed seed and fixed logits").

kernels/test_sample.mojo already device-tests amar_sample_row's distribution
and seed reproducibility (P-K1, P-K3/P-K4) on a small synthetic vocab (VS=64);
kernels/test_sample_ref.mojo already host-tests sample_ref.mojo including one
real-vocab row. Neither compares the two implementations against each other
on the same inputs, which is what gate 2 asks for -- this file is that cross
check, at the real VOCAB width serve/window.mojo actually calls with.

Build: ./.venv/bin/mojo build kernels/test_sample_device.mojo -I kernels -I serve -o .work/test_sample_device
Needs .work/draft-logits.bin (a real decode row, written as a side effect of
run-tests.sh / the one-shot engine path); skips cleanly if absent.
"""
from std.math import sqrt
from std.sys import has_accelerator, exit

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from registry import *
from sample import amar_sample_row, SAMP_THREADS
from sample_ref import sample_row_ref, is_valid

comptime FMAX = Float32(3.4028234663852886e38)
comptime x_l = row_major[1, VOCAB]()
comptime o_l = row_major[1]()


def load_logits() raises -> List[Float32]:
    var path = ".work/draft-logits.bin"
    with open(path, "r") as f:
        var data = f.read_bytes()
        var n = len(data) // 4
        var row = List[Float32](unsafe_uninit_length=n)
        var p = data.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n):
            row[i] = p[i]
        return row^


def device_sample(
    ctx: DeviceContext, mut xd: DeviceBuffer[f32], t: Float32, k: Int, p: Float32, mp: Float32, seed: UInt64, counter: UInt64,
) raises -> Tuple[Int, Float32]:
    var td = ctx.enqueue_create_buffer[DType.int32](1)
    var pd = ctx.enqueue_create_buffer[f32](1)
    comptime kern = amar_sample_row[type_of(x_l), type_of(o_l), type_of(o_l)]
    ctx.enqueue_function[kern](
        TileTensor(xd, x_l), TileTensor(td, o_l), TileTensor(pd, o_l), Int32(VOCAB),
        t, Int32(k), p, mp, seed, counter, grid_dim=1, block_dim=SAMP_THREADS,
    )
    var th = ctx.enqueue_create_host_buffer[DType.int32](1)
    var ph = ctx.enqueue_create_host_buffer[f32](1)
    ctx.enqueue_copy(dst_buf=th, src_buf=td)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    return (Int(th[0]), Float32(ph[0]))


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    var row: List[Float32]
    try:
        row = load_logits()
    except:
        print("SKIP: .work/draft-logits.bin not present (run run-tests.sh first)")
        return
    if len(row) != VOCAB:
        print("SKIP: draft-logits.bin has", len(row), "entries, this build's VOCAB is", VOCAB)
        return

    var ctx = DeviceContext()
    var xh = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.synchronize()
    for i in range(VOCAB):
        xh[i] = row[i]
    var xd = ctx.enqueue_create_buffer[f32](VOCAB)
    ctx.enqueue_copy(dst_buf=xd, src_buf=xh)
    ctx.synchronize()

    var fails = 0

    # Greedy (temperature = 0): device must match the host reference, which
    # kernels/test_sample_ref.mojo already showed matches plain argmax.
    var dg = device_sample(ctx, xd, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0))
    var hg = sample_row_ref(row, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0), 0)
    if dg[0] == Int(hg[0]) and dg[1] == hg[1]:
        print("PASS greedy (T=0): device token", dg[0], "prob", dg[1], "== host")
    else:
        print("FAIL greedy (T=0): device", dg[0], dg[1], "!= host", hg[0], hg[1])
        fails += 1

    # Sampling configs x many (seed, counter) pairs, row fixed at 0 (both
    # sides use the same "row" in the RNG stream, matching the single-row
    # decode step this kernel is actually called with in window.mojo).
    var configs = List[Tuple[String, Float32, Int, Float32, Float32]]()
    configs.append(("T1 k0 p1", Float32(1.0), 0, Float32(1.0), Float32(0.0)))
    configs.append(("T0.7 k20 p0.8", Float32(0.7), 20, Float32(0.8), Float32(0.0)))
    configs.append(("T1.3 k12 p0.9 minp0.05", Float32(1.3), 12, Float32(0.9), Float32(0.05)))
    configs.append(("T0.5 k0 p0.6", Float32(0.5), 0, Float32(0.6), Float32(0.0)))

    for ci in range(len(configs)):
        var name = configs[ci][0]
        var t = configs[ci][1]
        var k = configs[ci][2]
        var p = configs[ci][3]
        var mp = configs[ci][4]
        var mismatches = 0
        var checked = 0
        for counter in range(64):
            var d = device_sample(ctx, xd, t, k, p, mp, UInt64(42), UInt64(counter))
            var h = sample_row_ref(row, t, k, p, mp, UInt64(42), UInt64(counter), 0)
            checked += 1
            if d[0] != Int(h[0]) or abs(d[1] - h[1]) > Float32(1e-4):
                mismatches += 1
                if mismatches <= 3:
                    print("  mismatch counter", counter, ": device", d[0], d[1], "host", h[0], h[1])
        if mismatches == 0:
            print("PASS", name, ": device == host on all", checked, "(seed, counter) draws")
        else:
            print("FAIL", name, ":", mismatches, "of", checked, "draws disagree")
            fails += 1

    if fails == 0:
        print("PASS: device sampler matches serve/sample_ref.mojo, fixed logits")
    else:
        print("FAIL:", fails, "check(s) failed")
        exit(1)
