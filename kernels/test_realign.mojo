"""REALIGN live test (round 2): for 5 real prompts on .work/engine-pack-q4,
prefill through the prompt under mega=True (the E9/HARNESS configuration --
bench_latent_handoff.mojo runs every latent step through the megakernel),
call realign_expected_embedding on the final row, and dump the row's
pre-final-norm hidden (b.x_d, the one buffer every path keeps current) and
e to .work/realign-dump/ for tools/realign_oracle.py (numpy) to recompute
logits and e independently off the dumped hidden state and the pack's own
weights -- never against a Mojo-side logits dump, since realign_expected_embedding
no longer exposes logits at all (see serve/realign.mojo's docstring for why
b.logits_d/b.hn_d are not usable here).
"""
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major
from registry import *
from window import *
from realign import realign_expected_embedding
from realign_kernels import amar_realign_gather, amar_realign_reduce, REALIGN_SPLIT


def read_prompt(path: String) raises -> List[Int]:
    var prompt = List[Int]()
    with open(path, "r") as f:
        var data = f.read_bytes()
        var val = 0
        var have = False
        for i in range(len(data)):
            var b = Int(data[i])
            if b >= 48 and b <= 57:
                val = val * 10 + (b - 48)
                have = True
            else:
                if have:
                    prompt.append(val)
                val = 0
                have = False
        if have:
            prompt.append(val)
    return prompt^


def reset_buffers(ctx: DeviceContext, mut b: WindowBufs, prompt: List[Int], tmax: Int) raises:
    ctx.enqueue_memset(b.convstate_d, 0)
    ctx.enqueue_memset(b.sstate_d, 0)
    ctx.enqueue_memset(b.kc_d, 0)
    ctx.enqueue_memset(b.vc_d, 0)
    ctx.enqueue_memset(b.kc32_d, 0)
    ctx.enqueue_memset(b.vc32_d, 0)
    ctx.enqueue_memset(b.ctr_d, 0)
    ctx.enqueue_memset(b.prof_d, 0)
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.synchronize()
    for i in range(tmax):
        toks_h[i] = 0
    for i in range(len(prompt)):
        toks_h[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=b.toks_d, src_buf=toks_h)
    ctx.synchronize()


struct PackMin(Movable):
    var wbuf: DeviceBuffer[DType.uint8]
    var off: List[Int]
    var pack_q4: Bool
    var q4_off: Int
    var e: Int

    def __init__(out self, var wbuf: DeviceBuffer[DType.uint8], var off: List[Int], pack_q4: Bool, q4_off: Int, e: Int):
        self.wbuf = wbuf^
        self.off = off^
        self.pack_q4 = pack_q4
        self.q4_off = q4_off
        self.e = e


def load_pack_min(ctx: DeviceContext, packdir: String) raises -> PackMin:
    # trimmed load_pack: same index parse + staged read as serve/engine.mojo main().
    var off = List[Int]()
    var total = 0
    var q4_off = 0
    var have_q4_draft = False
    var pack_q4 = False
    with open(packdir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var n = Int(parts[3])
            var dt = String(parts[1])
            off.append(Int(parts[2]))
            if dt == "bf16":
                total += n * B2
            elif dt == "f32":
                total += n * B4
            elif dt == "q8":
                total += n + (n // 32) * 2
            elif dt == "q4":
                if String(parts[0]) == "output.weight.q4draft":
                    q4_off = Int(parts[2])
                    have_q4_draft = True
                else:
                    pack_q4 = True
                total += n // 2 + (n // 32) * 2
            else:
                raise Error("unknown pack dtype " + dt)
    var e = len(off) - (1 if have_q4_draft else 0) - 15

    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    comptime RSPLIT = 4
    var stage0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var stage1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    with open(packdir + "/pack.bin", "r") as f:
        var fd = f._get_raw_fd()
        var done = 0
        var flip = False
        while done < total:
            var want = min(CHUNK, total - done)
            var stage = stage1 if flip else stage0
            var sptr = stage.unsafe_ptr()
            var rerr = List[Int64](unsafe_uninit_length=RSPLIT)
            var rerr_ptr = rerr.unsafe_ptr()
            def rchunk(t: Int) {imm fd, imm want, imm done, imm sptr, imm rerr_ptr}:
                var lo = want * t // RSPLIT
                var hi = want * (t + 1) // RSPLIT
                var got = lo
                while got < hi:
                    var n = external_call["pread", c_ssize_t](fd, sptr.unsafe_offset(got), hi - got, Int64(done + got))
                    if n <= 0:
                        rerr_ptr[t] = 1
                        return
                    got += Int(n)
                rerr_ptr[t] = 0
            parallelize(rchunk, RSPLIT)
            for t in range(RSPLIT):
                if rerr[t] != 0:
                    raise Error("short read")
            ctx.synchronize()
            var dslice = DeviceBuffer[DType.uint8](ctx, wbuf.unsafe_ptr() + done, want, owning=False)
            var hslice = stage.create_sub_buffer[DType.uint8](0, want) if want != CHUNK else stage
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            done += want
            flip = not flip
    ctx.synchronize()
    return PackMin(wbuf^, off^, pack_q4, q4_off, e)


def alloc_bufs_min(ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], tmax: Int) raises -> WindowBufs:
    var tpages = ceildiv(tmax, KVPAGE)
    var kvpool = tpages * N_ATT * NKVH * KVHSTR
    var kvpool1 = tpages * NKVH * KVHSTR
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    var win_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)

    var x_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var curb_d = ctx.enqueue_create_buffer[bf16](MROWS * H)
    var qkv_d = ctx.enqueue_create_buffer[f32](MROWS * CONV)
    var z_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var eg_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](MROWS * CONV)
    var so_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V * SSTATE)
    var resb_d = ctx.enqueue_create_buffer[bf16](MROWS * H)
    var qf_d = ctx.enqueue_create_buffer[f32](MROWS * QF)
    var q_d = ctx.enqueue_create_buffer[f32](MROWS * NQH * HD)
    var gate_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var k_d = ctx.enqueue_create_buffer[f32](MROWS * KV)
    var v_d = ctx.enqueue_create_buffer[f32](MROWS * KV)
    var ao_d = ctx.enqueue_create_buffer[f32](MROWS * NQH * HD)
    var fgb_d = ctx.enqueue_create_buffer[bf16](MROWS * FFN)
    var aq_d = ctx.enqueue_create_buffer[DType.int8](MROWS * FFN)
    var asc_d = ctx.enqueue_create_buffer[DType.float16](MROWS * (FFN // 32))
    var logits_d = ctx.enqueue_create_buffer[f32](MROWS * VOCAB)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](tmax)
    var hn_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var de_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var hd_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var cc_d = ctx.enqueue_create_buffer[bf16](MROWS * QF)
    var dtok_d = ctx.enqueue_create_buffer[DType.int32](KMAX + 1)

    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_ffn_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_ffn2_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_v_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * VOCAB)

    var xp_d = ctx.enqueue_create_buffer[f32](CP * H)
    var curbp_d = ctx.enqueue_create_buffer[bf16](CP * H)
    var qkvp_d = ctx.enqueue_create_buffer[f32](CP * CONV)
    var zp_d = ctx.enqueue_create_buffer[f32](CP * H)
    var arp_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var brp_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var egp_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var betap_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var convp_d = ctx.enqueue_create_buffer[f32](CP * CONV)
    var sop_d = ctx.enqueue_create_buffer[f32](CP * NH_V * SSTATE)
    var resbp_d = ctx.enqueue_create_buffer[bf16](CP * H)
    var qfp_d = ctx.enqueue_create_buffer[f32](CP * QF)
    var qp_d = ctx.enqueue_create_buffer[f32](CP * NQH * HD)
    var gatep_d = ctx.enqueue_create_buffer[f32](CP * H)
    var kp_d = ctx.enqueue_create_buffer[f32](CP * KV)
    var vp_d = ctx.enqueue_create_buffer[f32](CP * KV)
    var aop_d = ctx.enqueue_create_buffer[f32](CP * NQH * HD)
    var gp_d = ctx.enqueue_create_buffer[f32](CP * FFN)
    var up_d = ctx.enqueue_create_buffer[f32](CP * FFN)
    var fgbp_d = ctx.enqueue_create_buffer[bf16](CP * FFN)

    var convstate_d = ctx.enqueue_create_buffer[f32](SLOTS * CONV_SLOT)
    var sstate_d = ctx.enqueue_create_buffer[f32](SLOTS * SSM_SLOT)
    var kc_d = ctx.enqueue_create_buffer[KVT](kvpool)
    var vc_d = ctx.enqueue_create_buffer[KVT](kvpool)
    var kc32_d = ctx.enqueue_create_buffer[KVT](kvpool1)
    var vc32_d = ctx.enqueue_create_buffer[KVT](kvpool1)
    ctx.enqueue_memset(convstate_d, 0)
    ctx.enqueue_memset(sstate_d, 0)
    ctx.enqueue_memset(kc_d, 0)
    ctx.enqueue_memset(vc_d, 0)
    ctx.enqueue_memset(kc32_d, 0)
    ctx.enqueue_memset(vc32_d, 0)
    ctx.synchronize()
    var off_h = ctx.enqueue_create_host_buffer[DType.int64](512)
    ctx.synchronize()
    for i in range(512):
        off_h[i] = Int64(off[i]) if i < len(off) else 0
    var off_d = ctx.enqueue_create_buffer[DType.int64](512)
    ctx.enqueue_copy(dst_buf=off_d, src_buf=off_h)
    var araw_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var braw_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var ctr_d = ctx.enqueue_create_buffer[DType.uint32](3)
    var prof_d = ctx.enqueue_create_buffer[DType.int64](16 * N_LAYERS + 4)
    ctx.enqueue_memset(ctr_d, 0)
    ctx.enqueue_memset(prof_d, 0)
    var dbg_d = ctx.enqueue_create_buffer[f32](2 * N_LAYERS * H)
    var hmax_d = ctx.enqueue_create_buffer[f32](MEGA_MR * MEGA_G_WIN)
    var hidx_d = ctx.enqueue_create_buffer[DType.int32](MEGA_MR * MEGA_G_WIN)
    var dump_h = ctx.enqueue_create_host_buffer[f32](GEN_N * 2 * N_LAYERS * H)
    var stream_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    ctx.synchronize()
    return WindowBufs(
        wbuf=wbuf.copy(), off=off.copy(), dtok_h=dtok_h.copy(), win_h=win_h.copy(), x_d=x_d.copy(),
        curb_d=curb_d.copy(), qkv_d=qkv_d.copy(), z_d=z_d.copy(), eg_d=eg_d.copy(), beta_d=beta_d.copy(),
        conv_d=conv_d.copy(), so_d=so_d.copy(), resb_d=resb_d.copy(), qf_d=qf_d.copy(), q_d=q_d.copy(),
        gate_d=gate_d.copy(), k_d=k_d.copy(), v_d=v_d.copy(), ao_d=ao_d.copy(), fgb_d=fgb_d.copy(),
        aq_d=aq_d.copy(), asc_d=asc_d.copy(), logits_d=logits_d.copy(), toks_d=toks_d.copy(), hn_d=hn_d.copy(),
        de_d=de_d.copy(), hd_d=hd_d.copy(), cc_d=cc_d.copy(), dtok_d=dtok_d.copy(), p_qf_d=p_qf_d.copy(),
        p_h_d=p_h_d.copy(), p_kv_d=p_kv_d.copy(), p_32_d=p_32_d.copy(), p_32b_d=p_32b_d.copy(),
        p_ffn_d=p_ffn_d.copy(), p_ffn2_d=p_ffn2_d.copy(), p_v_d=p_v_d.copy(), xp_d=xp_d.copy(),
        curbp_d=curbp_d.copy(), qkvp_d=qkvp_d.copy(), zp_d=zp_d.copy(), arp_d=arp_d.copy(), brp_d=brp_d.copy(),
        egp_d=egp_d.copy(), betap_d=betap_d.copy(), convp_d=convp_d.copy(), sop_d=sop_d.copy(),
        resbp_d=resbp_d.copy(), qfp_d=qfp_d.copy(), qp_d=qp_d.copy(), gatep_d=gatep_d.copy(), kp_d=kp_d.copy(),
        vp_d=vp_d.copy(), aop_d=aop_d.copy(), gp_d=gp_d.copy(), up_d=up_d.copy(), fgbp_d=fgbp_d.copy(),
        convstate_d=convstate_d.copy(), sstate_d=sstate_d.copy(), kvpool=kvpool, kc_d=kc_d.copy(),
        vc_d=vc_d.copy(), kc32_d=kc32_d.copy(), vc32_d=vc32_d.copy(), off_d=off_d.copy(), araw_d=araw_d.copy(),
        braw_d=braw_d.copy(), ctr_d=ctr_d.copy(), prof_d=prof_d.copy(), dbg_d=dbg_d.copy(), hmax_d=hmax_d.copy(),
        hidx_d=hidx_d.copy(), dump_h=dump_h.copy(), stream_h=stream_h.copy(),
    )


def dump_f32(path: String, host: HostBuffer[f32], n: Int) raises:
    with open(path, "w") as f:
        var p = host.unsafe_ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=n * 4))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var packdir = ".work/engine-pack-q4"
    var pack = load_pack_min(ctx, packdir)
    var pack_q4 = pack.pack_q4
    var q4_off = pack.q4_off
    var e = pack.e

    comptime tmax = TMAX
    var bufs = alloc_bufs_min(ctx, pack.wbuf, pack.off, tmax)
    var wst = WindowState(
        pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0,
        tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0,
        fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0],
    )

    var prompt_names = List[String]()
    prompt_names.append("p01-water")
    prompt_names.append("p02-python-fib")
    prompt_names.append("p03-story")
    prompt_names.append("p04-list-planets")
    prompt_names.append("p05-math")

    var e_dev = ctx.enqueue_create_buffer[f32](H)
    var x0_h = ctx.enqueue_create_host_buffer[f32](H)
    var e_h = ctx.enqueue_create_host_buffer[f32](H)

    print("REALIGN_SPLIT (gather row-split over VOCAB):", REALIGN_SPLIT)
    print("prompt | prefill_toks | realign_us")
    print("-------+--------------+-----------")

    for p_idx in range(len(prompt_names)):
        var pname = prompt_names[p_idx]
        var prompt = read_prompt("bench/mtp-prompts/" + pname + ".tokens")
        var plen = len(prompt)

        reset_buffers(ctx, bufs, prompt, tmax)
        var cfg_prompt = WindowCfg(
            pack_q4=pack_q4, draft_q4=False, q4_off=q4_off, e=e, kcfg=2,
            spec=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
            dump=False, mega=True, att_split=TMAX, mega_win=False, dot3=False, pf_chunk=1024,
            pf_rows=0, pf_tail=0, n_total=plen, n_prompt=plen,
        )
        wst.reset(perf_counter_ns())
        while wst.pos < plen:
            step_window(ctx, bufs, cfg_prompt, wst)
        ctx.synchronize()

        ctx.synchronize()
        var t0 = perf_counter_ns()
        realign_expected_embedding(ctx, bufs, e_dev, pack_q4)
        ctx.synchronize()
        var dt_us = Float64(perf_counter_ns() - t0) / 1e3

        ctx.enqueue_copy(
            dst_buf=x0_h,
            src_buf=DeviceBuffer[f32](ctx, bufs.x_d.unsafe_ptr(), H, owning=False),
        )
        ctx.enqueue_copy(dst_buf=e_h, src_buf=e_dev)
        ctx.synchronize()

        dump_f32(".work/realign-dump/" + pname + "-x0.f32", x0_h, H)
        dump_f32(".work/realign-dump/" + pname + "-e.f32", e_h, H)

        var line = pname + " | " + String(plen) + " | " + String(dt_us)
        print(line)

    print("REALIGN test: dump complete -> .work/realign-dump/")
