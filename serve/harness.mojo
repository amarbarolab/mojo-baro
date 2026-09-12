"""Engine harness: pack loading and buffer allocation shared by serve/engine.mojo
and kernels/test_prefix.mojo. Moved verbatim out of engine.mojo main (M1a);
no decode-path code lives here.
"""
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *
from window import *


struct Pack(Movable):
    var wbuf: DeviceBuffer[DType.uint8]
    var off: List[Int]
    var total: Int
    var q4_off: Int
    var have_q4_draft: Bool
    var pack_q4: Bool
    var e: Int
    var fr_off: Int
    var fr_ids_off: Int
    var fr_k: Int

    def __init__(out self, var wbuf: DeviceBuffer[DType.uint8], var off: List[Int], total: Int, q4_off: Int, have_q4_draft: Bool, pack_q4: Bool, fr_off: Int = 0, fr_ids_off: Int = 0, fr_k: Int = 0):
        self.wbuf = wbuf^
        self.off = off^
        self.total = total
        self.q4_off = q4_off
        self.have_q4_draft = have_q4_draft
        self.pack_q4 = pack_q4
        self.fr_off = fr_off
        self.fr_ids_off = fr_ids_off
        self.fr_k = fr_k
        self.e = len(self.off) - (1 if have_q4_draft else 0) - (2 if fr_k > 0 else 0) - 15


def load_pack(ctx: DeviceContext, packdir: String) raises -> Pack:
    var PACK = packdir + "/pack.bin"
    # --- offset table from the pack index (tools/engine-pack.py order) ----
    var off = List[Int]()
    var total = 0
    var q4_off = 0
    var have_q4_draft = False
    var pack_q4 = False
    var fr_off = 0
    var fr_ids_off = 0
    var fr_k = 0
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
                if String(parts[0]) == "output.weight.frdraft":
                    # trailing entry (tools/fr-draft.mojo): FR-Spec reduced draft head
                    fr_off = Int(parts[2])
                elif String(parts[0]) == "output.weight.q4draft":
                    # trailing entry (tools/engine-pack.py --q4-draft): the draft
                    # head's own q4 copy of output.weight, appended after the
                    # trunk order -- excluded from the blk.32 index math below.
                    q4_off = Int(parts[2])
                    have_q4_draft = True
                else:
                    # --q4 pack: every 2D trunk weight is ggml Q4_0
                    pack_q4 = True
                total += n // 2 + (n // 32) * 2
            elif dt == "i32" and String(parts[0]) == "frdraft.ids":
                fr_ids_off = Int(parts[2])
                fr_k = n
                total += n * 4
            else:
                raise Error("unknown pack dtype " + dt)

    # --- load pack into one device buffer -----------------------------------
    print("loading pack:", total, "bytes")
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    comptime RSPLIT = 4
    var stage0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var stage1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t_load = perf_counter_ns()
    with open(PACK, "r") as f:
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
                    var n = external_call["pread", c_ssize_t](
                        fd, sptr.unsafe_offset(got), hi - got, Int64(done + got)
                    )
                    if n <= 0:
                        rerr_ptr[unsafe_offset=t] = 1
                        return
                    got += Int(n)
                rerr_ptr[unsafe_offset=t] = 0
            parallelize(rchunk, RSPLIT)
            for t in range(RSPLIT):
                if rerr[t] != 0:
                    raise Error("short read")
            # The in-flight copy sources the OTHER stage: sync only after this
            # read has overlapped it, and always before this stage is enqueued.
            ctx.synchronize()
            var dslice = DeviceBuffer[DType.uint8](
                ctx, wbuf.unsafe_ptr().unsafe_offset(done), want, owning=False
            )
            var hslice = stage.create_sub_buffer[DType.uint8](0, want) if want != CHUNK else stage
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            done += want
            flip = not flip
    ctx.synchronize()
    print("pack loaded in", Float64(perf_counter_ns() - t_load) / 1e9, "s")
    return Pack(wbuf^, off^, total, q4_off, have_q4_draft, pack_q4, fr_off, fr_ids_off, fr_k)


def alloc_bufs(ctx: DeviceContext, pack: Pack, tmax: Int) raises -> WindowBufs:
    var wbuf = pack.wbuf
    var off = pack.off.copy()
    var tpages = ceildiv(tmax, KVPAGE)
    var kvpool = tpages * N_ATT * NKVH * KVHSTR
    var kvpool1 = tpages * NKVH * KVHSTR
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    var win_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)

    # --- activations / state -------------------------------------------------
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
    var mh_d = ctx.enqueue_create_buffer[f32](H)
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
    ctx.synchronize()






    var stream_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    return WindowBufs(wbuf=wbuf.copy(), off=off.copy(), dtok_h=dtok_h.copy(), win_h=win_h.copy(), x_d=x_d.copy(), curb_d=curb_d.copy(), qkv_d=qkv_d.copy(), z_d=z_d.copy(), eg_d=eg_d.copy(), beta_d=beta_d.copy(), conv_d=conv_d.copy(), so_d=so_d.copy(), resb_d=resb_d.copy(), qf_d=qf_d.copy(), q_d=q_d.copy(), gate_d=gate_d.copy(), k_d=k_d.copy(), v_d=v_d.copy(), ao_d=ao_d.copy(), fgb_d=fgb_d.copy(), aq_d=aq_d.copy(), asc_d=asc_d.copy(), logits_d=logits_d.copy(), toks_d=toks_d.copy(), hn_d=hn_d.copy(), de_d=de_d.copy(), hd_d=hd_d.copy(), cc_d=cc_d.copy(), dtok_d=dtok_d.copy(), p_qf_d=p_qf_d.copy(), p_h_d=p_h_d.copy(), p_kv_d=p_kv_d.copy(), p_32_d=p_32_d.copy(), p_32b_d=p_32b_d.copy(), p_ffn_d=p_ffn_d.copy(), p_ffn2_d=p_ffn2_d.copy(), p_v_d=p_v_d.copy(), xp_d=xp_d.copy(), curbp_d=curbp_d.copy(), qkvp_d=qkvp_d.copy(), zp_d=zp_d.copy(), arp_d=arp_d.copy(), brp_d=brp_d.copy(), egp_d=egp_d.copy(), betap_d=betap_d.copy(), convp_d=convp_d.copy(), sop_d=sop_d.copy(), resbp_d=resbp_d.copy(), qfp_d=qfp_d.copy(), qp_d=qp_d.copy(), gatep_d=gatep_d.copy(), kp_d=kp_d.copy(), vp_d=vp_d.copy(), aop_d=aop_d.copy(), gp_d=gp_d.copy(), up_d=up_d.copy(), fgbp_d=fgbp_d.copy(), convstate_d=convstate_d.copy(), sstate_d=sstate_d.copy(), kvpool=kvpool, kc_d=kc_d.copy(), vc_d=vc_d.copy(), kc32_d=kc32_d.copy(), vc32_d=vc32_d.copy(), off_d=off_d.copy(), araw_d=araw_d.copy(), braw_d=braw_d.copy(), ctr_d=ctr_d.copy(), prof_d=prof_d.copy(), dbg_d=dbg_d.copy(), hmax_d=hmax_d.copy(), hidx_d=hidx_d.copy(), dump_h=dump_h.copy(), stream_h=stream_h.copy())
