from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.math import log
from std.os import getenv
from std.sys import has_accelerator, get_defined_int
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from attn import KVQ, KVT, KVPAGE, KVPAD
from elementwise import amar_rmsnorm_cast, amar_tok_copy
from matmul_skinny import ROW_WAVES, ROW_THREADS
from tokenizer import Tokenizer
from minja import render_chat
from sample import amar_sample_row, amar_sample_row_masked, amar_topn_probs, SAMP_THREADS
from serve_proto import read_line, parse_request, default_sample_params, json_key, json_int, parse_schema_field, parse_reasoning_field
from grammar_rt import GrammarRuntime, compile_schema_matcher, reasoning_boundary_observe
from grammar.automaton import Bitset
from grammar.matcher import Matcher
from spark_kernels import (
    amar_embed_lookup_f32, amar_gemv_q8, amar_argmax_part, amar_argmax_final, amar_rope_kv_append, amar_attn_decode_swa_gated, amar_rope_plain, amar_bias_add,
    amar_trace_sum,
)
from profile import (
    H, FFN, VOCAB, N_LAYERS, NQH, NKVH, HD, NORM_EPS, NROT_FULL, BASE_FULL, NROT_SWA, BASE_SWA,
    SWA_WIN, SWA_PERIOD, SWA_FULL_PHASE, ROPE_NEOX, QKV_BIAS, HAS_GATE, ATTN_SCALE,
    ACTIVATION_GELU, TMAX,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime QDIM = NQH * HD
comptime KVDIM = NKVH * HD
comptime QKV = QDIM + 2 * KVDIM
comptime KVHSTR = KVPAGE * HD + KVPAD
comptime TPAGES = TMAX // KVPAGE
comptime KVPOOL = TPAGES * N_LAYERS * NKVH * KVHSTR
# Per-layer pack order: attn_norm, attn_qkv, [attn_qkv_bias], [attn_gate], attn_output,
# ffn_norm, ffn_gate, ffn_up, ffn_down -- the two bracketed tensors are present only when
# the model's recipe calls for them (spark2_5 has the gate, none of the others do).
comptime OFF_QKV_BIAS = 2 if QKV_BIAS else -1
comptime GATE_BASE = 2 + (1 if QKV_BIAS else 0)
comptime OFF_GATE = GATE_BASE if HAS_GATE else -1
comptime OFF_O = GATE_BASE + (1 if HAS_GATE else 0)
comptime OFF_FFN_NORM = OFF_O + 1
comptime OFF_FFN_GATE = OFF_O + 2
comptime OFF_FFN_UP = OFF_O + 3
comptime OFF_DOWN = OFF_O + 4
comptime LSTRIDE = OFF_O + 5
# -D BARO_TRACE_SUM=1 (diagnostic build, bench/p4-trace-soak.sh): one FNV checksum per
# (position, layer, stage) written on the device with no host sync, dumped per request to
# BARO_TRACE_SUM_DIR. Stages per layer: 0 = QKV after bias, 1 = residual after attention,
# 2 = residual after FFN; the last slot of a position is the logits row. Two runs of the same
# prompt must produce the same table, and the first differing cell names the kernel group
# that produced a transient wrong value. Off (the default) compiles to nothing.
comptime TRACE_SUM = get_defined_int["BARO_TRACE_SUM", 0]() == 1
comptime TRACE_STRIDE = 3 * N_LAYERS + 1

comptime x_l = row_major[1, H]()
comptime xb_l = row_major[1, H]()
comptime h_l = row_major[H]()
comptime toks_l = row_major[TMAX]()
comptime emb_l = row_major[VOCAB, H]()
comptime p_qkv_l = row_major[1, 1, QKV]()
comptime qkv_l = row_major[1, QKV]()
comptime qkv1_l = row_major[QKV]()
comptime h1_l = row_major[H]()
comptime ffn1_l = row_major[FFN]()
comptime v1_l = row_major[VOCAB]()
comptime dummy_l = row_major[1]()
comptime aob2_l = row_major[NQH, HD]()
comptime q_l = row_major[NQH, HD]()
comptime kv_l = row_major[NKVH, HD]()
comptime cache_l = row_major[KVPOOL]()
comptime aob_l = row_major[1, QDIM]()
comptime p_gate_l = row_major[1, 1, NQH]()
comptime gate2_l = row_major[1, NQH]()
comptime gate_l = row_major[NQH]()
comptime p_h_l = row_major[1, 1, H]()
comptime fgb_l = row_major[1, FFN]()
comptime p_v_l = row_major[1, 1, VOCAB]()
comptime AM_NB = 128
comptime amv_l = row_major[AM_NB]()

comptime q_qkv = row_major[QKV, H]()
comptime s_qkv = row_major[QKV, H // 32]()
comptime q_gate = row_major[NQH, H]()
comptime s_gate = row_major[NQH, H // 32]()
comptime q_o = row_major[H, QDIM]()
comptime s_o = row_major[H, QDIM // 32]()
comptime q_ffn = row_major[FFN, H]()
comptime s_ffn = row_major[FFN, H // 32]()
comptime q_down = row_major[H, FFN]()
comptime s_down = row_major[H, FFN // 32]()
comptime q_out = row_major[VOCAB, H]()
comptime s_out = row_major[VOCAB, H // 32]()

comptime k_emb = amar_embed_lookup_f32[type_of(emb_l), type_of(x_l), type_of(toks_l)]
comptime k_rms = amar_rmsnorm_cast[type_of(x_l), type_of(h_l), type_of(xb_l)]
comptime k_qkv = amar_gemv_q8[0, type_of(xb_l), type_of(q_qkv), type_of(s_qkv), type_of(qkv1_l), type_of(dummy_l)]
comptime k_bias = amar_bias_add[type_of(qkv1_l), type_of(qkv1_l)]
comptime k_rope_full = amar_rope_plain[NROT_FULL, type_of(q_l), NEOX=ROPE_NEOX]
comptime k_rope_swa = amar_rope_plain[NROT_SWA, type_of(q_l), NEOX=ROPE_NEOX]
comptime k_kv_full = amar_rope_kv_append[NROT_FULL, N_LAYERS, type_of(cache_l), type_of(kv_l), HD, NKVH, NEOX=ROPE_NEOX]
comptime k_kv_swa = amar_rope_kv_append[NROT_SWA, N_LAYERS, type_of(cache_l), type_of(kv_l), HD, NKVH, NEOX=ROPE_NEOX]
comptime k_att = amar_attn_decode_swa_gated[type_of(q_l), type_of(cache_l), type_of(gate_l), type_of(aob2_l), N_LAYERS, HD, NQH, NKVH, HAS_GATE=HAS_GATE]
comptime k_gate = amar_gemv_q8[0, type_of(xb_l), type_of(q_gate), type_of(s_gate), type_of(gate_l), type_of(dummy_l)]
comptime k_o = amar_gemv_q8[1, type_of(aob_l), type_of(q_o), type_of(s_o), type_of(h1_l), type_of(dummy_l)]
comptime k_ffn_gate = amar_gemv_q8[0, type_of(xb_l), type_of(q_ffn), type_of(s_ffn), type_of(ffn1_l), type_of(dummy_l)]
comptime k_ffn_up = amar_gemv_q8[2 if ACTIVATION_GELU else 3, type_of(xb_l), type_of(q_ffn), type_of(s_ffn), type_of(ffn1_l), type_of(ffn1_l)]
comptime k_down = amar_gemv_q8[1, type_of(fgb_l), type_of(q_down), type_of(s_down), type_of(h1_l), type_of(dummy_l)]
comptime k_head = amar_gemv_q8[0, type_of(xb_l), type_of(q_out), type_of(s_out), type_of(v1_l), type_of(dummy_l)]
comptime k_argmax = amar_argmax_part[AM_NB, type_of(v1_l), type_of(amv_l), type_of(amv_l)]
comptime k_argmax_final = amar_argmax_final[AM_NB, type_of(amv_l), type_of(amv_l), type_of(toks_l)]
# Item 2, briefs/2026-09-16-sampling-all-models-lane.md: amar_sample_row at
# this profile's VOCAB, instantiated the same way registry.mojo's
# sample_row_1 already is for the dense/MoE engine. samp_x_l is a 2-D
# [1, VOCAB] view of the same v1_l-shaped logits_d buffer the argmax path
# reads (sample_row_body requires X.flat_rank == 2); samp_o_l is the
# 1-element scratch the token id and its probability land in before
# amar_tok_copy moves the id into Toks, the same indirection registry.mojo
# uses via amar_tok_copy for the megakernel window.
comptime samp_x_l = row_major[1, VOCAB]()
comptime samp_o_l = row_major[1]()
comptime k_sample = amar_sample_row[type_of(samp_x_l), type_of(samp_o_l), type_of(samp_o_l)]
comptime k_sample_masked = amar_sample_row_masked[type_of(samp_x_l), type_of(samp_o_l), type_of(samp_o_l)]
comptime k_tokcp = amar_tok_copy[type_of(samp_o_l), type_of(toks_l)]
comptime NTOPLP = 20
comptime topn_l = row_major[1, NTOPLP]()
comptime k_topn = amar_topn_probs[type_of(samp_x_l), type_of(topn_l), type_of(topn_l)]


def wq[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[DType.int8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.int8](ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.int8]](), n, owning=False)
    return rebind[TileTensor[DType.int8, LT, MutAnyOrigin]](TileTensor(b, lt))


def ws[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](ctx, (wbuf.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](TileTensor(b, lt))


def wf[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[f32]](), n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(b, lt))


def sub_f32[LT: TensorLayout](ctx: DeviceContext, b: DeviceBuffer[f32], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var v = DeviceBuffer[f32](ctx, b.unsafe_ptr() + o, n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(v, lt))


def load_pack(ctx: DeviceContext, packdir: String, mut off: List[Int]) raises -> DeviceBuffer[DType.uint8]:
    var total = 0
    with open(packdir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var n = Int(parts[3])
            var dt = String(parts[1])
            off.append(Int(parts[2]))
            if dt == "f32":
                total += n * 4
            elif dt == "q8":
                total += n + (n // 32) * 2
            else:
                raise Error("bad dtype " + dt)
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    comptime RSPLIT = 4
    var stage0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var stage1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t_load = perf_counter_ns()
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
    print("pack loaded", total, "bytes in", Float64(perf_counter_ns() - t_load) / 1e9, "s")
    return wbuf^


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


def tok_line(id: Int, tok: Int) -> String:
    return String("{\"id\":") + String(id) + ",\"tok\":" + String(tok) + "}"


def err_line(id: Int, msg: String) -> String:
    return String("{\"id\":") + String(id) + ",\"error\":\"" + msg + "\"}"


# Export-only state file for the llama.cpp bridge (P1 item 4): serve/engine.mojo's
# BAROST01 with conv_n = ssm_n = 0 and a zero salt, the first ceil(pos/KVPAGE) pages
# of the page-major pool. Spark keeps the identity page table and has no state_load.
def save_kv_state(ctx: DeviceContext, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT], path: String, prompt: List[Int], pos: Int) raises:
    comptime if KVT != DType.float32:
        raise Error("state_save: state files store f32 KV; this engine has BARO_KVQ=" + KVQ)
    else:
        var kvn = ceildiv(pos, KVPAGE) * N_LAYERS * NKVH * KVHSTR
        var kh = ctx.enqueue_create_host_buffer[KVT](kvn)
        var vh = ctx.enqueue_create_host_buffer[KVT](kvn)
        ctx.enqueue_copy(dst_buf=kh, src_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr(), kvn, owning=False))
        ctx.enqueue_copy(dst_buf=vh, src_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr(), kvn, owning=False))
        ctx.synchronize()
        var head = List[UInt8]()
        var magic = String("BAROST01")
        for i in range(8):
            head.append(magic.as_bytes()[i])
        for v in [pos, 0, 0, kvn]:
            for b in range(8):
                head.append(UInt8((v >> (8 * b)) & 0xFF))
        for _ in range(32):
            head.append(0)
        for t in range(pos):
            for b in range(4):
                head.append(UInt8((prompt[t] >> (8 * b)) & 0xFF))
        with open(path, "w") as f:
            f.write_bytes(Span(head))
            f.write_bytes(Span[UInt8](unsafe_ptr=kh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
            f.write_bytes(Span[UInt8](unsafe_ptr=vh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
        print("state saved:", path, " pos", pos, " kv pages", ceildiv(pos, KVPAGE), " format BAROST01 (kv only)")


def main() raises:
    comptime if KVQ != "f32":
        raise Error("the spark profile keeps f32 KV; built with BARO_KVQ=" + KVQ)
    comptime assert has_accelerator(), "GPU required"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/spark/pack-q8")
    var serve = getenv("BARO_SERVE", "0") == "1"
    print("BARO_SERVE:", serve)
    # Item 2, briefs/2026-09-16-sampling-all-models-lane.md: dumps the last
    # decode step's target row (same buffer amar_argmax_part / amar_sample_row
    # read from), one profile-agnostic capture path for the distribution test.
    # Off by default, no effect on any existing path.
    var dump_logits_path = getenv("BARO_DUMP_LOGITS", "")
    var force = List[Int]()
    var force_path = getenv("BARO_FORCE", "")
    if force_path != "":
        force = read_prompt(force_path)

    var off = List[Int]()
    var wbuf = load_pack(ctx, packdir, off)

    # Buffers allocated once (serve/PROTOCOL.md); a served request refills
    # toks_h/toks_d from position 0 and every kernel below only ever reads a
    # position it has itself just written this request, so no explicit KV
    # cache clear is needed between requests -- position 0 always overwrites
    # whatever an earlier request left at that slot.
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var tok1_h = ctx.enqueue_create_host_buffer[DType.int32](1)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var xb_d = ctx.enqueue_create_buffer[bf16](H)
    var qkv_d = ctx.enqueue_create_buffer[f32](QKV)
    var kc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var vc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var kvtab_h = ctx.enqueue_create_host_buffer[DType.int32](TPAGES)
    for i in range(TPAGES):
        kvtab_h[i] = Int32(i)
    var kvtab_d = ctx.enqueue_create_buffer[DType.int32](TPAGES)
    ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)
    var aob_d = ctx.enqueue_create_buffer[bf16](QDIM)
    var gate_d = ctx.enqueue_create_buffer[f32](NQH)
    var p_g_d = ctx.enqueue_create_buffer[f32](FFN)
    var fgb_d = ctx.enqueue_create_buffer[bf16](FFN)
    var dummy_d = ctx.enqueue_create_buffer[bf16](1)
    var logits_d = ctx.enqueue_create_buffer[f32](VOCAB)
    var pred_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var trace_n = TMAX * TRACE_STRIDE if TRACE_SUM else 1
    var trace_d = ctx.enqueue_create_buffer[DType.uint32](trace_n)
    var trace_h = ctx.enqueue_create_host_buffer[DType.uint32](trace_n)
    var trace_dir = getenv("BARO_TRACE_SUM_DIR", "")
    ctx.synchronize()

    var Emb = wf(ctx, wbuf, off[0], VOCAB * H, emb_l)
    var X = TileTensor(x_d, x_l)
    var Xb = TileTensor(xb_d, xb_l)
    var Toks = TileTensor(toks_d, toks_l)
    var Pred = TileTensor(pred_d, toks_l)
    var Qkv1 = TileTensor(qkv_d, qkv1_l)
    var X1 = TileTensor(x_d, h1_l)
    var Dummy = TileTensor(dummy_d, dummy_l)
    var Q = sub_f32(ctx, qkv_d, 0, QDIM, q_l)
    var K = sub_f32(ctx, qkv_d, QDIM, KVDIM, kv_l)
    var V = sub_f32(ctx, qkv_d, QDIM + KVDIM, KVDIM, kv_l)
    var Kc = TileTensor(kc_d, cache_l)
    var Vc = TileTensor(vc_d, cache_l)
    var AoB = TileTensor(aob_d, aob_l)
    var AoB2 = TileTensor(aob_d, aob2_l)
    var G1 = TileTensor(p_g_d, ffn1_l)
    var Fgb1 = TileTensor(fgb_d, ffn1_l)
    var Gate = TileTensor(gate_d, gate_l)
    var Fgb = TileTensor(fgb_d, fgb_l)
    var Logits1 = TileTensor(logits_d, v1_l)
    var LogitsSample = TileTensor(logits_d, samp_x_l)
    var amv_d = ctx.enqueue_create_buffer[f32](AM_NB)
    var ami_d = ctx.enqueue_create_buffer[DType.int32](AM_NB)
    var Amv = TileTensor(amv_d, amv_l)
    var Ami = TileTensor(ami_d, amv_l)
    var stok_d = ctx.enqueue_create_buffer[DType.int32](1)
    var sprob_d = ctx.enqueue_create_buffer[f32](1)
    var Stok = TileTensor(stok_d, samp_o_l)
    var Sprob = TileTensor(sprob_d, samp_o_l)
    var sprob_h = ctx.enqueue_create_host_buffer[f32](1)
    var topn_ids_d = ctx.enqueue_create_buffer[DType.int32](NTOPLP)
    var topn_probs_d = ctx.enqueue_create_buffer[f32](NTOPLP)
    var topn_ids_h = ctx.enqueue_create_host_buffer[DType.int32](NTOPLP)
    var topn_probs_h = ctx.enqueue_create_host_buffer[f32](NTOPLP)
    var gmask_h = ctx.enqueue_create_host_buffer[DType.uint64]((VOCAB + 63) // 64)
    var gmask_d = ctx.enqueue_create_buffer[DType.uint64]((VOCAB + 63) // 64)
    var row_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
    var dump_dir = getenv("BARO_DUMP_LOGITS_DIR", "")
    var OutNorm = wf(ctx, wbuf, off[1 + LSTRIDE * N_LAYERS], H, h_l)
    var out_off = off[2 + LSTRIDE * N_LAYERS]
    var Woq = wq(ctx, wbuf, out_off, VOCAB * H, q_out)
    var Wos = ws(ctx, wbuf, out_off, VOCAB * H, s_out)
    var grt: Optional[GrammarRuntime] = None

    if serve:
        print("{\"ready\":true,\"tmax\":" + String(TMAX) + ",\"mrows\":1,\"kmax\":0,\"spec_k\":0,\"pack\":\"" + packdir + "\"}")

    # No cancel support this milestone (M4, briefs/2026-09-15-wiring-lane.md):
    # spark has no draft head and no spec decode, so a request always runs to
    # completion quickly; {"cancel":ID} lines are not sent by a client that
    # never streams past this engine's own doneness. A future round wanting
    # cancel reuses serve_proto.cancel_pending exactly as engine.mojo does.
    var req_id = 0
    while True:
        var prompt: List[Int]
        var gen_n = atol(getenv("BARO_GEN", "64"))
        var stop_seqs = List[List[Int]]()
        # Item 2, briefs/2026-09-16-sampling-all-models-lane.md: declared
        # outside the branch (engine.mojo's own one-shot path does the same)
        # so the decode loop below can read sample.temperature regardless of
        # mode; one-shot stays the default (temperature 0, greedy), matching
        # today's behaviour exactly.
        var sample = default_sample_params()
        var grammar: Optional[Matcher] = None
        var grammar_pending_think = False
        var grammar_think_buf = List[UInt8]()
        var grammar_stop = False
        var grammar_mask = Bitset(1)
        var sp_state_save = String("")
        if serve:
            var line_in = read_line(0)
            if not line_in:
                break
            prompt = List[Int]()
            var req_n = 0
            var req_spec = False
            var req_has_spec = False
            var ckpt_hints = List[Int]()
            var sp_state_load = String("")
            var perr = parse_request(line_in.value(), req_id, prompt, req_n, req_spec, req_has_spec, stop_seqs, ckpt_hints, sample, sp_state_save, sp_state_load)
            if perr == "" and len(prompt) < 1:
                perr = "empty prompt"
            if perr == "" and req_n < 1:
                perr = "n must be >= 1"
            if perr == "" and len(prompt) + req_n > TMAX:
                perr = "prompt+n exceeds TMAX " + String(TMAX)
            if perr == "" and sp_state_load != "":
                perr = "state_load is not wired for this engine (spark); its state files are export-only (tools/state-to-llama-slot)"
            if perr == "" and sp_state_save != "" and len(prompt) < 2:
                perr = "state_save needs a prompt of at least 2 tokens"
            # Items 3-4, briefs/2026-09-16-sampling-all-models-lane.md: spark
            # parses presence_penalty/frequency_penalty/top_logprobs (M5/C3)
            # but has never acted on them -- silently ignoring a parameter
            # the caller asked for is the inert-parameter defect P1 forbids,
            # so refuse loudly rather than serve a request that looks
            # penalized/logprob'd and isn't, until this is wired here too.
            if perr == "" and (sample.presence_penalty != 0 or sample.frequency_penalty != 0):
                perr = "presence_penalty/frequency_penalty are not yet wired for this engine (spark); only the dense/MoE engine (serve/engine.mojo) supports them"
            if perr == "" and sample.embed == 1:
                perr = "embed is not wired for this engine (spark); only the dense engine (serve/engine.mojo) emits it"
            if perr == "" and (sample.hidden == 1 or sample.logits_topk > 0):
                perr = "hidden/logits_topk are not wired for this engine (spark); use the dense engine (serve/engine.mojo)"
            var schema_raw = parse_schema_field(line_in.value()) if perr == "" else String("")
            if perr == "" and schema_raw != "":
                sample.top_p = 1.0
                sample.top_k = 0
                sample.min_p = 0.0
                if not grt:
                    try:
                        grt = Optional(GrammarRuntime(packdir))
                    except e:
                        perr = "response_format: grammar runtime failed to load: " + String(e)
                if perr == "" and grt.value().vocab_size != VOCAB:
                    perr = "response_format: grammar vocab_size " + String(grt.value().vocab_size) + " != engine VOCAB " + String(VOCAB)
                if perr == "":
                    try:
                        grammar = Optional(compile_schema_matcher(grt.value(), schema_raw))
                    except e:
                        perr = "response_format schema: " + String(e)
                    grammar_pending_think = parse_reasoning_field(line_in.value())
                    grammar_mask = Bitset(VOCAB)
            if perr != "":
                print(err_line(req_id, perr))
                continue
            gen_n = req_n
            # Per-request teacher forcing, the same "force":[ids] field
            # serve/engine.mojo reads; absent, the request inherits BARO_FORCE.
            force = read_prompt(force_path) if force_path != "" else List[Int]()
            var fi2 = json_key(line_in.value(), "force")
            if fi2 >= 0:
                var fb = line_in.value().as_bytes()
                if fi2 < len(fb) and fb[fi2] == 91:
                    fi2 += 1
                    while True:
                        while fi2 < len(fb) and (fb[fi2] == 32 or fb[fi2] == 44):
                            fi2 += 1
                        if fi2 >= len(fb) or fb[fi2] == 93:
                            break
                        var fv = 0
                        if not json_int(line_in.value(), fi2, fv):
                            break
                        force.append(fv)
        else:
            var text_path = getenv("BARO_PROMPT_TEXT", "")
            var chat_path = getenv("BARO_CHAT", "")
            if text_path != "" or chat_path != "":
                var gguf = getenv("BARO_GGUF", "")
                if gguf == "":
                    raise Error("BARO_PROMPT_TEXT / BARO_CHAT need BARO_GGUF (tokenizer + template source)")
                var t0 = perf_counter_ns()
                var tok = Tokenizer(gguf)
                var text: String
                if chat_path != "":
                    var case_json: String
                    with open(chat_path, "r") as f:
                        case_json = f.read()
                    text = render_chat(tok.chat_template, case_json, tok.token_str(tok.bos_id), tok.token_str(tok.eos_id), tok.token_str(tok.pad_id))
                    print("chat template rendered:", text.byte_length(), "bytes (mojo-minja)")
                else:
                    with open(text_path, "r") as f:
                        text = f.read()
                prompt = tok.encode(text)
                print("tokenized", len(prompt), "ids in", Float64(perf_counter_ns() - t0) / 1e9, "s (mojo tokenizer,", tok.pre, ")")
                var ps = String()
                for i in range(len(prompt)):
                    ps += String(prompt[i]) + " "
                print("prompt ids:", ps)
            else:
                prompt = read_prompt(getenv("BARO_PROMPT", packdir + "/prompt-tokens.txt"))
        var n_prompt = len(prompt)
        var n_total = n_prompt + gen_n
        if n_total > TMAX:
            raise Error("prompt + gen exceeds TMAX")
        print("prompt tokens:", n_prompt, "gen:", gen_n)

        for i in range(TMAX):
            toks_h[i] = Int32(prompt[i]) if i < n_prompt else Int32(0)
        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
        comptime if TRACE_SUM:
            for i in range(trace_n):
                trace_h[i] = 0
            ctx.enqueue_copy(dst_buf=trace_d, src_buf=trace_h)

        var t_gen_start: Int = 0
        var generated_ids = List[Int]()
        var finish = String("length")
        ctx.synchronize()
        var t_pf_start = perf_counter_ns()
        for pos in range(n_total - 1):
            if pos == n_prompt - 1:
                ctx.synchronize()
                t_gen_start = perf_counter_ns()
                print("prefill_s:", Float64(t_gen_start - t_pf_start) / 1e9, " prefill rows:", n_prompt - 1, " chunk: 0")
            ctx.enqueue_function[k_emb](Emb, X, Toks, Int32(pos), Int32(H), grid_dim=(ceildiv(H, 256), 1), block_dim=256)
            for i in range(N_LAYERS):
                var e = 1 + LSTRIDE * i
                var swa = (i % SWA_PERIOD) != SWA_FULL_PHASE
                var AttnNorm = wf(ctx, wbuf, off[e], H, h_l)
                var FfnNorm = wf(ctx, wbuf, off[e + OFF_FFN_NORM], H, h_l)
                ctx.enqueue_function[k_rms](X, AttnNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                ctx.enqueue_function[k_qkv](Xb, wq(ctx, wbuf, off[e + 1], QKV * H, q_qkv), ws(ctx, wbuf, off[e + 1], QKV * H, s_qkv), Qkv1, Dummy, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, ROW_WAVES), block_dim=ROW_THREADS)
                comptime if QKV_BIAS:
                    var QkvBias = wf(ctx, wbuf, off[e + OFF_QKV_BIAS], QKV, qkv1_l)
                    ctx.enqueue_function[k_bias](Qkv1, QkvBias, Int32(QKV), grid_dim=ceildiv(QKV, 256), block_dim=256)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](qkv_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(QKV), Int32(pos * TRACE_STRIDE + 3 * i), grid_dim=1, block_dim=1)
                if swa:
                    ctx.enqueue_function[k_rope_swa](Q, Int32(pos), Int32(NQH), BASE_SWA, grid_dim=(NQH, 1), block_dim=NROT_SWA // 2)
                    ctx.enqueue_function[k_kv_swa](Kc, Vc, K, V, kvtab_d.unsafe_ptr(), Int32(pos), BASE_SWA, Int32(i), grid_dim=(NKVH, 2), block_dim=HD)
                else:
                    ctx.enqueue_function[k_rope_full](Q, Int32(pos), Int32(NQH), BASE_FULL, grid_dim=(NQH, 1), block_dim=NROT_FULL // 2)
                    ctx.enqueue_function[k_kv_full](Kc, Vc, K, V, kvtab_d.unsafe_ptr(), Int32(pos), BASE_FULL, Int32(i), grid_dim=(NKVH, 2), block_dim=HD)
                comptime if HAS_GATE:
                    ctx.enqueue_function[k_gate](Xb, wq(ctx, wbuf, off[e + OFF_GATE], NQH * H, q_gate), ws(ctx, wbuf, off[e + OFF_GATE], NQH * H, s_gate), Gate, Dummy, Int32(NQH), Int32(H), grid_dim=ceildiv(NQH, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[k_att](Q, Kc, Vc, Gate, AoB2, kvtab_d.unsafe_ptr(), Int32(pos + 1), Int32(SWA_WIN if swa else 0), ATTN_SCALE, Int32(i), grid_dim=(NQH, 1), block_dim=HD)
                ctx.enqueue_function[k_o](AoB, wq(ctx, wbuf, off[e + OFF_O], H * QDIM, q_o), ws(ctx, wbuf, off[e + OFF_O], H * QDIM, s_o), X1, Dummy, Int32(H), Int32(QDIM), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](x_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(H), Int32(pos * TRACE_STRIDE + 3 * i + 1), grid_dim=1, block_dim=1)
                ctx.enqueue_function[k_rms](X, FfnNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                ctx.enqueue_function[k_ffn_gate](Xb, wq(ctx, wbuf, off[e + OFF_FFN_GATE], FFN * H, q_ffn), ws(ctx, wbuf, off[e + OFF_FFN_GATE], FFN * H, s_ffn), G1, Dummy, Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[k_ffn_up](Xb, wq(ctx, wbuf, off[e + OFF_FFN_UP], FFN * H, q_ffn), ws(ctx, wbuf, off[e + OFF_FFN_UP], FFN * H, s_ffn), G1, Fgb1, Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[k_down](Fgb, wq(ctx, wbuf, off[e + OFF_DOWN], H * FFN, q_down), ws(ctx, wbuf, off[e + OFF_DOWN], H * FFN, s_down), X1, Dummy, Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](x_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(H), Int32(pos * TRACE_STRIDE + 3 * i + 2), grid_dim=1, block_dim=1)
            if pos >= n_prompt - 1:
                ctx.enqueue_function[k_rms](X, OutNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                ctx.enqueue_function[k_head](Xb, Woq, Wos, Logits1, Dummy, Int32(VOCAB), Int32(H), grid_dim=ceildiv(VOCAB, ROW_WAVES), block_dim=ROW_THREADS)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](logits_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(VOCAB), Int32(pos * TRACE_STRIDE + 3 * N_LAYERS), grid_dim=1, block_dim=1)
                if dump_logits_path != "" and pos == n_total - 2:
                    var lg_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
                    ctx.enqueue_copy(
                        dst_buf=lg_h,
                        src_buf=DeviceBuffer[f32](ctx, logits_d.unsafe_ptr(), VOCAB, owning=False),
                    )
                    ctx.synchronize()
                    with open(dump_logits_path, "w") as f:
                        var p = lg_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=VOCAB * 4))
                    print("dumped final logits row to", dump_logits_path)
                # Item 2, briefs/2026-09-16-sampling-all-models-lane.md: at
                # temperature > 0 amar_sample_row replaces the argmax pair;
                # at 0 (including one-shot, which never sets sample) the
                # path below is untouched, byte for byte.
                var want_lp = serve and sample.top_logprobs > 0
                if want_lp and dump_dir != "":
                    ctx.enqueue_copy(dst_buf=row_h, src_buf=DeviceBuffer[f32](ctx, logits_d.unsafe_ptr(), VOCAB, owning=False))
                    ctx.synchronize()
                    with open(dump_dir + "/row-" + String(pos + 1 - n_prompt) + ".bin", "w") as f:
                        f.write_bytes(Span[UInt8](unsafe_ptr=row_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=VOCAB * 4))
                if want_lp:
                    ctx.enqueue_function[k_topn](
                        LogitsSample, TileTensor(topn_ids_d, topn_l), TileTensor(topn_probs_d, topn_l),
                        Int32(VOCAB), Int32(min(sample.top_logprobs, NTOPLP)),
                        Float32(sample.temperature), Int32(sample.top_k), Float32(sample.top_p), Float32(sample.min_p),
                        grid_dim=1, block_dim=SAMP_THREADS,
                    )
                    ctx.enqueue_copy(dst_buf=topn_ids_h, src_buf=topn_ids_d)
                    ctx.enqueue_copy(dst_buf=topn_probs_h, src_buf=topn_probs_d)
                var grammar_here = grammar.__bool__() and pos + 1 >= n_prompt and not grammar_pending_think
                if grammar_here:
                    ref mm = grammar.value()
                    mm.fill_mask(grammar_mask)
                    for i in range(len(grammar_mask.words)):
                        gmask_h[i] = grammar_mask.words[i]
                    ctx.enqueue_copy(dst_buf=gmask_d, src_buf=gmask_h)
                    ctx.enqueue_function[k_sample_masked](
                        LogitsSample, Stok, Sprob, Int32(VOCAB), Float32(sample.temperature), Int32(sample.top_k),
                        Float32(sample.top_p), Float32(sample.min_p), sample.seed, UInt64(pos),
                        gmask_d.unsafe_ptr(), Int32((VOCAB + 63) // 64), grid_dim=1, block_dim=SAMP_THREADS,
                    )
                    ctx.enqueue_function[k_tokcp](Stok, Toks, Int32(0), Int32(pos + 1), Int32(1), grid_dim=1, block_dim=32)
                elif sample.temperature > 0:
                    ctx.enqueue_function[k_sample](
                        LogitsSample, Stok, Sprob, Int32(VOCAB), Float32(sample.temperature), Int32(sample.top_k),
                        Float32(sample.top_p), Float32(sample.min_p), sample.seed, UInt64(pos),
                        grid_dim=1, block_dim=SAMP_THREADS,
                    )
                    ctx.enqueue_function[k_tokcp](Stok, Toks, Int32(0), Int32(pos + 1), Int32(1), grid_dim=1, block_dim=32)
                else:
                    ctx.enqueue_function[k_argmax](Logits1, Amv, Ami, Int32(VOCAB), grid_dim=AM_NB, block_dim=256)
                    var fi = pos + 1 - n_prompt
                    var forced = Int32(force[fi]) if fi < len(force) else Int32(-1)
                    ctx.enqueue_function[k_argmax_final](Amv, Ami, Toks, Pred, Int32(pos + 1), forced, grid_dim=1, block_dim=32)
                if serve:
                    ctx.enqueue_copy(dst_buf=tok1_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr().unsafe_offset(pos + 1), 1, owning=False))
                    ctx.synchronize()
                    var new_tok = Int(tok1_h[0])
                    if grammar.__bool__():
                        ref mm = grammar.value()
                        if new_tok < 0:
                            grammar_stop = True
                        elif grammar_pending_think:
                            if reasoning_boundary_observe(grammar_think_buf, mm.vocab[].token_bytes[new_tok]):
                                grammar_pending_think = False
                        else:
                            _ = mm.accept(new_tok)
                            if mm.is_terminated():
                                grammar_stop = True
                    if grammar_stop and new_tok < 0:
                        finish = "stop"
                        break
                    if want_lp:
                        var lp = 0.0
                        if sample.temperature > 0:
                            ctx.enqueue_copy(dst_buf=sprob_h, src_buf=sprob_d)
                            ctx.synchronize()
                            lp = log(Float64(sprob_h[0])) if sprob_h[0] > 0 else -1e30
                        var tl = String("{\"id\":") + String(req_id) + ",\"tok\":" + String(new_tok) + ",\"logprob\":" + String(lp) + ",\"top_logprobs\":["
                        for i in range(min(sample.top_logprobs, NTOPLP)):
                            var tid = Int(topn_ids_h[i])
                            if tid < 0:
                                break
                            if i > 0:
                                tl += ","
                            var pr = topn_probs_h[i]
                            tl += "{\"id\":" + String(tid) + ",\"logprob\":" + String(log(Float64(pr)) if pr > 0 else -1e30) + "}"
                        tl += "]}"
                        print(tl)
                    else:
                        print(tok_line(req_id, new_tok))
                    generated_ids.append(new_tok)
                    # Checked once per generated token (spark decodes m=1 at a
                    # time, unlike engine.mojo's per-window check): the tail
                    # of what has been generated so far against every stop
                    # sequence; a match ends generation right here.
                    for seq in stop_seqs:
                        if len(seq) > 0 and len(seq) <= len(generated_ids):
                            var matched = True
                            for k in range(len(seq)):
                                if generated_ids[len(generated_ids) - len(seq) + k] != seq[k]:
                                    matched = False
                                    break
                            if matched:
                                finish = "stop"
                    if finish == "stop":
                        break
                    if grammar_stop:
                        finish = "stop"
                        break
        ctx.synchronize()
        var dt = Float64(perf_counter_ns() - t_gen_start) / 1e9
        var n_gen = len(generated_ids) if serve else gen_n
        ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
        ctx.synchronize()
        var s = String("")
        for i in range(n_prompt, n_prompt + n_gen):
            s += String(Int(toks_h[i])) + " "
        print("generated:", s)
        comptime if TRACE_SUM:
            if trace_dir != "":
                ctx.enqueue_copy(dst_buf=trace_h, src_buf=trace_d)
                ctx.synchronize()
                with open(trace_dir + "/trace-" + String(req_id) + ".bin", "w") as f:
                    f.write_bytes(Span[UInt8](unsafe_ptr=trace_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=(n_prompt + n_gen - 1) * TRACE_STRIDE * 4))
                print("trace sums:", trace_dir + "/trace-" + String(req_id) + ".bin", " positions", n_prompt + n_gen - 1, " stride", TRACE_STRIDE)
        if len(force) > 0:
            ctx.enqueue_copy(dst_buf=toks_h, src_buf=pred_d)
            ctx.synchronize()
            var ps = String("")
            var agree = 0
            for i in range(n_prompt, n_prompt + n_gen):
                ps += String(Int(toks_h[i])) + " "
                if i - n_prompt < len(force) and Int(toks_h[i]) == force[i - n_prompt]:
                    agree += 1
            print("predicted:", ps)
            print("forced agreement:", agree, "/", min(n_gen, len(force)))
        print("tok/s_gen:", Float64(n_gen) / dt, "(", n_gen, "steps,", dt, "s )")
        if sp_state_save != "":
            save_kv_state(ctx, kc_d, vc_d, sp_state_save, prompt, n_prompt - 1)
        if serve:
            var tok_s = Float64(n_gen - 1) / dt if n_gen > 1 else 0.0
            print(
                "{\"id\":" + String(req_id) + ",\"done\":true,\"n\":" + String(n_gen)
                + ",\"prefill_s\":" + String(Float64(t_gen_start - t_pf_start) / 1e9)
                + ",\"decode_s\":" + String(dt) + ",\"tok_s\":" + String(tok_s)
                + ",\"finish\":\"" + finish + "\"}"
            )
        else:
            break
