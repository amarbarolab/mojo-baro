from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from attn import KVT, KVPAGE, KVPAD
from elementwise import amar_rmsnorm_cast
from matmul_skinny import ROW_WAVES, ROW_THREADS
from tokenizer import Tokenizer
from minja import render_chat
from spark_kernels import (
    amar_embed_lookup_f32, amar_gemv_q8, amar_argmax_part, amar_argmax_final, amar_rope_kv_append, amar_attn_decode_swa_gated, amar_rope_plain, amar_bias_add,
)
from profile import (
    H, FFN, VOCAB, N_LAYERS, NQH, NKVH, HD, NORM_EPS, NROT_FULL, BASE_FULL, NROT_SWA, BASE_SWA,
    SWA_WIN, SWA_PERIOD, SWA_FULL_PHASE, ROPE_NEOX, QKV_BIAS, ATTN_SCALE, TMAX,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime QDIM = NQH * HD
comptime KVDIM = NKVH * HD
comptime QKV = QDIM + 2 * KVDIM
comptime KVHSTR = KVPAGE * HD + KVPAD
comptime TPAGES = TMAX // KVPAGE
comptime KVPOOL = TPAGES * N_LAYERS * NKVH * KVHSTR
comptime OFF_QKV_BIAS = 2 if QKV_BIAS else -1
comptime OFF_GATE = 3 if QKV_BIAS else 2
comptime OFF_O = 4 if QKV_BIAS else 3
comptime OFF_FFN_NORM = 5 if QKV_BIAS else 4
comptime OFF_FFN_GATE = 6 if QKV_BIAS else 5
comptime OFF_FFN_UP = 7 if QKV_BIAS else 6
comptime OFF_DOWN = 8 if QKV_BIAS else 7
comptime LSTRIDE = 9 if QKV_BIAS else 8

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
comptime k_att = amar_attn_decode_swa_gated[type_of(q_l), type_of(cache_l), type_of(gate_l), type_of(aob2_l), N_LAYERS, HD, NQH, NKVH]
comptime k_gate = amar_gemv_q8[0, type_of(xb_l), type_of(q_gate), type_of(s_gate), type_of(gate_l), type_of(dummy_l)]
comptime k_o = amar_gemv_q8[1, type_of(aob_l), type_of(q_o), type_of(s_o), type_of(h1_l), type_of(dummy_l)]
comptime k_ffn_gate = amar_gemv_q8[0, type_of(xb_l), type_of(q_ffn), type_of(s_ffn), type_of(ffn1_l), type_of(dummy_l)]
comptime k_ffn_up = amar_gemv_q8[2, type_of(xb_l), type_of(q_ffn), type_of(s_ffn), type_of(ffn1_l), type_of(ffn1_l)]
comptime k_down = amar_gemv_q8[1, type_of(fgb_l), type_of(q_down), type_of(s_down), type_of(h1_l), type_of(dummy_l)]
comptime k_head = amar_gemv_q8[0, type_of(xb_l), type_of(q_out), type_of(s_out), type_of(v1_l), type_of(dummy_l)]
comptime k_argmax = amar_argmax_part[AM_NB, type_of(v1_l), type_of(amv_l), type_of(amv_l)]
comptime k_argmax_final = amar_argmax_final[AM_NB, type_of(amv_l), type_of(amv_l), type_of(toks_l)]


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


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/spark/pack-q8")
    var gen_n = atol(getenv("BARO_GEN", "64"))
    var prompt: List[Int]
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

    var off = List[Int]()
    var wbuf = load_pack(ctx, packdir, off)

    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    for i in range(TMAX):
        toks_h[i] = Int32(prompt[i]) if i < n_prompt else Int32(0)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var xb_d = ctx.enqueue_create_buffer[bf16](H)
    var qkv_d = ctx.enqueue_create_buffer[f32](QKV)
    var kc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var vc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var aob_d = ctx.enqueue_create_buffer[bf16](QDIM)
    var gate_d = ctx.enqueue_create_buffer[f32](NQH)
    var p_g_d = ctx.enqueue_create_buffer[f32](FFN)
    var fgb_d = ctx.enqueue_create_buffer[bf16](FFN)
    var dummy_d = ctx.enqueue_create_buffer[bf16](1)
    var logits_d = ctx.enqueue_create_buffer[f32](VOCAB)
    ctx.synchronize()

    var Emb = wf(ctx, wbuf, off[0], VOCAB * H, emb_l)
    var X = TileTensor(x_d, x_l)
    var Xb = TileTensor(xb_d, xb_l)
    var Toks = TileTensor(toks_d, toks_l)
    var pred_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var Pred = TileTensor(pred_d, toks_l)
    var force = List[Int]()
    var force_path = getenv("BARO_FORCE", "")
    if force_path != "":
        force = read_prompt(force_path)
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
    var amv_d = ctx.enqueue_create_buffer[f32](AM_NB)
    var ami_d = ctx.enqueue_create_buffer[DType.int32](AM_NB)
    var Amv = TileTensor(amv_d, amv_l)
    var Ami = TileTensor(ami_d, amv_l)
    var OutNorm = wf(ctx, wbuf, off[1 + LSTRIDE * N_LAYERS], H, h_l)
    var out_off = off[2 + LSTRIDE * N_LAYERS]
    var Woq = wq(ctx, wbuf, out_off, VOCAB * H, q_out)
    var Wos = ws(ctx, wbuf, out_off, VOCAB * H, s_out)

    var t_gen_start: Int = 0
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
            if swa:
                ctx.enqueue_function[k_rope_swa](Q, Int32(pos), Int32(NQH), BASE_SWA, grid_dim=(NQH, 1), block_dim=NROT_SWA // 2)
                ctx.enqueue_function[k_kv_swa](Kc, Vc, K, V, Int32(pos), BASE_SWA, Int32(i), grid_dim=(NKVH, 2), block_dim=HD)
            else:
                ctx.enqueue_function[k_rope_full](Q, Int32(pos), Int32(NQH), BASE_FULL, grid_dim=(NQH, 1), block_dim=NROT_FULL // 2)
                ctx.enqueue_function[k_kv_full](Kc, Vc, K, V, Int32(pos), BASE_FULL, Int32(i), grid_dim=(NKVH, 2), block_dim=HD)
            ctx.enqueue_function[k_gate](Xb, wq(ctx, wbuf, off[e + OFF_GATE], NQH * H, q_gate), ws(ctx, wbuf, off[e + OFF_GATE], NQH * H, s_gate), Gate, Dummy, Int32(NQH), Int32(H), grid_dim=ceildiv(NQH, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_att](Q, Kc, Vc, Gate, AoB2, Int32(pos + 1), Int32(SWA_WIN if swa else 0), ATTN_SCALE, Int32(i), grid_dim=(NQH, 1), block_dim=HD)
            ctx.enqueue_function[k_o](AoB, wq(ctx, wbuf, off[e + OFF_O], H * QDIM, q_o), ws(ctx, wbuf, off[e + OFF_O], H * QDIM, s_o), X1, Dummy, Int32(H), Int32(QDIM), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_rms](X, FfnNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
            ctx.enqueue_function[k_ffn_gate](Xb, wq(ctx, wbuf, off[e + OFF_FFN_GATE], FFN * H, q_ffn), ws(ctx, wbuf, off[e + OFF_FFN_GATE], FFN * H, s_ffn), G1, Dummy, Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_ffn_up](Xb, wq(ctx, wbuf, off[e + OFF_FFN_UP], FFN * H, q_ffn), ws(ctx, wbuf, off[e + OFF_FFN_UP], FFN * H, s_ffn), G1, Fgb1, Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_down](Fgb, wq(ctx, wbuf, off[e + OFF_DOWN], H * FFN, q_down), ws(ctx, wbuf, off[e + OFF_DOWN], H * FFN, s_down), X1, Dummy, Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
        if pos >= n_prompt - 1:
            ctx.enqueue_function[k_rms](X, OutNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
            ctx.enqueue_function[k_head](Xb, Woq, Wos, Logits1, Dummy, Int32(VOCAB), Int32(H), grid_dim=ceildiv(VOCAB, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[k_argmax](Logits1, Amv, Ami, Int32(VOCAB), grid_dim=AM_NB, block_dim=256)
            var fi = pos + 1 - n_prompt
            var forced = Int32(force[fi]) if fi < len(force) else Int32(-1)
            ctx.enqueue_function[k_argmax_final](Amv, Ami, Toks, Pred, Int32(pos + 1), forced, grid_dim=1, block_dim=32)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t_gen_start) / 1e9
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
    ctx.synchronize()
    var s = String("")
    for i in range(n_prompt, n_total):
        s += String(Int(toks_h[i])) + " "
    print("generated:", s)
    if len(force) > 0:
        ctx.enqueue_copy(dst_buf=toks_h, src_buf=pred_d)
        ctx.synchronize()
        var ps = String("")
        var agree = 0
        for i in range(n_prompt, n_total):
            ps += String(Int(toks_h[i])) + " "
            if i - n_prompt < len(force) and Int(toks_h[i]) == force[i - n_prompt]:
                agree += 1
        print("predicted:", ps)
        print("forced agreement:", agree, "/", min(gen_n, len(force)))
    print("tok/s_gen:", Float64(gen_n) / dt, "(", gen_n, "steps,", dt, "s )")
