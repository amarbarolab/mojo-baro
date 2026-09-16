"""baro engine: full-model greedy decode for qwen35 (Qwythos-9B), milestone 4.

Loads .work/engine-pack/ (fixed tensor order, 2D bf16 weights pre-transposed
to B-layout), reads prompt token ids from .work/engine-pack/prompt-tokens.txt,
runs the 32-block hybrid stack (24 gated-delta-net + 8 gated full-attention,
MTP block skipped) over a window of up to MROWS tokens at a time, and prints
greedy token ids.

Parity target: byte-identical token ids vs llama.cpp on the same GGUF.
"""
from std.math import ceildiv
from std.memory import memcpy, unsafe_memcpy
from std.os import getenv
from std.sys import exit, has_accelerator
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *


from window import *
from harness import *
from prefix import *
from moe_pack import parse_moe_index, resolve_plain
from latent import EngineLatentClient
from latentos.ipc import connect_unix_socket
from serve_proto import (
    read_line, cancel_pending, json_key, json_int, json_float,
    SampleParams, default_sample_params, parse_request,
    parse_schema_field, parse_reasoning_field,
)
from grammar_rt import GrammarRuntime, compile_schema_matcher
from grammar.automaton import Bitset




# --- engine state file (LatentOS use 2: a saved prefix survives the process) --
# BAROST01 | int64 pos, conv_n, ssm_n, kv_n | pack salt (32 B) | int32 tokens[pos]
# | f32 conv[conv_n] | f32 ssm[ssm_n] | f32 K[kv_n] | f32 V[kv_n]
# BAROST02 | same header | f32 conv[conv_n] | f32 ssm[ssm_n]
# | f32 Kscale[kv_n/KVHSTR] | int8 K[kv_n] | f32 Vscale[kv_n/KVHSTR] | int8 V[kv_n]
# K/V are the first ceil(pos/KVPAGE) pages of the page-major pool, so every
# position below pos is included; the checkpoint is the one saved at pos.
# BAROST02 quantizes one KVHSTR block (one page, one attention layer, one kv
# head; KVPAD included but always 0 today) per scale, matching the memory
# layout's own addressing unit exactly, so dequant needs no reshaping.
def _put_i64(mut out: List[UInt8], v: Int):
    for b in range(8):
        out.append(UInt8((v >> (8 * b)) & 0xFF))


def _get_i64(data: List[UInt8], off: Int) -> Int:
    var v = 0
    for b in range(8):
        v |= Int(data[off + b]) << (8 * b)
    return v


def save_state(
    ctx: DeviceContext, chain: Chain, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    path: String, prompt: List[Int], pos: Int,
) raises:
    comptime assert KVT == DType.float32, "state file stores f32 KV"
    var idx = -1
    for i in range(len(chain.items)):
        if chain.items[i].valid and chain.items[i].pos == pos:
            idx = i
    if idx < 0:
        raise Error("BARO_STATE_SAVE: no committed checkpoint at pos " + String(pos))
    var kvn = ceildiv(pos, KVPAGE) * N_ATT * NKVH * KVHSTR
    var kh = ctx.enqueue_create_host_buffer[KVT](kvn)
    var vh = ctx.enqueue_create_host_buffer[KVT](kvn)
    ctx.enqueue_copy(dst_buf=kh, src_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr(), kvn, owning=False))
    ctx.enqueue_copy(dst_buf=vh, src_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr(), kvn, owning=False))
    ctx.synchronize()
    var int8 = getenv("BARO_STATE_INT8", "0") == "1"
    var head = List[UInt8]()
    var magic = String("BAROST02" if int8 else "BAROST01")
    for i in range(8):
        head.append(magic.as_bytes()[i])
    _put_i64(head, pos)
    _put_i64(head, CONV_SLOT)
    _put_i64(head, SSM_SLOT)
    _put_i64(head, kvn)
    for b in chain.salt:
        head.append(b)
    for t in range(pos):
        for b in range(4):
            head.append(UInt8((prompt[t] >> (8 * b)) & 0xFF))
    with open(path, "w") as f:
        f.write_bytes(Span(head))
        f.write_bytes(Span[UInt8](unsafe_ptr=chain.items[idx].conv_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=CONV_SLOT * 4))
        f.write_bytes(Span[UInt8](unsafe_ptr=chain.items[idx].ssm_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=SSM_SLOT * 4))
        if int8:
            var ngroups = kvn // KVHSTR
            var kscale = List[Scalar[f32]](unsafe_uninit_length=ngroups)
            var kq = List[Scalar[i8]](unsafe_uninit_length=kvn)
            _quantize_kv_int8(kh, kvn, kscale, kq)
            var vscale = List[Scalar[f32]](unsafe_uninit_length=ngroups)
            var vq = List[Scalar[i8]](unsafe_uninit_length=kvn)
            _quantize_kv_int8(vh, kvn, vscale, vq)
            f.write_bytes(Span[UInt8](unsafe_ptr=kscale.unsafe_ptr().unsafe_bitcast[UInt8](), length=ngroups * 4))
            f.write_bytes(Span[UInt8](unsafe_ptr=kq.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn))
            f.write_bytes(Span[UInt8](unsafe_ptr=vscale.unsafe_ptr().unsafe_bitcast[UInt8](), length=ngroups * 4))
            f.write_bytes(Span[UInt8](unsafe_ptr=vq.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn))
        else:
            f.write_bytes(Span[UInt8](unsafe_ptr=kh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
            f.write_bytes(Span[UInt8](unsafe_ptr=vh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
    print("state saved:", path, " pos", pos, " kv pages", ceildiv(pos, KVPAGE), " format", magic)


def _quantize_kv_int8(hb: HostBuffer[KVT], kvn: Int, mut scales: List[Scalar[f32]], mut qdata: List[Scalar[i8]]):
    var ngroups = kvn // KVHSTR
    for g in range(ngroups):
        var base = g * KVHSTR
        var amax = Scalar[f32](0)
        for i in range(KVHSTR):
            var v = abs(hb[base + i])
            if v > amax:
                amax = v
        var scale = amax / 127 if amax > 0 else Scalar[f32](1)
        var inv = Scalar[f32](127) / amax if amax > 0 else Scalar[f32](0)
        scales[g] = scale
        for i in range(KVHSTR):
            var qf = round(hb[base + i] * inv)
            qf = min(max(qf, Scalar[f32](-127)), Scalar[f32](127))
            qdata[base + i] = qf.cast[i8]()


def _dequantize_kv_int8(mut hb: HostBuffer[KVT], data: List[UInt8], scale_off: Int, q_off: Int, kvn: Int):
    var scales = data.unsafe_ptr().unsafe_offset(scale_off).unsafe_bitcast[Scalar[f32]]()
    var qdata = data.unsafe_ptr().unsafe_offset(q_off).unsafe_bitcast[Scalar[i8]]()
    var ngroups = kvn // KVHSTR
    for g in range(ngroups):
        var scale = scales[g]
        var base = g * KVHSTR
        for i in range(KVHSTR):
            hb[base + i] = qdata[base + i].cast[f32]() * scale


def load_state(
    ctx: DeviceContext, mut chain: Chain, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    path: String, tmax: Int,
) raises -> Int:
    comptime assert KVT == DType.float32, "state file stores f32 KV"
    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    if len(data) < 72:
        raise Error("BARO_STATE_LOAD: file too short")
    var m2 = String("BAROST02")
    var is_v2 = True
    for i in range(8):
        if data[i] != m2.as_bytes()[i]:
            is_v2 = False
    if not is_v2:
        var m1 = String("BAROST01")
        for i in range(8):
            if data[i] != m1.as_bytes()[i]:
                raise Error("BARO_STATE_LOAD: not a BAROST01/BAROST02 state file")
    var pos = _get_i64(data, 8)
    var kvn = _get_i64(data, 32)
    if _get_i64(data, 16) != CONV_SLOT or _get_i64(data, 24) != SSM_SLOT:
        raise Error("BARO_STATE_LOAD: slot sizes differ from this engine build")
    for i in range(32):
        if data[40 + i] != chain.salt[i]:
            raise Error("BARO_STATE_LOAD: saved from a different pack")
    if pos < 1 or pos >= tmax or kvn != ceildiv(pos, KVPAGE) * N_ATT * NKVH * KVHSTR:
        raise Error("BARO_STATE_LOAD: bad pos/kv size for TMAX " + String(tmax))
    var off = 72
    var tokens = List[Int](capacity=pos)
    for t in range(pos):
        var v = 0
        for b in range(4):
            v |= Int(data[off + 4 * t + b]) << (8 * b)
        tokens.append(v)
    off += 4 * pos
    var ngroups = kvn // KVHSTR
    var kv_bytes = 2 * kvn * 4 if not is_v2 else 2 * (ngroups * 4 + kvn)
    if len(data) != off + (CONV_SLOT + SSM_SLOT) * 4 + kv_bytes:
        raise Error("BARO_STATE_LOAD: payload length mismatch")
    if chain.cap == 0:
        raise Error("BARO_STATE_LOAD: needs a checkpoint slot")
    var idx = 0
    for i in range(len(chain.items)):
        if not chain.items[i].valid:
            idx = i
            break
    chain.gen += 1
    chain.items[idx].pos = pos
    chain.items[idx].hash = prefix_hash(chain.salt, tokens, pos)
    chain.items[idx].gen = chain.gen
    chain.items[idx].valid = True
    chain.items[idx].pending = False
    chain.items[idx].pinned = True
    chain.items[idx].boundary = True
    unsafe_memcpy(dest=chain.items[idx].conv_h.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_ptr().unsafe_offset(off), count=CONV_SLOT * 4)
    off += CONV_SLOT * 4
    unsafe_memcpy(dest=chain.items[idx].ssm_h.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_ptr().unsafe_offset(off), count=SSM_SLOT * 4)
    off += SSM_SLOT * 4
    var kh = ctx.enqueue_create_host_buffer[KVT](kvn)
    var vh = ctx.enqueue_create_host_buffer[KVT](kvn)
    ctx.synchronize()
    if is_v2:
        var k_scale_off = off
        var k_q_off = off + ngroups * 4
        var v_scale_off = k_q_off + kvn
        var v_q_off = v_scale_off + ngroups * 4
        _dequantize_kv_int8(kh, data, k_scale_off, k_q_off, kvn)
        _dequantize_kv_int8(vh, data, v_scale_off, v_q_off, kvn)
    else:
        unsafe_memcpy(dest=kh.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_ptr().unsafe_offset(off), count=kvn * 4)
        unsafe_memcpy(dest=vh.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_ptr().unsafe_offset(off + kvn * 4), count=kvn * 4)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr(), kvn, owning=False), src_buf=kh)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr(), kvn, owning=False), src_buf=vh)
    ctx.synchronize()
    return pos


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var mega = getenv("BARO_MEGA", "1" if MEGA_ALLOWED else "0") == "1"
    if mega and not MEGA_ALLOWED and getenv("BARO_TIER", "") != "":
        raise Error("BARO_MEGA=1 (the MoE persistent token kernel) needs the routed experts in VRAM; unset BARO_TIER")
    if mega and not MEGA_ALLOWED and getenv("BARO_EXPERTS", "") != "":
        raise Error("BARO_MEGA=1 (the MoE persistent token kernel) does not write the expert trace; unset BARO_EXPERTS")
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var pack = load_pack(ctx, packdir)
    var wbuf = pack.wbuf
    var off = pack.off.copy()
    var q4_off = pack.q4_off
    var have_q4_draft = pack.have_q4_draft
    var pack_q4 = pack.pack_q4
    var e = pack.e
    var draft_q4 = getenv("BARO_DRAFT_Q4", "0") == "1" and have_q4_draft
    print("BARO_DRAFT_Q4:", draft_q4)
    var fr_k = pack.fr_k if getenv("BARO_FR", "0") == "1" else 0
    var fr_off = pack.fr_off
    var fr_ids_off = pack.fr_ids_off
    print("BARO_FR:", fr_k > 0, "k", fr_k)
    var dot3 = getenv("BARO_DOT", "0") == "1" and not pack_q4
    print("BARO_DOT:", dot3)
    print("pack q4 trunk:", pack_q4)
    print("BARO_MEGA:", mega)
    var mega_win = getenv("BARO_MEGA_WIN", "0") == "1"
    print("BARO_MEGA_WIN:", mega_win)
    var pf5 = getenv("BARO_PROFILE", "0") == "5"
    var dump_path = getenv("BARO_DUMP", "")
    var dump = dump_path != ""
    var dump4 = getenv("BARO_DUMP4", "0") == "1"
    var dump_layer = atol(getenv("BARO_DUMP_LAYER", "0"))
    # Item 4 verification (coordinator review, 2026-09-16): per-step raw-row
    # + history dump for an independent host comparison, read once here (the
    # harness), never in window.mojo. Off by default.
    var dump_pen_dir = getenv("BARO_DUMP_LOGITS_DIR", "")

    var serve = getenv("BARO_SERVE", "0") == "1"
    print("BARO_SERVE:", serve)
    var kcfg = 2
    try:
        with open(packdir + "/spec-k.txt", "r") as f:
            var kd = f.read_bytes()
            var kv_ = 0
            var kh = False
            for i in range(len(kd)):
                var b = Int(kd[i])
                if b >= 48 and b <= 57:
                    kv_ = kv_ * 10 + (b - 48)
                    kh = True
                elif kh:
                    break
            if kh:
                kcfg = kv_
    except:
        kcfg = 2
    var kenv = getenv("BARO_SPEC_K", "")
    if kenv != "":
        kcfg = atol(kenv)
    if kcfg < 1:
        kcfg = 1
    if kcfg > KMAX:
        kcfg = KMAX
    print("spec k:", kcfg)
    # Prefill plan (bench/prefill-protocol.md): prompt rows 0 .. L-2 go through
    # prefill_forward in chunks of pf_chunk; the last prompt token still runs
    # the megakernel. The MTP draft's "process" window is kept identical to
    # the decode window path: pos_prev is set so it covers the last
    # (L-1) mod 8 rows (8 when 0), whose post-final-norm hidden rows the last
    # chunk writes into hn_d. pf_rows/pf_tail are per request (below).
    var tmax = atol(getenv("BARO_TMAX", String(TMAX)))
    if tmax < CP + PF_MIN:
        tmax = CP + PF_MIN
    var att_split = atol(getenv("BARO_ATT_SPLIT_T", String(TMAX)))
    if getenv("BARO_ATT_SPLIT", "0") == "1":
        att_split = 0
    print("att split:", att_split)
    var pf_chunk = min(atol(getenv("BARO_PREFILL_C", String(CP))), CP)
    if pf_chunk < PF_MIN:
        pf_chunk = PF_MIN
    # The MoE profile has no m>1 MoE prefill yet (W5); replay its prompt rows
    # through the verified m=1 path. Dense profiles retain batched prefill.
    # Scoped on IS_MOE, not MEGA_ALLOWED: the two coincide today only because
    # qwen35moe is the single profile with the mega kernel off, so gating on
    # MEGA_ALLOWED would silently disable batched prefill for the first dense
    # profile that turns the mega kernel off for its own reasons.
    var pf_on = not IS_MOE and getenv("BARO_PREFILL", "1") == "1"
    # Default ON since 2026-09-12: draft decode measured 1.1042x on the frozen
    # 20-prompt median. Every BARO_FORCE caller pins BARO_SPEC=0 explicitly
    # (force-ab.sh, force-ab-serve.sh, dense-run.sh, ornith-run.sh,
    # llama-handoff.sh) and the engine raises if the two are combined, so an
    # unpinned identity gate fails loudly rather than measuring the wrong arm.
    # Still gated on MEGA_ALLOWED: the MoE profile has no draft head.
    var spec_env = MEGA_ALLOWED and getenv("BARO_SPEC", "1") == "1"
    print("BARO_SPEC:", spec_env)
    var spec_dbg = getenv("BARO_SPEC_DBG", "0") == "1"
    # B4 stage 2 (bench/moe-locality-protocol.md): BARO_EXPERTS=<path> writes
    # the router's top-8 per layer per decoded token after each request. A
    # traced run is an instrumentation run: it adds a device-to-device copy per
    # layer, so its tok/s is not a receipt for anything.
    var expert_trace_path = getenv("BARO_EXPERTS", "")
    var expert_trace = expert_trace_path != ""
    print("BARO_EXPERTS:", expert_trace_path if expert_trace else "off")
    # Teacher-forced agreement (identity gate, CLAUDE.md: never greedy equality
    # past ~256 ids). Off by default (empty path -> empty list, decode
    # unchanged): the model's own no-spec argmax is recorded as "predicted"
    # and the reference id is force-fed into the next step's context
    # regardless of what the model chose, same file format and semantics as
    # serve/spark.mojo:260-261.
    var force_env = List[Int]()
    var force_path = getenv("BARO_FORCE", "")
    if force_path != "":
        with open(force_path, "r") as ff:
            var fdata = ff.read_bytes()
            var fval = 0
            var fhave = False
            for i in range(len(fdata)):
                var fb = Int(fdata[i])
                if fb >= 48 and fb <= 57:
                    fval = fval * 10 + (fb - 48)
                    fhave = True
                else:
                    if fhave:
                        force_env.append(fval)
                    fval = 0
                    fhave = False
            if fhave:
                force_env.append(fval)
        print("BARO_FORCE:", force_path, " (", len(force_env), "ids )")
    var bufs = alloc_bufs(ctx, pack, tmax)
    # The dense path consumes Pack.off's historical order.  The MoE pack is
    # lexical by tensor name, so build a per-block semantic order by name;
    # the offsets remain byte offsets into the same Pack.wbuf blob.
    comptime if not MEGA_ALLOWED:
        var moe_tensors = parse_moe_index(packdir + "/index.txt")
        var moe_logical = List[Int]()
        moe_logical.append(resolve_plain(moe_tensors, "token_embd.weight").offset)
        for layer in range(N_LAYERS):
            var pfx = "blk." + String(layer) + "."
            if is_attn(layer):
                for name in [
                    "attn_norm.weight", "attn_q.weight", "attn_k.weight",
                    "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                    "attn_output.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
                moe_logical.append(resolve_plain(moe_tensors, pfx + "post_attention_norm.weight").offset)
                for name in [
                    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                    "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
            else:
                for name in [
                    "attn_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                    "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                    "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
                moe_logical.append(resolve_plain(moe_tensors, pfx + "post_attention_norm.weight").offset)
                for name in [
                    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                    "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
        moe_logical.append(resolve_plain(moe_tensors, "output_norm.weight").offset)
        moe_logical.append(resolve_plain(moe_tensors, "output.weight").offset)
        off = moe_logical.copy()
        bufs.off = moe_logical.copy()
        if len(moe_logical) > OFF_CAP:
            raise Error("MoE offset table exceeds OFF_CAP")
        var moff_h = ctx.enqueue_create_host_buffer[DType.int64](OFF_CAP)
        ctx.synchronize()
        for i in range(OFF_CAP):
            moff_h[i] = Int64(moe_logical[i]) if i < len(moe_logical) else 0
        ctx.enqueue_copy(dst_buf=bufs.off_d, src_buf=moff_h)
        ctx.synchronize()
    var toks_d = bufs.toks_d

    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0)
    var ckpt_cap = atol(getenv("BARO_CKPT", "8")) if serve else 0
    if ckpt_cap < 0:
        ckpt_cap = 0
    var state_load = getenv("BARO_STATE_LOAD", "")
    var state_save = getenv("BARO_STATE_SAVE", "")
    if ckpt_cap == 0 and (state_load != "" or state_save != ""):
        ckpt_cap = 1
    var chain = Chain(ctx, ckpt_cap, packdir)
    print("checkpoints: cap", ckpt_cap, ", bytes", Float64(CKPT_BYTES) / 1e6, "MB each, period", CKPT_PERIOD)
    # L1 (bench/chat-protocol.md): hand prefix checkpoints to an out-of-process
    # latentos-agent as sealed memfds. Unset, nothing here runs. Set and
    # unreachable, this RAISES: a handoff that silently does not happen is the
    # failure this sidecar exists to make visible.
    var latent_sock = getenv("BARO_LATENT_SOCK", "")
    var latent_fd = Int32(-1)
    var latent_gen = 0
    if latent_sock != "":
        latent_fd = connect_unix_socket(latent_sock)
        if latent_fd < 0:
            raise Error("BARO_LATENT_SOCK set but cannot connect: " + latent_sock)
        print("latent: exporting checkpoints to", latent_sock)
    if state_load != "":
        var t_ld = perf_counter_ns()
        var lpos = load_state(ctx, chain, bufs.kc_d, bufs.vc_d, state_load, tmax)
        print("state loaded:", state_load, " pos", lpos, " in", Float64(perf_counter_ns() - t_ld) / 1e9, "s")
    var req_id = 0
    # Requests read off fd 0 by the cancel probe mid-generation, in arrival
    # order; the request loop drains these before touching fd 0 again.
    var pending = List[String]()
    # JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
    # the grammar vocab/trie, built once on the first response_format
    # request (not at startup -- every existing run that never sets it pays
    # nothing, not even the trie build).
    var grt: Optional[GrammarRuntime] = None
    if serve:
        print("{\"ready\":true,\"tmax\":" + String(tmax) + ",\"mrows\":" + String(MROWS) + ",\"kmax\":" + String(KMAX) + ",\"spec_k\":" + String(kcfg) + ",\"kv\":\"" + String(KVT) + "\",\"pack\":\"" + packdir + "\"}")

    # --- request loop: one prompt from the file (BARO_SERVE=0, then exit) or
    # JSON lines from stdin until EOF (BARO_SERVE=1, serve/PROTOCOL.md) ------
    while True:
        var prompt = List[Int]()
        var gen_n = GEN_N
        var spec = spec_env
        var ckpt_idx = -1
        var cached = 0
        var restore_s = 0.0
        var force = force_env.copy()
        var stop_seqs = List[List[Int]]()
        var ckpt_hints = List[Int]()
        var sample = default_sample_params()
        var req_state_save = String("")
        var req_state_load = String("")
        if serve:
            var line_in = Optional[String](None)
            if len(pending) > 0:
                line_in = Optional(pending.pop(0))
            else:
                line_in = read_line(0)
            if not line_in:
                break
            var req_n = 0
            var req_spec = False
            var req_has_spec = False
            var perr = parse_request(line_in.value(), req_id, prompt, req_n, req_spec, req_has_spec, stop_seqs, ckpt_hints, sample, req_state_save, req_state_load)
            if perr == "" and len(prompt) < 1:
                perr = "empty prompt"
            if perr == "" and req_n < 1:
                perr = "n must be >= 1"
            if perr == "" and len(prompt) + req_n > tmax:
                perr = "prompt+n exceeds TMAX " + String(tmax)
            # C3 fix round (bench/chat-protocol.md, exchange/lane-WIRING-report.md
            # C3 section): serve/sample_ref.mojo's ceil'd top-p mass target was
            # the root cause of the M5 gate-2 finding above; fixed in 054bd22
            # and gate 2 now passes clean on all four truncated shapes
            # (top_p<1 and/or top_k>0, with or without min_p), so the refusal
            # for those shapes is lifted.
            #
            # C3 tail round (bench/chat-protocol.md): the untruncated shape
            # (top_p=1, top_k=0, min_p<=0) is a separate, still-open defect —
            if perr != "":
                print(err_line(req_id, perr))
                continue
            gen_n = req_n
            if req_has_spec:
                spec = req_spec
            # A1 (bench/spec-sample-protocol.md): temperature > 0 used to force
            # spec off, because accepting a draft on argmax equality under
            # sampling silently changes the output distribution. The window now
            # runs the real rule instead (accept with min(1, p/q) on the
            # truncated distributions, residual draw on the first rejection,
            # bonus token from p), so sampling and speculation compose.
            # JSON-enforcement item 1/2: reset every request, schema or not
            # -- wst is reused across requests, so a prior request's matcher
            # must never leak into one that never asked for response_format.
            wst.grammar = None
            wst.grammar_pending_think = False
            wst.grammar_think_buf = List[UInt8]()
            wst.grammar_stop = False
            wst.grammar_masked_draws = 0
            wst.grammar_accepted = 0
            var schema_raw = parse_schema_field(line_in.value())
            if schema_raw != "":
                # Interim (see window.mojo item 1 comment): the masked kernel
                # filters after truncation today, so a schema request that
                # also truncates could draw outside the allowed set. Dropped
                # once the masked-first kernel lands.
                sample.top_p = 1.0
                sample.top_k = 0
                sample.min_p = 0.0
                spec = False
                if not grt:
                    try:
                        grt = Optional(GrammarRuntime(packdir))
                    except e:
                        print(err_line(req_id, "response_format: grammar runtime failed to load: " + String(e)))
                        continue
                if grt.value().vocab_size != VOCAB:
                    print(err_line(req_id, "response_format: grammar vocab_size " + String(grt.value().vocab_size) + " != engine VOCAB " + String(VOCAB) + " (stale pack?)"))
                    continue
                try:
                    wst.grammar = Optional(compile_schema_matcher(grt.value(), schema_raw))
                except e:
                    print(err_line(req_id, "response_format schema: " + String(e)))
                    continue
                wst.grammar_pending_think = parse_reasoning_field(line_in.value())
                wst.grammar_mask = Bitset(VOCAB)
                print("response_format: schema compiled, reasoning wait:", wst.grammar_pending_think)
            print("prompt tokens:", len(prompt), " n:", gen_n, " spec:", spec, " temperature:", sample.temperature)
            # Per-request teacher forcing. BARO_FORCE is read once at startup,
            # so a forced run used to need one process per prompt: 20 pack
            # loads for a 20-prompt gate, 1.4 s each on the MoE pack. A
            # "force":[ids] field lets one resident process serve the whole
            # gate. Absent, the request inherits BARO_FORCE.
            var fi2 = json_key(line_in.value(), "force")
            if fi2 >= 0:
                force = List[Int]()
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
            # M1a prefix checkpoint: restore the SSM slot for the longest
            # hashed prefix and keep the KV pool (position addressed, [0, cached)
            # still in place); the draft KV is never prefilled, so it is zeroed
            # exactly as on the cold path. A miss is today's path.
            # spec mode replays at least one prompt row so the draft head's
            # hidden rows (hn_d) are fresh: the checkpoint at len-1 is skipped.
            if req_state_load != "":
                # Checkpoint API: bring the named state file into the chain
                # before the lookup; the lookup then finds it by prefix hash.
                # A refusal (different pack, bad file) is this request's
                # error, never the engine's death.
                var load_err = String("")
                try:
                    var t_ld = perf_counter_ns()
                    var lpos = load_state(ctx, chain, bufs.kc_d, bufs.vc_d, req_state_load, tmax)
                    print("state loaded:", req_state_load, " pos", lpos, " in", Float64(perf_counter_ns() - t_ld) / 1e9, "s")
                except e:
                    load_err = String(e)
                if load_err != "":
                    print(err_line(req_id, load_err))
                    continue
            ckpt_idx = chain.lookup(prompt, len(prompt) - 1 if spec else len(prompt))
            cached = chain.pos_of(ckpt_idx)
            chain.invalidate_above(cached)
            var t_restore = perf_counter_ns()
            if ckpt_idx >= 0:
                chain.restore(ctx, bufs.convstate_d, bufs.sstate_d, 0, ckpt_idx)
            else:
                ctx.enqueue_memset(bufs.convstate_d, 0)
                ctx.enqueue_memset(bufs.sstate_d, 0)
                ctx.enqueue_memset(bufs.kc_d, 0)
                ctx.enqueue_memset(bufs.vc_d, 0)
            ctx.enqueue_memset(bufs.kc32_d, 0)
            ctx.enqueue_memset(bufs.vc32_d, 0)
            ctx.enqueue_memset(bufs.ctr_d, 0)
            ctx.enqueue_memset(bufs.prof_d, 0)
            ctx.synchronize()
            restore_s = Float64(perf_counter_ns() - t_restore) / 1e9
        else:
            var prompt_path = getenv("BARO_PROMPT", packdir + "/prompt-tokens.txt")
            print("prompt file:", prompt_path)
            with open(prompt_path, "r") as f:
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
            print("prompt tokens:", len(prompt))
            if chain.cap > 0:
                ckpt_idx = chain.lookup(prompt, len(prompt) - 1 if spec else len(prompt))
                cached = chain.pos_of(ckpt_idx)
                if ckpt_idx >= 0:
                    var t_rs = perf_counter_ns()
                    chain.restore(ctx, bufs.convstate_d, bufs.sstate_d, 0, ckpt_idx)
                    ctx.synchronize()
                    restore_s = Float64(perf_counter_ns() - t_rs) / 1e9

        var pf_rows = 0
        var pf_tail = 0
        if pf_on and len(prompt) - 1 - cached >= PF_MIN:
            pf_rows = len(prompt) - 1
            pf_tail = (pf_rows - cached) % MROWS
            if pf_tail == 0:
                pf_tail = MROWS
        var prefill_rows = len(prompt) - 1 - cached
        print("TMAX:", tmax, " kv dtype:", String(KVT), " prefill chunk:", pf_chunk, " prefill rows:", pf_rows, " cached:", cached, " replay rows:", prefill_rows)
        # BARO_MARGIN=1 (with BARO_FORCE): after each forced step, recompute the
        # head from that step's final residual (final rmsnorm, q4 head GEMM,
        # reduce) and print the top-2 logits and their gap next to the argmax
        # the engine produced.
        var margin = getenv("BARO_MARGIN", "0") == "1" and len(force) > 0
        print("BARO_MARGIN:", margin)

        # --- decode loop ---------------------------------------------------------


        var n_total = len(prompt) + gen_n
        if n_total > tmax:
            print(
                "{\"error\":{\"code\":400,\"message\":\"the request exceeds the"
                " available context size, try increasing it\",\"type\":"
                "\"exceed_context_size_error\",\"n_prompt_tokens\":"
                + String(len(prompt)) + ",\"n_ctx\":" + String(tmax) + "}}"
            )
            exit(2)
        var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
        ctx.synchronize()
        for i in range(tmax):
            toks_h[i] = 0
        for i in range(len(prompt)):
            toks_h[i] = Int32(prompt[i])
        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var t_prefill_end = t0
        # BARO_PROFILE=1: synchronize at sub-block boundaries and attribute GPU
        # time to attn / ssm / ffn / head. Off by default; the timed path is
        # untouched.
        var prof = getenv("BARO_PROFILE", "0") != "0"
        var pf2 = getenv("BARO_PROFILE", "0") == "2"
        # BARO_PROFILE=3: split the draft path (blk32_forward's process + k-1
        # draft-step calls, plus the accept block) into layer body / head gemm+
        # reduce / argmax / accept-sync buckets; "other" is the leftover
        # tokcp/bookkeeping dispatch cost inside pf_proc+pf_draft.
        var pf3 = getenv("BARO_PROFILE", "0") == "3"
        # BARO_PROFILE=4: same idea as 2, on the ffn sub-block -- rmsnorm / gate
        # gemm / up gemm / swiglu / down gemm / residual add. Serialized, so the
        # stage sum exceeds the unsynchronized sub-block time; compare stages
        # within an arm and the same stage across m, never add these into a budget.
        var pf4 = getenv("BARO_PROFILE", "0") == "4"
        # Items 3-4 (briefs/2026-09-16-sampling-all-models-lane.md), fix for
        # the coordinator's P1 inert-parameter finding: penalties/top_logprobs
        # are only applied on window.mojo's plain (non-spec) single-token
        # decode step, so a default request (spec ON by default, temperature
        # possibly 0) could silently never reach it. Force both the
        # megakernel and speculation off for the rest of THIS request instead
        # of letting the parameter fall through unused.
        var want_extra = sample.presence_penalty != 0 or sample.frequency_penalty != 0 or sample.top_logprobs > 0
        var want_grammar = wst.grammar.__bool__()
        if want_extra or want_grammar:
            spec = False
        # A6 (bench/chat-protocol.md A6): every m == 1 decode step runs the
        # megakernel. window.mojo folds the head (argmax inside the launch)
        # only for a plain greedy request; sampling, penalties, top_logprobs
        # and grammar requests run the layers with fold_head 0 (the shape
        # the window already uses) and the launch-path head plus the
        # sampler after it, so their logits edits and mask sit between the
        # head and the draw as before (amar_sample_row is argmax-equivalent
        # at temperature <= 0, P-K2).
        var mega_req = mega
        # A1: the megakernel WINDOW writes the window's tokens itself, so it
        # cannot host the speculative sampling rule (which needs the target's
        # full probability rows, not its argmax). Sampling therefore stays on
        # the launch path for the window too, exactly as it already does for
        # the single-token megakernel above.
        var mega_win_req = mega_win and sample.temperature <= 0 and not want_extra and not want_grammar
        var dump_pen = want_extra and dump_pen_dir != ""
        var cfg = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, e=e, kcfg=kcfg, spec=spec, spec_dbg=spec_dbg, expert_trace=expert_trace, serve=serve, req_id=req_id, prof=prof, pf2=pf2, pf3=pf3, pf4=pf4, dump=dump, dump4=dump4, dump_layer=dump_layer, mega=mega_req, att_split=att_split, mega_win=mega_win_req, dot3=dot3, pf_chunk=pf_chunk, pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, fr_k=fr_k, fr_off=fr_off, fr_ids_off=fr_ids_off, n_prompt=len(prompt), sample=sample.copy(), dump_pen=dump_pen)
        if len(force) > 0 and cfg.spec:
            raise Error("BARO_FORCE requires BARO_SPEC=0 (teacher forcing is a no-spec identity gate)")
        wst.reset(t0)
        wst.pos = cached
        wst.pos_prev = cached
        var prefill_done = False
        var cancelled = False
        var stopped = False
        var predicted = List[Int]()
        var force_tok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        var lg_h = ctx.enqueue_create_host_buffer[f32](VOCAB if margin else 1)
        # The stopwatch stays here, in the harness that is never embedded in a
        # gguf: step_window cannot reach t0, t_prefill_end or dt (P-A, 2026-09-08).
        while wst.pos < n_total - 1:
            var pos_before = wst.pos
            if len(ckpt_hints) > 0 and wst.pos < pf_rows:
                # A hint may fall inside what would otherwise be one big
                # prefill chunk; cap this call's chunk so wst.pos actually
                # stops there (checkpoints are only taken between calls).
                var step_cfg = cfg.copy()
                step_cfg.pf_chunk = min(cfg.pf_chunk, next_ckpt_stop(ckpt_hints, wst.pos, pf_rows) - wst.pos)
                step_window(ctx, bufs, step_cfg, wst)
            else:
                step_window(ctx, bufs, cfg, wst)
            # Item 4 verification (coordinator review, 2026-09-16): the sync
            # and file write live here, in the harness, never in
            # window.mojo. bufs.dump_row_h and bufs.pen_hist_h were staged
            # by window.mojo's own enqueue_copy calls (cfg.dump_pen /
            # penalties-or-logprobs), synchronized here for the first time.
            if cfg.dump_pen and wst.pos == pos_before + 1 and pos_before + 1 >= len(prompt):
                ctx.synchronize()
                var step = pos_before + 1 - len(prompt)
                with open(dump_pen_dir + "/row-" + String(step) + ".bin", "w") as f:
                    var p = bufs.dump_row_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                    f.write_bytes(Span[UInt8](unsafe_ptr=p, length=VOCAB * 4))
                with open(dump_pen_dir + "/row-" + String(step) + ".json", "w") as jf:
                    var js = String("{\"temperature\":") + String(sample.temperature) + ",\"top_k\":" + String(sample.top_k)
                    js += ",\"top_p\":" + String(sample.top_p) + ",\"min_p\":" + String(sample.min_p)
                    js += ",\"presence_penalty\":" + String(sample.presence_penalty) + ",\"frequency_penalty\":" + String(sample.frequency_penalty)
                    js += ",\"top_logprobs\":" + String(sample.top_logprobs) + ",\"history\":["
                    for i in range(step):
                        if i > 0:
                            js += ","
                        js += String(Int(bufs.pen_hist_h[i]))
                    js += "]}"
                    jf.write_bytes(js.as_bytes())
            if len(force) > 0 and pos_before >= len(prompt) - 1:
                if wst.pos != pos_before + 1:
                    raise Error("BARO_FORCE: step advanced by more than one position (spec/prefill batching) -- void arm")
                var fi = pos_before - (len(prompt) - 1)
                if fi < len(force):
                    ctx.enqueue_copy(dst_buf=force_tok_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr().unsafe_offset(wst.pos), 1, owning=False))
                    ctx.synchronize()
                    var got = Int(force_tok_h[0])
                    predicted.append(got)
                    if margin:
                        var wf = 1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4
                        var Xm = TileTensor(bufs.x_d, xm_layout)
                        var CurBm = TileTensor(bufs.curb_d, xm_layout)
                        ctx.enqueue_function[rmsc_k](Xm, tens_f32(ctx, bufs.wbuf, bufs.off[wf], H, h_layout), CurBm, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
                        var Pv = TileTensor(bufs.p_v_d, p_v)
                        gemm_w[VOCAB, H](ctx, CurBm, bufs.wbuf, bufs.off[wf + 1], pack_q4, Pv, 1)
                        ctx.enqueue_function[r_head](Pv, TileTensor(bufs.logits_d, vm_layout), Int32(1), Int32(VOCAB), grid_dim=ceildiv(VOCAB, 256), block_dim=256)
                        ctx.enqueue_copy(dst_buf=lg_h, src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False))
                        ctx.synchronize()
                        var i1 = 0
                        var i2 = -1
                        for i in range(1, VOCAB):
                            if lg_h[i] > lg_h[i1]:
                                i2 = i1
                                i1 = i
                            elif i2 < 0 or lg_h[i] > lg_h[i2]:
                                i2 = i
                        var refv = force[fi]
                        print("MARGIN pos", fi, " engine_argmax", got, " ref", refv, " top1", i1, lg_h[i1], " top2", i2, lg_h[i2], " gap", lg_h[i1] - lg_h[i2], " ref_logit", lg_h[refv])
                    force_tok_h[0] = Int32(force[fi])
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr().unsafe_offset(wst.pos), 1, owning=False), src_buf=force_tok_h)
                    ctx.synchronize()
            if ckpt_cap > 0 and wst.pos > cached and wst.pos < len(prompt):
                if wst.pos == len(prompt) - 1 or wst.pos % CKPT_PERIOD == 0:
                    chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, False, False)
                else:
                    var hi = hint_index(ckpt_hints, wst.pos)
                    if hi >= 0:
                        chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, hi == 0, True)
            if not prefill_done and wst.pos >= len(prompt):
                ctx.synchronize()
                t_prefill_end = perf_counter_ns()
                prefill_done = True
            if serve and cancel_pending(0, req_id, pending):
                cancelled = True
                break
            if wst.grammar_stop:
                # JSON-enforcement item 1: the matcher reached an accepting
                # state with no further legal byte -- same "stop" finish as
                # a stop sequence, no new wire value.
                stopped = True
                break
            if len(stop_seqs) > 0 and wst.pos >= len(prompt):
                ctx.synchronize()
                ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
                ctx.synchronize()
                var gen_len = wst.pos + 1 - len(prompt)
                for si in range(len(stop_seqs)):
                    var seq = stop_seqs[si].copy()
                    var L = len(seq)
                    if L == 0 or L > gen_len:
                        continue
                    var ok = True
                    for k in range(L):
                        if Int(toks_h[wst.pos - L + 1 + k]) != seq[k]:
                            ok = False
                            break
                    if ok:
                        stopped = True
                        break
                if stopped:
                    break
        if not prefill_done:
            ctx.synchronize()
            t_prefill_end = perf_counter_ns()

        var t_host = Float64(perf_counter_ns() - t0) / 1e9
        ctx.synchronize()
        chain.commit()
        if latent_fd >= 0:
            var client = EngineLatentClient(latent_fd)
            var exported = 0
            for i in range(len(chain.items)):
                if chain.items[i].valid and chain.items[i].gen > latent_gen:
                    if not client.export_chain_slot(chain, i):
                        raise Error("latent export failed at slot " + String(i))
                    exported += 1
            latent_gen = chain.gen
            print("latent: exported", exported, "checkpoints, chain gen", latent_gen)
        if state_save != "":
            save_state(ctx, chain, bufs.kc_d, bufs.vc_d, state_save, prompt, len(prompt) - 1)
        if req_state_save != "":
            var t_sv = perf_counter_ns()
            save_state(ctx, chain, bufs.kc_d, bufs.vc_d, req_state_save, prompt, len(prompt) - 1)
            print("state saved:", req_state_save, " pos", len(prompt) - 1, " in", Float64(perf_counter_ns() - t_sv) / 1e9, "s")
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        print("host_enqueue_s:", t_host, " gpu_total_s:", dt)
        var flw = ctx.enqueue_create_host_buffer[DType.uint32](3)
        ctx.enqueue_copy(dst_buf=flw, src_buf=bufs.ctr_d)
        ctx.synchronize()
        print("mega fail word:", flw[2], "" if flw[2] == 0 else " NOT-RESIDENT: a grid barrier timed out, tokens after it are invalid")
        if wst.grammar.__bool__():
            # Gate 4 (bench/grammar-protocol.md): masked_draws must equal
            # accepted every time -- read on every grammar-constrained run,
            # not just when something looks wrong.
            print("grammar masked draws:", wst.grammar_masked_draws, " accepted:", wst.grammar_accepted, " terminated:", wst.grammar_stop, "" if wst.grammar_masked_draws == wst.grammar_accepted else " MISMATCH: masked draw and accept counts desynced")
        # P1 receipt from the device, not from the env echo: the grid-barrier
        # generation counter is written only by a persistent kernel's barriers
        # (the launch path never touches ctr_d), so gen = 0 means no persistent
        # kernel ran this process, and on the MoE profile gen / MOE_BARRIERS is
        # the number of tokens the MoE kernel produced.
        comptime if not MEGA_ALLOWED:
            print("mega barrier gen:", flw[1], " MoE kernel tokens:", Int(flw[1]) // MOE_BARRIERS, "(", MOE_BARRIERS, "barriers per token, grid", MOE_G, ")")
        else:
            print("mega barrier gen:", flw[1])
        if dump:
            ctx.synchronize()
            with open(dump_path, "w") as f:
                var dpp = bufs.dump_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                f.write_bytes(Span[UInt8](unsafe_ptr=dpp, length=wst.n_dumped * 2 * N_LAYERS * H * 4))
            print("dumped", wst.n_dumped, "tokens x", N_LAYERS, "layers to", dump_path)
        if dump and getenv("BARO_DUMP_DIR", "") != "":
            # Same container the packer writes and tools/tdiff.mojo reads:
            # data.bin plus one "name f32 byte_offset n_elem" line per tensor.
            # Names match llama.cpp's own (attn_residual-N, attn_post_norm-N,
            # l_out-N, linear_attn_out-N) so a dump joins against
            # llama-eval-callback's LLAMA_DUMP_DIR output by name, with no
            # mapping table to drift.
            var ddir = getenv("BARO_DUMP_DIR", "")
            ctx.synchronize()
            var dpp = bufs.dump_h.unsafe_ptr()
            var names = List[String]()
            var slots = List[Int]()
            var lens = List[Int]()
            comptime if not MEGA_ALLOWED:
                if dump4 and (dump_layer + 1) % 4 == 0:
                    # attention layer: the SSM slots never fire, and slots 4-6
                    # carry the attention captures instead.
                    names.append(String("attn_residual-") + String(dump_layer)); slots.append(0); lens.append(H)
                    names.append(String("attn_post_norm-") + String(dump_layer)); slots.append(1); lens.append(H)
                    names.append(String("l_out-") + String(dump_layer)); slots.append(2); lens.append(H)
                    # Names are llama's, exactly. The first pass called our
                    # pre-gate Ao "attn_output" and the o_proj result
                    # "attn_oproj"; llama calls those attn_pregate and
                    # attn_output, so tdiff was joining unrelated tensors of
                    # different lengths and reported relL2 22.69 on a layer
                    # whose downstream was provably at floor.
                    names.append(String("attn_pregate-") + String(dump_layer)); slots.append(4); lens.append(ATT)
                    names.append(String("attn_output-") + String(dump_layer)); slots.append(6); lens.append(H)
                    names.append(String("Qcur_full-") + String(dump_layer)); slots.append(8); lens.append(QF)
                    names.append(String("Qcur_normed-") + String(dump_layer)); slots.append(12); lens.append(ATT)
                    names.append(String("Qcur-") + String(dump_layer)); slots.append(14); lens.append(ATT)
                    names.append(String("Kcur_normed-") + String(dump_layer)); slots.append(16); lens.append(KV)
                    names.append(String("Kcur-") + String(dump_layer)); slots.append(17); lens.append(KV)
                elif dump4:
                    # SSM layer sub-block captures, slot order set in window.mojo
                    names.append(String("attn_residual-") + String(dump_layer)); slots.append(0)
                    names.append(String("attn_post_norm-") + String(dump_layer)); slots.append(1); lens.append(H)
                    names.append(String("l_out-") + String(dump_layer)); slots.append(2); lens.append(H)
                    names.append(String("final_norm-") + String(dump_layer)); slots.append(3); lens.append(H)
                    names.append(String("ssm_gates-") + String(dump_layer)); slots.append(5); lens.append(H)
                    names.append(String("ssm_state_out-") + String(dump_layer)); slots.append(6); lens.append(H)
                    names.append(String("linear_attn_out-") + String(dump_layer)); slots.append(7); lens.append(H)
                    names.append(String("conv_out-") + String(dump_layer)); slots.append(8); lens.append(H)
                else:
                    for layer in range(N_LAYERS):
                        names.append(String("attn_residual-") + String(layer))
                        slots.append(2 * layer)
                        lens.append(H)
            var off = 0
            var idx = String("")
            with open(ddir + "/data.bin", "w") as fb:
                for i in range(len(names)):
                    var base = slots[i] * H
                    var ln = lens[i]
                    fb.write_bytes(
                        Span[UInt8](
                            unsafe_ptr=dpp.unsafe_offset(base).unsafe_bitcast[UInt8](),
                            length=ln * 4,
                        )
                    )
                    idx += names[i] + " f32 " + String(off) + " " + String(ln) + "\n"
                    off += ln * 4
            with open(ddir + "/index.txt", "w") as fi:
                fi.write(idx)
            print("dump dir:", ddir, len(names), "tensors")
        if pf5:
            var ph = ctx.enqueue_create_host_buffer[DType.int64](16 * N_LAYERS + 4)
            ctx.enqueue_copy(dst_buf=ph, src_buf=bufs.prof_d)
            var fl = ctx.enqueue_create_host_buffer[DType.uint32](3)
            ctx.enqueue_copy(dst_buf=fl, src_buf=bufs.ctr_d)
            ctx.synchronize()
            var sub_ssm = 0.0
            var sub_att = 0.0
            var ffn_us = 0.0
            for layer in range(N_LAYERS):
                var a = Float64(ph[16 * layer + 7] - ph[16 * layer]) / 100.0
                var b = Float64(ph[16 * layer + 11] - ph[16 * layer + 7]) / 100.0
                if is_attn(layer):
                    sub_att += a
                else:
                    sub_ssm += a
                ffn_us += b
            var head_us = Float64(ph[16 * N_LAYERS + 3] - ph[16 * N_LAYERS]) / 100.0
            print("mega profile (last token, us): ssm sub-blocks", sub_ssm, " attn sub-blocks", sub_att, " ffn", ffn_us, " head", head_us, " total", Float64(ph[16 * N_LAYERS + 3] - ph[0]) / 100.0, " fail", fl[2])
            # BARO_PROFILE=5 raw: the prof buffer verbatim, 16 slots per layer
            # plus 4 tail slots, s_sendmsg.rtn(131) REALTIME ticks at 100 MHz.
            # Printed as raw ticks AND as us relative to slot 0 of layer 0, so
            # a reader can re-derive every aggregate above instead of taking
            # it on trust.
            print("PROFRAW slots", 16 * N_LAYERS + 4, " hz 100000000  base", ph[0])
            for layer in range(N_LAYERS):
                var line = String("PROFRAW L") + String(layer) + (" attn" if is_attn(layer) else " ssm ")
                for i in range(16):
                    line += " " + String(ph[16 * layer + i])
                print(line)
            var tail = String("PROFRAW TAIL")
            for i in range(4):
                tail += " " + String(ph[16 * N_LAYERS + i])
            print(tail)
            # per-phase sums over the layers of each kind, in stamp order (us)
            var ssm_seq: List[Int] = [0, 12, 1, 2, 3, 4, 5, 6, 7]
            var att_seq: List[Int] = [0, 1, 2, 3, 4, 5, 7]
            var ffn_seq: List[Int] = [7, 8, 9, 10, 11]
            var ssm_ph = InlineArray[Float64, 8](fill=0.0)
            var att_ph = InlineArray[Float64, 6](fill=0.0)
            var ffn_ph = InlineArray[Float64, 4](fill=0.0)
            for layer in range(N_LAYERS):
                var b = 16 * layer
                if is_attn(layer):
                    for i in range(6):
                        att_ph[i] += Float64(ph[b + att_seq[i + 1]] - ph[b + att_seq[i]]) / 100.0
                else:
                    for i in range(8):
                        ssm_ph[i] += Float64(ph[b + ssm_seq[i + 1]] - ph[b + ssm_seq[i]]) / 100.0
                for i in range(4):
                    ffn_ph[i] += Float64(ph[b + ffn_seq[i + 1]] - ph[b + ffn_seq[i]]) / 100.0
            var ssm_s = String("mega phases ssm (24 layers, us, stamps 0>12>1>2>3>4>5>6>7):")
            for i in range(8):
                ssm_s += " " + String(Int(ssm_ph[i]))
            print(ssm_s)
            var att_s = String("mega phases attn (8 layers, us, stamps 0>1>2>3>4>5>7):")
            for i in range(6):
                att_s += " " + String(Int(att_ph[i]))
            print(att_s)
            var ffn_s = String("mega phases ffn (32 layers, us, stamps 7>8>9>10>11):")
            for i in range(4):
                ffn_s += " " + String(Int(ffn_ph[i]))
            print(ffn_s)
            var head_s = String("mega phases head (us, stamps 0>1>2>3):")
            for i in range(3):
                head_s += " " + String(Int(Float64(ph[16 * N_LAYERS + i + 1] - ph[16 * N_LAYERS + i]) / 100.0))
            print(head_s)
            comptime if not MEGA_ALLOWED:
                # kernels/mega_moe.mojo stamps 0..12 per layer (attention layers
                # skip 6) and NPROF * N_LAYERS at the end; per-phase sums over
                # the layers of each kind, last token, us.
                var mssm_seq: List[Int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                var matt_seq: List[Int] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                var mssm = InlineArray[Float64, 12](fill=0.0)
                var matt = InlineArray[Float64, 11](fill=0.0)
                var mtot = 0.0
                for layer in range(N_LAYERS):
                    var b = 16 * layer
                    if is_attn(layer):
                        for i in range(11):
                            matt[i] += Float64(ph[b + matt_seq[i + 1]] - ph[b + matt_seq[i]]) / 100.0
                    else:
                        for i in range(12):
                            mssm[i] += Float64(ph[b + mssm_seq[i + 1]] - ph[b + mssm_seq[i]]) / 100.0
                    mtot += Float64(ph[b + 12] - ph[b]) / 100.0
                var ms = String("moe phases ssm (30 layers, us, stamps 0>1>2>3>4>5>6>7>8>9>10>11>12 = rms|proj|gates+conv|l2|delta|gated|ssm_out|rms|router|top8|gate_up|down):")
                for i in range(12):
                    ms += " " + String(Int(mssm[i]))
                print(ms)
                var ma = String("moe phases attn (10 layers, us, stamps 0>1>2>3>4>5>7>8>9>10>11>12 = rms|proj|heads|attn|gmul|out|rms|router|top8|gate_up|down):")
                for i in range(11):
                    ma += " " + String(Int(matt[i]))
                print(ma)
                print("moe kernel layers total us:", Int(mtot), " kernel span us:", Int(Float64(ph[16 * N_LAYERS] - ph[0]) / 100.0))
        if prof:
            var tot = Float64(wst.pf_att + wst.pf_ssm + wst.pf_ffn + wst.pf_head)
            print("profile: attn", Float64(wst.pf_att) / 1e9, Float64(wst.pf_att) / tot)
            print("profile: ssm", Float64(wst.pf_ssm) / 1e9, Float64(wst.pf_ssm) / tot)
            print("profile: ffn", Float64(wst.pf_ffn) / 1e9, Float64(wst.pf_ffn) / tot)
            print("profile: head", Float64(wst.pf_head) / 1e9, Float64(wst.pf_head) / tot)
            print("profile: mtp_proc", Float64(wst.pf_proc) / 1e9, " mtp_draft", Float64(wst.pf_draft) / 1e9)
        if pf3:
            var other = wst.pf_proc + wst.pf_draft - wst.p3[0] - wst.p3[1] - wst.p3[2]
            print(
                "profile3: layer", Float64(wst.p3[0]) / 1e9, " head", Float64(wst.p3[1]) / 1e9,
                " argmax", Float64(wst.p3[2]) / 1e9, " accept", Float64(wst.p3[3]) / 1e9,
                " other", Float64(other) / 1e9,
            )
            if wst.n_spec_windows > 0:
                var nw = Float64(wst.n_spec_windows)
                print(
                    "profile3_per_window_ms: layer", Float64(wst.p3[0]) / 1e6 / nw, " head", Float64(wst.p3[1]) / 1e6 / nw,
                    " argmax", Float64(wst.p3[2]) / 1e6 / nw, " accept", Float64(wst.p3[3]) / 1e6 / nw,
                    " other", Float64(other) / 1e6 / nw,
                )
        if pf4:
            var fnames = ["rmsnorm", "gemm_gate", "gemm_up", "swiglu", "gemm_down", "r_add"]
            var ft = Float64(wst.fc[0] + wst.fc[1] + wst.fc[2] + wst.fc[3] + wst.fc[4] + wst.fc[5])
            for i in range(6):
                print("ffn-kernel:", fnames[i], Float64(wst.fc[i]) / 1e9, Float64(wst.fc[i]) / ft)
        if pf2:
            var names = ["gemm4+reduce2", "rgates", "conv", "l2", "delta", "gated", "out_gemm+add", "-"]
            var st = Float64(wst.pc[0] + wst.pc[1] + wst.pc[2] + wst.pc[3] + wst.pc[4] + wst.pc[5] + wst.pc[6])
            for i in range(7):
                print("ssm-kernel:", names[i], Float64(wst.pc[i]) / 1e9, Float64(wst.pc[i]) / st)
        ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
        ctx.synchronize()
        var generated = List[Int]()
        for i in range(len(prompt), wst.pos + 1):
            generated.append(Int(toks_h[i]))
        var prefill_s = Float64(t_prefill_end - t0) / 1e9
        var decode_s = dt - prefill_s
        var n_gen = len(generated)
        # tok/s_gen is the only number comparable to llama.cpp: it divides
        # n_gen - 1 by decode time alone, matching timings.predicted_per_second
        # (n_gen, not the requested gen_n, so an early stop/cancel reports the
        # rate over what actually ran). tok/s_total includes prefill and is
        # reported for completeness only -- it is not the engine's throughput
        # against any external baseline.
        print("tokens:", n_gen, " prefill_s:", prefill_s, " decode_s:", decode_s)
        print("tok/s_total:", Float64(n_gen) / dt, " tok/s_gen:", Float64(n_gen - 1) / decode_s if n_gen > 1 else 0.0)
        var line = String("")
        for i in range(len(generated)):
            line += String(generated[i]) + " "
        print("GENERATED:", line)
        # B4 stage 2b gate 3: the tier's own counters, per request. Printed
        # here rather than in window.mojo so the wiring patch stays small.
        if bufs.tier.active:
            bufs.tier.report(n_gen)
        # B4 stage 2: one copy out, after the request, not per layer. The file
        # is appended so a 20-prompt sweep is one trace; each block names the
        # prompt's own token count so the replay can refuse a truncated trace.
        if expert_trace:
            ctx.enqueue_copy(dst_buf=bufs.etrace_h, src_buf=bufs.etrace_d)
            ctx.synchronize()
            var tl = String("# trace n_gen=") + String(n_gen) + " layers=" + String(N_LAYERS) + " topk=" + String(TOPK_TRACE) + " prompt_tokens=" + String(len(prompt)) + "\n"
            var rows = min(n_gen, TRACE_TOK)
            for tk in range(rows):
                for ly in range(N_LAYERS):
                    var base = (tk * N_LAYERS + ly) * TOPK_TRACE
                    var any_set = False
                    for e in range(TOPK_TRACE):
                        if bufs.etrace_h[base + e] >= 0:
                            any_set = True
                    if not any_set:
                        continue
                    tl += String(tk) + " " + String(ly)
                    for e in range(TOPK_TRACE):
                        tl += " " + String(bufs.etrace_h[base + e])
                    tl += "\n"
            with open(expert_trace_path, "a") as tf:
                tf.write(tl)
            print("expert trace appended:", expert_trace_path, " rows", rows, "x", N_LAYERS)
        if len(force) > 0:
            var ps = String("")
            var agree = 0
            var checked = min(len(predicted), len(force))
            for i in range(len(predicted)):
                ps += String(predicted[i]) + " "
                if i < len(force) and predicted[i] == force[i]:
                    agree += 1
            print("predicted:", ps)
            print("forced agreement:", agree, "/", checked)
        if spec:
            print("mtp: drafted", wst.n_drafted, " accepted", wst.n_accepted, " k", kcfg)
        if serve:
            var finish = String("length")
            if cancelled:
                finish = String("cancelled")
            elif stopped:
                finish = String("stop")
            var done_line = String("{\"id\":") + String(req_id) + ",\"done\":true,\"n\":" + String(n_gen)
            done_line += ",\"prefill_s\":" + String(prefill_s) + ",\"decode_s\":" + String(decode_s)
            done_line += ",\"tok_s\":" + String(Float64(n_gen - 1) / decode_s if n_gen > 1 else 0.0)
            done_line += ",\"cached\":" + String(cached) + ",\"prefill_rows\":" + String(prefill_rows) + ",\"restore_s\":" + String(restore_s) + ",\"checkpoints\":" + String(chain.count_valid())
            done_line += ",\"finish\":\"" + finish + "\""
            if spec:
                done_line += ",\"drafted\":" + String(wst.n_drafted) + ",\"accepted\":" + String(wst.n_accepted) + ",\"k\":" + String(kcfg)
            print(done_line + "}")
            continue

        # briefs/2026-09-16-sampling-all-models-lane.md item 1: the draft-head
        # receipt below is dense-only (MEGA_ALLOWED gates it off for
        # qwen35moe), so it never captures a real logits row for the MoE
        # sampler gates. This dumps the last decode step's target row
        # (b.logits_d row 0, the same buffer argmax_k/sample_row_k read from
        # in window.mojo) on either profile. Off by default, unconditional
        # on the profile, no effect on any existing path.
        var dump_logits_path = getenv("BARO_DUMP_LOGITS", "")
        if dump_logits_path != "":
            var lg_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
            ctx.enqueue_copy(
                dst_buf=lg_h,
                src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False),
            )
            ctx.synchronize()
            with open(dump_logits_path, "w") as f:
                var p = lg_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                f.write_bytes(Span[UInt8](unsafe_ptr=p, length=VOCAB * 4))
            print("dumped final logits row to", dump_logits_path)

        if not MEGA_ALLOWED:
            return

        # MTP (NextN) draft head receipt, blk.32: last generated token paired with
        # the last trunk hidden row, attended at position 0 (arm A: empty draft
        # KV, identical to the 2026-09-01 validation dump; arm B: draft KV holds
        # the run, so only arm A's DRAFT line is the receipt).
        var hn_last = DeviceBuffer[f32](ctx, bufs.hn_d.unsafe_ptr(), H, owning=False)
        blk32_forward(ctx, wbuf, off, e, 1, 0, n_total - 1, True, hn_last,
            bufs.x_d, bufs.curb_d, bufs.qf_d, bufs.q_d, bufs.k_d, bufs.v_d, bufs.gate_d, bufs.ao_d, bufs.resb_d, bufs.fgb_d, bufs.p_qf_d, bufs.p_kv_d, bufs.p_h_d,
            bufs.p_ffn_d, bufs.p_ffn2_d, bufs.p_v_d, bufs.logits_d, bufs.cc_d, bufs.de_d, bufs.hd_d, bufs.kc32_d, bufs.vc32_d, toks_d, bufs.dtok_d,
            False, wst.p3, False, 0, pack_q4)
        var last_tok = generated[len(generated) - 1]
        ctx.synchronize()
        var draft_logits_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
        ctx.enqueue_copy(
            dst_buf=draft_logits_h,
            src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False),
        )
        # h_nextn is dumped too: tools/draft-ref.py's numpy reference needs the
        # EXACT f32 hidden state the GPU consumed, or a mismatch says nothing about
        # correctness -- it would just mean the two sides were fed different inputs.
        var hn_h = ctx.enqueue_create_host_buffer[f32](H)
        ctx.enqueue_copy(
            dst_buf=hn_h,
            src_buf=DeviceBuffer[f32](ctx, bufs.hn_d.unsafe_ptr(), H, owning=False),
        )
        var dtok1_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(
            dst_buf=dtok1_h,
            src_buf=DeviceBuffer[DType.int32](ctx, bufs.dtok_d.unsafe_ptr(), 1, owning=False),
        )
        ctx.synchronize()

        with open(".work/draft-logits.bin", "w") as df:
            var dp = draft_logits_h.unsafe_ptr().unsafe_bitcast[UInt8]()
            var dspan = Span[UInt8](unsafe_ptr=dp, length=VOCAB * 4)
            df.write_bytes(dspan)

        with open(".work/draft-hn.bin", "w") as hf:
            var hp = hn_h.unsafe_ptr().unsafe_bitcast[UInt8]()
            var hspan = Span[UInt8](unsafe_ptr=hp, length=H * 4)
            hf.write_bytes(hspan)

        print("DRAFT: from_token", last_tok, "draft_argmax", Int(dtok1_h[0]))
        break
