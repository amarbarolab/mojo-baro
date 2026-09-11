"""baro engine: full-model greedy decode for qwen35 (Qwythos-9B), milestone 4.

Loads .work/engine-pack/ (fixed tensor order, 2D bf16 weights pre-transposed
to B-layout), reads prompt token ids from .work/engine-pack/prompt-tokens.txt,
runs the 32-block hybrid stack (24 gated-delta-net + 8 gated full-attention,
MTP block skipped) over a window of up to MROWS tokens at a time, and prints
greedy token ids.

Parity target: byte-identical token ids vs llama.cpp on the same GGUF.
"""
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.memory import memcpy
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


def read_line(fd: Int) raises -> Optional[String]:
    # One line from fd, newline stripped; None at EOF with nothing buffered.
    var buf = List[UInt8]()
    var b = List[UInt8](unsafe_uninit_length=1)
    while True:
        var n = external_call["read", c_ssize_t](fd, b.unsafe_ptr(), 1)
        if n <= 0:
            if len(buf) == 0:
                return None
            break
        if b[0] == 10:
            break
        buf.append(b[0])
    return String(from_utf8=Span[UInt8](buf))


def cancel_pending(fd: Int, req_id: Int) raises -> Bool:
    # Non-blocking check for a "{"cancel":ID}" line on fd, once per
    # step_window call. poll(fd, POLLIN, 0) never blocks; a hit means the
    # writer's single write() of a short line is already in the pipe, so the
    # blocking read_line below will not stall on a torn line.
    var pfd = List[UInt8](unsafe_uninit_length=8)
    var p = pfd.unsafe_ptr()
    p.unsafe_bitcast[Int32]().unsafe_offset(0)[] = Int32(fd)
    p.unsafe_bitcast[Int16]().unsafe_offset(2)[] = Int16(1)  # events = POLLIN
    p.unsafe_bitcast[Int16]().unsafe_offset(3)[] = Int16(0)  # revents
    var r = external_call["poll", Int32](p, UInt64(1), Int32(0))
    if r <= 0 or (Int(p.unsafe_bitcast[Int16]().unsafe_offset(3)[]) & 1) == 0:
        return False
    var line_in = read_line(fd)
    if not line_in:
        return False
    var line = line_in.value()
    var i = json_key(line, "cancel")
    var cid = 0
    if i < 0 or not json_int(line, i, cid):
        return False
    return cid == req_id


def json_key(line: String, key: String) -> Int:
    # Index of the first byte of the value for "key": ..., or -1.
    var i = line.find(String("\"") + key + "\"")
    if i < 0:
        return -1
    var b = line.as_bytes()
    i += key.byte_length() + 2
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    if i >= len(b) or b[i] != 58:
        return -1
    i += 1
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    return i


def json_int(line: String, mut i: Int, mut v: Int) -> Bool:
    var b = line.as_bytes()
    var neg = False
    if i < len(b) and b[i] == 45:
        neg = True
        i += 1
    var have = False
    v = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        v = v * 10 + Int(b[i] - 48)
        i += 1
        have = True
    if neg:
        v = -v
    return have


def json_float(line: String, mut i: Int, mut v: Float64) -> Bool:
    # Plain decimal (sign, digits, optional '.', digits); no exponent form --
    # none of the sampler fields need one.
    var b = line.as_bytes()
    var neg = False
    if i < len(b) and b[i] == 45:
        neg = True
        i += 1
    var have = False
    var ip: Float64 = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        ip = ip * 10 + Float64(Int(b[i] - 48))
        i += 1
        have = True
    var frac: Float64 = 0
    if i < len(b) and b[i] == 46:
        i += 1
        var scale: Float64 = 1
        while i < len(b) and b[i] >= 48 and b[i] <= 57:
            scale /= 10
            frac += Float64(Int(b[i] - 48)) * scale
            i += 1
            have = True
    v = ip + frac
    if neg:
        v = -v
    return have


@fieldwise_init
struct SampleParams(Copyable, Movable):
    # C3 (bench/chat-protocol.md): parsed from the wire, not yet acted on --
    # the live decode loop still always takes the greedy/MTP path. Ready for
    # serve/sample_ref.mojo (host reference) or kernels/sample.mojo (device,
    # lane-KSAMP) to read once either is wired in. temperature <= 0 means
    # "off" throughout, matching both references' own convention.
    var temperature: Float64
    var top_p: Float64
    var top_k: Int
    var min_p: Float64
    var seed: UInt64
    var presence_penalty: Float64
    var frequency_penalty: Float64


def default_sample_params() -> SampleParams:
    return SampleParams(temperature=0, top_p=1.0, top_k=0, min_p=0, seed=0, presence_penalty=0, frequency_penalty=0)


def parse_request(
    line: String, mut id: Int, mut prompt: List[Int], mut n: Int, mut spec: Bool, mut has_spec: Bool, mut stop: List[List[Int]], mut ckpt: List[Int], mut sample: SampleParams
) -> String:
    # {"id":INT,"prompt":[INT,...],"n":INT,"spec":BOOL,"stop":[[INT,...],...],
    #  "ckpt":[INT,...],"temperature":FLOAT,"top_p":FLOAT,"top_k":INT,
    #  "min_p":FLOAT,"seed":INT,"presence_penalty":FLOAT,
    #  "frequency_penalty":FLOAT}; everything past prompt/n optional. Returns
    # "" on success, else the error text (id is set when it parsed).
    id = 0
    var i = json_key(line, "id")
    if i < 0 or not json_int(line, i, id):
        return "missing or non-integer id"
    i = json_key(line, "n")
    if i < 0 or not json_int(line, i, n):
        return "missing or non-integer n"
    i = json_key(line, "prompt")
    var b = line.as_bytes()
    if i < 0 or i >= len(b) or b[i] != 91:
        return "missing prompt array"
    i += 1
    while True:
        while i < len(b) and (b[i] == 32 or b[i] == 44):
            i += 1
        if i >= len(b):
            return "unterminated prompt array"
        if b[i] == 93:
            break
        var v = 0
        if not json_int(line, i, v) or v < 0:
            return "prompt must hold non-negative integers"
        prompt.append(v)
    has_spec = False
    i = json_key(line, "spec")
    if i >= 0:
        if line.as_bytes()[i] == 116:
            spec = True
            has_spec = True
        elif line.as_bytes()[i] == 102:
            spec = False
            has_spec = True
        else:
            return "spec must be true or false"
    var si = json_key(line, "stop")
    if si >= 0:
        if si >= len(b) or b[si] != 91:
            return "stop must be an array of arrays"
        si += 1
        while True:
            while si < len(b) and (b[si] == 32 or b[si] == 44):
                si += 1
            if si >= len(b):
                return "unterminated stop array"
            if b[si] == 93:
                break
            if b[si] != 91:
                return "stop entries must be arrays of token ids"
            si += 1
            var seq = List[Int]()
            while True:
                while si < len(b) and (b[si] == 32 or b[si] == 44):
                    si += 1
                if si >= len(b):
                    return "unterminated stop sequence"
                if b[si] == 93:
                    si += 1
                    break
                var v2 = 0
                if not json_int(line, si, v2) or v2 < 0:
                    return "stop sequence must hold non-negative integers"
                seq.append(v2)
            stop.append(seq^)
    var ci = json_key(line, "ckpt")
    if ci >= 0:
        if ci >= len(b) or b[ci] != 91:
            return "ckpt must be an array of integers"
        ci += 1
        while True:
            while ci < len(b) and (b[ci] == 32 or b[ci] == 44):
                ci += 1
            if ci >= len(b):
                return "unterminated ckpt array"
            if b[ci] == 93:
                break
            var v3 = 0
            if not json_int(line, ci, v3) or v3 < 0:
                return "ckpt must hold non-negative integers"
            ckpt.append(v3)
    var fi = json_key(line, "temperature")
    if fi >= 0:
        var fv: Float64 = 0
        if not json_float(line, fi, fv):
            return "temperature must be a number"
        sample.temperature = fv
    fi = json_key(line, "top_p")
    if fi >= 0:
        var fv2: Float64 = 0
        if not json_float(line, fi, fv2):
            return "top_p must be a number"
        sample.top_p = fv2
    fi = json_key(line, "top_k")
    if fi >= 0:
        var iv = 0
        if not json_int(line, fi, iv):
            return "top_k must be an integer"
        sample.top_k = iv
    fi = json_key(line, "min_p")
    if fi >= 0:
        var fv3: Float64 = 0
        if not json_float(line, fi, fv3):
            return "min_p must be a number"
        sample.min_p = fv3
    fi = json_key(line, "seed")
    if fi >= 0:
        var iv2 = 0
        if not json_int(line, fi, iv2) or iv2 < 0:
            return "seed must be a non-negative integer"
        sample.seed = UInt64(iv2)
    fi = json_key(line, "presence_penalty")
    if fi >= 0:
        var fv4: Float64 = 0
        if not json_float(line, fi, fv4):
            return "presence_penalty must be a number"
        sample.presence_penalty = fv4
    fi = json_key(line, "frequency_penalty")
    if fi >= 0:
        var fv5: Float64 = 0
        if not json_float(line, fi, fv5):
            return "frequency_penalty must be a number"
        sample.frequency_penalty = fv5
    return ""




def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
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
    var dot3 = getenv("BARO_DOT", "0") == "1" and not pack_q4
    print("BARO_DOT:", dot3)
    print("pack q4 trunk:", pack_q4)
    var mega = getenv("BARO_MEGA", "1") == "1"
    print("BARO_MEGA:", mega)
    var mega_win = getenv("BARO_MEGA_WIN", "0") == "1"
    print("BARO_MEGA_WIN:", mega_win)
    var pf5 = getenv("BARO_PROFILE", "0") == "5"
    var dump_path = getenv("BARO_DUMP", "")
    var dump = dump_path != ""

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
    var pf_on = getenv("BARO_PREFILL", "1") == "1"
    var spec_env = getenv("BARO_SPEC", "0") == "1"
    print("BARO_SPEC:", spec_env)
    var spec_dbg = getenv("BARO_SPEC_DBG", "0") == "1"
    var bufs = alloc_bufs(ctx, pack, tmax)
    var toks_d = bufs.toks_d

    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0])
    var ckpt_cap = atol(getenv("BARO_CKPT", "8")) if serve else 0
    if ckpt_cap < 0:
        ckpt_cap = 0
    var chain = Chain(ctx, ckpt_cap, packdir)
    print("checkpoints: cap", ckpt_cap, ", bytes", Float64(CKPT_BYTES) / 1e6, "MB each, period", CKPT_PERIOD)
    var req_id = 0
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
        var stop_seqs = List[List[Int]]()
        var ckpt_hints = List[Int]()
        var sample = default_sample_params()
        if serve:
            var line_in = read_line(0)
            if not line_in:
                break
            var req_n = 0
            var req_spec = False
            var req_has_spec = False
            var perr = parse_request(line_in.value(), req_id, prompt, req_n, req_spec, req_has_spec, stop_seqs, ckpt_hints, sample)
            if perr == "" and len(prompt) < 1:
                perr = "empty prompt"
            if perr == "" and req_n < 1:
                perr = "n must be >= 1"
            if perr == "" and len(prompt) + req_n > tmax:
                perr = "prompt+n exceeds TMAX " + String(tmax)
            if perr != "":
                print(err_line(req_id, perr))
                continue
            gen_n = req_n
            if req_has_spec:
                spec = req_spec
            print("prompt tokens:", len(prompt), " n:", gen_n, " spec:", spec)
            # M1a prefix checkpoint: restore the SSM slot for the longest
            # hashed prefix and keep the KV pool (position addressed, [0, cached)
            # still in place); the draft KV is never prefilled, so it is zeroed
            # exactly as on the cold path. A miss is today's path.
            # spec mode replays at least one prompt row so the draft head's
            # hidden rows (hn_d) are fresh: the checkpoint at len-1 is skipped.
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

        var pf_rows = 0
        var pf_tail = 0
        if pf_on and len(prompt) - 1 - cached >= PF_MIN:
            pf_rows = len(prompt) - 1
            pf_tail = (pf_rows - cached) % MROWS
            if pf_tail == 0:
                pf_tail = MROWS
        var prefill_rows = len(prompt) - 1 - cached
        print("TMAX:", tmax, " kv dtype:", String(KVT), " prefill chunk:", pf_chunk, " prefill rows:", pf_rows, " cached:", cached, " replay rows:", prefill_rows)

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
        var cfg = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, e=e, kcfg=kcfg, spec=spec, spec_dbg=spec_dbg, serve=serve, req_id=req_id, prof=prof, pf2=pf2, pf3=pf3, pf4=pf4, dump=dump, mega=mega, att_split=att_split, mega_win=mega_win, dot3=dot3, pf_chunk=pf_chunk, pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, n_prompt=len(prompt))
        wst.reset(t0)
        wst.pos = cached
        wst.pos_prev = cached
        var prefill_done = False
        var cancelled = False
        var stopped = False
        # The stopwatch stays here, in the harness that is never embedded in a
        # gguf: step_window cannot reach t0, t_prefill_end or dt (P-A, 2026-09-08).
        while wst.pos < n_total - 1:
            if len(ckpt_hints) > 0 and wst.pos < pf_rows:
                # A hint may fall inside what would otherwise be one big
                # prefill chunk; cap this call's chunk so wst.pos actually
                # stops there (checkpoints are only taken between calls).
                var step_cfg = cfg.copy()
                step_cfg.pf_chunk = min(cfg.pf_chunk, next_ckpt_stop(ckpt_hints, wst.pos, pf_rows) - wst.pos)
                step_window(ctx, bufs, step_cfg, wst)
            else:
                step_window(ctx, bufs, cfg, wst)
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
            if serve and cancel_pending(0, req_id):
                cancelled = True
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
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        print("host_enqueue_s:", t_host, " gpu_total_s:", dt)
        var flw = ctx.enqueue_create_host_buffer[DType.uint32](3)
        ctx.enqueue_copy(dst_buf=flw, src_buf=bufs.ctr_d)
        ctx.synchronize()
        print("mega fail word:", flw[2], "" if flw[2] == 0 else " NOT-RESIDENT: a grid barrier timed out, tokens after it are invalid")
        if dump:
            ctx.synchronize()
            with open(dump_path, "w") as f:
                var dpp = bufs.dump_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                f.write_bytes(Span[UInt8](unsafe_ptr=dpp, length=wst.n_dumped * 2 * N_LAYERS * H * 4))
            print("dumped", wst.n_dumped, "tokens x", N_LAYERS, "layers to", dump_path)
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
