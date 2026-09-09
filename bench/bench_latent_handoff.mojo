# E8 HARNESS -- 5-arm dual-engine evaluator (exchange/e8-lane-plan-2026-09-09.md,
# item HARNESS). Two co-resident engine instances (A = producer, B = receiver)
# run each bench/data/e8_tasks.json item under one or more of:
#   0        B answers the prompt alone, greedy.
#   T        A emits <=300 tokens of CoT text; B answers prompt+text.
#   L8-raw   A feeds 8 raw h_L vectors back to itself and to B (bench_hidden_dtype
#            step_latent_raw, unmodified); B ingests them as a prefix, answers.
#   L8-soft  same, each vector passed through serve/realign.mojo first.
#   L32-soft same, 32 steps.
# Soft arms call serve/realign.mojo's stub, which raises "REALIGN not merged"
# until that lane's kernel lands -- caught per-arm, recorded as an error, never
# silently downgraded to the raw vector.
#
# Writes results/e8/<out-prefix>.raw.json: schema validity (json tasks) is
# computed here via grammar/'s Automaton/Matcher on B's decoded answer text;
# arithmetic extraction and JSON exact-match live in bench/e8_score.py, which
# turns the raw file into the final <out-prefix>.json / .md (bench/latent-handoff.sh).
from std.memory import ArcPointer
from std.os import getenv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer

from registry import *
from window import *
from latent_harness import load_pack, alloc_bufs, Pack
from realign import realign_expected_embedding
from tokenizer import Tokenizer

from grammar.automaton import Automaton
from grammar.json_value import parse_json_file, JSONDoc, JSONValue, JKindString, JKindNumber, JKindArray, JKindObject
from grammar.json_schema import compile_root_schema, str_bytes
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, TrieNode, build_trie
from grammar.matcher import Matcher

comptime COT_MAX = 300
comptime K8 = 8
comptime K32 = 32


def get_str(doc: JSONDoc, idx: Int, key: String) -> String:
    var vi = doc.get_field(idx, key)
    if vi < 0:
        return String("")
    return doc.get(vi).s


def get_int_list(doc: JSONDoc, idx: Int, key: String) -> List[Int]:
    var out = List[Int]()
    var vi = doc.get_field(idx, key)
    if vi < 0:
        return out^
    var v = doc.get(vi)
    for j in range(len(v.arr)):
        out.append(Int(doc.get(v.arr[j]).n))
    return out^


def contains(xs: List[Int], v: Int) -> Bool:
    for i in range(len(xs)):
        if xs[i] == v:
            return True
    return False


def trim_at_stop(ids: List[Int], stops: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(ids)):
        if contains(stops, ids[i]):
            break
        out.append(ids[i])
    return out^


def strip_ws(s: String) -> String:
    var b = s.as_bytes()
    var lo = 0
    var hi = len(b)
    while lo < hi and (b[lo] == 32 or b[lo] == 9 or b[lo] == 10 or b[lo] == 13):
        lo += 1
    while hi > lo and (b[hi - 1] == 32 or b[hi - 1] == 9 or b[hi - 1] == 10 or b[hi - 1] == 13):
        hi -= 1
    return String(StringSlice(unsafe_from_utf8=Span(b)[lo:hi]))


@fieldwise_init
struct GenResult(Copyable, Movable):
    var ids: List[Int]
    var elapsed_s: Float64


def reset_and_load(ctx: DeviceContext, mut b: WindowBufs, context: List[Int], tmax: Int) raises -> Tuple[Int, Int]:
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
    for i in range(len(context)):
        toks_h[i] = Int32(context[i])
    ctx.enqueue_copy(dst_buf=b.toks_d, src_buf=toks_h)
    ctx.synchronize()
    var pf_rows = 0
    var pf_tail = 0
    if len(context) - 1 >= PF_MIN:
        pf_rows = len(context) - 1
        pf_tail = pf_rows % MROWS
        if pf_tail == 0:
            pf_tail = MROWS
    return (pf_rows, pf_tail)


def make_cfg(pack_q4: Bool, q4_off: Int, e: Int, pf_rows: Int, pf_tail: Int, n_prompt: Int, n_total: Int) -> WindowCfg:
    return WindowCfg(
        pack_q4=pack_q4, draft_q4=False, q4_off=q4_off, e=e, kcfg=2,
        spec=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
        dump=False, mega=True, att_split=n_total, mega_win=False, dot3=False, pf_chunk=CP,
        pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, n_prompt=n_prompt,
    )


def read_toks(ctx: DeviceContext, b: WindowBufs, lo: Int, hi: Int, tmax: Int) raises -> List[Int]:
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=b.toks_d)
    ctx.synchronize()
    var out = List[Int]()
    for i in range(lo, hi):
        out.append(Int(toks_h[i]))
    return out^


def run_fresh_generate(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, context: List[Int], gen_max: Int, tmax: Int,
) raises -> GenResult:
    var plen = len(context)
    var pf = reset_and_load(ctx, b, context, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], plen, plen + gen_max)
    var t0 = perf_counter_ns()
    wst.reset(t0)
    while wst.pos < plen + gen_max - 1:
        step_window(ctx, b, cfg, wst)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var generated = read_toks(ctx, b, plen, plen + gen_max, tmax)
    return GenResult(generated^, dt)


def run_to_prompt_end(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, context: List[Int], tmax: Int,
) raises -> WindowCfg:
    var plen = len(context)
    var pf = reset_and_load(ctx, b, context, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], plen, plen + 1)
    wst.reset(perf_counter_ns())
    while wst.pos < plen:
        step_window(ctx, b, cfg, wst)
    return cfg^


def step_latent_raw(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_dev_vec: DeviceBuffer[f32],
) raises:
    var x_slice = DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False)
    ctx.enqueue_copy(dst_buf=x_slice, src_buf=latent_dev_vec)

    var Xm = TileTensor(b.x_d, xm_layout)
    var CurBm = TileTensor(b.curb_d, xm_layout)
    var Toks = TileTensor(b.toks_d, toks_layout)
    var ConvStateAll = TileTensor(b.convstate_d, csall_layout)
    var SStateAll = TileTensor(b.sstate_d, ssall_layout)
    var Hnm0 = TileTensor(b.hn_d, xm_layout)
    var Dtok0 = TileTensor(b.dtok_d, dtok_layout)

    ctx.enqueue_function[mega_token_q4_k](
        b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
        TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
        TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
        TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
        TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
        TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
        TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(),
        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(0), Int32(1), Int32(cfg.att_split),
        grid_dim=MEGA_G, block_dim=ROW_THREADS,
    )
    st.ring = (st.ring + 1) % SLOTS
    st.pos_prev = st.pos
    st.pos += 1


def collect_latent_raw(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_host: HostBuffer[f32], k: Int,
) raises:
    for s in range(k):
        var h_slice = DeviceBuffer[f32](ctx, b.hn_d.unsafe_ptr(), H, owning=False)
        var latent_dev = ctx.enqueue_create_buffer[f32](H)
        var host_dst = latent_host.create_sub_buffer[f32](s * H, H)
        ctx.enqueue_copy(dst_buf=host_dst, src_buf=h_slice)
        ctx.enqueue_copy(dst_buf=latent_dev, src_buf=h_slice)
        ctx.synchronize()
        step_latent_raw(ctx, b, cfg, st, latent_dev)


def collect_latent_soft(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_host: HostBuffer[f32], k: Int,
) raises:
    # never falls back to raw hn_d on failure -- the caller catches the raise
    # (REALIGN not merged) and records the arm as errored.
    for s in range(k):
        var e_dev = ctx.enqueue_create_buffer[f32](H)
        realign_expected_embedding(ctx, b, e_dev)
        var host_dst = latent_host.create_sub_buffer[f32](s * H, H)
        ctx.enqueue_copy(dst_buf=host_dst, src_buf=e_dev)
        ctx.synchronize()
        step_latent_raw(ctx, b, cfg, st, e_dev)


def apply_latent_to_receiver(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_host: HostBuffer[f32], k: Int,
) raises:
    for s in range(k):
        var latent_dev = ctx.enqueue_create_buffer[f32](H)
        var host_src = latent_host.create_sub_buffer[f32](s * H, H)
        ctx.enqueue_copy(dst_buf=latent_dev, src_buf=host_src)
        ctx.synchronize()
        step_latent_raw(ctx, b, cfg, st, latent_dev)


def greedy_tokenize(trie: TokenTrie, text_bytes: List[UInt8]) raises -> List[Int]:
    var out: List[Int] = []
    var pos = 0
    var n = len(text_bytes)
    while pos < n:
        var node_idx = trie.root_child(text_bytes[pos])
        if node_idx < 0:
            raise Error("no vocab token starts with byte at pos " + String(pos))
        var best_end = -1
        var best_token = -1
        if trie.nodes[Int(node_idx)].token_id >= 0:
            best_end = pos + 1
            best_token = Int(trie.nodes[Int(node_idx)].token_id)
        var cur = Int(node_idx)
        var i = pos + 1
        while i < n:
            var nxt = trie.nodes[cur].find_child(text_bytes[i])
            if nxt < 0:
                break
            cur = Int(nxt)
            i += 1
            if trie.nodes[cur].token_id >= 0:
                best_end = i
                best_token = Int(trie.nodes[cur].token_id)
        if best_token < 0:
            raise Error("cannot tokenize at pos " + String(pos))
        out.append(best_token)
        pos = best_end
    return out^


def check_schema_valid(schema_path: String, sample: String, vp: ArcPointer[Vocab], tp: ArcPointer[TokenTrie]) -> Bool:
    try:
        var doc = parse_json_file(schema_path)
        var a = Automaton()
        var rid = compile_root_schema(a, doc)
        var m = Matcher(a^, tp, vp)
        var toks = greedy_tokenize(tp[], str_bytes(strip_ws(sample)))
        for i in range(len(toks)):
            if not m.accept(toks[i]):
                return False
        return m.is_terminated()
    except:
        return False


def json_escape(s: String) -> String:
    var out = List[UInt8]()
    for c in s.as_bytes():
        if c == 34:
            out.append(92); out.append(34)
        elif c == 92:
            out.append(92); out.append(92)
        elif c == 10:
            out.append(92); out.append(110)
        elif c == 13:
            out.append(92); out.append(114)
        elif c == 9:
            out.append(92); out.append(116)
        elif c < 32:
            pass
        else:
            out.append(c)
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def ids_to_json(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i > 0:
            s += ","
        s += String(ids[i])
    s += "]"
    return s


def arm_json(name: String, producer_s: Float64, receiver_s: Float64, ids: List[Int], text: String, schema_valid: Int, error: String) -> String:
    var s = String("{\"arm\":\"") + name + "\",\"producer_s\":" + String(producer_s)
    s += ",\"receiver_s\":" + String(receiver_s)
    s += ",\"generated_ids\":" + ids_to_json(ids)
    s += ",\"answer_text\":\"" + json_escape(text) + "\""
    if schema_valid < 0:
        s += ",\"schema_valid\":null"
    else:
        s += ",\"schema_valid\":" + ("true" if schema_valid == 1 else "false")
    if error == "":
        s += ",\"error\":null"
    else:
        s += ",\"error\":\"" + json_escape(error) + "\""
    s += "}"
    return s


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var args = argv()
    var n_items = 40
    var arms_arg = String("0,T,L8-raw,L8-soft,L32-soft")
    var out_prefix = String("results/e8/topology1-q4")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--items" and i + 1 < len(args):
            n_items = atol(String(args[i + 1]))
            i += 2
        elif a == "--arms" and i + 1 < len(args):
            arms_arg = String(args[i + 1])
            i += 2
        elif a == "--out" and i + 1 < len(args):
            out_prefix = String(args[i + 1])
            i += 2
        else:
            i += 1
    var arms = arms_arg.split(",")
    var ans_max = atol(getenv("BARO_E8_ANS_MAX", "128"))
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var gguf_path = getenv(
        "BARO_E8_GGUF",
        "$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
    )
    # longest arm needs prompt + max(COT_MAX, K32) + ans_max; e8_tasks.json's
    # longest prompt is 98 tokens (measured) -- 640 leaves headroom without
    # paying registry.TMAX=1088's full KV-cache footprint. Two co-resident
    # engines at TMAX measured 25.48/25.75 GB used (bench/latent-handoff.sh
    # smoke run, 2026-09-09) -- 270 MB of headroom on a 24 GB card; this cuts
    # the KV pool this harness allocates (per engine, tpages=ceildiv(tmax,128))
    # without touching serve/window.mojo's own KVPAGE/TMAX.
    var tmax = atol(getenv("BARO_E8_TMAX", "640"))

    print("E8 HARNESS: items=", n_items, " arms=", arms_arg, " ans_max=", ans_max, " pack=", packdir)

    var ctx = DeviceContext()

    print("--- loading producer (A) pack ---")
    var packA = load_pack(ctx, packdir)
    var bufsA = alloc_bufs(ctx, packA, tmax)

    print("--- loading receiver (B) pack ---")
    var packB = load_pack(ctx, packdir)
    var bufsB = alloc_bufs(ctx, packB, tmax)

    # Marker for the wrapper script: both packs are resident now, so this is
    # the moment to read VRAM back -- a reading taken after this process exits
    # would just re-measure baseline (bench/latent-handoff.sh polls for this).
    with open(".work/e8-vram-ready.marker", "w") as f:
        f.write_bytes(String("1").as_bytes())

    var pack_q4 = packA.pack_q4
    var q4_off = packA.q4_off
    var eA = packA.e
    var eB = packB.e

    print("--- loading tokenizer + grammar vocab ---")
    var tok = Tokenizer(gguf_path)
    var stops = List[Int]()
    if tok.eos_id >= 0:
        stops.append(tok.eos_id)
    var im_end = tok.tok2id.get("<|im_end|>", -1)
    if im_end >= 0 and not contains(stops, im_end):
        stops.append(im_end)

    var vocab = load_vocab(packdir)
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    var wstA = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0])
    var wstB = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0])

    var doc = parse_json_file("bench/data/e8_tasks.json")
    var root = doc.get(doc.root)
    var n_avail = len(root.arr)
    var n_run = n_items if n_items < n_avail else n_avail

    var out = String("{\"topology\":\"topology1-q4\",\"pack\":\"") + packdir + "\",\"transport\":\"in-process host copy (memfd transport is E7's, not measured here)\",\"items\":["
    var first_item = True

    for it in range(n_run):
        var idx = root.arr[it]
        var task_id = get_str(doc, idx, "id")
        var task_type = get_str(doc, idx, "type")
        var schema_file = get_str(doc, idx, "schema_file")
        var tokens = get_int_list(doc, idx, "tokens")
        print("=== item", task_id, "(" + task_type + ") n_tok=" + String(len(tokens)), "===")

        if not first_item:
            out += ","
        first_item = False
        out += "{\"id\":\"" + json_escape(task_id) + "\",\"type\":\"" + json_escape(task_type) + "\",\"arms\":["
        var first_arm = True

        for arm_i in range(len(arms)):
            var arm = String(arms[arm_i])
            if not first_arm:
                out += ","
            first_arm = False
            print("  arm", arm)

            try:
                if arm == "0":
                    var r = run_fresh_generate(ctx, bufsB, wstB, pack_q4, q4_off, eB, tokens, ans_max, tmax)
                    var ans_ids = trim_at_stop(r.ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    if task_type == "json":
                        sv = 1 if check_schema_valid(schema_file, text, vp, tp) else 0
                    out += arm_json(arm, 0.0, r.elapsed_s, r.ids, text, sv, "")

                elif arm == "T":
                    var rA = run_fresh_generate(ctx, bufsA, wstA, pack_q4, q4_off, eA, tokens, COT_MAX, tmax)
                    var cot_ids = trim_at_stop(rA.ids, stops)
                    var b_context = tokens.copy()
                    for j in range(len(cot_ids)):
                        b_context.append(cot_ids[j])
                    var rB = run_fresh_generate(ctx, bufsB, wstB, pack_q4, q4_off, eB, b_context, ans_max, tmax)
                    var ans_ids = trim_at_stop(rB.ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    if task_type == "json":
                        sv = 1 if check_schema_valid(schema_file, text, vp, tp) else 0
                    out += arm_json(arm, rA.elapsed_s, rB.elapsed_s, rB.ids, text, sv, "")

                elif arm == "L8-raw" or arm == "L8-soft" or arm == "L32-soft":
                    var k = K32 if arm == "L32-soft" else K8
                    var latent_hA = ctx.enqueue_create_host_buffer[f32](k * H)
                    ctx.synchronize()
                    var t0a = perf_counter_ns()
                    var cfgA = run_to_prompt_end(ctx, bufsA, wstA, pack_q4, q4_off, eA, tokens, tmax)
                    if arm == "L8-raw":
                        collect_latent_raw(ctx, bufsA, cfgA, wstA, latent_hA, k)
                    else:
                        collect_latent_soft(ctx, bufsA, cfgA, wstA, latent_hA, k)
                    ctx.synchronize()
                    var producer_s = Float64(perf_counter_ns() - t0a) / 1e9

                    var t0b = perf_counter_ns()
                    var cfgB = run_to_prompt_end(ctx, bufsB, wstB, pack_q4, q4_off, eB, tokens, tmax)
                    apply_latent_to_receiver(ctx, bufsB, cfgB, wstB, latent_hA, k)
                    var start_pos = len(tokens) + k
                    var cfg_gen = make_cfg(pack_q4, q4_off, eB, 0, 0, start_pos, start_pos + ans_max)
                    while wstB.pos < start_pos + ans_max - 1:
                        step_window(ctx, bufsB, cfg_gen, wstB)
                    ctx.synchronize()
                    var receiver_s = Float64(perf_counter_ns() - t0b) / 1e9
                    var gen_ids = read_toks(ctx, bufsB, start_pos, start_pos + ans_max, tmax)
                    var ans_ids = trim_at_stop(gen_ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    if task_type == "json":
                        sv = 1 if check_schema_valid(schema_file, text, vp, tp) else 0
                    out += arm_json(arm, producer_s, receiver_s, gen_ids, text, sv, "")

                else:
                    out += arm_json(arm, 0.0, 0.0, List[Int](), "", -1, "unknown arm " + arm)
            except e:
                out += arm_json(arm, 0.0, 0.0, List[Int](), "", -1, String(e))

        out += "]}"

    out += "]}"

    with open(out_prefix + ".raw.json", "w") as f:
        f.write_bytes(out.as_bytes())
    print("wrote", out_prefix + ".raw.json")
