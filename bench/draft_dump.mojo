# A4 draft-head data + parity dump (bench/draft-head-protocol.md).
#
# --mode dump: real held-out text, one token at a time through step_window
# (the same per-token loop bench_latent_handoff.run_to_prompt_end already
# runs), reading serve/realign.final_norm_hidden after each step. That is
# model.norm(residual) at the position just processed, i.e. h_i (row i of
# the dump = h AFTER processing tokens[i]): the same tensor blk32_forward
# consumes as hsrc, one step later (blk32's own call is pos==tok_pos==P,
# hsrc=h_{P-1}, predicting Toks[P+1] -- confirmed against both real call
# sites in serve/window.mojo and the self-chained draft loop's own
# `tokcp_k(..., st.pos+j+1, ...)` write-back). So for training: row h[i]
# pairs with INPUT token tokens[i+1] (P=i+1) and LABEL tokens[i+2] (P+1),
# valid for i in 0..n-3. No collect_latent_raw here: that function runs a
# k-step self-feeding latent rollout (E13's own out-of-manifold input), not
# a read of the trunk's real per-position hidden state, and does not apply
# to this lane (bench/draft-head-protocol.md).
#
# --mode parity: for every interior position P, calls blk32_forward
# directly (m=1, pos=tok_pos=P, do_head=True) with hsrc=h_{P-1}, the exact
# call shape step_window's own spec branch uses, and records the real
# engine's own drafted argmax (dtok_d[0]) alongside h_{P-1} and the true
# next token Toks[P+1]. tools/mtp_head.py --mode parity compares its torch
# replica's argmax against this file before any training gradient is
# trusted.
from std.memory import bitcast
from std.os import getenv
from std.sys import argv, has_accelerator

from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer

from registry import H, f32
from window import WindowBufs, WindowState, WindowCfg, blk32_forward, step_window
from realign import final_norm_hidden
from bench_latent_handoff import run_to_prompt_end, reset_and_load, make_cfg
from harness import load_pack, alloc_bufs, Pack
from grammar.automaton import Bitset


def write_u32(mut f: FileHandle, v: Int) raises:
    var x = Int32(v)
    var b = List[UInt8]()
    b.append(UInt8(x & 0xFF)); b.append(UInt8((x >> 8) & 0xFF))
    b.append(UInt8((x >> 16) & 0xFF)); b.append(UInt8((x >> 24) & 0xFF))
    f.write_bytes(Span(b))


def write_f32_vec(mut f: FileHandle, v: HostBuffer[f32], n: Int) raises:
    var b = List[UInt8](unsafe_uninit_length=n * 4)
    for i in range(n):
        var bits = bitcast[DType.uint32, 1](v[i])[0]
        b[i * 4 + 0] = UInt8(bits & 0xFF)
        b[i * 4 + 1] = UInt8((bits >> 8) & 0xFF)
        b[i * 4 + 2] = UInt8((bits >> 16) & 0xFF)
        b[i * 4 + 3] = UInt8((bits >> 24) & 0xFF)
    f.write_bytes(Span(b))


def dump_document_stepwise(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, tokens: List[Int], tmax: Int,
    mut out: FileHandle,
) raises:
    var n = len(tokens)
    var pf = reset_and_load(ctx, b, tokens, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], n, n + 1)
    wst.reset(0)
    write_u32(out, n)
    for i in range(n):
        write_u32(out, tokens[i])
    var hbuf = ctx.enqueue_create_host_buffer[f32](H)
    var n_rows = n - 1
    if n_rows < 0:
        n_rows = 0
    for i in range(n_rows):
        while wst.pos <= i:
            step_window(ctx, b, cfg, wst)
        var hdev = ctx.enqueue_create_buffer[f32](H)
        final_norm_hidden(ctx, b, hdev)
        ctx.enqueue_copy(dst_buf=hbuf, src_buf=hdev)
        ctx.synchronize()
        write_f32_vec(out, hbuf, H)
    while wst.pos < n:
        step_window(ctx, b, cfg, wst)


def run_parity(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, draft_q4: Bool, q4_off_draft: Int,
    tokens: List[Int], tmax: Int, mut out: FileHandle,
) raises:
    """For every interior position P (1 <= P <= n-2), calls blk32_forward
    (m=1, pos=P, tok_pos=P, do_head=True) with hsrc = h_{P-1} -- the exact
    call shape both real call sites in serve/window.mojo use (pos==tok_pos
    always; the output is copied into Toks[tok_pos+1], confirmed from the
    self-chained draft loop's own `tokcp_k(..., st.pos+j+1, ...)` right
    after a `blk32_forward(..., st.pos+j, st.pos+j, ...)` call) -- and
    records the real engine's own drafted argmax dtok_d[0] alongside
    h_{P-1} and the true next token. Format: [n_pairs u32] then per pair
    [pos u32][tok_input u32][true_next u32][draft_argmax i32][H f32
    h_{pos-1}]. tok_input = Toks[pos] (the token blk32 is given as its
    embedding input); true_next = Toks[pos+1] (what the draft's argmax
    predicts, mirroring the trunk's own h_pos -> lm_head -> Toks[pos+1])."""
    var n = len(tokens)
    var pf = reset_and_load(ctx, b, tokens, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], n, n + 1)
    wst.reset(0)
    var hbuf = ctx.enqueue_create_host_buffer[f32](H)
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var p3_scratch = List[Int]()
    var n_pairs = n - 2
    if n_pairs < 0:
        n_pairs = 0
    write_u32(out, n_pairs)
    for i in range(n_pairs):
        var pos = i + 1
        while wst.pos <= pos - 1:
            step_window(ctx, b, cfg, wst)
        var hdev = ctx.enqueue_create_buffer[f32](H)
        final_norm_hidden(ctx, b, hdev)
        ctx.enqueue_copy(dst_buf=hbuf, src_buf=hdev)
        ctx.synchronize()
        blk32_forward(
            ctx, b.wbuf, b.off, e, 1, pos, pos, True, hdev,
            b.x_d, b.curb_d, b.qf_d, b.q_d, b.k_d, b.v_d, b.gate_d, b.ao_d,
            b.resb_d, b.fgb_d, b.p_qf_d, b.p_kv_d, b.p_h_d, b.p_ffn_d,
            b.p_ffn2_d, b.p_v_d, b.logits_d, b.cc_d, b.de_d, b.hd_d,
            b.kc32_d, b.vc32_d, b.kvtab_d, b.toks_d, b.dtok_d,
            False, p3_scratch, draft_q4, q4_off_draft, pack_q4,
        )
        ctx.enqueue_copy(dst_buf=dtok_h, src_buf=DeviceBuffer[DType.int32](ctx, b.dtok_d.unsafe_ptr(), 1, owning=False))
        ctx.synchronize()
        write_u32(out, pos)
        write_u32(out, tokens[pos])
        write_u32(out, tokens[pos + 1])
        write_u32(out, Int(dtok_h[0]))
        write_f32_vec(out, hbuf, H)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var args = argv()
    var mode = String("dump")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--mode" and i + 1 < len(args):
            mode = String(args[i + 1]); i += 2
        else:
            i += 1

    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var tmax = atol(getenv("BARO_E8_TMAX", "896"))
    var ctx = DeviceContext()
    var pack = load_pack(ctx, packdir)
    var buf = alloc_bufs(ctx, pack, tmax)
    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0)

    # bench/data/e8_tasks.json's own "tokens" field, same stored ids the
    # harness already scores against elsewhere (e13_engine_dump.mojo's own
    # --mode check convention): fine for a structural parity check, NOT a
    # held-out training corpus (bench/draft-head-protocol.md: the gate set
    # stays a gate). Real --mode dump data collection needs a genuinely
    # held-out text source, wired separately.
    var text_path = getenv("A4_TEXT_PATH", "bench/data/e8_tasks.json")
    var out_path = getenv("A4_OUT", ".work/a4/dump.bin")

    from grammar.json_value import parse_json_file
    from bench_latent_handoff import get_int_list
    var doc = parse_json_file(text_path)
    var root = doc.get(doc.root)
    var limit = atol(getenv("A4_LIMIT", "0"))
    var draft_q4 = pack.have_q4_draft and getenv("BARO_DRAFT_Q4", "0") == "1"
    var n_written = 0

    with open(out_path, "w") as out:
        for j in range(len(root.arr)):
            var tokens = get_int_list(doc, root.arr[j], "tokens")
            if len(tokens) < 4 or len(tokens) + 2 > tmax:
                continue
            if mode == "dump":
                dump_document_stepwise(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, tokens, tmax, out)
            else:
                run_parity(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, draft_q4, pack.q4_off, tokens, tmax, out)
            n_written += 1
            if limit > 0 and n_written >= limit:
                break
    print("draft_dump: wrote", n_written, "documents, mode=", mode, "->", out_path)
