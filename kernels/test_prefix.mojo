"""Byte-exact prefix checkpoint restore (bench/chat-protocol.md M1a, P-F1).

Drives the real engine path (serve/harness.mojo buffers, serve/window.mojo
step_window, serve/prefix.mojo Chain) on the q4 pack over P = A||B with
A = p8192.tokens[0:1088], B = the next 64 tokens, n = 1, and compares the
conv slot, the delta slot, every KV entry in [0, n_total), the megakernel
head partials (hmax/hidx) or the full logits (window path), and the generated
token as raw bits:

  cold_u   one process, prefill chunks as the engine cuts them (1024, 127)
  cold_s   one process, no checkpoint, chunk boundary forced at 1087
           (separates chunk arithmetic from checkpoint bookkeeping)
  restore  request A (checkpoints at 1024 and 1087), then P restored at 1087
  repeat   P again, restored at its own prompt-end checkpoint (1151)
  ckpt1024 P' = P with token 1050 mutated, restored at 1024 (chunk-identical
           to cold_u(P')) vs cold_u(P')

Lookup checks (chain = {1024, 1087, 1151} after `restore`): mutations at
token 0 and 1023 miss everything, 1025 and 1086 fall back to 1024, 1087
(outside tokens[0:1087]) keeps 1087, and a corrupted hash on the 1087
checkpoint falls back to 1024. Both the megakernel and the window path run.

Build: ./.venv/bin/mojo build kernels/test_prefix.mojo -I kernels -I serve -o .work/test_prefix
Env: BARO_PACK (default .work/engine-pack-q4), BARO_PREFIX_TOKENS (default
bench/prefill-prompts/p8192.tokens).
"""
from std.os import getenv
from std.sys import exit, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from registry import *
from window import *
from harness import *
from prefix import *
from attn import kv_off

comptime A_LEN = 1088
comptime B_LEN = 64
comptime T_MAX = 2048


def read_tokens(path: String, n: Int) raises -> List[Int]:
    var out = List[Int]()
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
                    out.append(val)
                    if len(out) == n:
                        return out^
                val = 0
                have = False
        if have:
            out.append(val)
    if len(out) < n:
        raise Error("prompt file holds " + String(len(out)) + " tokens, need " + String(n))
    return out^


struct Snap(Movable):
    var conv_h: HostBuffer[f32]
    var ssm_h: HostBuffer[f32]
    var kc_h: HostBuffer[KVT]
    var vc_h: HostBuffer[KVT]
    var hmax_h: HostBuffer[f32]
    var hidx_h: HostBuffer[DType.int32]
    var logits_h: HostBuffer[f32]
    var tok: Int
    var n_total: Int

    def __init__(out self, ctx: DeviceContext, bufs: WindowBufs, ring: Int, n_total: Int) raises:
        self.conv_h = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
        self.ssm_h = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
        self.kc_h = ctx.enqueue_create_host_buffer[KVT](bufs.kvpool)
        self.vc_h = ctx.enqueue_create_host_buffer[KVT](bufs.kvpool)
        self.hmax_h = ctx.enqueue_create_host_buffer[f32](MEGA_MR * MEGA_G_WIN)
        self.hidx_h = ctx.enqueue_create_host_buffer[DType.int32](MEGA_MR * MEGA_G_WIN)
        self.logits_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
        var tok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.synchronize()
        ctx.enqueue_copy(dst_buf=self.conv_h, src_buf=DeviceBuffer[f32](ctx, bufs.convstate_d.unsafe_ptr() + ring * CONV_SLOT, CONV_SLOT, owning=False))
        ctx.enqueue_copy(dst_buf=self.ssm_h, src_buf=DeviceBuffer[f32](ctx, bufs.sstate_d.unsafe_ptr() + ring * SSM_SLOT, SSM_SLOT, owning=False))
        ctx.enqueue_copy(dst_buf=self.kc_h, src_buf=bufs.kc_d)
        ctx.enqueue_copy(dst_buf=self.vc_h, src_buf=bufs.vc_d)
        ctx.enqueue_copy(dst_buf=self.hmax_h, src_buf=bufs.hmax_d)
        ctx.enqueue_copy(dst_buf=self.hidx_h, src_buf=bufs.hidx_d)
        ctx.enqueue_copy(dst_buf=self.logits_h, src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False))
        ctx.enqueue_copy(dst_buf=tok_h, src_buf=DeviceBuffer[DType.int32](ctx, bufs.toks_d.unsafe_ptr() + n_total - 1, 1, owning=False))
        ctx.synchronize()
        self.tok = Int(tok_h[0])
        self.n_total = n_total


def check_bits[dt: DType](label: String, a: HostBuffer[dt], b: HostBuffer[dt], n: Int, mut fails: Int):
    var ap = a.unsafe_ptr().unsafe_bitcast[UInt32]()
    var bp = b.unsafe_ptr().unsafe_bitcast[UInt32]()
    var first = -1
    var bad = 0
    for i in range(n):
        if ap[i] != bp[i]:
            if first < 0:
                first = i
            bad += 1
    if bad == 0:
        print("  PASS", label, "(" + String(n) + " words)")
    else:
        print("  FAIL", label, ": " + String(bad) + " / " + String(n) + " words differ, first at", first)
        fails += 1


def kv_row_equal(a: HostBuffer[KVT], b: HostBuffer[KVT], t: Int) -> Bool:
    var ap = a.unsafe_ptr().unsafe_bitcast[UInt32]()
    var bp = b.unsafe_ptr().unsafe_bitcast[UInt32]()
    for att_i in range(N_ATT):
        for kvh in range(NKVH):
            var o = kv_off[N_ATT](t, att_i, kvh)
            for d in range(HD):
                if ap[o + d] != bp[o + d]:
                    return False
    return True


def compare(label: String, x: Snap, y: Snap, mega: Bool, mut fails: Int):
    print("==", label, "(" + ("megakernel" if mega else "window path") + ")")
    check_bits("conv slot", x.conv_h, y.conv_h, CONV_SLOT, fails)
    check_bits("delta slot", x.ssm_h, y.ssm_h, SSM_SLOT, fails)
    var n = x.n_total
    var bad_k = 0
    var bad_v = 0
    var first_k = -1
    var first_v = -1
    for t in range(n):
        if not kv_row_equal(x.kc_h, y.kc_h, t):
            if first_k < 0:
                first_k = t
            bad_k += 1
        if not kv_row_equal(x.vc_h, y.vc_h, t):
            if first_v < 0:
                first_v = t
            bad_v += 1
    if bad_k == 0 and bad_v == 0:
        print("  PASS KV entries [0," + String(n) + ") all layers/heads")
    else:
        print("  FAIL KV entries: K rows differing", bad_k, "(first", first_k, ") V rows differing", bad_v, "(first", first_v, ")")
        fails += 1
    var pos_list: List[Int] = [0, 1, 1023, 1024, 1025, 1087, 1088, 1151]
    var line = String("  KV at positions:")
    for i in range(len(pos_list)):
        var t = pos_list[i]
        if t < n:
            line += " " + String(t) + ("=ok" if (kv_row_equal(x.kc_h, y.kc_h, t) and kv_row_equal(x.vc_h, y.vc_h, t)) else "=DIFF")
    print(line)
    if mega:
        check_bits("head partial max (hmax)", x.hmax_h, y.hmax_h, MEGA_MR * MEGA_G_WIN, fails)
        check_bits("head partial idx (hidx)", x.hidx_h, y.hidx_h, MEGA_MR * MEGA_G_WIN, fails)
    else:
        check_bits("next-token logits", x.logits_h, y.logits_h, VOCAB, fails)
    if x.tok == y.tok:
        print("  PASS next token", x.tok)
    else:
        print("  FAIL next token", x.tok, "vs", y.tok)
        fails += 1


def make_cfg(pack: Pack, mega: Bool, pf_rows: Int, pf_tail: Int, n_total: Int, n_prompt: Int) -> WindowCfg:
    return WindowCfg(
        pack_q4=pack.pack_q4, draft_q4=False, q4_off=pack.q4_off, e=pack.e, kcfg=2, spec=False, spec_dbg=False,
        serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False, dump=False, mega=mega, att_split=TMAX,
        mega_win=False, dot3=False, pf_chunk=CP, pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, n_prompt=n_prompt,
    )


def load_prompt(ctx: DeviceContext, mut bufs: WindowBufs, prompt: List[Int]) raises:
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](T_MAX)
    ctx.synchronize()
    for i in range(T_MAX):
        toks_h[i] = 0
    for i in range(len(prompt)):
        toks_h[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=bufs.toks_d, src_buf=toks_h)
    ctx.synchronize()


def cold_reset(ctx: DeviceContext, mut bufs: WindowBufs) raises:
    ctx.enqueue_memset(bufs.convstate_d, 0)
    ctx.enqueue_memset(bufs.sstate_d, 0)
    ctx.enqueue_memset(bufs.kc_d, 0)
    ctx.enqueue_memset(bufs.vc_d, 0)
    ctx.enqueue_memset(bufs.kc32_d, 0)
    ctx.enqueue_memset(bufs.vc32_d, 0)
    ctx.enqueue_memset(bufs.ctr_d, 0)
    ctx.enqueue_memset(bufs.prof_d, 0)
    ctx.synchronize()


def prefill_to(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, mut wst: WindowState, end: Int, n_total: Int, n_prompt: Int, prompt: List[Int], mut chain: Chain, take: Bool) raises:
    # prompt rows [wst.pos, end) through prefill_forward in chunks of CP, exactly
    # as the engine loop cuts them from wst.pos; checkpoints at the engine's
    # boundaries when `take`.
    if end - wst.pos < PF_MIN:
        raise Error("prefill_to: fewer than PF_MIN rows")
    var tail = (end - wst.pos) % MROWS
    if tail == 0:
        tail = MROWS
    var cfg = make_cfg(pack, mega, end, tail, n_total, n_prompt)
    while wst.pos < end:
        step_window(ctx, bufs, cfg, wst)
        if take and wst.pos < n_prompt and (wst.pos == n_prompt - 1 or wst.pos % CKPT_PERIOD == 0):
            chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt)


def finish(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, mut wst: WindowState, n_total: Int, n_prompt: Int) raises:
    var cfg = make_cfg(pack, mega, 0, 0, n_total, n_prompt)
    while wst.pos < n_total - 1:
        step_window(ctx, bufs, cfg, wst)
    ctx.synchronize()


def fresh_state() -> WindowState:
    return WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0])


def run_cold(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, prompt: List[Int], split_at: Int, mut chain: Chain) raises -> Snap:
    var n_prompt = len(prompt)
    var n_total = n_prompt + 1
    cold_reset(ctx, bufs)
    load_prompt(ctx, bufs, prompt)
    var wst = fresh_state()
    wst.reset(perf_counter_ns())
    if split_at > 0:
        prefill_to(ctx, bufs, pack, mega, wst, split_at, n_total, n_prompt, prompt, chain, False)
    prefill_to(ctx, bufs, pack, mega, wst, n_prompt - 1, n_total, n_prompt, prompt, chain, False)
    finish(ctx, bufs, pack, mega, wst, n_total, n_prompt)
    return Snap(ctx, bufs, wst.ring, n_total)


def run_request(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, prompt: List[Int], mut chain: Chain, mut cached_out: Int) raises -> Snap:
    # The engine serve path: lookup, restore into slot 0 or cold reset, replay,
    # checkpoints at the boundaries, commit after the final synchronize.
    var n_prompt = len(prompt)
    var n_total = n_prompt + 1
    var idx = chain.lookup(prompt, n_prompt)
    var cached = chain.pos_of(idx)
    chain.invalidate_above(cached)
    if idx >= 0:
        chain.restore(ctx, bufs.convstate_d, bufs.sstate_d, 0, idx)
        ctx.enqueue_memset(bufs.kc32_d, 0)
        ctx.enqueue_memset(bufs.vc32_d, 0)
        ctx.enqueue_memset(bufs.ctr_d, 0)
        ctx.enqueue_memset(bufs.prof_d, 0)
        ctx.synchronize()
    else:
        cold_reset(ctx, bufs)
    load_prompt(ctx, bufs, prompt)
    var wst = fresh_state()
    wst.reset(perf_counter_ns())
    wst.pos = cached
    wst.pos_prev = cached
    if n_prompt - 1 - cached >= PF_MIN:
        prefill_to(ctx, bufs, pack, mega, wst, n_prompt - 1, n_total, n_prompt, prompt, chain, True)
    var cfg = make_cfg(pack, mega, 0, 0, n_total, n_prompt)
    while wst.pos < n_total - 1:
        step_window(ctx, bufs, cfg, wst)
        if wst.pos > cached and wst.pos < n_prompt and (wst.pos == n_prompt - 1 or wst.pos % CKPT_PERIOD == 0):
            chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt)
    ctx.synchronize()
    chain.commit()
    cached_out = cached
    return Snap(ctx, bufs, wst.ring, n_total)


def expect_lookup(label: String, chain: Chain, tokens: List[Int], want: Int, mut fails: Int):
    var got = chain.pos_of(chain.lookup(tokens, len(tokens)))
    if got == want:
        print("  PASS lookup", label, "->", got)
    else:
        print("  FAIL lookup", label, "->", got, "want", want)
        fails += 1


def mutated(tokens: List[Int], at: Int) -> List[Int]:
    var out = tokens.copy()
    out[at] = out[at] + 1
    return out^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var tokpath = getenv("BARO_PREFIX_TOKENS", "bench/prefill-prompts/p8192.tokens")
    var pack = load_pack(ctx, packdir)
    var bufs = alloc_bufs(ctx, pack, T_MAX)
    var P = read_tokens(tokpath, A_LEN + B_LEN)
    var A = List[Int]()
    for i in range(A_LEN):
        A.append(P[i])
    var fails = 0
    print("test_prefix: pack", packdir, " q4", pack.pack_q4, " tokens", tokpath, " A", len(A), " B", B_LEN, " tmax", T_MAX)
    print("checkpoints: cap 8, bytes", Float64(CKPT_BYTES) / 1e6, "MB each, period", CKPT_PERIOD)

    for arm in range(2):
        var mega = arm == 0
        var chain = Chain(ctx, 8)
        var t0 = perf_counter_ns()
        var cold_u = run_cold(ctx, bufs, pack, mega, P, 0, chain)
        var cold_s = run_cold(ctx, bufs, pack, mega, P, A_LEN - 1, chain)
        compare("cold_s vs cold_u (chunk split at 1087, no checkpoint)", cold_s, cold_u, mega, fails)
        var c0 = 0
        var snapA = run_request(ctx, bufs, pack, mega, A, chain, c0)
        print("  request A: cached", c0, " checkpoints", chain.count_valid(), " tok", snapA.tok)
        var c1 = 0
        var rest = run_request(ctx, bufs, pack, mega, P, chain, c1)
        print("  request P: cached", c1, " checkpoints", chain.count_valid())
        if c1 != A_LEN - 1:
            print("  FAIL restore point", c1, "want", A_LEN - 1)
            fails += 1
        compare("restore(A)+replay(B) vs cold_u", rest, cold_u, mega, fails)
        compare("restore(A)+replay(B) vs cold_s", rest, cold_s, mega, fails)

        print("== lookup (chain 1024 / 1087 / 1151)")
        expect_lookup("exact P", chain, P, A_LEN + B_LEN - 1, fails)
        expect_lookup("token 0 mutated", chain, mutated(P, 0), 0, fails)
        expect_lookup("token 1023 mutated", chain, mutated(P, 1023), 0, fails)
        expect_lookup("token 1025 mutated", chain, mutated(P, 1025), 1024, fails)
        expect_lookup("token 1086 mutated (last hashed token of A)", chain, mutated(P, 1086), 1024, fails)
        expect_lookup("token 1087 mutated (last token of A, outside tokens[0:1087])", chain, mutated(P, 1087), A_LEN - 1, fails)
        expect_lookup("token 1100 mutated", chain, mutated(P, 1100), A_LEN - 1, fails)
        var ci = -1
        for i in range(len(chain.items)):
            if chain.items[i].valid and chain.items[i].pos == A_LEN - 1:
                ci = i
        var saved_hash = chain.items[ci].hash
        chain.items[ci].hash = saved_hash ^ UInt64(1)
        expect_lookup("corrupted hash on the 1087 checkpoint", chain, mutated(P, 1100), 1024, fails)
        chain.items[ci].hash = saved_hash

        var c2 = 0
        var rep = run_request(ctx, bufs, pack, mega, P, chain, c2)
        print("  request P again: cached", c2, " checkpoints", chain.count_valid())
        if c2 != A_LEN + B_LEN - 1:
            print("  FAIL repeat restore point", c2, "want", A_LEN + B_LEN - 1)
            fails += 1
        compare("repeat(P) restored at 1151 vs cold_u", rep, cold_u, mega, fails)

        var Pm = mutated(P, 1050)
        var cold_m = run_cold(ctx, bufs, pack, mega, Pm, 0, chain)
        var c3 = 0
        var rest_m = run_request(ctx, bufs, pack, mega, Pm, chain, c3)
        print("  request P' (token 1050 mutated): cached", c3, " checkpoints", chain.count_valid())
        if c3 != 1024:
            print("  FAIL P' restore point", c3, "want 1024")
            fails += 1
        compare("restore(P') at 1024 vs cold_u(P') (chunk-identical)", rest_m, cold_m, mega, fails)
        print("arm", "megakernel" if mega else "window", "done in", Float64(perf_counter_ns() - t0) / 1e9, "s")

    if fails == 0:
        print("PASS: prefix checkpoints byte-exact")
    else:
        print("FAIL: prefix checkpoints,", fails, "checks failed")
        exit(1)
