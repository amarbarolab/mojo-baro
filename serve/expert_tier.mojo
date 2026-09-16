"""B4 stage 2b: routed experts in host RAM, an LRU of resident experts in VRAM.

Preregistered in `bench/moe-tier-protocol.md`. The trunk stays in VRAM; the
120 routed expert tensors (40 layers x gate/up/down) live in a split pack's
`experts.bin` (`tools/pack-split-experts.py`) and are fetched into a VRAM
cache on demand.

Why this needs no kernel change, which is the whole design:

  `moe_gate_up_q4k_pack` reads expert `e` at `e * FFN * row_bytes` from a base
  pointer it is handed, and its gate-to-up distance is a RUNTIME argument
  (`up_offset`). `amar_moe_down_q4k` reads `e * N * row_bytes` from its own
  base. Neither knows how many experts exist. So a cache holding `cap`
  experts in the same inner layout, a base pointer into it, an `up_offset` of
  `cap * per_expert_bytes`, and an index array holding SLOTS instead of
  expert ids runs both kernels unchanged.

Cache layout per layer, one contiguous block:

  [ gate: cap x eb ][ up: cap x eb ][ down: cap x ebd ]

`eb` is 589,824 bytes for this pack's gate and up; `ebd` is 589,824 (q4_k) or
860,160 (q6_k: layers 34, 38, 39). At cap 64 that is 113 MB per layer and
4.53 GB over 40 layers, against 18.1 GB for all 256.

The host store is read through the page cache by default rather than held in
18 GB of pinned memory: stage 1 measured pinned and pageable host-to-device
at 28.78 against 28.60 GB/s on this box, a 0.6% difference, and this machine
runs other work. `BARO_TIER_PINNED=1` holds the whole store pinned instead,
and the tier prints which one it is using, because that is an arm-defining
parameter (P1).

A miss costs a host round trip: the top-8 is only known after the router
runs, so the ids come back to the host, the LRU decides, and the fetches are
enqueued before the gate/up launch. That is inherent to demand fetching and is
what stage 3's prefetch exists to remove.
"""
from std.collections import Dict, List
from std.ffi import c_ssize_t, external_call
from std.memory import UnsafePointer
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from moe_pack import (
    TensorInfo, parse_moe_index, tensor_byte_size, resolve_plain, N_EXP, TOPK,
)

# One staging slot per piece a single layer can fetch: 8 experts x gate, up,
# down. With a slot per piece nothing in flight is overwritten, so the copies
# need no synchronize between them; the next layer's prepare() already syncs to
# read the router's ids back, which is what makes the previous layer's fetches
# safe to overwrite. Syncing per piece instead cost 720 syncs per token and
# 13.6 tok/s on the first live run.
comptime STAGE_SLOTS = 24
comptime SLOT_BYTES = 1 << 20  # 1 MiB, above this pack's 860,160-byte q6_k expert
comptime STAGE_BYTES = STAGE_SLOTS * SLOT_BYTES


struct LayerGeom(Copyable, Movable):
    """Byte geometry of one layer's three routed expert tensors."""

    var gate_off: Int
    var up_off: Int
    var down_off: Int
    var eb: Int
    var ebd: Int

    def __init__(out self, tensors: Dict[String, TensorInfo], layer: Int) raises:
        var g = resolve_plain(tensors, "blk." + String(layer) + ".ffn_gate_exps.weight")
        var u = resolve_plain(tensors, "blk." + String(layer) + ".ffn_up_exps.weight")
        var d = resolve_plain(tensors, "blk." + String(layer) + ".ffn_down_exps.weight")
        self.gate_off = g.offset
        self.up_off = u.offset
        self.down_off = d.offset
        self.eb = tensor_byte_size(g.dtype, g.n_elem) // N_EXP
        self.ebd = tensor_byte_size(d.dtype, d.n_elem) // N_EXP
        var ub = tensor_byte_size(u.dtype, u.n_elem) // N_EXP
        if ub != self.eb:
            raise Error(
                "layer " + String(layer) + ": up expert bytes " + String(ub)
                + " differ from gate's " + String(self.eb)
                + ", so one up_offset cannot address both"
            )


struct LayerLru(Copyable, Movable):
    """Expert id to cache slot, least-recently-used eviction.

    Deliberately the same policy the offline replay used
    (`bench/moe-locality.py`): capacity per layer, reset per request, ties
    broken by recency of use. Gate 1 replays this code over the same trace and
    must reproduce the replay's hit rate, because a tier whose residency
    decisions differ from the measurement's is not the thing that was
    measured.
    """

    var cap: Int
    var slot_of: Dict[Int, Int]
    var expert_at: List[Int]
    var used_at: List[Int]
    var clock: Int

    def __init__(out self, cap: Int):
        self.cap = cap
        self.slot_of = Dict[Int, Int]()
        self.expert_at = List[Int](unsafe_uninit_length=cap)
        self.used_at = List[Int](unsafe_uninit_length=cap)
        for i in range(cap):
            self.expert_at[i] = -1
            self.used_at[i] = -1
        self.clock = 0

    def reset(mut self):
        self.slot_of = Dict[Int, Int]()
        for i in range(self.cap):
            self.expert_at[i] = -1
            self.used_at[i] = -1
        self.clock = 0

    def touch(mut self, expert: Int) raises -> Tuple[Int, Bool]:
        """-> (slot, hit). On a miss the caller must fill the slot."""
        self.clock += 1
        if expert in self.slot_of:
            var s = self.slot_of[expert]
            self.used_at[s] = self.clock
            return (s, True)
        var victim = 0
        var oldest = self.used_at[0]
        for i in range(self.cap):
            if self.expert_at[i] < 0:
                victim = i
                oldest = -1
                break
            if self.used_at[i] < oldest:
                oldest = self.used_at[i]
                victim = i
        var evicted = self.expert_at[victim]
        if evicted >= 0:
            _ = self.slot_of.pop(evicted)
        self.expert_at[victim] = expert
        self.used_at[victim] = self.clock
        self.slot_of[expert] = victim
        return (victim, False)


struct ExpertTier(Copyable, Movable):
    var active: Bool
    var cap: Int
    var pinned: Bool
    var n_layers: Int
    var geom: List[LayerGeom]
    var lru: List[LayerLru]
    var cache: DeviceBuffer[DType.uint8]
    var layer_bytes: Int
    var stage: HostBuffer[DType.uint8]
    var store: HostBuffer[DType.uint8]
    var store_path: String
    var slots_h: HostBuffer[DType.int32]
    var slot_next: Int
    var refs: Int
    var hits: Int
    var bytes_fetched: Int
    var fetch_ns: Int
    var b_open: List[Int]
    var b_readback: List[Int]
    var b_lru: List[Int]
    var b_pread: List[Int]
    var b_copy: List[Int]
    var b_writeback: List[Int]

    def __init__(
        out self, ctx: DeviceContext, packdir: String, cap: Int, n_layers: Int
    ) raises:
        """Sizes everything from the pack's own index, never from constants."""
        var tensors = parse_moe_index(packdir + "/index.txt")
        self.active = True
        self.cap = cap
        self.n_layers = n_layers
        self.pinned = getenv("BARO_TIER_PINNED", "0") == "1"
        self.geom = List[LayerGeom]()
        self.lru = List[LayerLru]()
        var per_layer = 0
        var store_bytes = 0
        for l in range(n_layers):
            var g = LayerGeom(tensors, l)
            var lb = cap * (2 * g.eb + g.ebd)
            if lb > per_layer:
                per_layer = lb
            store_bytes += N_EXP * (2 * g.eb + g.ebd)
            self.geom.append(g^)
            self.lru.append(LayerLru(cap))
        self.layer_bytes = per_layer
        self.cache = ctx.enqueue_create_buffer[DType.uint8](per_layer * n_layers)
        self.stage = ctx.enqueue_create_host_buffer[DType.uint8](STAGE_BYTES)
        self.slots_h = ctx.enqueue_create_host_buffer[DType.int32](TOPK)
        # The path, not a held descriptor: WindowBufs must stay copyable and a
        # FileHandle is not, so prepare() opens the store once per call. That
        # is one open per layer per token, microseconds against a 1.8 MB
        # transfer, and only on the miss path.
        self.store_path = packdir + "/experts.bin"
        self.store = ctx.enqueue_create_host_buffer[DType.uint8](
            store_bytes if self.pinned else 1
        )
        self.slot_next = 0
        self.refs = 0
        self.hits = 0
        self.bytes_fetched = 0
        self.fetch_ns = 0
        self.b_open = List[Int]()
        self.b_readback = List[Int]()
        self.b_lru = List[Int]()
        self.b_pread = List[Int]()
        self.b_copy = List[Int]()
        self.b_writeback = List[Int]()
        for _ in range(n_layers):
            self.b_open.append(0)
            self.b_readback.append(0)
            self.b_lru.append(0)
            self.b_pread.append(0)
            self.b_copy.append(0)
            self.b_writeback.append(0)
        ctx.synchronize()
        if self.pinned:
            with open(self.store_path, "r") as f:
                var fd = f._get_raw_fd()
                var got = 0
                while got < store_bytes:
                    var n = external_call["pread", c_ssize_t](
                        fd, self.store.unsafe_ptr().unsafe_offset(got),
                        store_bytes - got, Int64(got),
                    )
                    if n <= 0:
                        raise Error("expert tier: short read of experts.bin")
                    got += Int(n)
        print(
            "expert tier: cap", cap, " cache", Float64(per_layer * n_layers) / 1e9,
            "GB  host store", Float64(store_bytes) / 1e9,
            "GB  mode", "pinned" if self.pinned else "page-cache",
        )

    def __init__(out self, ctx: DeviceContext) raises:
        """BARO_TIER unset: everything stays in the pack and moe_ffn takes its
        old path. One-byte buffers rather than zero-length ones, because a
        zero-length device buffer is not worth the special case."""
        self.active = False
        self.cap = 0
        self.pinned = False
        self.n_layers = 0
        self.geom = List[LayerGeom]()
        self.lru = List[LayerLru]()
        self.layer_bytes = 0
        self.cache = ctx.enqueue_create_buffer[DType.uint8](1)
        self.stage = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self.store = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self.slots_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.store_path = ""
        self.slot_next = 0
        self.refs = 0
        self.hits = 0
        self.bytes_fetched = 0
        self.fetch_ns = 0
        self.b_open = List[Int]()
        self.b_readback = List[Int]()
        self.b_lru = List[Int]()
        self.b_pread = List[Int]()
        self.b_copy = List[Int]()
        self.b_writeback = List[Int]()

    def layer_base(self, layer: Int) -> Int:
        return layer * self.layer_bytes

    def gate_base(self, layer: Int) -> Int:
        return self.layer_base(layer)

    def up_offset(self, layer: Int) -> Int:
        """What the gate/up kernel needs as its runtime gate-to-up distance."""
        return self.cap * self.geom[layer].eb

    def down_base(self, layer: Int) -> Int:
        return self.layer_base(layer) + 2 * self.cap * self.geom[layer].eb

    def reset(mut self):
        """Per request: a new conversation starts with a cold tier, which is
        also what the offline replay assumed when it reset per prompt."""
        for i in range(self.n_layers):
            self.lru[i].reset()

    def _fetch_piece(
        mut self, ctx: DeviceContext, fd: Int, host_off: Int, dev_off: Int, nbytes: Int, layer: Int
    ) raises:
        """One expert projection, host to device, through the staging buffer
        unless the whole store is pinned (then the copy is direct)."""
        if self.pinned:
            var tc0 = perf_counter_ns()
            ctx.enqueue_copy(
                dst_buf=DeviceBuffer[DType.uint8](
                    ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
                ),
                src_buf=self.store.create_sub_buffer[DType.uint8](host_off, nbytes),
            )
            self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
            self.bytes_fetched += nbytes
            return
        if nbytes > SLOT_BYTES:
            raise Error("expert tier: staging slot too small for " + String(nbytes))
        var slot = self.slot_next % STAGE_SLOTS
        self.slot_next += 1
        var base = slot * SLOT_BYTES
        var got = 0
        var tp0 = perf_counter_ns()
        while got < nbytes:
            var n = external_call["pread", c_ssize_t](
                fd, self.stage.unsafe_ptr().unsafe_offset(base + got),
                nbytes - got, Int64(host_off + got),
            )
            if n <= 0:
                raise Error("expert tier: short read at " + String(host_off))
            got += Int(n)
        self.b_pread[layer] = self.b_pread[layer] + Int(perf_counter_ns() - tp0)
        var tc0 = perf_counter_ns()
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.uint8](
                ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
            ),
            src_buf=self.stage.create_sub_buffer[DType.uint8](base, nbytes),
        )
        self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
        self.bytes_fetched += nbytes

    def prepare(
        mut self, ctx: DeviceContext, layer: Int, mut idx_d: DeviceBuffer[DType.int32]
    ) raises:
        """Read the router's top-8 back, fill misses, write SLOTS in place.

        The device index buffer holds expert ids on entry and cache slots on
        exit, so the two expert kernels see slots and need no change. The read
        back is a real host round trip: the top-8 does not exist until the
        router has run.
        """
        var t0 = perf_counter_ns()
        var to0 = perf_counter_ns()
        var fh = open(self.store_path, "r")
        var fd = fh._get_raw_fd()
        self.slot_next = 0
        self.b_open[layer] = self.b_open[layer] + Int(perf_counter_ns() - to0)
        var trb0 = perf_counter_ns()
        ctx.enqueue_copy(
            dst_buf=self.slots_h,
            src_buf=DeviceBuffer[DType.int32](ctx, idx_d.unsafe_ptr(), TOPK, owning=False),
        )
        ctx.synchronize()
        self.b_readback[layer] = self.b_readback[layer] + Int(perf_counter_ns() - trb0)
        # Copied out, not held as a reference: _fetch_piece takes `mut self`
        # (it counts bytes), which invalidates an interior reference into
        # self.geom while the fetch loop still needs the geometry.
        var eb = self.geom[layer].eb
        var ebd = self.geom[layer].ebd
        var lbase = self.layer_base(layer)
        var gate_store = self.geom[layer].gate_off
        var up_store = self.geom[layer].up_off
        var down_store = self.geom[layer].down_off
        for j in range(TOPK):
            var e = Int(self.slots_h[j])
            if e < 0 or e >= N_EXP:
                raise Error("expert tier: router returned expert id " + String(e))
            var tl0 = perf_counter_ns()
            var r = self.lru[layer].touch(e)
            self.b_lru[layer] = self.b_lru[layer] + Int(perf_counter_ns() - tl0)
            var slot = r[0]
            self.refs += 1
            if r[1]:
                self.hits += 1
            else:
                self._fetch_piece(ctx, fd, gate_store + e * eb, lbase + slot * eb, eb, layer)
                self._fetch_piece(
                    ctx, fd, up_store + e * eb, lbase + self.cap * eb + slot * eb, eb, layer
                )
                self._fetch_piece(
                    ctx, fd, down_store + e * ebd,
                    lbase + 2 * self.cap * eb + slot * ebd, ebd, layer,
                )
            self.slots_h[j] = Int32(slot)
        var tw0 = perf_counter_ns()
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.int32](ctx, idx_d.unsafe_ptr(), TOPK, owning=False),
            src_buf=self.slots_h,
        )
        self.b_writeback[layer] = self.b_writeback[layer] + Int(perf_counter_ns() - tw0)
        var tc1 = perf_counter_ns()
        fh.close()
        self.b_open[layer] = self.b_open[layer] + Int(perf_counter_ns() - tc1)
        self.fetch_ns += Int(perf_counter_ns() - t0)

    def report(self, tokens: Int):
        var rate = Float64(self.hits) / Float64(self.refs) if self.refs > 0 else 0.0
        print(
            "expert tier: refs", self.refs, " hits", self.hits,
            " hit_rate", rate,
            " bytes_fetched", self.bytes_fetched,
            " bytes_per_token", Float64(self.bytes_fetched) / Float64(tokens) if tokens > 0 else 0.0,
            " fetch_s", Float64(self.fetch_ns) / 1e9,
        )
        if getenv("BARO_TIER_STAMP", "0") == "1":
            print(
                "tier stamp: layer  open_us  readback_us  lru_us  pread_us  copy_us",
                " writeback_us  row_us",
            )
            var sum_ns = 0
            for l in range(self.n_layers):
                var row_ns = (
                    self.b_open[l] + self.b_readback[l] + self.b_lru[l]
                    + self.b_pread[l] + self.b_copy[l] + self.b_writeback[l]
                )
                sum_ns += row_ns
                print(
                    "tier stamp:", l, Float64(self.b_open[l]) / 1e3,
                    Float64(self.b_readback[l]) / 1e3, Float64(self.b_lru[l]) / 1e3,
                    Float64(self.b_pread[l]) / 1e3, Float64(self.b_copy[l]) / 1e3,
                    Float64(self.b_writeback[l]) / 1e3, Float64(row_ns) / 1e3,
                )
            print(
                "tier stamp: sum_s", Float64(sum_ns) / 1e9,
                " fetch_s", Float64(self.fetch_ns) / 1e9,
                " sum_per_token_us",
                Float64(sum_ns) / Float64(tokens) / 1e3 if tokens > 0 else 0.0,
            )
