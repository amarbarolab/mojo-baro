"""Prefix checkpoints for one sequence (bench/chat-protocol.md M1a).

A checkpoint is the SSM state after `pos` tokens (one conv ring slot + one
delta ring slot, copied device -> pinned host) keyed by an FNV-1a 64 hash of
the little-endian i32 token ids `tokens[0:pos]`. The KV pool is position
addressed and never copied: for a single sequence the entries for `[0, pos)`
are still in place unless a later replay overwrote them, which is what
`invalidate_above` tracks. In-process only; SHA-256 and cross-process
reproducibility are M1b.
"""
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from registry import CONV_SLOT, SSM_SLOT

comptime f32 = DType.float32
comptime CKPT_PERIOD = 1024
comptime CKPT_BYTES = (CONV_SLOT + SSM_SLOT) * 4


def prefix_hash(tokens: List[Int], n: Int) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        var v = UInt32(tokens[i])
        for b in range(4):
            h ^= UInt64((v >> UInt32(8 * b)) & UInt32(0xFF))
            h *= UInt64(0x100000001B3)
    return h


@fieldwise_init
struct Checkpoint(Copyable, Movable):
    var pos: Int
    var hash: UInt64
    var gen: Int
    var valid: Bool
    var pending: Bool
    var conv_h: HostBuffer[f32]
    var ssm_h: HostBuffer[f32]


struct Chain(Movable):
    var cap: Int
    var gen: Int
    var items: List[Checkpoint]

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        self.cap = cap
        self.gen = 0
        self.items = List[Checkpoint]()
        for _ in range(cap):
            var c = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
            var s = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
            self.items.append(Checkpoint(pos=0, hash=0, gen=0, valid=False, pending=False, conv_h=c^, ssm_h=s^))
        ctx.synchronize()

    def lookup(self, tokens: List[Int], n: Int) -> Int:
        # Index of the valid checkpoint with the largest pos <= n - 1 whose
        # hash matches tokens[0:pos]; -1 when none. n = len(tokens) normally.
        var best = -1
        for i in range(len(self.items)):
            if not self.items[i].valid or self.items[i].pos < 1 or self.items[i].pos > n - 1:
                continue
            if best >= 0 and self.items[i].pos <= self.items[best].pos:
                continue
            if prefix_hash(tokens, self.items[i].pos) == self.items[i].hash:
                best = i
        return best

    def pos_of(self, idx: Int) -> Int:
        return self.items[idx].pos if idx >= 0 else 0

    def invalidate_above(mut self, pos: Int):
        # A replay from `pos` overwrites KV [pos, ..): every later checkpoint
        # loses its KV prefix. A miss (pos 0) drops everything.
        for i in range(len(self.items)):
            if self.items[i].pos > pos:
                self.items[i].valid = False
                self.items[i].pending = False

    def save(
        mut self, ctx: DeviceContext, convstate_d: DeviceBuffer[f32], sstate_d: DeviceBuffer[f32],
        slot: Int, pos: Int, tokens: List[Int],
    ) raises:
        # Device -> pinned host copy of ring slot `slot`, stream-ordered: valid
        # once the caller has synchronised (commit).
        if self.cap == 0:
            return
        var idx = -1
        for i in range(len(self.items)):
            if self.items[i].valid and self.items[i].pos == pos:
                idx = i
                break
        if idx < 0:
            for i in range(len(self.items)):
                if not self.items[i].valid and not self.items[i].pending:
                    idx = i
                    break
        if idx < 0:
            idx = 0
            for i in range(1, len(self.items)):
                if self.items[i].pos < self.items[idx].pos:
                    idx = i
        self.gen += 1
        self.items[idx].pos = pos
        self.items[idx].hash = prefix_hash(tokens, pos)
        self.items[idx].gen = self.gen
        self.items[idx].valid = False
        self.items[idx].pending = True
        var cs = DeviceBuffer[f32](ctx, convstate_d.unsafe_ptr() + slot * CONV_SLOT, CONV_SLOT, owning=False)
        var ss = DeviceBuffer[f32](ctx, sstate_d.unsafe_ptr() + slot * SSM_SLOT, SSM_SLOT, owning=False)
        ctx.enqueue_copy(dst_buf=self.items[idx].conv_h, src_buf=cs)
        ctx.enqueue_copy(dst_buf=self.items[idx].ssm_h, src_buf=ss)

    def commit(mut self):
        for i in range(len(self.items)):
            if self.items[i].pending:
                self.items[i].pending = False
                self.items[i].valid = True

    def restore(
        self, ctx: DeviceContext, convstate_d: DeviceBuffer[f32], sstate_d: DeviceBuffer[f32], slot: Int, idx: Int
    ) raises:
        var cs = DeviceBuffer[f32](ctx, convstate_d.unsafe_ptr() + slot * CONV_SLOT, CONV_SLOT, owning=False)
        var ss = DeviceBuffer[f32](ctx, sstate_d.unsafe_ptr() + slot * SSM_SLOT, SSM_SLOT, owning=False)
        ctx.enqueue_copy(dst_buf=cs, src_buf=self.items[idx].conv_h)
        ctx.enqueue_copy(dst_buf=ss, src_buf=self.items[idx].ssm_h)

    def count_valid(self) -> Int:
        var n = 0
        for i in range(len(self.items)):
            if self.items[i].valid:
                n += 1
        return n
