"""baro REALIGN: training-free expected embedding e = softmax(logits) @ W_emb
for the current row (see serve/window.mojo WindowBufs.logits_d/hn_d and
serve/registry.mojo embed_k/VOCAB/H for the buffers and table this reads).
"""
from std.math import ceildiv

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from registry import *
from window import WindowBufs, tens_bf16, row_f32
from elementwise import amar_softmax_rows
from realign_kernels import (
    amar_realign_copy_row, amar_realign_gather, amar_realign_reduce, REALIGN_SPLIT,
)

comptime probs_layout = row_major[1, VOCAB]()
comptime part_layout = row_major[REALIGN_SPLIT, H]()

comptime realign_copy_k = amar_realign_copy_row[type_of(probs_layout), type_of(probs_layout)]
comptime realign_softmax_k = amar_softmax_rows[type_of(probs_layout)]
comptime realign_gather_k = amar_realign_gather[type_of(emb_layout), type_of(probs_layout), type_of(part_layout)]
comptime realign_reduce_k = amar_realign_reduce[type_of(part_layout), type_of(h_layout)]


def realign_expected_embedding(
    ctx: DeviceContext,
    mut b: WindowBufs,
    mut e_dev: DeviceBuffer[f32],
) raises:
    var probs_d = ctx.enqueue_create_buffer[f32](VOCAB)
    var part_d = ctx.enqueue_create_buffer[f32](REALIGN_SPLIT * H)

    ctx.enqueue_function[realign_copy_k](
        row_f32(ctx, b.logits_d, 0, VOCAB, probs_layout), TileTensor(probs_d, probs_layout),
        Int32(VOCAB), grid_dim=ceildiv(VOCAB, 256), block_dim=256,
    )
    ctx.enqueue_function[realign_softmax_k](
        TileTensor(probs_d, probs_layout), Int32(VOCAB), grid_dim=1, block_dim=256,
    )

    var Embd = tens_bf16(ctx, b.wbuf, b.off[0], VOCAB * H, emb_layout)
    ctx.enqueue_function[realign_gather_k](
        Embd, TileTensor(probs_d, probs_layout), TileTensor(part_d, part_layout), Int32(VOCAB), Int32(H),
        grid_dim=(ceildiv(H, 256), REALIGN_SPLIT), block_dim=256,
    )

    ctx.enqueue_function[realign_reduce_k](
        TileTensor(part_d, part_layout), TileTensor(e_dev, h_layout), Int32(H),
        grid_dim=ceildiv(H, 256), block_dim=256,
    )
