# A3(c) kernel request

The A3(c) host contract is four `RowDesc` entries in one batch. Each entry
needs `req_id`, `pos`, `ring`, `kvtab_base`, `kv_base`, `conv_base`, and
`ssm_base`; host buffers already use those bases as four contiguous resident
slots. `SampleParams` stays per row on the host side and must reach the row
sampler without collapsing to row 0.

The current `m>1` kernels are same-sequence windows: they derive position as
`pos + row`, take one KV table pointer, and take one recurrent ring. They
cannot address four distinct request descriptors. The request below preserves
the current tensor layouts and changes only the scalar addressing ABI. `rows`
is a device pointer to four POD descriptors, and `row` is the existing
`block_idx.y` row. The exact new entry points are:

```text
amar_kv_append_rows[CLayout, NLayout, NAT](
    Kc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Kn: TileTensor[f32, NLayout, MutAnyOrigin],
    Vn: TileTensor[f32, NLayout, MutAnyOrigin],
    rows: DevicePointer[RowDesc], att_i: Int32,
)
amar_attn_decode_rows[QLayout, KLayout, OLayout, NAT](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    rows: DevicePointer[RowDesc], scale: Float32, att_i: Int32,
)
amar_ssm_conv_rows[QLayout, SLayout, WLayout, OLayout](
    Qkv: TileTensor[f32, QLayout, MutAnyOrigin],
    ConvState: TileTensor[f32, SLayout, MutAnyOrigin],
    ConvW: TileTensor[f32, WLayout, MutAnyOrigin],
    Out: TileTensor[f32, OLayout, MutAnyOrigin],
    rows: DevicePointer[RowDesc], ssm_i: Int32,
)
amar_ssm_delta_rows[MR, S0Layout, CLayout, GLayout, OLayout](
    SAll: TileTensor[f32, S0Layout, MutAnyOrigin],
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin],
    Eg: TileTensor[f32, GLayout, MutAnyOrigin],
    Beta: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    rows: DevicePointer[RowDesc],
)
amar_ssm_gated_out_rows_bf16[OLayout, ZLayout, NLayout, RLayout](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Z: TileTensor[f32, ZLayout, MutAnyOrigin],
    NormW: TileTensor[f32, NLayout, MutAnyOrigin],
    Res: TileTensor[DType.bfloat16, RLayout, MutAnyOrigin],
    rows: DevicePointer[RowDesc],
)
```

`DevicePointer[RowDesc]` is schematic Mojo spelling for the device-visible
POD pointer type already used by the kernel launch wrapper; it must be made a
concrete `MutPointer`/`UnsafePointer` at the call site. The descriptor layout
must be fixed as seven 32-bit fields in this order:
`req_id, pos, ring, kvtab_base, kv_base, conv_base, ssm_base`.

The attention append/decode pair uses `rows[row].pos` and
`rows[row].kvtab_base`/`kv_base`; the SSM pair uses `rows[row].ring` and
`rows[row].conv_base`/`ssm_base`. The output gate uses the existing row-major
residual/output stride and needs no additional descriptor field. This is the
only permitted ABI clarification before implementation.

Each row kernel must use its descriptor for position, page-table base, KV
base, ring and recurrent-state base. The sampler launch also needs row-wise
temperature, top-k, top-p, min-p and seed, or the T=0 gate must explicitly
scope itself to greedy argmax while the sampler ABI remains a follow-up.

This lane implements the row descriptor wire and serial per-row fallback first
so the four-request identity and accounting harness are live. It does not
claim one-launch batching until the row kernels land and the receipt prints a
single target launch with four ids.
