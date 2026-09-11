# Dense families protocol

Preregistration for the dense-families plan (`~/Brain/mojo/mojo-baro/briefs/2026-09-11-dense-families.md`).
Each lane owns one section; predictions are frozen by the commit that adds them, before any build or run.

## KATT: head dimension as a comptime parameter

Scope: the attention kernels the Spark path calls, `amar_attn_decode_swa_gated` and
`amar_rope_kv_append` in `kernels/spark_kernels.mojo`, and the helpers they share in
`kernels/attn.mojo` (`attn_head_span`, `kv_off`). Spark has no separate prefill attention
kernel: its multi-row path is the same decode kernel over grid y, so nothing else is in scope.
The head counts ride along as parameters because `kv_off`'s page stride and the GQA group
(`NQH // NKVH`) are part of the same indexing; every target has different head counts.
Defaults equal today's constants (HD 256, NQH 16, NKVH 4), so no call site changes.

Base: `0447adb`. Test: `kernels/test_spark_attn.mojo` runs rope+append then gated decode
attention at (HD, NQH, NKVH) = (64, 32, 8), (128, 28, 4), (256, 16, 4), two query rows,
NAT 2 with att_i 1, T spanning three KV pages, full window and a 100-token window.
Oracle: `tools/spark-attn-ref.py` (numpy, float64) on the dumped inputs.

Frozen predictions:

1. **Parity.** For every HD and both window modes, every output element satisfies
   `|got - ref| <= 2^-8 * |ref| + 1e-5` (half a bf16 ulp plus float32 slack). Zero violations.
2. **HD 256 bit-identity.** (a) The HD 256 test output from the new source is byte-identical
   (`cmp`) to the same test built against the base kernels. (b) The disassembly of every
   AMDGPU code object in the Spark binary (`serve/spark.mojo`) and in the qwen35 engine
   binary (`serve/engine.mojo`, which shares `attn_head_span`) is identical before and after.
   Control: two base builds of the Spark binary produce identical disassembly.
3. **ISA receipt.** At each HD, both kernels show 0 VGPR spills, 0 SGPR spills and 0 private
   segment bytes. VGPR count at HD 64 and 128 is at most the HD 256 count.
4. **Census.** `kernel-census --check` exits 0.
5. **Suite.** `./run-tests.sh` exits 0 before and after; the after run has exactly one more
   test binary (the new test) and its numpy oracle step.

Kill: any violation in 1 or 2 stops the lane after three repair attempts; 3 failing at HD 64
or 128 is reported, not repaired by changing the kernel family.

### KATT results (2026-09-11, commits `26fffe3`, `356d22f`; evidence `.work/KATT-gate.txt`)

| # | prediction | result |
|---|---|---|
| 1 | parity, 0 violations | PASS: 0/4096, 0/7168, 0/8192 per mode at HD 64/128/256; max abs err 1.5e-3 (HD 64 swa); 99.88-100% of elements equal the bf16 rounding of the float64 reference |
| 2a | HD 256 test output byte-identical to base | PASS: `o_full`, `o_swa`, appended row identical (`cmp`) |
| 2b | engine disassembly unchanged | PASS: 16/16 Spark and 186/186 qwen35 engine code objects identical after normalisation; base-vs-base control identical |
| 3 | 0 spills, VGPR(64,128) <= VGPR(256) | PASS: attention VGPR 34/36/40, rope 15/15/15, 0 spills, 0 scratch |
| 4 | census rc 0 | PASS |
| 5 | suite green, +1 test binary | PASS: exit 0 before and after; 101 -> 102 PASS lines |
