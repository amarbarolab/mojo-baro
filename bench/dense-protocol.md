# Dense families protocol

Preregistration for the dense-families plan (`~/Brain/mojo/mojo-baro/briefs/2026-09-11-dense-families.md`).
Each lane owns one section; predictions are frozen by the commit that adds them, before any build or run.

## PROFILE: comptime profile module from GGUF metadata (steps 1-2)

**Process note: this section is written after the step-1 build and gate ran, not before** --
the item started mid-conversation from the coordinator with KATT still in flight, and the
build/gate work (`tools/gen-profile.mojo`, the spark2_5 profile wiring) was already done and
verified before this preregistration was written up. Flagged here rather than silently
back-dated; no other lane's rule violation is implied.

Scope: `tools/gen-profile.mojo` (new), `serve/spark.mojo` (import constants from a generated
profile instead of a hardcoded block; recipe flags behind comptime ifs), `kernels/spark_kernels.mojo`
(ROPE_NEOX dispatch in `amar_rope_plain`/`amar_rope_kv_append`, new `amar_bias_add`).

Predictions (step 1, spark2_5 default profile):
1. `tools/gen-profile.mojo` run against all 5 targets' real GGUFs reproduces the plan's table
   (L, H, FFN, heads q/kv, HD) exactly, VOCAB read off `token_embd.weight`'s own tensor shape.
2. Spark rebuilt with `-I` pointed at the generated spark2_5 profile is bit-identical to the
   pre-change build on: the two frozen fixture gates (`.work/spark/ref` text 64/64,
   `.work/spark/chat-ref` chat 43/43) and a fresh 20-prompt greedy A/B against a baseline build
   of the same pre-change source (`bench/mtp-prompts/p*.txt`, `BARO_GEN=64`), 20/20.
3. `./run-tests.sh` exits 0 with the same PASS count as before the change (KATT's floor: 102).

Predictions (step 2, recipe-flag scaffolding only -- not exercised end to end):
4. QKV_BIAS's new `amar_bias_add` path is numerically inert (bit-identical output) when wired
   in with an all-zero synthetic bias on the spark2_5 shapes, proving the new offset-stride and
   kernel-launch code executes correctly without changing decode when the bias is zero.
5. ROPE_NEOX=False (the "norm"/interleaved rope path, needed for llama/lily) is not exercisable
   against a real model this round -- Spark's own weights are neox-native -- so it is verified
   only against a numpy rope reference on synthetic data, not an end-to-end model.
6. GRANITE_MULT's embedding/residual/logit scale multipliers are NOT wired into `spark.mojo`
   this round (design deviation, see report): they fold cleanly into pack-time weight scaling
   instead of runtime kernel branches, and that packer does not exist yet. Only `ATTN_SCALE` is
   wired (already generic; no boolean branch needed since it's a parameter substitution).

Kill: any mismatch in 1-3 stops the lane after three repair attempts and is reported, not
silently downgraded to "close enough".

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
