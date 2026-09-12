# Lane KATT report

Branch `lane-KATT` (worktree `~/Projects/mojo/mojo-baro-lanes/KATT`), base `0447adb`. Ran on Opus 5
because Fable was capped at dispatch (plan note, CLAUDE.md §10 fallback).

**Result: all gates pass.** Head dimension is now a comptime parameter of the Spark attention path.
HD 64 and 128 build from the same source as HD 256, and HD 256 is bit-identical to today.

## Flags (read first)

- **Design call: the head counts ride along with HD.** `kv_off`'s page stride includes `NKVH`, and the
  GQA group is `NQH // NKVH`. Every target has different head counts, so an HD-only parameter would be
  unusable. The kernels take `HD_`, `NQH_`, `NKVH_`, all defaulting to today's 256/16/4. No call site
  changed.
- **Files touched outside the item's kernel list:** `kernels/test_spark_attn.mojo` (new test),
  `tools/spark-attn-ref.py` (new numpy oracle), `run-tests.sh` (+7 lines wiring both in), and
  `bench/dense-protocol.md` (preregistration, as the plan's rules require). The gates needed the test
  and oracle, and the suite has to run them.
- **Spark has no separate prefill attention kernel.** Its multi-row path uses the same decode kernel
  over grid y, so `amar_attn_prefill` (qwen35 only) was left alone.
- **`docs/KERNELS.md` was not regenerated.** `--check` passes without it. A regen would change my two
  signature rows and add the test row, but it also rewrites three rows of pre-existing drift from main.
  One of those rows is wrong because of a census bug (last line). Regenerate at merge once that bug
  is fixed.

## Gate

```
gpu-wait run --vram 20 --timeout 3600 -- ./run-tests.sh
```

| | exit | PASS lines | test binaries |
|---|---|---|---|
| before (`.work/KATT-gate-before.txt`) | 0 | 101 | 3 + census |
| after (`.work/KATT-gate.txt`) | 0 | 102 | 4 + numpy oracle + census |

The plan item gives no numeric floor. The new test adds one binary and one oracle PASS line.
Evidence is in `.work/KATT-gate.txt`: suite output, then the ISA, bit-identity and census sections.
Raw dumps are in `.work/katt/`.

## Results against the frozen predictions (`bench/dense-protocol.md`, KATT)

The test runs rope plus KV append, then gated decode attention. It covers two query rows, NAT 2 with
att_i 1, and 301 tokens across three KV pages. Each head dim runs with the full window and a
100-token window.

| HD (NQH/NKVH) | violations full / swa | max abs err | attn VGPR | rope VGPR | spills | LDS |
|---|---|---|---|---|---|---|
| 64 (32/8) | 0/4096, 0/4096 | 1.5e-3 | 34 | 15 | 0 | 520 |
| 128 (28/4) | 0/7168, 0/7168 | 8.3e-4 | 36 | 15 | 0 | 1040 |
| 256 (16/4) | 0/8192, 0/8192 | 9.7e-4 | 40 | 15 | 0 | 2080 |

- **Parity bound.** Each element must satisfy `|got - ref| <= 2^-8 |ref| + 1e-5`. Between 99.88% and
  100% of elements equal the bf16 rounding of the float64 reference.
- **HD 256 bit-identity, test output.** The same test was built against the base kernels
  (`git archive 0447adb`). A sed dropped the explicit HD arguments and the HD 64/128 cases. Its
  `o_full`, `o_swa` and appended cache row match the new build byte for byte.
- **HD 256 bit-identity, disassembly.** All 16 Spark (`serve/spark.mojo`) and 186 qwen35
  (`serve/engine.mojo`) code objects have identical disassembly before and after. The comparison drops
  the objdump path line and masks the 16-hex parameter hash in kernel names. A base-vs-base control
  build was also identical.
- **Census.** `kernel-census --check` rc 0: 85 kernels, 0 orphans.

## Commits on `lane-KATT`

- `17056d4` bench(dense): preregister KATT head-dimension gates
- `26fffe3` feat(kernels): head dim as a comptime parameter of the Spark attention path (+28/-27 in the two kernel files)
- `356d22f` test(kernels): KATT parity for the Spark attention path at HD 64/128/256
- `74f4d29` bench(dense): record KATT results against the frozen predictions

## For PROFILE (what `serve/spark.mojo` must change to use HD 64/128)

- Pass `HD_, NQH_, NKVH_` positionally after `NAT` in the `amar_attn_decode_swa_gated`
  instantiation. `amar_rope_kv_append` takes `HD_, NKVH_` after its layouts.
- Compute `KVPOOL` from the profile's HD, not `attn.KVHSTR`. Launch with `block_dim` set to the
  profile HD. HD must be a multiple of 32, and a comptime assert enforces it.
- The attention scale is hardcoded as `Float32(0.0625)`, which is 1/sqrt(256). It must become
  1/sqrt(HD), or Granite's attention multiplier.

Finding outside scope: the census gets a test's runner by substring-matching its stem against the
gate script, so it reports `test_sample.mojo` as run by `run-tests.sh` because `test_sample_ref`
contains it (`tools/kernel-census.mojo:291`).
