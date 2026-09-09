# E8 HARNESS -- Topology 2 (Spark-4B) blocker

Checked before writing any harness code, per the item's instruction to verify
the structure transfers before building against it.

`serve/spark.mojo` is a self-contained, single-file engine (342 lines, its own
`def main()`) with **no** `WindowBufs`, `WindowCfg`, `WindowState`, `step_window`
or `hn_d` -- none of those names appear in the file (`grep -n
"WindowBufs\|step_window\|hn_d\|from window import" serve/spark.mojo` -> only
`def main` matches). It has its own:

- `load_pack(ctx, packdir, mut off) -> DeviceBuffer[DType.uint8]` (different
  signature from `latent_harness.load_pack`/`serve/harness.load_pack`: no
  `Pack` struct, offsets passed by out-parameter).
- its own dims (`H=2560`, `FFN=10240`, `VOCAB=131072`, `N_LAYERS=36`, its own
  `TMAX=4096`/`KVPOOL`), its own kernel set (`spark_kernels.amar_*`, RoPE with
  a full/SWA split, `amar_attn_decode_swa_gated`), and no megakernel /
  `step_latent_raw`-shaped injection point at all.

None of `bench_latent_handoff.mojo`'s machinery (`WindowBufs`/`WindowCfg`/
`WindowState`/`step_window`, the `step_latent_raw` raw-vector injection, or
the `serve/realign.mojo` interface contract, which is typed against
`WindowBufs`) transfers to Spark without a dedicated Spark-side harness --
its own buffer-alloc, its own prompt/latent-injection loop, and either a
second `realign`-equivalent typed against Spark's state or a shared
abstraction over both engines. That is new engine work, and HARNESS's
non-goals are explicit: "no changes under kernels/ or serve/ except the
stub... do not build a Spark engine."

**Estimate** (order of magnitude, if a future lane takes this on): a
Spark-side `spark_harness.mojo` (pack-load + buffer-alloc + a
`step_latent_raw`-equivalent single-token injection, mirroring this item's
`bench/latent_harness.mojo` + the `step_latent_raw`/`collect_latent_*`
helpers in `bench_latent_handoff.mojo`) is roughly 150-250 LOC; a
`spark_realign.mojo` kernel (softmax-over-VOCAB + weighted embedding gather,
VOCAB=131072 vs Qwythos's 248320) is REALIGN-item-sized work, not a HARNESS
sub-task. Topology 1 (this item) was built first and is the gate-defining
run, as the plan requires; Topology 2 is not attempted here.
