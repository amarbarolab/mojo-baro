# Wiring lane report

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-15-wiring-lane.md`. Order M1-M5,
each gated, committed on `main` as it lands.

## M1 -- vendor mojo-uregex and mojo-minja (DONE, `d7b3237`)

`tools/ci-checks.sh:115` built with `-I $HOME/Projects/mojo/mojo-{uregex,minja}/src`,
so a clone of this public repo could not build the tokenizer or Spark chat
templating; ci-checks only passed because it hardcoded the machine paths.

Vendored both packages as real files at the repo root (`uregex/`, `minja/`),
same precedent as `b465058`'s GGUF reader: a 5-line banner per file naming
the upstream, `tools/ci-checks.sh` diffs the vendored copy against its
upstream when present and skips cleanly when not. Every build command that
pointed at the sibling repos (`tools/ci-checks.sh`, `bench/dense-run.sh`,
`bench/e13-dump.sh`, `bench/latent-handoff.sh`, plus comments in
`bench/ruler/to_e8.mojo`, `tools/generate_e8_tasks.mojo`, `docs/TOKENIZER.md`,
`serve/tokenizer.mojo`, this repo's `CLAUDE.md`) now resolves through `-I .`.

**Gates, all met:**
- (a) `tools/ci-checks.sh` green, no `$HOME/Projects` path required for any
  build to succeed (the one remaining `$HOME/Projects` reference is the
  drift-check's own upstream pointer, optional and skip-cleanly, same class
  as the gguf_reader precedent's `~/iTools` reference).
- (b) drift check verified both ways: in sync passes; a one-line perturbation
  to a vendored file fails, naming the file (`uregex: drifted from
  .../mojo-uregex/src/uregex: pattern.mojo`).
- (c) `./run-tests.sh` exit 0 (test_gemm, test_prefix, test_sample_ref,
  test_spark_attn + KATT parity, test_latent, kernel-census --check, all PASS,
  0 orphans).
- (d) tokenizer gate unchanged: `tools/test_tokenizer_mojo.py` against the
  default Qwythos target, 61/61 cases, 0 failures, built with `.work/baro-tokenize`
  through the vendored `uregex/` (`-I .`).

Also proved the vendored copy is what actually resolves, not a stale/cached
read of the external path: injected a deliberate syntax error into
`uregex/parser.mojo` and the build failed on it.

## M2 -- persist the in-stream stamping helper (DONE, `3bafc78`)

`.work/carryover/{mkprobe.py,run.sh,analyze.py}` (2026-09-15 carry-over probe)
was the only instrument that could judge a geometry change on this card
(Gate-2-style per-kernel timing with host syncs provably cannot, verdict 3 of
`exchange/2026-09-15-carryover-probe.md`), and it lived in scratch where it
would be deleted.

Promoted to `bench/carryover-{stamp.py,run.sh,analyze.py}`, this repo's flat
bench/ naming (matches `dattn-run.sh`, `clock-probe.sh`: a documented
instrument, no separate protocol.md needed, same as `clock-probe.sh`).
Generalized via two registries so the tool remains able to stamp a dispatch
site other than the three FFN GEMVs the probe measured: `KERNELS` (reusable
per-kernel timer insertion, anchor strings only) and `SITES` (which call
sites route through a stamped kernel); `carryover-analyze.py` takes
`--site-names` instead of hardcoding gate/up/down. Verified this is not a
rewrite: the promoted `stamp.py` reproduces byte-identical patched sources to
the original tool (diffed directly; the one file that differs,
`serve/tokenizer.mojo`, differs only by M1's comment edit, which landed on
the working tree after the original probe ran), and the promoted
`analyze.py` reproduces byte-identical report text on the original probe's
own logs.

**Gate:** fresh 5-round-per-arm run through `carryover-run.sh S` (10 gpu-wait
jobs, 0 WARN, identity PASS all 10 -- every Cs/D2s run's `GENERATED` hash
matches `c00468774758`). Reproduces the probe's finding: D2/C in-kernel
shader clock ratio 1.002-1.006 at gate/up/down (published 1.0033/1.0047/1.0059),
and the absolute per-site clock lands within ~1% of the published
2949/2954/2976 MHz (this session: 2980/2986/3001). No formal spread was
published for those specific per-site numbers to check against more
tightly; the qualitative finding and the ratios reproduce closely, and a ~1%
cross-session gap is smaller than the cross-session decode variance already
on record in this repo (baton 2026-09-11: w3 run-to-run spread 7.3% vs
champion's 1.8%). This session's overall decode_s also ran ~3.5-8% slower
than the original probe's, consistent with ordinary thermal/ambient
variance rather than a tool defect -- the ratio between arms (what the probe
actually claims) held.

## M3 -- MSPEC precondition (BLOCKED, question sent, moving to M4)

"Stamp every dispatch of one MSPEC verify window" is not a small extension of
M2's tool. Enumerated the launch-path window's kernel dispatches
(`serve/window.mojo:842-1290`, `MEGA_ALLOWED` profile): ~25 distinct kernel
functions beyond the one GEMV kernel (`q4rowb`/`gemm_w`) M2 already
generalized -- `rmsc_k`, `split_k`, `hrms_q`/`hrms_kv`, `rope_q`/`rope_k`,
`append_k`, `datt_k`, `dcomb_k`, `gmul_k`, `r_add`, `rgates_k`, `r_h`,
`r_swiglu`, `conv_k`, `gated_k`, `l2_k`, `amar_ssm_gated_out_bf16`, `r_head`,
`argmax_d`/`argmax_k`, `tokcp_k`, `rms_m`, and more. Stamping only the
already-generalized GEMV class would lump every other kernel's real execution
time into "gap," which would not answer the actual question (whether the
8.4-9.2% corrected bound holds) -- it would systematically overstate gap
share. Doing all ~25 properly is the same TECHNIQUE as M2 (instrumented copy
under `.work/`, tracked `kernels/*.mojo` never touched) but is real
kernel-body surgery at 25x the scale, each kernel needing its own
control-flow anchors gotten right; that reads like the brief's own
kernel-authoring stop condition in spirit, even though tracked source stays
untouched. Sent the scope question to the coordinator
(`herd tell w7T:p1`) with three options (do the full 25-kernel
instrumentation / report a partial lower-bound gap share from the
already-generalized kernels only / skip M3 and report it scoped too large
for this lane) rather than deciding silently. Moving to M4, not gated on
this answer.

## M4-M5

Not started.
