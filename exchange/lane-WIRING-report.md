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

## M3 -- MSPEC precondition (DONE, `c9298f9`, INCONCLUSIVE by the preregistered rule)

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
for this lane) rather than deciding silently.

**the maintainer's ruling:** confirmed the full 25-kernel instrumentation is the
stop condition in spirit -- do not do it. Corrected the framing of the
narrower option: stamping a SUBSET of the window's kernels gives an UPPER
bound on gap share, not a lower one (gap_share = (window - sum_kernel_spans)
/ window; fewer measured kernels means a smaller measured sum, which means a
larger apparent gap). Preregistered decision rule: stamp the already-
generalized `gemm_w`-class kernels, report the result explicitly as an upper
bound; under 10% validates MSPEC's correction and clears the kill line,
at/over 10% is INCONCLUSIVE (not a failure or refutation) -- report which
kernels would need stamping next to tighten it, and stop.

**Result.** Extended `bench/carryover-stamp.py`'s `SITES` from the 3 FFN
GEMVs to all 13 `gemm_w[...]` call sites in one decode-loop window
(`serve/window.mojo:842-1290`): 3 FFN (gate/up/down), 4 attn-subblock (qf,
k, v, output), 5 ssm-subblock (qkv, z, a, b, out) -- mutually exclusive per
layer on `is_attn(layer)` -- and 1 head/logits GEMV once per window.
`NSLOT` raised 8192 -> 32768: the 13-site run needs ~8715 stamp slots
(3-site needed ~3360), which would have silently overflowed the old cap
(`stamp_slot()` has no bounds check). Along the way found and fixed a real
bug in `carryover-analyze.py` that the bigger, branching SITES list exposed:
the pairwise gap-name scheme collided on first-letter labels
(`att_qf`/`att_k`/`att_v`/`att_out` all reducing to "a") and divided by zero
on a gap category that structurally never fires for a branching site list;
fixed with full names and a zero guard, plus a docstring note that the
pairwise breakdown (unlike the per-site and window sums, which are
order-independent) only means what it says for a single sequential chain.

Preregistered in `bench/ssm-occupancy-protocol.md` before the run (hash
stamped to `.work/carryover/prereg.sha256`, same convention as the carry-over
probe). 5 runs, Cs arm: identity PASS 10/10 across both arms, 0 WARN,
`STAMPS 8715` every run. Sum of the 13 stamped sites' medians 330.67 ms
against a 479.08 ms window -> **gap_share_upper_bound = 31.0%**, over the
10% line. **INCONCLUSIVE per the rule.** Consistent with expectation, not a
surprise: the 13 sites cover ~249 of ~678 dispatches per window (~37% by
count), and 31% > MSPEC's own traced 15.65% (tracer overhead included) >
its corrected 8.4-9.2% -- the right direction, since measuring fewer kernels
should raise the apparent gap monotonically. Named the highest-yield next
addition: `delta_dispatch` (the SSM scan itself) and `datt_k`/`att_k` (the
attention kernel), since those are the only unstamped kernels that scale
with sequence/state length rather than being O(1) elementwise; the rest
(`rope_q`/`rope_k`, `hrms_q`/`hrms_kv`, `append_k`, `conv_k`, `l2_k`,
`gated_k`, `rgates_k`, `rmsc_k`, `r_swiglu`, `r_add`, `gmul_k`, `split_k`,
`argmax_d`/`argmax_k`, `tokcp_k`, `embed_k`) are lower priority. Stopped
here per the rule; full writeup and the decision rule's exact wording are in
`bench/ssm-occupancy-protocol.md` ("MSPEC precondition check").

## M4 -- make the dense families servable (DONE, `44a1742`)

`serve/spark.mojo` decoded llama/qwen2/granite/spark2_5 at 95.3-100%
teacher-forced agreement but was a one-shot CLI (`grep -rn spark
serve/src/*.rs` = nothing). Extracted the generic byte-scanner protocol out
of `serve/engine.mojo` into `serve/serve_proto.mojo` (`read_line`,
`cancel_pending`, `json_key`/`json_int`/`json_float`, `SampleParams`,
`parse_request`) -- a verbatim move. Verified behavior-preserving with a
same-tree A/B: `git stash` the M4 changes, build `engine.mojo` before and
after on the q4 pack, run both one-shot on the same prompt through one
gpu-wait job. `GENERATED` output byte-identical, `BARO_SPEC`/`BARO_MEGA`
readback and mega fail word identical, only wall-clock timings differ
(run-to-run noise); qwen35moe profile also confirmed still building clean
with the extraction in place.

Wired the same `BARO_SERVE=1` request loop into `spark.mojo`: ready line,
per-token streaming, done line, clean exit on stdin EOF. No explicit
per-request KV cache reset needed -- the cache is position-addressed and
every kernel only reads a position the current request just wrote, so
position 0 always overwrites the previous request's data there.
`BARO_SERVE=0` (every existing dense-protocol.md gate) is untouched: same
range, same final prints, verified by inspection and by the one-shot smoke
runs below reproducing the exact greedy output the CLI path already gave.
No cancel support this round (spark has no draft head, so nothing runs long
enough to need it). Stop-sequence checking IS wired, and turned out to
matter immediately: the first smoke test without it showed granite looping
past EOS to `max_tokens` with `finish_reason: "length"`; added a per-token
check against every `stop` sequence in the request (spark always decodes
m=1, so this is simpler than engine.mojo's per-window check), reran, got
`"finish_reason": "stop"` and a clean two-sentence answer.

**`serve/src/engine.rs` and `main.rs` needed zero changes.** `Engine::spawn`
already takes an arbitrary `--engine` binary and `--pack` dir and speaks
exactly this line protocol -- "make the Rust front able to select the
engine" was already true at the CLI level, just undiscovered because
nothing had ever pointed `--engine` at spark before.

**Gate, verified end to end for 2 of 4 targets** (built profiles + packs via
`tools/gen-profile.mojo` + `tools/engine-pack.py --dense`, fetched
`tokenizer.json` from HF for the open/ungated targets, built `baro-serve`
release, launched it through `gpu-wait run` and hit it with `curl`):

- Qwen2.5-7B-Instruct: `POST /v1/chat/completions` -> `"The capital of
  France is Paris."`, `finish_reason: "stop"`, 94.6 tok/s_gen.
- granite-4.2-3b: `POST /v1/chat/completions` -> `"<think></think>Paris."`,
  `finish_reason: "stop"`, 164.4 tok/s_gen (the empty think-tag is the
  model's own behavior, not a serving defect).

Llama-3.2-1B and lily-cybersecurity-7b are wired through the identical code
path (same `serve_proto`, same spark.mojo request loop) but not verified
through HTTP this round: no `tokenizer.json` fetched for them, and
Llama-3.2-1B-Instruct is gated on HF (needs an accepted-license token this
session does not have). Reported, not silently skipped.

**Also landed the two M1-class leftovers the maintainer caught in review**
(`tools/embed-files.py:17-18`, `bench/ruler/tok.py:31` -- both still pointed
at `$HOME/Projects/mojo/mojo-uregex`/`mojo-minja`, one layer deeper than
M1's fix: the self-describing GGUF closure was still pulling from outside
the repo). `EXT` now points at `ROOT/uregex`/`ROOT/minja`; verified by
re-embedding a fresh spark gguf (`tools/gguf-embed.py` against
`~/Models/spark-x2.5-4b`'s base gguf) and checking the result: the file list
includes `uregex/*.mojo`/`minja/*.mojo`, their embedded content is
byte-identical to the working tree, and the `baro.kernel.ext.*.commit`
provenance keys these two used to need (external package, tracked outside
the repo) no longer appear -- they're covered by the repo's own
`baro.kernel.commit` now, correctly, since they're no longer external.
`tools/gguf-closure.sh`'s rebuild+run gate on that fresh embed was **not**
run: it needs a runtime pack and a `ref-tokens-64.txt` for spark2_5 that
don't exist on this box, its own provisioning round, not a cheap add-on to
this one. Said plainly rather than claimed done.

`README.md` and `docs/ENGINE-ROADMAP.md` updated: the four dense targets are
no longer described as CLI-only.

## M5 -- greedy/sampling toggle (SHIPPED WITH A NAMED OPEN DEFECT, not done)

`kernels/sample.mojo` (device, distribution-tested) and `serve/sample_ref.mojo`
(host oracle) both already existed; `grep -n sample serve/registry.mojo
serve/window.mojo` was zero hits, confirming the brief. Wired
`temperature == 0` as the toggle at `serve/window.mojo`'s non-spec
token-selection call site: `<= 0` keeps `argmax_k` unchanged (byte-for-byte),
`> 0` calls `amar_sample_row` (new `registry.mojo` alias `sample_row_k`) into
`b.dtok_d`/`b.hmax_d` (both free scratch once spec and the megakernel are
off), then the same `tokcp_k` the spec path already uses places the sampled
token at the real position -- no new buffers. Scope limit as specified:
`temperature > 0` forces `spec = False` (`engine.mojo`) and forces the
megakernel off (`mega_token_*` kernels bake greedy argmax into their own
launch, no sampler params, so a sampling request always takes the launch
path where the sampler is wired).

**Gate 1 (byte-identical at temperature = 0): PASS.** Same before/after A/B
methodology as M4's `serve_proto` extraction -- built `engine.mojo` before
and after, ran both one-shot on the q4 pack, same prompt, through one
`gpu-wait` job. `GENERATED` output byte-identical, `BARO_SPEC`/`BARO_MEGA`
readback and mega fail word identical, mega still engages by default (the
new `mega_req = mega and sample.temperature <= 0` local only affects a
per-request WindowCfg field, never the session-level `mega` var other
requests see).

**Gate 2 finding, and the coordinator's diagnosis.** Wrote
`kernels/test_sample_device.mojo`: the first check that runs
`amar_sample_row` (device) and `sample_row_ref` (host) on the *same* inputs
and compares tokens directly, rather than each against its own target
(`kernels/test_sample.mojo` already device-tests the kernel's own
distribution via chi-square on a 64-token synthetic vocab, and
`kernels/test_sample_ref.mojo` already host-tests the reference, but neither
compares the two implementations to each other). Two of four configs
disagreed on every draw checked; sent the raw finding up rather than
guessing which side was wrong. The coordinator's diagnosis
(`exchange/2026-09-15-m5-sampler-diagnosis.md`) resolved it: **the host
reference is the defective side, not the device kernel.**
`serve/sample_ref.mojo` ports the device's `pmass_target` formula
(`W = ceil(top_p * Z)`) from the device's fixed-point mass (unit `2^-40`,
where `ceil` is exact) onto a float64 mass whose unit is `exp(lmax) = 1`; on
a peaked real row, `ceil` rounds the target up to most or all of the top-k
set. The device was never collapsing to one candidate -- it was returning
the correct, small nucleus, while the host's inflated set kept low-mass tail
tokens live. `min_p` masked this in the one config that had it set, on both
sides, which is why that config alone matched.

**Corrected gate status** (per the diagnosis and the coordinator's steer,
`briefs/2026-09-15-m5-steer.md`):
- Gate 1 (temperature 0 byte-identical): **PASS**, real engine A/B, rerun
  again after the refusal below and still byte-identical.
- Gate 2 (device == host per token, real row): **FAIL for `top_p < 1`
  without `min_p`, failing side is the host**, not a device defect --
  `exchange/2026-09-15-m5-sampler-diagnosis.md`.
- Gates 3 (seed reproducibility) and 4 (T=1 chi-square): **PASS at VS=64,
  UNVERIFIED at real vocab.** Both ran only in `kernels/test_sample.mojo`'s
  64-token synthetic vocab at `T1 k- p-`; they say nothing about the top-p,
  top-k, or top-k+top-p shapes at 248320. The missing check is preregistered
  in `bench/chat-protocol.md` ("C3 fix round").
- Also UNVERIFIED at real vocab: `top_k > 0` alone (never in gate 2's
  config list).

**M5 ships as wiring plus a loud refusal, not a fix.** `serve/engine.mojo`'s
`BARO_SERVE` request validation now rejects any request with
`temperature > 0`, `top_p < 1` and `min_p <= 0` before any GPU work, with:
`top_p without min_p is unverified at real vocab
(exchange/2026-09-15-m5-sampler-diagnosis.md, bench/chat-protocol.md C3 fix
round); use min_p > 0 or top_p = 1`. `temperature = 0` is unaffected (gate 1
above). `min_p > 0` and `top_p = 1` configs still sample normally. Verified
with a real `POST /v1/chat/completions` against a running `baro-serve`
(qwen35, `.work/engine-pack-q4`), all three required cases:

- `{"temperature":0.7,"top_p":0.8}` (no `min_p`) -> `502`,
  `{"error":{"code":502,"message":"engine: top_p without min_p is
  unverified at real vocab (exchange/2026-09-15-m5-sampler-diagnosis.md,
  bench/chat-protocol.md C3 fix round); use min_p > 0 or top_p = 1",
  "type":"invalid_request_error"}}`. (502, not 400: every engine-level
  request rejection in `serve/src/main.rs` maps to `BAD_GATEWAY` today,
  including the pre-existing `prompt+n exceeds TMAX` case -- pre-existing
  behavior, not introduced here, and out of this steer's scope to change.)
- `{"temperature":0.7,"top_p":0.8,"min_p":0.05}` -> a real completion
  (`"<think>\n1.  **Identify the core question:** ..."`, `finish_reason:
  "length"`, server log confirms `spec: False`).
- `{"temperature":1,"top_p":1}` -> a real completion (same content, cache
  hit on the identical prompt), `spec: False` confirmed in the server log.

Committed as its own commit, why-body citing the diagnosis file; no
attribution lines. `run-tests.sh` and `tools/ci-checks.sh` both green
(also regenerated `docs/KERNELS.md`, stale from the prior M5 commit missing
`kernels/test_sample_device.mojo` as a caller of `amar_sample_row`).

**Not touched:** `kernels/sample.mojo`, `serve/sample_ref.mojo`. The fix
round (host top-p mass target, arms H0/H1 against a numpy oracle, real-vocab
gate 2 and a real-vocab chi-square) is preregistered in
`bench/chat-protocol.md` ("C3 fix round: host top-p mass target") and waits
for a session with room for it.

Also noted, not fixed (unrelated to this steer): `kernels/test_realign.mojo`
and `bench/bench_latent_handoff.mojo` have their own older `WindowCfg` calls
already missing `dump4`/`dump_layer` from a change that predates this
session; neither is in `run-tests.sh`, and `ci-checks.sh` already skips
`bench_latent_handoff.mojo` for its external `grammar` import, so both are
out of this lane's gates and left as a separate, pre-existing finding.

## C3 -- host top-p target fix (SOURCE FIX COMMITTED, refusal NOT lifted, new finding escalated)

Brief `briefs/2026-09-15-c3-b3-lane.md`. Preregistered arms and gates
(`bench/chat-protocol.md` "C3 fix round: host top-p mass target") run as
frozen, not redesigned.

**Fix.** `serve/sample_ref.mojo`: dropped `ceil()` from the top-p mass
target in both `sample_row_ref` and `sample_probs_ref` (the bug was
duplicated in both functions; fixing only one would have broken gate 3's own
cross-check between them). `w = top_p * zc`, same clamps as before
(`[1, zc]`, where `1` is the top-1 token's own mass in these units).
`kernels/sample.mojo` untouched.

**Three real rows, not one**: `.work/m5/logits-p01.bin` (p01-water, on
record from M5), `.work/m5/logits-p02.bin` (p02-python-fib),
`.work/m5/logits-p03.bin` (p03-story), each a one-shot engine run's own MTP
draft-head logits (regeneration commands in
`kernels/test_sample_device.mojo`'s docstring). Extended
`kernels/test_sample_device.mojo` to loop 3 rows x 5 config shapes (added
the missing standalone top-k-only shape) for gate 1, and added gate 2 (a
20000-draw chi-square per row/config against a new independent oracle,
`tools/sample-nucleus-oracle.py`, numpy, computing the correct nucleus set
and its tempered distribution from scratch -- not derived from either
sampler file). Real-vocab distributions with more than 60 candidates are
capped at 60 explicit bins plus one aggregate "rest" bin, a discretionary
binning choice this round's preregistration did not specify (it froze the
fix formula, not the real-vocab chi-square's bin scheme).

**Gates 1, 3, 4: clean.** Gate 1: 959/960 per-token draws match (the one
miss is a deep-tail probability tie, ~1e-11 range, at a scale where
floating-point noise plausibly flips an argmax between two near-equally
unlikely candidates on either side, and is not a top-p issue). Gate 3:
`kernels/test_sample_ref.mojo` stays green; one VS=64 fixture's `df` moved
10 -> 1 as the preregistration said to expect and report (ceil was not a
no-op there either -- confirms the bug was never exclusively a real-vocab
phenomenon, just far more visible there). Gate 4: M5's temperature-0
byte-identical gate rerun and still PASS -- expected, since
`serve/sample_ref.mojo` is test-only and never linked into
`serve/engine.mojo`'s actual decode path, verified rather than assumed.

**Gate 2: mixed, and this is the open item.** The four shapes this round
actually targets (`T0.8_k30_p1`, `T0.7_k20_p0.8`,
`T1.3_k12_p0.9_minp0.05`, `T0.5_k0_p0.6`) all PASS cleanly on all three
rows, chi2 well under the p=0.001 critical value, zero draws landing outside
the correct candidate set. `T1_k0_p1` (no truncation, `top_p = 1`, the one
shape the ceil fix cannot touch since `p_on = top_p < 1.0` is false for it)
**FAILS on 2 of 3 rows**, not marginally (p02: chi2 136.0 vs crit 77.5; p03:
chi2 555.7 vs crit 86.7; p01 passes, chi2 75.8 vs crit 99.7). In every
failing case the excess sits almost entirely in the aggregate "rest" bin:
device draws land in the deep tail more often than the independent oracle
predicts, and the excess grows as the row's true tail mass shrinks (p01
true tail 16.3% / observed 17.2%, close; p02 3.8% / 5.2%; p03 0.9% / 2.6%,
nearly 3x). Not explained by the bug just fixed. Not visible at
`kernels/test_sample.mojo`'s VS=64 (which already runs the equivalent "T1
k- p-" case and passes, chi2 11.6 vs crit 40.9 -- a 64-token vocab has no
deep tail to expose this in).

**Per the round's own frozen falsifier** ("the device fails gate 2 at real
vocab on any shape; then the device is also wrong and the round widens to
`kernels/sample.mojo`"), this literally fires. Full numbers in
`bench/chat-protocol.md`'s "Result 2026-09-15" subsection (append-only,
under the C3 preregistration). Did not touch `kernels/sample.mojo` and did
not lift the M5 refusal (which only ever covered `top_p < 1` without
`min_p` -- `T1_k0_p1` was always allowed and unaffected by the refusal
either way). Escalated to the coordinator via `herd tell` with the full
picture and three explicit options (lift for the four covered shapes while
T1 is tracked separately / hold the whole round / other), rather than
deciding alone. Committed the source fix and the extended test on their own
strength regardless of the answer: three of four gates are clean, and the
fix is correct independent of the T1 finding. `kernels/test_sample_device.mojo`
is deliberately not wired into `run-tests.sh` (not green yet, per the
preregistration's own "wire it into run-tests.sh only once green").
`./run-tests.sh` and `tools/ci-checks.sh` both green (regenerated
`docs/KERNELS.md` again for the same reason as the M5 commit).
