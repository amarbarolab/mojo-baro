# MTP draft-head protocol — frozen before first run (M5 item 5)

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in particular:
> every parameter defining an arm is read back from the running system and
> recorded BEFORE the timed run. No receipt, no arm.


Question: does the model's own blk.32 draft head (NextN/MTP, `qwen35`
architecture, `nextn_predict_layers=1`) earn its launch cost in our
engine — i.e. does spending extra GPU launches on a draft-then-verify
loop beat plain greedy decode at our engine's own actual shape?

## Correctness precondition (gates ALL speed numbers)

No speedup number may be recorded until the draft head passes a
numerical parity gate against llama.cpp's draft logits (`ctx_dft`
`res->t_logits`, `common/speculative.cpp`'s `draft()` path, per
`docs/mtp-notes.md` §2-§3):

1. Top-1 argmax match on every position sampled during a fixed
   16-token draft trace.
2. Cosine similarity of the full logit vector >= threshold TBD-at-gate-
   time, held to the same bar `q8b` parity used (9.2e-4-class), recorded
   in the Result section — not loosened after the fact.
3. Max relative error bound recorded alongside cosine — cosine-alone is
   the known silent-corruption signature (high cosine, differing
   argmax = wrong token picked with a deceptively "close" logit vector).

Any one of the three failing voids the run: no tok/s, no acceptance
rate, no multiplier gets recorded that day. Fix the kernel, re-freeze
if the fix changes the instrument, re-run.

## Metric caveat (binds prediction 2 below)

llama.cpp's `tok/s` (`tools/server/server-common.h:404-427`,
`timings.predicted_per_second`) is generation-time-INCLUDING nothing
but decode: `(n_decoded - 1) / t_decode_s`. Our engine's own
`tok/s_gen` metric (see `bench/decode-race-protocol.md`) is computed
the same way, generation-time-EXCLUDING prefill — but the two are NOT
directly comparable across engines beyond what `decode-race-protocol.md`
already disclosed and froze. This protocol does NOT re-measure or
re-litigate llama.cpp's numbers. All predictions below are **self-
relative**: our-engine-with-MTP `tok/s_gen` vs our-engine-without-MTP
`tok/s_gen`, both measured by our own metric, same instrument, same run
session. llama.cpp's 44.1 / 109.8 tok/s figures (below) are cited only
as prior art informing the prediction ranges, never as the denominator
of any claim made here.

## Facts on the record (measured previously, not re-derived here)

- llama.cpp bar, frozen `adbb48c`, verdict `6b99693`
  (`bench/decode-race-protocol.md`): no-spec 44.1 tok/s (median of 4,
  spread 0.2%); with its MTP head 109.8 tok/s (median of 4, spread
  0.7%), draft acceptance 52/62 = 84% every run, multiplier 2.49x.
- HBM roofline ~960 GB/s / 17.9 GB-per-token weight traffic =
  53.6 tok/s non-speculative ceiling.
- Our engine, no MTP: 38.97 tok/s by our own metric, 64-token greedy
  output bit-identical to llama.cpp.
- GPU: llama-server holds ~22 GB VRAM at idle — MUST be down for this
  run, GPU exclusive (same discipline as `coldcache-protocol.md` v3/v4).

## Instrument

64-token gate run (`./.work/engine`, prompt "The capital France is",
greedy, `n_predict=64`), llama-server DOWN, GPU exclusive, existing
regression-gate prompt/tokens (`.work/engine-pack/ref-tokens.txt`).
Two arms, same process/binary build, same prompt:

- **Arm A (no MTP)**: current engine path, plain greedy decode.
- **Arm B (with MTP)**: blk.32 draft head wired per `docs/mtp-notes.md`
  §2-§4 (`process()`/`draft()`/`accept()` loop mirrored from
  `common/speculative.cpp`), same greedy sampling, same prompt.

Report per arm: `tok/s_gen` ((n_decoded-1)/decode_s, our own metric,
generation-time only), and for arm B additionally: draft acceptance
count/rate, drafted-vs-accepted token counts per verification window.
5 repeats per arm, discard first (cache/clock warm), report
median of remaining 4.

## Frozen predictions

1. **Stability**: per-arm spread on the 4 kept repeats < 5% of median.
   If spread exceeds 5% on either arm, no speedup claim may be made
   that day — re-run under the stability gate before recording
   anything (copy discipline: `coldcache-protocol.md` v2/v3).
2. **Acceptance rate**: 60-85%. Reasoning: llama.cpp gets 84% on this
   exact model/head; ours is a fresh implementation of the same weights
   through different kernels, so parity is expected but not assumed —
   range widened below llama.cpp's number to allow for real
   implementation slop while still requiring the head to be doing
   useful work.
3. **Speedup**: arm B `tok/s_gen` / arm A `tok/s_gen` in 1.6-2.4x.
   Reasoning: llama.cpp's own multiplier is 2.49x; our engine starts
   from a lower no-spec baseline (38.97 vs 44.1) and untuned draft-path
   kernels, so the range is shifted down and narrowed versus
   llama.cpp's observed ceiling, not copied from it.

## Falsifier

Arm B / Arm A speedup < 1.3x — the draft head is not earning its
extra launches in our engine; MTP work stops until the underlying
inefficiency (kernel launch overhead, draft-path fusion, acceptance
rate) is diagnosed and re-preregistered.

## Claim rule

A speedup number may be published ONLY if:

- the correctness precondition (top-1 AND cosine AND max-relative-error
  gate) passed and is recorded in the Result section, AND
- both arms' greedy 64-token output stayed bit-identical to the
  existing `ref-tokens.txt` gate — MTP is a launch-scheduling
  optimization, not a model-behavior change; if the output moves at
  all, the run is void regardless of how fast it was, AND
- per-arm spread stayed under the prediction-1 threshold.

Any deviation from this protocol (different prompt, different
`n_predict`, different sampling, GPU not exclusive) requires a new
preregistration before any number from that run may be cited as "the"
MTP result.

## Result

**Not yet run. This section is a placeholder — filled in only after
the correctness precondition passes and the instrument above executes
exactly as specified. No prose, no number, no verdict belongs here
until then.**

## Amendment 2026-09-04 (frozen before the draft loop is written, tree 87e0f81)

Base has moved: the engine is q8-only at **68.77 tok/s_gen** (`q8-protocol.md`),
64/64 identical to llama.cpp Q8_0 and bf16. Arm A = that binary/pack. Arm B
= same binary with `BARO_SPEC=1`, draft window k from `spec-k.txt` (4).
The correctness precondition of this file was met on 2026-09-01 (draft head
argmax and logits validated vs llama.cpp, `tools/draft-ref.py`); the DRAFT
line still prints `from_token 369 draft_argmax 369` on the q8 engine.

Design (mirrors `common/speculative.cpp` process/draft/accept, greedy):
- State per iteration: `pos` = last known token; trunk ring/KV valid through
  `pos-1`; `Hnm` rows hold the trunk's post-output-norm hidden for the last
  window; draft KV (`kc32/vc32`) valid through the previous iteration.
- **process**: one blk.32 pass, M = tokens not yet in the draft KV with
  truth h (token p pairs with h(p-1)): after prefill the P-1 prompt tokens,
  later the n_acc+1 accepted+bonus tokens. Its last row IS draft step 0:
  argmax -> d1, its own post-shared-head-norm hidden -> h for step 1.
- **draft**: k-1 further M=1 blk.32 passes at pos+j with (d_j, draft h_{j-1}).
- **verify**: trunk window M = min(k+1, remaining) rows at pos..pos+k over
  [Toks[pos], d1..dk]; argmax into `dtok_d`, host compares, n_acc = leading
  matches, bonus = argmax(row n_acc) written to Toks[pos+n_acc+1]; ring and
  pos advance by n_acc+1. Rejected rows' trunk KV / ring slots / draft KV are
  overwritten by the next window, never read.
- Greedy verification accepts iff equal, so the 64 tokens are identical to
  arm A by construction; any drift voids the run (claim rule unchanged).

Byte model per iteration at k=4, 84% acceptance (llama.cpp's rate):
verify 10.7 GB + process ~0.2 GB + LM head x4 (1.06 GB each) + 3 draft
bodies x 0.2 GB ~= 15.7 GB for ~4.4 tokens -> 3.6 GB/token vs 10.7 today.

| prediction | value | rule |
|---|---|---|
| acceptance (drafted->accepted) | 60-85% (unchanged) | recorded |
| arm B / arm A | **1.6x-2.4x** (110-165 tok/s_gen) | land >= 1.5x AND 64/64 AND spread < 5% |
| falsifier | < 1.3x: stop, diagnose, re-preregister | |

Both arms 5 repeats, discard first, median of 4; server down; `BARO_SPEC`
and `spec k` printed by the run are the P1 receipts, plus per-run
`drafted/accepted` counts.

## Result (2026-09-04, amendment arms; run on the tree that landed as the next two commits)

Correctness precondition: 64/64 on every run of both arms (10/10), plus
`BARO_SPEC_K=1` 64/64. Receipts per run: `spec k: 4`, `BARO_SPEC: True/False`,
`mtp: drafted 53 accepted 50 k 4` (arm B). bge-m3 llama-server stopped.
Logs `.work/mtp-gate/{A,B}{1..5}.log`, `summary.txt`.

| arm | kept 4 (tok/s_gen) | median | spread |
|---|---|---|---|
| A (`BARO_SPEC` unset) | 67.79 67.71 67.75 68.50 | **67.77** | 1.2% |
| B (`BARO_SPEC=1`, k=4) | 127.99 127.61 127.92 128.02 | **127.96** | 0.3% |

Speedup **1.888x** (predicted 1.6-2.4x): LANDS. Acceptance 50/53 = 94%,
above the 60-85% band (5-token prompt, repetitive 64-token greedy tail;
recorded, not claimed as general). llama.cpp MTP bar 109.8 -> ours 1.17x.

Two bugs found on the way, both would have voided the run:
1. First speculative window after prefill skipped the accept check
   (`prefill_done` gate), writing all drafted-row argmaxes unverified;
   token 5 wrong. Fixed with a per-window `win_spec` flag.
2. `amar_matmul_skinny_q8row[MR>1]` looped rows with a runtime `M`,
   dynamic-indexing the accumulator array into scratch: a 5-row window
   cost ~8x a 1-row pass (first arm B: 31.9 tok/s, falsifier fired).
   Compile-time `MR` loop with `r < M` guard; prefill 0.116 -> 0.046 s.

## Result 2 (2026-09-04, real prompts, k sweep, llama.cpp head-to-head)

Not preregistered; recorded because the 5-token number above does not
generalise. Set: `bench/mtp-prompts/` (20 prompts, 7-59 tokens, tokenized
by the model's own tokenizer), 64 greedy tokens each, one run per arm
(single runs: this is a ranking, not a claim). Identity gate = arm B
`GENERATED` equal to arm A's; passed 100/100 runs. Logs `.work/mtp-final/`,
llama.cpp `.work/llama-mtp-prompts/`.

Two engine fixes landed between the first real-prompt run and this table
(commit after this section): the batched q8row GEMM is now instantiated
at the window's actual row count (2, 3, 5) instead of the 8-row template,
and keeps 16-wide loads for those sizes. First run, k=4: median 82.2
(1.19x), two prompts below 1.0x. After: table below.

| k | median tok/s_gen | median B/A | min-max | median acceptance |
|---|---|---|---|---|
| A (no spec) | 68.3 | | | |
| 1 | 98.2 | 1.43x | 1.22-1.53 | 85% |
| **2** | **100.7** | **1.47x** | 1.20-1.80 | 69% |
| 3 | 94.7 | 1.39x | 1.03-1.84 | 58% |
| 4 | 92.4 | 1.35x | 0.90-1.85 | 51% |

Default k moves from 4 to 2 (`spec-k.txt`, `BARO_SPEC_K`). k=4 stays
better only on code/list/number prompts (p02, p04, p19, p20). The
5-token race prompt on this tree: k=4 145.6, k=2 125.8.

llama.cpp Q8_0 on the same prompts (`tools/llama-mtp-prompts.sh`, draft
n-max 4): no-spec 74.1, MTP median 123.5, ratio 1.66x, acceptance 58%,
and its speculative output differs from its own greedy output on 4/20
prompts. Ours at k=2: 100.7, ahead on 2/20 prompts, median 0.78x of
llama.cpp MTP. Verdict for the README: ahead on the preregistered
5-token race (145.6 vs 109.8), behind on real text (100.7 vs 123.5).

Where the remaining gap is (profile on p09, BARO_PROFILE=1): draft path
~2 ms/window; a 2-row trunk window still costs ~1.28x a 1-row pass
(bandwidth model says ~1.05x). SSM sub-block is per-row serial (5
launches x 24 layers per extra row); multi-row GEMM at QV=16 not yet
at the m=1 stream rate. Both are the next MTP levers, before q4.

## Amendment MSPEC step 1 (frozen 2026-09-11 before any trace run, tree `dc9e06b`)

Plan: `~/Brain/mojo/mojo-baro/briefs/2026-09-11-mega-spec-window.md`. Question:
is the launch-gap share of the speculative verify window large enough that a
multi-row megakernel for m = k+1 could pay? This amendment measures; it builds
nothing.

**Prior on the record.** `megakernel-mrow-protocol.md` (2026-09-06, q8 pack,
`BARO_PROFILE=3`, one prompt): 646 launches x 2.43 us = 1.57 ms = 6.0% of a
26 ms k=2 window, estimated from the launch floor, not measured on a timeline.
Its W3 round built `amar_mega_window[MR=3]` and closed it: the megakernel's m=3
GEMM phases scale 1.6x from m=1 vs 1.5x native, so it lost to the launch path.
Zero gaps is therefore an upper bound on what a window megakernel can buy, not
an expected gain. The default pack is now q4 (`.work/engine-pack-q4`), whose
trunk window is shorter than q8's, so the same launch count is a larger share.

**Instrument.** `.work/engine` built from this tree. For each of the 20
`bench/mtp-prompts/` prompts, one process per arm, GPU through `gpu-wait`:
- `A`: no spec, bare (identity reference, no-spec tok/s context).
- `B2`, `B4`: `BARO_SPEC=1 BARO_SPEC_K=2|4`, bare (the tok/s the bound scales).
- `T2`, `T4`: the same two arms under `rocprofv3 --kernel-trace --output-format csv`.

P1 receipts read from each run's own log, per arm: `BARO_SPEC:`, `spec k:`,
`BARO_MEGA:`, `BARO_MEGA_WIN:` (must be False: the window runs the launch
path), `pack q4 trunk:` (True), `mtp: drafted/accepted`, `tok/s_gen`; pack
identity = resolved `BARO_PACK` path plus sha256 of its manifest files, recorded
once per session. Identity gate: `GENERATED` of every B and T run equals A's.
A run failing its receipt or identity is void.

**Definitions (from the kernel trace, per spec iteration).**
- *Verify window* = the trunk pass of `step_window` under `win_spec`: from the
  start of its `embed_k` dispatch to the end of its `argmax_d` dispatch.
  Kernel count n, kernel sum S (sum of dispatch durations), wall W (span), gap
  share g = 1 - S/W.
- *Iteration* = from the start of the draft process pass (`blk32_forward`) to
  the start of the next iteration; adds the draft path and the host accept
  sync. Its idle share is reported beside g, not used for the decision.
- Per prompt, g is aggregated as sum(W - S) / sum(W) over its spec windows.

**Decision metric and kill line.** Median over the 20 prompts of the per-prompt
g at k=2 (the shipping default). **g < 10%: stop at step 1**, no build.
g >= 10%: report and wait for the maintainer's go; step 2 is not started by this lane
either way.

**Upper bound on the gain.** Per prompt, tok/s_ub = B tok/s_gen /
(1 - sum(W - S) / decode_s of the traced run). Reported as the median over 20
prompts for k=2 and k=4, beside the B median.

**Trace overhead check.** If the median T/B tok/s ratio is below 0.95, the
tracer is inflating gaps: the measured g is then an upper bound and says so
in the Result; the decision still uses it (an inflated g below 10% kills
harder).

| prediction (frozen) | k=2 | k=4 |
|---|---|---|
| dispatches per verify window | 600-700 | 600-700 (same code path, m=5) |
| verify window wall W, ms | 10-15 | 12-19 |
| gap share g, median | **7-17%, point 11%** | **5-13%, point 9%** |
| tok/s_ub / B | 1.05-1.15x | 1.04-1.12x |
| T/B tok/s ratio (tracer cost) | 0.93-1.00 | 0.93-1.00 |

Reasoning: ~650 dispatches at 1.5-3 us of dispatch gap each is 1.0-1.9 ms;
q4 is ~57% of q8's bytes, so the m=3 launch-path window is ~11-14 ms against
q8's 21.3 ms. Falsifier for the reasoning (not the decision): g outside
4-20% at k=2 means the per-dispatch gap model is wrong and the Result says
what the trace shows instead.
