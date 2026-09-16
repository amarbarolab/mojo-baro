# Protocol rules

Rules that bind every protocol in this directory. P15 to P20 (GPU time, loud failure, minimal code); the plan to apply them to existing scripts is `docs/design/gpu-efficiency.md`. A protocol may add
constraints; it may not relax these. Cited by `coldcache-protocol.md`,
`decode-race-protocol.md`, `mtp-protocol.md`, `ssm-occupancy-protocol.md`.

## P1. Parameter verification is mandatory and precedes every run

**Passing a parameter is not evidence that the parameter took effect.**
Before any timed run, every parameter that defines an arm must be proven
active from the instrument's own reporting, and that proof recorded in the
run record next to the numbers.

An arm's parameters are the ones whose values distinguish it from another
arm, plus any value quoted in the result (dtype, shape, KV type, context
length, spec type and draft width, thread/block dims, n_predict, temperature,
buffer count, clock state).

The check is: **read the value back from the running system, not from the
command you typed.** Sources that count as read-back —

- llama-server: `GET /props` for effective load-time settings; the response's
  `timings` block (`draft_n`, `draft_n_accepted`, `speculative`) for per-request
  settings; server stderr at load for what the flags resolved to.
- our engine: values printed by the run itself (`prompt tokens:`, `spec k:`,
  `tokens:`), and for comptime constants the binary being timed rebuilt in the
  same command as the run.
- hipBLASLt / vendor calls: the algo actually selected and its tuned attributes
  (splitK, wgm, workspace), dumped from the shim, not assumed from the heuristic.
- kernels: grid/block dims and template parameters echoed by the bench harness.

Sources that do NOT count: the flag string, the request JSON, the protocol
text, a previous run's receipt, or a maintainer's recollection.

**No receipt, no arm.** An arm whose parameters were not verified before its
timed run is VOID and its numbers may not be recorded, cited, or compared —
identical standing to the >10% spread rule. A void arm is re-run, not
retro-justified.

If a parameter cannot be read back, it may not be set per-request. Define the
arm by a controlled restart or rebuild whose configuration IS observable,
record that this was necessary, and restore the original configuration with a
health check afterwards.

### Why this rule exists

Both of this repo's known poisonings were silently-inert parameters, not bad
math:

- **decode-race arm B**: `"speculative.n_max": 0` was accepted by the server
  and ignored by that build. `draft_n` in the response was unchanged, so the
  "no-MTP" arm was still speculating. Caught only because someone read the
  timings block. Had it gone unread, the published no-spec bar would have been
  the MTP number and every later comparison would have inherited it.
- **hipBLASLt shim**: splitK and wgm were never actually set, so our GEMM
  measured ~2x faster than "the vendor". The parameter absence *was* the
  result. Fixing it took hipBLASLt 2497 -> 5201 GFLOP/s and erased the lead.

Both cases produced clean-looking numbers with tight spreads. **Spread does not
detect an inert parameter** — a consistently mis-configured arm is consistently
mis-configured. P1 is the only check that catches this class.

## P2. Verification comes before prediction freeze

Verify parameters, then freeze predictions, then run. Verifying after the run
lets the observed number inform which knobs get scrutinised, which is the same
defect as an unfrozen prediction.

## P3. The receipt is part of the result

Record, in the protocol's Result/Verdict section: what was checked, where the
value was read from, and the observed value. A verdict without its receipt is
incomplete and may not be promoted to "the bar".

## P4. Decode and speculation claims come from the prompt set, never one prompt

Any tok/s, speedup or acceptance number quoted as "ours" or "the bar" is the
median over `bench/mtp-prompts/` (20 prompts, `bench/mtp-prompts.sh`), with
the min-max range beside it, and the competitor measured on the same set in
the same session (`tools/llama-mtp-prompts.sh`). A single-prompt run is an
instrument receipt or a bug-isolation probe; it may appear in a protocol as
such, never in README or a verdict line.

Why: 2026-09-04 the MTP loop was reported at 128 tok/s and 1.17x llama.cpp
from the 5-token race prompt (94% acceptance). On 20 prompts it was 82 and
0.66x. Both numbers were correct; only one described the engine.

## P5. Every kernel run at m > 1 carries a row-scaling receipt

Before a kernel is dispatched at m > 1 on a timed path (prefill, speculative
window), the cold-cache bench records its cost at m = 1, 2, 4, 8 on the ffn
shape, and the m-row / 1-row ratio is in the landing commit. A ratio above
1.25x at m = 4 on a bandwidth-bound shape is a defect to fix or a documented
falsifier, not a property to live with.

Why: the q8row kernel's m > 1 path shipped inside the q8 round on the m = 1
receipt alone. It cost 8x per window (runtime-indexed accumulators), then
2.9x (8-row template with half-width loads), and every MTP number sat on it
for a day.

## P6. Profile the harness before the kernel

When a sweep's wall time is more than 2x its summed GPU time, or a run prints
a load/setup time above 10% of its total, the harness (loader, host copies,
per-window synchronizes) is profiled and fixed before any kernel is touched.
Run receipts include `pack loaded in` next to `tok/s_gen`.

Why: the engine loader copied 10.7 GB with a per-byte loop, 12.6 s of every
17 s run; a 100-run sweep cost 30 min and a whole optimisation stint was
planned around waiting for it. memcpy: 1.8 s.

## P7. The arm file must be written by the run, never alongside it

Every A/B harness prints each arm's binary sha256 **from inside the code path
that executes that arm**, and refuses to run when the two arms hash equal.
An arm file assembled separately from the execution can name a binary that
never ran.

Why: 2026-09-12, the YaRN ramp A/B. The harness ran `.work/engine-yarn$arm`
in a loop while a `sed` had rewritten only the arm-header line to say
`engine-yarnB2`. The receipt reported sha `dbb3a84c`; the binary that
actually decoded was `96752fb4`. The numbers were real, reproducible, and
described the wrong experiment. `bench/force-ab.sh` had the refusal check
from the start and is the pattern to copy.

## P8. A null result is not a finding until the changed code is proven reached

A change that compiles, produces a different binary hash, and moves nothing
has two explanations: the hypothesis is wrong, or the edited line never
executed. Distinguish them before recording a falsifier. Cheapest proof:
make the change absurd (invert it, zero it, delete the call) and confirm the
output *does* move; then apply the real change.

Why, three times in this repo:
- The FREQ_SCALE falsifier edited `kernels/attn.mojo` where those constants
  no longer lived. Byte-identical binary; would have read as "rope
  exonerated" from an experiment that never ran. Only the matching hash
  exposed it.
- The MoE name-resolution change rewrote every weight offset in `moe_ffn`
  and produced twenty prompts identical to the digit. Reached, correct, and
  equivalent: the old arithmetic already resolved to the same addresses.
  Proven only by corrupting it deliberately (gate/up swap -> 1/64).
- The YaRN ramp flip patched `kernels/attn.mojo` while dense decode runs the
  megakernel's own copy at `kernels/mega.mojo:747`. 15 of 20 prompts came
  back byte-identical, which is impossible if 14 of 32 rotary pairs had
  changed. The data shape caught it, not the hash.

Corollary: when one formula exists in more than one file, a fix that lands
in some of them is the default outcome. The YaRN ramp has five copies
(`kernels/attn.mojo`, `kernels/mega.mojo`, `tools/attn-ref.py`,
`tools/draft-ref.py`, `tools/model-ref.py`).

## P9. A sum is not a comparison

Compare tensors element by element. Aggregates cancel: sign-mixed vectors of
2048 elements agree in sum while disagreeing everywhere.

Why: layer 0's post-SSM residual matched llama.cpp to 1.2% on the sum while
its individual components were 6%, 12% and 78% out. That sum is why the SSM
was not a suspect for three rounds. The RMS of an RMSNorm output is the
right diagnostic for scale; mean-abs is not, and neither is a sum for
direction.

## P10. A gate that never runs is not coverage, and a void is a failure

Every committed gate script is executed once, in the commit that adds it.
A gate that skips, voids or errors reports FAIL and exits non-zero; it never
averages over the arms that survived.

Why: `bench/force-ab-serve.sh` was committed at `ba9b832` and first run on
2026-09-12, where it returned 19 of 20 arms VOID and printed
`prompts 1/20  min 100.0%  mean 100.0%`, which reads as a pass. The void was
a real bug in the shipped serve path: the cancel probe read a line from fd 0
mid-generation and discarded it when it was not a cancel, so every queued
request after the first was eaten.

## P11. Prove the gate can fail before trusting that it passed

A gate is characterised by feeding it a known-bad input and confirming it
fails loudly. Until that is on record, a pass means the gate ran, not that
the artifact is correct.

Why: W3 gate 1 passes at ~1e-4 on all four layers while decode produced
garbage, because gate 1 feeds the MoE block the **oracle's** fixed-seed
input and never the engine's own activation. It was also confirmed
falsifiable (`gate 1 expert id mismatch at 0` on another layer's fixtures),
which is what makes its pass meaningful. Separately,
`bench/moe-gate2-force.sh` hardcoded `BARO_PREFILL=0` two commits before the
m=1 replay it was meant to validate was preregistered, so it would have
produced identical numbers with or without the commit under test.

## P12. Reading the code produces hypotheses; only the oracle produces findings

Before proposing a cause from a code reading, state it as a hypothesis with
the measurement that would kill it. Where a reference implementation can run
the same input, diff against it rather than reasoning about intent.

Why: three consecutive confident readings of the MoE SSM path were wrong
(positional offsets, SPLITK partials, head-layout transpose). The actual
defects, found in one pass by comparing against llama.cpp's own per-tensor
dump, were that `ssm_alpha`/`ssm_beta` are f32 in the MoE pack and were read
through a q8_0 kernel, and that both projections launched `grid_dim=1`
against a kernel writing 8 rows per block so 24 of 32 heads were never
computed. The tell was in the data, not the source: 24 beta values sitting
at exactly 0.5000, which is sigmoid(0) on memory nobody wrote.

`tools/llama-oracle.py` parses `llama-eval-callback` into per-tensor sums and
sample values (1398 tensors for RegesCore), and is the reference arm for any
model llama.cpp can run.

## P13. Verify a lane's claim by rebuilding from its committed tree

A dispatched lane's report is a claim. Rebuild from the commit it names, in
your own worktree, and re-run the gate before believing any number. Check
that the receipt postdates the commit it describes.

Why: in the MoE lane the receipt predated its own commit five times
(W0 gate 00:04 vs commit 00:09; "corrected binary" 02:23 vs 02:25; 9e5cc60
log 02:53:58 vs commit 02:56:48; gate2-force-m1prefill dir 03:44:24 vs
commit 03:44:39). Every number happened to be accurate; none of the receipts
covered the tree being claimed. The same lane's falsifier verdict was
reported from 2 of 20 prompts, and the one long prompt it tested was the
shortest of the nine that mattered.

## P14. A gate's bar must be a number some known-good configuration has hit

Before a pass threshold is frozen, measure what the mature path on the same
hardware actually reaches against the same reference on a quant-matched arm.
A bar nobody has ever cleared is not a bar, it is an open-ended hunt, and it
will be read as "the new thing is broken" for as long as it stands.

Why: W3 gate 2 required 20 prompts at 64/64 teacher-forced agreement against
llama.cpp. On 2026-09-12 the MoE path reached 53.20 and was treated as
failing for a whole session. Measured the same day, our DENSE path -- the one
that ships, verified, on Qwythos -- reaches **51.90** against llama on a
quant-matched arm (our q4 pack against Qwythos-9B Q4_0-pure, engine
318cc092, 20 prompts, min 38, max 60). The MoE path was already ABOVE the
ceiling of the known-good path while being called broken.

The cause is not a defect: our activations pass through bf16 (`curb_d`)
before every matmul while llama keeps f32. bf16 carries 8 mantissa bits,
2^-8 = 0.39% relative, and the measured per-layer floor against llama is
0.44%. Over 40 layers, with MoE routing making a discrete top-8-of-256
choice, that is enough to flip a share of tokens.

This repo already knew the principle and did not apply it: CLAUDE.md records
that llama.cpp's own f16-KV config fails its own f32 reference at 5 of 7
lengths, which is exactly why identity gates are teacher-forced agreement
"never greedy 64-token equality past ~256 ids". Gate 2 asked for the thing
the rule says not to ask for.

Corollary, learned the same day: before calling any per-layer divergence a
defect, check whether the model CANCELS there. Layer 31's residual read 10x
worse than its neighbours and was localised as the break. It is not: that
layer subtracts two vectors of RMS 0.85 into a result of RMS 0.095, and
llama does the same, 8.9x, with our magnitudes matching to three decimals.
The relative error is amplified cancellation of a constant input error.

## P15. Preflight on the CPU; a GPU job never discovers a harness error

Before any `gpu-wait run`, every binary the gate uses is built from the current tree and the gate runs
once on its smallest input (one item, smallest model) outside the queue where possible. A GPU job that
fails on a build error, a bad flag, an unread config field or an over-length request is a process defect,
logged as one.

Why: on 2026-09-16, 34 of 135 GPU jobs failed (25%). The traced causes were all CPU-discoverable:
`run-tests.sh` red for 5 hours after a merge nobody rebuilt the tests for, `max_tokens` + prompt over TMAX,
a bake's `baro.run.env` ignored so the engine refused to start, an eval tool that no longer built.

## P16. Failures are loud and stop the run

Every gate script runs with `set -euo pipefail`. A failing step prints `FAIL <step>: <reason>` with the
log path and exits non-zero. A sweep may continue past one failed item only to finish the others; it then
ends with `FAIL k/N: <names>` and a non-zero exit. A skip is never exit 0. No `|| true` on a step whose
output a later step reads. A script is never edited while a job is executing it (bash reads it as it runs).
Every job has a wall budget; exceeding it kills the job and says so.

Why: `bench/quality-sweep.sh` printed `FAILED ... continuing to next model` and exited 0, so the background
notification read as success while RegesCore's task arm had failed. Granite's row went void because
`quality-run.sh` was edited mid-sweep. A `torch.compile` arm held the GPU 32 minutes with no step done.

## P17. Cache the reference arm, never ours

A reference arm (llama.cpp, a frozen champion binary) whose output is a pure function of its inputs is
computed once and stored under a key of every input that can change it: model file sha256, reference
binary commit or sha, sampler parameters, input ids. The receipt prints `refcache hit|miss key=...`.
Any key change is a miss. The arm under test is never cached.

Why: the 2026-09-16 quality sweep regenerated llama.cpp's T=0 answers for every model, about half of each
row's GPU time, although nothing on the llama.cpp side had changed.

## P18. Iterate on the quick gate, claim on the full gate

Each gate has a named quick subset (fixed items, under 2 GPU minutes) for iteration and the full set for
the merge candidate. Commit messages, reports and the board cite only full-gate numbers, and say which
commit the full gate ran on.

## P19. Spend GPU only on the question

The GPU is held only while GPU work runs: CPU servers, scoring and report writing happen outside the job,
one resident engine per model serves all of a session's gates, and the next known job is queued before
the current result is read. Eval generation stops at the answer (stop strings identical on both arms),
shared prompt prefixes use `ckpt` hints, and our arm runs spec on at T=0 once identity with spec off is
on record for that eval.

## P20. The shorter implementation wins

If a smaller change (fewer lines, fewer files, fewer processes) meets the same gate, it replaces the
larger one, and the replaced code is deleted in the same commit, not left beside it. Duplicate harness
copies are not kept "for reference"; the reference is git history.

Why: `bench/latent_harness.mojo` was a hand copy of `serve/harness.mojo` that fell 23 fields behind and
broke the E13 tools; `bench/quality-task-eval.py` survived as a second eval path after
`quality-task-ids.py` replaced it.
