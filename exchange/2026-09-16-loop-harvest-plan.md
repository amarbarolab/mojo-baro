# Self-optimising loop and model-harvest plan (2026-09-16)

Research deliverable. File citations against the tree at `91bb16b` (main).
Hypotheses are labelled (P12).


## A. Loop state

### What exists

| component | path | purpose |
|---|---|---|
| proposer | `tools/loop-propose.py` | serve the baked GGUF on llama.cpp, feed it kernel sources + profile shares + identity framings, collect one unified diff per branch |
| gate ladder | `tools/loop-gate.sh` | scope -> compile -> identity@64 (+ second fixture) -> perf (+2%, spread <5%, wall-clock plausibility) -> ISA vs champion build |
| orchestrator | `tools/loop-run.sh` | one GPU job: server up -> propose -> server down -> gguf-closure -> gate |
| embedder | `tools/loop-embed-winner.sh` | re-embed committed sources into a new GGUF with `baro.kernel.parent` |
| diff normaliser | `tools/diff-normalise.py` | rewrite hunk headers from the body before `patch` sees them |
| ISA checker | `tools/isa-spills.py` | compare spill/scratch counts per kernel family against a baseline |
| protocol | `bench/loop-protocol.md` | receipts for iterations 001 through 007, gate amendments, worth-it rule |
| second fixture | `bench/loop-prompt2.txt` | 27 prompt tokens, unpublished |

### What iterations 001 through 007 produced

7 iterations, 19 candidates with a diff, **0 survivors, 0% gain**. Deepest reaches:

| iter | deepest | what happened |
|---|---|---|
| 001 | parse | 9B proposer; 3 hallucinated bindings, 0 parseable diffs |
| 002 | apply | hunk counts wrong; all 4 diffs rejected by `patch` |
| 003 | identity | 27B proposer; cand-1 compiled + ran, first token wrong |
| 004 | compile | 27B self-describing; cand-1 commented out profiling guard |
| 005 | scope | new identities; cand-2 deleted `perf_counter_ns()` |
| 006 | perf | skeleton prompt; cand-2 appended `# noqa` (no-op), +1.7% against stale denominator |
| 007 | parse | megakernel region; 0/4 produced a diff fence at all |

Across 002 through 005: **8 of 14 diffs echoed the RULES example back** (invented symbols, literal `...`). Replacing the example with a format-only skeleton (iter 006) killed that class and the proposer started editing real code, but what it edited was profiling guards and no-ops, not kernels. The megakernel (iter 007) was beyond the 27B entirely: every branch traced the code in prose and stopped.

### The 27B proposer swap (iter 003 through 005)

A 3x larger model (Qwen3.8-27B vs Qwythos-9B) cleared the output format but did not do the engineering. Two harness defects had to be fixed before anything was measurable: `reasoning_content` discarded silently (`67b456d`), and llama.cpp sampler defaults degenerated the output (`presence_penalty` + documented Qwen3 params fixed it). The proposer is not the bottleneck; the prompt context is insufficient for the task.

### R2 no-op lottery

Iteration 006 cand-2 was a `# noqa` no-op that reached +1.7% because the acceptance denominator was the `gguf-closure` median from a different session (65.91, with a 62.2 outlier) while the in-gate build of the same sources measured 67.40. **P-D was applied the same day**: the gate now builds the iteration's pristine sources in the same run and compares against that median. The gate code at `tools/loop-gate.sh:46-55` implements this: champion = median of 3 in-gate runs. **R2 is closed by construction.** A no-op on the in-gate pair read +0.6%, well under the +2% bar.

### Runability on today's tree

**Not runnable without edits.** Four breakages, all mechanical:

1. **Reference fixture path.** `loop-gate.sh:13` hardcodes `ref=.work/engine-pack/ref-tokens-64.txt`. That directory does not exist; the current fixture is at `.work/engine-pack-q4/ref-tokens-64.txt` (the default pack became `q4` around 2026-09-04). Fix: one `sed` or a `BARO_PACK`-aware variable.

2. **No baked GGUF.** The last BARO-embedded 27B is `Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-3242573.gguf` from iteration 007 (2026-09-08). The tree is hundreds of commits ahead. A new bake is required: `tools/gguf-embed.py` + `tools/embed-files.py` on the current sources, then verify with `tools/gguf-closure.sh`.

3. **Embed list has grown.** `tools/embed-files.py` now emits **27 files** (was ~10 at iteration 007). New files include `kernels/dattn.mojo`, `kernels/mega_moe.mojo`, `kernels/sample.mojo`, `kernels/moe.mojo`, `serve/expert_tier.mojo`, `serve/grammar_rt.mojo`, `serve/model*.mojo`, `serve/moe_pack.mojo`, `serve/serve_proto.mojo`, and the 5-file `latentos/` package. The GGUF's KV overhead grows (hypothesis: still small relative to the 16.8 GB model, ~300 KB extrapolating from the 134 KB at 14 files).

4. **Proposer region maps are stale.** `loop-propose.py`'s `KFILES` dict knows only `attn.mojo`, `ssm.mojo`, `elementwise.mojo`, `matmul_skinny.mojo`. The engine now has `dattn.mojo`, `mega_moe.mojo`, `sample.mojo`, `moe.mojo`, `matmul_prefill.mojo`, `matmul_prefill_lds.mojo`, `matmul_ternary.mojo`. The megakernel path (`MEGA_DEF`) should still work if `mega.mojo` still has the named defs, but none of the MoE or sampling code is reachable by the proposer.

**Estimated fix: S (under 60 LOC).** Pack path variable, KFILES update, a fresh bake. No gate logic changes needed; the acceptance rule, scope stage, ISA check, wall-clock plausibility and P-D denominator are all sound.


## B. Is the loop still the right shape?

### What the gate should measure now

| dimension | gate today | gap |
|---|---|---|
| decode tok/s | median of 3 in-gate runs, +2% bar, spread <5%, wall-clock plausibility | sound; consider adding a warm-up run (iter 007 showed 5.5% spread from a cold GPU after model unload) |
| identity | 64/64 greedy on the published fixture + optional second fixture, on every run | sound |
| ISA regression | no family with more scratch/spills than champion build | sound |
| quality rows | not in the gate | **gap**: a candidate that speeds up decode by cutting precision would pass today. Adding a task score or perplexity check is out of scope for a kernel loop (hundreds of tokens needed), but a 20-prompt forced-agreement check against the champion's own output (not llama.cpp's) is cheap (the T=0 identity gate already does this implicitly at 64 tokens, which is enough for decode-only changes) |
| MoE path | not in the gate | **gap**: the gate runs only the dense profile; a MoE candidate would need a second `gguf-closure` + identity run against the MoE pack |

### What the search space should be

**Today's pool, by GPU time share (dense q4 megakernel, `BARO_PROFILE=5` last token):**

| region | share | file(s) | notes |
|---|---|---|---|
| ffn | 51% | `mega.mojo::ffn_phases` | q4 dot loop schedule is the known pool (BASELINE) |
| ssm | 32% | `mega.mojo::ssm_phases` | delta linear in m, <5% share; conv1d + l2norm |
| attn | 9% | `mega.mojo::attn_phases` + `dattn.mojo` | dattn landed 3824e20, ~15% of floor |
| head | 9% | `mega.mojo::mega_body` tail | argmax/sample path |

For the MoE profile (launch path, 727 launches/token): the launched kernels in `moe.mojo`, the expert GEMVs in `matmul_skinny.mojo`, and the router in `ssm.mojo`.

Widening the search space to **launch partition counts**, **VGPR pressure** and **comptime schedule parameters** (not kernel rewrites) is the highest-EV move. The proposer can reason about numeric values; it cannot write correct Mojo diffs.

### Proven dead ends (from memory and the board)

| idea | evidence | citation |
|---|---|---|
| launch fusion / megakernel for decode | floor 2.57 us, ceiling 6.9% | `memory/launch-fusion-closed.md` |
| JSPLIT (j-split SSM delta kernel) | flat at kernel level, 3.3% slower e2e via FFN carry-over; closed twice | `memory/jsplit-closed-twice.md`, `exchange/2026-09-15-jsplit-review.md` |
| int8 WMMA for speed | same 512 ops/clk/CU as bf16 on gfx1100 (ratio 1.007) | `memory/int8-wmma-rate-gfx1100.md` |
| megakernel warm-touch | +0.42% below kill line, reverted | `memory/megakernel-warm-touch-closed.md` |
| persistent MoE kernel | 1.027x, below +5% line; default stays launch path | `memory/moe-persistent-kernel-r6.md` |
| register-prefetch double buffering (fp16 WMMA) | <=1.01x at BLK_K 16 and 32 | `docs/BASELINE.md` Round 1 |
| int8 MMQ for decode | 0.91x of bf16-lds | `memory/chat-lane-m0-complete.md` |
| `rocdl.waves_per_eu` metadata | six spellings rejected by Mojo | `docs/BASELINE.md` |
| `@no_inline` pins | dead end | `memory/rmsnorm-fold-closed.md` |
| M3-dot (int8-dot FFN at k=2) | wins only m>=5/prefill | `memory/m3dot-closed-at-k2.md` |

### What is the loop's right shape, given all of this?

The loop in its current form (model proposes kernel diffs, gate checks) has **zero successes in 7 iterations**. The failure mode is not the gate (which is sound) but the proposer's ability to write correct, non-trivial Mojo diffs against a 27-file closure. Specifically:

- The proposer cannot hold 18k+ chars of megakernel code and produce a correct hunk.
- Even a 27B model that carries its own sources echoes the prompt format, attacks the timing code, or writes no-ops. None has proposed a mechanism.
- The worth-it rule fired at iteration 5. Widening the region (the prescribed remedy) made it worse (iteration 7: zero diffs).

**Hypothesis (P12): the loop's shape should change from "model writes diffs" to "model proposes numeric parameters, script measures".** The gate stays; the proposer output format changes from unified diff to a parameter vector. The search space is comptime constants (already swept by `bench/sweep.py` for the GEMM) and runtime env knobs. This is where the model's reasoning is useful (explain WHY a parameter value should help) without requiring it to write correct code.


## C. What else can we harvest from models

### C1. Parameter search via model proposals (recommended)

A model proposes values for a bounded set of numeric parameters (chunk sizes, VGPR cap, launch grid dimensions, loop unroll factors, MoE expert batch size) with a rationale. A mechanical harness builds the binary with those constants, gates on identity, and measures tok/s. The model sees the receipt and proposes again.

- **Expected value**: the fp16 WMMA sweep (`bench/sweep.py`, 198 points) found the winning config at 1.38x over the hand-tuned one. A model-directed search over 5 to 10 parameters in the megakernel decode path could find schedule improvements the register allocator expresses differently. The q4 dot-loop schedule is the known pool (BASELINE: lost dual-issue pairs 102 -> 74, gained `s_delay_alu` 184 -> 454 after the rmsnorm fold).
- **Cost**: 0 GPU minutes for proposals (API tokens only, ~5k tokens per proposal round at the 27B). 2 to 5 GPU minutes per measurement (one identity check + 3 timed runs). A 10-round search: ~30 GPU minutes, ~50k API tokens.
- **Gate**: the existing loop gate (identity + perf + ISA), unchanged.
- **Past measurement**: `bench/sweep.py` already proved model-free search works on a smaller parameter space. Model-directed search adds the rationale (which direction to try) and should converge faster.
- **Effort**: S (<60 LOC). A `tools/loop-param-propose.py` that formats the current parameter values + profile shares + ISA fingerprint as a prompt, parses a JSON response `{"param": value, ...}`, and writes a `comptime` override file the build picks up.

### C2. Test and oracle generation

A model generates differential test cases: prompt token sequences that exercise edge cases (long repetitions for the sampler, adversarial routing for MoE, near-capacity sequences for the expert tier). It also writes the reference oracle in Python (using `sample_ref.mojo` patterns or `tools/model-ref.py`).

- **Expected value**: moderate. The existing test suite (129 PASS in `run-tests.sh`) was written by hand and misses real-vocab edge cases (the 2^-24 Gumbel floor was found by a device test at real vocab, not by tests). Coverage of `kernels/sample.mojo` at real vocab widths is the highest-value gap.
- **Cost**: API tokens only (no GPU for generation; GPU for running tests, but existing `run-tests.sh` overhead). ~20k tokens per 5 test cases. Reviewing generated tests is the real cost.
- **Gate**: generated tests must find the same defects that hand-written tests find on a deliberately broken binary (P11 check).
- **Past measurement**: the `test_sample_device.mojo` gate 2 finding (two top_p/top_k configs disagree at real vocab) was found by a hand-written fixture, not generated. No measurement rules this out, but no measurement supports it either.
- **Effort**: M (~100 LOC for a prompt template + runner + P11 gate).

### C3. Reading other engines' kernels at scale

Feed llama.cpp, vLLM, or MAX kernel source to a strong model with a specific question and get a structured answer. The pattern from `2026-09-12-max-colibri-harvest.md` worked once (found the repeat_interleave lead, which turned out to be wrong for our layout, and the gated_delta conventions, which were right) and the `ggml-harvest` tool (`~/iTools/llm/ggml-harvest/`) catalogs which kernels fire.

- **Expected value**: the highest-value target is decode attention, where llama.cpp's `flash_attn_ext_vec` runs at 13 to 31% of bandwidth peak (`2026-09-11-llama-cpp-kernel-harvest.md` B3). Understanding WHY is cheaper than reimplementing blind. A model reading llama.cpp's `ggml-cuda/flash_attn_ext_vec.cuh` and our `kernels/dattn.mojo` side by side, asked "what memory access pattern does theirs use that ours does not", is a well-shaped question.
- **Cost**: API tokens only (~50k tokens per kernel read at opus). No GPU.
- **Gate**: the reading produces a hypothesis (P12); the hypothesis is tested by a preregistered kernel experiment with a known-good bar (P14).
- **Past measurement**: the colibri harvest produced one false positive (the `h % NH_K` lead) caught in 2 minutes by measurement, and one useful convention check. Hit rate is low but cost is near zero.
- **Effort**: XS (<20 LOC for the prompt; the reading itself is analyst work).

### C4. Distillation or self-training data

Generating training data for the draft head (A4 in NEXT-PLAN) or for a task-specific LoRA. The draft head already has a trainer (`~/AMDHQ`, `ca10db1`), and acceptance is the lever (42% vs DeepSeek-V3's 85-90%).

- **Expected value**: bounded by the draft head training ceiling. E13-mini showed NO SIGNAL (`AMDHQ 030bd64`); a full E13 run is the maintainer's decision (up to 3 GPU-h per k). Model-generated training data adds nothing until the training pipeline produces signal.
- **Cost**: high (GPU hours for training, API tokens for data generation).
- **Gate**: acceptance on the 20-prompt set at k=2 above the untrained 42% (NEXT-PLAN A4 frozen bar: >=50%).
- **Past measurement**: E13-mini NO SIGNAL.
- **Effort**: L+ (training infrastructure, data pipeline).
- **Verdict**: blocked on E13 signal; do not start until E13 full produces it.

### C5. Nightly skill improvement from transcripts (SkillOpt-Sleep)

The dry run (`~/Brain/agents/SkillOpt/2026-09-16-sleep-dry-harvest.md`, mock backend) mined 39 tasks from 72h of transcripts. Findings:
- 25 of 39 are dispatched briefs (agent sessions, not recurring user tasks).
- 8 of the 14 human asks are the same task: "status / where are we / what next".
- All 39 have `reference_kind: none` (no checkable gate under the mock backend).
- The one recurring human task worth a skill is project status/next action (closest existing: `checkpoints`, `whiteboard-command-center`).

- **Expected value**: low for mojo-baro specifically. The recurring task is already partially covered. A real backend (`llm_mine`) would need to filter out dispatched-brief sessions, which are agent work judged by their own gates.
- **Cost**: API tokens for `llm_mine` (the real backend); no GPU.
- **Gate**: a generated skill handles the "status" task faster than the current manual flow.
- **Past measurement**: the dry run's mock backend produced no actionable rubrics.
- **Effort**: S to hook up `llm_mine` with session filtering; M to act on findings.
- **Verdict**: worth one real-backend run with session filtering, but not a priority for this repo.

### C6. Model-directed launch partition search (MoE specific)

The MoE path runs 727 launches per token. R6.0/R6.0b folded 390 of the original 1117 launches. The remaining launches are individually small kernels where the question is not "can this kernel be faster" but "can two adjacent kernels share a launch". A model reading the launch trace (from `bench/moe-launch-count.sh`) and the kernel bodies can propose merge candidates with a rationale. The gate is identity + launch count + 20-prompt tok/s.

- **Expected value**: moderate. R6.0b's 1.043x from 130 fewer launches gives ~0.33% per 100 launches eliminated. The 727 remaining launches include ~500 that are elementwise chains or SSM small kernels. Ceiling: ~2x fewer launches = ~2.4% at the measured rate, which is below the +5% kill line used for R6.2.
- **Cost**: API tokens for proposals; 5 GPU minutes per gate.
- **Gate**: existing MoE gate (identity 20/20, launch count from rocprofv3, 20-prompt tok/s).
- **Past measurement**: R6.0 to R6.0b measured ~0.33% per 100 launches. The persistent MoE kernel (ALL launches collapsed to 8) gained only 2.7%, confirming diminishing returns.
- **Effort**: S for the prompt; M for any kernel merge that lands.
- **Verdict**: below the kill line by the measured rate. Not recommended.

### Summary table

| harvest | EV | GPU min | API tokens | gate | blocked by | effort | recommend |
|---|---|---|---|---|---|---|---|
| C1 param search | high | 30 per 10 rounds | 50k | existing loop | nothing | S | **yes** |
| C2 test/oracle gen | moderate | existing | 20k per 5 | P11 | nothing | M | second priority |
| C3 kernel reading | moderate | 0 | 50k per read | P12 hypothesis | nothing | XS | **yes** |
| C4 distillation | unknown | GPU-hours | varies | E13 acceptance | E13 signal | L+ | blocked |
| C5 SkillOpt-Sleep | low | 0 | varies | skill coverage | nothing | S | one run |
| C6 MoE launch merge | low | 5 per gate | 20k | MoE gate | nothing | S-M | no (below kill line) |


## D. Recommended next experiment: parameter search on the q4 megakernel FFN

### Hypothesis

The q4 dot-loop schedule in `mega.mojo::ffn_phases` left performance on the table when the rmsnorm fold (`9e6feaa`) re-rolled it: dual-issue pairs dropped 102 -> 74, `s_delay_alu` rose 184 -> 454 (BASELINE). A model-directed search over comptime schedule parameters can find a configuration where the register allocator produces a better ISA, recovering some or all of the lost pairs. The ISA fingerprint (`isa-loops` dual/delay columns) is the leading indicator; the 20-prompt median is the gate.

### Arms

- **Champion (control):** main at `91bb16b`, q4 megakernel, 20-prompt median. Expected: ~136 tok/s (BASELINE: 136.37 at `3824e20`; verify with an in-gate run).
- **Candidate (treatment):** same binary with comptime overrides from the model's proposal. Built with a `-D PARAM=VALUE` mechanism or a generated `overrides.mojo` file included at the top of `mega.mojo`'s FFN section.

### Parameters in scope

Read from the ISA and the source before proposing:
- `ROW_WAVES` (current: profile-dependent, ~4 for ffn)
- `SPLITK` / `SK_THREADS` (ffn gate/up/down GEMVs)
- `SBN2` (skinny matmul block N)
- `BLK_K` and unroll factors in the q4 dot loop
- `RELOAD` (chunked delta, currently True at m=1)
- Grid dimensions for the ffn elementwise kernels

Parameters NOT in scope: anything in the timing/profiling path, anything that changes output (the identity gate catches it), anything in the attention or SSM paths (different experiment).

### Gate

1. **ISA fingerprint:** `tools/isa-spills.py` + `isa-loops` dual/delay columns on the candidate binary. No family with more spills or scratch than the champion (existing rule). Record dual-issue pairs and delay-alu count for the ffn families.
2. **Identity:** 64/64 greedy on `ref-tokens-64.txt` + the second fixture, on every timed run (existing rule).
3. **Perf:** median of 3 in-gate `tok/s_gen` >= champion + 2%, spread < 5%, wall-clock plausibility >= 0.5x claimed saving (existing rule, P-D denominator). Bar: the champion has hit 136.37; any candidate must hit 139.1.
4. **Preflight (P15):** build both binaries on CPU and run each once with `--quick 1` before `gpu-wait run`.

### Kill line

If 10 rounds of model-directed proposals produce no candidate reaching perf (stage 3): the parameter space is exhausted at the granularity the register allocator expresses, and the next move is a manual ISA-level investigation of the q4 dot loop, not more proposals.

### GPU budget

- Per round: 1 champion build (CPU, ~11 s) + 1 candidate build (CPU) + 1 identity run (~2 s) + 3 timed runs (~6 s) + 1 warm-up (~2 s). Total: ~10 s GPU per round.
- 10 rounds: ~100 s GPU = 1.7 GPU minutes. Add gguf-closure identity: 2 GPU minutes total.
- `gpu-wait run --timeout 300 --vram 22 -- tools/loop-run.sh ...` (the existing 5-minute budget is generous for this; 180 s would suffice).

### Preflight step (P15)

Before `gpu-wait run`:
```
bench/preflight.sh loop-gate   # builds champion + one candidate, identity on CPU fixture
```

### Which model proposes and why

**Opus 5 via the `claude` CLI**, not the local 27B. Rationale:
- The local 27B failed to produce a single mechanism across 7 iterations. Its failure mode is not format (the skeleton fixed that) but reasoning about ISA-level effects of parameter changes on register allocation and instruction scheduling. That requires the level of technical reasoning only frontier models demonstrate.
- API cost is small (~5k tokens per proposal, $0.15 per round at opus pricing, $1.50 for 10 rounds).
- The proposer does not need to be self-describing: it reads the ISA fingerprint and profile shares, not its own weights.
- The local 27B is still the right proposer if the experiment is repeated at scale (100+ rounds) where API cost matters. But 10 rounds is a feasibility probe.

### Preregistration

This section is written so it can be committed as `bench/param-search-protocol.md` before any run:

- Prediction: at least 1 of 10 rounds produces a candidate that changes the ISA fingerprint (dual-issue pairs or delay-alu count moves). Whether that translates to +2% is uncertain: the q4 dot loop's sensitivity to schedule parameters is the open question.
- Falsifier: 10 rounds, 0 ISA changes = the comptime parameters do not influence the register allocator's schedule at the granularity proposed. The experiment is killed.
- If 1+ candidates change the ISA but none reaches +2%: report the ISA deltas and the tok/s ratios. The finding is whether the parameter space connects to the schedule, not whether it wins.

---

Sources: `bench/loop-protocol.md` (iterations 001-007), `docs/BASELINE.md`, `docs/NEXT-PLAN.md`, `bench/PROTOCOL-RULES.md` (P1-P20), `docs/design/gpu-efficiency.md`, `tools/loop-{run,propose,gate,embed-winner}.sh`, `~/Brain/mojo/mojo-baro/2026-09-01-fp16-wmma-gap.md`, `~/Brain/mojo/mojo-baro/2026-09-05-proposer-swap-27b.md`, `~/Brain/mojo/mojo-baro/2026-09-01-loop-001-candidate-transcripts.md`, `~/.claude/projects/-home-user-Projects-mojo-mojo-baro/memory/loop-gate-stage4-unpassable.md`, `~/Brain/mojo/mojo-baro/2026-09-11-llama-cpp-kernel-harvest.md`, `~/Brain/mojo/mojo-baro/2026-09-12-max-colibri-harvest.md`, `~/iTools/llm/ggml-harvest/tool.toml`, `~/Models/library/INDEX.md`, `~/Brain/agents/SkillOpt/2026-09-16-sleep-dry-harvest.md`, `~/.claude/projects/-home-user-Projects-mojo-mojo-baro/memory/MEMORY.md`.
