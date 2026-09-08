# chat lane protocol — preregistrations for the agent-engine milestones

Design of record: `docs/design/agent-engine-2026-09.md`. Work order: design §10.
Binding rules: `bench/PROTOCOL-RULES.md` P1-P6. Every milestone below is
preregistered (prediction + gate + falsifier) BEFORE the build starts; the
Result section is filled in after the gated run and never edited to fit.

## Baseline (frozen 2026-09-08, `main` at 5ae2925, worktree `lane-chat`)

Read back from `tools/merge-gate.sh` on the binary built in the same command
(`.work/merge-gate.txt`, run 2026-09-08T06:14:41+02:00):

| receipt | value | source |
|---|---|---|
| engine build | exit 0 | merge-gate |
| run-tests.sh | exit 0, 63 kernels, 35 in registry, 0 orphans | merge-gate |
| ci-checks.sh | exit 0, all non-GPU checks passed | merge-gate |
| test_prefill | exit 0, PASS | merge-gate |
| one-shot q4 identity | PASS: 64 tokens match | `tools/check-tokens.sh` |
| one-shot q4 tok/s_gen | 134.52702393163582 | engine stdout |
| one-shot q8 identity | PASS: 64 tokens match | `tools/check-tokens.sh` |
| one-shot q8 tok/s_gen | 77.04891558274252 | engine stdout |
| TMAX / prefill chunk | 1088 / 1024 | engine stdout |
| prefill p0512 | prefill rows 511, prefill_s 0.424297253, tok/s_gen 118.86931574785959 | engine stdout |
| mega fail word | 0 | engine stdout |

P4 note: the single-prompt 134.5 above is an **instrument receipt**, not the
bar. The bar is the 20-prompt median from `bench/ab-prompts.sh` /
`bench/mtp-prompts.sh`; `main`'s recorded champion is 133.9 tok/s_gen (9e6feaa).

## Where the 1088-token ceiling actually lives (read from source, 2026-09-08)

1. `kernels/attn.mojo:12` `comptime MAX_T = 1088` sizes an LDS score array
   `stack_allocation[f32, row_major[MAX_T]]` in `amar_attn_decode` (line 155)
   and in the megakernel attention phase (`kernels/mega.mojo:807`). 4352 B of
   LDS per block, and a hard cap on T.
2. `serve/registry.mojo:36` `comptime TMAX = 1088` sizes the KV cache layout
   `row_major[NKVH, TMAX, HD]` (f32) and the token buffer `row_major[TMAX]`,
   and is baked into the three megakernel instantiations.
3. `serve/engine.mojo` rejects `len(prompt) + n > TMAX` and allocates
   `N_ATT * NKVH * TMAX * HD` f32 for K and for V (71 MB at 1088; 8.6 GB at
   128k if left f32 and contiguous).

The prefill attention kernel `amar_attn_prefill` is already online-softmax
(running `m_run`/`l_run`, PA_TK=16 tiles) and carries no MAX_T array: it is
T-agnostic today. Only the decode path holds the ceiling.

M0 is therefore split into three preregistered steps, each with its own gate,
each landing on `lane-chat` with its numbers in the commit.

---

## M0a — decode attention without the MAX_T score array

**Change.** Rewrite `attn_head_body` (shared by `amar_attn_decode` and the
megakernel attention phase, so both keep computing the same bits) from
"score all T into LDS, then reduce" to a chunked online-softmax loop over
T in chunks of C = HD = 256: per chunk, 256 threads each score one position
into a 256-entry LDS array, the chunk max and chunk sum are warp-reduced, the
running `(m, l, o)` are rescaled by `exp(m_run - m_new)`, and the output row
is accumulated against V for that chunk. LDS for scores drops 4352 B -> 1024 B
and T stops being a compile-time quantity in the kernel.

**Not in this step.** KV dtype (stays f32), paging, TMAX in the registry, the
engine's `prompt+n > TMAX` check. Those are M0b/M0c.

**Predictions (frozen before the build; commit of the frozen text is the
receipt).**
- P-A1 Identity holds: one-shot q4 and q8 both `PASS: 64 tokens match`, and
  `bench/mtp-prompts.sh` k=2 identity 20/20 PASS, 0 FAIL.
  Rationale: the change reorders the softmax rescale only; the K-dot and the
  V-accumulate keep their existing per-element order and 8-wide grouping.
  This is a non-bit-exact change, so `tools/model-ref.py` decode identity is
  the gate, per the standing rule.
- P-A2 Decode throughput at T <= 1088 is within +-2 % of baseline on the
  20-prompt median. Two effects roughly cancel: 3.3 KB less LDS per block
  (occupancy-neutral at the current 31 KB budget, so no win expected), and
  ~4 extra `s_barrier` per 256 positions (5 chunks at T ~ 1088, so ~20
  barriers vs ~6 today).
- P-A3 The falsifier is a regression worse than 2 % on the 20-prompt median,
  or any identity failure. Either sends the step back, not the gate.
- P-A4 `mega fail word: 0` on every run (the grid-barrier health word).

**Verification before timing (P1).** Every timed arm rebuilds the binary in
the same command as the run. Arm-defining values read back from the run's own
output: `TMAX:` line, `spec k:`, `prompt tokens:`, `tok/s_gen`,
`mega fail word`. The chunk constant is comptime, so the rebuild-in-command
is its receipt.

**Gate.** `tools/merge-gate.sh` all lines at least as green as the baseline
table above; 20-prompt A/B (`bench/ab-prompts.sh`, `AB_ENGINE_B`, `arm.txt`
read first) within P-A2.

**Result.** (filled after the gated run)

---

## M0b — KV cache: paged, and narrower than f32

**Change (to be preregistered in full before its build).** KV pages of 128
tokens with a per-slot page table, static per-page layout
`row_major[NKVH, 128, HD]` so no kernel needs a runtime-strided layout, page
count allocated at runtime. KV element type becomes a comptime switch over
{f32, f16, bf16} so the identity gate decides the default rather than the
design note deciding it: f16 carries 10 mantissa bits and is what llama.cpp
stores by default, bf16 carries 7, and the reference token stream this repo
gates against came from llama.cpp. The design (§2) names bf16; if f16 holds
identity and bf16 does not, f16 ships and the design gets the correction.

**Open until measured.** Whether the paged indirection costs decode
throughput (one extra dependent load per 128 positions).

---

## M0c — runtime T end to end

**Change (to be preregistered in full before its build).** `TMAX` moves from a
comptime constant to a runtime capacity (env + per-request), the token buffer
and page pool size with it, `amar_attn_prefill` chunking extends past CP, and
the engine's overflow path returns llama.cpp's `exceed_context_size` shape
instead of raising.

**Gate.** TTFT table at 8k / 32k / 100k vs llama.cpp on the same prompts;
identity at T <= 1088 unchanged.
