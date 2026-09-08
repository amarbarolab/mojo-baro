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

**Result.** PASS, 2026-09-08.

Arms: A = `.work/engine-base`, built by `mojo build` from `git archive HEAD`
(b64821b, pre-change) in the same stint; B = `.work/engine`, built from the
working tree in the same stint. `arm.txt`: `power_cap_uW=290000000`,
`vddgfx=-100mV`, both arms `BARO_PACK=.work/engine-pack-q4`.

| receipt | base | M0a | source |
|---|---|---|---|
| 20-prompt median tok/s_gen (P4) | 132.68 (spread 9.9 %) | 132.99 (spread 5.1 %) | `bench/ab-prompts.sh` `.work/ab-m0a` |
| 20-prompt identity B == A | - | 20/20 PASS | same |
| one-shot q4 identity | PASS 64/64 | PASS 64/64 | merge-gate |
| one-shot q4 tok/s_gen | 134.53 | 132.90 | merge-gate |
| one-shot q8 identity | PASS 64/64 | PASS 64/64 | merge-gate |
| one-shot q8 tok/s_gen | 77.05 | 82.51 | merge-gate |
| prefill p0512 tok/s_gen (T ~ 575) | 118.87 | 128.03 | merge-gate |
| mtp k=2 identity | 20/20 | 20/20 | merge-gate |
| test_server.sh | ALL PASS | ALL PASS | merge-gate |
| mega fail word | 0 | 0 | engine stdout |

ISA receipt (`tools/isa-receipt.py`, same two binaries), the mechanism:

| kernel | `group_segment_fixed_size` base -> M0a | vgpr | vgpr spill |
|---|---|---|---|
| `amar_attn_decode` | 5408 -> 2080 | 26 -> 39 | 0 -> 0 |
| `amar_mega_token` q4 | 31264 -> 27936 | 239 -> 239 | 0 -> 0 |
| `amar_mega_token` q8 | 6688 -> 3360 | 256 -> 256 | 78 -> 76 |
| `amar_mega_window` | 6816 -> 3488 | 192 -> 192 | 7 -> 7 |

-3328 B of LDS per block is exactly the 4352 - 1024 the score array gave back.

Verdict against the frozen predictions:
- P-A1 held: every identity check PASS (q4 64/64, q8 64/64, mtp 20/20,
  A/B 20/20, server suite ALL PASS).
- P-A2 held: +0.2 % on the 20-prompt median, inside the +-2 % band. The
  spread also halved (9.9 % -> 5.1 %).
- P-A3 not triggered.
- P-A4 held: `mega fail word: 0` on every run.

Unpredicted, and larger than the gated metric: **q8 decode +7.1 %** (77.05 ->
82.51) and **q4 at T ~ 575 +7.7 %** (118.87 -> 128.03). Both are LDS-occupancy
effects the prediction did not anticipate -- P-A2 assumed the 31 KB budget made
the saving occupancy-neutral, which is true for the q4 megakernel (27936 B is
still one block per allocation granule) but false for the q8 megakernel, where
6688 -> 3360 B changes how many blocks fit. Recorded as an observation, not
promoted: neither number is a 20-prompt median, and the q8 path is not this
lane's champion.

---

## M0b — KV cache: token-major, runtime capacity, narrower than f32

**Change.** The KV cache layout goes from `row_major[NKVH, TMAX, HD]` to
`row_major[TCAP, NKVH, HD]` (token-major) and its element type from f32 to a
comptime `KVT`.

Why token-major removes the ceiling: in `row_major[TCAP, NKVH, HD]` the
outermost extent has no stride of its own, so the address of (t, kvh, d) is
`t*NKVH*HD + kvh*HD + d` -- `TCAP` never enters the arithmetic. It is a
nominal extent, and the real capacity is whatever the runtime allocation
holds. In the old order `TMAX` was the *middle* extent and therefore the
stride between kv heads, which is why it had to be a compile-time constant.
The per-layer stride in the megakernel (`kc + att_i * ATT32`) becomes a
runtime argument for the same reason.

Token-major is also the layout paging wants: a 128-token page is one
contiguous `128*NKVH*HD` run, so M1 turns `t*NKVH*HD` into
`page_table[t >> 7]*PAGESZ + (t & 127)*NKVH*HD` and nothing else moves. The
page table itself is NOT built here -- with one sequence and no prefix sharing
it would be an identity map, i.e. dead weight until M1 gives it a job.

Coalescing is preserved: the V accumulation still has thread `tid` reading
`V[t, kvh, tid]` across a wave, 64 consecutive elements for one t.

**Element type.** `KVT` is a comptime switched per build (the rebuild-in-the-
same-command is its P1 receipt, and the engine prints `kv dtype:` from it).
Arms: f32, f16, bf16. Design §2 names bf16; f16 carries 10 mantissa bits
against bf16's 7 and is what llama.cpp stores by default, which is where this
repo's reference token stream comes from. The identity gate picks, not the
design note.

**Not in this step.** The page table, `TMAX` as a runtime request parameter,
the `exceed_context_size` error shape, chunked prefill past CP. Those are M0c.
`TMAX = 1088` stays the engine's capacity, so nothing user-visible changes:
this step is the re-layout alone, gated on identity.

**Predictions (frozen before the build).**
- P-B1 The f32 arm is **bit-exact** against the M0a binary: `GENERATED`
  byte-identical on all 20 `bench/mtp-prompts/`, q4 and q8 identity 64/64.
  Only addresses change; every value, cast and summation order is preserved.
  This is the load-bearing prediction -- it is what makes the re-layout
  separable from the dtype question.
- P-B2 The f16 arm holds identity (q4 64/64, q8 64/64, mtp 20/20). The bf16
  arm does **not**: at least one identity check fails. Frozen deliberately as
  the asymmetric prediction -- if bf16 also holds, the design's choice stands
  and bf16 ships.
- P-B3 No arm separates from another on the 20-prompt median by more than
  2 %. At T ~ 100 the KV read is 2*T*NKVH*HD*4 B per layer over 8 layers =
  6.5 MB per token against a 6.2 GB q4 pack, i.e. under 0.2 % of the bytes
  moved, so halving KV width cannot show up here. The KV-width win is a
  long-context effect and is claimed only when M0c can measure it.
- P-B4 `mega fail word: 0` on every run.
- Falsifier: the f32 arm not bit-exact against M0a. That is a re-layout bug,
  and the step goes back rather than the gate being relaxed.

**Verification before timing (P1).** Each arm is a rebuild in the same command
as its run; `kv dtype:`, `TMAX:`, `spec k:`, `prompt tokens:`, `tok/s_gen`
and `mega fail word` all read back from the run's own output.

**Gate.** `tools/merge-gate.sh` ALL PASS on the chosen default; 20-prompt A/B
against the M0a binary within P-B3; P-B1 exact.

**Result.** (filled after the gated run)

---

## M0c — runtime T end to end

**Change (to be preregistered in full before its build).** `TMAX` moves from a
comptime constant to a runtime capacity (env + per-request), the token buffer
and page pool size with it, `amar_attn_prefill` chunking extends past CP, and
the engine's overflow path returns llama.cpp's `exceed_context_size` shape
instead of raising.

**Gate.** TTFT table at 8k / 32k / 100k vs llama.cpp on the same prompts;
identity at T <= 1088 unchanged.
