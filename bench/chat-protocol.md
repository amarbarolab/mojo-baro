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

**Result.** GATE NOT MET, recorded 2026-09-08. The build is complete and
identity-clean but is not committed; the working tree carries it and
`.work/briefs/status-chat.md` has the handover.

Arms: A = `.work/engine-m0a`, built from `4027b6e` in the same stint;
B = the working tree, `KVT = float32`, layer-in-page layout.

| receipt | M0a | M0b tree | source |
|---|---|---|---|
| q4 one-shot identity | PASS 64/64 | PASS 64/64 | `tools/check-tokens.sh` |
| tok/s_gen at T ~ 70, 3 runs | 133.26 / 133.59 / 133.54 | 133.26 / 133.59 / 133.54 | engine stdout |
| tok/s_gen at T ~ 575 (`p0512`), 2 runs | 129.00 / 128.85 | **116.74 / 116.80** | engine stdout |
| `prefill_s` at T ~ 575 | 0.4175 / 0.5479 | 0.4211 / 0.4255 | engine stdout |
| `mega fail word` | 0 | 0 | engine stdout |

- P-B1 held on tokens: identity PASS everywhere measured, and an earlier
  20-prompt A/B of an intermediate M0b build was 20/20 identical to M0a.
  Bit-exactness at the buffer level was not separately checked.
- P-B3 **failed**: -9.5 % at T ~ 575, well outside the 2 % band. The
  prediction reasoned only about KV *bytes* (correctly negligible at
  T <= 1088) and never about the instruction budget the layout change spends,
  which is where the loss is.
- P-B2 not run. q8 identity, the merge gate and the dtype arms not run.

**Mechanism, measured.** The q4 megakernel is on an instruction-footprint
cliff. `tools/isa-receipt.py --hist` on `mega_amar_mega_token` (q4 code
object), four builds with identical semantics:

| decode-attention V-loop form | instructions | tok/s_gen at T ~ 70 |
|---|---|---|
| M0a, 8-wide unrolled | 12727 | 133.2 |
| unrolled over the chunk's 2 pages | 12905 | 127.4 |
| scalar, longer address expression | 12671 | 133.5 |
| scalar, algebraically simplified address | 12742 | 127.3 |

The last two compute the same address and differ by 71 instructions and 5 %.
Keeping the megakernel under the cliff forced the scalar V loop, and the
scalar V loop is what costs 9.5 % once T is large enough for the V
accumulation to matter. Both cannot hold inside one megakernel, so the fix is
the separate GQA-grouped decode kernel design §4 already specifies (split-K
over page spans, (m, l, O) partials, one merge workgroup per head) rather than
another attempt inside `attn_phases`.

**Falsified along the way** (both recorded so they are not retried):
- token-major `[t][kvh][HD]` costs 3.9 % at T ~ 70 — it loses kv-head
  locality, putting consecutive tokens of one head 4 KB apart instead of 1 KB.
- page-major with a **runtime** per-layer stride costs 5 % and takes
  `sgpr_spill_count` from 69 to 74: the stride argument stays live across the
  whole 32-layer loop. Layer-inside-the-page makes every stride comptime and
  is the only variant that keeps capacity out of the address arithmetic.
- **Cache-set aliasing is not the problem.** Layer-in-page puts the four kv
  heads exactly 128 KB apart, so 8 KB of padding per head block was added to
  break the power-of-two stride: 126.6 tok/s before, 126.6 after. `KVPAD` is
  dead weight and should go back to 0 with a re-measurement.

---

## M0b attempt 2 — same layout, M0a's V-loop form, KVPAD = 0

**Diagnosis receipts (CPU only, taken before this text, 2026-09-08).**
- T ~ 575 loss: the scalar V loop compiles to `global_load_b32` →
  `s_waitcnt vmcnt(0)` → `v_fmac_f32` per position (one load in flight);
  M0a's issues 8 loads under `s_clause` before its FMAs. Memory-latency
  serialisation, proportional to T. Attempt 1's diagnosis stands.
- "Instruction cliff": falsified. The "simplified address" build was
  reproduced CPU-side (12742 instructions, their number) and `isa-loops`
  shows its q4 *dot loops* rescheduled (VOPD 80/80/60 → 75/75/57, +26..33
  instructions each, more `s_delay_alu`) by a one-line edit in the attention
  V loop. The T ~ 70 losses were the known schedule lottery, fingerprintable
  at build time; the megakernel may unroll the V loop.

**Change.** `KVPAD = 0`. `attn_head_body`'s V accumulation returns to M0a's
8-wide form (8 loads into an `InlineArray`, then 8 in-order `o += sc*v`),
group base `vb + (tt >> KVPSH) * PGSTR + (tt & (KVPAGE-1)) * HD`, loads at
`+ j*HD` (8 | KVPAGE, no page crossing inside a group), scalar tail. Nothing
else in the attempt-1 tree changes.

**Predictions (frozen before the build).**
- P-C1 Build fingerprint, read BEFORE any GPU run: q4 `amar_mega_token`
  12727 ± 40 instructions, vgpr spill 0, `isa-loops` fused dot loop
  `dual` ≥ 105, loops 2-4 `dual` within ±2 of 80 / 80 / 60. A miss is a lost
  lottery ticket, not a layout verdict: one re-roll by spelling only, a second
  miss stops the step (brief: ask, file not terminal).
- P-C2 Identity: q4 64/64, q8 64/64, mtp k=2 20/20, 20-prompt A/B 20/20
  identical to `.work/engine-m0a`. Same FMA form and order as M0a.
- P-C3 p0512 (T ~ 575) tok/s_gen within ±2 % of M0a's 129.0 in the same
  stint. Falsifier: < 126.4 means the serialisation diagnosis was wrong.
- P-C4 20-prompt median within ±2 % of M0a (P-B3 restated).
- P-C5 `mega fail word: 0` on every run.
- Receipt run, not a prediction: `.work/engine-diag` one-shot at T ~ 70 in
  the same stint, expected ~127 (ties the losing fingerprint to its number).

**Verification before timing (P1).** Rebuild in the same stint; `kv dtype:`,
`TMAX:`, `spec k:`, `prompt tokens:`, `tok/s_gen`, `mega fail word` read from
each run's output; `arm.txt` first for the A/B.

**Gate.** P-C1 before GPU; then `tools/merge-gate.sh` ALL PASS; 20-prompt A/B
vs `engine-m0a` within P-C4; P-C3 within band. Then the dtype arms under
P-B2 as frozen, each arm fingerprinted before its run.

**Result.** PASS on the f32 arm, recorded 2026-09-08. The dtype arms both
fail identity; f32 ships (P-B2 as frozen).

Arms: A = `.work/engine-m0a` (4027b6e); B = `.work/engine` = `.work/engine-c2`
(sha `ba6eebfb1b53911c`, working tree, `KVT = float32`, `KVPAD = 0`, M0a V-loop
form). `arm.txt`: `power_cap_uW=290000000`, `vddgfx=-100mV`, both arms
`BARO_PACK=.work/engine-pack-q4`.

| receipt | M0a | M0b attempt 2 | source |
|---|---|---|---|
| P-C1 fingerprint (q4 `amar_mega_token`) | 12727 instr, dual 117/80/80/60 | 12746 instr, dual 115/80/80/60, spill 0 | `.work/isa/c2.txt`, status 07:20 |
| 20-prompt median tok/s_gen (P4) | 133.59 (spread 0.8 %) | 133.41 (spread 0.8 %), ratio 0.999 | `.work/ab-c2y/results.txt` |
| 20-prompt identity B == A | - | 20/20 PASS | same |
| one-shot q4 identity / tok/s_gen | PASS 64/64 | PASS 64/64 / 133.92 | `.work/merge-oneshot-q4.log` |
| one-shot q8 identity / tok/s_gen | PASS 64/64 | PASS 64/64 / 83.53 | `.work/merge-oneshot-q8.log` |
| prefill p0512 (T ~ 575) tok/s_gen | 129.00 / 128.85 | 128.67 (merge-gate), 128.48 / 128.61 (one-shots) | `.work/merge-pf512.log` |
| mtp k=2 identity | 20/20 | 20/20 | `.work/merge-mtp/results.txt` |
| test_server.sh | ALL PASS | ALL PASS | `.work/merge-server-test/SUMMARY.txt` |
| test_prefill / test_attn_block / test_mega_block | PASS | PASS / PASS / PASS (0 mismatches, q4 and q8, m=1 and m=3) | `.work/c2-stint4.out`, `.work/c2-run-test_mega_block.log` |
| mega fail word | 0 | 0 | engine stdout |

A first A/B run (`.work/ab-c2x`) was VOID: spread 32.5 % from one 124 outlier
under a co-running llama-server; the rerun above was exclusive.

Verdict against the frozen predictions:
- P-C1 held (one ticket, no re-roll).
- P-C2 held: every identity check PASS.
- P-C3 held: 128.5-128.7 vs 129.0, inside the 2 % band; the serialisation
  diagnosis stands.
- P-C4 held: ratio 0.999.
- P-C5 held.

Dtype arms (P-B2, 20-prompt A/B vs M0a, same stint, `.work/ab-c2f16`,
`.work/ab-c2bf16`): **f16 identity 17/20** (fails p12-rust, p13-haiku,
p15-bash), **bf16 identity 17/20** (fails p07-json, p12-rust, p13-haiku).
The one-shot 64-token checks passed for both, which is why the status file
briefly said "bf16 PASS everywhere"; the 20-prompt set is the gate and it
says otherwise. P-B2's f16 half is falsified (f16 does not hold either), its
bf16 half held. Neither narrow format ships; `KVT = float32` is the default
and the narrow KV question moves to M5 with the RULER gate, where the
long-context effect it is for can be measured.

---

## M0c — runtime T end to end

**Change (to be preregistered in full before its build).** `TMAX` moves from a
comptime constant to a runtime capacity (env + per-request), the token buffer
and page pool size with it, `amar_attn_prefill` chunking extends past CP, and
the engine's overflow path returns llama.cpp's `exceed_context_size` shape
instead of raising.

**Gate.** TTFT table at 8k / 32k / 100k vs llama.cpp on the same prompts;
identity at T <= 1088 unchanged.
