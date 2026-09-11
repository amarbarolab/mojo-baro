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

## M0c — runtime T end to end (frozen 2026-09-08, before its build)

**Where the ceiling lives after M0b.** `serve/registry.mojo:36` `TMAX = 1088`
sizes `TPAGES`/`KVPOOL`/`KVPOOL1` and `toks_layout`; `serve/engine.mojo`
allocates `toks_d` (286), the four KV pools (326-329), the host token buffer
(446), rejects `len(prompt) + n > TMAX` in both the serve loop (389) and the
one-shot path (441), and prints `TMAX:` (435). The kernels carry no T. The
prefill loop already walks the prompt in chunks of `pf_chunk <= CP = 1024`
(`serve/window.mojo:572`), so no kernel change is expected.

**Change.**
- `TMAX` becomes a runtime capacity `tcap`: env `BARO_TMAX` (default 1088,
  the identity reference) read once at start; `TPAGES = ceildiv(tcap,
  KVPAGE)` and the two pool sizes computed from it; token buffers sized
  `tcap`. The ready JSON and the `TMAX:` line print the runtime value (P1
  receipt). No per-request capacity yet: one process, one capacity.
- Overflow: both checks return llama.cpp's error shape instead of a bare
  string. Serve mode: `{"error":{"code":400,"message":"the request exceeds
  the available context size, try increasing it","type":
  "exceed_context_size_error","n_prompt_tokens":N,"n_ctx":C}}`; one-shot
  mode prints the same object and exits 2. `baro-serve` maps it to HTTP 400
  verbatim.
- Long prompt files: `bench/prefill-prompts/p8192.tokens`, `p32768.tokens`,
  `p100000.tokens`, produced by `tools/baro-tokenize encode` over a fixed
  text (this repo's `docs/*.md` concatenated, then `~/llama.cpp/docs/*.md`,
  truncated to the exact count) and committed with their sha256 in this
  section's Result.

**Not in this step.** Per-request capacity, page table, checkpoints (M1);
the GQA-grouped decode kernel (M0d, only if P-D5 demands it); the int8
prefill GEMM (lane int8).

**Predictions.**
- P-D1 At `BARO_TMAX=1088` the binary is bit-exact against `9d3b228`'s
  build: `GENERATED` byte-identical on all 20 `bench/mtp-prompts/`, q4 and
  q8 one-shots 64/64, mtp 20/20. Only allocation sizes and error strings
  change. Falsifier: any difference, which would mean a size leaked into
  the address arithmetic after all.
- P-D2 20-prompt median within ±2 % of `9d3b228` (same stint), fingerprint
  in class (12727 ± 40, dual ≥ 105, 80/80/60 ± 2, spill 0). The megakernel
  source does not change, so the fingerprint is expected identical.
- P-D3 TTFT table, `prefill_s` from the run at `BARO_TMAX=102400`,
  `BARO_PREFILL_C=1024`: predicted from the measured p0512 rate
  (511 rows in 0.42 s, of which ~0.05 s is fixed) as ~0.75 ms/row, linear
  in chunks: 8192 → ~6 s, 32768 → ~24 s, 100000 → ~75 s. llama.cpp
  `--pure Q4_0` on the same token files (`tools/llama-ref-run.sh`,
  `/props` read back) predicted ~2.4× faster (int8 MMQ prefill). Recorded
  as the gap the int8 lane owes; not gated.
- P-D4 Memory: at `BARO_TMAX=102400`, f32 KV = 2 × 102400 × 8 × 4 × 256 ×
  4 B = 6.7 GB; with the 6.2 GB pack the engine prints its ready line and
  runs the 100000-token prompt plus 64 generated tokens with `mega fail
  word: 0`. Falsifier: allocation failure or a residency fault, which
  would move K8/V4 forward from M5.
- P-D5 Decode after a 32768-token prompt (`tok/s_gen` of the 64 tokens):
  predicted ≥ 100 tok/s_gen (KV read per token at T=32k is 2 × 32768 × 4 ×
  256 × 4 B × 8 layers = 2.1 GB, ~2.3 ms at 900 GB/s on top of the 7.4 ms
  token). At T=100000: predicted ~60 tok/s_gen (6.5 GB per token). Reading
  rule: below 80 at T=32k triggers the M0d preregistration (design §4
  decode kernel); above it M0d waits for M5.
- P-D6 `mega fail word: 0` on every run; the overflow request returns the
  JSON shape above in both modes (`tools/test_server.sh` gains one
  oversize case).

**Verification before timing (P1).** `TMAX:` and the ready JSON `tmax` field
read back from every run; `prompt tokens:` equals the file's word count;
`prefill chunk:` 1024; rebuild in the same stint; `arm.txt` first for the
A/B; llama.cpp `/props` saved beside its timings.

**Gate.** P-D1 exact; `tools/merge-gate.sh` ALL PASS at the default 1088;
P-D2 within band; P-D3/P-D4/P-D5 tables filled (P-D4 must hold; P-D3 and
P-D5 are recorded against their predictions, not gated); P-D6.

**Result.** PASS on the gate; P-D3 and P-D5 missed their predictions,
recorded 2026-09-08 (`.work/m0c/stint.txt`, exclusive GPU, ref =
`.work/engine-m0b` sha `321c5f25b3e172b3` built from 9d3b228, new =
`.work/engine-m0c`).

| receipt | M0b | M0c | source |
|---|---|---|---|
| fingerprint q4 `amar_mega_token` | 12746 / dual 115/80/80/60 / spill 0 | identical | `.work/isa/m0c/co64.s` |
| one-shot q4 at 1088 | 133.75, 64/64 | **bit-exact**, 133.75 | `.work/m0c/one-q4-*.log` |
| one-shot q8 at 1088 | 82.08, 64/64 | **bit-exact**, 83.63 | `.work/m0c/one-q8-*.log` |
| 20-prompt median (P4) | 133.26 (spread 3.8 %) | 133.26 (spread 1.9 %), ratio 1.000, identity 20/20 | `.work/m0c/ab/results.txt` |
| overflow, one-shot (8192 prompt at TMAX 1088) | raise | exit 2, `exceed_context_size_error` JSON | `.work/m0c/overflow.log` |
| overflow, server (`max_tokens` 1e6) | 400 plain | 400 `exceed_context_size_error`, n_prompt_tokens 43, n_ctx 1088 | merge-gate `PASS overflow` |
| merge-gate | ALL PASS | ALL PASS | `.work/merge-gate.txt` 11:35 |

Long context, `BARO_TMAX=102400`, f32 KV, prefill chunk 1024, 64 generated
tokens, prompt files sha256 (first 16) p8192 `3ff81ea490f46aa6`, p32768
`a922d0b1d2f3b6f8`, p100000 `112722f7857c9962`:

| T | prefill_s ours | predicted | llama.cpp Q4_0-pure prompt_ms | ratio | tok/s_gen ours | predicted | llama.cpp | mega fail |
|---|---|---|---|---|---|---|---|---|
| 8192 | 8.33 | ~6 | 2.59 | 3.2x | 89.4 | - | 109.2 | 0 |
| 32768 | 50.46 | ~24 | 12.84 | 3.9x | 45.7 | >= 100, floor 80 | 100.2 | 0 |
| 100000 | 298.3 | ~75 | 63.67 | 4.7x | 19.2 | ~60 | 78.8 | 0 |

Verdict against the frozen predictions:
- P-D1 held: bit-exact at 1088 on both packs, 20/20.
- P-D2 held: fingerprint identical, ratio 1.000.
- P-D3 missed: prefill is superlinear (8.3 -> 50 -> 298 s, i.e. T^1.5-2),
  not the linear ~0.75 ms/row the prediction assumed. The prediction
  modelled the GEMMs only; `amar_attn_prefill` re-reads the whole KV per
  1024-row chunk, so its cost grows with T per chunk and the sum is
  quadratic. Against llama.cpp the gap grows 3.2x -> 3.9x -> 4.7x from 8k to 100k
  (2.4x expected from the int8 MMQ GEMM alone; the growth is the
  attention sweep). llama.cpp's own decode falls 109 -> 100 -> 79 over the
  same range, i.e. its attention costs ~0.04 ms per 1k, a tenth of ours.
- P-D4 held: 6.7 GB f32 KV + pack, 100k prompt + 64 tokens, fail word 0.
- P-D5 **falsified**: 45.7 at 32k, below the 80 floor. Per-token decode
  cost is linear in T at 0.44 ms per 1k tokens (7.5 ms + 3.7 at 8k, + 14.4
  at 32k, + 44.6 at 100k), six times the KV-byte floor (67 MB per 1k
  tokens = 0.075 ms at 900 GB/s). Bytes do not explain it; the attention
  phase inside the megakernel runs on a fraction of the CUs while the rest
  hold at the grid barrier. This triggers M0d: the GQA-grouped, split-K
  decode attention kernel of design §4, as a separate launch (or phase
  with every block participating), preregistered next.
- P-D6 held in both modes; `tools/test_server.sh` gained the case.

Observation, not promoted: decode after an 8k prompt is 89 vs llama.cpp's
109 on the same card; at T <= 1088 ours leads 133 vs ~110. The crossover is
the M0d target.

---

## M0d — split-K decode attention inside the megakernel (frozen 2026-09-08, before its build)

**Mechanism, read from source.** `attn_phases` (`kernels/mega.mojo`) runs
the decode attention as `if bid < M * NQH:` — at m = 1, 16 of the 96
resident blocks, one q head each, while 80 blocks wait at the grid barrier.
Each active block walks all T positions in 256-position chunks
(`attn_head_body`), four `barrier()`s per chunk, one position per thread,
and threads 256..511 of the 512-thread block idle. M0c measured the cost
of that at 0.44 ms per 1k tokens of context, 6x the KV-byte floor; per
active block that is ~4.6 GB/s, latency-bound by construction.

**Change.** Split-K over page spans in the same phase, no new launch:
- Work item = (q head h, span s), s in 0..NSPLIT-1, NSPLIT = 6 at m = 1
  (96 / 16; `MEGA_G // (M * NQH)` in general, at least 1). Span s covers
  pages `[s * P / NSPLIT, (s + 1) * P / NSPLIT)` of the T positions
  (P = ceildiv(T, KVPAGE)), so each item is contiguous and page-aligned.
- Every block runs `attn_head_body` over its span with the same chunk loop
  and writes a partial `(m, l, o[HD])` to scratch instead of `O`. Scratch
  = the head split-K buffer `p_v` (`row_major[SPLITK, SM, VOCAB]` f32,
  idle during attention; 16 * 6 * 258 floats needed, millions available),
  no engine change.
- Grid barrier, then blocks `bid < M * NQH` merge their head's NSPLIT
  partials: `m = max m_s`, `O = sum_s exp(m_s - m) * o_s / sum_s exp(m_s
  - m) * l_s`, in span order.
- **T <= 1088 keeps NSPLIT = 1 and the direct `O` write**, i.e. today's
  code path, so the champion path stays bit-exact and the P4 A/B is
  untouched. `BARO_ATT_SPLIT=1` forces the split at every T for the
  identity receipts. The threshold is a runtime compare on T.

**Not in this step.** Using the idle upper 256 threads (a second chunk in
flight per block, the next lever, predicted a further ~1.6x); GQA-grouping
(one K/V read for the four q heads of a kv head; bytes are not the bound
yet); a separate launch (design §4's standalone kernel) — the megakernel
stays one launch per token.

**Predictions.**
- P-E1 Fingerprint: the dot loops are untouched, but `attn_phases` gains
  code, so the schedule lottery applies: q4 `amar_mega_token` within
  12746 ± 60 instructions, dual >= 105 / 80 ± 2 / 80 ± 2 / 60 ± 2, spill 0.
  One re-roll by spelling; second miss stops the step.
- P-E2 Identity: default (split off at T <= 1088) bit-exact vs `a86516b`'s
  binary on q4/q8 one-shots and 20/20 A/B. Forced split (`BARO_ATT_SPLIT=1`)
  at T <= 1088: q4 64/64 and q8 64/64 vs `tools/model-ref.py` reference
  tokens, 20-prompt identity 20/20 vs the unsplit binary (the merge changes
  summation order, so this is identity, not bit-exact). At 8192 and 32768
  the split binary's `GENERATED` equals M0c's unsplit `GENERATED`
  (`.work/m0c/long-*.log`); a mismatch there is arbitrated by
  `tools/model-ref.py` on the 8192 prompt (falsifier if model-ref sides
  with unsplit).
- P-E3 Long-context decode, same prompts as M0c: 8192 >= 115 tok/s_gen
  (from 89.4: 3.7 ms attention -> ~0.6 + 0.3 merge/barrier), 32768 >= 90
  (from 45.7: 14.4 -> ~2.4 + 0.3), 100000 >= 55 (from 19.2: 44.6 -> ~7.4
  + 0.4). Falsifier: 32768 < 70 means the per-block cost is not the
  bound (look at the barrier count / merge before touching occupancy).
- P-E4 20-prompt median within ±2 % of `a86516b` (path unchanged at
  T <= 1088; only the fingerprint could move it).
- P-E5 `mega fail word: 0` on every run; prefill_s unchanged (prefill
  attention is not touched).

**Verification before timing (P1).** `att split:` printed by the engine
(threshold and forced flag), `TMAX:`, `prompt tokens:`, `tok/s_gen`,
`mega fail word` from each run; fingerprint before any GPU run; rebuild in
the same stint; `arm.txt` first for the A/B.

**Gate.** P-E1 before GPU; P-E2 all four identity receipts; merge-gate ALL
PASS; P-E4 within band; P-E3 recorded against its predictions (32768 >= 70
is the hard floor, the three numbers are the claim).

**Result.** PASS on the gate, recorded 2026-09-08 (`.work/m0d/stint.txt`,
exclusive GPU; ref = `.work/engine-m0c` (a86516b), new = `.work/engine-m0d`).

P-E1 fingerprint, two builds: (1) head body inlined in both branches, 13511
instructions; (2) re-spelled to one `attn_head_span` call shared by both
paths, **12977** instructions, vgpr 249, spill 0, `isa-loops` dual
122 / 84 / 84 / 61. The count band (12746 ± 60) is missed by construction:
the partial write and the merge are +231 new instructions that no spelling
removes. The loop class, which is what the band was guarding, is faster in
every column (115/80/80/60 before). Recorded as a deviation and the GPU
stint was run; the question stands in `.work/briefs/status-chat.md`.

| receipt | M0c | M0d | source |
|---|---|---|---|
| test_mega_block (split forced, `ATT_SPLIT = 0`) | PASS | PASS, bit-identical to the launch path, m=1 q4/q8 and m=3 q8 | `.work/m0d/test_mega_block.log` |
| test_attn_block | PASS | PASS | `.work/m0d/test_attn_block.log` |
| one-shot q4 / q8 default (split off at T <= 1088) | 64/64 | **bit-exact** vs M0c, 137.6 / 81.3 | `.work/m0d/one-*-new.log` |
| one-shot q4 / q8 forced split | - | 64/64 vs model-ref tokens, 136.8 / 80.9 | `.work/m0d/one-*-split.log` |
| 20-prompt A/B default | 133.24 | 133.22, ratio 1.000, identity 20/20 | `.work/m0d/ab/results.txt` |
| 20-prompt A/B forced split vs unsplit (same binary) | 136.22 | 136.13, ratio 0.999, identity 20/20 | `.work/m0d/ab-split/results.txt` |
| merge-gate | ALL PASS | ALL PASS | `.work/merge-gate.txt` 11:55 |

Long context, `BARO_TMAX=102400`, default threshold 1088 (split on):

| T | tok/s_gen M0c | M0d | predicted | GENERATED vs M0c | prefill_s | mega fail |
|---|---|---|---|---|---|---|
| 8192 | 89.4 | **117.5** | >= 115 | identical | 8.30 (8.33) | 0 |
| 32768 | 45.7 | **85.0** | >= 90, floor 70 | identical | 50.24 (50.46) | 0 |
| 100000 | 19.2 | **48.7** | ~55 | differs from token 11 | 298.6 (298.3) | 0 |

Verdict against the frozen predictions:
- P-E1: count band missed (explained above), loop class held.
- P-E2 held on every frozen receipt: default path bit-exact; forced split
  64/64 on both packs and 20/20; 8192 and 32768 byte-identical to M0c.
  At 100000 (not in the prediction) the two summation orders diverge at
  token 11 of 64. Both continuations loop on the docs text (the prompt has
  no question); a low-margin distribution is where a merge-order
  difference flips an argmax. No 100k reference decode exists to
  arbitrate; recorded as an observation. Whether the model itself is sound
  at 100k is M5's RULER question (llama.cpp niah_single is 100 % at 32k,
  nothing measured beyond).
- P-E3: 8192 held (117.5 >= 115); 32768 above the floor, below the
  prediction (85.0 vs >= 90); 100000 below (48.7 vs ~55). The attention
  cost per token went 3.7 -> 1.0 ms at 8k, 14.4 -> 4.3 ms at 32k, 44.6 ->
  13.0 ms at 100k: a 3.3-3.7x cut against the 6x a perfect split over
  6 spans would give. The remainder is the extra grid barrier plus the
  merge (~0.3 ms) and the per-block chunk loop itself, which is still one
  position per thread with four barriers per 256 positions. The next
  lever is inside the block (both halves of the 512-thread block in
  flight, or two chunks per iteration), not more spans.
- P-E4 held: ratio 1.000.
- P-E5 held: fail word 0 everywhere, prefill_s unchanged.

Observation, not promoted: the M0d schedule's one-shot at T ~ 70 reads
136-137.6 on both the default and forced paths (M0c 133.7), and the
same-binary A/B medians read 136.1-136.2 while the cross-binary A/B read
133.2. The 20-prompt A/B against M0c is the P4 number (1.000); the 136s
are the lottery being kind on this build and are not claimed.

---

## M1a — prefix checkpoints for one sequence (frozen 2026-09-08, before its build)

**Where the state lives (read from source).** Per request the engine memsets
conv state, delta state, both KV pools and the counters
(`serve/engine.mojo` serve loop), then prefills the whole prompt. The SSM
state is `csall_layout = [SLOTS, N_SSM, 3, CONV]` f32 (1.77 MB per slot) and
`ssall_layout = [SLOTS, N_SSM, NH_V, SSTATE, SSTATE]` f32 (50.3 MB per
slot), `SLOTS = KMAX + 1 = 9`, ring index `st.ring` advancing by the rows
processed; the slot holding the state *after* position p is `ring` after
that step. The KV cache is one pool, position-addressed through `kv_off`,
so for a single sequence the KV for tokens `[0, p)` is already in place
when the next request shares that prefix; only what a replay overwrites
changes. The design's 26.25 MiB figure assumed 16 v-heads; the real
checkpoint is **52.1 MB** (conv + delta) per position.

**Change.**
- `serve/prefix.mojo` (new, host side): `Checkpoint {pos, hash, gen,
  conv_h, ssm_h}` in pinned host buffers; `hash = sha256(canonical bytes of
  tokens[0:pos])` computed on the host in Mojo (a 64-bit FNV-1a over the
  little-endian i32 ids is the first cut, upgraded to SHA-256 in M1b when
  cross-process reproducibility matters: this step is in-process only).
  A `Chain` of at most `BARO_CKPT` (default 8) checkpoints ordered by pos,
  each valid for the prefix it hashes; `lookup(tokens) -> best` returns the
  checkpoint with the largest `pos <= len(tokens) - 1` whose hash matches
  `tokens[0:pos]`.
- Boundaries where a checkpoint is taken: prompt end (after prefill, before
  the first generated token), and every 1024 tokens of prefill as the
  periodic safety net. Role boundaries and branch points are M1b (they
  need the template renderer and a second sequence).
- Restore: copy `conv_h`/`ssm_h` into the ring slot the replay will read
  from, set `st.pos = pos`, `st.ring` accordingly, skip the memsets of KV
  (the pool keeps `[0, pos)`), prefill only `tokens[pos:]`. If no match,
  today's path (memset, full prefill).
- Save: `enqueue_copy` device -> pinned host of the two slot regions,
  once per boundary; the copy is off the critical path of the response
  (issued after the boundary, synchronised before the next request).
- Receipts in the `done` line: `"cached": <pos restored, 0 if none>`,
  `"prefill_rows": <rows actually prefilled>`; the engine prints
  `checkpoints: cap 8, bytes 52.1 MB each` at start (P1 read-back).
- Server: passes `cached` through in `usage` as `baro.cached_tokens`.

**Not in this step.** Radix pages / multiple sequences, salt, retention
priorities, branch-point snapshots, cancel semantics, `prefix_churn`
(M1b). Cross-process reproducible hashes (M1b). Any kernel change.

**Predictions.**
- P-F1 Byte-exact restore, test `kernels/test_prefix.mojo` (new,
  `run-tests.sh`): for prompt A||B (A = 1088 tokens of `p8192.tokens`,
  B = the next 64), cold(A||B) vs restore(A)+replay(B) give
  `memcmp`-equal conv, delta, KV pages and next-token logits at the
  positions 0 / 1 / 1023 / 1024 / 1025 / 1087 / 1088 / 1151; a mutation at
  token 0, at pos-1, at pos+1 and at the last token of A misses the
  checkpoint (lookup returns none or an earlier one); a corrupted hash
  fails restoration. Falsifier: any byte difference, which means the
  ring/slot bookkeeping or the KV memset assumption is wrong.
- P-F2 Engine identity: q4/q8 one-shots and the 20-prompt A/B are
  bit-exact vs `48a48b1` (one-shot mode takes no checkpoints; the serve
  path with a cold cache is today's path).
- P-F3 Serve path, two requests with the 7,914-token DeerFlow system
  prompt prefix (tap replay from `.work/chat/tap.jsonl`): request 2
  reports `cached` >= 7,900 and `prefill_rows` < 200; its wall TTFT
  (prefill_s + restore) < 60 ms at the 8k prefix, from 8.3 s cold.
  Restore copy predicted ~2 ms (52 MB at PCIe 4 x16).
- P-F4 20-prompt A/B median within ±2 % of `48a48b1` (decode untouched;
  the fingerprint must be unchanged since no kernel changes).
- P-F5 Memory: 8 checkpoints = 417 MB pinned host, printed at start;
  `mega fail word: 0`.

**Verification before timing (P1).** `checkpoints:` line, `cached` and
`prefill_rows` in the `done` line of each request, `TMAX:`, fingerprint
unchanged (12977 / dual 122/84/84/61) read before the stint.

**Gate.** P-F1 all positions and mutations; merge-gate ALL PASS; P-F2
exact; P-F3 both numbers; P-F4 within band.

**Result (2026-09-10, `7b4b9d0` on `lane-chat`). The gate is NOT met: P-F3
misses both of its numbers. Everything else passes.**

P-F1 byte-exact restore: PASS on 2026-09-08 (`kernels/test_prefix.mojo`, all
positions and all four mutations, megakernel and window paths).

P-F2 / P-F4 engine identity and A/B (2026-09-10 01:12, queue empty,
`.work/m1a/ab/arm.txt`): `engA=./.work/engine engB=.work/m1a/engine`
shaA 644c23560ed499d4 shaB dfd578701f8b9f90, power cap 290000000 uW,
vddgfx -100mV, both arms `BARO_MEGA=1`. champ median 136.97 tok/s_gen
spread 0.5 %, m1a median 136.98 spread 0.6 %, **ratio 1.000**, identity
fails none. Checkpointing costs the decode path nothing.

Merge gate (01:12-01:14, `.work/merge-gate.txt`): **ALL PASS** — engine and
test builds, run-tests, ci-checks, test_prefill, one-shot q4 64/64 at 137.2
and q8 64/64 at 81.6, p0512, mtp 20/20, and the full server suite.

P-F5: engine prints `checkpoints: cap 8 , bytes 52.690944 MB each, period
1024` = 421 MB pinned; `mega fail word: 0` on every request.

**P-F3 tap replay** (`tools/tap-replay.py`, rows 0,1,2 x2 against baro-serve
on the m1a engine, `BARO_TMAX=16384`, `.work/m1a/tap-replay.log`):

| row | prompt_tokens | cached | prefill_rows | prefill_s | restore_s | wall_s |
|---|---|---|---|---|---|---|
| 0 (cold) | 5755 | 0 | 5754 | 5.229 | 0.003 | 5.279 |
| 1 | 5770 | 5120 | 649 | 0.651 | 0.002 | 0.695 |
| 2 | 5858 | 5120 | 737 | 0.722 | 0.002 | 0.766 |
| 0 (2nd) | 5755 | 5120 | 634 | 0.616 | 0.002 | 0.659 |
| 1 (2nd) | 5770 | 5120 | 649 | 0.652 | 0.002 | 0.696 |
| 2 (2nd) | 5858 | 5120 | 737 | 0.721 | 0.002 | 0.762 |

Frozen: `cached >= 7900`, `prefill_rows < 200`, wall TTFT `< 60 ms`.
Measured: cached 5120, rows 634-737, wall 659-766 ms. **Both numbers
missed.** Two separate reasons, neither of them the restore path:

1. The fixture is not the 7,914-token prompt the prediction named — these
   tap rows are 5,755-5,858 tokens. The prediction should have been read off
   the fixture before it was frozen.
2. The real miss: `cached` is 5120 = 5 x 1024, the periodic grid, not the
   5,754-token prompt-end checkpoint that request 0 wrote. Request 1's ids
   diverge from request 0's somewhere between 5120 and 5754, because the
   re-rendered template replaces request 0's trailing generation prompt with
   the assistant turn. So the end-of-prompt checkpoint is worth nothing for
   the next turn of the same conversation, and every restore falls back to
   the last 1024-boundary — up to 1023 replayed rows, whatever the prefix
   length. **Role-boundary checkpoints, deferred to M1b, are exactly the fix
   for this**, which the M1a design says in as many words; the frozen P-F3
   simply predicted M1b's behaviour and measured M1a.

What M1a does deliver, measured: multi-turn prefill **5.229 s -> 0.616-0.722 s,
7.2-8.5x**, with a restore copy of 2.1-3.2 ms against the ~2 ms predicted,
and no cost to decode. That is the honest claim. The 100x-class number needs
M1b.

Not run: nothing further; M1b preregistration is unwritten.

---

## C1 — control block: stop sequences, EOS-in-engine, cancel, FIFO+503 (frozen 2026-09-11, before its build)

**Where the gap is (read from source, 2026-09-11).** `serve/PROTOCOL.md` says it
in as many words: "there is no stop-token or cancel in the protocol." The
engine (`serve/engine.mojo`) always decodes exactly the requested `n`; a stop
token is only ever noticed by `serve/src/main.rs`'s `Acc::take`, client-side,
after the token has already been generated. There is no way to end a running
request early. The queue (`serve/src/engine.rs`, `mpsc::channel(QUEUE_CAP=64)`,
one worker) is already FIFO and already returns 503 past the cap
(`Engine::submit`'s `try_send` error mapped to `StatusCode::SERVICE_UNAVAILABLE`
in `main.rs::check_and_submit`) -- nothing to change there, only to keep gated.

**Change.**
- Wire: the request line gains `"stop":[[id,...],...]` (a list of token-id
  sequences, default `[]`); `serve/engine.mojo::parse_request` parses it.
  `serve/src/main.rs` builds it from two sources: the tokenizer's own
  `stop_ids` (each as a length-1 sequence, moving today's client-side EOS cut
  into the engine) and an OpenAI-style `stop` request field (string or array
  of strings, tokenized with `encode(s, false)`).
- Engine: after each `step_window` call where `wst.pos >= len(prompt)`, if any
  stop sequence is configured, sync + copy `toks_d` back and check whether the
  generated tail ends with it; on a match, break the decode loop early
  (`finish:"stop"` in the done line). Only requests that actually set `stop`
  pay this sync; the no-stop path is untouched.
- Cancel: a second line shape, `{"cancel":ID}`, written to the SAME stdin the
  worker uses (now behind an `Arc<tokio::sync::Mutex<ChildStdin>>` so a cancel
  can interleave with the in-flight request's line) whenever
  `Engine::cancel(id)` is called and `id` is the request currently running.
  The engine polls fd 0 with `poll(..., timeout=0)` once per `step_window`
  call (prefill chunk or decode window alike); a hit is read and, if it is a
  cancel for the running `req_id`, breaks the loop (`finish:"cancelled"`).
  Polling costs one syscall per window (~us) against a ~7-10 ms window, and
  never touches the GPU queue.
- `done` gains `"finish":"length"|"stop"|"cancelled"`; `n`/`tok_s` are computed
  from tokens actually generated, not the request's `n`, so an early stop or
  cancel reports its real count. `serve/src/protocol.rs::DoneStats` gains
  `finish: Option<String>` (absent = old engine = today's client-side
  `Acc::finish_reason` fallback, so the wire change is backward compatible).
- room for sampler settings (brief M3): the object is JSON, so a later field
  needs no reshaping here -- nothing to add speculatively now (§7).

**Not in this step.** Cancelling a request that is still queued, not yet
running (the worker only understands cancel for what it is actively decoding;
a queued job simply has not started). Any sampler field. Removing text from
the *displayed* string when a multi-token stop string's tail tokens are not
themselves flagged EOS by the tokenizer (`Acc` still only hides text at a
known stop id) -- token-level stop is exact; text-level trimming for
arbitrary stop strings is a finish-polish item, not gated here.

**Predictions (frozen before the build).**
- P-G1 Identity: the 20-prompt A/B (`bench/ab-prompts.sh`) at temperature 0,
  no `stop`/`cancel` set, is bit-exact vs `main` -- the added `poll()` per
  window is a host-side syscall with no GPU-visible effect, and the stop-check
  sync never runs when `stop` is empty (the default). Falsifier: any
  regression outside the standing +-2% band on the median, or any identity
  mismatch, since neither should be possible from this change.
- P-G2 Stop strings: a request with `stop` sequences ends generation at the
  first token whose tail matches one of them; `finish:"stop"`; `n` is less
  than the requested `max_tokens` when the stop occurs before the length
  limit, and the emitted tokens end in the stop sequence.
- P-G3 EOS moved server-side: a request with no explicit `stop` but whose
  reference continuation reaches the tokenizer's EOS before `max_tokens`
  now stops the engine loop at that token (todayâ€™s engine would keep
  computing to `n`); `finish:"stop"`, and the emitted tokens are a prefix of
  what the old (always-run-to-n) path would have emitted.
- P-G4 Cancel: `Engine::cancel(id)` on a request mid-decode returns `Ok(true)`
  and the engine's `done` line for that request arrives within one
  `step_window` call of the cancel line reaching stdin, `finish:"cancelled"`,
  `n` less than the requested length; a request submitted immediately after
  completes normally and matches `ref-tokens-64.txt` (the worker, queue and
  engine process are unharmed by a cancel).
- P-G5 FIFO + 503: unchanged from today -- two requests submitted back to
  back are served in submission order, and a request beyond `QUEUE_CAP` (64)
  gets 503. Recorded, not expected to move (no code in this path changes).
- Falsifier for the whole item: any identity break on the no-`stop`,
  no-cancel path (P-G1), or a cancel/stop that corrupts the next request.

**Verification before timing (P1).** `TMAX:`, `spec k:`, `prompt tokens:`
read back as usual; the done line's `finish` field read back on every request
that exercises stop or cancel; rebuild engine + `baro-serve` in the same
stint; `arm.txt` first for the A/B.

**Gate.** `tools/test_server.sh` ALL PASS, extended with a `stop`-string case
(P-G2) and a cancel case (P-G4, via `/v1/chat/completions` streaming: the
first SSE chunk's `id` is now `chatcmpl-<internal id>`, letting the test call
`POST /v1/cancel {"id":...}` mid-stream); `run-tests.sh` unaffected (no kernel
change); 20-prompt A/B (P-G1) within band.

**Result.** pending -- filled in after the gated run.
