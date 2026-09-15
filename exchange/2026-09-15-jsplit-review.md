# JSPLIT falsification review, with the re-run (2026-09-15)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-jsplit-review.md`, plus two
mid-turn extensions from the maintainer (optimizations allowed; then the GPU freed and
timed runs were allowed under the protocol's own rules). Repo `main` @
`a9c0106`. Nothing committed; every touched file is listed at the end.
Artifacts: `.work/jsplit-review/` (variant sources, four engine binaries, ISA
dumps, run logs, preregistration hashes in `prereg.sha256`).

## Verdict

1. **The 2026-09-04 falsification was not sound.** Arm D's 85% regression was
   the generalization's codegen, not the split. Verified by compile receipt
   (the reconstruction reproduces the recorded 603 / 2029 / 2009 spill counts
   exactly) and then by timing: with spills equalized, the split arms land
   within 2.3% of the control at kernel level.
2. **Item 2 is nevertheless a confirmed null, now for a measured reason.**
   At MR=1, the shape it was written for, neither JSPLIT=4 nor JSPLIT=2 beats
   the control (+1.5% / -0.6%, ranges overlap). JSPLIT=2 is 2.3% faster at
   MR=3 with disjoint ranges, a kernel-level receipt only.
3. **End-to-end both split arms are 3.3% slower** (one prompt, 5 rounds,
   disjoint ranges), and the loss sits mostly in the untouched FFN sub-block
   that follows the split kernel. Mechanism not identified; candidates and
   the missing instrument are named below. JSPLIT does not land.
4. The recorded explanation (redundant `kq` fill, spill pressure from the
   split) is withdrawn. The `kq` fill is forced by the split but costs ~2% of
   the kernel's bytes from L2; the spills were the index expression.

`bench/ssm-occupancy-protocol.md` carries the preregistration (frozen before
the first run, four amendments each timestamped before the run they govern,
hashes in `.work/jsplit-review/prereg.sha256`) and a Results 2026-09-15
section with the same numbers as below.

## Part 1: the confound, by compile receipt (verified)

All variants are the `main` kernel body (`kernels/ssm.mojo:182-232` before my
edit, byte-identical to `af30ac9`) with only the column-index expression
changed, compiled at the engine's layouts (SLOTS=9, N_SSM=24, MROWS=8).
`vgpr_count` is 192 in every row.

| variant | grid | column index `j` | MR | spills | scratch B |
|---|---|---|---|---|---|
| `k_orig` (= `main`, arm C) | (32) x 128 | `thread_idx.x` | 8 | **603** | 360 |
| `k_orig` | | | 1 | 81 | 308 |
| `k_gen` JSPLIT=1 (the broken control, reconstructed) | (32,1) x 128 | `block_idx.y * JW + thread_idx.x` | 8 | **2029** | 1180 |
| `k_gen` JSPLIT=4 (arm D as run on 09-04, reconstructed) | (32,4) x 32 | same | 8 | **2009** | 1260 |
| `k_gen` JSPLIT=4 | | | 1 | 211 | 488 |
| `k_gen32` JSPLIT=1 / 4 | 2-D | 32-bit `UInt32(block_idx.y) * JW + tid` | 8 | 1995 / 1978 | 1208 / 1336 |
| `k_gen1d` JSPLIT=4 | (128) x 32 | `(bx % 4) * JW + tid`, 32-bit | 8 / 1 | 612 / 78 | 436 / 296 |
| `k_genmask` JSPLIT=4 | (32,4) x 32 | `(block_idx.y & 3) * JW + tid` | 8 / 1 | **599** / 78 | 388 / 296 |

Three exact matches with the recorded numbers (603, 2029, 2009) say the
reconstruction of the never-committed 09-04 kernel is faithful at the receipt
level. Within one index style JSPLIT=1 and JSPLIT=4 sit within 1% of each
other; across styles the same JSPLIT=4 geometry moves 2009 to 599. The spill
count belongs to the index expression. The protocol's attribution to the
`kq` fill loop is falsified by the `k_gen` JSPLIT=1 row: one fill iteration,
byte-identical to the original fill, 2029 spills.

Instruction shape at MR=8: `k_gen` carries the same global traffic as
`k_orig` (1062 loads / 1032 stores) plus ~1500 extra spill instructions, 64-bit
scratch pairs the original never emits (1165 vs 0), and 4x the selects (2352
vs 572). Reading (reasoned, not proven at IR level): with `j = thread_idx.x`
the backend knows `j < 1024` and forms the 128 per-column addresses of
`SAll[rs, si, h, i, j]` as bounded 32-bit offsets; an unbounded
`block_idx.y` term turns them into per-element 64-bit values with sign-extend
and select, live across the `col` load. `k_gen32` shows width alone does not
help (a 32-bit `block_idx.y` is still unbounded); `k_genmask` and `k_gen1d`
show any provable small range restores the original codegen.

Per-MR receipts of the engine binaries (spills MR=8..1):

| arm | MR 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| C, J1 | 603 | 526 | 458 | 380 | 312 | 235 | 167 | 81 |
| D (JSPLIT=4 masked) | 599 | 530 | 461 | 405 | 336 | 267 | 181 | 78 |
| D2 (JSPLIT=2 masked) | 573 | 504 | 435 | 366 | 297 | 227 | 155 | 78 |

At MR=2..5 the JSPLIT=4 forms carry +32 spills over C (the 1-D form lands on
the same counts), so that residue is the 4-way split itself, not the index.
It made D fail my own 3% receipt gate at MR=3, which is why D2 was added
before any run (Amendment 2) and D kept as exploratory at MR=3.

## Part 2: the `kq` fill and the geometry (verified from the kernel body)

Each thread owns one column `j` and reduces over the row index `i`:
`sk = sum_i col[i] * eg * kq[1, i]` and `o += s * kq[0, i]` run over all 128
`i`, so every thread needs the whole `k` and `q` vectors regardless of which
columns its block owns. The full-width fill is forced by a `j` split. Its
cost at JSPLIT=4 is 3 x 1 KB extra L2-resident reads per head per row
against 128 KB of state traffic, ~2.3% of the kernel's bytes. Not 85%.

JSPLIT does not change the wave count (32 x 4 waves = 128 x 1 wave = 128).
It changes spread: 4-wave workgroups land on one CU each, so JSPLIT=1 uses at
most 32 of 96 CUs. That is the whole upside, and it is bounded by whatever
memory latency the kernel hides badly on 32 CUs. The 128-wide `col` and the
192-VGPR spilling allocation are untouched by any `j` split; only a split
along `i` (partial `sk` combine, not bit-exact) can shrink them.

## Part 3: the re-run (measured)

Arms: C = untouched `main` (`fc976cd9`); D = JSPLIT=4 masked
(`501fd19b`); D2 = JSPLIT=2 masked (`614968b6`); J1 = original body under the
`comptime if JSPLIT == 1` branch with the tuple grid `(NH_V, 1)`
(`e7ab45bd`). Every launch through `gpu-wait run`, queue read empty before
each, GPU otherwise idle. Every one of the 64 runs: `GENERATED` sha
`c00468774758` (identical, and identical to the 2026-09-04 reference),
`mega fail word: 0`, `BARO_SPEC` / `spec k` / `BARO_MEGA` read back from the
log. The 0.0478 s reference did not reproduce (0.0415 today): 136 decode-path
commits since `af30ac9`, stated in the preregistration before running.

**Gate 2, `ssm-kernel: delta` (s), 5 rounds alternating C, D, D2:**

| cfg | arm | median | min..max | spread | vs C | ranges vs C |
|---|---|---|---|---|---|---|
| P: spec k=2, MR=3 windows | C | 0.041477 | 0.041158..0.041810 | 1.59% | 1.000 | |
| P | D | 0.041045 | 0.040674..0.041451 | 1.91% | 0.990 | overlap |
| P | D2 | 0.040508 | 0.040127..0.040774 | 1.61% | 0.977 | **disjoint** |
| S: no mega, no spec, MR=1 | C | 0.041784 | 0.041520..0.041924 | 0.97% | 1.000 | |
| S | D | 0.042405 | 0.041798..0.043472 | 4.01% | 1.015 | overlap |
| S | D2 | 0.041533 | 0.041223..0.042276 | 2.55% | 0.994 | overlap |

Prediction 1 (confound thesis, split within 10% of control once spills are
equal): **confirmed**, worst case +1.5%. Prediction 2 (item 2 as written,
per-kernel time drops at MR=1): **falsified for both widths**; item 2 is a
confirmed null at MR=1. D2 at MR=3: 2.3% faster, disjoint, one prompt.

**Prediction 3 (tok/s moves under 1%): falsified.** Non-profiled pass, same
prompt, 5 rounds:

| arm | tok/s_gen median | min..max | decode_s median |
|---|---|---|---|
| C | 118.90 | 118.12..118.98 | 0.5299 |
| D | 114.99 | 114.31..115.55 | 0.5479 |
| D2 | 115.00 | 114.81..115.71 | 0.5478 |
| C (J1 pass) | 118.98 | 118.88..119.15 | |
| J1 | 118.33 | 118.14..119.03 | |

Both split arms 3.3% slower with disjoint ranges, the same amount for both
widths. J1 sits on C (overlapping), so the tuple launch form is not the cost.
`BARO_PROFILE=1` sub-block attribution, median of 3 (s):

| sub-block | C | D | D2 |
|---|---|---|---|
| attn | 0.0481 | 0.0495 | 0.0495 |
| ssm | 0.1947 | 0.1993 | 0.1994 |
| ffn | 0.2218 | 0.2361 | 0.2330 |
| head | 0.0351 | 0.0350 | 0.0350 |

The FFN sub-block, whose kernels the change never touches, carries 11 to 14
of the 20 ms; ssm +4.5 ms; attn +1.4 ms; head flat. The split kernel is not
slower when measured alone, so the loss is a carry-over from its execution
into what follows. Candidates: clock or power state after a kernel that
lights all 96 CUs instead of 32; cache state left by the different write
order of the 2 MB state. Separating them needs a clock probe, which this
protocol does not have and which `docs/BASELINE.md` forbids inside a timing
loop. **Unresolved, stated as such.** Whatever it is, it makes JSPLIT a
3.3% end-to-end loss on this prompt at k=2, and nothing lands.

One-prompt caveat on everything in Part 3: these are receipts under the
protocol's own gates, not 20-prompt medians (P4). They settle the confound
and the direction; they do not put a number in `docs/BASELINE.md`.

## What was wrong on 2026-09-04, in one paragraph

The control was repaired to escape a codegen penalty, the treatment was left
paying it, and the treatment's receipt (2009 spills, the same count the broken
control had just been fixed away from) was read as a property of the split.
The first-pass timing recorded in the protocol ("~0.09 s for BOTH arms" with
the broken control) was the tell: same spills, same time, different geometry.
Rule already in `m.ledger/kernels.md` (verify a no-op with an ISA receipt)
needs its mirror: **when the control is repaired, re-derive the treatment
from the repaired control and read its receipt against the control's, not
against the broken control's.**

## Optimizations spotted (scope extension)

### O1. `bench/bench_launch_floor.mojo` did not compile on `main`. DONE, compile check

Five stale call sites, not one: every SSM kernel in the stage-(c) chain had
moved to the multi-row signature. Rewritten to MR=1 at engine per-launch
shapes (2-slot rank-5 state ring, 2-slot conv ring, `[1, CONV]` conv rows,
`[1, NH_V]` gates, `[1, NH_V, SSTATE]` output, `[1, H]` residual); unused
`cs_layout` and `cs_w` removed; stage (d) untouched. Check: builds with
`-I kernels -I serve`; `isa-receipt.py` on the binary lists all five SSM
kernels, delta at 192 VGPR / 81 spills = `k_orig` MR=1. Not run on the GPU;
its numbers are not claimed.

### O2. `tools/ci-checks.sh` never built `bench/`. DONE, the step runs

Compile sweep of all 31 `bench/*.mojo`: 22 built, 2 are libraries with no
`main` (`e13_projector.mojo`, `latent_harness.mojo`), 4 import the external
`grammar` package that lives in the sibling repo `mojo-baro-clean`
(`bench_latent_handoff`, `e12long_check`, `e13_engine_dump`, `e8_batch`), and
3 were stale: `bench_launch_floor` (O1), `bench_hidden_dtype` (two
`WindowCfg(...)` calls missing `dump4`, `dump_layer`; fixed, builds),
`latent_ingest_bench` (`Chain(ctx, cap)` needs `packdir`; `save(...)` needs
`pinned, boundary`; fixed with the engine's own values, builds).

Added a CI step that builds every `bench/*.mojo` with a `main`, skipping and
naming the `grammar` importers. Check: `tools/ci-checks.sh` ran end to end,
step reports `OK 25 bench sources build`, whole script 34 s. The same run
flagged `docs/KERNELS.md` stale because of the new `JSPLIT` parameter;
regenerated with `tools/kernel-census.mojo`, census now `93 kernels, 46 in
registry, 0 orphans`.

### O3. `slots` is always compile-time `SLOTS` at its one call site. Candidate, mixed receipt, NOT applied

`serve/window.mojo:1062` passes `Int32(SLOTS)`; the kernel takes a runtime
`Int32` and computes `(rg + r) % sl` twice per row as a 64-bit signed modulo
(the `v_cvt_f32_u32` / `s_mul_hi_u32` prologue: 19 conversions, 378 scalar
multiplies at MR=8). A comptime `SLOTS` (`k_origcs` in the variant file):

| | MR | instructions | `s_mul*` | spills |
|---|---|---|---|---|
| `k_orig` / `k_origcs` | 8 | 13795 / 12806 | 378 / 90 | 603 / **881** |
| `k_orig` / `k_origcs` | 1 | 2067 / 1712 | 84 / 13 | 81 / 79 |

Fewer instructions at both MR, but spills rise 603 to 881 at MR=8. Time
effect UNVERIFIED; the scalar work overlaps VALU work and may cost nothing.
Check that would settle it: Gate 2 above per MR in {1, 3, 8}. Not applied: it
changes the kernel signature and eight dispatch rows, and the MR=8 receipt is
a plausible loss.

### O4. Split along `i` instead of `j`. Idea only, UNVERIFIED

The spills exist because each thread holds a 128-wide `col`. A 4-way split
along the reduction axis inside the block (512 threads per head, 4 per
column, 32-wide `col`, `sk` combined across the 4 with a shuffle before `d`)
would cut the footprint to ~48 VGPRs and remove spills, at the cost of a
changed summation order (not bit-exact; `model-ref.py` decode is the gate).
A different kernel, not a JSPLIT variant; nothing here measures it.

## Working tree state and every file touched (uncommitted)

- `kernels/ssm.mojo`: `amar_ssm_delta_step` gained `JSPLIT: Int = 1`;
  `comptime if JSPLIT == 1` holds the original body verbatim, `else` the
  masked split path with the full-width fill. At JSPLIT=1 the engine receipt
  is identical to `main` at every MR (603..81) and J1 ran the prompt at
  118.33 vs 118.98 tok/s with identical tokens. `./run-tests.sh` (through `gpu-wait run`) exits 0 with every PASS line
  present on this tree. `kernels/test_ssm_block.mojo` builds but did NOT run:
  it needs the `.work/gguf/*.bin` fixtures (blk.0 tensors plus
  `tools/ssm-ref.py` outputs) which are not on disk; that parity test is the
  one check the JSPLIT=2 / 4 paths still lack beyond the 64-run token
  identity, and the JSPLIT=1 path lacks beyond the byte-identical ISA receipt.
- `serve/registry.mojo`: `comptime SSM_JSPLIT = 1`; the eight dispatch rows
  pass it and launch `grid_dim=(NH_V, SSM_JSPLIT), block_dim=SSTATE //
  SSM_JSPLIT`. Set to 4 and 2 for the D and D2 builds, back to 1 now.
- `docs/KERNELS.md`: one row regenerated (the new parameter).
- `bench/ssm-occupancy-protocol.md`: preregistration, amendments 2..5,
  Results 2026-09-15, Verdict 2026-09-15 (append-only; the 09-04 text stands
  as history).
- `bench/bench_launch_floor.mojo` (O1), `bench/bench_hidden_dtype.mojo`,
  `bench/latent_ingest_bench.mojo` (O2 fixes; all three build).
- `tools/ci-checks.sh` (O2 step).
- `exchange/2026-09-15-jsplit-review.md` (this file).
- `.work/jsplit-review/` (gitignored): `ssm_jsplit.mojo`, `drv*.mojo`,
  binaries `engine-C`, `engine-D`, `engine-D2`, `engine-J1`, ISA dumps,
  `runs/`, `runs-tps/`, `runs-j1/`, `runs-p1/`, `prereg.sha256`, logs.

Whether to keep the `JSPLIT` machinery at 1 or revert `kernels/ssm.mojo` and
`serve/registry.mojo` to `main` is the maintainer's call; both are one `git checkout`
away and the receipts say they are the same kernel.
