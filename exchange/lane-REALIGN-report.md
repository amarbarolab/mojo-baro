# Lane REALIGN — build report

Branch `lane-REALIGN`, worktree `~/Projects/mojo-baro-lanes/REALIGN`, commit `3d7bfb2`.

## Files (all new, all in item scope)

- `serve/realign.mojo` — `realign_expected_embedding(ctx, mut b: WindowBufs, mut e_dev: DeviceBuffer[f32])`,
  matching the E8 plan's interface contract verbatim (the plan's own pseudo-signature omits `mut`
  on `e_dev`; Mojo requires it since the function writes through it — see "Design calls" below).
- `kernels/realign_kernels.mojo` — `amar_realign_copy_row`, `amar_realign_gather`, `amar_realign_reduce`,
  `REALIGN_SPLIT = 32`.
- `kernels/test_realign.mojo` — live test, 5 real prompts on `.work/engine-pack-q4`.
- `tools/realign_oracle.py` — numpy oracle, never touches the GPU.

## What it computes

`e = softmax(logits) @ W_emb` for the row-0 logits already sitting in `WindowBufs.logits_d`,
against the pack's resident `token_embd.weight` (bf16, `[VOCAB, H]`, offset 0 — no dequant scale,
same table `embed_k` reads). Pipeline: copy row 0 of `logits_d` into a scratch buffer (so the
in-place softmax never mutates the caller's logits), `amar_softmax_rows` (reused from
`elementwise.mojo`, not duplicated per item non-goals/§7), then a split-VOCAB gather-accumulate
(`REALIGN_SPLIT=32` blocks along V, one thread per output `h`, reads each table row exactly once,
fully coalesced across the H-contiguous inner dimension) and a reduce over the 32 partials.

## Design call flagged

**The fused megakernel decode path (`WindowCfg.mega=True`) never materializes VOCAB-sized logits
at all** — `serve/window.mojo:945-946` is a literal `if use_mega: pass` where the launch-path
computes `Logitsm` via `r_head`/`gemm_w[VOCAB,H]`. The interface contract cites
`serve/window.mojo:275` for "logits already computed by the window/megakernel path", but line 275
is inside `blk32_forward` (the MTP draft path), not the mega decode path the plan's own prior
receipt (E9) exercises. Caught in test: `mega=True` gave `argmax_tok=0` on every one of 5 different
prompts (the tie-break return value for an all-untouched sentinel buffer). Fixed by setting
`mega=False` in the test's `WindowCfg` (the non-mega "launch path", which does populate
`logits_d`/`hn_d`). This is a config choice in my own new test file, not a change to
`window.mojo`/`mega.mojo` — non-goals ("no change to the megakernel") respected.

`serve/harness.mojo` (referenced by the interface contract for `load_pack`) does not exist on this
branch (confirmed: untracked/uncommitted in the main checkout, absent from every lane worktree's
git history). Rather than create it — out of this item's file list — `kernels/test_realign.mojo`
carries its own trimmed, self-contained pack-load/buffer-alloc (`load_pack_min`/`alloc_bufs_min`),
copied from the same logic already duplicated in `serve/engine.mojo:main()`.

## Gate

Full log: `.work/REALIGN-gate.txt`. Commands:
```
./run-tests.sh
./.venv/bin/mojo build kernels/test_realign.mojo -I kernels -I serve -o .work/test_realign
gpu-wait run --priority 30 --vram 8 -- ./.work/test_realign
./.venv/bin/python tools/realign_oracle.py
```

| step | exit | note |
|---|---|---|
| `test_gemm` (via run-tests.sh) | 0 | `GEMM OK — 4 x 3 @ 3 x 2 matches host reference` |
| `kernel-census.py --check` (via run-tests.sh) | **1** | 3 orphans, **all pre-existing** — see below |
| `test_realign` build | 0 | clean |
| `test_realign` run (gpu-wait) | 0 | 5/5 prompts, see below |
| `realign_oracle.py` | 0 | 5/5 PASS |

**`run-tests.sh` itself exits 1**, solely from `kernel-census.py --check`'s pre-existing orphans in
`kernels/spark_kernels.mojo` (`amar_attn_decode_swa`, `amar_head_gate_mul_cast`,
`amar_skinny_reduce_gelu_par_bf16`) — verified by moving all 4 of my new files out of the tree and
re-running the census: identical 3 orphans, identical exit 1, with zero of my files present. Not
introduced by REALIGN, not touched by REALIGN; reported, not chased.

**Test count (kernel census):** 77 kernels tracked → **80** after (my 3 new `amar_realign_*`
kernels), orphans **3 → 3** (unchanged — mine are reachable via `kernels/test_realign.mojo`'s
import, the census's own criterion for "used"). No floor was given for this item; reporting the
delta since it's the only test-count instrument in the repo's gate.

## Correctness (oracle)

5/5 prompts (`bench/mtp-prompts/p01..p05`, pack `.work/engine-pack-q4`), threshold
`max|e_diff|/max|e| <= 1e-3`:

| prompt | argmax tok | nearest-emb tok (cos) | max\|diff\|/max\|e\| | verdict |
|---|---|---|---|---|
| p01-water | 279 | 279 (0.8743) | 1.28e-05 | PASS |
| p02-python-fib | 262 | 262 (0.9998) | 1.74e-06 | PASS |
| p03-story | 995 | 995 (0.6605) | 2.53e-05 | PASS |
| p04-list-planets | 42120 | 42120 (0.9944) | 5.87e-05 | PASS |
| p05-math | 220 | 220 (0.9541) | 5.50e-06 | PASS |

Worst case 5.87e-05, ~17x under the 1e-3 threshold. Argmax token and nearest-embedding token
(cosine over all VOCAB rows of `W_emb`, computed by the oracle) agree on all 5 — expected, since a
confident greedy step has softmax mass concentrated on one token, so `e` sits close to that token's
own row.

## Performance

Kernel time (copy + softmax + gather + reduce, `ctx.synchronize()`-bracketed), measured per prompt:
**5.0–5.9 ms** (4990–5883 us in the raw log), all above the ≤1ms target. Reported per plan ("report
the number either way"): reading the full ~2.03 GB bf16
embedding table once per call (`VOCAB * H * 2` bytes, unavoidable — every table row is read exactly
once by construction) is the floor; at the pack's measured 855–917 GB/s this repo has hit on other
skinny-GEMM kernels, that alone is ~2.2–2.4 ms, so the remaining ~2.5–3.5 ms is launch overhead
across 4 sequential kernels plus `REALIGN_SPLIT=32` leaving some occupancy on the table — a lead
for a follow-up round, not chased here (non-goal: no megakernel/harness changes).

## Commits

`3d7bfb2` — `realign: expected-embedding kernel e = softmax(logits) @ W_emb` (single commit, all
four files, conventional subject + why-body, no attribution trailers per brief).

## Non-goals honored

No change to `kernels/mega.mojo`, `serve/window.mojo`, `serve/registry.mojo`, or any harness file.
No bf16 path changes. `serve/realign.mojo` contains a stub-free real implementation (not the
`raise Error("REALIGN not merged")` placeholder HARNESS lane is building against) — coordinator
resolves the merge per the plan.

## Round 2

Branch `lane-REALIGN`, commit `798e27e`. Coordinator's brief (`exchange/lane-REALIGN-brief-r2.md`)
correctly flagged that round 1's `b.logits_d` read blocks the real caller (`bench_latent_handoff.mojo`
runs every latent step through the megakernel, exactly E9's `step_latent_raw` loop, `mega=True`
always). Its suggested fix — read `b.hn_d` instead — turns out to be **equally broken, one layer
deeper**, and fixing that (not just swapping the buffer) is what actually landed.

### What was wrong with `b.hn_d`

`kernels/mega.mojo`'s only write to `Hn_` (`amar_mega_token`'s `rms_f32_phase` call) is gated on an
internal `fold_head == 2` argument. Grepped every call site in the repo:

- `serve/window.mojo`'s three mega/mega_win launches (`mega_token_q4_k`, `mega_token_k`,
  `mega_win_k`) pass `fold_head` = 1, 1, 0 respectively — never 2.
- `bench/bench_hidden_dtype.mojo`'s `step_latent_raw` (the function HARNESS's brief says its own
  loop is "exactly") also passes `fold_head=1`.

So `b.hn_d` is not stale under `mega=True` — it is **never written by the fused kernel at all**.
Verified empirically before touching any code: a throwaway diagnostic build (mega=True, dumping
`hn_d[0:8]` + its L2 norm after prefill) showed real, prompt-varying, plausible-magnitude values on
every one of 5 prompts — looking exactly like "it works." It doesn't: those values come from the
row-0 slot of whatever **earlier, non-mega prefill chunk** last wrote it (`rms_m` at
`serve/window.mojo:930-934` runs on every chunk where `m != 1`, i.e. every chunk except the final
single-token one, and always writes to `hn_d[0..m-1]`, not offset by position) — for a 13-20 token
prompt with `MROWS=8`, that's the hidden state for an early-middle token, not the final one. Reading
`b.hn_d` after round 2's literal suggestion would have passed a smoke test and been silently wrong
in production — the same failure class round 1 hit with `b.logits_d`, one buffer over.

### The actual fix

`b.x_d` (the pre-final-norm residual stream) is the one buffer every path — mega, non-mega, spec —
genuinely updates every single step, regardless of `fold_head`/`use_mega`. `realign_expected_embedding`
now:

1. Norms `b.x_d` row 0 into bf16 via `rmsc_h2` (reused, registry alias for `amar_rmsnorm_cast`),
   using `output_norm.weight` at a **fixed, comptime-derived offset**
   `HEAD_NORM_IDX = 1 + N_SSM*10 + N_ATT*7 + N_LAYERS*4` into `b.off[]` — the same formula
   `serve/window.mojo:918` computes at runtime for its own `w`, verified against
   `.work/engine-pack-q4/index.txt` lines 426/427 (`output_norm.weight` / `output.weight`).
2. Runs the head GEMM (`gemm_w[VOCAB,H]`, reused from `window.mojo`, dispatches `gemm_q4`/`gemm_q8`
   per `pack_q4`) against `output.weight` at `off[HEAD_NORM_IDX+1]`, reusing `b.p_v_d` as scratch —
   the same partial buffer the launch path already carries, no new ~250MB allocation per call.
3. Reduces the partial into a fresh `[1,VOCAB]` scratch via a new instantiation of the existing
   `amar_skinny_reduce` (no new kernel).
4. Softmax + gather + reduce over `W_emb`, unchanged from round 1, except the softmax's input is now
   this freshly-computed scratch buffer instead of a copy of `b.logits_d` — **the round-1 copy kernel
   (`amar_realign_copy_row`) is gone**, nothing left to copy from.

Final signature, adopted by HARNESS verbatim per the brief:
```mojo
realign_expected_embedding(ctx, mut b: WindowBufs, mut e_dev: DeviceBuffer[f32], pack_q4: Bool) raises
```

### Test changes

`kernels/test_realign.mojo` now runs with **`mega=True`** (the real HARNESS/E9 configuration).
Round 1's mega=False test is gone (the whole point was to stop testing the path HARNESS doesn't
use). It dumps `b.x_d` row 0 (`.work/realign-dump/<prompt>-x0.f32`) instead of logits, since
`realign_expected_embedding` no longer exposes logits at all — per the brief's offered choice, the
oracle recomputes logits itself rather than comparing against a Mojo-side dump.

`tools/realign_oracle.py` is now a full independent reimplementation of the head + embedding
pipeline in numpy: ggml Q4_0 dequant of `output.weight` (block-32, 16 packed nibble bytes + fp16
scale, formula inverted from `tools/engine-pack.py:quantize_q4_0`), rmsnorm against
`output_norm.weight`, **round-to-nearest-even bf16 truncation of the normalized activation before
the head matmul** (matching what `rmsc_h2`'s cast actually does on the GPU — the first version of
this oracle skipped that and got 2/5 prompts to ~5e-3, an order of magnitude over threshold; adding
the bf16 truncation dropped all 5 back to ~1e-5, confirming the gap was oracle precision, not a
kernel bug), softmax, then the same embedding gather as round 1.

### Gate

Full log: `.work/REALIGN-gate.txt`. Same commands as round 1 (mega now `True` inside the test
itself, no CLI flag). `run-tests.sh` still exits 1 from the same 3 pre-existing `spark_kernels.mojo`
census orphans (untouched by this round, `git diff --stat` confirms only the 4 REALIGN files
changed). Kernel census: 79 kernels tracked (down from round 1's 80 — one fewer kernel, the removed
copy), 3 orphans (unchanged, pre-existing).

### Correctness (oracle, mega=True)

5/5 PASS, same threshold `max|e_diff|/max|e| <= 1e-3`:

| prompt | argmax tok | nearest-emb tok (cos) | max\|diff\|/max\|e\| | verdict |
|---|---|---|---|---|
| p01-water | 279 | 279 (0.8743) | 1.40e-05 | PASS |
| p02-python-fib | 262 | 262 (0.9998) | 1.75e-06 | PASS |
| p03-story | 995 | 995 (0.6605) | 2.52e-05 | PASS |
| p04-list-planets | 42120 | 42120 (0.9944) | 5.87e-05 | PASS |
| p05-math | 220 | 220 (0.9541) | 5.59e-06 | PASS |

Worst case 5.87e-05 — identical to round 1's number, and **every argmax token matches round 1's
mega=False run exactly**, cross-validating that the mega and launch paths compute the same next
token (as they must) and that this round's fix reproduces it correctly from `x_d` alone.

### Performance

5.8–6.5 ms per call (up from round 1's 5.0–5.9 ms — the added head GEMM: `output.weight` is q4,
~508MB packed, read once per call, on top of the ~2.03GB embedding-table read round 1 already paid).
Still above the ≤1ms target; not folded further (copy+softmax already collapsed to just softmax by
construction, not by an explicit fusion) — a candidate for a follow-up round, not chased here.

### Commits

`798e27e` — `realign round 2: derive logits from x_d, not the mega-dead logits_d/hn_d` (single
commit, all four changed files, conventional subject + why-body, no attribution trailers).

## Round 3

Branch `lane-REALIGN`, commit `55dc19e`. Brief (`exchange/lane-REALIGN-brief-r3.md`): HARNESS's
`L8-raw` arm (and E9 before it) reads `b.hn_d` directly for the post-final-norm hidden and ships it
as f32 — dead under `mega=True` for the same reason round 2 found (`kernels/mega.mojo`'s only write
to `Hn_` gates on `fold_head == 2`, no real call site passes that). Add a small function reusing
step 1's `b.x_d` row read to serve this need without touching `b.hn_d`.

### `final_norm_hidden`

```mojo
final_norm_hidden(ctx: DeviceContext, mut b: WindowBufs, mut h_dev: DeviceBuffer[f32]) raises
```

Same `b.x_d` row-0 read as `realign_expected_embedding`'s step 1, same `output_norm.weight` at
`off[HEAD_NORM_IDX]` — but through `rms_h2` (registry alias for `amar_rmsnorm`, f32-in/f32-out)
instead of `rmsc_h2` (`amar_rmsnorm_cast`, casts to bf16). The raw arm wants the un-rounded rmsnorm
output, not the bf16-truncated value the head GEMM consumes — those are two different consumers of
the same normalized activation with two different precision requirements, and round 2's `rmsc_h2`
call only serves the GEMM's. No new kernel: `rms_h2` already existed in `registry.mojo:181` for
exactly this `[1,H]` shape, just unused by this file until now.

### Test + oracle

`kernels/test_realign.mojo` calls `final_norm_hidden` right after `realign_expected_embedding` each
prompt and dumps `h_dev` to `.work/realign-dump/<prompt>-h.f32`. `tools/realign_oracle.py` adds an
independent f32 rmsnorm of the dumped `x0` (no bf16 truncation — this path has none) and diffs it
against the dumped `h`, threshold `max|h_diff|/max|h| <= 1e-4` per the brief, alongside the existing
`e` check.

### Gate

`.work/REALIGN-gate.txt` rewritten (same commands as round 2, plus the new `h` oracle column).
`run-tests.sh` still exits 1 from the same 3 pre-existing `spark_kernels.mojo` orphans (`git diff
--stat` confirms only the 3 REALIGN files changed this round — no kernel file touched, so census is
identical to round 2, 79 kernels / 3 orphans).

5/5 PASS on both checks, `mega=True`, same 5 prompts:

| prompt | max\|e_diff\|/max\|e\| | max\|h_diff\|/max\|h\| | verdict |
|---|---|---|---|
| p01-water | 1.40e-05 | 6.93e-08 | PASS |
| p02-python-fib | 1.75e-06 | 6.59e-08 | PASS |
| p03-story | 2.52e-05 | 7.23e-08 | PASS |
| p04-list-planets | 5.87e-05 | 2.91e-08 | PASS |
| p05-math | 5.59e-06 | 9.85e-08 | PASS |

Worst `h` case 9.85e-08 — two orders of magnitude under the 1e-4 threshold, as expected for a path
with no precision-losing cast at all (the `e` numbers are unchanged from round 2, since
`realign_expected_embedding` itself was not touched this round).

### Commit

`55dc19e` — `realign round 3: add final_norm_hidden, the f32 post-final-norm hidden HARNESS's raw
arm needs` (3 files: `serve/realign.mojo`, `kernels/test_realign.mojo`, `tools/realign_oracle.py`;
no attribution trailers).
