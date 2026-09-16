# Masked greedy and masked probs for grammar decoding (fable, lane-samplemask, 2026-09-16)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-fable-masked-sampler.md`, for the JSON lane
(w82:p9). Kernels in `kernels/sample.mojo`, host reference in `serve/sample_ref.mojo`, test
`kernels/test_sample_mask.mojo` in `run-tests.sh`. Branch `lane-samplemask`, worktree
`.work/wt/samplemask`; merge is the coordinator's.

## The three items

1. **Bug confirmed and fixed: T <= 0 ignored the mask.** `sample_row_body` called `greedy_tok`
   over the full row before any mask check. `greedy_tok` now takes `MASK` and reads through
   `mload4[MASK]`, so the argmax is over allowed tokens only. Unmasked `MASK = False` compiles to
   the previous code.
2. **Order checked, found wrong, fixed: mask before truncation.** The cut (`sample_cut`,
   `fast_cut`, `refine`) ran on the unmasked row and the mask only filtered members of the
   draw loop, so a mask whose allowed tokens all sat outside the top-k/top-p cut returned -1.
   Now every pass reads the row through `mload4[MASK]` / `mload1[MASK]`, which turn masked
   tokens into NaN, and NaN is what `is_valid` already rejects: the mask is applied before
   `lmax`, the valid count, every histogram and mass, the cut, and the draw, exactly as an
   invalid logit would be. This is llama.cpp's order (grammar before the samplers). A draw
   returns -1 only when no allowed token exists. The host reference got `mask_logits(logits,
   words)` (masked -> NaN); `sample_row_ref` and `sample_probs_ref` on that copy are the masked
   reference with no second code path.
3. **New `amar_sample_probs_masked`.** `amar_sample_probs` is now a wrapper over
   `sample_probs_body[MASK]`; the masked variant takes the same `mask, mask_stride` pair as the
   row sampler, writes 0 for masked tokens, and at T <= 0 is one-hot on the masked argmax. The
   draft head stays unmasked; the accept rule rejects a forbidden draft once Pt is masked.

## Interface (sent to w82:p9 before implementation)

```
mask: MutPointer[Scalar[uint64]], bit i set = token i allowed; mask_stride = words per row
      ((VOCAB + 63) // 64); row r reads words [r * mask_stride, ...).
amar_sample_row_masked[XL, OL, PL, CAP](X [R,VOCAB], Out [R] i32, Prob [R] f32, n, temperature,
      top_k, top_p, min_p, seed, counter, mask, mask_stride: Int32)   (signature unchanged)
amar_sample_probs_masked[XL, PL, CAP](X [R,VOCAB], P [R,VOCAB] f32, n, temperature, top_k,
      top_p, min_p, mask, mask_stride: Int32)   grid R, block SAMP_THREADS
serve/sample_ref.mojo: mask_logits(logits, words) -> List[Float32]
```

## Gates (`.work/test_sample_mask` under gpu-wait, 3 real decode rows at VOCAB 248320, `.work/m5/logits-p0{1,2,3}.bin`)

| gate | result on p01 / p02 / p03 |
|---|---|
| A, T = 0 masked argmax (top-3 cleared; 5 allowed ids) | device == host masked argmax on both masks, all rows; Prob = 1 |
| B, T > 0 masked draws, 3 configs x 16 counters x 2 masks | 96/96 token and prob (1e-5) equal to `sample_row_ref` on the masked copy per row; the 5-allowed mask with top_k 20/40 and top_p 0.8/0.95 never returned -1 and always drew an allowed id |
| C, chi-square, 20000 draws under the top-3-cleared mask, ranks 0..9 + rest, p = 0.001 | 19.9 / 9.1 / 9.0 against critical 29.76 at df 10; forbidden tokens drawn 0 |
| D, masked probs rows vs `sample_probs_ref` (T 0.9, k 40, p 0.95) | max abs dp 3.0e-8 / 3.0e-8 / 1.5e-8 (bar 1e-5); masked tokens exactly 0; T = 0 one-hot on the masked argmax |
| E, unmasked identity | `amar_sample_row` T = 0 == host argmax on every row; full mask == unmasked draws 16/16 |

ISA (`tools/isa-receipt.py` on the test binary): `amar_sample_row_masked` 101 VGPRs, 0 spills,
0 scratch; `amar_sample_row` 102 VGPRs, 0 spills; `amar_sample_probs_masked` 119 VGPRs, 0
spills, 0 scratch. Census 103 kernels, 0 orphans.

Repo gates in the worktree: see the commit body.

## Notes for the wiring

- The mask applies to the cut, so `top_k` counts allowed tokens: `top_k = 20` with 5 allowed
  tokens keeps all 5.
- One extra 64-bit load per 4 logits in every pass when `MASK` is on; the unmasked kernels are
  unchanged (E and the identical VGPR count are the evidence).
- The draw loop's own mask-bit test is kept (redundant with the NaN, harmless).
