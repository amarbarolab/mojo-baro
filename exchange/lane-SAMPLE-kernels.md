# Sample kernels (device halves of penalties and top_logprobs), fable, 2026-09-16

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-fable-sample-kernels.md`, for
the sampling lane's items 3 and 4 (`exchange/lane-SAMPLE-report.md`, w82:p7).
Kernels in `kernels/sample.mojo`; test `kernels/test_sample_pen.mojo`, in
`run-tests.sh`.

## Decisions (the coordinator's three questions)

**Sparse, not dense.** Dense would read and write 248,320 f32 per row per
token (2 MB, about 2 us of bandwidth plus a launch) and keep a 1 MB Counts
plane per row plus a bump launch. Sparse touches at most the distinct
generated ids of the request (21 elements in the gate's 14-token history),
one launch, no device state, a few KB uploaded per token from the history
the host already keeps for stop matching. Launch cost is the same one
dispatch; bytes are three orders of magnitude fewer. Chosen: sparse.

**Speculation.** The layout is per row: `Ids/Cnt [R, CAP]`, `Npen [R]`, so
row j of a k+1 window carries the shared history plus drafts 0..j-1 (the
host merges the multiset into distinct ids with counts). The gate exercises
three rows with drafts and checks each against `sample_ref.apply_penalties`
on that row's own history. Nothing forces spec off.

**Order and semantics.** Penalties subtract from the raw logits in place,
before temperature and truncation, in the exact float form of
`sample_ref.apply_penalties` (`x - (presence + frequency * Float32(count))`),
so the row `amar_sample_row` then draws from is the penalized row. top-N
probabilities are of the **truncated sampling distribution at the request
temperature, after penalties**: the distribution the draw came from, the one
`sample_probs_ref` computes, and the one whose `Prob` `amar_sample_row`
already returns, so the chosen-token logprob and the top-N list agree. With
`top_k = 0, top_p = 1, min_p = 0` this is OpenAI's meaning (the untruncated
model distribution at T). At `temperature <= 0` the sampler is one-hot, so
top-N reports the untruncated softmax at T = 1 (entry 0 is the argmax) and
the chosen-token logprob should be read from `TopProbs[0]` there.

## Interface (sent to w82:p7 before implementation)

```
amar_apply_penalties[XL, IL, CL, NL](
    X: [R, VOCAB] f32 (in place), Ids: [R, CAP] i32, Cnt: [R, CAP] i32, Npen: [R] i32,
    n_vocab: Int32, presence: Float32, frequency: Float32)
    grid_dim = R, block_dim = PEN_THREADS (256); ids outside [0, n) or count <= 0 ignored;
    Ids within a row are distinct (host contract), Npen[row] <= CAP.

amar_topn_probs[XL, IL, PL, CAP = SAMP_CAP](
    X: [R, VOCAB] f32, TopIds: [R, NMAX] i32, TopProbs: [R, NMAX] f32,
    n_vocab: Int32, nsel: Int32 (<= 20, <= NMAX), temperature, top_k, top_p, min_p)
    grid_dim = R, block_dim = SAMP_THREADS; entries sorted by logit desc, ties lower id first;
    unused slots id = -1, prob = 0. Call after apply_penalties on the same X.
```

## Gates (`.work/test_sample_pen`, run under gpu-wait, 3 real decode rows from `.work/m5/logits-p0{1,2,3}.bin`)

- Gate A, penalties: for (presence, frequency) in (0.7, 0.3), (1.5, 0), (0, 0.9),
  device rows **bit-equal** to `apply_penalties` on the host copy, 21 elements
  touched per row set; then `amar_sample_row` (T 0.8, top_k 40, top_p 0.95) on
  the penalized device rows draws the same token as `sample_row_ref` on the
  penalized host rows with prob within 1e-5, rows 0, 1, 2 (rows 1 and 2 carry
  one and two drafts respectively).
- Gate B, top-N: configs (T 0.7, k 40, p 0.9, N 20), (T 1, untruncated, N 20),
  (T 0.8, p 0.95, min_p 0.05, N 5), (T 0, N 20): ids equal to the host top-N by
  (logit desc, index asc) on all rows; max |dp| 4.2e-7 against
  `sample_probs_ref` (bar 1e-5).
- ISA (`tools/isa-receipt.py`): `amar_apply_penalties` 6 VGPRs, 18 SGPRs, 0
  spills, 0 scratch; `amar_topn_probs` 184 VGPRs, 52 SGPRs, 0 spills, 36 B
  scratch. Census 102 kernels, 0 orphans.
- Repo gates at the commit: see the commit body.

## Cost note for the wiring

`amar_topn_probs` runs the sampler's cut twice (the sampling cut and a
top-N cut) plus two row passes; it is only launched when `top_logprobs > 0`.
