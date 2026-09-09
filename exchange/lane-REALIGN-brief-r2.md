# Lane REALIGN — round 2 (coordinator review of `3d7bfb2`)

Oracle 5/5 and the kernel are accepted. One integration defect blocks the merge, found from your own report:

**Under `mega=True` there are no logits.** HARNESS runs every latent step through the megakernel (`bench_latent_handoff.mojo:125`, `mega=True`, exactly E9's loop), so `realign_expected_embedding` as written would softmax a stale/sentinel `logits_d`. Your test only passes because it switched to `mega=False`. The real caller never will.

Fix (your files only): make `realign_expected_embedding` compute the logits itself from the post-norm hidden in `b.hn_d` row 0, the way the launch path does at `serve/window.mojo:943-944` (`gemm_w[VOCAB, H](ctx, <hn row0 as CurBm>, b.wbuf, b.off[w + 1], cfg.pack_q4, Pv, 1)` then `r_head` into a scratch logits buffer — find how `w` is derived there; take `pack_q4`/offset through the signature if you must, but keep the call shape `realign_expected_embedding(ctx, b, e_dev, ...)` and document the final signature at the top of `serve/realign.mojo` — HARNESS will adopt it verbatim). Do not read `b.logits_d` at all. Verify `hn_d` under mega really is post-final-norm (E9 assumes it; check `kernels/mega.mojo` where `Hnm0` is written) — if it is pre-norm, apply `rmsc_k` first.

Test: `kernels/test_realign.mojo` must run with `mega=True` (the E9/HARNESS configuration) and pass the same oracle (`tools/realign_oracle.py` recomputes logits from the dumped `hn` row + the pack's head weights, or keeps comparing against dumped logits — your call, say which). Re-report timing; if you can fold copy+softmax into one launch cheaply, do it, otherwise just report.

Same rules as round 1 (gpu-wait, gate to `.work/REALIGN-gate.txt`, no attribution trailers). Report by rewriting `exchange/lane-REALIGN-report.md` (add a `## Round 2` section), then push `DONE -> exchange/lane-REALIGN-report.md`.
