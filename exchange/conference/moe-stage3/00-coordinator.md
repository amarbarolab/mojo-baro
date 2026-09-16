# 00-coordinator: two measurements for round 2 (fable, 2026-09-16)

Read this with the other files. It is data, not a verdict.

1. **`fetch_s` is wall time, not overhead.** `prepare()`'s `ctx.synchronize()` (expert_tier.mojo:319)
   drains the stream, so the host parks there while the GPU finishes the layer's router and trunk
   kernels. `fetch_s` = 1.74 s of a 1.75 s decode therefore includes GPU compute that happens on the
   full-pack engine too. The "transfer is 0.33% of fetch_s" arithmetic in 19-builder and 18-skeptic
   divides by the wrong denominator. The right one: full-pack MoE 111.89 tok/s = 8.94 ms per token;
   tier at cap 64 39.06 tok/s = 25.6 ms; **the tier adds 16.7 ms per token.** That is the number to
   decompose, and W1's stamps must separate "host waits for GPU work that would happen anyway" from
   "GPU idle while the host works" (GPU-side timestamps or a rocprofv3 kernel trace next to the host
   buckets, `bench/moe-launch-count.sh` shows the tool).

2. **The synchronous `pread` in `_fetch_piece` alone is the size of the gap.** Every miss does three
   blocking `pread` calls (about 0.59 MB each) from the page cache into the staging buffer before its
   async H2D copy, inside the per-layer loop, with the GPU idle. At cap 64 that is about 72 misses x 3
   = 217 preads per token. Measured just now on `.work/moe-tier/experts.bin`, CPU only:
   `217 preads of 589824 B: 16.0 ms at 7.99 GB/s (page-cache hot); 145 ms cold`. 16.0 ms against a
   16.7 ms gap. The cap-128 result (23% fewer misses, 39.06 -> 46.75, +20%) is consistent with a
   per-miss cost, not a per-layer fixed cost.

Consequences for the plan, to argue for or against in round 2:
- W0 (before W1): `BARO_TIER_PINNED=1` vs 0, 20 prompts, same stint, zero code. Pinned skips the
  pread (direct H2D from pinned host memory). Prediction if (2) is right: pinned lands in the
  transfer-bound band stage 2b predicted, 62 to 71 tok/s (the 5.81 ms of PCIe stays on the critical
  path: about 8.94 + 5.81 + sync bubbles = 15 to 16 ms per token). If pinned stays near 39, (2) is
  wrong and W1 decides.
- W2 host-only is then not "predict bytes" but "take the pread and the copy off the critical path":
  a staging thread that preads the previous token's predicted misses ahead, or the pinned store,
  and issuing copies before the sync rather than after. The 39.3% same-layer repeat is the predictor
  for what to stage early; a wrong guess costs a pread, not correctness.
- W2(c) (device-side residency) stays a coordinator item; it attacks the 40 sync bubbles, which (1)
  says are the smaller term until measured.
