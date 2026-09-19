# MoE prompt-lookup verification feasibility

Status: RAN 2026-09-19, 20 prompts. Identity 20/20, acceptance 37.7%, 0.76x (76.8 vs 101.1 tok/s spec-off, control 92.3). Rows cost full price (about 11 ms per row vs 9.9 ms per token): the row-count falsifier below fired. Receipt `.work/moe-ngram/full/report.json`.

The first implementation is greedy-only and opt-in: `BARO_NGRAM=1`, or
`spec:true` in the engine line protocol. It searches the last 4096 committed
tokens for the longest repeated suffix (up to 8 tokens), taking at most
`BARO_SPEC_K` continuations. The default width is 2. No separate model loads.
Sampling, penalties, grammar, hidden/logit output, explicit stop sequences,
and diagnostic tensor/expert dumps retain ordinary decoding.

Verification processes candidate rows in layer-major order with the existing
MoE row kernels, preserving each candidate's KV position and SSM/conv ring
slot, then batches the output head. It commits the matching prefix and one
target token. It does not fuse the expert GEMVs or promise a speedup.

## Checks before interpreting speed

1. `serve/test_ngram.mojo`: empty/no-match history, longest and latest match,
   width bound, and overlap.
2. Build unchanged HEAD control, candidate MoE, and candidate dense engine.
3. Run `bench/moe-ngram.py CONTROL CANDIDATE PACK OUT --smoke` through
   gpu-wait, then the same command without `--smoke` for 20 prompts.
4. Require identical generated tokens for clean control, candidate spec-off,
   and candidate spec-on. Require actual drafted tokens and the engine's
   `draft_kind:ngram` receipt. Inspect rejection-first, rejection-later, and
   all-accepted coverage; missing coverage is unverified, not a pass.
5. Report 20-prompt median/min/max decode tok/s, acceptance, proposal time,
   verification time, rows/windows and cost per committed token. The latter
   includes the target bonus/correction token. Do not present a smoke result
   or an unsynchronized timing as a throughput claim.

The expected limitation is serialized per-row expert/projection work. If
verification cost scales with row count, record that falsifier before investing
in fused multi-row kernels. This prototype does not establish P5 row scaling
for a future fused kernel or a 2x target.

Artifacts: `.work/moe-ngram/`. The clean control snapshot is from HEAD;
pre-existing edits to build tools and quality scripts are not part of this work.

## LatentOS evidence checked during this work

E12 and E14 prove same-model KV/SSM prefix reuse, saving repeated prefill.
E12: 120/120 identical answers, receiver median 0.530 to 0.346 seconds.
E14: three followers 57.07 to 22.66 seconds, 3/3 identity. E13-mini's
compact-hidden-state projector showed no signal. These do not establish a
cross-model draft or a RegesCore decode speedup. Source receipts live in
`~/AMDHQ/runs/latent-os/`; current transport/identity APIs are in `latentos/`
and `serve/latent.mojo`. The target verifier can be reused with a future draft.
