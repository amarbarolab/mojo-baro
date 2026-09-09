# lane-attn report — Round A in-block decode attention (2026-09-08, PAUSED 13:34)

Branch `lane-attn` at `d337fa1` (worktree `~/Projects/mojo-baro-lanes/attn`).
Status file: `~/Projects/mojo-baro/.work/briefs/status-attn.md` (HANDOFF 13:34).
Protocol: `bench/attn-protocol.md` (Round A + Amendment 1, both frozen before
the run; Result not yet written).

## Finding 1 — the brief's premise was wrong (no GPU time spent on it)
The megakernel block is 8 x wave32 = **256 threads** (ISA metadata
`.wavefront_size: 32`, `.max_flat_workgroup_size: 256`); `if tid < HD` idles
no thread. `bench/chat-protocol.md` M0d L482 ("512-thread block, threads
256..511 idle") assumed wave64. Recorded in `attn-protocol.md` Amendment 1;
lane-chat's file untouched. The two-halves `attn_head_span2` was built,
fingerprinted, never run; diff parked at `.work/span2-wrong-premise.patch`.

## Finding 2 — the real bound, from the M0d ISA
Attention K loop: 32 iterations of `2 x global_load_b128 -> s_waitcnt
vmcnt(1) -> 8 v_fmac` (one load pair in flight per wave); V loop: 8 loads in
flight x 32 iterations. ~64 memory-latency steps per 256-position chunk,
~21 us/chunk measured at 32k. Memory-level parallelism, not thread count.

## Change (d337fa1, bit-exact on both paths)
`attn_head_span`: K loads 8 vec8 slices per position before their FMAs
(`AK_U = 8`), V hoists 32 loads + 32 score reads per step (`AV_U = 32`),
tails kept; arithmetic order unchanged. Fingerprint: vgpr 249, spill 0, loop
class unchanged (dual 122/84/84/61); K loop 7 loads in flight, V loop 32.
Predictions (Amendment 1): 8192 >= 119, 32768 >= 91 (floor 88), 100000 >= 55
tok/s_gen, capped by the 800 GB/s KV-byte floor (unique KV/token = T x 73.7
KB); 20-prompt within +-2 %, identity 20/20.

## Receipts in hand (`.work/a/stint.txt`)
- test_mega_block PASS (bit-identical to launch path, m=1 q4/q8, m=3 q8).
- One-shot q4 default EXACT vs ref (138.0 tok/s_gen), q8 default EXACT
  (81.8); forced split q4 64/64, q8 64/64 vs model-ref; fail word 0.
- Stint still running at 13:34: forced-split A/B, long 8192/32768/100000
  (`.work/a/long-*.log`, compare GENERATED to `.work/ref/long-*.log`).
- NOT run: default 20-prompt A/B — `AB_ENGINE_B` does not cross `gpu-wait
  run`, ab-prompts.sh REFUSED. Fixed command in the HANDOFF section.
- test_attn_block: fixture symlink was missing in the worktree (now linked);
  re-run needed.

## Defect found in a prior receipt (not mine to edit)
M0d's default 20-prompt A/B (`chat-protocol.md` P-E4 "ratio 1.000, identity
20/20") compared `engine-m0c` with itself: `lanes/chat/.work/m0d/ab/arm.txt`
reads `engA=.work/engine-m0c engB=.work/engine-m0c` (same `AB_ENGINE_B`
propagation bug; the REFUSED guard in ab-prompts.sh now catches it). The
forced-split A/B in that stint was valid (same binary, different env).

## Round B
Not started. Design notes in the HANDOFF section of the status file.

PAUSED: budget cutoff (the maintainer 13:3x); Round A receipts partial, stint in flight, default A/B and test_attn_block re-run pending.
