# Question for the builder: how do you want W2 to W4 shaped?

Answer this AFTER you finish and commit the step you are on. Do not change course on your own: write
the answer file, say it is written, and wait. Nothing here is a decision yet.

The plan (`$HOME/Brain/mojo/mojo-baro/briefs/2026-09-11-moe-engine-wiring.md`) was written without
asking you, and its own front matter says the builder conference is still pending. the maintainer wants the
remaining LOC (about 800 to 1150 of the 1000 to 1400 estimate) cut where cutting is real, not where it
just moves work into a harder step.

## What I want from you

1. **Where is the plan wrong?** Anything in W2, W3 or W4 you would not build that way, and why.
2. **What would make this lane fail?** The step, the gate, or the missing fact most likely to cost hours.
3. **How do you want items shaped?** Size, ordering, what a step should hand the next one, what you want
   preregistered versus left open. If you want the briefs written differently, say exactly how.
4. **What is missing?** A fact, a fixture, a reference or a tool you need that no step currently produces.

## Two LOC-cut candidates, for you to accept or reject

**A. Requantise experts in W1 and delete W2.** The engine already has a q4 GEMV path (`gemm_w` in
`serve/window.mojo:304`, `q4: Bool`, weights as raw bytes at an offset in `wbuf`). If `tools/engine-pack.py`
converted Q4_K expert tensors into our own q4 layout instead of copying raw ggml Q4_K blocks, the expert
GEMVs might reuse that path and W2's 250 to 350 LOC of Q4_K dequant could shrink or vanish. Costs to weigh:
requantisation is lossy against llama.cpp, so parity becomes "within a bound" rather than exact; pack size
must still fit 21.5 GB; our q4 layout may not match the gather pattern the MoE kernels need (`amar_moe_gate_up`
indexes `row = e * FFN + r` over a 2D bf16 tile, not a byte offset). Is that reuse real, or does the
per-expert gather make it a rewrite anyway?

**B. Make W2 a dtype parameter on the existing kernels instead of new ones.** `amar_moe_gate_up[NSEL, FFN]`
and `amar_moe_down[NSEL, FFN]` already carry the routing, the gather and the wave layout, and only the
weight load is bf16-specific. A comptime dtype or a small load helper could give the Q4_K variant without
duplicating either kernel. Does that hold once 6-bit scales and mins land in the inner loop, or does the
super-block structure force a separate kernel?

Reject either one if it is false economy. Fewer LOC is not the goal if it costs a real gate.

## Deliverable

Write `$HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-conference-answer.md`. Your chat reply is only:
`written to $HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-conference-answer.md`. Then continue the
lane as briefed, unless the answer says a step should change, in which case stop and wait.
