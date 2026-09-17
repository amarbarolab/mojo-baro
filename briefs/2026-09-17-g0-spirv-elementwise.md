# G0: every elementwise kernel through SPIR-V onto aihq-lab's Radeon R5 M330

the maintainer opened G0 on 2026-09-17. It is the first stage of "mojo-baro on aihq-lab, CPU + GPU":
G0 elementwise -> G1 general phase-splitting converter -> G2 q4 GEMM/GEMV + SSM delta -> G3 small
model decode vs llama.cpp Vulkan. G0 decides whether G1 is worth building.

## Where

- Worktree `~/Projects/mojo/mojo-baro-lanes/g0`, branch `lane-g0`. Edit and commit only there,
  `git commit -- <paths>` (pathspec, never bare).
- Starting point, read fully first: `tools/spirv-probe/` (`run.sh`, `air2spv.py`, `host.c`,
  `rms2spv.py`, `host_rms.c`) and `~/Brain/mojo/mojo-baro/2026-09-17-mojo-to-spirv-probe.md`.
  swiglu and rmsnorm already PASS on the card.
- Target: `root@lab-host.example` (ssh key works, root). Run OpenCL with `RUSTICL_ENABLE=radeonsi`.
  Max buffer 512 MiB. The lab GPU is not managed by gpu-wait; the workstation GPU is not used at
  all in G0 (no gpu-wait jobs; `mojo build --target-accelerator apple-m1 --emit asm` is CPU only).
- A CPU setup script `/root/mojo-baro/.work/cpu-stack.sh` may be running on the lab box; do not
  touch `/root/mojo-baro`, work in `/root/g0/`.

## Task

For every `amar_*` kernel in `kernels/elementwise.mojo` (rmsnorm, rmsnorm_cast, rmsnorm_cast2,
swiglu, rope_rows, softmax_rows, embed_lookup, embed_lookup_pos, argmax_pos, argmax_row, tok_copy,
tok_remap, quantize_q8_rows):

1. Emit its Metal IR (the `test_elementwise.mojo` build already emits most; add a probe-only Mojo
   file under `tools/spirv-probe/` that instantiates any kernel the test does not).
2. Inventory: every `air.*` call, address space, `air.wg.barrier`, `simd_shuffle*` pattern, shared
   memory global, loop shape. One table row per kernel.
3. Lower it to SPIR-V (extend the probe scripts; shared logic goes into one module, no copy-paste
   per kernel) and run it on the card with a host check against a CPU reference implementing the
   kernel's formula. Parity bar: max relative error <= 1e-5 for float outputs, exact for integer
   outputs (argmax, token ids, q8 codes).
4. Extend `tools/spirv-probe/run.sh` so one command runs every kernel and ends `FAIL k/N <names>`
   non-zero or `PASS N/N`.

## Kill line

G0 fails, and G1 is not built, if any kernel contains a construct with no phase-split lowering
(write down exactly which construct and why), or if a kernel needs more than 8 host-dispatched
phases.

## Rules

- Global rules bind: no em dashes anywhere, loud failures (`set -euo pipefail`, FAIL lines),
  kernel files under `kernels/` untouched (no comments added there), Python allowed here only as the
  IR rewriter and oracle next to the tools it checks.
- DONE means `run.sh` PASS N/N was run end to end on the card and pasted in the report.
- Deliverable: `exchange/lane-G0-report.md` on `lane-g0` with the inventory table, per-kernel phase
  count, parity numbers, the kill-line verdict, and an honest size estimate for G1. Reply to the
  coordinator pane `w82:pC` only with "written to <path>".
