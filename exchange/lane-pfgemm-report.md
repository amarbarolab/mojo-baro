# lane-pfgemm report (2026-09-08)

Branch `lane-pfgemm` (worktree `~/Projects/mojo-baro-lanes/pfgemm`), from main
`b3da39a`. Protocol `bench/pfgemm-protocol.md` (this branch). Status
`.work/briefs/status-pfgemm.md`.

## What landed

- `0128bb8` kernels: `kernels/matmul_prefill_lds.mojo` -- the bf16 arm of
  lane-int8 `813c999` with the int8 MMQ path REMOVED (not kept dead: the
  R5 verdict is a hardware fact, int8 WMMA = bf16 rate, and dead code in a
  zero-comment kernel file has no way to say why it is dead) and an
  int8-weight loader added for the q8 pack. Brief named
  `kernels/matmul_wmma_lds.mojo`; that file is the live fp16 dense kernel
  (`bench_fp16*.mojo`, BASELINE), so the new file has its own name.
  `kernels/test_prefill.mojo` 1b: lds vs R4 bit-exact (7 checks, q4 + q8).
- `26c1971` serve: `gemm_prefill_q4/q8` dispatch m > 128 -> lds 128x128;
  m <= 128 stays on the R4 kernel (R5 receipt: lds slower below 128 rows).
  `prefill_forward` prints the per-chunk GEMM share under `BARO_PROFILE`.
- `62adaf8` bench: `bench_prefill.mojo` with the bf16-lds arms, bit-exact
  gate vs wmma.

## Files outside the lane's ownership

None. `.venv` (tracked self-loop symlink from `4027b6e`) is untracked in
`0128bb8`, as the driver did on main.

## Measured so far (`.work/logs/`, protocol `bench/pfgemm-protocol.md`)

- Main engine bar re-measured this stint: 8k 8.32 s (8.31-8.34), 32k 51.40 s (50.55-51.49).
- GEMM share on the R4 path (new `BARO_PROFILE` print): 0.614 at 8k, 0.409 at 32k; 0.515 s of GEMM per 1024-row chunk, the remainder grows with position.
- Frozen prediction (`10d6ca5`, from the R5 1.86x and the share): prefill_s 5.95 s at 8k, 41.6 s at 32k, ~275 s at 100k; gate <= 6.6 / <= 44 s.
- Kernel-level try 1 this session: lds/wmma 1.74x at n=1024 (R5: 1.86x), a foreign build job co-admitted.
- Parity: 7 lds-vs-R4 bit-exact checks PASS.

## Not measured

lds engine timed runs, prefill identity vs main, merge gate, 20-prompt decode A/B, lds-path share receipt. All scripted in `.work/pfgemm-stage2.sh`; see the HANDOFF section of the status file for the one command.

PAUSED: budget stop (the maintainer 13:3x) before the lds timed runs; prediction is frozen, nothing has been measured against it.
