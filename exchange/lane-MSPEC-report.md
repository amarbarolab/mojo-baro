# Lane MSPEC report: step 1 (measure)

Plan: `~/Brain/mojo/mojo-baro/briefs/2026-09-11-mega-spec-window.md`, item MSPEC, step 1 only. Step 2 was not started; it needs the maintainer's go.

## Verdict

The frozen rule passes on the upper bound only. The median verify-window gap share at k=2 is **15.65%** on the rocprofv3 timeline, above the 10% kill line. The tracer itself slowed the engine to 0.93x, below the preregistered 0.95 allowance, so 15.65% is an upper bound. Charging the tracer's extra decode time to window gaps gives 8.35%, and charging only the windows' share of dispatches gives about 9.2%. Both corrected estimates sit below the kill line.

The ceiling for a zero-gap window is +16% traced and +8% corrected, at k=2 150.96 tok/s_gen. The 2026-09-06 W3 round already measured an m=3 megakernel losing more than that to row scaling: GEMM phases at 1.6x of m=1 against 1.5x native, plus 300 us on the head.

**Recommendation: do not start step 2.** The realistic ceiling is +8%, and the one measured m=3 megakernel gave back more than 8%. the maintainer decides.

## Numbers (20-prompt medians, q4 pack)

| | k=2 | k=4 |
|---|---|---|
| dispatches per verify window | 678 | 678 |
| verify window wall, ms | 14.27 | 19.31 |
| gap share, traced | 15.65% (15.49-16.23) | 14.95% (14.69-20.51) |
| gap share, tracer-corrected lower bound | 8.35% | 9.21% |
| spec tok/s_gen, bare | 150.96 | 127.01 |
| tok/s if gaps were zero, traced / corrected | 174.71 / ~163 | 144.68 / ~137 |
| traced / bare tok/s | 0.93 | 0.95 |

No-spec arm A: 137.02 tok/s_gen, so on q4 k=2 speculation is 1.10x and k=4 is 0.93x. No llama.cpp arm ran, so no ratio against it is claimed.

## Evidence

- Protocol, predictions frozen at `7df2c61` before any trace, Result appended: `bench/mtp-protocol.md`, section "Amendment MSPEC step 1".
- Per-prompt table: `$HOME/Projects/mojo/mojo-baro-lanes/MSPEC/.work/mspec/run1/summary.md`.
- P1 receipts and identity: `.work/mspec/run1/receipts.txt`. All 100 speculative runs read back spec on, k, `BARO_MEGA: True`, `BARO_MEGA_WIN: False` and q4 pack. Identity passed 100/100 against the no-spec output. Pack and engine sha256 are recorded there.
- Cross-check: `.work/mspec/run1/oracle.txt`, a Python csv oracle, gives the same gap share on all 40 traces with 0 malformed windows.
- Brain note: `~/Brain/mojo/mojo-baro/2026-09-11-mspec-step1-gap-share.md`.

## Gate

- Command: `gpu-wait run --vram 16 -- bash -c './run-tests.sh > .work/MSPEC-gate.txt 2>&1; echo "run-tests exit $?" >> .work/MSPEC-gate.txt'`.
- Exit code: 0 (`run-tests exit 0`).
- Test count: 72 PASS lines before (`.work/MSPEC-gate-before.txt`) and 72 after, plus the kernel census showing 85 kernels and 0 orphans both times. The item sets no test floor, and no test was added or edited.
- Evidence: `$HOME/Projects/mojo/mojo-baro-lanes/MSPEC/.work/MSPEC-gate.txt`.
- The gate ran to completion before the gpu-waitd restart. The file holds the full suite output.

## Commits on lane-MSPEC

- `7df2c61` bench(mtp): preregister MSPEC step 1 (predictions frozen before any trace).
- `b4d17bb` bench(mspec): per-window launch-gap analyzer and 20-prompt trace runner.
- `95f4c89` bench(mtp): MSPEC step 1 result.

## Flags

- The plan named the `rocprof-kernels` iTool as the instrument. It aggregates per kernel and has no timeline segmentation, so I used rocprofv3 directly with a new analyzer, `bench/mspec_gaps.mojo`, and a runner, `bench/mspec-trace.sh`. Neither file is listed in the item.
- The first probe failed on a wrong prompt filename, `p01.tokens`. It produced no numbers and was superseded by the full run.
- p20 at k=4, traced: one 120.6 ms host-side stall, cause not diagnosed. Dropping it moves the k=4 median by 0.04 points.
- The README MTP numbers (100.7 vs 123.5) are from the q8 pack on 2026-09-04 and are stale for q4. I did not edit them, since README is outside the item.
