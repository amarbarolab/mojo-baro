# Lane MOEPF: batched prefill for qwen35moe (RegesCore-35B), 2026-09-19

Branch `lane-moepf` (worktree `.work/lanes/moepf`), base `ff8e523` = lane-r63 head, because
`main` lacked `BARO_SEQ_CAP`; the branch carries all of lane-r63 (32 commits) plus this lane.
Skills read in full before code: inference-kernels, gate-authoring, preregister-experiment,
llm-benchmark-method, prefix-cache-discipline, checkpoints. iTools used: rocprof-kernels,
gate-dryrun, lane-prep, lane-status, lane-merge. One sonnet assistant (`moepf-asst`, w9C:p3) did
the Build_Area read, rocm-doctor, the speed and needle gate scripts and the MAX stream API read.

## Verdicts

| gate | result | check |
|---|---|---|
| G1 identity, tier | PASS 23/23 equal over 64 tokens, 12 exercise prefill, 256-row chunks | `bench/moe-prefill-identity.sh`, receipts `exchange/receipts/MOEPF/g1-tier-*` |
| G1 identity, resident | PASS 23/23 equal over 64 tokens, 12 exercise prefill | same script, `g1-resident-*` |
| G2 speed | PENDING at the time of this commit (running); receipts below are a timing job, not the gate | `bench/moe-prefill-speed.sh` |
| G3 needle, about 100k tokens, tier | PENDING (running) | `bench/moe-prefill-needle.sh` |
| G4 `run-tests.sh` | rc=0 under gpu-wait on the lane tree | `exchange/receipts/MOEPF/run-tests-tail.log` |
| G4 `tools/ci-checks.sh` | see the merge commit | |

`BARO_PREFILL` stays OPT-IN for the MoE profile (`BARO_PREFILL=1`). The protocol's adoption rule
flips the default only after G3; that is a one-line follow-up, not part of this merge.

Of the 20 `bench/mtp-prompts` the brief names, only 9 have the 17 tokens that reach PF_MIN, the
longest prefills 58 rows and none crosses a chunk. The gate therefore adds 128, 512 and 1024
token prompts and runs tier mode with 256-row chunks so every long prompt crosses boundaries.

## Speed (receipts: tier mode, engine clock, final build `c5b664a`, 2 repeats, spread under 1.2%)

| prompt | replay | exact chunk path, round 1 `f0f0671` | final | x replay |
|---|---|---|---|---|
| 1k | about 67 tok/s | 220 tok/s (4.66 s) | 524 to 531 tok/s (1.93 to 1.95 s) | 7.9 |
| 8k | about 67 | 201 (40.7 s) | 428 to 432 (18.9 to 19.1 s) | 6.4 |
| 32k | about 67 | not run | 246 to 247 (132.5 to 133.4 s) | 3.7 |

Frozen predictions (`70de96a`): tier 1k 350 to 900 HELD (524); tier 8k 330 to 850 HELD (428);
tier 32k 250 to 700 MISSED LOW by a hair (246); every cell above 3x replay HELD; G1 20/20 HELD only
after design item 4 was FALSIFIED: the dense chunk attention and chunk delta scan both reorder
sums (3 of 23 and 4 of 23 prompts diverged) and had to be replaced. Resident 8k and 32k cells
cannot be run: the 21 GB pack leaves 0.08 GB of MAX's memory pool at tmax 10240 (out of memory)
and at 1k ran 51 tok/s against tier's 220, so resident is an invalid speed arm on this card.

Round 1b, asked for by the maintainer mid-lane (each step token-equal to the exact baseline):

| step | 1k | 8k | commit |
|---|---|---|---|
| exact chunk path | 4.66 s | 40.7 s | `f0f0671` |
| shared q8 projection loads, 8 rows per thread | 3.82 | 33.8 | `f084880` |
| experts grouped by token sort, 8 pairs of one expert per thread | 2.94 | 24.5 | `f084880` |
| grouped q6k down | | | `426e622` |
| tier copy overlapped with compute on a second stream | 1.89 to 2.45 | 15.1 (with WMMA attention) | `488ed55` |
| WMMA attention above 256 tokens | REJECTED: agreement 61, 63, 64 of 64, mean 97.92% vs the 99% bar set before the run, control 64/64 | | `c5b664a` |

Profile before grouping (1150 rows): routed experts 62% of device time (gate/up 1.44 s, down
1.07 s), projections 0.57 s, delta scan 0.30 s, attention 0.05 s. After: gate/up 0.54, down 0.38.
The exact attention kernel is the long-context limiter now (32k: 133 s vs 84 s with WMMA).

## Design, as built

- `kernels/moe_rows.mojo`: every MoE phase is a sibling of the m=1 kernel with the same dot
  helper and summation order; shared-load variants feed 8 comptime-indexed accumulators from one
  weight load; (token, k) pairs are sorted by expert on the device, dots are written per pair and
  the top-k sum runs afterwards in k order. `kernels/test_moe_rows.mojo`: 16 outputs bit-exact.
- `amar_ssm_delta_rows`: one-launch delta scan in the decode step's per-column order, bit-exact
  over 37 rows through the 9-slot ring (`kernels/test_ssm_rows.mojo`); the dense chunk scan
  differs from the decode step in 131837 of 151552 outputs.
- Attention in the chunk is `amar_attn_decode` over rows (exact). `BARO_PF_ATT=wmma` and
  `BARO_PF_SSM=chunk` keep the dense kernels as opt-in.
- Tier mode: a chunk touches nearly all 256 experts of a layer, so prefill bypasses the 64-slot
  cache: one layer's experts are staged from the pinned store into one of two VRAM slots
  (0.52 GB each), layer L+1 copying on a second `DeviceStream` while layer L computes, ordered
  by events. Rejected: per-row `prepare` (a host sync per row per layer) and in-kernel zero-copy
  (PCIe once per token, expert and row). `BARO_PF_OVERLAP=0` keeps one stream.
- Bug fixed on the way: `amar_rmsnorm_cast2` writes its f32 copy for row 0 only; the rows
  sibling writes every row, the decode kernel is untouched.

## Step 0

ATOM `f9127a88e755` -> `d07089373a6d` (150 commits; the 13 on MoE paths are EP/TP transport,
capacity and new-model support, none change the intra-GPU sort, grouped GEMM indexing,
router-weight point or top-k order; nothing gfx11). rocMLIR `518955cab3ec` -> `9d49902e3b8f`
(40 commits, tuning driver and gemm+gemm quick-tune lists, nothing gfx11-specific); not rebuilt
and not used as an oracle: the oracle is the repo's own m=1 kernels compared in bytes. The
rocMLIR install at `~/Archive/migraphx-session-20260629/rocmlir-install` holds libraries only.
`tools/rocm-doctor`: PCIe root-port check added, two wrong existing checks fixed (env read via
`< /proc/self/environ`, file-valued `MIGRAPHX_PROBLEM_CACHE`), exit 0 on this boot.

## Deviations and open items

- The gate sequence was stopped once by the maintainer to build round 1b first; the claimed gates ran on
  the final tree after a passing `bench/preflight.sh`.
- G2 runs 2 repeats per cell, not the protocol's 3, and resident at 1k only (time, and the
  memory finding above).
- One unexplained flake: `q8d_rows_add` in `test_moe_rows` failed 8 of 10 runs in one window and
  passed every run before and after (about 20), no cause found; the kernel it tested is no longer
  used by the engine, the diagnostic print stays in the test.
- Coordinator findings outside the lane: the primary `.work/` was recreated empty 2026-09-19
  03:07 (fixtures and all cited receipts lost; `engine-pack-q4` relinked to the cached pack of
  the real Qwythos gguf); R6.3 (`0931e90`) on lane-r63 still has a frozen prediction and no
  numbers; `lane-merge lane-r63` fails only on its 8 lost receipt paths.
- GPU queue, last day, all clients: 89 jobs, 57 ok, 21 failed, 11 cancelled, 2164 busy minutes
  (1981 of them a resident whisper server). This lane's failures: one out-of-memory (resident at
  tmax 10240), one empty-stdin job, the divergence runs that found the inexact chunk kernels.
- Next lever, not started: an exact attention with the decode summation order that scales at
  32k and beyond, then a default flip after G3.
