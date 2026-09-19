# Lane MOEPF: batched prefill for qwen35moe (interim, 2026-09-19 21:40)

Status: round 1 (bit-exact chunk path) built and exploratory-verified; claimed gates NOT run.
the maintainer stopped the gate sequence at 21:35 to go straight to the speed round (1b). Everything
below marked UNVERIFIED has no claimed gate behind it.

Skills read in full before code: inference-kernels, gate-authoring, preregister-experiment,
llm-benchmark-method, prefix-cache-discipline, checkpoints. Branch `lane-moepf` in
`.work/lanes/moepf`, base `ff8e523` (lane-r63 head; `main` lacks `BARO_SEQ_CAP`), plus the
coordinator's ci fixes cherry-picked (`b6e40b6`, `318adaa`).

## Step 0

- ATOM `f9127a88e755` -> `d07089373a6d` (150 commits; 13 touch the MoE paths, all EP/TP
  transport, capacity or new-model support, none change the intra-GPU sort, grouped GEMM
  indexing, router-weight point or top-k order; table in `.work/step0/assist-1.md`). Nothing
  gfx11. rocMLIR `518955cab3ec` -> `9d49902e3b8f` (40 commits; tuning driver and gemm+gemm
  quick-tune lists touched, nothing gfx11-specific). Local rocMLIR CMake edit kept.
- rocMLIR not rebuilt: not used as an oracle. The identity oracle is the repo's own m=1 kernels,
  compared in bytes, which is stricter than a vendor GEMM could be. The install at
  `~/Archive/migraphx-session-20260629/rocmlir-install` holds libraries only, no `rocmlir-gen`.
- `tools/rocm-doctor`: PCIe root-port check added (link speed 16.0 GT/s and no kernel
  bandwidth warning). Two existing checks were wrong and are fixed: the env read through
  `< /proc/self/environ` came back empty (now `printenv`), and `MIGRAPHX_PROBLEM_CACHE` is a
  file, not a directory. Exit 0 on this boot, output in `.work/step0/rocm-doctor-run2.txt`.

## What was built (commits 70de96a, 87e99e8, 9627cc5, be3c96e, f0f0671)

- Protocol and predictions frozen before any run: `bench/moe-prefill-protocol.md` (`70de96a`).
- `kernels/moe_rows.mojo`: row-batched siblings of the m=1 MoE kernels, same dot helpers, token
  on the grid's y axis. `kernels/test_moe_rows.mojo`: 11 outputs bit-exact vs the m=1 kernels.
- `moe_prefill_forward` in `serve/window.mojo`, opt-in `BARO_PREFILL=1`, echo `BARO_PREFILL:`.
- Tier mode: one layer's 256 experts staged per chunk from the pinned store into one VRAM buffer
  (about 453 MB), slot cache untouched. Chosen because a 1024-row chunk touches nearly every
  expert; per-row prepare is a host sync per row per layer, and in-kernel zero-copy would cross
  PCIe once per (token, expert, row) thread. UNVERIFIED: tier mode has not run yet.
- Bug found by the first smoke and fixed: `amar_rmsnorm_cast2` writes its f32 copy for row 0
  only; `amar_rmsnorm_cast2_rows` writes every row, decode kernel untouched.

## Identity so far (exploratory, resident pack, EXPLORE=1 so the verdict is capped at UNVERIFIED)

| chunk attention | chunk delta scan | equal of 23 | divergences (prompt@token) |
|---|---|---|---|
| WMMA chunk | chunk_w | 20 | p09@52 p17@22 p1024@35 |
| decode kernel over rows | chunk_w | 19 | p06@20 p17@52 p0512@2 p1024@51 |
| decode kernel over rows | `amar_ssm_delta_rows` (new) | 23 | none |

Design item 4 of the protocol was falsified as written: the dense chunk attention and the dense
chunk scan both reorder sums. `kernels/test_ssm_rows.mojo`: the dense scan differs from the decode
step in 131837 of 151552 outputs; `amar_ssm_delta_rows` is bit-exact over 37 rows through the
9-slot ring, outputs and final state. Conv, l2 and gates match the decode arithmetic by source.
Receipts: `.work/moepf/id-res-explore`, `id-res-attexact`, `id-res-exact`.

Finding about the gate itself: only 9 of the 20 `bench/mtp-prompts` have the 17 tokens that
exercise prefill, the longest prefills 58 rows, and none crosses a chunk boundary. The identity
script adds a long set (128, 512, 1024 tokens) and reports NOT EXERCISED prompts by name.

## Speed so far: receipts, not claims

Engine's own `prefill_s` inside the identity run: 127 rows 0.51 s, 384 rows 1.53 s, 512 rows
2.04 s, about 250 tok/s, under the frozen 600 to 1500 band for resident 1k. If the baseline job
confirms it the round 1 speed prediction is FALSIFIED; mechanism to check first: every
(token, expert) pair re-reads its expert rows and every row re-reads the trunk weights.

## Open

- Gates 1 to 4 as claimed runs. Speed and needle scripts written by the assistant
  (`bench/moe-prefill-speed.{sh,py}`, `bench/moe-prefill-needle{.sh,-gen.py}`), CPU dry run only;
  needle length is approximated in words, port 8097 was taken at its test time.
- Round 1b, asked for by the maintainer 2026-09-19 21:35: shared weight loads across rows in the trunk
  projections and the experts (token sort), tier copy overlapped with compute, and above 256
  prompt tokens the WMMA attention and chunk scan under a teacher-forced agreement gate.
