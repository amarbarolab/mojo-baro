# Lane dattn: generalised decode attention (brief, 2026-09-11)

You own this lane end to end. Work in this worktree on branch `lane-dattn` (based on `main`).

## Read first, in this order
1. `bench/dattn-protocol.md`: the frozen protocol. Scope, arms, gates, predictions and stop rules there are binding; do not edit the Frozen predictions section.
2. Repo `CLAUDE.md` and `bench/PROTOCOL-RULES.md` (P1-P6). Kernel files carry zero comments and zero docstrings.
3. `kernels/attn.mojo` (today's `amar_attn_decode`, `attn_head_span`, `kv_off`), `kernels/mega.mojo` `attn_phases` (the split-across-blocks path), `kernels/test_attn_block.mojo`.
4. `bench/ggml-harness/` (the reference arm) and `results/ggml-harness-headroom-2026-09-11.md` (the R receipts).
5. Skills: `kernel-parity`, `inference-kernels`, `mojo-syntax`, `mojo-gpu-fundamentals`.

## Deliver
- Kernel: generic decode attention over comptime `HD`, `NQH`, `NKVH`, `KVT`, runtime scale, paged `kv_off` layout, split + combine for long caches.
- Numerics test (gate 2) with an fp64 numpy reference in `tools/`, and gate 3 bit-identity at the shipped Qwythos instantiation.
- Standalone cold-cache bench for arm O with the same rotation rule as `op_bench` and a full parameter echo.
- Same-stint timed runs of R and O on S1-S3 plus the S1 scaling receipt, device time from rocprofv3.
- Fill the protocol's `## Result` section with receipts (P3), and a report at `exchange/lane-dattn-report.md`: verdict (land / close / between), numbers, what was read back and from where, open questions.

## Rules
- Every GPU run goes through the queue: `gpu-wait run [--vram GB] -- <cmd>`. gpu-wait drops the environment: pass settings as arguments, never exported variables.
- R is an arm: re-run `bench/ggml-harness/run.sh` on the three decode targets in the same stint as O; never compare against the old receipt alone.
- Commit on `lane-dattn` as each coherent unit passes its check, conventional subject + why-body. No `Co-Authored-By` or any model attribution line. No em dashes anywhere.
- Do not merge into `main`, do not touch `lane-attn`, do not push.
- Stop and write the report if a stop rule fires, or after two failed attempts at the same gate.
- "Done" means the gate or timed run passed and you name it; otherwise say UNVERIFIED and what is missing.

When finished, reply only: `written to exchange/lane-dattn-report.md`.
