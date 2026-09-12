# Lane PROFILE report

Branch `lane-PROFILE` (worktree `~/Projects/mojo/mojo-baro-lanes/PROFILE`), rebased onto
main twice mid-lane (KATT `088c940`, then TOK `3c1f877`) as those landed. Final base: TOK's
merge. All three plan steps done: gen-profile module (1), recipe flags behind comptime ifs
(2), and per-target build+verify for all four remaining targets (3) — Llama-3.2-1B,
Qwen2.5-7B, Granite-4.2-3B, Lily-7B, in the plan's order.

## Result: all steps PASS, 4/4 targets verified against llama.cpp

## Gate

```
gpu-wait run --priority 50 --vram 20 --timeout 3600 -- ./run-tests.sh
```

| | exit | PASS lines |
|---|---|---|
| before (post-KATT/TOK merge, this lane's starting point) | 0 | 102 |
| after (this lane's final commit) | 0 | 102 |

No numeric floor given by the item; PROFILE added no new `run-tests.sh` binary (its gate is
the bit-identity checks below plus the per-target BARO_FORCE runs), so the count is
unchanged by design. Evidence: `.work/PROFILE-gate-final.txt`.

## Step 1+2 gate: bit-identical to today (spark2_5, default profile)

- `.work/spark/gate-text-profiled4.log` / `.work/spark/gate-chat-profiled4.log`: both frozen
  fixture gates PASS (64/64, 43/43) against `.work/spark/ref` / `.work/spark/chat-ref`.
- `.work/spark/ab20-baseline/` vs `.work/spark/ab20-profiled2/`: 20/20 `bench/mtp-prompts`
  greedy generations byte-identical between a rebuilt pre-change baseline and the
  profile-driven build.

## Step 3 gate: BARO_FORCE teacher-forced agreement vs llama.cpp, 20 prompts each

Full method, frozen predictions and per-target results in `bench/dense-protocol.md`
(sections "PROFILE step 3" onward). Summary:

| target | result | tok/s_gen | evidence |
|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | PASS, 20/20 (95.3-100%) | 451-458 | `.work/dense/llama32-1b-run2/` |
| Qwen2.5-7B-Instruct-Q4_K_M | PASS, 20/20 (96.9-100%) | 96.8-97.8 | `.work/dense/qwen25-7b-run/` |
| Granite-4.2-3B-BF16 | PASS, 20/20 (98.4-100%), after 1 repair round | 164.5-169.0 | `.work/dense/granite42-3b-run2/` |
| Lily-cybersecurity-7B-Q6_K | PASS, 20/20 (96.9-100%) | 94.1-95.0 | `.work/dense/lily7b-run/` |

Chat-completions smoke: **not attempted, reported as N/A per the plan's escape hatch**.
`serve/src/engine.rs` drives a long-running request/response protocol that only
`serve/engine.mojo` implements; `serve/spark.mojo` is a one-shot CLI with no such protocol,
and wiring one up is outside this lane's file list.

## Flags (read first)

- **Two flags found outside the plan's step-2 list, both required for correctness, both
  fixed before any target ran to completion:**
  - **HAS_GATE**: spark2_5's per-head attention output gate has no analogue in
    llama/qwen2/granite. Without a way to disable it, every non-spark target's attention
    output would be silently multiplied by sigmoid(0)=0.5. Added as a fifth comptime flag,
    same shape as the four the plan named (`cdfc2eb`).
  - **Activation**: `amar_gemv_q8`'s fused FFN-up epilogue was hardcoded to spark2_5's GELU.
    All four remaining targets use SiLU. New EPI=3 branch, selected by `ACTIVATION_GELU`
    (`8615841`).
- **One wrong assumption, caught by a failing run, not by inspection**: the first
  `tools/gen-profile.mojo` put Granite in the NeoX rope group. `granite-4.2-3b` scored 40/64
  average forced agreement (well under the 90% floor) on the first attempt.
  `llama_model_rope_type` in llama.cpp's `llama-model.cpp` groups `LLM_ARCH_GRANITE` with
  `LLM_ARCH_LLAMA` under "normal RoPE" (interleaved pairs), not with `LLM_ARCH_QWEN2`'s
  half-offset group — the table was assembled from recollection instead of checked against
  the source for this one arch. Fixed (`31f1830`), re-ran clean (98.4-100%). One repair
  round total across the whole lane.
- **GRANITE_MULT's embedding_scale/residual_scale are wired nowhere** (design deviation from
  a literal reading of the plan's step-2 flag list, flagged in `bench/dense-protocol.md`
  before the granite run). Both multiply the residual stream and can't be handled by simple
  parameter substitution the way `ATTN_SCALE` was; the correct place for them is pack-time
  weight scaling in `tools/engine-pack.py`, not a runtime kernel branch, and that packer
  didn't exist at the time this decision was made. **Not exercised**: the one Granite
  checkpoint available has both multipliers at 1.0 (no-ops), so this is a real gap for a
  future Granite checkpoint that sets them, not something this run could have caught either
  way. `LOGIT_SCALE` is correctly never wired: scaling logits by any positive constant can't
  change an argmax, and this engine only ever greedy-decodes.
- **`tools/engine-pack.py --dense`** (new mode, alongside the untouched pre-existing modes)
  fuses the three separate GGUF Q/K/V tensors into the one `[QKV, H]` matrix the fused-QKV
  kernel expects, reusing Ornith's K-quant dequant path. Refuses a GGUF with a per-head
  attention gate outright (spark2_5 stays on `tools/spark-pack.py`).
- **Two stale `git stash` entries** left in this worktree from mid-lane checkpointing
  (`PROFILE-spark-wip-for-baseline-build`, `PROFILE-spark-wip-pre-rebase`) — both already
  applied and their content is in history; the auto-mode classifier blocked `git stash drop`
  as an irreversible action. Safe to drop, left for the maintainer or an explicit ask.
- Files touched outside a literal reading of the plan's PROFILE file list: none beyond what
  the plan itself names (`tools/gen-profile.mojo` instead of `.py` per the maintainer's instruction
  mid-lane; `bench/dense-run.sh`, new, the step-3 verify harness).

## Commits on `lane-PROFILE` (8, in order)

- `17cecf5` tools(gen-profile): emit a comptime profile module from GGUF metadata
- `19d8120` serve(spark): source recipe constants from a generated profile module
- `cdfc2eb` serve(spark): add HAS_GATE recipe flag, a gap the plan's flag list missed
- `8615841` tools(engine-pack): add --dense mode for plain llama/qwen2/granite packs
- `90e0154` bench(dense): Llama-3.2-1B PASS, 20/20 prompts >=95% forced agreement
- `7219bca` bench(dense): Qwen2.5-7B PASS, 20/20 prompts >=96.9% forced agreement
- `31f1830` tools(gen-profile): fix Granite's rope type, wrong since the first table
- `d178d1f` bench(dense): Lily-7B PASS, 20/20 prompts >=96.9% forced agreement

## Not done / next

- Pack-time embedding_scale/residual_scale folding for a Granite checkpoint that actually
  sets them (none available to test against right now).
- Chat-completions smoke, N/A as explained above.
- `docs/KERNELS.md` regen: KATT's report already noted `--check` passes without it and a
  regen would also touch pre-existing drift unrelated to this lane; left as-is, same call
  KATT made.
