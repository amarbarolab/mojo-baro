# Lane MOE-BASE: the llama.cpp reference arm for W4 (codex, gpt-5.6-luna)

You measure ONE side of a comparison: llama.cpp decoding RegesCore-35B. You do not touch mojo-baro's
engine, and you do not measure our side. Another lane is still building it.

## Where

- Worktree: `$HOME/Projects/mojo/mojo-baro-lanes/MOE-BASE`, branch `lane-MOE-BASE`. `cd` there
  first. Never edit `$HOME/Projects/mojo/mojo-baro` (main) or any other worktree.
- Model: `$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf` (20.9 GB, Q4_K_S).
  Read-only. Never copy it, never convert it.
- llama.cpp: `$HOME/llama.cpp`, built at `ca3d5a3e1` (2026-08-28), binaries
  `build/bin/llama-cli` and `build/bin/llama-bench`.
- Plan context: `$HOME/Brain/mojo/mojo-baro/briefs/2026-09-11-moe-engine-wiring.md`, step W4.
- Repo rules: `CLAUDE.md` in the worktree root, and `bench/PROTOCOL-RULES.md`. They bind you.

## Ownership, so three lanes never collide

- **You own, and create:** `bench/moe-baseline-protocol.md`, `bench/moe-baseline.sh`, and your own
  receipts under `.work/moe-base/`.
- **You must NOT touch:** anything else. In particular `bench/moe-protocol.md`, `kernels/*`,
  `serve/*`, `tools/*`. Two other lanes hold those right now. If your step seems to need one of them,
  stop and write the report instead.

## The step

Produce the reference-arm receipt W4 needs, so that when our engine can decode this model the two sides
are comparable without argument.

1. **Preregister first.** Write `bench/moe-baseline-protocol.md` and commit it BEFORE any timed run:
   what you will measure, the exact invocation, how many warmups and repetitions, and what would make
   the run void. No predictions about our side; this is a measurement, not a contest.
2. **Record the arm.** llama.cpp commit and build flags (how it was configured, backend, ROCm version),
   GPU and driver, model path and its sha256 (first 16 chars is enough), context length, batch, threads,
   sampling parameters, and the number of GPU layers. Every one read back from the running system or
   the binary's own output, never assumed. `bench/PROTOCOL-RULES.md` P1: no receipt, no arm.
3. **Measure decode.** The 20 prompts in `bench/mtp-prompts/` (the `.txt` files are the text, the
   `.tokens` files are our tokenised form; say in the report which you fed llama.cpp and how). Same
   generation length on every prompt, stated in the protocol. Report per-prompt tok/s, then the median
   AND the min-max range, not just the median.
4. **Record prefill separately** if llama.cpp reports it (prompt eval time), same discipline.
5. **The reference arm is an arm** (CLAUDE.md): it gets the same care as ours. State whether weights were
   cold or warm, and do not let a warm cache stand in for a cold one silently.

## Rules

- **GPU: every run goes through `gpu-wait run --priority 20 --preemptible --vram 22 -- <cmd>`.** Low
  priority and preemptible are deliberate: two other lanes are on the critical path and their gates must
  win the card. Never run a GPU command bare. Re-read the head of
  `$HOME/Brain/mojo/mojo-baro/whiteboard.md` before EVERY launch; if it records a GPU hold, stop
  and write the report.
- Commits: conventional subject, why in the body, numbers in the body. No `Co-Authored-By` or any model
  attribution line. Never push; there is no remote.
- No em dashes anywhere.
- Nothing is done until the run happened and you name the numbers. A plan to measure is not a measurement.
- Blocked, or a run voids twice: stop, write the report, wait. Never substitute a different model,
  quantisation or prompt set to make a number appear.

## Deliverable

Write `$HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-BASE-report.md`: the arm receipt table, the
per-prompt numbers, median and min-max, the verbatim invocation, and anything that would make the
comparison unfair if our side did it differently. Your final chat reply is only:
`written to $HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-BASE-report.md`.
