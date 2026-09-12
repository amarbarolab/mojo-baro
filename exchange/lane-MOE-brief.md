# Lane MOE: RegesCore-35B (qwen35moe) in the engine, W0 to W4 (codex, gpt-5.6-luna)

You are the only agent on this lane. Run the steps in the order below and commit each one as it lands.

## Where

- Worktree: `$HOME/Projects/mojo/mojo-baro-lanes/MOE`, branch `lane-MOE`. `cd` there first; never edit
  `$HOME/Projects/mojo/mojo-baro` (main) directly.
- The plan: `$HOME/Brain/mojo/mojo-baro/briefs/2026-09-11-moe-engine-wiring.md`. Read it in full,
  including "Traps already known" and the "Skills" table. Read every skill file the table names before
  your first line of code.
- Repo rules: `$HOME/Projects/mojo/mojo-baro/CLAUDE.md` (the same file is in the worktree). They apply
  to every step.

## Order

W0, W1, W2, W3, W4, all in this one lane. W3 starts only after W2's parity gate passes. W5 is out of scope.

## Rules (these are not optional)

1. Before building each step, preregister its prediction and gate in `bench/moe-protocol.md` and commit that.
   Then build, gate and commit with the numbers.
2. Every GPU command (build tests that run on the GPU, benches, the engine, llama.cpp) runs as
   `gpu-wait run [--priority N] [--vram GB] [--timeout S] -- <cmd>`, never bare. Before every GPU launch,
   re-read the head of `$HOME/Brain/mojo/mojo-baro/whiteboard.md`. If it records a GPU hold, do not
   launch; write the report and stop.
3. Identity gates are teacher-forced agreement (`BARO_FORCE`), never greedy token equality. On every step,
   check the Qwythos regression: `bench/force-ab.sh` between a main-built engine and your build, plus
   `./run-tests.sh` green.
4. Commits: conventional subject plus a body that explains why, with the step's numbers. No
   `Co-Authored-By` or any model attribution line. Never push; there is no remote.
5. No em dashes anywhere (code, comments, commits, docs). Kernel files `kernels/matmul*.mojo` and
   `kernels/elementwise.mojo` carry zero comments.
6. If you are blocked, or a gate fails twice, stop and write the report. Never substitute a different
   approach, model or target silently.
7. Nothing is "done" until its gate ran and passed; name the gate and its numbers in the report.

## Deliverable

Write `$HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-report.md`, and update it after each step:
the step, its commits, gate results with numbers, and which skill files you read. Your final chat reply is
only: `written to $HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-report.md`.
