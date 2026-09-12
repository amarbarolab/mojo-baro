# Lane KSAMP — build brief

Worktree: `$HOME/Projects/mojo/mojo-baro-lanes/KSAMP` (branch `lane-KSAMP`). Plan:
`$HOME/Brain/mojo/mojo-baro/briefs/2026-09-11-chat-engine-next.md`, item **KSAMP**
(section "Lane items") — build exactly that item, nothing else. The interface in the plan is
fixed: another lane (CHAT) targets it.

Read first: the `mojo-syntax` and `mojo-gpu-fundamentals` skills, then the
`mojo-nightly-lane-builder` skill (~/.claude/skills/.user/mojo-nightly-lane-builder/SKILL.md):
its error table and bug-class list are the review checklist. Kernel files follow the repo
convention in CONTRIBUTING.md (no comments or docstrings in `kernels/matmul*.mojo` /
`elementwise.mojo` style kernels; rationale goes in commit messages and docs).

Standing rules:
- Preregister the prediction + gate for KSAMP in `bench/chat-protocol.md` BEFORE building.
- GPU only through the waiting room: `gpu-wait run --vram <GB> -- env VAR=val <cmd>` (env
  vars INSIDE the `--` command, never before `gpu-wait`). Before every GPU launch, re-read
  line 10 of `~/Brain/mojo/mojo-baro/whiteboard.md`: this lane is allowed to gate through
  gpu-wait during the current hold (the maintainer, 2026-09-11); if that line says otherwise, stop.
- Gate: `./run-tests.sh` (includes the kernel census) plus `kernels/test_sample.mojo`,
  output captured to `.work/KSAMP-gate.txt`; report the test count before/after.
- Commit each passing sub-step on `lane-KSAMP`; three repair attempts, then write
  `.work/KSAMP-failure.md` and stop; never edit or weaken a test to pass; files outside
  `kernels/sample.mojo`, `kernels/test_sample.mojo`, `bench/chat-protocol.md` and the census
  wiring are out of scope — stop and report. Do not touch the greedy path or the engine.
- Commit messages: conventional subject + why-body, NO attribution trailers (no
  Co-Authored-By, no 'Generated with', no session links).

Report to `$HOME/Projects/mojo/mojo-baro/exchange/lane-KSAMP-report.md`: gate command,
exit code, test count before/after, the chi-square numbers, the per-token time at the real
vocabulary size, evidence paths, commits. Then push
`DONE -> $HOME/Projects/mojo/mojo-baro/exchange/lane-KSAMP-report.md`.
