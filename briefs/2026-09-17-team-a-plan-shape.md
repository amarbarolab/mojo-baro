# Team A conference: how do you two want the platform plan shaped?

You are one of two peers of equal standing on the same tree: `sonnet` (Claude Sonnet, pane `w8K:p1`)
and `codex` (pane `w8K:p2`). The coordinator is pane `w82:pC`. Repo: `~/Projects/mojo/mojo-baro`
(cd there first). This round is a CONFERENCE. No source edits, no builds, no GPU jobs. You read,
you talk, you write one agreed document.

## Why

the maintainer wants two mixed teams (one sonnet + one codex each) to build `docs/PLATFORM-PLAN.md`, both
members working over the same files. Before the coordinator writes the build plan, you two decide
the shape that lets BOTH of you work at maximum performance. The plan will be written in the shape
you ask for, and rules you insist on are adopted verbatim.

## Read first (once, in one burst)

- `docs/PLATFORM-PLAN.md` (the ten items and their gates)
- `CLAUDE.md` (repo rules), `docs/BASELINE.md` head, `bench/PROTOCOL-RULES.md`
- `$HOME/Brain/Skills/teamson/SKILL.md` (two peers on one tree: deadlock, invariants)
- `$HOME/Brain/Skills/herdr-ops/SKILL.md` section on codex gotchas (both of you: know your partner's limits)
- `~/iTools/harness/room/room.sh` header (your chat tool), `~/iTools/harness/lane-merge/lane-merge.sh` header (what a merge will demand)

## How you talk

The room lives in the Brain: `$HOME/Brain/mojo/mojo-baro/rooms/A/` (`room.md` chat, `cards.md` cards).
Always by absolute path, from the repo directory:

    ~/iTools/bin/room say A <your-name> "text"      # appends and pings your partner's pane
    ~/iTools/bin/room read A <your-name>            # what you have not seen yet
    ~/iTools/bin/room cards A
    ~/iTools/bin/room card A <your-name> "<match>" doing|done "<receipt, required for done>"

Content goes in the room, never in the terminal. A ping in your pane is only a pointer: run the
read command it names. Never idle waiting on a reply without having posted the question in the
room. If you are blocked on the coordinator: `~/iTools/bin/herd tell w82:pC "QUESTION: ..."` and
keep working on what does not depend on the answer. No em dashes anywhere (messages are refused).

## What to settle (turn each card as you close it)

1. **Strengths and limits, honestly.** Each of you states what you are fast and reliable at in this
   repo (Mojo 1.0, GPU gates under `gpu-wait`, HTTP surface in `serve/`, Python oracles, Go client)
   and what makes you fail (sandbox, context, tool gaps, long builds, ambiguity). Codex: your sandbox
   and skill-loading limits. Sonnet: yours.
2. **Item shape.** What does one plan item need to contain so you can build it without a question:
   files to touch, the gate command, the receipt path, the kill line, preflight steps, size limit?
   Give the template you want, filled in for P0a as the worked example.
3. **Split protocol on shared files.** How two of you edit the same files (`serve/engine.mojo` is
   the hot one) without collisions: who owns what, announce rule, commit cadence (pathspec only),
   per-agent build dirs under `.work/`, who runs the suite and when (one GPU, `gpu-wait`, the whole
   `./run-tests.sh` wrapped once).
4. **Review loop.** Do you review each other's diffs before commit, after, or at the gate? What does
   the reviewer check that the author cannot?
5. **Which items for which team, and in what order.** The coordinator's draft: Team A = P0a, P0b,
   P1 build; Team B = P4, P5a, P3a, P5b, P6, P3b; P2 kernels stay with the coordinator (fable); P1
   design spec comes from the coordinator. Say what is wrong with this.
6. **What the plan gets wrong.** Anything in `docs/PLATFORM-PLAN.md` that would make you fail or
   that you believe is false, with the file and line that shows it.
7. **What you need from the coordinator** (tools, fixtures, decisions) before the first build hour.

Disagreement is wanted. Do not defer to your partner to be polite; if you agree on everything in
two messages, you have not read the code.

## Deliverable

One file, written jointly: `exchange/2026-09-17-team-A-plan-shape.md`, sections 1 to 7 as above,
each with the agreed answer and, where you did not agree, both positions. Sonnet creates the file
with the section skeleton; each of you writes your own paragraphs; announce in the room before
editing a section your partner wrote. When both of you have signed the last line
(`signed: sonnet`, `signed: codex`), sonnet commits it by pathspec (no attribution lines) and runs:

    ~/iTools/bin/herd tell w82:pC "team A plan shape written to exchange/2026-09-17-team-A-plan-shape.md"

Then stop. Do not start building.
