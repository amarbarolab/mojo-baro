# Lane MOE-LOADER: load the qwen35moe pack, W3's loader half (sonnet)

You run one bounded step, in parallel with another lane. Read this whole file before touching anything.

## Where

- Worktree: `$HOME/Projects/mojo/mojo-baro-lanes/MOE-LOADER`, branch `lane-MOE-LOADER`. `cd` there
  first. Never edit `$HOME/Projects/mojo/mojo-baro` (main) or any other worktree.
- Plan for context (read it, you implement only the part below):
  `$HOME/Brain/mojo/mojo-baro/briefs/2026-09-11-moe-engine-wiring.md`.
- Repo rules: `CLAUDE.md` in the worktree root. They apply to every commit.
- The pack already exists, built by the other lane: `.work/moe-w1/pack` (`index.txt` plus `pack.bin`,
  733 tensors, 19.56 GiB). Do not rebuild it. Symlink or point at
  `$HOME/Projects/mojo/mojo-baro-lanes/MOE/.work/moe-w1/pack` read-only.

## Ownership, so two lanes never touch one file

- **You own:** `serve/harness.mojo` (its `load_pack` path, line 41), a new `serve/moe_pack.mojo` module if
  you want one, and your own test or probe files.
- **You must NOT touch:** `kernels/moe.mojo`, `serve/window.mojo`, `serve/engine.mojo`,
  `tools/engine-pack.py`, `bench/moe-protocol.md`. Another lane owns those right now. If your step seems
  to need one of them, stop and write the report instead.

## The step

Make the qwen35moe pack loadable and its layout resolvable, without any MoE kernel and without wiring
decode.

1. Extend the pack loader to the qwen35moe index: every tensor resolved by name to its dtype, byte offset
   and element count, with Q4_K, Q8_0, Q6_K and F32 all recognised. The existing q4/q8 packs must keep
   loading byte-identically.
2. Give the loader a lookup that answers, for a routed or shared expert tensor: layer, expert id,
   projection and row range, plus the byte offset and the Q4_K superblock geometry of that row range. The
   index carries `name dtype offset n_elem` already, so this is the semantic layer over it. Nothing may
   infer layout from lexical tensor order.
3. Size the profile-dependent host state from the selected model profile (KV pool and SSM state slots,
   qwen35moe: H 2048, QF 8192, KV 512, 40 layers, 30 SSM, 10 attention, HD 256, NKVH 2). Read the profile;
   do not hardcode a second copy of these numbers.

## Gate (preregister it first, then run it)

Write the prediction and the gate into `bench/moe-loader-protocol.md` and commit that BEFORE you build.
Then:

- A standalone probe binary (yours, under `tools/` or `kernels/`, built AOT into `.work/`) opens the real
  `.work/moe-w1/pack`, resolves every one of the 733 tensors, and prints: tensor count, the sum of the
  resolved sizes against the pack's real byte size, and for `blk.0` the resolved offsets of the router, the
  shared expert and expert 0 of each projection. Every number is read back from the pack, never assumed.
- Any tensor that fails to resolve, or a size sum that disagrees with the file, is a FAIL. Report it, do
  not paper over it.
- Regression: `./run-tests.sh` green, and `bench/force-ab.sh` between a main-built engine and yours at
  20/20 prompts, so the existing qwen35 path is provably untouched.
- Every GPU command runs through `gpu-wait run ... -- <cmd>`, never bare. Re-read the head of
  `$HOME/Brain/mojo/mojo-baro/whiteboard.md` before each GPU launch; if it records a hold, stop.

## Rules

- Commits: conventional subject, why in the body, numbers from the gate. No `Co-Authored-By` or any model
  attribution line. Never push; there is no remote.
- No em dashes anywhere.
- Nothing is done until its gate ran and passed, and you name the gate and its numbers in the same
  sentence. "It should work" is not done.
- Blocked, or a gate fails twice: stop, write the report, wait. Never substitute a different approach.

## Deliverable

Write `$HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-LOADER-report.md`: what you changed, the
commits, the gate output with its numbers, and anything the wiring lane must know about the layout you
exposed. Your final chat reply is only:
`written to $HOME/Projects/mojo/mojo-baro/exchange/lane-MOE-LOADER-report.md`.
