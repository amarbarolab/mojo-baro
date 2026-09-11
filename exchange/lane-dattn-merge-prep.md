# lane-dattn: merge prep (2026-09-11)

Nothing is merged. This file is the checklist for the maintainer's go.

## State

- `lane-dattn`: 11 commits over the branch point `d8fbb44`; code complete at `7fd5680`, this checklist is the commit after it.
- `main` at `09731a5`, 7 commits over the branch point: E12 bench scripts and results only.

## Pre-merge checks (done)

| check | result |
|---|---|
| main touched any lane file | no (empty intersection of changed-file lists) |
| main touched engine / kernel / tool files since the branch point | no (`git diff --name-only d8fbb44 main -- serve kernels tools run-tests.sh shim` empty) |
| trial merge `git merge-tree --write-tree main lane-dattn` | clean, tree `b12aa7a`; its `serve/ kernels/ tools/ run-tests.sh shim/` equal the lane's, so every lane gate describes the merged engine |
| merge gate on the lane (`tools/merge-gate.sh`) | builds, `run-tests.sh`, `ci-checks.sh`, test_prefill PASS; one-shot q4 64/64 (136.6 tok/s_gen), q8 64/64; p0512 prefill fail word 0; mtp k=2 identity 20/20; server suite 16 PASS / 0 FAIL, ALL PASS (cargo-test 9 passed); gate log `.work/merge-gate.txt`, exit 0 |
| wiring stint `.work/dattn-wire/w1` | mega-gate ALL PASS, split identity 64/64, 20-prompt identity 20/20, see `bench/dattn-wire-protocol.md` |
| re-embed tool | fixed (`7fd5680`): re-embedding from the existing BARO file no longer duplicates `baro.kernel.*` keys; `tools/test_gguf_embed.py` PASS on real tensor data |

Other branches ahead of `main`, untouched and not part of this merge: `lane-attn` (+3, must not
merge as it stands: it reintroduces the page arithmetic `1e7ab91` fixed), `lane-b-4x4` (+1),
`lane-mrow-kernel` (+2).

## Merge steps

In the main checkout (`~/Projects/mojo/mojo-baro`):

1. `git merge --no-ff lane-dattn` (expected clean; the trial tree is `b12aa7a`).
2. Post-merge gate, one gpu-wait job: `tools/merge-gate.sh`, then the fingerprint on the merged
   engine: `tools/isa-receipt.py .work/engine --dump DIR` and `isa-loops DIR/co78.s --match
   amar_mega_token --top 4`. Expected class 124 / 79 / 79 / 59, 0 spills (sources identical to
   the lane build); a different ticket means the short-context number must be re-measured
   before it is quoted.
3. 20-prompt A/B, merged vs pre-merge main (`bench/ab-prompts.sh`, `AB_ENGINE_B`), in the same
   stint: the P4 bar. Lane receipt 127.96 -> 136.41, identity 20/20.
4. Re-embed the self-describing model (writes a new 18.4 GB file, never touches the source):
   `tools/gguf-embed.py ~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-e3948ba.gguf ~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-<merge>.gguf $(tools/embed-files.py)`
   then `tools/gguf-closure.sh <new file>` under gpu-wait (builds the engine from the embedded
   sources, gates on reference tokens). The current file embeds `e3948ba`, 223 commits behind
   main, 33 of them in embedded sources (including `1e7ab91`): it is stale regardless of this
   lane. The old file stays until the maintainer deletes it (disk: 319 GB free).
5. Docs: `docs/M5-PLAN.md` names the BARO file `...-9b8a399.gguf` (on disk it is `e3948ba`);
   point it at the new file. `docs/BASELINE.md` champion row only if step 3 confirms.
6. Board, baton, Brain note.

## After the merge (the maintainer's step)

Worktree and branch removal is blocked for the agent. One line once the merge is on `main`:

    git -C ~/Projects/mojo/mojo-baro worktree remove ~/Projects/mojo/mojo-baro-lanes/dattn && git -C ~/Projects/mojo/mojo-baro branch -d lane-dattn
