# LatentOS cross-node fork (P1 design items 2 and 4, gates 2 and 4)

the maintainer, 2026-09-17: "launch a fable 5.1 on LatentOS", scope chosen: **cross-node fork**. Builder: fable.

## Read first

- `docs/PLATFORM-PLAN.md` section P1 (the whole section; this lane is items 2 and 4, gates 2 and 4).
- `exchange/lane-P1-report.md` and `docs/P1-STATE-API.md`: export/import, LAT1 streaming, identity
  409, `GET /v1/state` are merged on main (`6c54d1d`). Build on them, do not re-implement.
- `~/AMDHQ/docs/design/latent-os/` (03 latent IPC, 04 identity, 10 checkpoint API) and the receipts the
  plan names: E12/E12-long, E15 (llama.cpp state bridge), B4-mini (`bench/b4-cross-host.sh`).
- `bench/PROTOCOL-RULES.md` (P1 parameter read-back, P4 20-prompt medians, P6 harness before kernel),
  `bench/coldcache-protocol.md` style preregistration: freeze predictions by commit before timed runs.
- Memory: `exchange/lane-P4-report.md`. The iGPU arm showed request-state bleed at p06; any node on the
  iGPU inherits that open defect.

## Where

Worktree `~/Projects/mojo/mojo-baro-lanes/fork`, branch `lane-fork` from main. Pathspec commits only.
`serve/src/bin/router.rs` and router modules belong to team A (P0b in flight on `lane-team-a`): do not
edit them. Fork targets a node by address in this lane; router placement comes after P0b merges.
Kernel files carry zero comments (repo CLAUDE.md). Every GPU process through `gpu-wait run ... --timeout`,
re-read the whiteboard head before each GPU launch. One MAX engine process holds about 22 GB, so two
engines cannot share the XTX at once: design the rig around that (time-sliced nodes, or the XTX plus a
llama.cpp node), and write the choice down before building.

## Build

1. `/v1/fork` gains `target` (node address): export on the holder (int8 state format where the engine
   allows it), stream to the target's `/v1/state/import`, answer from the target. Identity check on
   import stays the 409 path.
2. The E15 bridge forward direction: LAT1 KV state to a llama.cpp slot file (about 150 LOC, the reverse
   `tools/llama-slot-to-state.mojo` exists), so a llama.cpp node (desktop, aihq-lab CPU, the phone app
   later) continues a conversation our engine started.

## Gates (receipts under `.work/fork/`, report `exchange/lane-FORK-report.md`)

- **Gate 2:** cross-node through the B4-mini veth rig at 100 Mbit, 1 Gbit, 10 Gbit: fork-on-target
  token ids equal single-node ids on the 20-prompt set (teacher-forced agreement where greedy equality
  is not the right bar, per repo CLAUDE.md), and for a 32k prefix the payload arm beats re-prefill
  (the recipe arm) at 1 Gbit and above. Predictions frozen by commit before the timed run.
- **Gate 4:** E15's three models continue from our exported state through the bridge with the first 32
  tokens identical.
- **Kill line (plan):** an identity miss outside the documented E14 one, or a cross-node fork slower
  than re-prefill at 1 Gbit for 32k.

## Rules

Global CLAUDE.md binds: no em dashes, loud failures, DONE only with the named receipt, reference arm
treated as an arm (read-back and cold-cache rules apply to it), no self-tagging commits. GPU jobs:
CPU preflight first, `--timeout` always. Reply to the coordinator pane `w82:pC` only with
"written to <path>".
