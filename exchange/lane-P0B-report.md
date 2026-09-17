# Team A P0b report

Date: 2026-09-17
Item: P0b CPU router skeleton and gate 3 fixture
Status: **BLOCKED: gate 3 does not pass**

## Landed

- `c411fd5` adds `serve/src/bin/router.rs` with the named proxy phases, sent-aware
  retry state, 3-failure/2-success health hysteresis, pending accounting, node-info
  without the P1 locality term, port-probe adoption, and CPU proxy scaffolding.
- `814244f` adds `mdns-sd` advertisement/browse support and the router binary's
  CPU smoke coverage.
- `63845c5` publishes the PAIR UUID and the PAIR node-info fields
  (`hostUuid`, `GPUs`, `telemetryValid`, `msSince`).
- `ca11fe1` uses PAIR's fixed `_nvpair-node._tcp` SRV port `14318`; the node-info
  port remains in the `ni` TXT field, and adds `BARO_ROUTER_ADVERTISE_IP` plus
  `BARO_ROUTER_MDNS_HOST` overrides for real-interface bring-up.
- `b4820c3` adds `bench/p0b-router-gate.sh`. The fixture starts
  `nvpair-node-scanner` before the router, uses an isolated output directory,
  checks `/v1/node-info`, requests `discovery:get-nodes`, asserts the exact UUID
  and `ni` port, and terminates both processes.

## Verification

CPU checks for the router passed:

```text
cargo test --bin router                  4 passed
cargo clippy --bin router -- -D warnings  exit 0
cargo build --release --bin router       exit 0
```

The gate dry-run passed and stopped before process launch:

```text
GATE_DRYRUN_OUT=.work/team-A/codex/p0b/pair-gate-r8 \
  ~/iTools/bin/gate-dryrun --arm ... --expect cpu_only=true --expect port=18113 \
  --stop 'FAIL.*GPU' -- bench/p0b-router-gate.sh ...
exit 0; PASS gate-dryrun; rc=97 at the intentional FAIL GPU stop
```

## Gate 3 receipt

The requested live CPU gate was run as:

```text
bench/p0b-router-gate.sh
exit 1
```

Receipt directory: `.work/team-A/codex/p0b/pair-gate/`

- `/v1/node-info` passed for UUID
  `00000000-0000-4000-8000-0000000000b3`; `state_locality` was absent.
- PAIR's `discovery:get-nodes` response did **not** contain that UUID or its
  `ni=18103` entry. It contained the stale UUID `00000000-0000-4000-8000-000000000095`
  and the local `hercule` entry instead.
- `contract.log` contains the assertion and exit; `node-info.json`,
  `pair-frames.json`, `router.stderr`, and process exit files preserve the
  evidence.

This is an honest gate failure, not a pass claim. The current environment's
PAIR scanner sees older mDNS records, while the router's own mDNS browse and
the HTTP node-info endpoint see the new advertisement. Gates 1, 2, and 4 remain
deferred until this gate passes and the P1 contract is available to the router.

## Suite and readiness

Sonnet reported a clean lane suite in room A at line 223 (`./run-tests.sh`, exit
0, 146 PASS), but no corresponding suite log exists in this worktree. The only
local full-suite receipt is the earlier P0a receipt
`.work/team-A/codex/p0a-suite.log`, so P0b is **not merge-ready**.

## Receipts

- `.work/team-A/codex/p0b/receipt.txt`
- `.work/team-A/codex/p0b/mdns-receipt.txt`
- `.work/team-A/codex/p0b/pair-gate/contract.log`
- `.work/team-A/codex/p0b/pair-gate/pair-frames.json`
- `.work/team-A/codex/p0b/pair-gate/node-info.json`
