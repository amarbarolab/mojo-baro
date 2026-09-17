# Team A P0b report

Date: 2026-09-17
Item: P0b CPU router skeleton and gate 3 fixture
Status: **gate 3 PASS** (2/2 clean runs). Gates 1, 2, 4 not started (blocked
on gate 3 until now; team A takes those next).

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
- `b7a23a6` records the honest FAIL receipt this report originally carried
  (below, kept for the record).
- `d603136` (sonnet, codex out of budget, coordinator handoff 2026-09-17): sends
  mDNS goodbyes on shutdown, cleans up three orphaned test processes, and
  (with the user's explicit approval) opens inbound mDNS on the `public`
  firewalld zone. Diagnosis and fix below.

## The original failure (codex, kept for the record)

```text
bench/p0b-router-gate.sh
exit 1
```

`/v1/node-info` passed for UUID `00000000-0000-4000-8000-0000000000b3`;
`state_locality` was absent (correct). PAIR's `discovery:get-nodes` response
did **not** contain that UUID or its `ni=18103` entry. It contained the stale
UUID `00000000-0000-4000-8000-000000000095` and the local `hercule` entry
instead. Receipts: `.work/team-A/codex/p0b/pair-gate/{contract.log,
node-info.json, pair-frames.json, router.stderr}`.

## Diagnosis (sonnet, 2026-09-17, room A)

Two independent problems stacked, found in order:

**1. Orphaned processes, and no mDNS goodbye on exit.** `ps` found three
`router` processes from earlier ad-hoc test runs still alive (started 08:12,
08:13, 08:16), one of them `--node-id 00000000-0000-4000-8000-000000000095`
on port `18095` -- the exact stale UUID in the failure receipt. It was not
cached, stale data; it was a live process still re-announcing. Killing it
made the specific stale entry disappear, but reading `mdns-sd`'s own source
(`service_daemon.rs`) showed why any past run, not just orphans, poisons
future ones: `ServiceInfo`'s PTR/TXT records carry `DNS_OTHER_TTL = 4500`
(75 minutes, RFC 6762), and `ServiceDaemon` has no `Drop` impl at all --
`unregister()`/`shutdown()` exist but `MdnsRuntime` never called them, so a
killed router (however cleanly) leaves its record believed by every peer
that ever saw it for up to 75 minutes. Fixed in `d603136`: `MdnsRuntime`
tracks each registered fullname and unregisters explicitly, wired through
`with_graceful_shutdown` on SIGINT/SIGTERM. Verified live: `mDNS unregister
baro-...: OK` for both records on a clean SIGTERM.

**2. This did not fully explain the failure.** After killing the orphans and
landing the fix, gate 3 still failed the same way, now against a *clean*
router record with no stale competitor. Isolated the remaining cause from
both our code and PAIR's Go implementation with a from-scratch Python UDP
multicast probe -- a sender and receiver on the same interface, nothing to
do with `mdns-sd` or `nvpair-node-scanner`:

```text
python3 -c '... IP_MULTICAST_IF=192.0.2.10, IP_ADD_MEMBERSHIP on enp16s0 ...'
TIMEOUT: no multicast reflection received on enp16s0-bound socket
(the identical probe against 127.0.0.1 receives its own packet immediately)
```

`ip maddr show enp16s0` confirmed our router genuinely joins the multicast
group (`224.0.0.251 users` incremented while it ran); `avahi-browse -r -t`
confirmed the OS-level picture independently: our record was visible **only
via `lo`**, never `enp16s0`, and PAIR's own `-log-level debug` scanner output
confirmed it queries `enp16s0` on every cycle (`mdns send: query sent
service=_nvpair-node._tcp iface=enp16s0 ip=192.0.2.10`) and simply never
receives an answer -- it does not browse `lo` at all, a reasonable design
choice that happens to be exactly the interface this box's mDNS traffic was
confined to.

Root cause: `firewall-cmd --zone=public --query-service=mdns` answered `no`.
`enp16s0` sits in firewalld's `public` zone, which allows only
`dhcpv6-client` and `ssh`; UDP 5353 was never permitted inbound on that
zone, so *no* local process -- not our router, not avahi, not PAIR's
scanner -- ever received multicast traffic tied to that interface, only
`lo` (which firewalld does not filter). This explains every symptom
observed with no further guessing needed.

## Fix applied

With the user's explicit approval (asked because it is a firewall posture
change, not something to decide unilaterally):

```text
sudo firewall-cmd --zone=public --add-service=mdns
```

Non-permanent, per the approval given (session-only; reverts on the next
firewalld reload or reboot -- whoever picks up gates 1/2/4 later, or anyone
rerunning this gate after a reboot, needs to reapply it or ask for the
`--permanent` version).

## Gate 3 receipt

```text
ROUTER_BIN=.work/team-A/sonnet/target/release/router bench/p0b-router-gate.sh .work/team-A/sonnet/p0b/pair-gate
PAIR node discovery PASS uuid=00000000-0000-4000-8000-0000000000b3 ni.port=18103
node-info hostUuid PASS and state_locality absent
P0b gate 3 PASS: PAIR discovered router and verified /v1/node-info
exit 0
```

Run twice (`.work/team-A/sonnet/p0b/pair-gate`, `.work/team-A/sonnet/p0b/pair-gate-r2`),
both exit 0. `node-info.json` in both receipts: `hostUuid` matches, no
`state_locality` key (CONTRACT 4's term is still absent, as designed until
gate 4).

## Checks

```text
cargo clippy --workspace -- -D warnings   clean
cargo nextest run --workspace             72 tests run: 72 passed, 0 skipped
```

No stray `router` or `nvpair-node-scanner` processes remain (`ps` checked
clean after every run in this session).

## Suite and readiness

`./run-tests.sh` (Mojo/kernel suite, unaffected by this Rust-only lane): last
run in room A at line 223 (sonnet), exit 0, 146 PASS -- receipt predates this
gate's own work but nothing since has touched Mojo sources.

## Next action

Gates 1 and 2 need real proxy pass-through (`proxy()` in `router.rs` still
answers 501 `"proxy transport pending"`); gate 4 needs the P1 state-locality
term wired into `choose_engine`. Coordinator assigned these to team A next,
now that gate 3 is green. `lane-team-a` is current with `main` (merged
2026-09-17, `be9a002`) so this can proceed without re-conflicting.
