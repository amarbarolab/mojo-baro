# Team A P0b report

Date: 2026-09-17
Item: P0b CPU router skeleton, gate 3, and gates 1/2/4's real proxy + state
locality
Status: **gate 3 PASS** (2/2 clean runs). Gates 1/2/4's mechanism is built
and live-proven with one real engine; the two-engine placement-spread and
live-failover claims those gates actually ask for are **BLOCKED on P4**
(multi-GPU wiring, XTX + iGPU, a separate platform item nobody has built
yet). See "Gates 1/2/4" below for exactly what is proven and what is not.

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
- `65d59a8` (sonnet): real proxy pass-through (`reqwest`, streaming,
  connect-failure retry) and CONTRACT 4 state locality (`choose()`'s
  preferred-engine bonus, `probe_loop`'s `GET /v1/state` poll,
  `locality_preference()`'s `POST /tokenize` + hash match). "Gates 1/2/4"
  below.

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

## Gates 1/2/4

The plan's own text for all three names "two engines on this box (P4's
rig)" or a live kill of one of two. P4 (multi-GPU wiring: `baro-serve` on
the XTX plus a second one on the Raphael iGPU under the HSA override, a
small Qwen2.5-0.5B pack) is its own platform item and nobody has built it
yet (`exchange/` has no P4 report; no iGPU pack exists under `.work/`). That
is a real, single, cross-item blocker for all three gates' literal claims,
not a per-gate list -- named here once rather than three times.

What is genuinely built and what a live single-engine smoke can and cannot
prove:

- **`proxy()` forwards for real now.** It answered 501 before this commit.
  It now forwards via a pooled `reqwest::Client`, streams the response body
  straight through (`Body::from_stream` over `bytes_stream()`, so SSE
  token-by-token output reaches the client as it arrives), and on a
  connect/timeout failure *before* any response arrives, demotes the engine
  immediately and retries the next-ranked one (not waiting for
  `probe_loop`'s next 5 s cycle) -- gate 2's "new ones route to the other,"
  minus the "kill a real engine mid-generation, in-flight requests fail
  loudly" half, which needs a second real engine to demonstrate without
  faking it.
- **CONTRACT 4 (P1-STATE-API.md) is wired end to end**, not just at the
  unit level: `choose()` takes a preferred engine id and treats its pending
  count as one lower, breaking ties in its favor; `probe_loop` polls each
  engine's own `GET /v1/state` every cycle and caches both the raw response
  (`node_info` embeds it per engine, CONTRACT 4's literal wording) and the
  prefix-hash list; `locality_preference()` computes CONTRACT 1's hash via
  `POST /tokenize`, scoped to `/v1/completions` and `/api/generate`'s raw
  `prompt` string (already-rendered text) -- a `messages`-shaped chat
  request would need this process to apply the chat template itself to get
  a "rendered prompt" at all, which only the engine's own tokenizer+template
  pipeline can do, so it falls back to plain rank honestly rather than
  faking a hash. Caught live, before trusting the mechanism: the hash was
  first computed at the full token length, but a checkpoint's `pos` is
  `len - 1` (the same reservation `checkpoints::create`'s own `c.pos`,
  `state.rs::default_pos`'s fallback, and `Chain::lookup` on the engine side
  all use, room A 2026-09-17's P1 fix) -- fixed to check `len - 1` first,
  `len` too.
- **Gate 1's placement-spread across two engines** and **gate 2's live
  "kill one engine mid-generation, in-flight fails loudly, new routes to
  the survivor"** genuinely need two real engines and are not claimed here.

### Live receipt (one real engine)

```text
bench/p0b-proxy-smoke.sh
gate1 mechanism OK: proxied text matches direct byte-for-byte: ' Paris.\nThe capital of France is'
SSE streaming pass-through OK: 10 data frames, terminal [DONE] reached
catalog OK: 1 completions row(s), last one engine=real placement=rank
resident prefix_hash after checkpoint: 524c3fca17fd65ee
gate4 OK: locality-prefixed request placed with placement=locality
PASS p0b-proxy-smoke
```

`node-info` before the checkpoint existed: `real` healthy, `dead` (an
intentionally-unlistened port, to prove `choose()`'s health filter and the
retry-exclude path have something to skip) not, `resident_state` already
present (an empty `GET /v1/state` response, since nothing was checkpointed
yet). After a real `POST /v1/checkpoints`, the same engine's
`resident_prefix_hashes` carries the new hash and a following
`/v1/completions` request for that exact prompt is placed with
`placement=locality`, not `rank` -- the full wire path fires, not just its
pieces.

### Unit coverage

11 router tests (6 new this commit): `choose_plain_rank_picks_the_lowest_pending`,
`choose_locality_bonus_breaks_a_tie_in_the_preferred_engines_favor`,
`choose_locality_bonus_never_overrides_a_genuinely_worse_engine`,
`choose_excludes_listed_ids`, `choose_returns_none_when_every_engine_is_excluded_or_unhealthy`,
`is_hop_by_hop_strips_connection_framing_not_ordinary_headers`, plus the
`prefix_hash` module's own test against `state.rs`'s exact vector (must
agree with the engine's own number, not just be internally consistent).

## Checks

```text
cargo clippy --workspace -- -D warnings   clean
cargo nextest run --workspace             73 tests run: 73 passed, 0 skipped
```

No stray `router`, `baro-serve`, or `nvpair-node-scanner` processes remain
(`ps` checked clean after every run in this session).

## Suite and readiness

`./run-tests.sh` (Mojo/kernel suite, unaffected by this Rust-only lane): last
run in room A at line 223 (sonnet), exit 0, 146 PASS -- receipt predates this
gate's own work but nothing since has touched Mojo sources.

## Next action

Everything buildable without a second engine is built and live-proven.
What remains for gates 1/2/4 is P4 itself: `baro-serve` on the iGPU
(Qwen2.5-0.5B, WMMA-free kernel set, `HSA_OVERRIDE_GFX_VERSION=10.3.0` --
`igpu-env --probe` already confirms the device works, `vadd mismatches 0`)
plus the pack bake for it. Once that exists, `bench/p0b-proxy-smoke.sh`'s
shape (one real engine, a `dead` port to exercise the exclude path) extends
directly to a genuine two-engine `pair-dispatch --count 20` run for gate 1
and a real kill-mid-generation for gate 2; nothing in this commit needs
rework for that, it only needs a second `BARO_ROUTER_ENGINES` entry that
answers. `lane-team-a` is current with `main` (merged 2026-09-17, `be9a002`).
