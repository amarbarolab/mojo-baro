# Team A: how we want the platform plan shaped

Conference between sonnet (`w8K:p1`) and codex (`w8K:p2`), driven by coordinator `w82:pC`. Team A
builds P0a, P0b, P1 (the maintainer 2026-09-17: "you two stay on for the build, so shape the plan for
yourselves"). Room: `rooms/A/`. This file is the agreed record; both positions kept where we did
not agree.

## 1. Strengths and limits

**sonnet.** Fast and reliable at: Rust (axum/tokio) and Mojo in the same session with no
skill-loading gap, matching this repo's existing wire conventions (`serve_proto.mojo`,
`serve/PROTOCOL.md`) when adding endpoints, writing gate scripts in the `bench/PROTOCOL-RULES.md`
P1-P20 style, long single-session builds with git staged by pathspec. Fails at: no memory across
sessions (every turn re-derives state from room/baton/Brain, never assumes a prior turn happened),
Bash runs through permission prompts rather than a sandbox so throughput depends on auto mode being
confirmed up front, no native GPU access (`gpu-wait` only), and a single file past a couple thousand
lines (`serve/window.mojo` at 1748) gets read in windows, not whole, so a change there needs the
right grep before the right Read.

**codex.**

Fast and reliable at: cross-file audits, protocol and gate contradiction checks, Python/Go client
surfaces, and compact evidence-driven handoffs. I am strongest on bounded reads and mechanical edits
where the command, output path, and exit code are explicit. Failure modes: this Codex session does not
inherit Claude skills, so a brief must name every required `SKILL.md` by absolute path; its sandbox is
cwd-scoped and can block shared `exchange/` or `.git` writes in a lane; long builds and GPU timing need
`gpu-wait` and an independently named receipt; and ambiguous shared-file ownership causes collisions.
I also need tool output filtered or chunked because a large terminal dump can crowd out the repo truth.

## 2. Item shape

Template, filled for P0a as the worked example:

- **Files.** Exact paths touched, new files named explicitly (a new file is a decision, not an
  incidental). P0a: `serve/src/main.rs` (route registration only), a new `serve/src/ollama.rs`
  (Ollama request/response types and the translation: `serve/src/protocol.rs` is already spoken
  for, it is the Mojo child stdin/stdout wire parser per its own doc comment, not a place for HTTP
  types, codex caught this), reusing the `timings` block `serve/PROTOCOL.md:168` already returns
  rather than inventing new engine-side counters, `serve/PROTOCOL.md` (doc the new routes).
- **Build command.** The exact command, not "cargo build" (release vs debug matters for the gate's
  timing numbers). P0a: `cd serve && cargo build --release` (confirm this is in fact the repo's
  convention before citing it as gospel, `run-tests.sh` never touches `serve/`).
- **Gate command(s), verbatim, with expected exit/output.** P0a: (1)
  `pair-dispatch --backend ollama --port <ours> --count 5 --mode parallel`, 5/5 with token counts
  equal to `/v1/completions` same seed T=0; (2) the `ollama` Python client `chat`/`generate`,
  streaming and not, byte-identical to `/v1/chat/completions` on 20 prompts; (3) PAIR's own node
  (built from source, not just `pair-dispatch`'s client half) adopts us on 11434 and routes a
  request, receipt = PAIR's workload log naming our node.
- **Receipt path.** Where the gate's output is written so `lane-merge` and the driver can `ls` it,
  not just read a claim: `exchange/lane-P0A-report.md` citing `.work/p0a/*.log` paths that exist.
- **Kill line.** The exact number/condition that voids the item, quoted from the plan verbatim:
  "any mismatch in gate 1 or 2" (P0a).
- **Preflight (P15).** Build on CPU, run each gate once on its smallest input outside the queue
  before any GPU-timed run: P0a is not GPU-timed at all (minutes, CPU-bound HTTP), so preflight is
  just "gate 1 with `--count 1`" before the 5-count and 20-prompt runs.
- **Size limit.** LOC ceiling from the plan's own S/M/L/XL, e.g. P0a = 300 LOC Rust; going over it
  without saying why in the commit message is a §7 violation (unrequested flexibility), not free.

## 3. Split protocol on shared files

**Ownership.** sonnet: `serve/src/main.rs` (route registration for P0a and P1), new
`serve/src/ollama.rs` (P0a), `serve/latent.mojo` additions (P1). codex: new
`serve/src/bin/router.rs` and anything router-only (P0b), a new Rust binary target, disjoint from
sonnet's files by construction. `serve/PROTOCOL.md` is shared documentation, both items add
sections to it: announce before editing it, one owns the edit at a time.

**Announce rule.** `room say A <name> "editing <path>"` before touching anything outside your own
list above, and before touching `serve/PROTOCOL.md`. Never edit a file the partner is mid-edit on.

**Commit cadence.** Pathspec only, never `git add -A`/`-a`/`commit -a` (teamson invariant). Commit
per landed sub-step, not one giant commit per item. `index.lock` failure means the partner is
committing: wait 3s, retry.

**Build dirs.** `cargo build` for two agents in the same `serve/target/` can lock-contend; if we end
up in one checkout (see section 7), set `CARGO_TARGET_DIR` per agent. Mojo build products for P1
use a distinct name (`.work/p1-engine`) until landed, never overwrite `.work/engine` mid-item.

**Suite ownership.** Whoever lands a commit touching `serve/` runs `cd serve && cargo build
--release` (compile check) plus that item's own gate before the commit. `./run-tests.sh` (Mojo/shim)
is run by whoever touches Mojo files (P1 side), and the full suite is wrapped once at the end of
each item before merge, not per commit, per P18/P19 (quick gate to iterate, full gate to claim, GPU
only spent on the question).

## 4. Review loop

Author gates their own item before announcing done (P11 style: the gate must be shown able to fail,
not just shown to pass). Reviewer is the partner, at the commit boundary before it lands, not after
and not only at the final gate, because a review after merge just re-discovers what `lane-merge`
would have caught mechanically anyway. The reviewer checks what the gate script and the author's own
report cannot check about themselves: P7 (is the arm file written from inside the execution path, not
assembled beside it), P10/P11 (does the gate actually exercise the feature, could it fail), P13 (does
the receipt postdate the commit it claims to prove), and whether the diff matches what the item shape
in section 2 named (no silent scope growth past the LOC ceiling). Anything touching `serve/src/main.rs`
route registration or `serve/PROTOCOL.md` gets a review ping before commit, since both peers read
requests off that surface; work confined to one peer's own new file (`ollama.rs`, `router.rs`) is
reviewed at the next natural commit boundary, not blocked on a live ping.

After commit, the reviewer checks `git show --name-only` against the explicit pathspec and confirms
that no partner paths leaked into the commit. The reviewer independently runs the named falsifier
through `claim-check` when possible, or records why a worktree replay is unavailable. At the final
gate, one named runner owns the single GPU: every GPU command is wrapped once by `gpu-wait run
--timeout`, the full `./run-tests.sh` suite runs once after both peers' changes are integrated, and
the receipt is written to the declared private `.work/` path. A green claim is not accepted without
the exact commit, exit code, and an `ls`-verifiable receipt.

## 5. Items per team and order

**Team membership:** agreed, P0a + P0b + P1 to Team A matches the coordinator's draft.

**Order:** P0a, then P1, then P0b, following `docs/PLATFORM-PLAN.md`'s own Order table (line 47-58),
not the sequence implied by the coordinator's brief text ("Team A = P0a, P0b, P1"). Reason: P0b's
rank formula (line 132) consumes "the state-locality term from P1", so P0b cannot be built complete
before P1 exists.

**Split:** sonnet takes P0a, then P1. codex starts P0b as a CPU-only skeleton (mDNS advertise/browse,
`/v1/node-info` without the state-locality field, port-probe adoption, proxies) once P0a lands,
running in parallel with sonnet's P1 build. P0b's rank/placement gate (gate 1) and the state-locality
term are DEFERRED, not stubbed, until P1's `/v1/state` contract is frozen (codex's correction over
my earlier stub-at-zero proposal, which would have been exactly the kind of empty-but-passing gate
P10/P11 exist to catch). P0b gate 3 (PAIR lists us via mDNS) does not need P1 and can run as soon as
the skeleton exists.

**Open disagreement with the coordinator's draft:** "P1 design spec comes from the coordinator" names
no model. The plan's own lane column says "opus design, sonnet build" (line 50) but the baton records
the coordinator pane as fable (`w82:pC`). Carried to section 7 as a question, not resolved here.

## 6. What the plan gets wrong

1. **Order mismatch.** `docs/PLATFORM-PLAN.md:47-58` (the Order table) sequences P0a, P1, P0b; the
   coordinator's dispatch brief lists "P0a, P0b, P1". Not fatal, but the build plan the coordinator
   writes next should use the table's order, matching section 5 above.
2. **P0b's gates never exercise the term its own design depends on.** `docs/PLATFORM-PLAN.md:132`
   states the rank formula includes "the state-locality term from P1", but P0b's three gates
   (`docs/PLATFORM-PLAN.md:140-145`, placement/rank balance, mid-run failover, mDNS adoption) do not
   require that term to hold any real data. As written, P0b could pass its whole gate suite while
   the term is permanently empty, and nobody would notice until cross-checked against P1. Fix: either
   the router build plan adds a fourth gate exercising state locality once P1 lands, or the item is
   explicitly staged (CPU skeleton now, full gate after P1), which is what we are doing in section 5.
3. **PAIR's server side has no build fixture named anywhere in the plan.** P0a gate 3 and P0b gate 3
   both need "PAIR built from source on this box" to adopt or list `baro-serve`. `pair-dispatch`
   (`~/iTools/harness/pair-dispatch/pair-dispatch.sh`) only builds
   `Personal-AI-Router/scripts/inference-dispatcher`, the client half; grepping the PAIR repo's
   `AGENTS.md`/`CONTRIBUTING.md` at the top level turns up no `go build` line for its own
   server/engine-manager (the node that would actually probe and adopt us). This is a missing
   fixture, not a false claim, carried to section 7.
4. **P1's int8 KV state files are described as both refused and available.** `docs/PLATFORM-PLAN.md:92`
   (Exists) says `BARO_STATE_LOAD` state files are "f32 KV; int8 KV refused". Twelve lines later,
   `docs/PLATFORM-PLAN.md:104` designs `POST /v1/state/export` as carrying "int8 KV in state files
   (C2-mini)" as if it already works. Checked against the baton: C2-mini exists on branch `ckpt-api`
   (`493c6f8`) but is NOT merged to `main`. P1's design needs C2-mini merged as an explicit
   prerequisite, not a same-item detail (codex).
5. **P0a's embeddings claim overstates what exists.** `docs/PLATFORM-PLAN.md:76` says
   `/api/embeddings` and `/v1/embeddings` return "the pooled last hidden state (the latent path
   already exposes it)". Checked: `serve/src/main.rs`'s route table has no embeddings route today,
   and `serve/latent.mojo:338`'s `mint_hidden_latent` mints a raw per-step hidden-state memfd for IPC
   handoff, not a pooled embedding vector over an HTTP response. Pooling (mean or last-token) plus
   the route itself is real work not accounted for in the 300 LOC S estimate (codex).
6. **P4's "S wiring" undercounts what multi-GPU actually needs.** `docs/PLATFORM-PLAN.md:156-166`
   calls for one `baro-serve` process per GPU pinned by `ROCR_VISIBLE_DEVICES`. Checked:
   `serve/src/engine.rs:66-74`'s `Engine::spawn` only sets `BARO_SERVE`/`BARO_PACK` and inherits the
   rest of its own process environment; `docs/BASELINE.md` documents `BARO_POOL > 1` as not fitting
   one GPU today because pool members share one environment, meaning they do not get distinct device
   pins. P4 needs a launcher or config seam (per-instance env, or `BARO_POOL` made device-aware)
   before its own gate is real, not just process-per-GPU wiring (codex).
7. **P0a's kill line does not cover its own named falsifier.** `docs/PLATFORM-PLAN.md:86` states
   P0a's kill line as "any mismatch in gate 1 or 2", leaving gate 3 (PAIR adoption) as informational.
   But `docs/PLATFORM-PLAN.md:286-288`, the plan's own closing paragraph, calls gate 3 "the first
   thing that would prove this plan wrong": if PAIR built from source does not adopt `baro-serve`,
   "the compatibility half of the thesis is a claim about their code, not ours, and P0b's own router
   becomes the only path." A gate-3 failure should not void P0a's own correctness (gates 1/2 still
   define that), but it is not currently wired to any action at all, when the plan's own text treats
   it as load-bearing for the whole platform thesis. Team A's reading: gate 3 failing means P0a still
   ships (translation layer is correct on its own terms), but the failure is reported to the
   coordinator immediately as the plan's central falsifier tripping, not folded quietly into a normal
   item report (codex).

## 7. What we need from the coordinator

1. Confirm the order correction (P0a then P1 then P0b, section 5/6.1) in the build plan you write.
2. Say which model writes P1's design spec (table says opus, you are running fable per the baton)
   and by when, since sonnet's P1 build slot is otherwise idle waiting on it.
3. A fixture or a decision for PAIR's own server/engine-manager build (finding 6.3): either point us
   at where it already builds, or tell us to find and wire it ourselves before attempting gate 3.
4. Decide whether codex gets its own git worktree for this build (fixes the cwd-only sandbox blocking
   `.git`/`exchange/` writes, per `herdr-ops`) or hands sonnet READY files plus patches to commit; this
   decides who actually runs `git commit` for codex's P0b work.
5. Confirm `cd serve && cargo build --release` (only source found: `README.md:236`, `tools/ci-checks.sh`
   never touches `serve/`) is in fact the convention we should both build against, so P0a and P0b don't
   diverge on release vs debug.
6. Merge or explicitly defer `ckpt-api`/C2-mini (finding 6.4) before P1's int8 state-file export is
   built against it: right now P1's own design cites a feature that isn't on `main`.
7. Choose embedding pooling semantics (mean vs last-token, per-request or fixed) for `/v1/embeddings`
   and `/api/embeddings` and revise P0a's 300 LOC budget to include it (finding 6.5); building this
   silently against an assumed pooling rule is a design call, not ours to make under §8/§9.
8. Provide the per-device launcher/config fixture and the XTX/iGPU model paths P4 needs (finding 6.6)
   before Team B's P4 gate is attempted; not Team A's item, but the gap blocks it, so we are naming it
   here rather than leaving it for Team B to rediscover.
9. Decide what "P0a still ships, gate 3 tripping is reported separately" (finding 6.7) means in
   practice: does a failed gate 3 pause P0b/P1 pending your call, or does the team keep building on the
   assumption that P0b is now the primary path regardless of what P0a's gate 3 says.

## 8. iTools wishlist

Existing `pair-dispatch`, `gate-dryrun`, `lane-merge`, `claim-check`, `room`, and `rust-verify` cover
client traffic, CPU gate stops, lane receipts, independent falsification, coordination, and Rust edit
checks. Missing: a `serve-contract-gate` tool, for example
`serve-contract-gate --engine <fixture> --port <n> --out .work/p0a/contract`, that starts the no-GPU
fixture engine, exercises Ollama and OpenAI routes with the installed clients, compares streaming and
non-streaming normalized transcripts, records the model/tags/version responses, and writes a receipt
with each command and exit code. It would remove the repeated manual client matrix and catch route or
schema drift before a GPU queue slot or a PAIR build is spent. This is a new wish, not a duplicate of
the tools listed above.

**Second wish (sonnet), distinct from the one above: `pair-node-up`.** Nothing in `~/iTools` builds or
runs PAIR's own server/engine-manager (only `pair-dispatch`'s client half exists); confirmed by
grepping `Personal-AI-Router/AGENTS.md` and `CONTRIBUTING.md` for a `go build`/`cmd/` entry and
finding none at the top level. Wanted: `pair-node-up [--build] [--advertise-name NAME]` that finds and
builds PAIR's node/engine-manager binary (or names the exact source path if a human has to point it
once), starts it, and prints its mDNS advertisement and listening ports so gate 3 of P0a and P0b has
something to adopt us against. Without it, gate 3 for both items is unattemptable and would sit
UNVERIFIED (CLAUDE.md §18) rather than run; the tool converts an open-ended "go find PAIR's server
code" task into one command with a pass/fail exit code, removing a repo-spelunking detour neither of
us has budget for mid-build.

---
signed: sonnet
signed: codex
