# Team B plan shape: P4, P5a, P3a, P5b, P6, P3b

Conference between sonnet (w8M:p1) and codex (w8M:p2), coordinator w82:pC. Room:
`~/Brain/mojo/mojo-baro/rooms/B/room.md`. This document is the agreed answer; where we did not
agree, both positions are recorded.

## 1. Strengths and limits

**Sonnet:** Fast and reliable at Rust/Python cross-file reasoning, HTTP protocol and schema surfaces, gate scripts, and receipt-first reporting. It will not interpret GPU timings without a frozen protocol note, `gpu-wait`, and effective-parameter read-back. It will stop on ambiguous ownership rather than guess, and must guard against unnecessary rereads despite its large context.

**Codex:** Fast and reliable at evidence-led repository archaeology, Mojo/Rust/Python interface tracing, comparing plan claims with exact source lines, and turning gates into falsifiable commands and receipts. Limits are material: the sandbox is cwd-scoped, so writes to `~/Android`, Brain memory, or a shared lane/exchange path are unavailable unless explicitly present in writable roots; Codex cannot assume Claude skills are loaded, so needed `SKILL.md` files must be named by absolute path; long builds and GPU jobs must be delegated to the prescribed queue, not improvised; and tool output can truncate, so claims must come from files and exit codes. Ambiguous ownership or a missing fixture is a coordinator question, not permission to edit a neighboring hot file.

## 2. Item shape (worked example: P4)

An item is buildable without a question back to the coordinator only if it names, in one place:

| field | meaning |
|---|---|
| Files to touch | exact paths, `NEW` marked; if a route/CLI item, the one line in `main.rs` it adds (see §3) |
| Preflight | CLAUDE.md P1/P6: build once on CPU/dry-run, read every arm-defining parameter (env var, device id) back from the running process before any timed run |
| Gate command | the literal invocation, argv and all, not prose |
| Receipt path | where the gate's stdout/log lands, exact pattern |
| Kill line | the numeric or binary condition that stops the item, stated even when the answer is "any mismatch" |
| GPU budget | queue priority, `--timeout`, `--vram`, and whether the job is inside or outside `gpu-wait` |
| Dependencies | which OTHER item's shipped artifact this item's gate needs (own team or the other team) |
| Owner | sonnet or codex, or split further with a sub-boundary |

**P4 filled in** (Multi-GPU data parallel, `docs/PLATFORM-PLAN.md:147-166`):

- **Files:** `serve/src/main.rs` (one new CLI flag, e.g. `--rocr-visible-devices`, threaded to engine
  spawn) or `serve/src/engine.rs` if pinning lives there; NEW `.work/p4/run-two-engines.sh`; NEW
  `bench/p4-multigpu-protocol.md` (frozen prediction before the timed run, per repo perf-claim rule).
  Does not touch `serve/engine.mojo` or any kernel file.
- **Preflight:** (1) `igpu-env --probe` proves the iGPU override works standalone, before wiring a
  second engine around it. (2) Start each `baro-serve` process, then read back which GPU it actually
  attached (`rocm-smi --showpids` or `rocminfo` per-PID): a `ROCR_VISIBLE_DEVICES` that silently did
  not take effect still produces two engines that both land on the XTX with clean, tight-spread
  numbers (CLAUDE.md's named failure mode). Log the read-back in the receipt before any prompt runs.
- **Gate command:** after the CPU dry-run and preflight stamp, invoke the timed harness as
  `gpu-wait run --priority 20 --timeout 3600 --vram 22 -- .work/p4/run-two-engines.sh`; that script
  starts both engines, records effective device read-back, performs the 20-prompt split, and diffs
  each result against that engine's solo arm. The router-backed variant repeats the same script
  against the P0b endpoint with a separate receipt. If P0b or P1 is not shipped, record BLOCKED
  rather than silently substituting a weaker claim.
- **Receipt path:** `exchange/lane-P4-report.md` + `.work/p4/*.log`.
- **Kill line:** any identity miss between the split run and either engine's solo run (as stated,
  `docs/PLATFORM-PLAN.md:165`).
- **GPU budget:** 1 hour; both devices are admitted through `gpu-wait` in one serialized harness job.
  The plan's iGPU-outside-queue wording at `docs/PLATFORM-PLAN.md:166` conflicts with the standing
  rule at lines 60-64 and must be corrected before build.
- **Dependencies:** gate 1 needs something to split the 20 prompts across two engines. The plan
  says "by the router" (`docs/PLATFORM-PLAN.md:164`), i.e. P0b, which is Team A's item, not yet
  built when Team B starts. Gate 2 needs P1's state export/import API, also Team A/coordinator.
  **P4 as specced cannot pass its own gates until Team A ships P0b and P1.** Resolution we propose
  in §6: gate 1 substitutes a 20-line round-robin dispatch script for the router (same identity
  check, no ranking/catalog logic), re-run through the real router once P0b lands; gate 2 waits on
  P1 for real, or is deferred to after P1 merges. Either way the dependency is now explicit instead
  of discovered mid-build.
- **Owner:** split by path: Sonnet owns the `serve/src/main.rs` and `serve/src/engine.rs` wiring;
  Codex owns `bench/p4-multigpu-protocol.md`, `.work/p4/run-two-engines.sh`, preflight/read-back,
  and the receipt. Codex is the gate runner; Sonnet reviews the gate and the final pathspec.

## 3. Split protocol on shared files

Ownership is by exact path, not by topic. Announce a path in Room B before editing it, and never edit a path while the partner owns it. Sonnet owns route registration and CLI wiring in `serve/src/main.rs` for its assigned item; Codex owns that item's fixtures, gate scripts, and receipt/report paths unless we explicitly trade ownership. `serve/engine.mojo` is Team A's hot file and is read-only for Team B unless the coordinator records a handoff. New P6 client directories and separate Python tools are one-owner paths. If two items must touch `main.rs`, serialize them into one edit window and have the second agent review rather than concurrently patch it.

Each agent uses a distinct repo-local build/output root such as `.work/team-B/sonnet/<item>/` or `.work/team-B/codex/<item>/`; no shared generated binary, log, arm file, port, or model cache is reused without a recorded hash. Announce `RUNNING GPU`, `GPU DONE`, `RUNNING SUITE`, and `SUITE DONE` in Room B. There is one GPU queue: every GPU action, including the iGPU proof or gate, uses the prescribed `gpu-wait` path unless the coordinator explicitly documents a non-GPU command. Only one named agent runs a gate at a time. After edits and review, that gate owner runs the complete `./run-tests.sh` once, captures its exit code and log in the item receipt, then the partner checks the receipt. No concurrent full suites, server ports, or model bakes. Commits are coherent units staged by explicit pathspec only, never `git add .`, `git add -A`, or `commit -a`.

## 4. Review loop

The author first runs the cheapest non-GPU syntax or unit check and records the exact command. Before commit, the partner reviews the diff and the touched-path list, checking API compatibility, effective parameter read-back, fixture identity, failure and kill-line behavior, and that no unrelated or partner-owned file leaked in. The partner also checks that the gate can fail for the claimed reason rather than merely pass with a disabled feature or stale artifact. After both approve, the author runs the item gate, including the single full-suite receipt, and the partner independently verifies the exit code, cited paths, hashes, and commit pathspec. A report's prose is not evidence: the reviewer trusts command output and files on disk. If review finds a defect, the author fixes the owning path and repeats the cheap check before the timed gate.

## 5. Which items for which team, and order

The Team B item set is reasonable, but the coordinator's order and dependency boundaries need correction. P3a and P3b are mostly independent sidecars and can proceed first, with serialized route registration because both touch `serve/src/main.rs`; P3b is optional and must not block the core. P5a should precede P5b because it validates the training and evaluation fixture shape, while P5b needs its own frozen corpus, LoRA tensor set, generalized write-back contract, and multi-hour preemptible budget. P4 wiring can be prepared early, but its router placement gate must wait for the P0b engine-registration contract and its state-move gate must wait for the P1 LAT1 API. P6 probes can start after the platform contracts freeze, but clients and cross-device continuation belong after P0b, P1, and the relevant audio/vision surfaces are stable. Proposed Team B sequence: P3a, P3b in parallel by path with one `main.rs` merge window; P5a; P4 contract and harness wiring; P5b; P4 timed gate; P6 probes, then clients.

P2 kernels remain with the coordinator, but Team B needs an explicit API seam and must not patch kernel files as an incidental dependency. Team A's P0a/P0b/P1 work must publish versioned request/response schemas, error behavior, engine identity, and state-transfer semantics before P4/P6 timed gates. Team A should also reserve the shared `serve/src/main.rs` route-registration window instead of assuming both teams can edit it freely.
## 6. What the plan gets wrong

Three concrete gaps in `docs/PLATFORM-PLAN.md`, beyond the P0b/P1 cross-team dependency in §2:

1. **P5b has no kill line** (`docs/PLATFORM-PLAN.md:185-191`). Every other item in the plan (P0a:86,
   P1:121-122, P0b:144, P4:165, P5a:183, P3a:201, P2:243-245) states one explicitly. P5b lists four
   gates (write-back receipt, forced agreement at the class bar, quality table, 20-prompt identity)
   but never says what failing one of them means: revert the LoRA merge and keep the base bake, or
   retry with a different tensor set, or park the item like P5a's kill line does for A4. Without a
   stated line, a partial pass reads as a judgment call at build time; whether an ambiguous outcome
   ships is a coordinator decision, not a builder's to make silently. We propose: kill line = any nonzero differing
   byte outside the named tensor ranges, OR forced agreement below the class bar with one repair round
   exhausted, OR any regression in the quality table without an explicit written accepted trade-off.

2. **P3a names no fixture path** (`docs/PLATFORM-PLAN.md:195-201`). "A fixed 20-clip set" is the only
   input named, with no source or path, unlike every other item's fixtures (E12/E14/E15's receipts,
   `dense-run.sh`'s reference GGUF, the 20 images for P2). CLAUDE.md's kernel rules require "reference
   fixture snapshotted before any run"; the same discipline should apply here. We need either an
   existing clip set named by path, or an explicit instruction to record one (source, license, and a
   checksum) before P3a's first gate run, so the gate is reproducible by anyone re-running it later.

3. **P3b's gate proves reproducibility, not correctness** (`docs/PLATFORM-PLAN.md:203-204`). "Fixed-seed
   audio identical across two runs (sha256)" only shows the sidecar is deterministic; it never compares
   the output to anything, so a broken voice, wrong language, or garbled phonemes passes the gate as
   long as it is garbled the same way twice. Every other item's gate compares against a reference
   (llama.cpp, HF tower, or the model's own solo run per the teacher-forced identity rule in
   `CLAUDE.md:21-23`).
   We propose adding a round-trip check: feed the P3b output back through the P3a whisper sidecar and
   diff the transcript against the input text (normalized), which is nearly free once P3a exists and
   catches wrong-language or garbled output that a hash cannot.

4. **P4 exempts the iGPU from `gpu-wait`** (`docs/PLATFORM-PLAN.md:166`: "the iGPU jobs run outside
   the queue"). `docs/PLATFORM-PLAN.md:60-64` makes `gpu-wait run` a blanket rule with no hardware
   carve-out: every GPU workload we or an agent starts runs through it, and the queue's own accounting (depth, timeout, VRAM)
   depends on nothing bypassing it. The plan should either name the iGPU job as a `gpu-wait` job at a
   low `--vram` reservation, or the coordinator states the carve-out explicitly as an override (the
   way §10's fable-for-kernels override is stated), not leave it as a line in an item spec that reads
   as contradicting a standing rule. We treat this as a defect until the coordinator confirms it is an
   intended, named exception.

5. **Partial disagreement between us, recorded as asked.** Codex reads P4's gate 1 as underspecified
   because the XTX and iGPU engines run different models (dense q4 vs Qwen2.5-0.5B), so no throughput
   or cross-engine equivalence is stated. Sonnet's reading: the plan already forecloses that concern at
   `docs/PLATFORM-PLAN.md:151-154` ("a functional second device for the harness, never a performance
   arm"), and the identity gate itself is well-defined as written, since "each response identical to
   the same engine alone" compares each engine only to its own solo baseline, never across engines. We
   agree the P0b/router substitution question in §2 still needs a coordinator answer regardless of
   which reading is right.

## 7. What we need from the coordinator

Before the first build hour we need:

- frozen schemas for P0a/P0b/P1, including route registration ownership, streaming/error behavior, engine identity, model naming, and the LAT1 export/import request shape;
- exact fixtures and paths: P4 model packs and per-engine prompt set, P5a held-out prompts and untrained baseline, P5b corpus/checkpoint/tensor allowlist, P3 audio clips plus whisper and Chatterbox settings, and P6 Android/PWA scaffold location;
- a corrected P4 identity design. The plan pairs an XTX dense q4 engine with an iGPU Qwen2.5-0.5B engine, so “each response identical to the same engine alone” is undefined unless each engine is compared with its own single-engine arm or both load the same model;
- the actual `gpu-wait` command, priority, timeout, VRAM reservation, and queue policy for XTX and iGPU work, plus confirmation that the iGPU is not exempt from the standing queue rule;
- named per-item owners for `serve/src/main.rs`, a shared-port policy, and permission/writable-root confirmation for Codex to write the exchange and repo-local `.work` paths;
- acceptance bars for P5a/P5b quality and write-back, and deterministic audio settings plus a normalized transcript/audio comparison rule for P3a/P3b;
- the "every GPU job through `gpu-wait run`" rule (currently in the maintainer's global config, unverifiable from Codex's cwd-scoped sandbox) restated in `bench/PROTOCOL-RULES.md` so both of us can check it directly instead of one peer trusting the other's unverifiable quote.

## 8. iTools wishlist

These are the highest-value tools, checked against `~/iTools/INDEX.md`:

- `pair-dispatch --backend ollama --port 11434 --count 5 --mode parallel`: exercise P0a/P0b as a third-party PAIR client, removing hand-written self-test bias and client-protocol uncertainty.
- `igpu-env --probe` and `igpu-env --run ./engine-small ...`: prove the iGPU environment and apply it reproducibly, removing the current risk of testing the filtered XTX or an inert override.
- `gate-dryrun --arm .work/<item>/arm.txt --expect key=value -- env ... bench/<gate>.sh ...`: drive every gate to its first GPU step on CPU, removing wasted queue slots from argument, fixture, or arm-file errors.
- `claim-check <commit> <falsifier-cmd...>`: rerun the kill line in an isolated worktree and assert the caller tree is unchanged, removing “reviewed the diff but did not independently falsify it” failure.
- `rust-verify.sh [-p serve]`: run the fast Rust check, clippy, and focused tests after route edits, removing slow full-suite feedback from ordinary edit loops.
- `lane-merge lane-<item>`: verify branch receipts, cited paths, trial merge, and `ci-checks`, removing report-only acceptance and missing-receipt surprises.
- `disk-dupes ~/Models ~/.cache/baro ~/Projects/mojo/mojo-baro/.work`: measure exclusive bytes before P5 bakes, removing accidental multi-copy disk pressure.
- `audio-audit [--json] FILE...`: audit P3 fixtures for codec, bandwidth, loudness, and headroom, removing hidden input-quality differences from audio gates. It complements, rather than replaces, transcript/audio identity checks.

---
signed: sonnet
signed: codex
