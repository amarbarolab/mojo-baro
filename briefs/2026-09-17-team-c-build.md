# Team C build: the P6 pieces that need no GPU and no router

You are one of two peers of equal standing: `sonnet` (Claude Sonnet) and `codex`. The coordinator
is pane `w82:pC`. Your panes were started INSIDE your worktree
`~/Projects/mojo/mojo-baro-lanes/team-c` (branch `lane-team-c`); never edit or commit anywhere else.
Your room is C: `~/iTools/bin/room say|read|cards|card C <name> ...` (the pane ids are in the room's
first line). Teams A and B are building other items in their own worktrees; you talk to them only
through the coordinator.

## Read first, once, in one burst

1. `briefs/2026-09-17-team-build-rules.md`: the rules. Everything in it binds you (room protocol,
   ownership by path, review loop, pathspec commits, no em dashes, reports, the GPU rule).
2. `$HOME/Brain/Skills/mixed-pair-teams/SKILL.md`, the sections "What each kind is for",
   "Split protocol", "Review loop" and "What codex's sandbox cannot do". Codex: that last section
   is about you; sonnet launches anything it lists.
3. `docs/PLATFORM-PLAN.md` section P6, and `serve/PROTOCOL.md` section "HTTP surface".
4. `exchange/2026-09-17-team-A-plan-shape.md` section 2 (the item template you write items in).

`serve/src/main.rs` is shared by all three teams: before touching it run
`~/iTools/bin/room window C <name> take "item, what you edit"`, edit the registration lines only,
commit, `~/iTools/bin/room window C <name> release <commit>`. Refused means another team holds it.

## Item P6-pwa: the iPad and phone client as a PWA, against today's `baro-serve`

The plan serves the PWA from the router at `/`. The router does not exist yet, so build it against
`baro-serve` directly; the router will serve the same files later.

- **Files.** NEW `serve/web/` (index.html, app.js, app.css, manifest.webmanifest, sw.js, icons):
  plain HTML, CSS and JS, NO framework, NO build step, NO CDN (it must work on a LAN with no
  internet). NEW `serve/src/web.rs` (serves `serve/web/` embedded at compile time with correct
  content types, `GET /` and `GET /web/*`). One registration line in `main.rs` through the window.
- **Features, in this order, each landing with its check:** (1) model list from `/v1/models` and
  streaming chat over `/v1/chat/completions` (SSE), with stop, and the conversation kept in
  `localStorage`; (2) installable: manifest, service worker caching the shell, works offline to the
  point of showing the last conversation; (3) server address field (the iPad gets it from a QR code
  later; for now typed), remembered; (4) voice input through `POST /v1/audio/transcriptions`
  (P3a, merged on main) using `MediaRecorder`, hidden when the page is not a secure context;
  (5) a settings sheet: temperature, max tokens, system prompt.
- **Design.** Phone-first, works at 400 px, dark and light from `prefers-color-scheme`, no horizontal
  scroll, safe-area insets respected. the maintainer judges the look; you make it clean and quiet.
- **Owner.** sonnet: `serve/web/*`, `web.rs`, the `main.rs` line. codex: NEW
  `bench/p6-pwa-gate.sh`, the fake engine fixture, the report.
- **Gate (no GPU).** Against `baro-serve` started with a FAKE engine (codex writes
  `bench/fixtures/fake-engine.py`: speaks the engine stdin/stdout protocol of `serve/PROTOCOL.md`,
  prints `ready`, answers every request with a fixed token sequence, never touches a GPU):
  (a) `GET /` is 200 text/html, every asset it references is 200 with the right content type,
  manifest parses, service worker registers (headless Chrome through
  `~/iTools/bin/cdp-headless`, read its tool.toml first); (b) a typed message streams a reply into
  the DOM and the text equals the fake engine's sequence detokenized, read from the DOM, not from
  the network; (c) reload restores the conversation; (d) at 400 px wide the document has no
  horizontal overflow (`scrollWidth <= clientWidth`); (e) with the server stopped the shell still
  loads from the service worker. A screenshot at 400x860 and one at 1024x768 go in the receipts and
  the report names them so the coordinator can show the maintainer.
- **Receipts.** `.work/team-C/<agent>/p6-pwa/`, report `exchange/lane-P6PWA-report.md`.
- **Kill line.** (b) or (d) failing. **Size.** 600 lines of web code, 80 of Rust.
- **Not in this item:** mDNS, QR pairing, the router, "continue on this device". Leave a visible,
  disabled "Pair" entry in settings so the place exists.

## Item P6-aarch64: can our tooling build on aarch64 Linux (S probe)

The plan's facts: Mojo 1.0 emits aarch64 objects but this x86 box cannot link them; Modular
publishes a Linux aarch64 package. Question to answer with receipts, not to assume: do
`baro-serve` (Rust) and the CPU-only Mojo tools build and pass their checks on aarch64?

- **Step 1, Rust, no VM needed.** `rustup target add aarch64-unknown-linux-gnu`, cross-build
  `serve` with an aarch64 linker if one is installed (`aarch64-linux-gnu-gcc`); if none is
  installed, say so and stop this step: installing system packages is the maintainer's call, ask the
  coordinator. Receipt: the `file` output of the binary.
- **Step 2, find the VM rig.** The plan names a `labiso` QEMU rig. Find it
  (`~/iTools/bin/brain-recall labiso qemu aarch64`, `~/iTools/INDEX.md`). Report what exists and
  what an aarch64 guest would need. Do NOT download OS images or multi-GB packages without the
  coordinator's word; propose the exact download with its size.
- **Owner.** codex leads the investigation and the report; sonnet runs anything the sandbox refuses.
- **Report.** `exchange/lane-P6A64-report.md`: what built, what did not, what it would take.
  This item may end as "blocked on a download the maintainer must approve"; that is a valid report.

## Order and done

P6-pwa is the main item; P6-aarch64 is codex's background item while sonnet builds the web files.
Write each item in the room in the template before building, let your partner object, add your
cards (`~/iTools/bin/room card+ C "..."`) and turn them. One report per item, then
`~/iTools/bin/herd tell w82:pC "ITEM <X> report written to exchange/lane-<X>-report.md"`.
