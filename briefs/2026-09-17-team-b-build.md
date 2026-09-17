# Team B build: P3a and P3b now, then P5a, P4 wiring, P5b, P4 timed gate, P6

Read `briefs/2026-09-17-team-build-rules.md` first (in your worktree after `cd`), then the P3, P5,
P4 and P6 sections of the corrected `docs/PLATFORM-PLAN.md`. The conference is over; this is the build.

## Item P3a: speech in (your template, completed)

- **Files.** NEW `serve/src/audio.rs` (multipart parse, sidecar lifecycle: start on demand, stop
  after idle, proxy to whisper-server); `serve/src/main.rs` (registration of
  `POST /v1/audio/transcriptions`, inside a MAIN.RS WINDOW); `serve/PROTOCOL.md` section; NEW
  `bench/fixtures/p3a/` (20 clips + `MANIFEST.tsv`: file, text, source, license, sha256); NEW
  `bench/p3a-gate.sh` with a `GATE_DRYRUN` stop.
- **Fixture (coordinator's decision, no speech corpus is on disk).** 19 clips synthesized on CPU
  with `espeak-ng` from 19 fixed English sentences of 8 to 20 words (varied digits, names,
  punctuation), 16 kHz mono wav through `ffmpeg`, plus `~/Models/whisper.cpp/samples/jfk.wav` as the
  one human clip. Deterministic, no license question. Check them with `~/iTools/bin/audio-audit`,
  write the manifest, commit the clips (they are small), snapshot before the first gate run.
- **Owner.** sonnet: `audio.rs`, `main.rs` line. codex: fixture, manifest, gate script, whisper
  settings read-back (model file, beam, language, threads: read them back from the running sidecar
  or its command line, record in the receipt), `serve/PROTOCOL.md` section, report.
- **Build.** `cd serve && CARGO_TARGET_DIR=... cargo build --release`; `~/iTools/bin/rust-verify -p serve`.
- **Preflight.** One clip through `whisper-cli` and through the endpoint with `ggml-base.bin` on
  CPU; `gate-dryrun` on the gate script.
- **Gate.** The 20 clips through our endpoint equal `whisper-cli` output byte for byte, same model
  (`ggml-large-v3-turbo-q5_0.bin`) and beam settings. If the sidecar runs on the GPU it is a
  `gpu-wait` resident like every other; the gate does not need `baro-serve`'s LLM engine loaded, so
  do not load it.
- **Receipts.** `.work/team-B/<agent>/p3a/*.log`, report `exchange/lane-P3A-report.md`.
- **Kill line.** Any difference. **GPU.** Minutes, `--vram 4 --timeout 1200`.
- **Dependencies.** None. **Size.** 150 LOC Rust plus fixture and gate.

## Item P3b: speech out (optional, must not block the core)

- **Files.** `serve/src/audio.rs` (same file, same owner, `POST /v1/audio/speech`), the `main.rs`
  line in the SAME window as P3a's, NEW `bench/p3b-gate.sh`.
- **Gates.** (1) fixed-seed audio identical across two runs (sha256). (2) Round trip: the 20
  manifest sentences synthesized, fed through P3a's endpoint, normalized transcript equals the
  normalized input text. Normalization rule is codex's to freeze in the gate script BEFORE the run
  (lowercase, strip punctuation, digits to words or the reverse, one choice), and the report states it.
- **First step, before any code:** find how `~/Models/tts/chatterbox` is run here (venv, entry
  point, seed control, VRAM) and post it in the room. If it cannot be run deterministically or needs
  more than an S item, report that to the coordinator and skip P3b; do not sink the team into it.
- **Kill line.** A round-trip miss after one repair round. **GPU.** `--vram 8 --timeout 1800`.
- **Dependencies.** P3a. **Size.** 120 LOC.

## After P3

Post each report, tell the coordinator, continue with P5a. For P5a, P4, P5b and P6: write the item
in the template above IN THE ROOM first (files, owner, gate, receipts, kill line, GPU budget,
dependencies), let your partner object, then build. Fixtures the plan does not name: propose one in
the room with its path and tell the coordinator in one line; if the coordinator does not answer
within your current sub-step, proceed with your proposal and mark it in the report. P4's gates that
need P0b or P1 record BLOCKED until team A ships them. P5a and P5b training runs are preemptible at
priority 10 with the GPU-hour caps from the plan. `serve/engine.mojo` and every kernel file are
read-only for you.

Add your own cards for the sub-steps (`~/iTools/bin/room card+ B "..."`) and turn them as you go.
