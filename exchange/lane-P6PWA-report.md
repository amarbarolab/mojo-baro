# P6-pwa report

## Result

PASS. The CPU-only fake-engine browser gate exercised the PWA against `baro-serve`.
The kill conditions passed: streamed DOM text matched the fixed fake sequence and the
400 px document had no horizontal overflow.

## Files and commits

- `2c2a90a` `serve: P6-pwa client shell (chat, install, offline, voice, settings)`
  added the client, embedded web server, and route registration.
- `e7817a6` `test: add P6 PWA fake-engine browser gate` added
  `bench/fixtures/fake-engine.py` and `bench/p6-pwa-gate.sh`.
- `git show --name-only` confirmed the codex commit touched only the two `bench/` files.

## Commands and receipts

- Build: from `serve/`,
  `CARGO_TARGET_DIR=$HOME/Projects/mojo/mojo-baro-lanes/team-c/.work/team-C/codex/target cargo build --release`
  exited 0. Binary: `.work/team-C/codex/target/release/baro-serve`.
- Preflight and gate: `BARO_SERVE_BIN=.work/team-C/codex/target/release/baro-serve bench/p6-pwa-gate.sh .work/team-C/codex/p6-pwa`
  exited 0.
- Gate receipt: `.work/team-C/codex/p6-pwa/SUMMARY.txt`.
- Gate checks: `/` and 4 referenced assets returned 200 with expected content types;
  manifest parsed; service worker registered; typed chat streamed `hello world !` into
  the DOM; reload restored localStorage conversation; 400 px overflow was 400/400;
  offline reload loaded a controlled shell with the server stopped.
- Screenshots: `.work/team-C/codex/p6-pwa/p6-pwa-400x860.png` and
  `.work/team-C/codex/p6-pwa/p6-pwa-1024x768.png`.
- Falsifier: `~/iTools/bin/claim-check e7817a6 bash -n bench/p6-pwa-gate.sh`
  passed and restored a clean checkout.
- Full suite: `GPUWR_SOCKET=/run/user/1000/gpu-waiting-room.sock $HOME/.local/bin/gpu-wait run --vram 24 --timeout 3600 -- ./run-tests.sh`
  exited 0; 47 lines beginning `PASS`; log `.work/team-C/codex/p6-pwa/run-tests.log`.

## Unverified and variance

- Voice input is implemented as `MediaRecorder` to `/v1/audio/transcriptions`, but is
  unverified by this gate because the fake engine has no whisper sidecar. The real P3a
  gate is on its merged commit.
- Web assets are 719 LOC against the 600 LOC item limit. Sonnet disclosed this before
  review; the excess is the phone-first dark/light CSS surface required by the design.

