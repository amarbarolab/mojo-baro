# Team B P3a report

Status: complete. All P3a gates, cleanup checks, preflight, and the required whole suite passed.

## Commits

- `2885007` added the 20 speech fixtures, manifest, gate script, and P3a protocol section.
- `235c47b` decoupled `--audio-only` from pack and tokenizer loading and added CPU preflight behavior.
- `6089b37` added whisper-server `-nt` so endpoint segmentation matches the CLI reference.
- `bf73a2a` made the whisper sidecar use an absolute gpu-wait client and skip nested queueing when already admitted.
- `ae2b593` made the gate queue client explicit and fixed the in-job PATH failure for gpu-wait snapshots.
- `f2731e4` added SIGTERM graceful shutdown and explicit audio-sidecar reaping.
- `ffd16b6` added direct-child PDEATHSIG hardening for SIGKILL and OOM parent death.

## Fixtures and static checks

`bench/fixtures/p3a/` contains 19 deterministic espeak-ng English clips and the human
`20-jfk.wav` fixture copied from `~/Models/whisper.cpp/samples/jfk.wav`. `MANIFEST.tsv`
has 20 rows with source, text, license, and SHA-256. Manifest hashes, ffprobe format checks,
and `~/iTools/bin/audio-audit bench/fixtures/p3a` completed. All clips are readable,
16 kHz mono where synthesized; audio-audit reported only the expected source-rate,
transcode, and loudness-spread warnings.

## Gate dry run

`bash -n bench/p3a-gate.sh` passed. The gate dry-run completed with exit code 0 and
`PASS P3a dryrun` in `.work/team-B/codex/p3a/dryrun4/receipt.log`. The dry run performed
no GPU work.

## CPU preflight

The CPU preflight used the release binary, `BARO_WHISPER_MODEL=$HOME/Models/whisper/ggml-base.bin`,
and `BARO_WHISPER_NO_GPU=1` on `20-jfk.wav`. Exit code was 0. The health receipt proves
audio-only mode has no tokenizer and no LLM engine; `/v1/completions` returned the expected
503. Endpoint text matched the `whisper-cli` reference after the gate's edge-whitespace
comparison. Receipts:

- `.work/team-B/codex/p3a/cpu/health-pass.json`
- `.work/team-B/codex/p3a/cpu/completions-pass.status`
- `.work/team-B/codex/p3a/cpu/ref-pass.txt`
- `.work/team-B/codex/p3a/cpu/response-pass.json`
- `.work/team-B/codex/p3a/cpu/children-pass.txt`
- `.work/team-B/codex/p3a/cpu/receipt-pass.log`
- `.work/team-B/codex/p3a/cpu/direct-nt.json`

The first CPU attempt used a stale release binary and was rejected. After rebuilding, the
endpoint initially differed from the CLI at punctuation because the server lacked `-nt`.
`6089b37` fixed that; the release rebuild and the direct `-nt` receipt then passed.

## Timed GPU gate

The timed invocation was:

    GPUWR_SOCKET=/run/user/1000/gpu-waiting-room.sock BARO_SERVE_BIN=.work/team-B/sonnet/target/release/baro-serve BARO_GPU_WAIT=$HOME/.local/bin/gpu-wait bench/p3a-gate.sh .work/team-B/codex/p3a/timed

The gate budget was 4 GiB VRAM and 1200 seconds. The admitted job was
`mu52bqev24pd`. It processed all 20 fixtures and ended with `PASS P3a 20/20` in
`.work/team-B/codex/p3a/timed/receipt.log`. The receipt contains the `gpu-before` and
`gpu-after` snapshots, sidecar command read-back, health response, 503 completions check,
and all 20 reference and endpoint response files. `baro.stderr` contains the audio-only
identity, no-engine evidence, and:

    audio: whisper sidecar running bare, already admitted (GPU_WAITING_ROOM_JOB=mu52bqev24pd)

The first queued attempt failed before transcription because the admitted job did not have
`gpu-wait` on PATH. `.work/team-B/codex/p3a/timed/gpu-before.txt` records the failure.
`bf73a2a` fixed the sidecar path and `ae2b593` fixed the gate's in-job snapshot path.

The successful run exposed a cleanup defect after its 20/20 pass: the gate sent SIGTERM to
`baro-serve`, which had only a Ctrl-C handler, so its `whisper-server` child was orphaned
and held about 24.5 GB VRAM. The orphan was manually killed and VRAM returned to baseline.
The required fix is an explicit `AudioSidecar::shutdown()` call during server shutdown,
followed by a rerun proving no orphan and an empty `gpu-wait list`.

That fix is `f2731e4`. It installs a SIGTERM handler alongside SIGINT and calls
`app.audio.shutdown()` after the engine shutdown. Rust verification passed 31/31 and both
release binaries were rebuilt with mtimes after the source files. A fresh preflight passed
with exit code 0 and stamp `9ce4d88748f1aedf35f3d3cdb1657e71` at 07:09:27; the coordinator
validated it against HEAD `f2731e4`. The same explicit shutdown also fixes the LLM engine
orphan risk on SIGTERM because `app.engine.shutdown()` was previously unreachable when the
kernel's default SIGTERM disposition terminated the server.

The rerun used a fresh output directory, `.work/team-B/codex/p3a/timed2/`, and admitted job
`mu52lyjchz52`. Its receipt ended `PASS P3a 20/20` at 07:10:08. It again proved audio-only
health, no engine, 503 completions, and all 20 transcript comparisons. Post-run checks found
`pgrep -x whisper-server` count 0, `pgrep -x baro-serve` count 0, and `gpu-wait list` empty.
The `gpu-after.txt` snapshot returned to the recorded baseline range with no reclaim leftovers.

The coordinator requested a follow-up hardening test for SIGKILL and OOM: set
`PR_SET_PDEATHSIG(SIGTERM)` in the sidecar child pre-exec, then kill the server with SIGKILL
and verify the child disappears within 2 seconds. Commit `ffd16b6` adds this direct-child
hardening and the direct `libc` dependency. Rust verification remained 31/31 and both release
binaries were rebuilt. A corrected CPU-only test used distinct ports 18200 and 18201, got HTTP
200 from the transcription endpoint, identified the real parent-child pair, killed the
baro-serve parent with SIGKILL, and found no whisper-server at T+0.5, 1.5, or 2.5 seconds.
VRAM returned to baseline. The test is recorded in Sonnet's room receipt. The queued
`gpu-wait run --shared` path remains unverified because its direct child is the gpu-wait
wrapper, not whisper-server; all P3a gates run inside an admitted job and use the direct-child
path. The same PDEATHSIG protection should later be applied to the LLM engine child.

## Preflight and full suite

An earlier full `bench/preflight.sh` run outside the Codex sandbox passed with exit code 0
after `ae2b593`. The coordinator validated `.work/preflight.ok` stamp
`995c7449f86011f882d9c4b5458fab09`, mtime `07:01:25`, against HEAD `ae2b593`; the Codex
sandbox's differing stamp was an environment artifact and is not treated as a tree failure.

The final whole-suite command was:

    GPUWR_SOCKET=/run/user/1000/gpu-waiting-room.sock $HOME/.local/bin/gpu-wait run --vram 24 --timeout 3600 -- ./run-tests.sh

After `ffd16b6`, preflight was rerun with exit code 0 and
`.work/preflight.ok=bf082843124fa85a61c294b6e633863e` at 07:18:58; `--check` also passed.
The timed gate was rerun after that receipt in
`.work/team-B/codex/p3a/timed3/receipt.log` and `.work/p3a-gate-run6.log`, ending
`PASS P3a 20/20` at 07:19:47. Post-reclaim checks found no real sidecar or server process,
`gpu-wait list` empty, and `gpu-wait gpu` at `1067028480` bytes versus baseline
`1084987392` bytes.

The whole suite then ran once under the required 24 GiB, 3600 second gpu-wait wrapper.
Receipt/log: `.work/run-tests-suite.log`. It exited 0, had no FAIL or ERROR lines, and ended
`104 kernels, 58 in registry, 0 orphans` plus `PASS`. The log contains 47 lines beginning
`PASS` and the final suite `PASS` marker. The suite emitted only existing deprecation and
Crashpad availability warnings.
