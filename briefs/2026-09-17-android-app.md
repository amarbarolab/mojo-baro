# Android app: baro on the phone (P6 item 1 plus the Android half of item 3)

the maintainer, 2026-09-17: "build the APK, so we can run models from the phones too." Read
`docs/PLATFORM-PLAN.md` section P6 and `serve/PROTOCOL.md` "HTTP surface" first. Builder: opus.

## Goal

One Android app, two ways to run a model:

1. **Remote:** a client of `baro-serve` on the workstation (and of the P0b router once it merges):
   server address entry now, mDNS discovery + QR pairing when P0b lands; model list, streaming chat,
   voice input through `POST /v1/audio/transcriptions` (P3a, on main).
2. **On the phone:** llama.cpp arm64 inside the app (JNI, CPU first; Vulkan only if it builds and
   passes the same gate), a small GGUF (Qwen2.5-0.5B-Instruct q4 class, pick the file from `~/Models`
   and name it with its sha256), chat fully offline. Reference: `~/llama.cpp/examples/llama.android`.
3. **Node (stretch, only after 1 and 2 pass):** the phone serves the same OpenAI-style HTTP subset on
   the LAN so the router can adopt it as kind `llamacpp` later. Do not build router code.

## Where and how

- New project `~/Android/baro/` (own local git, no remote ever), scaffolded with
  `~/iTools/bin/android-new` (read its tool.toml first; it builds what it emits). Skills to read:
  `~/Brain/Skills/android-clean-architecture`, `android-ui-fitness`, `adb-transport`,
  `local-models`. iTools to check before improvising: `apk-ship`, `apk-sync`, `adb-wifi`,
  `android-ui-audit`, `gradle-verify`, `android-exit-info-pull`.
- Do not edit mojo-baro except a NEW report `exchange/lane-ANDROID-report.md` (pathspec commit on
  main in `~/Projects/mojo/mojo-baro`). Server changes you need go in the report as asks.
- The workstation GPU is used only by the `baro-serve` engine you talk to, started through
  `gpu-wait run --priority 50 --vram 22 --timeout 3600` (read the whiteboard head for GPU holds first).
  llama.cpp desktop runs for reference outputs also go through gpu-wait.
- No phone is attached right now (`adb devices` empty). Build and test everything that needs no
  device first; when a device is needed, stop and ask the coordinator (pane `w82:pC`) with ONE
  instruction for the maintainer (e.g. enable wireless debugging, the exact `adb pair` line).

## Gates (each with a receipt under `~/Android/baro/.work/gates/`)

1. `assembleRelease` signed APK + Robolectric/unit tests green, on a clean checkout of the commit.
2. **Remote identity:** the 20 prompts (`bench/mtp-prompts` or the repo's 20-prompt identity set, name
   it) at temperature 0 through the app's HTTP client code path (JVM test against the live server is
   acceptable before a device exists) produce byte-identical text to `curl` against the same server.
3. **On-device model, emulator or phone:** arm64 build of the JNI lib, the GGUF loads, 5 fixed prompts
   at temperature 0 match llama.cpp desktop CPU on the same file token for token for the first 32
   tokens (report any divergence with the first differing position), tok/s recorded.
4. **UI looked at:** screenshots of chat (remote and local) at phone width, opened for the maintainer with
   `setsid nohup xdg-open`, plus `android-ui-audit` if it applies.

Kill line: on-device gate 3 cannot pass on CPU for the chosen model after one repair round, report
it and ship the remote client alone.

## Rules

Global CLAUDE.md binds: no em dashes, loud failures, commit as work lands with pathspec, DONE only
with the gate receipt named, no self-tagging commits. Report: what passed, what is UNVERIFIED, APK
path and sha256, size. Reply to the coordinator only "written to <path>".
