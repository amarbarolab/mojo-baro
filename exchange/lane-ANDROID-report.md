# Lane ANDROID report: baro on the phone (P6 item 1, Android half of item 3)

Brief: `briefs/2026-09-17-android-app.md`. Builder: opus. Date 2026-09-17.
App repo: `~/Android/baro/` (local git, HEAD `34e827c`). Receipts: `~/Android/baro/.work/gates/`.

## Verdict

| gate | result | receipt |
|---|---|---|
| 1 signed release + unit tests, clean clone | **PASS** on `34e827c` | `.work/gates/gate1/SUMMARY.txt` |
| 2 remote identity, 20 prompts, client vs curl | **PASS 80/80** byte-identical | `.work/gates/gate2/SUMMARY.txt` |
| 3 on-device vs llama.cpp desktop CPU | **FAIL 1/5**, after one repair round (kill line reached) | `.work/gates/gate3-phone/`, `.work/gates/gate3-phone-nokleidi/` |
| 4 UI looked at, phone width | **PASS** with audit caveats below | `.work/gates/gate4/` |
| Node (stretch) | not started: gate 3 did not pass | |

**Decision needed (coordinator/the maintainer):** gate 3's kill line says ship the remote client alone. The
evidence says the one failing prompt diverges at a near-tie on a flat distribution, not at an app
bug (details below). The on-phone tab is still in the APK; I did not remove it. Remove it, or accept
gate 3 as 4/5 exact with a documented near-tie divergence?

## APK

- Path: `~/Android/baro/.work/gates/gate1/baro-34e827c.apk`
- sha256: `b0cffd0edfd9807cce7b2332a651678997ef8f4fc78b2ae2648539633d550144`
- Size: 70195904 bytes (67 MB). ABIs arm64-v8a and x86_64 only. versionName `0.1.0+34e827c`.
- Signed with `~/Android/.keystores/baro.jks` (alias `release`, CN=baro), verified by `apksigner verify`.
  The keystore password is in the desktop keyring: `secret-tool lookup service android-keystore app baro`.
- Size note: most of the 67 MB is llama.cpp's runtime CPU variants (7 arm64 plus 14 x86_64, each
  about 1.1 to 1.6 MB, extracted). A phone-only build (`-Pbaro.abis=arm64-v8a`) drops the x86_64 half.

## What was built

- `core` (pure JVM): `BaroClient` for `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`
  (streamed via SSE and not), `/v1/audio/transcriptions` (P3a, multipart WAV). 7 MockWebServer tests.
- `llama` (Android library): llama.cpp from `~/llama.cpp` by path (commit `ca3d5a3e1`), flags from the
  upstream `examples/llama.android` (runtime-loaded CPU variants, KleidiAI and OpenMP on arm64), a
  JNI bridge using only `llama.h`: GGUF chat template, tokenize, greedy decode with a per-token callback,
  prefill and decode timings.
- `app` (Compose): two tabs. "Workstation": server address entry, model name, streaming chat, mic
  button that records 16 kHz WAV and transcribes through baro-serve. "On this phone": import a GGUF
  through the system picker, offline greedy chat, tok/s shown. Brand colors from iTools `android-tokens`.
- Not built (brief: when P0b lands): mDNS discovery, QR pairing, router catalog.

## Gate 1

Clean clone of HEAD, `assembleRelease :app:testDebugUnitTest :core:test :llama:testDebugUnitTest`,
5 suites / 12 tests / 0 failures, every result XML fresh from that build. The gate test for gate 2
skips there by design (it needs a live server). Script `tools/gate1-release.sh`.

## Gate 2

Prompt set: `bench/mtp-prompts/p01..p20-*.txt` (the 20-prompt identity set), temperature 0,
max_tokens 64. For each prompt, four client paths (completion, completion streamed, chat, chat
streamed) through `BaroClient` in a JVM test; curl sends the same four requests; `cmp` byte for byte.
80/80 PASS. Server: baro-serve and engine built from mojo-baro `21e4dbe` into `~/Android/baro/.work/serve/`
(nothing written in the mojo-baro tree), pack `.work/engine-pack-q4`, under
`gpu-wait run --priority 50 --vram 22 --timeout 3600`. Script `tools/gate2-remote-identity.sh`.

First run failed 4/80 because the gate compared the streamed chat against curl's non-streamed chat;
that is a server behaviour (see ask 1), so each path is now compared with curl on the same request
and the stream/non-stream difference is logged as NOTE lines, not gated.

## Gate 3

Model: Qwen2.5-0.5B-Instruct q4_K_M, `~/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf`,
sha256 `8d7026fa27ff7fbf56ee0a9e47235ad6d03d242537a84daa5002d16d72e7c686`.
Device: OnePlus HD1913 (SM8150), Android 12, arm64-v8a lib packaged alone, 4 threads.
Desktop: `llama-server` from the same commit `ca3d5a3e1`, `--device none -ngl 0 -t 8`, fed the
device's own prompt token ids, top_k 1, under gpu-wait (priority 20, timeout 300).
Five fixed chat prompts (`llama/src/androidTest/assets/gate3-prompts.txt`), 32 tokens or EOG.
Prompt tokenization matched on all 5.

| prompt | KleidiAI ON (default) | KleidiAI OFF (repair round) |
|---|---|---|
| 0 capital of France | PASS, 7 tokens to EOG | PASS |
| 1 haiku | PASS, 17 to EOG | PASS |
| 2 sky is blue | **FAIL at position 29**: desktop top-2 logprobs -1.3208 (id 304) vs -1.3447 (id 438); device picked id 13 | **FAIL at position 28**: desktop top-2 -0.6851 vs -0.7639 |
| 3 primes | PASS, 22 to EOG | PASS |
| 4 translate | PASS, 15 to EOG | PASS |

Both failures sit where the desktop distribution is nearly flat (top token about 27% in the ON run),
and the repair build moved the divergence rather than removing it. That reads as cross-architecture
q4_K numerics (NEON/KleidiAI vs AVX-512) at a near-tie, not a bridge bug; the four exact prompts
exercise the same template, tokenizer, prefill and decode code. This is the reading, not a proof:
no x86_64 on-device run exists to separate arch numerics from device code (see UNVERIFIED).

tok/s on the phone (decode, tokens after the first): 28.4 to 38.6 in the first run (KleidiAI ON),
21.4 to 29.5 in the second ON run (battery 21 to 27%, warm), 19.4 to 21.4 with KleidiAI OFF.
Desktop CPU clean run 129 to 144 tok/s. Thermal (`android-thermal-log`, 240 s over the first run):
time-to-throttle NOT REACHED. No crash in `android-exit-info-pull` was needed: the one crash found was
on the emulator (below).

Repair found and fixed on the way (both would have shipped broken otherwise):
1. `useLegacyPackaging=false` left `nativeLibraryDir` empty, zero ggml backends registered, every
   model load failed. Libs are now extracted, and init throws when no backend loads.
2. The comparator counted llama-server's EOG id as a generated token; four prompts failed with
   identical content. Fixed; top-2 logprobs are now recorded at any divergence.

Emulator finding: on the x86_64 API 36 emulator the arm64 lib runs through `libndk_translation` and
aborts inside `libomp.so` (`__kmp_fork_call`) on the first decode. OpenMP under the translator, not
an app bug; a real arm64 phone runs the same build. The emulator job was ended as soon as the phone
appeared (coordinator's note on queue use acknowledged).

## Gate 4

Screenshots from the phone at 1080x2340, 420 dpi = 411 dp: `.work/gates/gate4/remote-chat.png`
(streamed reply from baro-serve over `adb reverse`) and `local-chat.png` (offline reply, 20.1 tok/s),
opened for the maintainer. Looked at: insets respected under the status bar and above the gesture bar, Send
and Mic pinned outside the transcript scroller, bubbles within width, brand colors applied.

`android-ui-audit` on both screens: 4 FAIL, 1 WARN each, all of one kind: clickable Compose nodes
(tabs, Connect, text fields) whose label sits in a merged child node, so the raw uiautomator dump shows
no text on the clickable parent. Compose merges child semantics for TalkBack, so this is probably a
tool false positive for Compose, but TalkBack itself was not run. WARN: Connect is 40 dp tall in the
dump.

Seen and not fixed (cosmetic): the model file name wraps mid-extension in the phone tab header; the
served pack is a reasoning model and its `<think>` block shows verbatim in chat.

## UNVERIFIED

- Voice input end to end: the client call is unit-tested against a mock; recording on the phone and a
  real transcription through P3a were not exercised.
- TalkBack reading of the controls the audit flagged.
- Gate 3 on an x86_64 device (would separate arch numerics from device code for prompt 2): the x86_64
  emulator needs a `-gpu host` queue slot, not spent.
- The release APK was not installed on the phone; the gate 4 screenshots come from the debug build of
  the same source (`2491906` plus the ABI filter and gate tooling commits).
- Import GGUF through the system picker: the model was copied in with `run-as` for the screenshot.

## Asks for mojo-baro (not done here, no mojo-baro edits)

1. `/v1/chat/completions` non-streamed trims trailing whitespace from `message.content`; the stream
   sends it (p04, p09, p10, p18: streamed text 1 to 4 bytes longer). Pick one behaviour for both.
2. Reasoning models: split `<think>...</think>` into a separate field (`reasoning_content`, as
   llama-server does) so clients can hide it.
3. P0b: mDNS service name and QR payload format, so the app's address entry can be replaced.
4. `baro-serve --host 0.0.0.0` works for a phone on the LAN; a documented default port would let the
   app prefill it.

## GPU use (my jobs, 2026-09-17)

gate2 twice (first a comparator FAIL, second PASS), emulator once (14 min at 0.5 GB, ended early),
gate3 desktop reference three times (CPU, short), UI-session baro-serve once (killed after the
screenshot). Day total from `gpu-wait stats --days 1`: 372 jobs, 252 ok / 108 failed / 12 cancelled,
544 GPU-min busy (all lanes).

## Machine changes

- `sdkmanager` installed `ndk;29.0.13113456` and `cmake;3.31.6`; root-umask exec bits fixed.
  Logged in `~/Brain/OS/android-toolchain.md`.
- iTools `android-tokens`: added `~/Android/baro` as a target (`cc12f88`). Its three older targets
  point at renamed app directories and are skipped.
- Phone cleanup: test APK uninstalled, model removed from `/data/local/tmp`, `adb reverse` removed.
  The debug app `com.amarbaro.baro` stays installed with the 0.5B model in its private storage.

## Skills and tools read

Skills: `adb-transport`, `android-clean-architecture`, `android-ui-fitness`,
`compose-multiplatform-patterns`, `kotlin-patterns`, `kotlin-coroutines-flows`, `kotlin-testing`,
`local-models`, plus `prefix-cache-discipline` (long lane).
iTools (tool.toml each; none has a README): `android-new` (scaffolded the project, used),
`gradle-verify` (used for core and app test freshness), `apk-ship` (read, not used: it prompts for
passwords interactively and requires a literal `versionCode = N`, while the scaffold computes
versionCode from the commit count; `tools/gate1-release.sh` reuses its keystore location instead),
`apk-sync` (read, not used: gates install through gradle and adb), `adb-wifi` (read, not needed: the
phone was on USB), `android-ui-audit` (used, gate 4), `android-exit-info-pull` (read, not needed on
the phone), `android-thermal-log` (used, gate 3), `android-tokens` (used, brand colors).
