# Team B P3b report

Status: SKIPPED under the brief's exit clause; no source changes or GPU work.

- Runner found at `~/Projects/ai-models/tts-local`; `~/Models/tts/chatterbox` contains weights only.
- Required `.venv-chatterbox` is absent; `scripts/check` fails its chatterbox hard check immediately.
- Setup would require a multi-GB `chatterbox-tts` and ROCm torch install from `requirements-chatterbox.txt`.
- Existing scripts and README contain no seed control; `ChatterboxTTS.generate()` has no known seed argument in this code.
- Device support is unresolved in the README, which still questions `cuda` versus CPU or HIP.
- Determinism cannot be verified without installing and inspecting the package, beyond the S-item scope.
- Therefore fixed-seed audio and P3a round-trip gates were not run; P3b is skipped per the coordinator ruling.
