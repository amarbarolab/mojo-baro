# Team B P3b report

Status: SKIPPED, per the item's own exit clause.

Chatterbox has no runner in `~/Models/tts/chatterbox` (raw safetensors only, no
wrapper code). The real project is `~/Projects/ai-models/tts-local`, whose
`.venv-chatterbox` does not exist on disk; its own `scripts/check` hard-fails
at "venv missing" today. Standing it up means installing `chatterbox-tts` plus
ROCm torch 2.6.0+rocm6.1 from `requirements-chatterbox.txt` from scratch,
several GB of wheels, well past an S item.

No seed or determinism control was found anywhere in the project's README,
`benchmarks/bench_chatterbox.py`, or `scripts/tts-chatterbox`; the README's
own example even leaves the device string an open question ("cuda # or cpu
/ hip?"). `ChatterboxTTS.generate()` is called with no seed parameter in any
existing script, and confirming one exists would require installing the
package first.

Per the brief: "If it cannot be run deterministically or needs more than an
S item, report that to the coordinator and skip P3b; do not sink the team
into it." No code changed, no GPU used.
