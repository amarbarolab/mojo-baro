# Lane tokenizer — report (2026-09-06)

Branch `lane-tokenizer` in `$HOME/Projects/mojo-baro-lanes/tokenizer`, two commits on top of `751bc3c`.

## What landed

- `tools/gguf-tokenizer.py` — reads `tokenizer.ggml.{model,pre,tokens,merges,token_type}`, eos/pad ids, `add_bos/add_eos`, `tokenizer.chat_template` from the GGUF header only (0.3 s, no weights) and writes `<pack>/tokenizer.json` (HF byte-level BPE, llama.cpp regex per `pre`: `qwen2`, `qwen35`) + `<pack>/tokenizer-meta.json`. Written to `.work/engine-pack-q4`, `-q8`, `-q8d` (same tokenizer in every GGUF of this model: metadata hash `35142bfedaceec8e` on BF16 and Q4_0).
- `tools/test_tokenizer.py` — the gate: 20 `bench/mtp-prompts` vs `.tokens`, 41-case hard set vs `llama-tokenize` on the source GGUF, `decode(encode(x)) == x` on all 61. Exit non-zero on mismatch. Not in `tools/ci-checks.sh` (needs the GGUF + `~/llama.cpp`), command documented.
- `tools/baro-tokenize` — `encode` (`--chat`, `--system`, `-o`), `decode` (`--keep-special`), `prompt` (writes `<pack>/prompt-tokens.txt` in the layout `serve/engine.mojo` parses), `info`. Re-execs under `.venv/bin/python` if the caller lacks `tokenizers`.
- `docs/TOKENIZER.md` — file layout, meta fields ↔ GGUF keys, assembly rules, gate, CLI, engine prompt-file contract, Rust `tokenizers` crate loading notes, residual risk.
- `README.md` — "Tokenizer" subsection under Engine status.
- `pyproject.toml` `[dependency-groups] dev = ["tokenizers>=0.21"]` via `uv add --dev`; `uv.lock` is new.

Findings worth knowing: `tokenizer.ggml.pre` is `qwen35`, not `qwen2` — llama.cpp's qwen35 regex adds `\p{M}` to the letter runs; the Qwen2 regex would have been wrong on combining marks. No BOS token exists in this GGUF (`add_bos` false, as llama.cpp defaults for gpt2 vocab). EOS = 248046 `<|im_end|>`, pad = 248044 `<|endoftext|>`. No NFC normalizer (llama.cpp does none); NFD input tested identical.

## Gate

Output: `$HOME/Projects/mojo-baro-lanes/tokenizer/.work/tokenizer-gate.txt` (run at commit `2099f03`; the second commit changed only docs/README/CLI and `tools/ci-checks.sh` was re-run after it with the same result).

| step | command | result |
|---|---|---|
| CI checks | `tools/ci-checks.sh` | exit 1 — **one pre-existing failure**: `docs/KERNELS.md is stale` (census last regenerated `bda99f7` 2026-09-05, `kernels/test_mega_block.mojo` changed `bb797dc` 2026-09-06; diff is only that test's kernel references). Every other check OK, incl. 192 referenced doc paths resolve, 27 python files parse. Not my file; not touched. |
| tokenizer identity | `.venv/bin/python tools/test_tokenizer.py --pack .work/engine-pack-q4` | exit 0 — `PASS: 61 cases, 0 failures` (20 prompts + 41 hard set incl. rendered chat template, empty string) |
| GPU tests | `gpu-wait run --priority 60 --timeout 1800 -- ./run-tests.sh` | exit 0 — `GEMM OK`, `54 kernels, 26 in registry, 0 orphans` |
| q4 engine identity | `gpu-wait run --priority 60 --timeout 1800 -- env BARO_PACK=.work/engine-pack-q4 ./.work/engine` then `tools/check-tokens.sh .work/engine-pack-q4/ref-tokens-64.txt .work/tokenizer-engine-q4.log` | exit 0 — `PASS: 64 tokens match`; receipt in log: `prompt tokens: 5`, `mega fail word: 0`, `pack loaded in 0.44 s`, tok/s_gen 130.8 (one-prompt receipt, not a claim) |

Negative control for the test (run by hand): `llama-tokenize --no-parse-special` on the chat-shaped case gives a different id list than ours, so the comparison is live. Engine build log: `.work/tokenizer-build-engine.log` (warnings only).

20/20 `bench/mtp-prompts` identity is the first block of `test_tokenizer.py`; decode is untouched by this lane, so no MTP re-run.

## Numbers (with the command)

- `tools/gguf-tokenizer.py <Q4_0.gguf> .work/engine-pack-q4` → vocab 248320, merges 247587, 27 special (control) + 6 added (user-defined) tokens, 243 `[PAD…]` unused kept in vocab; `tokenizer.json` 9.8 MB.
- `tools/baro-tokenize decode - < .work/engine-pack-q4/ref-tokens-64.txt` → " Paris.\nThe capital of France is Paris.\n…" (the current `prompt-tokens.txt` decodes to "The capital of France is").
- `tools/baro-tokenize encode --chat "What is 2+2?"` → 45 ids, decodes back to the `<|im_start|>system … <|im_start|>assistant\n<think>\n` rendering (template's default identity system line included).

## Evidence paths

- `.work/tokenizer-gate.txt`, `.work/tokenizer-engine-q4.log`, `.work/tokenizer-build-engine.log` (worktree)
- `.work/engine-pack-{q4,q8,q8d}/tokenizer.json`, `tokenizer-meta.json` (main repo `.work`, via the symlink — the item said "next to each pack"; two new files per pack, nothing existing touched)
- Reference llama.cpp: `~/llama.cpp` at `ca3d5a3e1` (2026-08-28); GGUF `$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0.gguf` (recorded in `tokenizer-meta.json` `source_gguf`)

## Commits (lane branch only)

- `2099f03` tokenizer: build an HF tokenizers BPE from GGUF metadata, identity-tested vs llama.cpp
- `a9247db` tokenizer: baro-tokenize CLI, docs/TOKENIZER.md contract, README section

## Side effect to know about

`.venv` in the worktree is a symlink to the main repo's `.venv`, so `uv add --dev tokenizers` synced the **shared** venv: `tokenizers 0.23.0rc0 → 0.22.2`, `regex 2026.8.31 → 2026.9.3`, `pydantic-core 2.46.5` added. `mojo`, `max`, `numpy`, `yaml` verified importable afterwards; the engine and GEMM test built and ran on it. If the rc tokenizers was wanted, pin it in the dev group.

## What is left

- Nothing from the item's floor. Test count +1 tool test (`tools/test_tokenizer.py`), gate green except the pre-existing `docs/KERNELS.md` staleness (regenerate with `python3 tools/kernel-census.py`, outside this lane's file list).
- `docs/ENGINE-ROADMAP.md` still says "Tokenizer: still none — deliberate non-goal"; not in my ownership list, left for the integrator.
- Server lane: load `<pack>/tokenizer.json` with the Rust crate as in `docs/TOKENIZER.md`; chat template needs a Jinja engine with `raise_exception`/`strftime_now`/`tojson`.

## Questions

None blocking.
