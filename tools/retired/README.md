# retired

Python tools superseded by Mojo implementations. Kept for history and as oracle
material; not on any path, not maintained.

| retired | replaced by | since |
|---|---|---|
| `gguf-tokenizer.py` (HF `tokenizer.json` builder from GGUF metadata) | `serve/tokenizer.mojo` reads the GGUF header directly | 2026-09-08 (`890a452`) |
| `baro-tokenize.py` (Python CLI over `tokenizers`) | `tools/baro-tokenize.mojo` → `.work/baro-tokenize` | 2026-09-08 |
| `test_tokenizer.py` (gate for the Python tokenizer) | `tools/test_tokenizer_mojo.py` (same 61 cases + Spark ids, ref = `llama-tokenize`) | 2026-09-08 |

Not carried over yet: `--chat` rendering of `tokenizer.chat_template` (jinja2)
in the CLI; the gate renders the template in Python as oracle-side input.
