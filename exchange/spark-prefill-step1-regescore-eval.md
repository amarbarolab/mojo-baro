# RegesCore-35B doing Spark prefill step 1 — evaluation (2026-09-08)

Drafts: `spark-prefill-step1-regescore-draft.md` (v1), `-v2.md`. Brief: `.work/briefs/spark-prefill.md` step 1 + gates.
Serving: the self-describing bake on llama-server :8099, thinking off, temp 0.2, 5.5k prompt tokens.

**v1 (9000-token budget): FAIL at G2 by construction.** It wrote "The quick brown fox…" inline into two prompt files
until the budget ran out; no protocol, no scripts. The brief said the prompts come from existing text; it did not
read that as "write a script". Same failure class as the wiring eval: the cheap step is where it burns the budget.

**v2 (with hard rules: no inline prompts, file order, 5k cap): 4261 tokens, finish=stop.** Per file:

| file | verdict | what was wrong |
|---|---|---|
| `bench/spark-prefill-protocol.md` | usable after edits | arm A row says our engine runs "f16 KV, flash-attn on, -b 2048 -ub 512" (fabricated: those are llama flags); read-back list names `spec k:` (Qwythos print, not this engine); no power-cap/vddgfx read-back; bands copied from the brief verbatim, no independent estimate |
| `tools/spark-prefill-ref.sh` | **rejected as written** | kills the fast server, starts the f32 server, THEN runs the whole loop — every "fast" `prompt_ms` would be f32/fa-off timing while props-fast.json says f16. Exactly the silent-parameter failure P1 exists for. Also no gpu-wait (listed as its own unknown), `return_timings` is not a llama.cpp field (harmless) |
| `tools/spark-prefill-prompts.py` | usable after 1 fix | `baro-tokenize count MODEL -` assumes stdin; the CLI reads files and takes the model LAST (`count NUL_FILE MODEL.gguf`); binary search over chars is fine (≈12 tokenizer calls per length) |
| `serve/spark.mojo` diff | **rejected** | invented file header (`from engine.ctx import Context`, `SparkModel`), duplicates the entire 36-layer loop inside the token loop (every layer would run twice per token), calls `k_rope_swa(Q, Kc, Vc, …)` with a made-up signature, prints `prefill rows: N_LAYERS`. The real change is 6 lines around the existing `t_gen_start` |

**Genuinely useful:** the UNKNOWNS section is honest and names the two things that were in fact wrong (kernel names,
gpu-wait). The protocol's "No receipt → arm is VOID" line is a good compression of P1.

**Verdict:** as before — a draft with a load-bearing verify gate. Two of four files were unusable, one of them in the
way that produces clean-looking wrong numbers. Time: 94 s + 45 s of generation, ~20 min of review and repair.
Step 1 was done by hand from the salvageable parts (commit in this repo: protocol + tools + engine print).
