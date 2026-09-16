# lane BAROSERVE report (2026-09-16)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-16-baro-serve-cache-lane.md`. Lane pane `w82:pA` was
killed by the 14:33 OOM; items 1-4 were already committed, the gates were run and the lane closed
from the main session.

## Items

1. Structural id, `tools/model-id.py` (`69806b1`).
2. Cache `~/.cache/baro/<id>/{engine,manifest.json,packs/<gguf-sha256>/}`, eviction manual (`337e443`).
3. Engine selection by architecture, refusal for unsupported arch or tokenizer (`337e443`).
4. `tools/baro serve MODEL.gguf [--port] [--rebuild]` with an id / hit-miss / seconds receipt (`337e443`).
   README Building section and the gate script `tools/baro-serve-gate.sh` (`ff9ecda`).

## Gates

Run: `tools/baro-serve-gate.sh .work/serve-gate MODEL.gguf ...`. Per model: CPU build, then one
gpu-wait job that starts `baro serve` twice and sends a T=0 `/v1/completions` by token ids each time;
PASS = generated ids equal the file's own `baro.run.ref.tokens` and the second start reports
`engine=hit pack=hit`.

| model | id | first run | tokens run 1 / run 2 | warm |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | llama-531b581265e3c68f | hit (built by lane) | 64/64, 64/64 | hit hit |
| granite-4.2-3b-BF16 | granite-d8ea7d76a98aa5af | hit (built by lane) | 64/64, 64/64 | hit hit |
| Spark-X2.5-4B-Q8_0 | spark2_5-6be66f945d465e82 | hit (built by lane) | 64/64, 64/64 | hit hit |
| Qwen2.5-7B-Instruct-Q4_K_M | qwen2-9cb3d655831d981e | hit (built by lane) | 64/64, 64/64 | hit hit |
| qwen2.5-coder-7b-instruct-q4_k_m | qwen2-9cb3d655831d981e | hit (built by lane) | 64/64, 64/64 | hit hit |
| lily-cybersecurity-7b-v0.2-q6_k | llama-d0306a87d21aa507 | hit (built by lane) | 64/64, 64/64 | hit hit |
| Ornith-1.5-9B-Q4_K_M | qwen35-c19706672fb25b4a | hit (built by lane) | 64/64, 64/64 | hit hit |
| Qwythos-9B-v2-MTP-Q6_K | qwen35-c19706672fb25b4a | engine hit, pack miss, 112 s | 64/64, 64/64 | hit hit |
| Qwythos-9B-...-MTP-BF16 | qwen35-c19706672fb25b4a | engine hit, pack miss, 68 s | 64/64, 64/64 | hit hit |
| RegesCore-1.0-35B-UD-Q4_K_S | qwen35moe-f2628b18234dbbb4 | engine miss, pack miss, 62 s | 64/64, 64/64 | hit hit |

The superseded `Qwythos-...-BF16-BARO-8184f7d` bake was not run (no embedded ref tokens).

- **Cold cache:** 10/10 PASS. Cold builds for the seven entries marked "built by lane" were recorded
  by the lane before the kill (commit message of `337e443`); this run saw them as hits.
- **Warm start:** 10/10 print `engine=hit pack=hit`, 0-1 s, same tokens.
- **Id discipline:** Qwen2.5-7B and Qwen2.5-Coder-7B share `qwen2-9cb3d655831d981e` with separate packs;
  the three qwen35 checkpoints share one engine entry with three packs. A copy of the Llama bake with
  `llama.feed_forward_length` edited 8192 -> 8256 gets `llama-d1e9be0d00484c99`; a copy with one weight
  byte changed keeps `llama-531b581265e3c68f`, `baro serve` reports `engine=hit pack=miss` and built a
  separate pack (deleted after).
- **Unsupported model:** `tools/model-id.py MiniCPM5-2B-Q4_K_M.gguf` refuses with
  `tokenizer.ggml.pre='minicpm5' has no pre-tokenizer regex in serve/tokenizer.mojo`. `tools/baro serve`
  on that raw GGUF stops earlier with `not a BARO file: no baro.kernel.commit`, which is true but names
  the bake, not the tokenizer.

Logs: `.work/serve-gate/<model>/` in the lane worktree.
