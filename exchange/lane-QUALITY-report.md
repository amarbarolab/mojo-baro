# lane QUALITY report (2026-09-16)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-quality-rows-lane.md`. Pane `w82:pB` killed by the 14:33
OOM after the prereg; the main session rebuilt the harness (amendments 2 and 3 in
`bench/quality-protocol.md`, frozen before any scored run) and ran the rows.

| model | engine | PPL ours / llama.cpp (ratio) | task ours / llama.cpp | delta | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 22/120 / 24/120 | -1.7 pp | task PASS |
| lily-cybersecurity-7b-v0.2-Q6_K | spark | pending (spark logprobs landed after this sweep) | 13/120 / 15/120 | -1.7 pp | task PASS |
| Qwen2.5-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 46/120 / 46/120 | +0.0 pp | task PASS |
| Qwen2.5-Coder-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 21/120 / 20/120 | +0.8 pp | task PASS |
| Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16 | dense | 9.301 / 8.410 (1.106) | 70/120 / 80/120 | -8.3 pp | PASS |
| RegesCore-1.0-35B-UD-Q4_K_S | moe | 6.233 / 6.242 (0.999) | 11/120 / 12/120 | -0.8 pp | PASS |

Items: 1 instrument DONE (`bench/quality-ppl-run.py`, `bench/quality-task-ids.py`, `bench/quality-run.sh`);
2 prereg DONE (a0aafb5, amendments 806bdf5, c4bcfac); 3 run: 6 of 10 rows, all PASS their bands, 4 not run
(listed in the protocol result); 4 BASELINE table DONE, library `quality` field on 5 entries
(Qwen2.5-Coder has no BARO library entry file).

Open: Spark-family perplexity rerun (spark.mojo logprobs landed today); Ornith, Qwythos-v2, Spark-X2.5,
Granite rows; a q4-vs-q4 Qwythos row to separate quantization from engine.
