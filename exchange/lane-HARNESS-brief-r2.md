# Lane HARNESS — round 2 (coordinator review of `7f980b5`)

The evaluator runs, VRAM read-back and the Topology 2 blocker are accepted. The smoke is NOT a passing check: arm 0 scores 0/4 on items the model visibly answers correctly. Read `results/e8/topology1-q4-2026-09-09.raw.json` json_01 arm 0: the answer is `<think>\n\n</think>\n\n```json\n{ "name": "Alice", "age": 29, "occupation": "engineer" }\n```` and it is scored schema-invalid. Five defects, fix all in your files:

1. **Scoring input is the raw text.** Before schema check (Mojo) and parse (`e8_score.py`): strip every `<think>...</think>` block, strip ``` fences, take the first `{...}` object. Math: last integer after the same stripping.
2. **Exact match is too strict for the schema.** Output BOTH `correct_exact` (parsed == expected) and `correct_subset` (every expected key present with equal value, extra keys ignored) per json cell; the md table shows both columns. The coordinator picks which one the gate uses.
3. **`expected` and `type` are missing from the raw json** per item — include them so the raw file is self-describing.
4. **B never gets an assistant turn after a handoff.** Arm T's B emits `<|im_end|>` immediately (your report saw it), and L8-raw's first generated id is 248046 for the same reason. After the handoff — after A's trimmed CoT ids in arm T, after the k latent vectors in every latent arm — append the ids of `<|im_end|>\n<|im_start|>assistant\n` before B generates. Arm 0 already ends there (`full_prompt`), so all five arms then start B at the same turn boundary. State this in the output header.
5. **B's own thinking eats the budget.** json_03/json_04 arm 0 spent all 128 tokens inside `<think>`. Add `BARO_E8_NOTHINK` (default `1`): when set, append the ids of `<think>\n\n</think>\n\n` after the assistant opener for B in every arm, so B answers without its own chain-of-thought and the only reasoning channel is the handoff under test. Raise `ans_max` default to 256.

Smoke (the check that must pass before DONE): `bench/latent-handoff.sh --items 8 --arms 0,T,L8-raw` with items = json_01..04 + math_01..04 (add an item-selection flag if `--items N` cannot express that). Pass = arm 0 has ≥ 1 `correct_subset` on json AND ≥ 1 correct on math, and arm T's `answer_text` is non-empty on ≥ 6/8. If arm 0 still scores 0, do not tune — dump the stripped text next to expected in the report and stop.

Do not merge REALIGN yet; its signature may change (`realign_expected_embedding(ctx, b, e_dev, ...)` — round 2 there computes logits from `hn_d` because the megakernel never fills `logits_d`). Keep the stub, adopt the final signature from the top of REALIGN's `serve/realign.mojo` only when the coordinator says so.

Same rules as round 1. Rewrite `exchange/lane-HARNESS-report.md` with a `## Round 2` section (stripped-answer examples, the 8-item table), commit results, push `DONE -> exchange/lane-HARNESS-report.md`.
