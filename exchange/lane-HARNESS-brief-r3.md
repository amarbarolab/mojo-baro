# Lane HARNESS — round 3

Round 2 accepted (`003b01b`, smoke passes). Two more defects, one from REALIGN's finding, one from your own report. Part A now; Part B when the coordinator sends GO.

## Part A (now, your files)

1. **`schema_valid` is always false because the grammar is whitespace-intolerant.** Before the Mojo `Matcher` check, re-serialize the extracted object compactly (parse `scored_text` → JSON → compact string with no spaces, the form `grammar/test_accept_known_good.mojo` uses; if `grammar/json_value.mojo` has no serializer, write a minimal compact one in `bench/`). Schema check runs on the compact string; `scored_text` stays as-is in the output. Expect json_02 arm 0 to flip to `schema_valid: true`; report the 4 json cells before/after.

## Part B (on GO — `lane-REALIGN` will then carry `serve/realign.mojo` with two functions)

2. **`L8-raw` ships a stale vector.** `b.hn_d` is never written under `mega=True` (`kernels/mega.mojo:1182` gates the write on `fold_head == 2`; every launch passes 1). `collect_latent_raw` must call `final_norm_hidden(ctx, b, h_dev)` from `serve/realign.mojo` (post-final-norm f32 `[H]` computed from `b.x_d`) instead of copying `hn_d`. The receiver side (`step_latent_raw`, `apply_latent_to_receiver`) is unchanged — it injects into `x_d`, which is correct.
3. `git merge lane-REALIGN` into `lane-HARNESS` (delete your stub first so the merge takes theirs), adopt the final signature `realign_expected_embedding(ctx, b, e_dev, pack_q4)` for the soft arms.
4. Smoke = all five arms on the same 8 ids: `bench/latent-handoff.sh --ids json_01,json_02,json_03,json_04,math_01,math_02,math_03,math_04 --arms 0,T,L8-raw,L8-soft,L32-soft` under `gpu-wait run --priority 30 --vram 14 --`. Pass = no `error` cells, soft arms produce non-empty `answer_text` on ≥ 6/8, per-arm producer medians reported. Do NOT run the 40-item set; do NOT compute the verdict.

Append `## Round 3` to `exchange/lane-HARNESS-report.md` (Part A table now, Part B after GO), commit results, push `DONE -> exchange/lane-HARNESS-report.md` after Part B.
