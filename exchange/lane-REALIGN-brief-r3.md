# Lane REALIGN — round 3 (XS)

Round 2 accepted (`798e27e`): the `fold_head==2` finding is verified against `kernels/mega.mojo:1182` and the launch sites. Consequence for the other lane: HARNESS's `L8-raw` arm (and E9 before it) reads `b.hn_d`, so it ships a stale early-chunk vector. HARNESS needs the same post-final-norm hidden you already compute in step 1, as f32.

Add to `serve/realign.mojo` (your file), reusing your step 1:
```mojo
def final_norm_hidden(ctx: DeviceContext, mut b: WindowBufs, mut h_dev: DeviceBuffer[f32]) raises
```
= rmsnorm(`b.x_d` row 0, `output_norm.weight` at `off[HEAD_NORM_IDX]`, eps 1e-6) written as **f32** `[H]` (no bf16 cast — the raw arm ships f32 per E9). If `rmsc_h2` only emits bf16, use `rmsc_k` (f32 path at `serve/window.mojo:937`) or the f32 rms kernel the launch path's `rms_m` uses. Document the signature at the top of the file next to the other one.

Test: extend `kernels/test_realign.mojo` to dump `h_dev` per prompt; `tools/realign_oracle.py` compares it to its own f32 rmsnorm of the dumped `x0` (max-abs relative error ≤ 1e-4 — no bf16 in this path). Same 5 prompts, `mega=True`, gpu-wait, gate to `.work/REALIGN-gate.txt`. Append `## Round 3` to `exchange/lane-REALIGN-report.md`, push `DONE -> exchange/lane-REALIGN-report.md`.
