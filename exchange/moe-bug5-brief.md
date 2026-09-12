# Brief: find bug 5 and close W3 gate 2 (RegesCore-35B, qwen35moe)

Worktree: `$HOME/Projects/mojo/mojo-baro-lanes/MOE`, branch `lane-MOE`,
head `80cdac1`. Work only in this worktree. No git remote, ever. No
`Co-Authored-By` or model-attribution lines on commits. No em dashes in
anything you write.

Report to `exchange/moe-bug5-report.md`. Reply with one line naming that
path, nothing else. Terminal output is status, never content.

Read first, in this order, and do not re-derive what they already settle:
`bench/PROTOCOL-RULES.md` (all of P1-P13), `bench/moe-fix-plan.md`,
`bench/moe-protocol.md`, `exchange/lane-MOE-report.md`.

## The goal

W3 gate 2, frozen and not negotiable:

1. 20 prompts at 64/64 teacher-forced agreement against llama.cpp
2. `./run-tests.sh` green on a CLEAN fixture
3. one non-empty Rust-front `/v1/chat/completions` answer with `BARO_PACK`
   set to the MoE pack

Current state: mean 39.0/64 over the 20 prompts, nothing at 64. At least one
defect remains. Four have already been found and fixed; each one hid the
next, so expect more than one and do not assume the first thing you find is
the last.

## Run it like this

    BARO_MEGA=0 BARO_PACK=.work/moe-w1/pack BARO_PROMPT=<p>.tokens \
      BARO_FORCE=<p>.ref.ids <engine>

Build: `./.venv/bin/mojo build serve/engine.mojo -I kernels -D BARO_MODEL=qwen35moe -o <out>`
(about 6 seconds). Every GPU run goes through `gpu-wait run`, never bare.
`bench/moe-gate2-force.sh` runs the whole gate. Re-read the board head
(`~/Brain/mojo/mojo-baro/whiteboard.md`) before each GPU launch: a hold
written there is invisible to `gpu-wait list`.

`BARO_SPEC=0` is pinned by the harnesses. Teacher forcing and spec decode
are mutually exclusive and the engine raises if you combine them.

## The method that works here, and the one that does not

Reading the code produced THREE confident wrong diagnoses on this exact
path (positional offsets, SPLITK partials, a head-layout transpose). The
oracle found the real defects in one pass. Use the oracle.

    tools/llama-oracle.py <evalcb.log> [out.json]

Generate the log with a SINGLE token so sums are per-token and comparable:

    llama-eval-callback -m ~/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf \
      -ngl 99 -c 512 -n 1 --temp 0 -f <one-token-prompt.txt>

Already on disk: `.work/moe-w3/evalcb1.log` (single token, 1398 tensors),
`.work/moe-w3/llama-1tok.json`, and `.work/moe-w3/one.tokens` (the matching
single token, id 27336) so our side can be run on the same input.

Our side dumps with `BARO_DUMP=<path> BARO_DUMP4=1`. Layout is
`2 * N_LAYERS * H` f32 per token; with `dump4` the slots for layer 0 are

    slot0 post-SSM residual     -> llama attn_residual-0
    slot1 post-FFN norm         -> llama attn_post_norm-0
    slot2 post-MoE residual     -> llama l_out-0
    slot3 final norm
    slot5 Eg then Beta, NH_V=32 each -> llama a_softplus-0 / beta_sigmoid-0
    slot6 So, NH_V*SSTATE=4096  -> pre-gate, NOT llama final_output
    slot7 ssm_out result        -> llama linear_attn_out-0
    slot8 Conv, CONV=8192       -> post-l2norm, llama's is pre-norm

P9 is the rule that matters most here: COMPARE ELEMENTS, NOT SUMS. Layer 0
agreed with llama to 1.2% on the sum of the post-SSM residual while its
individual components were 6%, 12% and 78% out. That sum is why the SSM was
not a suspect for three rounds.

## Already proven correct. Do not spend the round re-checking these.

- MoE weight offsets resolve BY NAME (`serve/moe_pack.mojo build_moe_off`),
  verified: engine layer 2 maps to `blk.2`, not `blk.10`.
- SSM weight offsets are correct: `moe_base+1..4` resolve to attn_qkv,
  attn_gate, ssm_alpha, ssm_beta for layers 0, 1 and 2.
- SSM geometry matches the GGUF: `ssm.group_count 16` (NH_K),
  `ssm.inner_size 4096`, `ssm.state_size 128` (so NH_V 32), `conv_kernel 4`.
  The hardcoded values in `kernels/ssm.mojo` are right for this model.
- The delta-rule recurrence matches llama's published form
  `S_t = S_{t-1}*exp(gate) + beta*(v - S_{t-1}^T k) (x) k^T`, `out = S_t q`.
- MoE FFN kernels pass gate 1 exact (expert ids 8/8, outputs ~1e-4) BUT gate
  1 feeds the block the ORACLE's input, never the engine's, so it proves the
  kernels and not the wiring (P11).
- Norm weights are already `(1+w)`-folded in the GGUF (mean|g| ~0.88-1.03),
  so the Gemma RMSNorm form is NOT a defect here.
- The dense path is exact at 64/64 on BOTH the mega and non-mega paths, with
  prefill on and off, at every prompt length up to 58 replay rows. Whatever
  is wrong is qwen35moe-specific.
- llama's `ffn_moe_weights_sum == ffn_moe_weights_sum_clamped` at every
  layer, so their top-8 clamp is inert on this model.

## The four bugs already fixed, as a guide to the shape of the fifth

1. `9e5cc60` weights read positionally from a pack with different strides
2. `4d255da` router fed un-normalised `curb_d` (RMS 21x too small)
3. `d79d78a` batched m>1 prefill broken; every prompt >=18 tokens scored 0
4. `4aee3be` `ssm_alpha`/`ssm_beta` are F32 in the MoE pack and were read
   through a q8_0 kernel, AND both launched `grid_dim=1` against a kernel
   writing `MOE_WAVES=8` rows per block, so 24 of 32 heads were never
   computed

Two of those four were dtype or launch-geometry mistakes that a code reading
had walked past several times. The tell each time was a value that could not
occur naturally: 24 beta gates sitting at exactly 0.5000 (sigmoid(0) on
memory nobody wrote), an RMS 21x below its norm weight. Look for impossible
values, not for wrong-looking code.

## Where to look first

Layer 0 now matches llama to about 2% elementwise, so the remaining defect
is probably NOT in layer 0's SSM. Walk outward:

- the attention layers (`is_attn(i) = (i+1)%4==0`, so 10 of 40), which have
  had far less scrutiny than the SSM and MoE blocks
- the q/k/v slicing of `ConvOut` (q at `[0,KDIM)`, k at `[KDIM,2*KDIM)`,
  v at `[2*KDIM, ...)`) against llama's `build_qwen3next_linear_attn`
- the q6_k expert path, which only fires on layers 34, 38 and 39
- per-layer divergence: dump WITHOUT `dump4` to get `2*layer*H` per layer and
  find the first layer whose elements diverge, then bisect inside it

## Rules that decide whether the round counts

- P7: an A/B harness prints each arm's sha256 FROM INSIDE the code path that
  runs it, and refuses equal hashes. An arm file assembled separately can
  name a binary that never ran, and did, twice, on 2026-09-12.
- P8: a change that compiles, differs by hash and moves nothing may simply
  never have executed. Prove reachability by making the change absurd first
  and confirming the output DOES move.
- P10: a void is a failure. Never average over the arms that survived.
- P13: if you hand off or take over work, rebuild from the committed tree
  and re-run before believing any number.
- Preregister any prediction by commit BEFORE the timed run.
- Commit as work lands, on `lane-MOE`, conventional subject plus a why-body.
  Never commit red or unverified.
- Before `run-tests.sh`: delete `.work/draft-logits.bin` and
  `.work/draft-hn.bin`. A moe-written pair makes `test_sample_ref` fail
  spuriously and has already cost this lane a false failure.

## Falsifier for the round

If the 20-prompt mean does not move above 45/64 after two distinct fixes,
stop and report what the oracle says diverges first, rather than continuing
to iterate. Five bugs deep, the next one may not be findable from layer 0
alone and the round should hand back a localisation, not a guess.

A round that ends "here is the first divergent tensor and why I could not
fix it" is a successful round. A round that ends "improved to 44/64" without
naming what was wrong is not.
