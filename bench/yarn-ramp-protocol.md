# YaRN ramp direction: preregistration

Frozen BEFORE any arm is built or run. the maintainer: "do YaRN ramp inversion".

## Claim under test

`kernels/attn.mojo` `amar_rope_yarn` mixes interpolated and extrapolated
angles in the opposite direction to llama.cpp and HF.

Ours (`kernels/attn.mojo:266-268`):

    ramp  = clamp((j - YARN_LOW) / (YARN_HIGH - YARN_LOW))
    theta = theta_in * (1 - ramp) + theta_ex * ramp

so `ramp` is the weight on EXTRAPOLATION and it is 0 for low `j`. Low `j` is
the HIGH-frequency end, so we interpolate the fast-rotating pairs and
extrapolate the slow ones.

llama.cpp (`ggml/src/ggml-cpu/ops.cpp`, `rope_yarn_ramp` + `rope_yarn`):

    ramp_mix = (1 - clamp((i0/2 - low) / (high - low))) * ext_factor
    theta    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix

so their weight on EXTRAPOLATION is 1 for low `j`. HF
`_compute_yarn_parameters` matches llama. Ours is inverted.

## Not in doubt, checked before freezing

The Qwythos GGUF really does ask for YaRN, so this code path is live:
`qwen35.rope.scaling.type = "yarn"`, `scaling.factor = 4.0`,
`rope.scaling.original_context_length = 262144`, `rope.freq_base = 1e7`,
`rope.dimension_count = 64`.

The constants are right, only the direction is suspect. Recomputing
llama's correction dims from those metadata values with the usual
beta_fast=32 / beta_slow=1:

    corr_dim(b) = n_dims * ln(n_ctx_orig / (b * 2pi)) / (2 * ln(base))
    corr_dim(32) = 64 * ln(262144/201.06) / (2*ln(1e7)) = 14.24 -> low  = floor = 14
    corr_dim(1)  = 64 * ln(262144/6.283)  / (2*ln(1e7)) = 21.12 -> high = ceil  = 22

which is exactly `YARN_LOW = 14.0` and `YARN_HIGH = 22.0`. FREQ_SCALE 0.25
= 1/4.0 and MSCALE 1.1386294361119891 = 1 + 0.1*ln(4) also match llama's
`mscale *= 1 + 0.1*log(1/freq_scale)`.

## Arms

- A: current `main`, unchanged kernel.
- B: A with one line changed, `ramp = 1 - clamp(...)`.

Same pack (`.work/engine-pack-q4`), same prompts, same llama.cpp reference
ids, `BARO_SPEC=0` on every run (teacher forcing is a no-spec gate and the
engine raises otherwise). Both engine sha256s recorded in the arm file.

Reference: llama.cpp on the Qwythos BF16 GGUF, greedy, 64 tokens,
`cache_prompt: false`. Our pack is Q4_0 against their BF16, so neither arm
will reach 64/64; that quantisation gap is SHARED by both arms and cancels
in the comparison. The comparison is A vs B, not either against 64.

## Prediction, frozen

If the ramp is inverted, B scores strictly higher teacher-forced agreement
than A across the 20-prompt set, and the gap is large rather than marginal:
the two mixes differ by 0.75 * theta_ex on every pair with j < 14, which is
most of the rotation at any position.

PASS: B's mean agreement over 20 prompts exceeds A's by more than 2 tokens
per prompt, and B is not worse than A on more than 2 of the 20 prompts.

FALSIFIER: B equal to or worse than A. Then the ramp direction is not the
defect on this model and the change is reverted, whatever the source
comparison says. If that happens, do NOT keep iterating on rope: the
next suspect is that ext_factor is effectively 0 in llama for this model,
which would make both mixes moot and point the disagreement elsewhere.

## What is deliberately NOT changed in this round

`tools/attn-ref.py`, `tools/draft-ref.py`, `tools/model-ref.py` and
`kernels/mega.mojo` carry the same ramp. They are only updated once the
measurement decides, so the numpy oracles keep describing arm A until then
and cannot silently drag the comparison with them.
