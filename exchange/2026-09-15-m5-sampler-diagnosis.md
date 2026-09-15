# M5 sampler disagreement: the host oracle is the wrong side (diagnosis only, 2026-09-15)

Scope: diagnosis of the gate-2 failure in `kernels/test_sample_device.mojo`
(two configs disagree at real vocab). Nothing in `kernels/sample.mojo` or
`serve/sample_ref.mojo` is changed here; the fix is preregistered in
`bench/chat-protocol.md` ("C3 fix round: host top-p mass target") and waits
for a session with room for a distribution gate.

## Verdict

**`serve/sample_ref.mojo` is wrong; `kernels/sample.mojo` is right, on the
evidence below.** The prior ("host ref is the oracle by construction, exact
sorted cumulative sum") does not hold for its top-p cutoff: the host ports
the device's `pmass_target` formula, `W = ceil(top_p * Z)`, from the device's
fixed-point mass (unit 2^-40, where `ceil` is exact) onto a float64 mass whose
unit is `exp(lmax) = 1`. On a peaked row `ceil` then rounds the target up to
the next whole unit of mass, which is most or all of the top-k set. The device
does not "collapse to one element"; it returns the correct nucleus, and the
argmax simply carries 79 to 91 percent of the tempered mass inside it.

## Evidence

Row: `.work/draft-logits.bin` (248320 logits, a real decode row). argmax 4904
with softmax mass 0.416; next 0.097, 0.079, 0.048, 0.019. Reference arms:
numpy float64 (`.work/m5/oracle.py`, output `.work/m5/oracle.txt`).

| config | intended nucleus (smallest prefix with mass >= p x top-k mass) | host set (`ceil` on float mass) | device prob of 4904 (test run) | numpy P(4904) in intended set | numpy P(4904) in host set | host prob of 4904 (test run) |
|---|---|---|---|---|---|---|
| T0.7 k20 p0.8 | 4 tokens | 20 tokens (p x Z = 1.4649, ceil = 2 > Z = 1.8311, so W = Z) | 0.7913294 | 0.7913293 | 0.7578927 | 0.7578927 |
| T0.5 k0 p0.6 | 4 tokens | 56 tokens (p x Z = 1.4415, ceil = 2, prefix to mass 2.0 = 0.833 of total) | 0.9059879 | 0.9059879 | 0.8999498 | 0.8999498 |

Each side's reported probability identifies its candidate set to seven
digits. The device's set is the one top-p means; the host's is the ceil
artifact. The min_p config passes because its floor (`exp(v - lmax) >= 0.05`)
cuts both sides down to the same 4 tokens after the host's inflated top-p,
which is exactly the "min_p accidentally supplies the floor" mechanism, with
the roles reversed: min_p hides the host's defect, not the device's.

Why the host's own tests never saw it: `kernels/test_sample_ref.mojo` checks
`sample_row_ref` against `sample_probs_ref` (same formula on both sides) and
its real-vocab check is temperature 0 only. `kernels/test_sample.mojo`'s
VS=64 chi-square runs at `T1 k- p-` (no truncation), so the top-p branch was
never distribution-tested against an independent oracle at any vocab.

Origin: `8792c58` (2026-09-11, "C3: sampler host reference, matched to KSAMP's
semantics"), the port of `pmass_target` to floats.

## Gate status, corrected

- Gate 1 (temperature 0 byte-identical): PASS, real engine A/B (lane's receipt).
- Gate 2 (device == host per token, real row): FAIL for `top_p < 1` without
  `min_p`, and the failing side is the host. Not a device defect.
- Gate 3 (seed reproducibility) and gate 4 (T=1 chi-square): **PASS at VS=64,
  UNVERIFIED at real vocab.** Both ran only in `kernels/test_sample.mojo`'s
  64-token synthetic vocab and at `T1 k- p-`; they say nothing about the
  top-p, top-k or top-k+top-p shapes at 248320. Missing check: the
  preregistered real-vocab distribution gate below.
- Also UNVERIFIED at real vocab: `top_k > 0` alone (never in gate 2's config
  list).

## What M5 ships as (the maintainer's call, 2026-09-15)

The wiring (`8ddd477`) stays. Requests with `temperature > 0`, `top_p < 1`
and `min_p <= 0` are refused with an explicit error naming this file and the
open round, not sampled; `temperature = 0` stays byte-identical; `min_p`
configs and `top_p = 1` configs work. Refusing is honest until the fix round
lands; sampling from a set the gate cannot yet confirm is the §18 failure this
repo has a rule about, even when the evidence says the device is right.

## Artifacts

- `.work/m5/oracle.py`, `.work/m5/oracle.txt`: numpy arms above.
- `.work/m5/test_sample_device.txt`: the gate-2 run this diagnosis reads.
- `.work/m5/refine.txt`, `.work/m5/fast_cut.txt`: device cut excerpts read.
