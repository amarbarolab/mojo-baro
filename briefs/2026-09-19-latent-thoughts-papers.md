# Brief: every paper on latent thoughts (2026-09-19)

You are a sonnet research agent. Working directory: this repo. Do not edit code.

## Goal

Collect ALL papers in the domain of latent thoughts: models reasoning,
drafting or communicating in hidden-state space instead of tokens. Exhaustive,
not representative. the maintainer wants everything.

## Already done, do not repeat, extend it

`exchange/2026-09-17-latent-communication-survey.md` (47 arXiv ids, 7 families)
and its check file `exchange/2026-09-17-survey-id-check.md`. Read both first.
List every paper there as "known" and spend your effort on what is missing.

## Families to sweep (add any family you discover)

1. Latent / continuous reasoning: Coconut, chain of continuous thought, pause
   and filler tokens, Quiet-STaR, looped and recurrent-depth transformers,
   latent-space test-time compute, soft thinking, implicit CoT.
2. Speculative decoding that drafts from the target's hidden states: EAGLE 1/2/3,
   Medusa, Hydra, MTP heads (DeepSeek-V3, Qwen), ReDrafter, Clover, HASS,
   feature-level drafting, self-speculation, layer skip.
3. Speculative decoding for MoE models and for offloaded weights (host RAM,
   CPU-GPU): SpecExec, MoE expert-overlap studies, expert prefetch and caching,
   KTransformers, Fiddler, MoE-Infinity, Pre-gated MoE, offload plus speculation.
4. Model-to-model latent communication: Cache-to-Cache, KV or activation
   sharing between models, cross-model hidden-state translation, relative
   representations, vec2vec, model stitching.
5. Latent state transfer and reuse in serving: KV cache transfer,
   prefill-decode disaggregation, SSM and hybrid state checkpointing.
6. Oversight and interpretability of latent channels: steganography, latent CoT
   faithfulness, decoding hidden thoughts (logit lens, tuned lens, patchscopes).

## Tools (HARD RULE)

Web search and fetch ONLY through firecrawl: `firecrawl_search` (use
`categories: ["research"]` for literature) and `firecrawl_scrape` with
`onlyMainContent: true`. Built-in WebSearch and WebFetch are forbidden. Scrape
arXiv abs pages, never PDFs. Follow citation trails and "related work" sections
of the strongest papers in each family until a pass finds nothing new.

## Verification (HARD RULE)

Every arXiv id must be verified by scraping its abs page: title, first author,
year must match what you wrote. An id you could not verify is listed as
UNVERIFIED, never silently kept. No paper from memory alone.

## Output (files, not terminal)

1. `exchange/2026-09-19-latent-thoughts-papers.md`: per family, one line per
   paper: arXiv id, title, first author, year, one sentence on what it does,
   one sentence on relevance to: a drafter fed the target's hidden states over
   a wire, verified by a fused multi-row MoE kernel, experts in VRAM or host
   RAM, on one consumer GPU. End with: closest prior work to that exact
   combination, and what nobody has done.
2. `exchange/2026-09-19-latent-thoughts-id-check.md`: the verification table.

No em dashes anywhere in either file. No git remote, no pushing. Commit both
files when done, conventional subject, no model attribution line. Reply in the
terminal only: "written to <path>".
