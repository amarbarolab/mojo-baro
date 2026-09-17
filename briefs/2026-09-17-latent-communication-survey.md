# Brief: research on models communicating without words

You are a research agent. Deliverable is a FILE. Write it to:

    $HOME/Projects/mojo/mojo-baro-lanes/fork/exchange/2026-09-17-latent-communication-survey.md

Reply with one line only: "written to <path>". Do not paste findings into the terminal.

## The question

the maintainer is building LatentOS: nodes that exchange model STATE (KV pages, SSM slots,
hidden activations) rather than text. He wants the literature on models that
communicate with each other WITHOUT natural language as the channel.

Collect everything real that exists on this. The families to cover, at minimum,
and add any you find that I have not listed:

- Latent / activation-space communication between LLMs or agents: passing hidden
  states, KV cache, soft prompts or embeddings from one model to another instead
  of tokens. "Neuralese", "latent communication", "communicating in embeddings",
  "soft-prompt handoff", "activation transfer".
- Emergent communication / language games in multi-agent RL: agents inventing a
  discrete or continuous protocol, referential games, Lewis games, the whole
  EmeCom line. Include the older foundational work, not just recent papers.
- Continuous / differentiable inter-agent channels: CommNet, DIAL/RIAL, TarMAC,
  IC3Net, and successors. Anything where gradients flow through the message.
- Model stitching, representation alignment and translation between two models'
  latent spaces: relative representations, Platonic Representation Hypothesis,
  vec2vec / universal embedding translation, model grafting.
- Latent-space reasoning as a channel, where it bears on inter-model transfer:
  Coconut (chain of continuous thought), looped/recurrent-depth latent reasoning,
  latent tokens.
- KV-cache transfer and reuse across models or nodes, including the systems
  literature (disaggregated prefill, cache sharing, CacheBlend / prompt-cache
  style work) where the point is moving state, not text.
- Anything on interpretability or oversight of non-linguistic model-to-model
  channels, including steganography in model communication. the maintainer needs to know
  the known failure modes, not just the capability results.

## Rules

- Search with firecrawl ONLY: firecrawl_search, firecrawl_scrape,
  firecrawl_research_search_papers, firecrawl_research_related_papers.
  WebSearch and WebFetch are FORBIDDEN in this task.
- Pass onlyMainContent on scrapes. Do not dump raw pages.
- Every entry needs: title, authors (first author + et al is fine), year, venue,
  arXiv id or DOI, and a LINK. If you cannot find the identifier, say so on the
  entry rather than inventing one. An invented arXiv number is the one failure
  that makes the whole file worthless.
- Two to four sentences per paper: what it actually does, what the channel IS
  (discrete symbols? continuous vector? KV pages? gradients?), and what it
  measured. Not an abstract paraphrase.
- Do NOT rank, filter to "the important ones", or decide what the maintainer cares about.
  Breadth is the deliverable. If a family turns out to have 30 papers, list 30.
- No em dashes anywhere in the file.
- Mark clearly where a claim is yours rather than the paper's.

## Shape of the file

One section per family above. Inside a section, papers in chronological order,
oldest first, so the line of development is visible. End the file with:

1. "What nobody seems to have done" - gaps you actually observed while reading,
   each tied to the papers that would have covered it.
2. "Closest to what we are building" - the handful whose setup is nearest to
   moving real engine state between two running nodes, with why.
3. Search log: the queries you ran, so the next agent does not repeat them.
