# Brief: verify every identifier in the latent-communication survey

File under check:
    $HOME/Projects/mojo/mojo-baro-lanes/fork/exchange/2026-09-17-latent-communication-survey.md

It was written by another agent. It cites 47 distinct arXiv ids. Your job is to
find the ones that do not exist, or that exist but are a DIFFERENT paper than the
file claims. An invented or misattributed id is the one defect that makes the file
worthless, so this check is the whole task.

Deliverable, a FILE:
    $HOME/Projects/mojo/mojo-baro-lanes/fork/exchange/2026-09-17-survey-id-check.md

Reply with one line only: "written to <path>".

## Method

For each arXiv id cited in the file, fetch https://arxiv.org/abs/<id> with
firecrawl_scrape (onlyMainContent true). firecrawl ONLY: WebSearch and WebFetch
are forbidden. Compare three things against what the survey says:

1. Does the id resolve at all?
2. Is the TITLE the one the survey gives it? A well-known nickname is fine
   (vec2vec is "Harnessing the Universal Geometry of Embeddings"); a different
   paper is not.
3. Is the first author and the year the one the survey gives?

Do NOT re-read the papers or judge whether the survey's description is a good
summary. Only identity: id, title, first author, year.

Work through them in batches, and do not stop at the first bad one.

## The file to write

A table: id, survey's title, actual title, verdict (OK / WRONG PAPER / DOES NOT
RESOLVE / COULD NOT CHECK), and for anything not OK, one line on what is actually
at that id. Count at the top: N checked, N OK, N wrong, N unresolved.

If a page will not load after two tries, mark it COULD NOT CHECK and move on.
Never guess a verdict from memory of the paper: the point of this task is that
memory is exactly what cannot be trusted here.

No em dashes in the file.
