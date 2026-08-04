# NOTES.md — plain-language design log

Twenty minutes at the end of each day. If I can't answer a question here in my
own words, that component is not done. These are the interview questions.

---

## Day 1 — Environment & foundation

**Deviation from spec: Python 3.12 instead of 3.11.**
The spec fixes Python 3.11, but 3.11 was not installed on my machine (only 3.10
and 3.12). I chose 3.12 because it is my default interpreter and every library
in the stack (PyMuPDF, sentence-transformers, ChromaDB, torch) ships 3.12 wheels
in 2026, so installs are clean. 3.11 vs 3.12 makes no defensible difference to
this project's behaviour. Recorded here so it's an intentional choice.

**Why a virtual environment?**
A venv is a private, isolated copy of Python + libraries living inside this
project folder. It stops this project's exact package versions from colliding
with my other projects (Orbit, SmogSense). Reproducibility: anyone can recreate
my exact environment from requirements.txt.

**Why is the raw layer immutable?** _(to answer end of Day 1-2)_

**Why gitignore derived data (parsed/clean/chunks/index)?**
Everything downstream of `data/raw` is reproducible by re-running the pipeline,
so committing it just bloats the repo and invites stale artifacts. The manifest
+ acquisition script can rebuild `raw`, and `raw` rebuilds everything else.

---

## Questions to answer as we build (from spec §9)

- [ ] Why is the raw layer immutable, and what does that buy?
- [ ] Why parse with layout information instead of plain text?
- [ ] What breaks if chunk IDs are not stable across runs?
- [ ] Why prepend title and section before embedding?
- [ ] Why keep both a vector index and a BM25 index?
- [ ] Why did we chunk this way, and what broke with the naive approach?
- [ ] What does the re-ranker do that the embedding model cannot?
- [ ] Why is metric extraction agentic when chunking is not?
- [ ] How do you stop the metric extractor inventing numbers?
- [ ] Walk through one full agent trace.
- [ ] How does the agent avoid looping forever?
- [ ] What does one query cost, and where does the time go?
- [ ] What percentage of papers failed to parse, and why?
- [ ] What does LangGraph do that you cannot do yourself?
- [ ] What still fails?
