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

## Day 2 — LLM provider: Groq instead of Claude

**Deviation from spec: Groq (Llama 3.3 70B, free tier) instead of Claude.**
The spec fixes the LLM as Claude via the anthropic SDK, but I don't have paid
API access. Groq's free tier gives me a capable model with OpenAI-compatible
tool calling, which is all the architecture needs. Every design decision is
provider-agnostic: only `src/llm.py` (the wrapper) knows which provider is
behind it. Trade-off: open models are weaker than Claude on the hardest parts
(metric-extraction verification, multi-hop reasoning), so final accuracy numbers
may be a bit lower. Interview answer: "provider swap for cost/access; the
hand-written tool-calling loop is written at the API level and ports to any
provider." The hand-written-vs-LangGraph story (Part B) is unaffected.

---

## Day 3 — Retrieval evaluation (the credibility day)

**Built:** 52 questions labelled with gold chunk IDs (41 single-hop pinpoint,
11 comparative multi-hop), an eval harness (recall@5, MRR, paper-hit@5), and a
retrieval module with 4 methods (vector, BM25, hybrid via RRF, cross-encoder
rerank).

**Result:** vector is the best retriever here — single-hop chunk recall@5 = 0.585,
paper-hit@5 ~ 0.85. BM25, hybrid, and reranking all FAILED to improve exact-chunk
recall; hybrid and rerank actually hurt it.

**Why (the interesting part):** the answers live in number-heavy result *tables*.
(a) Tables have almost no query keywords, so BM25 can't surface them (chunk
recall 0.000) — though it still finds the right *paper* 83% of the time via
architecture/dataset names. (b) The cross-encoder scores tables below prose, so
reranking demotes the answer. (c) RRF rewards vector+BM25 *agreement*, and what
they agree on is prose, so it promotes prose above the table. Diagnosed by adding
a paper-level metric: paper-hit@5 stays ~0.85 for every method, i.e. all methods
find the right document; only exact-chunk pinpointing separates them.

**Decisions:** use vector for chunk retrieval + BM25/hybrid as a paper-level
signal; drop the reranker (no gain AND ~0.6 passage-scorings/sec on CPU, far too
slow). This is a stronger story than "reranking helped": a textbook technique
measured, found to backfire on table-heavy data, and explained.

**Multi-hop failure = the agent's justification:** every single-query method gets
~0.03 recall on the comparative questions ("which architectures work best for
head-and-neck?") because the answer needs 3-4 tables from different papers and one
query embedding can't gather them. Plain RAG cannot do it. The hand-written agent
(Days 4-6) decomposes the question and retrieves per sub-question.

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
