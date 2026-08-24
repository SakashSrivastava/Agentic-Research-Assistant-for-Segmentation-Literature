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

## Days 4-6 — the agent (metric extraction + hand-written loop)

**Agent 2 (metric extraction, `extract.py`):** one LLM call per paper over its
tables + results text pulls structured rows (architecture, dataset, anatomy,
metric, value, case_count) into a SQLite `metrics` table. **Verification pass:**
every extracted value must appear verbatim in the source or it's discarded (~5%
caught). That is how the extractor is stopped from inventing scores. Source is
tables-first so the numbers survive the input cap.

**Hand-written loop (`agent.py`):** a plain `while` loop on Groq's
(OpenAI-compatible) tool-calling API, no LangChain. Send question + tool schemas;
the model replies with either a final answer or tool-call requests; we execute
the Python tool, append the result as a `tool` message, loop. Tools:
query_metrics / compare_across_papers (read the verified table), search_corpus
(vector retrieval), fetch_paper_section. Planning + reflection are prompt
instructions; guardrails are max 8 iterations + a token budget; a trace records
every tool call. **Proven across tasks** (head-neck, pancreas), 2 steps each when
query_metrics matches.

**Why the agent (not plain RAG):** single-query retrieval scored ~0.03 on the
comparative multi-hop questions (Day 3). The agent decomposes the question and
queries the metrics table per sub-question, which is what makes "which
architecture is best across papers" answerable.

**Free-tier rate-limit engineering (real lesson):** Groq free tier = 8000
tokens/min (TPM) + 200k tokens/day (TPD). Naively firing calls failed papers on
the *per-minute* limit while daily budget remained. Fix in `llm.py`: wait out TPM
429s (they clear in ~8s) but fail fast on TPD (its retry-after is ~40 min). The
agent catches a TPD stop and returns a partial answer instead of crashing.
Extraction and the agent share the daily budget, so the metrics table grows ~50
papers/day; currently 132/275 papers, 829 verified rows across ~10 anatomies.

## Day 7 — end-to-end eval (agent vs baseline RAG, LLM-judged)

**Built:** `eval_e2e.py` answers each question two ways -- baseline RAG (retrieve
top-5, one stuffed LLM call) and the agent -- then an LLM judge scores each answer
1-5 on faithfulness, completeness, citation. Balanced sampling (`--per-hop N`) so
single- and multi-hop are compared fairly.

**First run looked bad for the agent** (baseline won every metric), but inspecting
the actual answers showed the cause was two *agent robustness bugs*, not a quality
gap -- and the eval is what surfaced them:
  1. **Malformed tool-call JSON.** gpt-oss occasionally emits invalid JSON for a
     tool call (e.g. a trailing comma); Groq 400s the whole turn. The old loop
     treated that as fatal and returned nothing. Fix: catch it, nudge the model
     to retry with valid JSON, continue.
  2. **No answer on max-iters.** Hitting the 8-step guardrail returned "reached
     max iterations" instead of using the evidence already gathered. Fix:
     `_finalize()` forces a tool-free answer (tool_choice='none') from the
     gathered context when any guardrail trips.
After the fixes both failing questions produce real, cited answers.

**Two honest findings that survive the fixes:**
  - **Single-hop pinpoint is retrieval's home turf.** For "what DSC does WAU-net
    get on the optic chiasm?", baseline retrieval grabs the exact table chunk and
    answers; the agent's `fetch_paper_section` returned prose that omitted the
    table value, so it (correctly, but incompletely) said "not available". The
    agent earns its cost on comparative multi-hop, not pinpoint lookups.
  - **Judge grounding (fixed).** The first judge scored blind (no ground truth),
    so it could reward a confident baseline number and penalize the agent's honest
    "not available". Fix: the judge now receives the gold passages
    (`gold_chunk_ids`) as REFERENCE EVIDENCE and grades faithfulness against them
    -- a number absent from the evidence is a hallucination. Bonus insight this
    exposed: for q02 the gold evidence IS the results table (optic-chiasm DSC is
    there), so the agent's "not available" is really an *incompleteness* -- its
    `fetch_paper_section` read prose and missed the table, while baseline
    retrieval grabbed the table chunk. Confirms single-hop pinpoint favours
    retrieval, and the agent should prefer search_corpus/query_metrics over
    section prose for exact values.

The agent is token-heavy (~15-25k tokens/question once it finalizes), so a full
clean re-run of the headline table wants a fresh daily budget run on its own.

## Day 9 — deployment (Docker + CI/CD), token-free

Multi-stage `Dockerfile`: CPU-only torch + baked embedding model, no secrets and
no `data/` in the image. `/health` endpoint for the load balancer.
`.github/workflows/deploy.yml`: build -> push to ECR -> deploy on EC2 over SSH ->
`/health` check. Secrets in GitHub Secrets; `data/` mounted read-only on the
instance (never rebuilt in the container). Configs tested locally (Flask test
client: `/health` 200, `/` renders); live AWS provisioning left manual to avoid
idle billing. The workflow is `workflow_dispatch`-only for now: with no AWS
secrets set, a `push`-triggered run red-X's on every commit, so it's manual until
the instance exists (flip to a push trigger once the Secrets are in place).

## Questions to answer as we build (from spec §9)

These are the interview questions. Short answers in my own words.

**Q. Why is the raw layer immutable, and what does that buy?**
Everything downstream (parsed, clean, chunks, index, metrics) is a pure function
of `data/raw`, so I never edit raw in place. That buys reproducibility (any stage
can be rebuilt from raw), safe experiments (a chunking change re-runs from
`parsed/`, it never re-downloads), and a clean audit trail: if a number looks
wrong I can trace it back to an unchanged source PDF.

**Q. Why parse with layout information instead of plain text?**
Plain text extraction reads a two-column paper straight across, so the two
columns interleave into nonsense, and it flattens tables into a run of numbers
with no rows or columns. The scores live in those tables. Reading font size,
boldness, and position lets me rebuild reading order, detect columns, and pull
tables out as structured rows so the numbers survive.

**Q. What breaks if chunk IDs are not stable across runs?**
The retrieval gold set is labelled by chunk ID. If IDs are re-assigned on every
run (e.g. a running counter over a dict), the same text gets a new ID and every
labelled question silently points at the wrong chunk, so the eval measures noise.
IDs are deterministic (`{paper_id}::{section}::{index}`) so the gold set stays
valid across rebuilds.

**Q. Why prepend title and section before embedding?**
A bare results chunk ("0.92 vs 0.88 ...") has no topic signal. Prepending the
paper title and section name gives the embedding context ("this is the Results of
a head-and-neck OAR paper"), so a query about head-and-neck lands near it. The
prefix is embedded but not shown in the answer.

**Q. Why keep both a vector index and a BM25 index?**
They fail in opposite ways. Vector search is strong on meaning ("head-and-neck"
matching "OARs") but fuzzy on exact tokens. BM25 is exact on literal strings
(architecture names like "nnU-Net", dataset names, metric abbreviations) but
blind to synonyms. The eval showed vector wins for exact-chunk recall while BM25
still finds the right paper 83% of the time, so BM25 is kept as a paper-level
signal and for exact-name lookups.

**Q. Why did we chunk this way, and what broke with the naive approach?**
Section-aware, tables kept whole, sized to the embedding model's real 512-token
limit. The naive approach (fixed 800-token windows with overlap) produced chunks
that exceeded 512, and BGE-small silently truncates past that, so the tail of
every long chunk was never embedded. It also split tables across chunk
boundaries, destroying the very rows the system needs.

**Q. What does the re-ranker do that the embedding model cannot?**
In theory: a cross-encoder reads the query and passage together and scores true
relevance, rather than comparing two independently-made vectors. In practice here
it backfired: it scores number-heavy tables below fluent prose, so it demoted the
exact answers. Measured, found to hurt, and dropped (also ~0.6 passages/sec on
CPU, far too slow). That is a stronger story than "reranking helped".

**Q. Why is metric extraction agentic when chunking is not?**
Chunking is a deterministic rule (split on sections, count tokens), so it is
plain code, faster and reproducible. Pulling a clean (architecture, dataset,
metric, value, cases) tuple out of an inconsistent table plus its caption plus
prose needs judgement about what maps to what, which is what an LLM call is for.
Rule of thumb: rules where rules suffice, an agent only where judgement is
genuinely required.

**Q. How do you stop the metric extractor inventing numbers?**
A verification pass: every extracted value must appear verbatim in the source
text or it is discarded (about 5% are caught and dropped). The LLM proposes;
string matching against the source disposes. Nothing reaches the metrics table
that is not literally in a paper.

**Q. Walk through one full agent trace.**
"Best Dice on head-and-neck?" -> step 1 `query_metrics(anatomical_target="head
and neck", metric_name="Dice")` returns ranked rows with architecture / value /
case_count / paper_id -> the model sees enough and writes a ranked, cited answer.
Two steps. Harder comparative questions add a `search_corpus` step for context
before answering. Every call is recorded in a trace shown under the answer.

**Q. How does the agent avoid looping forever?**
Two guardrails: a hard cap of 8 iterations and a token budget. When either trips,
`_finalize()` forces a tool-free answer from the evidence already gathered, so a
guardrail produces a real (if partial) answer instead of an abort. It also
recovers from a malformed tool call by nudging the model to retry rather than
dying.

**Q. What does one query cost, and where does the time go?**
From the end-to-end eval: plain RAG is ~1.3k tokens and ~16s; the agent is
~10-25k tokens and ~60-95s. The extra cost is the multiple tool-calling round
trips, and most of the wall-clock time is waiting out the 8000 tokens/min free
tier limit between calls, not compute. The agent is only worth that cost on
comparative multi-hop questions, where plain RAG scores ~0.03.

**Q. What percentage of papers failed to parse, and why?**
1 of 276 (0.4%) was excluded: a scanned PDF with no text layer, detected and set
aside rather than ingested as empty text. Two more parse but have no detectable
section headings (unusual formatting) and are flagged for a repair pass. Text
coverage vs source is 0.9997 mean.

**Q. What does LangGraph do that you cannot do yourself? _(Part B, pending)_**
To be answered after the LangGraph re-implementation. Expected answer: nothing I
cannot do by hand, but it provides state management, retries, branching, and
persistence as reusable primitives instead of hand-rolled loop code. The point of
building the loop by hand first is to know exactly what the framework is doing.

**Q. What still fails?**
Single-hop pinpoint answers where the value is in a table but the agent fetches
prose (it should prefer search_corpus/query_metrics for exact values). Complex
tables where `find_tables()` merges cells (values stay correct, grid does not).
The metrics table covers 132/275 papers so far (daily-budget-limited). The LLM
judge is only as good as the gold evidence it is shown. Corpus is arXiv-only.
