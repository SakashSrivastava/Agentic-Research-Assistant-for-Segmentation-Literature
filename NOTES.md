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
  - **Judge caveat.** The judge scores without ground-truth evidence, so it
    rewards a confident baseline number and penalizes the agent's honest "not
    available" -- even if the baseline number is the hallucinated one. A stronger
    judge should see the gold passage. Recorded as a known limitation; the raw
    per-question answers matter more than the aggregate here.

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
