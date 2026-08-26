# Segmentation Literature Assistant — Complete Study & Interview Guide

This is your single source of truth for talking about the project in depth. Read it
top to bottom once, then re-skim sections 1, 14, and 16 the night before an interview.
Everything here maps to real code in this repo.

---

## 0. What the project is, in one line

An agentic RAG system over ~276 medical image segmentation papers: it ingests PDFs,
extracts reported results into a verified, queryable table, and answers comparative
questions with citations using a hand-written tool-calling agent, served as a secure
multi-user web app.

---

## 1. Pitches (memorize these)

**30-second version:**
"I built an agentic research assistant over ~276 medical image segmentation papers.
The hard part is that the answers live in inconsistent result tables, and the
questions are comparative, like 'which architectures report the best Dice on
head-and-neck, on which datasets, and how many cases each.' Plain RAG can't answer
that, so I built a layout-aware ingestion pipeline, a verified metric-extraction
step, and a hand-written tool-calling agent that decomposes the question and queries
a structured metrics table. I evaluated retrieval with a labelled set, wrapped it in
a secure multi-user Flask app, and containerized it with a CI/CD pipeline."

**2-minute version:** expand each clause with a number:
- Ingestion: layout-aware PDF parsing (columns + tables), section recovery, cleaning,
  section-aware chunking sized to the embedding model's real 512-token limit, BGE-small
  embeddings in ChromaDB plus a BM25 keyword index. 8,139 chunks over 275 usable papers.
- Retrieval eval: 52 labelled questions; dense vector wins (single-hop chunk recall@5
  ~0.59, paper-hit@5 ~0.85); BM25/hybrid/reranking all failed to beat it because the
  answers are in keyword-poor tables. Multi-hop single-query recall is ~0.03, which is
  the measured justification for an agent.
- Extraction: one LLM call per paper pulls (architecture, dataset, anatomy, metric,
  value, cases) into SQLite, with a verification pass that discards any number not found
  verbatim in the source (~5% caught) - the anti-hallucination guarantee.
- Agent: a plain Python loop on Groq's tool-calling API, no framework; tools read the
  verified table and the corpus; guardrails cap iterations and tokens.
- Product: multi-user app with accounts, per-user history, an admin dashboard,
  bring-your-own-key + a free daily allowance, and a semantic answer cache with a
  feedback loop. Hardened auth, then Dockerized with CI/CD. I also re-implemented the
  agent in LangGraph to compare hand-written vs framework.

---

## 2. The problem and why it is hard

Researchers need to compare reported results across many papers, but:
- The numbers live in **result tables** with wildly inconsistent formats.
- Questions are **comparative and multi-hop**: they need several tables from several
  papers plus their case counts, then a ranking.
- Plain "embed the question, retrieve top-k chunks, stuff into the LLM" fails because
  (a) one query embedding can't gather 4 different papers' tables, and (b) tables are
  keyword-poor so they retrieve poorly, and (c) LLMs hallucinate numbers.

The project exists to solve exactly those three failures: structure the numbers,
decompose the question, and verify every value.

---

## 3. Architecture at a glance

```
INGESTION (deterministic code, run once)
  arXiv PDFs
    -> parse (layout-aware: columns + tables)
    -> structure (sections, cut references)
    -> clean (6 transforms; tables untouched)
    -> enrich (1 LLM call/paper: anatomy + modality)
    -> chunk (section-aware, tables whole, deterministic IDs)
    -> embed (BGE-small, contextual prefix, cached)
    -> index -> ChromaDB (vectors) + rank_bm25 (keyword) + SQLite (papers)

EXTRACTION (1 LLM call/paper) -> SQLite metrics table (verified verbatim)

QUERY TIME
  user question
    -> hand-written agent (plan -> tool-call -> reflect -> answer)
         tools: query_metrics / compare_across_papers (SQLite metrics)
                search_corpus (vector)  |  fetch_paper_section (clean text)
    -> cited answer + agent trace

SERVING
  Flask web app: accounts, history, admin, BYOK, semantic cache + feedback
  Docker image (self-contained) + GitHub Actions CI/CD
```

Key principle: **use plain code where rules suffice (hashing, parsing, chunking) and
an LLM only where judgement is genuinely required (metadata enrichment, metric
extraction, the agent).** This is faster, cheaper, reproducible, and easy to defend.

---

## 4. The stack and why each choice

| Component | Choice | Why this, not the alternative |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | Fast, pure-Python, gives font size/position/columns and `find_tables()`. No Java (vs Tika/pdfbox). |
| Embeddings | BGE-small-en-v1.5 (384-dim) | Strong small retrieval model, runs on CPU, 512-token limit. Small enough to bake into a Docker image. |
| Vector store | ChromaDB | Local, persistent, no server. Good enough at this scale; no need for a hosted vector DB. |
| Keyword search | rank_bm25 | Exact matching for architecture/dataset names that vectors blur. |
| Structured store | SQLite | The metrics table is relational; SQL is the natural query language for the agent's tools. Zero-ops. |
| LLM | Groq `openai/gpt-oss-120b` (free tier) | I had no paid API budget. Groq is OpenAI-compatible with tool calling, so the agent ports to any provider by editing one wrapper file. |
| Web | Flask + Flask-Login | Lightweight, full control; Flask-Login for sessions. |
| Rate limiting | Flask-Limiter | Per-route abuse protection. |
| Sanitization | bleach | Whitelist-sanitize LLM-generated HTML (XSS). |
| Tokens | itsdangerous | Signed, expiring tokens for email verification + password reset. No DB table needed. |
| Agent (Part B) | LangGraph | To compare a framework against my hand-written loop. |
| Deploy | Docker + GitHub Actions | Reproducible image + CI/CD to AWS ECR/EC2. |

**One-liner to remember:** "The LLM sits behind a one-file wrapper (`src/llm.py`), so
the provider is a single-line change; the architecture is provider-agnostic."

---

## 5. Part A: the ingestion pipeline, stage by stage

Each stage is a `src/*.py` module, restartable and idempotent (a manifest tracks how
far each paper has progressed).

### 5.1 Acquire (`acquire_arxiv.py`)
- Pulls PDFs + metadata from the arXiv API, one request / 3s with backoff.
- Trusts the API for title/authors/year/DOI rather than scraping the PDF.
- Idempotent: seeds bucket counts from existing metadata so re-runs don't inflate.
- **Interview Q: why trust the API for metadata?** Because PDF-scraped metadata is
  noisy; the API is authoritative and free.

### 5.2 Manifest + dedup (`manifest.py`)
- SHA-256 fingerprint of each PDF + normalized-title matching, so a preprint and its
  published version collapse to one record.
- Also tracks `stage_completed` per paper -> the whole pipeline is restartable.
- **Why hashing?** Content-addressing gives free dedup and change detection.

### 5.3 Parse, layout-aware (`parse.py`)
- Reads font size, boldness, position of every block; detects columns; rebuilds reading
  order; pulls tables out separately with `find_tables()`.
- A fill-ratio filter (>= 0.4, >= 4 filled cells) rejects flowcharts that `find_tables`
  mistakes for tables.
- **Why layout-aware, not plain text?** Plain extraction reads two columns straight
  across into nonsense and flattens tables into a meaningless run of numbers. The scores
  live in the tables, so preserving structure is the whole game.

### 5.4 Structure recovery (`structure.py`)
- Groups blocks into sections (Abstract, Methods, Results...) via a section-name regex
  plus font signals; cuts everything after References so the bibliography doesn't
  pollute retrieval.

### 5.5 Clean (`clean.py`)
- Six ordered transforms per section: Unicode NFKC, de-hyphenate across line breaks,
  strip repeated headers/footers, join paragraph lines, replace display equations with a
  token, collapse whitespace. Tables are left untouched.
- **Gotcha I hit:** the equation filter deleted result rows like `0.88 ± 0.02`. Fix:
  removed +-*/ and ± from the "math" set, and refuse to treat a block as an equation if
  it has >= 2 decimals or >= 4 numbers. Great "attention to data quality" story.

### 5.6 Enrich (`enrich.py`)
- One cheap LLM call per paper labels `anatomical_target` and `imaging_modality` from the
  abstract, as controlled-vocabulary metadata for filtered retrieval.
- **Why is this LLM but chunking isn't?** Mapping free-text abstracts to a fixed
  vocabulary needs judgement; splitting on sections is a deterministic rule.

### 5.7 Chunk (`chunk.py`)
- Section-aware; tables kept whole; deterministic IDs `{paper_id}::{section}::{index}`.
- Sized to the model's real **512-token** limit (not the spec's 800).
- **Why deterministic IDs?** The retrieval gold set is labelled by chunk ID. If IDs
  changed each run, every labelled question would silently point at the wrong chunk and
  the eval would measure noise.
- **Why 512 not 800?** BGE-small silently truncates inputs over 512, so 800-token chunks
  would lose their tail at embed time.

### 5.8 Embed (`embed.py`)
- Prepends title + section to each chunk **for the vector only** (display text unchanged),
  embeds with BGE-small in batches of 64, L2-normalized. Content-hash cache so a re-run
  only embeds changed chunks.
- **Why the prefix?** A bare results chunk ("0.92 vs 0.88...") has no topic signal;
  prepending title+section gives context so a topical query lands near it.

### 5.9 Index (`index.py`)
- Loads vectors into ChromaDB (with paper_id/section/year/anatomy/modality as filter
  metadata), builds a persisted rank_bm25 index over the same chunks, populates a SQLite
  `papers` table.

### 5.10 Validation (`validate_parse.py`)
- Compares parsed output word-for-word against the raw PDFs and verifies extracted table
  numbers appear in the source. Text coverage 0.9997 mean; table numeric fidelity 100%.
- **Why validate?** "The output is checked against the source rather than assumed
  correct" is a credibility line most projects can't say.

---

## 6. Retrieval and the evaluation that shaped it (the credibility section)

`retrieve.py` implements four methods: vector, BM25, hybrid (Reciprocal Rank Fusion),
and cross-encoder rerank. `eval_retrieval.py` scores them on 52 labelled questions
(41 single-hop, 11 comparative multi-hop) with gold chunk IDs.

| method | chunk recall@5 | paper-hit@5 | single-hop r@5 | multi-hop r@5 |
|---|--:|--:|--:|--:|
| vector | 0.468 | 0.846 | 0.585 | 0.030 |
| bm25 | 0.000 | 0.827 | 0.000 | 0.000 |
| hybrid (RRF) | 0.019 | 0.846 | 0.024 | 0.000 |
| rerank | 0.404 | 0.827 | 0.512 | 0.000 |

**Two findings that drove the design (this is the part interviewers love):**
1. **Dense vector wins, and BM25/hybrid/reranking all failed to beat it - they hurt.**
   Why? The answers live in number-heavy tables: BM25 can't keyword-match them (chunk
   recall 0.000), the cross-encoder scores tables below fluent prose so reranking demotes
   the answer, and RRF rewards the prose that vector+BM25 agree on. I diagnosed this by
   adding a paper-level metric (paper-hit@5 stays ~0.85 for every method - they all find
   the right document; only exact-chunk pinpointing separates them).
2. **Every single-query method scores ~0.03 on multi-hop.** One query embedding can't
   gather tables from several papers. **This is the measured justification for the
   agent** - not a hunch, a number.

**Decision:** use vector for chunk retrieval, keep BM25/hybrid as a paper-level signal,
drop the reranker (no gain and ~0.6 passages/sec on CPU, far too slow).

**Why this is a strong story:** "I measured a textbook technique (reranking), found it
backfires on table-heavy data, and explained why" beats "I added reranking and it
helped." It shows you evaluate rather than cargo-cult.

---

## 7. Verified metric extraction (Agent 2, `extract.py`)

- One LLM call per paper over its tables + results text pulls structured rows:
  architecture, dataset, anatomical_target, metric_name, metric_value, case_count.
- **Verification pass:** every extracted value must appear **verbatim** in the source or
  it is discarded (~5% caught). Source is tables-first and capped so it fits the token
  budget.
- Stored to a SQLite `metrics` table (276 papers in the DB, 78 with verified metrics,
  829 rows, 267 architectures, 72 datasets, 15 anatomies).

**Interview Q: how do you stop the model inventing numbers?** "The LLM proposes,
string-matching against the source disposes. Any value not found verbatim in the paper
text is dropped before it reaches the table, so nothing the agent later cites is
fabricated."

**Interview Q: why is extraction agentic but chunking isn't?** Chunking is a rule;
extracting a clean tuple from an inconsistent table + caption + prose needs judgement.

---

## 8. The hand-written agent (the core of Part A, `agent.py`)

A plain Python `while` loop on Groq's OpenAI-compatible tool-calling API. No framework.

The loop:
```python
for step in range(1, MAX_ITERS + 1):
    msg, usage = llm.chat_tools(messages, TOOLS, api_key=api_key)   # model turn
    messages.append(_msg_to_dict(msg))
    if not msg.tool_calls:                    # model produced the final answer
        return {...answer...}
    for tc in msg.tool_calls:                 # run each requested tool
        result = DISPATCH[tc.function.name](**json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": ...})
    if tokens_in + tokens_out > TOKEN_BUDGET: # cost guardrail
        break
final, ti, to = _finalize(messages, api_key)  # force a tool-free answer from evidence
```

**Tools (`DISPATCH` / `TOOLS`):**
- `query_metrics(anatomical_target, metric_name, ...)` - reads the verified metrics table.
  Uses **word-level matching** so "brain tumor" matches rows tagged "brain", and treats
  Dice/DSC as synonyms.
- `compare_across_papers(metric_name, anatomical_target)` - ranked results across papers.
- `search_corpus(query, k)` - vector retrieval for conceptual questions.
- `fetch_paper_section(paper_id, section)` - full section text.

**Guardrails (this is a common interview question):**
- `MAX_ITERS = 8` - never loops forever.
- `TOKEN_BUDGET = 60_000` - cost cap.
- `_finalize()` - when a guardrail trips, force one tool-free answer from the evidence
  already gathered, so the agent never returns "gave up" if it has evidence.
- Malformed tool-call JSON from the model is caught and the model is nudged to retry,
  rather than crashing the run.

**Two robustness bugs I found via the eval and fixed (great story):**
1. The model sometimes emits invalid JSON for a tool call; Groq 400s the whole turn. The
   old loop treated that as fatal. Fix: catch it, tell the model to retry with valid JSON.
2. Hitting the step limit returned "reached max iterations" with no answer. Fix:
   `_finalize()` synthesizes from gathered evidence.

**Free-tier rate-limit engineering (`llm.py`):** Groq free tier is 8,000 tokens/min (TPM)
+ 200,000 tokens/day (TPD). I wait out TPM 429s (they clear in seconds) but fail fast on
TPD (its retry is ~40 min). I also learned to keep the running context lean (cap
`search_corpus` fan-out and snippet size) so the final answer call fits under the
per-minute limit - otherwise a bloated context makes the finalize call impossible.

**Interview Q: walk me through one trace.** "Best Dice on head-and-neck?" ->
`query_metrics(anatomical_target="head and neck", metric_name="Dice")` returns ranked
rows with value/case_count/paper_id -> the model has enough and writes a ranked, cited
answer. Two steps. Conceptual questions ("which methods use uncertainty estimation?")
use `search_corpus` once or twice, then synthesize.

---

## 9. The web application (`app.py`, `db.py`, `qcache.py`, `templates/`)

- **Accounts:** email/password signup + login (Flask-Login sessions). User data lives in
  a **separate writable `users.db`**, kept apart from the read-only corpus so it's never
  rebuilt with the index.
- **Per-user history:** every search is logged; past questions reopen their full answer.
- **Admin dashboard:** usage analytics across all users; admin granted only server-side
  (`manage_admin.py`), never through the website.
- **BYOK + free allowance:** 5 free shared-key searches/user/day, then the user pastes
  their own Groq key. The key lives in the browser (localStorage), is sent per request,
  used once, and never written to the DB or logs. This is what lets the app scale to many
  users past a single free-tier budget. `llm._client_for(api_key)` builds an ephemeral
  client for a supplied key.
- **Semantic answer cache + feedback (`qcache.py`):** a new question is embedded with the
  same BGE-small model and matched (cosine >= 0.93) against past answers; a close,
  not-downvoted, not-stale match is served instantly for **0 tokens**. Thumbs up/down
  decides what stays reusable. The cache is global, so popular questions are answered once
  for everyone.

**Interview Q: honest caveat on BYOK "never touches the server"?** "It never touches the
database or logs, which is the achievable promise. The key does pass through the server
in memory because the agent's tool loop runs server-side; the only way to avoid that is
to call the LLM from the browser, which would expose the key in client JS."

---

## 10. Security (`security.py`, `app.py`)

Treated as a deliberate hardening pass. Each measure and why:

- **Password hashing** (werkzeug pbkdf2/scrypt, salted) - never store plaintext.
- **Sessions expire** (12h idle) with HttpOnly + SameSite + Secure (in prod) cookies.
- **Email verification + password reset** via signed, expiring tokens (itsdangerous:
  24h / 1h). Expired or wrong-purpose tokens are rejected.
- **CSRF** token on every POST (session token compared on each request).
- **Rate limiting** (Flask-Limiter) on login, signup, AI generation, feedback - brute
  force, abuse, scraping.
- **XSS**: rendered answers are `bleach`-whitelist-sanitized before being marked safe
  (the one place we emit LLM-generated HTML).
- **SQL injection**: all queries parameterized; no string interpolation into SQL.
- **IDOR**: history is scoped to `current_user.id`; no endpoint takes a user-owned
  resource by raw id, so there's no direct-object-reference surface. The cache is global
  by design, not user-owned.
- **Admin cannot be self-assigned**: signup always creates a regular user; admin is
  granted only from the server shell (or an env-gated bootstrap on shell-less hosts).
- **Deployment headers**: HTTPS redirect + HSTS + CSP + X-Frame-Options + nosniff when
  `HTTPS_ONLY` is set (ProxyFix trusts the load balancer's X-Forwarded-Proto).
- **Audit logging** for auth attempts, admin actions, CSRF rejects, errors, rate-limit
  hits.
- **Secrets** only in environment variables; a secret scan confirmed none are committed;
  `.env` and `users.db` are gitignored.

**Interview Q: what was the subtlest security fix?** "Early on, admin was granted to
whoever signed up with ADMIN_EMAIL. With no email verification that's a
pre-registration race, so I removed all web paths to admin and made it server-side only."

---

## 11. Evaluation and metrics (know exactly what each number means)

**Metrics inside the papers (data we extract, not compute):**
- **Dice / DSC**: overlap of prediction P and ground truth G, `2|P∩G| / (|P|+|G|)`,
  0 to 1, higher better. "Dice" and "DSC" are the same thing.
- **IoU / Jaccard**: `|P∩G| / |P∪G|`, stricter than Dice, always lower.
- **HD95**: 95th-percentile Hausdorff boundary distance in mm, **lower is better**.
- Always paired with **case count** - a score on 300 cases beats the same score on 12.

**Metrics that grade the system:**
- **recall@5**: fraction of gold answer-chunks in the top 5. Did the evidence reach the
  model?
- **paper-hit@5**: was the right paper in the top 5, even if the exact chunk was missed?
  (I added this to separate "found the doc" from "found the table".)
- **MRR**: `1/rank` of the first correct hit, averaged. Rewards ranking near the top.
- **faithfulness / completeness / citation (1-5)**: an LLM judge grades each final answer
  against the gold evidence (a number not in the evidence is a hallucination).
- **tokens / latency per query**: the cost side; quantifies the agent's overhead vs plain
  RAG.
- **extraction verification catch rate (~5%)**: numbers discarded because not found
  verbatim - the anti-hallucination guarantee.

**End-to-end eval (`eval_e2e.py`):** answers each question with baseline RAG vs the agent,
LLM-judged. Its first run exposed the two agent robustness bugs (section 8) - the eval
earning its keep. The judge grades against the gold passages, so faithfulness is grounded.

---

## 12. Deployment (`Dockerfile`, `docker-compose.yml`, `.github/workflows/deploy.yml`)

- **Multi-stage Dockerfile**: builder installs CPU-only torch (keeps the image ~2 GB
  smaller than the CUDA build) + requirements; runtime copies site-packages, bakes the
  BGE model, and **bakes the 149 MB serve-time corpus** so the image is self-contained
  and runs on any host with no mounted volume. No secrets, no raw PDFs.
- **docker-compose.yml**: one-command local run; mounts the corpus read-only, the user DB
  on a writable volume.
- **CI/CD**: GitHub Actions builds -> pushes to ECR -> deploys on EC2 over SSH -> `/health`
  gate. Secrets in GitHub Secrets. Manual-trigger until AWS is provisioned so it doesn't
  red-X on every push.
- **The bug that only appeared in the container (great story):** every `search_corpus`
  failed in Docker with "attempt to write a readonly database". ChromaDB opens its SQLite
  backend read-write and needs to write WAL/lock files **even to read**, but I'd mounted
  the corpus read-only. Local dev never hit it because `data/` is writable there. Fix: a
  nested read-write mount over just the ChromaDB directory (baking the corpus into the
  image also fixes it, since the chroma dir is then in the writable image layer).
- Verified in-container: `/health` 200, both mounts work, security headers present, two
  gunicorn workers boot, baked model loads with no runtime download, and the agent answers
  real questions.

**Interview Q: why did it work locally but not in the container?** Exactly the ChromaDB
read-only story above - a textbook "reproducible environment surfaced a hidden write
dependency."

---

## 13. Part B: LangGraph vs the hand-written loop (`agent_langgraph.py`, `compare_agents.py`)

Re-implemented the same agent as a LangGraph `StateGraph`, holding model, tools, and
prompt constant so the only variable is orchestration. Two nodes - an `llm` node (one
model turn) and a `tools` node - with a conditional edge that loops back to `llm` while
the model asks for tools and routes to END when it answers. It drives our own `llm`
wrapper inside the nodes (not `langchain-groq`), which also avoids a version conflict.

**What LangGraph makes explicit:** the state schema, transitions, and stop condition are
declared as data; `recursion_limit` is the loop guard. Nicer for branching, retries,
checkpointing, and visualization as agents grow.

**What was easier by hand:** forcing a final answer when a guardrail trips - a plain
function call vs a dedicated node/edge around LangGraph's recursion limit.

**The line to say:** "Building the loop by hand first is what lets me tell you exactly
what the framework is doing under the hood. For a single linear plan-act loop, the
framework is more machinery for the same behaviour; its value shows up with complexity."

---

## 14. Key design decisions and trade-offs (the "why" bank)

Have a crisp answer ready for each:
- **Rules vs LLM**: deterministic code for hashing/parsing/chunking; LLM only for
  enrichment, extraction, and the agent. Cheaper, reproducible, defensible.
- **Vector over hybrid/rerank**: measured; hybrid/rerank hurt on table-heavy data.
- **Agent over plain RAG**: multi-hop single-query recall ~0.03 - measured necessity.
- **Verify-verbatim extraction**: trust; nothing cited is invented.
- **Hand-written agent first, LangGraph second**: understand the primitive, then compare.
- **Groq free tier + provider-agnostic wrapper**: no budget; ports to Claude/OpenAI by
  editing one file.
- **Separate user DB, read-only corpus**: user data is precious + writable; corpus is
  regenerable + immutable.
- **BYOK + free allowance + semantic cache**: scale past a single free-tier budget and
  make repeats free.
- **Admin server-side only**: no web path to privilege escalation.
- **Chunk to 512, deterministic IDs**: match the model's real limit; keep the eval valid.

---

## 15. Known limitations and future work (say these before they ask)

- Corpus is arXiv-only; PubMed Central is a planned second source.
- `find_tables()` occasionally merges cells on complex tables (values stay correct,
  grid isn't perfect); the extractor also reads prose to compensate.
- Metric extraction covers 78/276 papers so far (free-tier token limited); it grows daily.
- The LLM judge is only as good as the gold evidence it's shown.
- Rate-limit storage is in-memory (per worker); use Redis for multi-worker/instance.
- No live 24/7 hosting (cost); demoed locally or via a tunnel. Fully containerized and
  CI/CD-ready for AWS.

---

## 16. Interview questions with strong answers (rapid-fire)

**RAG / agents**
- *What is agentic RAG vs plain RAG?* Plain RAG = retrieve then generate, one shot.
  Agentic = the model plans, calls tools, inspects results, and decides the next action
  in a loop. Needed here because comparative questions require multiple targeted lookups.
- *What is a tool-calling loop?* Send the question + tool schemas; the model replies with
  either a final answer or tool-call requests; you execute the tool, feed the result back
  as a tool message, and repeat.
- *How do you stop an agent looping forever?* Max-iteration cap + token budget, and a
  finalize step that produces a best-effort answer when a guardrail trips.
- *Hallucination control?* Verified extraction (verbatim check) + "cite paper_id for
  every claim, say 'not available' if the tools return nothing".

**Retrieval / eval**
- *Why keep both vector and BM25?* They fail oppositely - vector blurs exact tokens, BM25
  misses synonyms; BM25 still finds the right paper 83% of the time via names.
- *Why did reranking hurt?* Cross-encoder scores keyword-poor tables below fluent prose.
- *What's recall@5 vs paper-hit@5?* Exact-chunk vs right-document; the gap explained the
  method behaviour.

**Security / systems**
- *How is auth secured?* Hashed passwords, expiring sessions, verification + reset tokens,
  CSRF, rate limiting, XSS sanitization, secure headers, audit logging.
- *IDOR prevention?* Ownership scoping on every user resource; no raw-id endpoints.
- *Scaling past the free tier?* BYOK per-user keys + a semantic answer cache (0-token
  repeats).

**Deployment**
- *Walk me through the Docker setup.* Multi-stage, CPU torch, baked model + corpus,
  self-contained, `/health`, gunicorn, CI/CD to ECR/EC2.
- *A bug you fixed in deployment?* ChromaDB read-only-mount story.

**Behavioural / design**
- *Hardest part?* Getting multi-hop comparative questions to work - it drove the whole
  agent + verified-table design, backed by the ~0.03 multi-hop number.
- *What would you do differently / next?* PubMed source, finish extraction, Redis for
  rate limits, a stronger evidence-grounded judge, live hosting.

---

## 17. The live demo script (exact commands)

Have Docker Desktop running. Open a fresh terminal so `docker` is on PATH.

**Option A - Docker (shows containerization; most impressive):**
```bash
docker compose up -d --build          # first time builds (~10 min); later runs are instant
```
Open http://localhost:5000 , sign up, then in another terminal:
```bash
docker compose exec web python -m src.manage_admin grant YOUR_EMAIL
```
Refresh the page - the Admin link appears. Ask a question in the UI, show the cited
answer + collapsible agent trace + the admin dashboard. Show the CLI agent too:
```bash
docker compose exec web python -m src.agent "Which architectures report the best Dice on head and neck segmentation?"
```
Stop it when done:
```bash
docker compose down
```

**Option B - local venv (fastest to start, no Docker):**
```powershell
.\venv\Scripts\python.exe -m src.app          # then open http://localhost:5000
```
```powershell
.\venv\Scripts\python.exe -m src.manage_admin grant YOUR_EMAIL
```
Run the test suite live (impressive - 26 green checks):
```powershell
.\venv\Scripts\python.exe tests\test_app.py
```
Show the retrieval eval numbers:
```powershell
.\venv\Scripts\python.exe -m src.eval_retrieval
```
Show Part B (LangGraph) and the comparison:
```powershell
.\venv\Scripts\python.exe -m src.agent_langgraph "Which methods use uncertainty estimation in medical image segmentation?"
```

**Optional - share a public URL during a remote interview (free, no card):**
```bash
cloudflared tunnel --url http://localhost:5000
```
This prints a temporary `https://...trycloudflare.com` URL that proxies to your local
app while your machine runs. (Install cloudflared first; the URL changes each run.)

**Demo narration order (2-3 min):** stats dashboard -> ask a head-and-neck question ->
point at the citations and the agent trace (tools it called) -> open the admin dashboard
-> mention BYOK + the semantic cache -> optionally run the test suite.

---

## 18. Glossary (quick reference)

- **RAG**: Retrieval-Augmented Generation - retrieve relevant text, then generate an
  answer conditioned on it.
- **Agent / tool-calling**: an LLM that decides to call functions (tools) and reacts to
  their outputs in a loop.
- **Embedding**: a vector representation of text; similar meaning -> nearby vectors.
- **Cosine similarity**: dot product of unit vectors; how "close" two embeddings are.
- **BM25**: a classic keyword-ranking function (term frequency + rarity).
- **RRF (Reciprocal Rank Fusion)**: combine ranked lists by summing 1/(k+rank).
- **Cross-encoder / reranker**: scores a (query, passage) pair jointly for relevance.
- **Dice / IoU / HD95**: segmentation quality metrics (see section 11).
- **recall@5 / MRR / paper-hit@5**: retrieval quality metrics (see section 11).
- **CSRF / XSS / IDOR**: web vulns - forged requests / injected scripts / accessing
  another user's object by id.
- **TPM / TPD**: Groq tokens-per-minute / per-day rate limits.
- **BYOK**: bring your own (API) key.
- **StateGraph**: LangGraph's declarative graph of nodes + edges with shared state.
```
