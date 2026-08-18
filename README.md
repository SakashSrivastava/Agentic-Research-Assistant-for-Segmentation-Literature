# Agentic Research Assistant for Segmentation Literature

A research assistant over medical-imaging segmentation literature. It ingests a
corpus of papers, extracts the reported results into a structured, queryable
table, and answers questions about them with citations. For harder questions it
plans multiple steps, calls tools, checks its own evidence, and then answers.

The project came out of working on AI + medTech research, where finding specific
results across a large body of segmentation papers is slow. The information is
there, but it is scattered across tables, prose, and appendices in inconsistent
formats: which architecture, on which dataset, with what Dice score, over how
many cases. This tool makes that searchable instead of manual.

The question it is built to answer:

> *Which architectures report the best Dice on head-and-neck segmentation, on
> which datasets, and how many cases was each evaluated on?*

That single query needs multi-step retrieval, numeric extraction from messy
tables, and comparison across papers. Plain embed-and-retrieve RAG cannot do it,
and getting it to work is the core of the project.

## Status

Work in progress, built one stage at a time. The text-preparation pipeline
(acquire through clean) is done and validated. Chunking, embedding, indexing,
retrieval, the agent, the web UI, and deployment are next (see the roadmap
below).

To be clear about what this is not: it is a working system with measured results,
not a production service. There is no monitoring, auth, or unattended operation.

## Where it stands right now

The pipeline turns raw PDFs into clean, section-aware, layout-faithful text, and
the output is checked against the source rather than assumed correct.

| | |
|---|---|
| Papers acquired | 276 (arXiv) |
| Usable after dedup / OCR filter | 275 |
| Excluded (scanned, no text layer) | 1 |
| Exact / near duplicates found | 0 |
| Year range | 2011 to 2026 |
| Results tables extracted | 732 |
| Two-column pages reconstructed | 2,435 |
| Papers with a Results / Experiments section | 250 (91%) |
| Parse text coverage vs. source PDFs | 0.9997 mean (min 0.987) |
| Table numeric fidelity (values that trace back to source) | 100% |
| Cleaning: headers/footers and equations stripped | 3,093 / 1,181 |
| Result numbers retained through cleaning | 98.3% (rest are page numbers) |

The corpus is weighted toward head-and-neck, organs-at-risk, orbital/ocular, and
small-structure segmentation, with a minority of broader-AI work (diffusion
models, foundation/SAM segmentation, general segmentation, AI-in-medtech) so
there is real architectural variety to compare.

## How the pipeline works

Most of this is plain deterministic code, not LLM calls, because hashing,
parsing, and chunking are faster, cheaper, and reproducible as ordinary code.
Only the parts that genuinely need judgement get an agent.

1. **Acquire:** pull papers from the arXiv API, one request every 3 seconds with
   backoff, trusting the API for metadata (title, authors, year, DOI) instead of
   scraping it from the PDF. Restartable: it skips anything already downloaded.
2. **Manifest + dedup:** SHA-256 fingerprint every PDF and normalize titles, so a
   preprint and its published version collapse to one record. This file also
   tracks how far each paper has moved through the pipeline.
3. **Parse (layout-aware):** read font size, boldness, and position of every text
   block, detect columns, and rebuild proper reading order. Two-column papers
   read straight across into nonsense otherwise. Tables come out separately, as
   structured rows, since that is where the scores live and flattening them
   destroys the numbers. Scanned PDFs are detected and set aside.
4. **Structure recovery:** group blocks into sections (Abstract, Methods,
   Results, and so on) using the font signals plus a section-name regex, and cut
   everything after References so the bibliography does not pollute retrieval.
5. **Clean:** apply six ordered transforms per section (Unicode NFKC,
   de-hyphenate across line breaks, strip repeated headers/footers, join
   paragraph lines, replace display equations with a token, collapse whitespace),
   logging what each removes. Tables are left untouched, and the equation filter
   is tuned to never mistake a result row (`0.88 ± 0.02`) for an equation.
6. **Enrich (one cheap LLM call/paper):** label each paper's anatomical target
   and imaging modality from its abstract, as controlled-vocabulary metadata for
   filtered retrieval ("search only CT head-and-neck papers"). The one part of
   ingestion that needs judgement rather than rules.
7. **Validation:** before trusting any of the above, compare the parsed output
   word for word against the raw PDFs and verify that extracted table numbers
   actually appear in the source. Results are saved to
   `data/parse_validation.json`.

Still to come: chunking, embedding, hybrid retrieval (vector + BM25 + reranker),
the hand-written agent loop, a verified metric-extraction agent, a Flask UI, and
deployment.

## Stack

Python 3.12, PyMuPDF, sentence-transformers (BGE-small), ChromaDB, rank_bm25,
bge-reranker cross-encoder, SQLite, Groq (gpt-oss-120b, free tier), Flask,
Docker / AWS.

The LLM sits behind a one-file wrapper (`src/llm.py`), so the provider is a
single-line change. I use Groq's free tier rather than a paid API; the
architecture is provider-agnostic.

One deliberate choice worth calling out: the agent loop is hand-written on the
LLM's tool-calling API (OpenAI-compatible), not LangChain or LangGraph. The goal
was to understand tool calling at the API level rather than hide it behind a
framework. A later phase re-implements it in LangGraph specifically to compare
the two.

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

copy .env.example .env               # then add your free GROQ_API_KEY
```

Run the pipeline stages (each is restartable and idempotent):

```bash
python -m src.acquire_arxiv          # download the corpus (~25 min, rate limited)
python -m src.manifest               # fingerprint + dedup
python -m src.parse                  # layout-aware parsing
python -m src.structure              # section recovery
python -m src.clean                  # text cleaning
python -m src.enrich                 # LLM metadata (anatomy + modality)
python -m src.validate_parse         # QA report vs. source PDFs
```

Sanity-check any single paper against its PDF:

```bash
python -m src.validate_parse --paper arxiv_1808.05238
```

## Repository layout

```
data/
  raw/            source PDFs + API metadata (immutable, re-downloadable)
  parsed/         page-level blocks with layout info
  clean/          section-structured, cleaned text
  chunks/         (coming) final chunks with metadata
  index/          (coming) chroma collection + bm25 index
  manifest.jsonl  per-paper record + pipeline status
  app.db          (coming) sqlite: extracted metrics table
src/              pipeline + agent code
evals/            (coming) retrieval + end-to-end evaluation sets
NOTES.md          plain-language design log, the reasoning behind each decision
```

`data/raw` is immutable on purpose: everything downstream is reproducible from
it, so a chunking change means re-running from `parsed/`, never re-downloading.

## Roadmap

- [x] Acquisition, manifest + dedup
- [x] Layout-aware parsing (columns, tables, OCR detection) + validation
- [x] Structure recovery (sections, reference truncation)
- [x] Cleaning (NFKC, de-hyphenation, header/footer and equation stripping)
- [x] Metadata enrichment (anatomy + modality via LLM, for filtered retrieval)
- [ ] Chunking, embedding, indexing
- [ ] Hybrid retrieval + reranking, with a labelled retrieval eval
- [ ] Hand-written planning / tool-calling agent + verified metric extraction
- [ ] Flask interface with inline citations and an agent trace
- [ ] Docker + AWS deployment
- [ ] LangGraph re-implementation and a head-to-head comparison

## Known limitations (so far)

- Corpus is arXiv-only for now. PubMed Central is a planned second source.
- `find_tables()` occasionally merges cells on complex tables. The values are
  always correct (verified), but the grid structure is not always perfect. The
  metric-extraction agent is designed to handle this by also reading the prose.
- Two papers have no detectable section headings (unusual formatting) and are
  flagged for a parse-repair pass.
- One scanned paper is excluded. Scanned cases are detected and skipped rather
  than ingested as empty text.
