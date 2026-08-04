# Agentic Research Assistant for Segmentation Literature

I read a lot of medical-imaging segmentation papers, and I kept losing hours to
the same boring task: hunting through PDFs to find *which architecture reported
which Dice score, on which dataset, over how many cases*. The information is all
there — but it's scattered across tables, prose, and appendices in wildly
inconsistent formats. This project is my attempt to make that queryable.

It ingests a corpus of papers, pulls the reported results into a structured
table, and answers questions about them with citations. For the hard questions
it doesn't just do one retrieval — it plans, calls tools, checks its own
evidence, and then answers.

The question I'm building it to answer:

> *Which architectures report the best Dice on head-and-neck segmentation, on
> which datasets, and how many cases was each evaluated on?*

That single query needs multi-step retrieval, numeric extraction from messy
tables, and comparison across papers. Plain "embed-and-retrieve" RAG can't do it,
and getting it to work is basically the whole point of the project.

## Status

Work in progress, built one stage at a time. The **ingestion pipeline is done
and validated**; retrieval, the agent, the web UI, and deployment are next (see
the roadmap below).

To be clear about what this *isn't*: it's a working system with measured
results, not a production service. There's no monitoring, auth, or unattended
operation, and I'm not going to pretend otherwise.

## Where it stands right now

The pipeline turns raw PDFs into clean, section-aware, layout-faithful text — and
I've checked that it actually matches the source, rather than assuming it does.

| | |
|---|---|
| Papers acquired | 276 (arXiv) |
| Usable after dedup / OCR filter | 275 |
| Excluded (scanned, no text layer) | 1 |
| Exact/near duplicates found | 0 |
| Year range | 2011 – 2026 |
| Results tables extracted | 732 |
| Two-column pages reconstructed | 2,435 |
| Papers with a Results/Experiments section | 250 (91%) |
| Parse text coverage vs. source PDFs | 0.9997 mean (min 0.987) |
| Table numeric fidelity (values that trace back to source) | 100% |

The corpus is deliberately weighted toward my domain — head-and-neck,
organs-at-risk, orbital/ocular, small-structure segmentation — with a minority of
broader-AI work (diffusion models, foundation/SAM segmentation, general
segmentation, AI-in-medtech) so there's real architectural variety to compare.

## How the pipeline works

Most of this is plain deterministic code, not LLM calls — because hashing,
parsing, and chunking are faster, cheaper, and reproducible as ordinary code.
Only the parts that genuinely need judgement get an agent.

1. **Acquire** — pull papers from the arXiv API, one request every 3 seconds with
   backoff, trusting the API for metadata (title/authors/year/DOI) instead of
   scraping it from the PDF. Restartable: it skips anything already downloaded.
2. **Manifest + dedup** — SHA-256 fingerprint every PDF and normalize titles, so
   a preprint and its published version collapse to one record. This file also
   tracks how far each paper has moved through the pipeline.
3. **Parse (layout-aware)** — read font size, boldness, and position of every
   text block, detect columns, and rebuild proper reading order. Two-column
   papers read straight-across into nonsense otherwise. Tables come out
   *separately, as structured rows* — that's where the scores live and flattening
   them destroys the numbers. Scanned PDFs are detected and set aside.
4. **Structure recovery** — group blocks into sections (Abstract, Methods,
   Results, …) using the font signals plus a section-name regex, and cut
   everything after References so the bibliography doesn't pollute retrieval.
5. **Validation** — before trusting any of the above, compare the parsed output
   word-for-word against the raw PDFs and verify that extracted table numbers
   actually appear in the source. Results are saved to
   `data/parse_validation.json`.

Still to come: cleaning, chunking, embedding, hybrid retrieval (vector + BM25 +
reranker), the hand-written agent loop, a verified metric-extraction agent, a
Flask UI, and deployment.

## Stack

Python 3.12 · PyMuPDF · sentence-transformers (BGE-small) · ChromaDB · rank_bm25
· bge-reranker cross-encoder · SQLite · Claude (Anthropic SDK) · Flask · Docker /
AWS.

One deliberate choice worth calling out: the agent loop is **hand-written on the
Anthropic tool-calling API**, not LangChain/LangGraph. I wanted to understand
tool calling at the API level, not hide it behind a framework. (A later phase
re-implements it in LangGraph specifically to compare the two.)

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

copy .env.example .env               # then add your ANTHROPIC_API_KEY
```

Run the pipeline stages (each is restartable and idempotent):

```bash
python -m src.acquire_arxiv          # download the corpus (~25 min, rate-limited)
python -m src.manifest               # fingerprint + dedup
python -m src.parse                  # layout-aware parsing
python -m src.structure              # section recovery
python -m src.validate_parse         # QA report vs. source PDFs
```

Sanity-check any single paper against its PDF:

```bash
python -m src.validate_parse --paper arxiv_1808.05238
```

## Repository layout

```
data/
  raw/            source PDFs + API metadata (immutable; re-downloadable)
  parsed/         page-level blocks with layout info
  clean/          section-structured text
  chunks/         (coming) final chunks with metadata
  index/          (coming) chroma collection + bm25 index
  manifest.jsonl  per-paper record + pipeline status
  app.db          (coming) sqlite: extracted metrics table
src/              pipeline + agent code
evals/            (coming) retrieval + end-to-end evaluation sets
NOTES.md          plain-language design log — the "why" behind each decision
```

`data/raw` is immutable on purpose: everything downstream is reproducible from
it, so a chunking change means re-running from `parsed/`, never re-downloading.

## Roadmap

- [x] Acquisition, manifest + dedup
- [x] Layout-aware parsing (columns, tables, OCR detection) + validation
- [x] Structure recovery (sections, reference truncation)
- [ ] Cleaning, chunking, embedding, indexing
- [ ] Hybrid retrieval + reranking, with a labelled retrieval eval
- [ ] Hand-written planning/tool-calling agent + verified metric extraction
- [ ] Flask interface with inline citations and an agent trace
- [ ] Docker + AWS deployment
- [ ] LangGraph re-implementation and a head-to-head comparison

## Known limitations (so far)

- Corpus is arXiv-only for now; PubMed Central is a planned second source.
- `find_tables()` occasionally merges cells on complex tables — the *values* are
  always correct (verified), but the grid structure isn't always perfect. The
  metric-extraction agent is designed to handle this by also reading the prose.
- Two papers have no detectable section headings (unusual formatting) and are
  flagged for a parse-repair pass.
- One scanned paper is excluded; I detect and skip OCR cases rather than ingest
  empty text.
