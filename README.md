# Agentic Research Assistant for Segmentation Literature

An assistant over medical-imaging **segmentation** literature. It retrieves
relevant passages, extracts reported results (Dice, IoU, Hausdorff, case counts)
into a structured, queryable table, and for complex questions plans multiple
steps, calls tools, and synthesises an answer with citations.

**Target query the system must answer:**
> Which architectures report the best Dice on head-and-neck segmentation, on
> which datasets, and how many cases was each evaluated on?

That needs multi-step retrieval, structured numeric extraction from tables, and
comparison across papers — which plain RAG cannot do.

> ⚠️ Status: **work in progress**, built one day at a time. This is a working
> system with measured results, **not** production-ready (no monitoring, auth,
> tests-at-scale, or unattended operation).

## Stack
Python 3.12 · PyMuPDF · sentence-transformers (BGE-small) · ChromaDB ·
rank_bm25 · bge-reranker cross-encoder · SQLite · Claude (anthropic SDK) ·
Flask · Docker / AWS.

## Setup
```bash
# 1. Create and activate the virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env   # then edit .env with your ANTHROPIC_API_KEY
```

## Repository layout
```
data/
  raw/       immutable source PDFs (re-downloadable via acquisition script)
  parsed/    page-level blocks with layout info
  clean/     normalised text with sections
  chunks/    final chunks with metadata
  index/     chroma collection + bm25 index
  app.db     sqlite: manifest + extracted metrics
  manifest.jsonl
src/         pipeline + agent code
evals/       retrieval and end-to-end evaluation sets
NOTES.md     plain-language design log (the interview prep)
```

## Ingestion report
_(printed here after Day 2)_

## Results
_(retrieval table after Day 3; end-to-end + extraction accuracy after Day 7)_

## Limitations
_(honest list, filled in as we go)_
