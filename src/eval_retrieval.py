"""Day 3 retrieval evaluation.

Runs each retrieval method against evals/retrieval_questions.jsonl and reports
recall@5, hit@5, MRR, split by single-hop vs multi-hop. Shows the effect of each
change in isolation: vector (baseline) -> +BM25 (hybrid RRF) -> +reranking.

  python -m src.eval_retrieval
"""
from __future__ import annotations

import json

from src import config, retrieve

METHODS = ["vector", "bm25", "hybrid", "rerank"]
QUESTIONS = config.ROOT / "evals" / "retrieval_questions.jsonl"
RESULTS = config.ROOT / "evals" / "retrieval_results.json"


def _paper(cid: str) -> str:
    return cid.split("::")[0]


def _eval_method(qs, method: str, topn: int = 20):
    rows = []
    for q in qs:
        ids = [cid for cid, _ in retrieve.search(q["question"], k=topn, method=method)]
        gold = set(q["gold_chunk_ids"])
        gold_papers = {_paper(g) for g in gold}
        top5 = ids[:5]
        rows.append((q["hop"],
                     len(gold & set(top5)) / len(gold),                  # chunk recall@5
                     1.0 if gold & set(top5) else 0.0,                   # chunk hit@5
                     next((1 / (i + 1) for i, c in enumerate(ids) if c in gold), 0.0),  # MRR
                     1.0 if gold_papers & {_paper(c) for c in top5} else 0.0))  # paper hit@5
    return rows


def _agg(rows, hop=None):
    sel = [r for r in rows if hop is None or r[0] == hop]
    n = len(sel) or 1
    return {"recall@5": sum(r[1] for r in sel) / n,
            "hit@5": sum(r[2] for r in sel) / n,
            "mrr": sum(r[3] for r in sel) / n,
            "paper_hit@5": sum(r[4] for r in sel) / n}


def run() -> None:
    qs = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
    print(f"{len(qs)} questions "
          f"({sum(q['hop']=='single' for q in qs)} single, "
          f"{sum(q['hop']=='multi' for q in qs)} multi)\n")
    header = (f"{'method':8} | {'recall@5':>8} {'paperhit@5':>10} {'MRR':>5} "
              f"| {'single r@5':>10} {'multi r@5':>9}")
    print(header)
    print("-" * len(header))

    results = {}
    for method in METHODS:
        rows = _eval_method(qs, method)
        a, s, m = _agg(rows), _agg(rows, "single"), _agg(rows, "multi")
        results[method] = {"all": a, "single": s, "multi": m}
        print(f"{method:8} | {a['recall@5']:8.3f} {a['paper_hit@5']:10.3f} {a['mrr']:5.2f} "
              f"| {s['recall@5']:10.3f} {m['recall@5']:9.3f}")

    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {RESULTS}")


if __name__ == "__main__":
    run()
