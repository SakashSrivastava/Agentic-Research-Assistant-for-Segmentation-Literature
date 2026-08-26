"""End-to-end answer-quality eval: baseline RAG vs the hand-written agent.

For each question we produce an answer two ways:
  - baseline RAG : retrieve top-k chunks, stuff them into one LLM call
  - agent        : the multi-step tool-calling agent
Both answers are scored by an LLM judge on faithfulness, completeness, and
citation accuracy (1-5). Reports a comparison split by single/multi-hop, plus
tokens and latency per query.

  python -m src.eval_e2e --limit 6     # small, budget-friendly run
  python -m src.eval_e2e               # full question set
"""
from __future__ import annotations

import argparse
import json
import time

from src import agent, config, llm, retrieve

QUESTIONS = config.ROOT / "evals" / "retrieval_questions.jsonl"
RESULTS = config.ROOT / "evals" / "e2e_results.json"

BASELINE_SYS = (
    "Answer the question using ONLY the passages provided. Cite the paper_id "
    "(arxiv_...) for each claim. If the passages do not contain the answer, say so. "
    "Do not invent numbers. Be concise.")

JUDGE_SYS = (
    "You grade an answer to a question about medical image segmentation papers. "
    "You are given REFERENCE EVIDENCE (the gold passages that contain the answer). "
    "Grade the answer against that evidence, not your own knowledge.\n"
    "Return a JSON object with integer keys faithfulness, completeness, citation "
    "(each 1-5) and a short string 'note'.\n"
    "  faithfulness: is every claim/number in the answer supported by the evidence? "
    "A confident number that is NOT in the evidence is a hallucination -> score low.\n"
    "  completeness: does the answer cover what the evidence shows is askable?\n"
    "  citation: are paper_ids cited for the claims?\n"
    "Grade strictly.")


def baseline_rag(question: str, k: int = 5):
    passages = retrieve._passages()
    ctx = "\n\n".join(f"[{cid.split('::')[0]}] {passages.get(cid, '')[:600]}"
                      for cid, _ in retrieve.search(question, k=k, method="vector"))
    text, usage = llm.chat(BASELINE_SYS, f"Passages:\n{ctx}\n\nQuestion: {question}",
                           max_tokens=500)
    return text, usage.prompt_tokens + usage.completion_tokens


def _gold_evidence(gold_chunk_ids: list, cap: int = 2500) -> str:
    passages = retrieve._passages()
    ev = "\n\n".join(f"[{cid.split('::')[0]}] {passages.get(cid, '')}" for cid in gold_chunk_ids)
    return ev[:cap] if ev.strip() else "(no gold evidence available)"


def judge(question: str, answer: str, evidence: str):
    # gpt-oss reasons before answering, so give room for both the reasoning and
    # the JSON. A judge failure records null scores rather than killing the run.
    user = (f"REFERENCE EVIDENCE:\n{evidence}\n\n"
            f"Question: {question}\n\nAnswer to grade:\n{answer}")
    try:
        data, _ = llm.chat_json(JUDGE_SYS, user, max_tokens=900)
    except Exception as e:
        print(f"    (judge failed: {str(e)[:80]})")
        data = {}
    return {k: data.get(k) for k in ("faithfulness", "completeness", "citation")}


def _avg(rows, method, key, hop=None):
    sel = [r for r in rows if hop is None or r["hop"] == hop]
    vals = [r[method].get(key) or 0 for r in sel]
    return sum(vals) / len(vals) if vals else 0.0


def run(limit: int | None = None, per_hop: int | None = None) -> None:
    qs = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
    if per_hop:
        single = [q for q in qs if q["hop"] == "single"][:per_hop]
        multi = [q for q in qs if q["hop"] == "multi"][:per_hop]
        qs = single + multi
    elif limit:
        qs = qs[:limit]
    rows = []
    for q in qs:
        evidence = _gold_evidence(q.get("gold_chunk_ids", []))
        t = time.time()
        b_ans, b_tok = baseline_rag(q["question"])
        b = {**judge(q["question"], b_ans, evidence), "tokens": b_tok,
             "latency": round(time.time() - t, 1)}
        t = time.time()
        a_out = agent.answer(q["question"], verbose=False)
        a = {**judge(q["question"], a_out["answer"], evidence), "tokens": sum(a_out["tokens"]),
             "latency": round(time.time() - t, 1), "steps": a_out["steps"]}
        rows.append({"id": q["id"], "hop": q["hop"], "baseline": b, "agent": a})
        print(f"  {q['id']} ({q['hop']}): baseline F{b['faithfulness']}/C{b['completeness']} "
              f"| agent F{a['faithfulness']}/C{a['completeness']}")

    RESULTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n----- end-to-end eval -----")
    hdr = f"{'group':10} | {'method':8} | {'faith':>5} {'compl':>5} {'cite':>5} | {'tokens':>6} {'lat(s)':>6}"
    print(hdr + "\n" + "-" * len(hdr))
    for hop in (None, "single", "multi"):
        label = hop or "all"
        for method in ("baseline", "agent"):
            print(f"{label:10} | {method:8} | "
                  f"{_avg(rows, method, 'faithfulness', hop):5.2f} "
                  f"{_avg(rows, method, 'completeness', hop):5.2f} "
                  f"{_avg(rows, method, 'citation', hop):5.2f} | "
                  f"{_avg(rows, method, 'tokens', hop):6.0f} "
                  f"{_avg(rows, method, 'latency', hop):6.1f}")
    print(f"\nsaved {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end eval (baseline RAG vs agent).")
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--per-hop", type=int, default=None, help="N single + N multi (balanced)")
    args = ap.parse_args()
    run(limit=args.limit, per_hop=args.per_hop)
