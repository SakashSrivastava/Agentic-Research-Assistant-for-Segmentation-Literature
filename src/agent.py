"""The hand-written agent loop (Part A).

A plain Python while-loop drives tool calling on the LLM's API directly (no
LangChain/LangGraph): send the question + tool definitions, receive either a
final answer or tool-call requests, execute the requested Python functions, feed
the results back, and repeat until the model answers or a guardrail stops it.

  python -m src.agent "Which architectures report the best Dice on head and neck?"
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3

from src import config, llm, retrieve

MAX_ITERS = 8          # guardrail: never loop forever
TOKEN_BUDGET = 60_000  # guardrail: stop if the run gets too expensive

SYSTEM = (
    "You are a research assistant over a corpus of medical image segmentation papers. "
    "Answer the user's question using ONLY the tools and the evidence they return.\n\n"
    "Work in explicit steps:\n"
    "1. PLAN: break the question into the sub-facts you need (e.g. which metric, which "
    "anatomy, which datasets, how many cases).\n"
    "2. GATHER: call tools. `query_metrics` and `compare_across_papers` read a table of "
    "VERIFIED reported results (Dice, IoU, Hausdorff, ...) with case counts and paper ids "
    "-- use these for any numeric result. `search_corpus` finds relevant passages; "
    "`fetch_paper_section` reads a paper's full section.\n"
    "3. REFLECT: before answering, check the evidence is sufficient; if not, gather more.\n\n"
    "Prefer query_metrics / compare_across_papers for numeric results; use search_corpus "
    "only when the metrics table lacks what you need. Once the metric rows answer the "
    "question, STOP gathering and write the answer -- do not keep searching. "
    "Cite the paper_id for every claim. Never invent numbers -- if the tools do not return "
    "a value, say it is not available. Give a concise, structured final answer."
)


# ---------------- tools (the functions the model can ask us to run) ----------------

def _query_metrics(anatomical_target=None, metric_name=None, architecture=None,
                   dataset=None, min_value=None, limit=40):
    # Join papers so an anatomy filter matches either the per-row structure
    # ("Brain Stem") or the paper's region ("head and neck") from enrichment.
    q = ("SELECT m.architecture,m.dataset,m.anatomical_target,m.metric_name,"
         "m.metric_value,m.case_count,m.paper_id,p.anatomical_target "
         "FROM metrics m LEFT JOIN papers p ON m.paper_id=p.paper_id WHERE 1=1")
    p = []
    if anatomical_target:
        # Match on individual words so "brain tumor"/"glioma segmentation" still
        # hits rows tagged with the region ("brain") or a specific structure.
        words = [w for w in re.findall(r"[a-z0-9]+", anatomical_target.lower()) if len(w) >= 4]
        if words:
            clauses = []
            for w in words:
                clauses.append("(lower(m.anatomical_target) LIKE ? OR lower(p.anatomical_target) LIKE ?)")
                p += [f"%{w}%", f"%{w}%"]
            q += " AND (" + " OR ".join(clauses) + ")"
    if metric_name:
        mn = metric_name.lower()
        if mn in ("dice", "dsc"):   # Dice and DSC are the same metric
            q += " AND (lower(m.metric_name) LIKE '%dice%' OR lower(m.metric_name) LIKE '%dsc%')"
        else:
            q += " AND lower(m.metric_name) LIKE ?"
            p.append(f"%{mn}%")
    if architecture:
        q += " AND lower(m.architecture) LIKE ?"
        p.append(f"%{architecture.lower()}%")
    if dataset:
        q += " AND lower(m.dataset) LIKE ?"
        p.append(f"%{dataset.lower()}%")
    if min_value is not None:
        q += " AND m.metric_value >= ?"
        p.append(min_value)
    q += " ORDER BY m.metric_value DESC LIMIT ?"
    p.append(limit)
    con = sqlite3.connect(config.DB_PATH)
    rows = con.execute(q, p).fetchall()
    con.close()
    cols = ["architecture", "dataset", "anatomical_target", "metric_name",
            "metric_value", "case_count", "paper_id", "region"]
    return [dict(zip(cols, r)) for r in rows]


def _compare_across_papers(metric_name, anatomical_target=None):
    return _query_metrics(anatomical_target=anatomical_target, metric_name=metric_name, limit=25)


def _search_corpus(query, k=5):
    passages = retrieve._passages()
    out = []
    for cid, _ in retrieve.search(query, k=k, method="vector"):
        out.append({"chunk_id": cid, "paper_id": cid.split("::")[0],
                    "text": passages.get(cid, "")[:400]})
    return out


def _fetch_paper_section(paper_id, section):
    f = config.CLEAN_DIR / f"{paper_id}.json"
    if not f.exists():
        return {"error": f"unknown paper_id {paper_id}"}
    d = json.loads(f.read_text(encoding="utf-8"))
    for s in d["sections"]:
        if s["name"] == section or section.lower() in s["name"].lower():
            return {"paper_id": paper_id, "section": s["name"], "text": s["text"][:3000]}
    return {"error": f"section '{section}' not found",
            "available": [s["name"] for s in d["sections"]]}


DISPATCH = {
    "query_metrics": _query_metrics,
    "compare_across_papers": _compare_across_papers,
    "search_corpus": _search_corpus,
    "fetch_paper_section": _fetch_paper_section,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "query_metrics",
        "description": "Query the table of verified reported segmentation results. Filter by "
                       "anatomical_target, metric_name, architecture, dataset, or min_value. "
                       "Returns rows with architecture, dataset, anatomical_target, metric_name, "
                       "metric_value, case_count, paper_id, sorted by value descending.",
        "parameters": {"type": "object", "properties": {
            "anatomical_target": {"type": "string", "description": "e.g. 'head and neck', 'brain stem'"},
            "metric_name": {"type": "string", "description": "e.g. 'Dice', 'IoU', 'HD95'"},
            "architecture": {"type": "string"},
            "dataset": {"type": "string"},
            "min_value": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "compare_across_papers",
        "description": "Rank reported results across papers for a metric and (optional) anatomy. "
                       "Returns a ranked list of architecture/value/case_count/paper_id.",
        "parameters": {"type": "object", "properties": {
            "metric_name": {"type": "string"},
            "anatomical_target": {"type": "string"}}, "required": ["metric_name"]}}},
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Semantic search over paper passages. Returns top chunks with a text snippet.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_paper_section",
        "description": "Return the full text of one section of a paper (e.g. 'results', 'methods').",
        "parameters": {"type": "object", "properties": {
            "paper_id": {"type": "string"}, "section": {"type": "string"}},
            "required": ["paper_id", "section"]}}},
]


def _msg_to_dict(msg):
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
    return d


def answer(question: str, verbose: bool = True) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    trace, tokens_in, tokens_out = [], 0, 0

    for step in range(1, MAX_ITERS + 1):
        try:
            msg, usage = llm.chat_tools(messages, TOOLS)
        except Exception as ex:  # noqa: BLE001 - e.g. daily token budget exhausted
            return {"answer": f"(stopped: LLM unavailable - {ex})", "trace": trace,
                    "steps": step, "tokens": (tokens_in, tokens_out)}
        tokens_in += usage.prompt_tokens
        tokens_out += usage.completion_tokens
        messages.append(_msg_to_dict(msg))

        if not msg.tool_calls:                                  # model produced the answer
            return {"answer": msg.content, "trace": trace, "steps": step,
                    "tokens": (tokens_in, tokens_out)}

        for tc in msg.tool_calls:                               # run each requested tool
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = DISPATCH[name](**args)
            except Exception as ex:  # noqa: BLE001
                result = {"error": str(ex)}
            trace.append({"step": step, "tool": name, "args": args,
                          "result_preview": str(result)[:200]})
            if verbose:
                print(f"  [step {step}] {name}({args}) -> {str(result)[:110]}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, default=str)[:2000]})

        if tokens_in + tokens_out > TOKEN_BUDGET:               # guardrail: cost cap
            return {"answer": "(stopped: token budget exceeded)", "trace": trace,
                    "steps": step, "tokens": (tokens_in, tokens_out)}

    return {"answer": "(stopped: reached max iterations)", "trace": trace,
            "steps": MAX_ITERS, "tokens": (tokens_in, tokens_out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hand-written research agent.")
    ap.add_argument("question")
    args = ap.parse_args()
    print(f"\nQ: {args.question}\n")
    out = answer(args.question)
    print(f"\n=== ANSWER ({out['steps']} steps, "
          f"{sum(out['tokens'])} tokens) ===\n{out['answer']}")
