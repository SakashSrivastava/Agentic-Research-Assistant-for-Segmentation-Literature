"""Head-to-head: hand-written loop (src.agent) vs LangGraph (src.agent_langgraph).

Same model, tools, and prompt, so this isolates the orchestration. Compares steps,
tokens, and latency per question, split by single/multi-hop.

  python -m src.compare_agents --per-hop 2
"""
from __future__ import annotations

import argparse
import json
import time

from src import agent, agent_langgraph, config

QUESTIONS = config.ROOT / "evals" / "retrieval_questions.jsonl"
IMPLS = {"hand-written": agent.answer, "langgraph": agent_langgraph.answer}


def _avg(rows, impl, key):
    vals = [r[impl][key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def run(per_hop: int | None = 2, limit: int | None = None) -> None:
    qs = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
    if per_hop:
        single = [q for q in qs if q["hop"] == "single"][:per_hop]
        multi = [q for q in qs if q["hop"] == "multi"][:per_hop]
        qs = single + multi
    elif limit:
        qs = qs[:limit]

    rows = []
    for q in qs:
        row = {"id": q["id"], "hop": q["hop"]}
        for name, fn in IMPLS.items():
            t = time.time()
            out = fn(q["question"], verbose=False)
            row[name] = {"steps": out["steps"], "tokens": sum(out["tokens"]),
                         "latency": round(time.time() - t, 1), "chars": len(out["answer"])}
        rows.append(row)
        print(f"  {q['id']} ({q['hop']}): "
              f"hand-written {row['hand-written']['steps']}st/{row['hand-written']['tokens']}tok "
              f"| langgraph {row['langgraph']['steps']}st/{row['langgraph']['tokens']}tok")

    print("\n----- hand-written vs LangGraph (same model, tools, prompt) -----")
    hdr = f"{'impl':13} | {'avg steps':>9} {'avg tokens':>11} {'avg lat(s)':>11}"
    print(hdr + "\n" + "-" * len(hdr))
    for impl in IMPLS:
        print(f"{impl:13} | {_avg(rows, impl, 'steps'):9.1f} "
              f"{_avg(rows, impl, 'tokens'):11.0f} {_avg(rows, impl, 'latency'):11.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare the hand-written and LangGraph agents.")
    ap.add_argument("--per-hop", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(per_hop=args.per_hop, limit=args.limit)
