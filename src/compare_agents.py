"""Head-to-head: hand-written loop (src.agent) vs LangGraph (src.agent_langgraph).

Same model, tools, and prompt, so this isolates the orchestration. For each
question it records steps, tokens, latency, and whether a real answer was produced,
for both implementations, then writes a CSV and a short Markdown report.

  python -m src.compare_agents --custom          # curated question set (default)
  python -m src.compare_agents --per-hop 2        # sample from the labelled eval set
"""
from __future__ import annotations

import argparse
import csv
import json
import time

from src import agent, agent_langgraph, config

QUESTIONS = config.ROOT / "evals" / "retrieval_questions.jsonl"
IMPLS = {"hand-written": agent.answer, "langgraph": agent_langgraph.answer}

# Curated questions spanning the agent's real use cases: comparative metric lookups,
# a different-anatomy lookup, and a conceptual (search-heavy) question.
CUSTOM_QUESTIONS = [
    ("head_neck_dice", "Which architectures report the best Dice on head and neck segmentation, and on how many cases each?"),
    ("pancreas_dice", "What Dice scores are reported for pancreas segmentation, and by which methods?"),
    ("uncertainty", "Which methods use uncertainty estimation in medical image segmentation?"),
    ("brain_unet", "Compare U-Net variants for brain tumor segmentation by Dice."),
]


def _avg(rows, impl, key):
    vals = [r[impl][key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _one(fn, question):
    t = time.time()
    out = fn(question, verbose=False)
    ans = out["answer"]
    return {"steps": out["steps"], "tokens": sum(out["tokens"]),
            "latency": round(time.time() - t, 1), "chars": len(ans),
            "ok": 0 if ans.strip().startswith("(stopped") else 1}


def _write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question",
                    "hw_steps", "hw_tokens", "hw_latency_s", "hw_ok",
                    "lg_steps", "lg_tokens", "lg_latency_s", "lg_ok"])
        for r in rows:
            hw, lg = r["hand-written"], r["langgraph"]
            w.writerow([r["id"], r["question"],
                        hw["steps"], hw["tokens"], hw["latency"], hw["ok"],
                        lg["steps"], lg["tokens"], lg["latency"], lg["ok"]])


def _write_report(rows, path):
    lines = [
        "# Hand-written agent vs LangGraph: head-to-head",
        "",
        "Same model (Groq gpt-oss-120b), same tools, same system prompt. The only",
        "variable is orchestration: a hand-written Python while-loop (`src/agent.py`)",
        "vs a LangGraph `StateGraph` (`src/agent_langgraph.py`). Lower steps/tokens/",
        "latency is better; `ok` is whether a real (non-stopped) answer was produced.",
        "",
        "## Per-question",
        "",
        "| Question | Impl | Steps | Tokens | Latency (s) | Answer |",
        "|---|---|--:|--:|--:|:--:|",
    ]
    for r in rows:
        for impl in IMPLS:
            d = r[impl]
            lines.append(f"| {r['id']} | {impl} | {d['steps']} | {d['tokens']:,} | "
                         f"{d['latency']} | {'ok' if d['ok'] else 'stopped'} |")
    lines += ["", "## Averages", "",
              "| Impl | Avg steps | Avg tokens | Avg latency (s) | Answered |",
              "|---|--:|--:|--:|--:|"]
    n = len(rows) or 1
    for impl in IMPLS:
        answered = sum(r[impl]["ok"] for r in rows)
        lines.append(f"| {impl} | {_avg(rows, impl, 'steps'):.1f} | "
                     f"{_avg(rows, impl, 'tokens'):,.0f} | {_avg(rows, impl, 'latency'):.1f} | "
                     f"{answered}/{n} |")
    lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def run(custom: bool, per_hop: int | None, limit: int | None, out_prefix: str) -> None:
    if custom:
        items = [{"id": qid, "question": q, "hop": "custom"} for qid, q in CUSTOM_QUESTIONS]
    else:
        qs = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
        if per_hop:
            items = ([q for q in qs if q["hop"] == "single"][:per_hop]
                     + [q for q in qs if q["hop"] == "multi"][:per_hop])
        else:
            items = qs[:limit] if limit else qs

    rows = []
    for q in items:
        row = {"id": q["id"], "question": q["question"]}
        for name, fn in IMPLS.items():
            row[name] = _one(fn, q["question"])
        rows.append(row)
        print(f"  {q['id']}: hand-written {row['hand-written']['steps']}st/"
              f"{row['hand-written']['tokens']}tok ok={row['hand-written']['ok']} | "
              f"langgraph {row['langgraph']['steps']}st/{row['langgraph']['tokens']}tok "
              f"ok={row['langgraph']['ok']}")

    csv_path = config.ROOT / "evals" / f"{out_prefix}.csv"
    md_path = config.ROOT / "evals" / f"{out_prefix}.md"
    _write_csv(rows, csv_path)
    _write_report(rows, md_path)

    print("\n----- averages (same model, tools, prompt) -----")
    for impl in IMPLS:
        answered = sum(r[impl]["ok"] for r in rows)
        print(f"{impl:13} | avg steps {_avg(rows, impl, 'steps'):.1f} | "
              f"avg tokens {_avg(rows, impl, 'tokens'):,.0f} | "
              f"avg latency {_avg(rows, impl, 'latency'):.1f}s | answered {answered}/{len(rows)}")
    print(f"\nsaved {csv_path}\nsaved {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare the hand-written and LangGraph agents.")
    ap.add_argument("--custom", action="store_true", help="use the curated question set (default)")
    ap.add_argument("--per-hop", type=int, default=None, help="sample N single + N multi from the eval set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="agent_comparison", help="output filename prefix (in evals/)")
    args = ap.parse_args()
    use_custom = args.custom or (args.per_hop is None and args.limit is None)
    run(custom=use_custom, per_hop=args.per_hop, limit=args.limit, out_prefix=args.out)
