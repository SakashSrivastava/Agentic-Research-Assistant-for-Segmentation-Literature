"""Agent 2: metric extraction into the SQLite `metrics` table.

For each paper the LLM reads its results/experiments text + tables and extracts
structured records (architecture, dataset, anatomy, metric, value, case count).
A verification pass then DISCARDS any metric value that does not appear verbatim
in the source text, so the model cannot invent scores. Idempotent per paper.

  python -m src.extract --limit 3
  python -m src.extract
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3

from src import config, llm, manifest

METRIC_NAMES = ("Dice, DSC, IoU, Jaccard, Hausdorff, HD95, ASSD, MSD, ASD, "
                "Accuracy, Sensitivity, Specificity, Precision, Recall, AUC, F1")

SYSTEM = (
    "You extract reported quantitative segmentation results from a paper's results "
    "text and tables. Return a JSON object {\"metrics\": [...]}. Each item has keys:\n"
    "  architecture: the model/method name (e.g. 'U-Net', 'AnatomyNet', 'nnU-Net')\n"
    "  dataset: dataset or challenge name if stated, else null\n"
    "  anatomical_target: the organ/structure the score is for (e.g. 'brain stem', "
    "'optic chiasm', 'liver'), or 'overall' for an aggregate\n"
    f"  metric_name: one of {METRIC_NAMES}\n"
    "  metric_value: the numeric value EXACTLY as written in the source (e.g. 85.1 or 0.744)\n"
    "  case_count: number of cases/patients/images the score was computed on if stated, else null\n"
    "  source: a short label or quote showing where the value appears\n"
    "Extract only numbers explicitly present in the source. Do NOT invent, infer, or "
    "compute values. When several methods are compared, extract each method's values. "
    "Return at most 40 of the most important records."
)


def paper_source(pid: str) -> str:
    d = json.loads((config.CLEAN_DIR / f"{pid}.json").read_text(encoding="utf-8"))
    parts = []
    # Tables first: they hold the reported numbers, so they must survive the cap.
    for t in d.get("tables", []):
        rows = "\n".join(" | ".join(str(c) for c in r if c is not None) for r in t["rows"])
        parts.append(f"[table in {t['section']}]\n{rows}")
    for s in d["sections"]:
        if s["name"] in ("results", "experiments"):
            parts.append(f"[{s['name']}]\n{s['text']}")
    return "\n\n".join(parts)[:5000]   # cap input: dense tables are ~1 tok/char, keep request under Groq's 8k-tok/min limit


def _num(v):
    m = re.search(r"-?\d+\.?\d*", str(v))
    return m.group() if m else None


def verify(records, source):
    """Keep only records whose metric value appears verbatim in the source."""
    src = re.sub(r"\s+", "", source)
    kept = []
    for r in records:
        num = _num(r.get("metric_value"))
        if num and len(num.replace(".", "")) >= 3 and num in src:
            kept.append(r)
    return kept


def _to_float(v):
    n = _num(v)
    return float(n) if n else None


def _to_int(v):
    m = re.search(r"\d+", str(v)) if v is not None else None
    return int(m.group()) if m else None


def store(con, pid, records):
    con.execute("DELETE FROM metrics WHERE paper_id=?", (pid,))
    for r in records:
        con.execute(
            "INSERT INTO metrics(paper_id,architecture,dataset,anatomical_target,"
            "metric_name,metric_value,case_count,source) VALUES(?,?,?,?,?,?,?,?)",
            (pid, r.get("architecture"), r.get("dataset"), r.get("anatomical_target"),
             r.get("metric_name"), _to_float(r.get("metric_value")),
             _to_int(r.get("case_count")), str(r.get("source"))[:200]))


def _ensure_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS metrics(
        id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT, architecture TEXT,
        dataset TEXT, anatomical_target TEXT, metric_name TEXT, metric_value REAL,
        case_count INTEGER, source TEXT)""")


def run(limit: int | None = None, force: bool = False) -> None:
    con = sqlite3.connect(config.DB_PATH)
    _ensure_table(con)
    papers = manifest.active_papers()
    if limit:
        papers = papers[:limit]
    done = skipped = failed = 0
    tot_extracted = tot_kept = tot_in = tot_out = 0

    for row in papers:
        pid = row["paper_id"]
        if row.get("stage_completed") == "extracted" and not force:
            skipped += 1
            continue
        source = paper_source(pid)
        if not source.strip():
            manifest.update_manifest({pid: {"stage_completed": "extracted", "n_metrics": 0}})
            done += 1
            continue
        try:
            data, usage = llm.chat_json(
                SYSTEM, f"Anatomy hint: {row.get('anatomical_target')}\n\nSOURCE:\n{source}",
                max_tokens=1500)
        except Exception as ex:  # noqa: BLE001
            print(f"  FAIL {pid}: {ex}")
            failed += 1
            continue
        records = data.get("metrics", []) if isinstance(data, dict) else []
        kept = verify(records, source)
        store(con, pid, kept)
        con.commit()
        tot_extracted += len(records)
        tot_kept += len(kept)
        tot_in += usage.prompt_tokens
        tot_out += usage.completion_tokens
        # mark done immediately so a rerun after a rate limit resumes here
        manifest.update_manifest({pid: {"stage_completed": "extracted", "n_metrics": len(kept)}})
        done += 1
        print(f"  {pid}: {len(kept)}/{len(records)} verified")

    con.close()
    discarded = tot_extracted - tot_kept
    print("\n----- extraction summary -----")
    print(f"  papers processed : {done} | skipped: {skipped} | failed: {failed}")
    print(f"  records verified : {tot_kept} / {tot_extracted} extracted")
    print(f"  discarded (unverifiable): {discarded} "
          f"({discarded / tot_extracted:.0%} caught)" if tot_extracted else "")
    print(f"  tokens: {tot_in} in / {tot_out} out | cost: $0.00 (Groq free tier)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agent 2 metric extraction.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
