"""QA: verify parsed JSON faithfully represents the source PDFs.

Full run saves a report to data/parse_validation.json and prints a summary:
  python -m src.validate_parse

Match a single paper against its PDF (coverage + any missing words):
  python -m src.validate_parse --paper arxiv_1808.05238
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys

import fitz  # PyMuPDF

from src import config, manifest

fitz.TOOLS.mupdf_display_errors(False)
try:  # Windows consoles default to cp1252 and choke on PDF Unicode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

COVERAGE_MIN = 0.95
REPORT_PATH = config.DATA_DIR / "parse_validation.json"


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _numbers(s: str) -> list[str]:
    return re.findall(r"\d+\.?\d+", s)


def _parsed(pid: str) -> dict:
    return json.loads((config.PARSED_DIR / f"{pid}.json").read_text(encoding="utf-8"))


def _parsed_text(parsed: dict) -> str:
    out = []
    for pg in parsed["pages"]:
        out += [b["text"] for b in pg["blocks"]]
        for t in pg["tables"]:
            out += [" ".join(str(c) for c in r if c) for r in t["rows"]]
    return " ".join(out)


def _raw_text(pid: str) -> str:
    doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
    raw = " ".join(p.get_text() for p in doc)
    doc.close()
    return raw


def check_coverage() -> dict:
    per_paper = {}
    for row in manifest.active_papers():
        pid = row["paper_id"]
        our_w, raw_w = _words(_parsed_text(_parsed(pid))), _words(_raw_text(pid))
        if raw_w:
            per_paper[pid] = round(len(our_w & raw_w) / len(raw_w), 4)
    covs = list(per_paper.values())
    below = {p: c for p, c in per_paper.items() if c < COVERAGE_MIN}
    return {
        "papers": len(covs),
        "mean": round(statistics.mean(covs), 4),
        "median": round(statistics.median(covs), 4),
        "min": min(covs),
        "below_threshold": below,
        "per_paper": per_paper,
        "ok": not below,
    }


def check_tables(sample: int = 20) -> dict:
    with_tables = [r["paper_id"] for r in manifest.active_papers()
                   if _parsed(r["paper_id"])["n_tables"] > 0]
    picks = random.Random(0).sample(with_tables, min(sample, len(with_tables)))
    samples = []
    for pid in picks:
        parsed = _parsed(pid)
        doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
        for pi, pg in enumerate(parsed["pages"]):
            if not pg["tables"]:
                continue
            raw = doc[pi].get_text()
            nums = [n for r in pg["tables"][0]["rows"] for c in r if c for n in _numbers(str(c))]
            if not nums:
                continue
            hit = sum(1 for n in nums if n in raw)
            samples.append({"paper_id": pid, "page": pi + 1, "hit": hit,
                            "total": len(nums), "frac": round(hit / len(nums), 4)})
            break
        doc.close()
    mean = statistics.mean(s["frac"] for s in samples)
    return {"mean_numeric_fidelity": round(mean, 4), "samples": samples, "ok": mean >= 0.98}


def check_ocr() -> dict:
    out = []
    for row in [r for r in manifest.load_manifest() if r["status"] == "needs_ocr"]:
        pid = row["paper_id"]
        doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
        out.append({"paper_id": pid, "pages": doc.page_count,
                    "chars": sum(len(p.get_text()) for p in doc),
                    "images": sum(len(p.get_images()) for p in doc)})
        doc.close()
    return {"papers": out, "ok": True}


def match_one(pid: str) -> None:
    """Show coverage for one paper and the exact words present in the PDF but not
    in our parsed output (empty == perfect match)."""
    parsed = _parsed(pid)
    our_w, raw_w = _words(_parsed_text(parsed)), _words(_raw_text(pid))
    missing = sorted(raw_w - our_w)
    cov = len(our_w & raw_w) / len(raw_w) if raw_w else 1.0
    print(f"paper        : {pid}")
    print(f"pages        : {parsed['num_pages']} | tables: {parsed['n_tables']} | "
          f"needs_ocr: {parsed['needs_ocr']}")
    print(f"coverage     : {cov:.4f}  ({len(our_w & raw_w)}/{len(raw_w)} raw words present)")
    print(f"missing words: {len(missing)}")
    if missing:
        print("  " + ", ".join(missing[:60]) + (" ..." if len(missing) > 60 else ""))
    print("\n--- parsed reading-order text (first 600 chars) ---")
    print(_parsed_text(parsed)[:600])
    print("\n--- raw PDF text (first 600 chars) ---")
    print(_raw_text(pid)[:600])


def main() -> None:
    cov, tab, ocr = check_coverage(), check_tables(), check_ocr()
    result = "PASS" if (cov["ok"] and tab["ok"] and ocr["ok"]) else "FAIL"
    report = {"result": result, "coverage": cov, "table_fidelity": tab, "needs_ocr": ocr}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== TEXT COVERAGE ===")
    print(f"  papers {cov['papers']} | mean {cov['mean']} | median {cov['median']} "
          f"| min {cov['min']} | below {COVERAGE_MIN}: {len(cov['below_threshold'])}")
    print("=== TABLE FIDELITY ===")
    print(f"  mean numeric fidelity {tab['mean_numeric_fidelity']:.1%} over {len(tab['samples'])} tables")
    print("=== needs_ocr ===")
    for o in ocr["papers"]:
        print(f"  {o['paper_id']}: pages={o['pages']} chars={o['chars']} images={o['images']}")
    print(f"\nRESULT: {result}")
    print(f"report saved: {REPORT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate parsed JSON vs source PDFs.")
    ap.add_argument("--paper", help="match a single paper_id against its PDF")
    args = ap.parse_args()
    if args.paper:
        match_one(args.paper)
    else:
        main()
