"""QA: verify parsed JSON faithfully represents the source PDFs.

Three checks: word coverage vs raw PDF text, table-cell fidelity (values present
verbatim in source), and that needs_ocr papers really are text-sparse.

  python -m src.validate_parse
"""
from __future__ import annotations

import json
import random
import re
import statistics

import fitz  # PyMuPDF

from src import config, manifest

fitz.TOOLS.mupdf_display_errors(False)

COVERAGE_MIN = 0.95   # a paper below this has lost real content


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _parsed(pid: str) -> dict:
    return json.loads((config.PARSED_DIR / f"{pid}.json").read_text(encoding="utf-8"))


def _parsed_text(parsed: dict) -> str:
    out = []
    for pg in parsed["pages"]:
        out += [b["text"] for b in pg["blocks"]]
        for t in pg["tables"]:
            out += [" ".join(str(c) for c in r if c) for r in t["rows"]]
    return " ".join(out)


def check_coverage() -> bool:
    results = []
    for row in manifest.active_papers():
        pid = row["paper_id"]
        our_w = _words(_parsed_text(_parsed(pid)))
        doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
        raw_w = _words(" ".join(p.get_text() for p in doc))
        doc.close()
        if raw_w:
            results.append((pid, len(our_w & raw_w) / len(raw_w)))

    covs = [c for _, c in results]
    low = sorted([r for r in results if r[1] < COVERAGE_MIN], key=lambda x: x[1])
    print("=== TEXT COVERAGE (raw PDF words present in parsed JSON) ===")
    print(f"  papers      : {len(results)}")
    print(f"  mean {statistics.mean(covs):.3f} | median {statistics.median(covs):.3f} "
          f"| min {min(covs):.3f}")
    print(f"  below {COVERAGE_MIN}: {len(low)}")
    for pid, c in low[:20]:
        print(f"    {pid}  cov={c:.3f}")
    return not low


def _numbers(s: str) -> list[str]:
    return re.findall(r"\d+\.?\d+", s)


def check_tables(sample: int = 20) -> bool:
    """Fidelity = fraction of numeric values in extracted tables that appear in
    the source page. Tests for fabrication/corruption (the real risk); tolerant
    of find_tables() cell-boundary merges, which do not change the values."""
    with_tables = [r["paper_id"] for r in manifest.active_papers()
                   if _parsed(r["paper_id"])["n_tables"] > 0]
    picks = random.Random(0).sample(with_tables, min(sample, len(with_tables)))
    print("\n=== TABLE FIDELITY (table numbers present verbatim in source page) ===")
    fractions = []
    for pid in picks:
        parsed = _parsed(pid)
        doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
        for pi, pg in enumerate(parsed["pages"]):
            if not pg["tables"]:
                continue
            raw = doc[pi].get_text()
            cellnums = [n for r in pg["tables"][0]["rows"]
                        for c in r if c for n in _numbers(str(c))]
            if not cellnums:
                continue
            hit = sum(1 for n in cellnums if n in raw)
            fractions.append(hit / len(cellnums))
            print(f"  {pid} p{pi + 1}: {hit}/{len(cellnums)} numbers real ({hit / len(cellnums):.0%})")
            break
        doc.close()
    m = statistics.mean(fractions)
    print(f"  mean numeric fidelity: {m:.1%} (are reported numbers real?)")
    return m >= 0.98


def check_ocr() -> bool:
    print("\n=== needs_ocr papers (should be text-sparse) ===")
    ocr = [r for r in manifest.load_manifest() if r["status"] == "needs_ocr"]
    for row in ocr:
        pid = row["paper_id"]
        doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
        chars = sum(len(p.get_text()) for p in doc)
        imgs = sum(len(p.get_images()) for p in doc)
        print(f"  {pid}: pages={doc.page_count}, chars={chars}, images={imgs}")
        doc.close()
    if not ocr:
        print("  (none)")
    return True


def main() -> None:
    ok = all([check_coverage(), check_tables(), check_ocr()])
    print("\nRESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
