"""Layout-aware parsing -> data/parsed/{paper_id}.json.

Per page we keep text blocks (with font size, bold, bbox) in reconstructed
reading order, plus tables extracted separately as structured rows. Scanned
PDFs (too few characters) are flagged needs_ocr and excluded.

  python -m src.parse --limit 3   # smoke test
  python -m src.parse             # all active papers
"""
from __future__ import annotations

import argparse
import json
from statistics import median

import fitz  # PyMuPDF

from src import config, manifest

# Silence non-fatal "unknown keyword" content-stream warnings MuPDF recovers from.
fitz.TOOLS.mupdf_display_errors(False)

OCR_CHAR_THRESHOLD = 100   # avg chars/page below this => likely scanned
FULL_WIDTH_FRAC = 0.70     # block wider than this * page => spans both columns


def _is_bold(span) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _block_text_and_style(block) -> tuple[str, float, bool]:
    """Flatten a block's spans into text, plus its median font size and whether
    it is majority-bold (both used by structure recovery to detect headings)."""
    lines, sizes, bold, n = [], [], 0, 0
    for line in block["lines"]:
        parts = []
        for s in line["spans"]:
            parts.append(s["text"])
            sizes.append(s["size"])
            n += 1
            bold += _is_bold(s)
        lines.append("".join(parts))
    text = " ".join(lines).strip()
    return text, (round(median(sizes), 1) if sizes else 0.0), (n and bold >= n / 2)


def _detect_columns(blocks, page_width) -> int:
    """1 or 2 columns, from how block centers split around the page midline."""
    mid = page_width / 2
    centers = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks]
    if len(centers) < 6:
        return 1
    left = sum(1 for c in centers if c < mid)
    right = len(centers) - left
    return 2 if left >= 0.25 * len(centers) and right >= 0.25 * len(centers) else 1


def _column_of(bbox, page_width, columns) -> int:
    """0 = left/full-width, 1 = right. Full-width blocks (titles, wide tables)
    go to column 0 so they sort ahead of the right column at their y-position."""
    x0, _, x1, _ = bbox
    if columns == 1 or (x1 - x0) > FULL_WIDTH_FRAC * page_width:
        return 0
    return 0 if (x0 + x1) / 2 < page_width / 2 else 1


def _is_real_table(rows) -> bool:
    """Drop find_tables() false positives (flowcharts, diagrams): keep only grids
    that are densely filled — real results tables are full of values."""
    if not rows or len(rows) < 2:
        return False
    cells = [c for row in rows for c in row]
    filled = sum(1 for c in cells if c is not None and str(c).strip())
    return filled >= 4 and filled >= 0.4 * len(cells)


def _in_any_table(bbox, table_bboxes) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(tx0 <= cx <= tx1 and ty0 <= cy <= ty1
               for tx0, ty0, tx1, ty1 in table_bboxes)


def parse_page(page) -> dict:
    data = page.get_text("dict")
    page_width = page.rect.width

    # Tables first, so we can drop text blocks that live inside them.
    tables, table_bboxes = [], []
    try:
        for t in page.find_tables().tables:
            rows = t.extract()
            if not _is_real_table(rows):
                continue
            tables.append({"bbox": [round(v, 1) for v in t.bbox], "rows": rows})
            table_bboxes.append(t.bbox)
    except Exception:  # noqa: BLE001 - table finder can throw on odd pages
        pass

    blocks = []
    for b in data["blocks"]:
        if b.get("type") != 0 or _in_any_table(b["bbox"], table_bboxes):
            continue
        text, size, bold = _block_text_and_style(b)
        if not text:
            continue
        blocks.append({
            "text": text, "size": size, "bold": bold,
            "bbox": [round(v, 1) for v in b["bbox"]],
        })

    columns = _detect_columns(blocks, page_width)
    for b in blocks:
        b["column"] = _column_of(b["bbox"], page_width, columns)
    blocks.sort(key=lambda b: (b["column"], round(b["bbox"][1])))  # column, then top->bottom

    chars = sum(len(b["text"]) for b in blocks)
    return {"columns": columns, "blocks": blocks, "tables": tables, "chars": chars}


def parse_pdf(pid: str) -> dict:
    doc = fitz.open(config.RAW_DIR / f"{pid}.pdf")
    pages = [parse_page(p) for p in doc]
    doc.close()
    total_chars = sum(p["chars"] for p in pages)
    avg = total_chars / len(pages) if pages else 0
    return {
        "paper_id": pid,
        "num_pages": len(pages),
        "avg_chars_per_page": round(avg, 1),
        "needs_ocr": avg < OCR_CHAR_THRESHOLD,
        "n_tables": sum(len(p["tables"]) for p in pages),
        "pages": pages,
    }


def run(limit: int | None = None, force: bool = False) -> None:
    papers = manifest.active_papers()
    if limit:
        papers = papers[:limit]
    totals = {"parsed": 0, "needs_ocr": 0, "failed": 0, "skipped": 0,
              "cols2": 0, "tables": 0}
    updates: dict[str, dict] = {}

    for row in papers:
        pid = row["paper_id"]
        out = config.PARSED_DIR / f"{pid}.json"
        if out.exists() and not force:
            totals["skipped"] += 1
            continue
        try:
            parsed = parse_pdf(pid)
        except Exception as ex:  # noqa: BLE001
            print(f"  FAIL {pid}: {ex}")
            totals["failed"] += 1
            updates[pid] = {"status": "parse_failed", "error": str(ex)[:200]}
            continue

        out.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        totals["parsed"] += 1
        totals["tables"] += parsed["n_tables"]
        totals["cols2"] += sum(1 for p in parsed["pages"] if p["columns"] == 2)
        if parsed["needs_ocr"]:
            totals["needs_ocr"] += 1
            updates[pid] = {"status": "needs_ocr", "stage_completed": "parsed",
                            "needs_ocr": True}
            print(f"  needs_ocr {pid} (avg {parsed['avg_chars_per_page']} chars/page)")
        else:
            updates[pid] = {"stage_completed": "parsed", "needs_ocr": False}

    if updates:
        manifest.update_manifest(updates)

    print("\n----- parse summary -----")
    print(f"  parsed        : {totals['parsed']}")
    print(f"  needs_ocr     : {totals['needs_ocr']} (excluded from later stages)")
    print(f"  parse_failed  : {totals['failed']}")
    print(f"  skipped (done): {totals['skipped']}")
    print(f"  tables found  : {totals['tables']}")
    print(f"  2-column pages: {totals['cols2']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Layout-aware PDF parsing.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-parse even if output exists")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
