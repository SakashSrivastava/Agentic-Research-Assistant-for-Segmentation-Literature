"""Structure recovery -> data/clean/{paper_id}.json.

Detect headings (font size vs body median + bold + section-name regex), assign
every block to its section, truncate after References, tag tables by section.

  python -m src.structure --limit 3
  python -m src.structure
"""
from __future__ import annotations

import argparse
import json
import re
from statistics import median

from src import config, manifest

# Ordered: first matching pattern names the section. Patterns allow a leading
# section number ("3. Results", "IV. Experiments").
_NUM = r"(?:\d{1,2}(?:\.\d+)*\.?\s+|[ivxIVX]+\.\s+)?"
SECTION_PATTERNS = [
    ("abstract",        rf"^{_NUM}abstract\b"),
    ("introduction",    rf"^{_NUM}introduction\b"),
    ("related_work",    rf"^{_NUM}(related work|background|literature review|prior work)\b"),
    ("methods",         rf"^{_NUM}(methods?|materials?|methodology|proposed (method|approach)|"
                        rf"our (method|approach)|network|architecture|model)\b"),
    ("experiments",     rf"^{_NUM}(experiments?|experimental (setup|results)|implementation|"
                        rf"datasets?|evaluation|training|setup)\b"),
    ("results",         rf"^{_NUM}(results?|findings|quantitative|comparison)\b"),
    ("discussion",      rf"^{_NUM}(discussion|ablation)\b"),
    ("conclusion",      rf"^{_NUM}(conclusions?|concluding|summary)\b"),
    ("references",      r"^(references|bibliography)\b"),
    ("acknowledgments", r"^acknowledge?ments?\b"),
    ("appendix",        rf"^{_NUM}(appendix|supplementary)\b"),
]

CORE_SECTIONS = {"methods", "experiments", "results"}  # what the target query needs


def _canonical(low: str) -> str | None:
    for name, pat in SECTION_PATTERNS:
        if re.match(pat, low):
            return name
    return None


def _is_heading(text: str, size: float, bold: bool, body: float) -> bool:
    words = text.split()
    if not (1 <= len(words) <= 12):
        return False
    low = text.lower().strip()
    if _canonical(low):
        return True
    # Numbered heading ("3.1 Network Architecture") that is bold or slightly larger.
    if re.match(r"^(\d{1,2}(\.\d+)*\.?|[IVX]+\.)\s+[A-Za-z]", text) and (bold or size >= body + 0.5):
        return True
    # A short bold line noticeably larger than body text.
    return bold and size >= body + 1.5


def build_structure(parsed: dict) -> dict:
    sizes = [b["size"] for pg in parsed["pages"] for b in pg["blocks"] if b["size"] > 0]
    body = median(sizes) if sizes else 10.0

    sections: list[dict] = []
    current = {"name": "front_matter", "heading": "", "page": 1, "_parts": []}
    truncated = False

    for pi, pg in enumerate(parsed["pages"], start=1):
        for b in pg["blocks"]:
            if _is_heading(b["text"], b["size"], b["bold"], body):
                name = _canonical(b["text"].lower().strip())
                if name == "references":       # drop references and everything after
                    truncated = True
                    break
                sections.append(current)
                current = {"name": name or "section", "heading": b["text"].strip(),
                           "page": pi, "_parts": []}
            else:
                current["_parts"].append(b["text"])
        if truncated:
            break
    sections.append(current)

    for s in sections:
        s["text"] = "\n".join(s.pop("_parts")).strip()
    sections = [s for s in sections if s["text"] or s["heading"]]

    # Tag each table with the section active on its page (last section starting <= page).
    tables = []
    for pi, pg in enumerate(parsed["pages"], start=1):
        sect = "front_matter"
        for s in sections:
            if s["page"] <= pi:
                sect = s["name"]
        for t in pg["tables"]:
            tables.append({"page": pi, "section": sect, "rows": t["rows"]})

    present = {s["name"] for s in sections}
    return {
        "paper_id": parsed["paper_id"],
        "sections": sections,
        "tables": tables,
        "n_sections": len(sections),
        "has_core": sorted(CORE_SECTIONS & present),
        "references_truncated": truncated,
    }


def run(limit: int | None = None, force: bool = False) -> None:
    papers = manifest.active_papers()
    if limit:
        papers = papers[:limit]
    totals = {"done": 0, "skipped": 0, "no_headings": 0, "has_results": 0}
    updates: dict[str, dict] = {}

    for row in papers:
        pid = row["paper_id"]
        out = config.CLEAN_DIR / f"{pid}.json"
        if out.exists() and not force:
            totals["skipped"] += 1
            continue
        parsed = json.loads((config.PARSED_DIR / f"{pid}.json").read_text(encoding="utf-8"))
        struct = build_structure(parsed)
        out.write_text(json.dumps(struct, ensure_ascii=False), encoding="utf-8")
        totals["done"] += 1
        if struct["n_sections"] <= 1:
            totals["no_headings"] += 1
        if "results" in struct["has_core"]:
            totals["has_results"] += 1
        updates[pid] = {"stage_completed": "structured"}

    if updates:
        manifest.update_manifest(updates)

    print("\n----- structure summary -----")
    print(f"  structured        : {totals['done']}")
    print(f"  skipped (done)    : {totals['skipped']}")
    print(f"  with results sect.: {totals['has_results']}")
    print(f"  no headings found : {totals['no_headings']}  (candidates for parse-repair)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Structure recovery.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
