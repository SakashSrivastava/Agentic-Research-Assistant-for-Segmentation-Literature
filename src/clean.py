"""Stage 4: text cleaning, applied per section in clean/{paper_id}.json.

Six ordered transforms: NFKC -> de-hyphenate -> strip repeated headers/footers
-> (lines already joined in parse) -> replace display equations -> collapse
whitespace. Each paper's clean/ file gains a `cleaning` log; the manifest moves
to stage_completed=cleaned.

  python -m src.clean --limit 3
  python -m src.clean
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter

from src import config, manifest

# Equation-only operators. Deliberately EXCLUDES + - * / ± ×, which also appear
# in numeric ranges and uncertainties (i.e. inside reported results like
# "0.88 ± 0.02"), so results rows are never mistaken for equations.
_MATH = set("=∑∏∫√≤≥≈≠∇∂∈∉∀∃⊗⊕→↔⇒∝")


def _headerfooter_set(parsed: dict) -> set[str]:
    """Block texts repeated on >= half the pages = running headers/footers."""
    pages = parsed["pages"]
    if len(pages) < 4:
        return set()
    counts: Counter = Counter()
    for pg in pages:
        for t in {unicodedata.normalize("NFKC", b["text"]).strip() for b in pg["blocks"]}:
            if 0 < len(t) <= 120:
                counts[t] += 1
    thresh = max(3, len(pages) // 2)
    return {t for t, c in counts.items() if c >= thresh}


def _looks_equation(t: str) -> bool:
    if not (3 <= len(t) <= 300):
        return False
    if len(re.findall(r"\d+\.\d+", t)) >= 2:   # 2+ decimals => measurements, not an equation
        return False
    if len(re.findall(r"\d+", t)) >= 4:        # many numbers => a data row, not an equation
        return False
    if not any(c in _MATH for c in t):
        return False
    ascii_alpha = sum(c.isalpha() and c.isascii() for c in t)
    long_words = len(re.findall(r"[A-Za-z]{4,}", t))
    return ascii_alpha / len(t) < 0.4 and long_words <= 2


def clean_text(text: str, hf: set[str]) -> tuple[str, dict]:
    log = {"headerfooter_removed": 0, "equations_replaced": 0, "hyphens_joined": 0}
    kept = []
    for block in text.split("\n") if text else []:
        t = unicodedata.normalize("NFKC", block).strip()          # 1. NFKC
        if not t:
            continue
        if t in hf or re.fullmatch(r"\d{1,4}", t):                 # 3. headers/footers, page nums
            log["headerfooter_removed"] += 1
            continue
        if _looks_equation(t):                                     # 5. display equations
            kept.append("[EQUATION]")
            log["equations_replaced"] += 1
            continue
        kept.append(t)

    joined = "\n".join(kept)                                       # 4. blocks = paragraphs
    joined, n = re.subn(r"([A-Za-z])-\s+([a-z])", r"\1\2", joined)  # 2. de-hyphenate
    log["hyphens_joined"] = n
    joined = re.sub(r"[ \t]+", " ", joined)                        # 6. collapse whitespace
    joined = re.sub(r" *\n *", "\n", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined, log


def clean_paper(struct: dict, parsed: dict) -> dict:
    hf = _headerfooter_set(parsed)
    totals = Counter()
    for s in struct["sections"]:
        s["heading"] = unicodedata.normalize("NFKC", s["heading"]).strip()
        s["text"], log = clean_text(s["text"], hf)
        totals.update(log)
    struct["sections"] = [s for s in struct["sections"] if s["text"] or s["heading"]]
    struct["cleaned"] = True
    struct["cleaning"] = dict(totals)
    return struct


def run(limit: int | None = None, force: bool = False) -> None:
    papers = manifest.active_papers()
    if limit:
        papers = papers[:limit]
    totals = Counter()
    done = skipped = 0
    updates: dict[str, dict] = {}

    for row in papers:
        pid = row["paper_id"]
        struct = json.loads((config.CLEAN_DIR / f"{pid}.json").read_text(encoding="utf-8"))
        if struct.get("cleaned") and not force:
            skipped += 1
            continue
        parsed = json.loads((config.PARSED_DIR / f"{pid}.json").read_text(encoding="utf-8"))
        struct = clean_paper(struct, parsed)
        (config.CLEAN_DIR / f"{pid}.json").write_text(
            json.dumps(struct, ensure_ascii=False), encoding="utf-8")
        totals.update(struct["cleaning"])
        done += 1
        updates[pid] = {"stage_completed": "cleaned"}

    if updates:
        manifest.update_manifest(updates)

    print("\n----- cleaning summary -----")
    print(f"  cleaned            : {done}")
    print(f"  skipped (done)     : {skipped}")
    print(f"  headers/footers cut: {totals['headerfooter_removed']}")
    print(f"  equations replaced : {totals['equations_replaced']}")
    print(f"  hyphenations joined: {totals['hyphens_joined']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 4 text cleaning.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
