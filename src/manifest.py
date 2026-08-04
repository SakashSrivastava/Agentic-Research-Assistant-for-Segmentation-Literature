"""Stage 1: build data/manifest.jsonl with content hashes + duplicate flags.

Deterministic, no LLM. Rebuilds from data/raw each run.

  python -m src.manifest
"""
from __future__ import annotations

import hashlib
import json
import re

from src import config


def sha256_of(path) -> str:
    """Byte-fingerprint of a file, read in 1 MB chunks so large PDFs never
    load fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_title(title: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace -> preprint and
    published versions of the same paper map to the same string."""
    t = re.sub(r"[^a-z0-9 ]+", " ", title.lower())
    return re.sub(r"\s+", " ", t).strip()


def build_manifest() -> list[dict]:
    rows = []
    for pdf in sorted(config.RAW_DIR.glob("*.pdf")):
        pid = pdf.stem
        meta_path = config.RAW_DIR / f"{pid}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows.append({
            "paper_id": pid,
            "content_hash": sha256_of(pdf),
            "title": meta.get("title", ""),
            "normalized_title": normalize_title(meta.get("title", "")),
            "source": meta.get("source"),
            "year": meta.get("year"),
            "doi": meta.get("doi"),
            "query_bucket": meta.get("query_bucket"),
            "status": "active",
            "stage_completed": "acquired",
            "duplicate_of": None,
            "error": None,
        })

    # Canonical preference: a paper with a DOI (usually the published version)
    # wins over one without; ties break by paper_id for determinism.
    rows.sort(key=lambda r: (0 if r["doi"] else 1, r["paper_id"]))

    seen_hash: dict[str, str] = {}
    seen_title: dict[str, str] = {}
    for r in rows:
        h, nt = r["content_hash"], r["normalized_title"]
        dup_of = seen_hash.get(h) or (seen_title.get(nt) if nt else None)
        if dup_of:
            r["status"] = "duplicate"
            r["duplicate_of"] = dup_of
        else:
            seen_hash[h] = r["paper_id"]
            if nt:
                seen_title[nt] = r["paper_id"]

    rows.sort(key=lambda r: r["paper_id"])  # stable on-disk order
    config.MANIFEST_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def load_manifest() -> list[dict]:
    if not config.MANIFEST_PATH.exists():
        return []
    return [json.loads(line) for line in
            config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def active_papers() -> list[dict]:
    """Non-duplicate papers — what later stages should process."""
    return [r for r in load_manifest() if r["status"] == "active"]


def update_manifest(updates: dict[str, dict]) -> None:
    """Merge per-paper field updates into the manifest in place, preserving all
    other rows and fields. `updates` maps paper_id -> {field: value}."""
    rows = load_manifest()
    for r in rows:
        if r["paper_id"] in updates:
            r.update(updates[r["paper_id"]])
    config.MANIFEST_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _report(rows: list[dict]) -> None:
    dups = [r for r in rows if r["status"] == "duplicate"]
    by_hash = sum(1 for r in dups
                  if any(o["content_hash"] == r["content_hash"] and o["status"] == "active"
                         for o in rows))
    print("\n----- manifest summary -----")
    print(f"  total papers   : {len(rows)}")
    print(f"  active (unique): {len(rows) - len(dups)}")
    print(f"  duplicates     : {len(dups)}  (exact-hash: {by_hash}, title-only: {len(dups) - by_hash})")
    print(f"  manifest       : {config.MANIFEST_PATH}")
    for r in dups:
        kind = "hash" if by_hash and any(
            o["content_hash"] == r["content_hash"] and o["status"] == "active" for o in rows
        ) else "title"
        print(f"    DUP ({kind}): {r['paper_id']}  ->  {r['duplicate_of']}")


if __name__ == "__main__":
    _report(build_manifest())
