"""Section-aware chunking -> data/chunks/{paper_id}.json.

800-token chunks with 150 overlap, measured with the embedding model's tokenizer.
Chunks never span sections; tables are kept whole; text chunks under 100 tokens
(parsing debris) are dropped. Chunk IDs are deterministic and stable across runs.

  python -m src.chunk --limit 3
  python -m src.chunk
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from transformers import AutoTokenizer

from src import config, manifest

_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    return _tok


def n_tokens(text: str) -> int:
    return len(tok().encode(text, add_special_tokens=False))


def _split_big(text: str) -> list[str]:
    """Token-window a single paragraph longer than CHUNK_TOKENS (rare)."""
    ids = tok().encode(text, add_special_tokens=False)
    step = config.CHUNK_TOKENS - config.CHUNK_OVERLAP
    out = []
    for i in range(0, len(ids), step):
        out.append(tok().decode(ids[i:i + config.CHUNK_TOKENS]))
        if i + config.CHUNK_TOKENS >= len(ids):
            break
    return out


def chunk_section_text(text: str) -> list[str]:
    """Greedily pack paragraphs to ~800 tokens, carrying ~150 tokens of trailing
    paragraphs into the next chunk as overlap. Paragraph boundaries keep chunks
    clean (no mid-word cuts)."""
    paras = [p for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    ctok = 0
    for para in paras:
        pt = n_tokens(para)
        if pt > config.CHUNK_TOKENS:
            if current:
                chunks.append("\n".join(current))
                current, ctok = [], 0
            chunks.extend(_split_big(para))
            continue
        if current and ctok + pt > config.CHUNK_TOKENS:
            chunks.append("\n".join(current))
            overlap, ot = [], 0
            for p in reversed(current):
                pc = n_tokens(p)
                if ot + pc > config.CHUNK_OVERLAP:
                    break
                overlap.insert(0, p)
                ot += pc
            current, ctok = overlap[:], ot
        current.append(para)
        ctok += pt
    if current:
        chunks.append("\n".join(current))
    return chunks


def _table_text(rows) -> str:
    return "\n".join(" | ".join(str(c) for c in r if c is not None) for r in rows)


def chunk_paper(struct: dict) -> list[dict]:
    pid = struct["paper_id"]
    out: list[dict] = []
    idx: Counter = Counter()   # per-section-name index -> deterministic unique IDs

    for s in struct["sections"]:
        name = s["name"]
        for piece in chunk_section_text(s["text"]):
            nt = n_tokens(piece)
            if nt < config.MIN_CHUNK_TOKENS:
                continue
            out.append({"chunk_id": f"{pid}::{name}::{idx[name]}", "paper_id": pid,
                        "section": name, "heading": s.get("heading", ""),
                        "is_table": False, "n_tokens": nt, "text": piece})
            idx[name] += 1

    for t in struct.get("tables", []):
        name = t.get("section", "table")
        txt = _table_text(t["rows"])
        out.append({"chunk_id": f"{pid}::{name}::{idx[name]}", "paper_id": pid,
                    "section": name, "heading": "", "is_table": True,
                    "page": t.get("page"), "n_tokens": n_tokens(txt), "text": txt})
        idx[name] += 1

    return out


def run(limit: int | None = None, force: bool = False) -> None:
    papers = manifest.active_papers()
    if limit:
        papers = papers[:limit]
    updates: dict[str, dict] = {}
    total = tables = done = skipped = 0
    counts: list[int] = []

    for row in papers:
        pid = row["paper_id"]
        out_path = config.CHUNKS_DIR / f"{pid}.json"
        if out_path.exists() and not force:
            skipped += 1
            continue
        struct = json.loads((config.CLEAN_DIR / f"{pid}.json").read_text(encoding="utf-8"))
        chunks = chunk_paper(struct)
        out_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        done += 1
        total += len(chunks)
        tables += sum(1 for c in chunks if c["is_table"])
        counts.append(len(chunks))
        updates[pid] = {"stage_completed": "chunked", "n_chunks": len(chunks)}

    if updates:
        manifest.update_manifest(updates)

    print("\n----- chunk summary -----")
    print(f"  papers chunked : {done} (skipped {skipped})")
    print(f"  total chunks   : {total}")
    print(f"  table chunks   : {tables}")
    if counts:
        counts.sort()
        print(f"  chunks/paper   : min {counts[0]}, median {counts[len(counts)//2]}, "
              f"max {counts[-1]}, mean {sum(counts)/len(counts):.1f}")
        bad = [c for c in counts if not (5 <= c <= 500)]
        print(f"  outside [5,500]: {len(bad)}  (validation will flag these)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Section-aware chunking.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
