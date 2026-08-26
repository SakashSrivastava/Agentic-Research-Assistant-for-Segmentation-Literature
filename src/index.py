"""Build the three retrieval stores from chunks + cached vectors.

- ChromaDB   : vectors + filter metadata (paper_id, section, year, anatomy, modality)
- rank_bm25  : keyword index over the same chunk texts, persisted (pickle)
- SQLite     : papers table (document metadata; the metrics table is added by Agent 2)

Both a vector store and a keyword store are needed: vectors for semantic
similarity, BM25 for exact metric/architecture names ("Dice", "nnU-Net").

  python -m src.index
"""
from __future__ import annotations

import json
import pickle
import re
import sqlite3

import chromadb
from rank_bm25 import BM25Okapi

from src import config, embed, manifest

CHROMA_DIR = config.INDEX_DIR / "chroma"
BM25_PATH = config.INDEX_DIR / "bm25.pkl"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _meta(chunk: dict, paper: dict) -> dict:
    # Chroma metadata must be non-null scalars, so coerce None -> "" / 0.
    return {
        "paper_id": chunk["paper_id"],
        "section": chunk["section"],
        "is_table": bool(chunk["is_table"]),
        "year": paper.get("year") or 0,
        "anatomical_target": paper.get("anatomical_target") or "",
        "imaging_modality": paper.get("imaging_modality") or "",
        "title": paper.get("title") or "",
    }


def load_all_chunks() -> list[tuple[dict, dict]]:
    papers = {r["paper_id"]: r for r in manifest.load_manifest()}
    out = []
    for row in manifest.active_papers():
        pid = row["paper_id"]
        f = config.CHUNKS_DIR / f"{pid}.json"
        if not f.exists():
            continue
        for c in json.loads(f.read_text(encoding="utf-8")):
            out.append((c, papers[pid]))
    return out


def build_chroma(chunks, cache) -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("chunks")
    except Exception:  # noqa: BLE001 - fine if it doesn't exist yet
        pass
    col = client.create_collection("chunks", metadata={"model": config.EMBED_MODEL})
    batch = 1000
    for i in range(0, len(chunks), batch):
        ids, embs, docs, metas = [], [], [], []
        for c, paper in chunks[i:i + batch]:
            ei = embed.embed_input(paper.get("title", ""), c["section"], c["text"])
            v = cache.get(embed._hash(ei))
            if v is None:
                continue
            ids.append(c["chunk_id"])
            embs.append(v.tolist())
            docs.append(c["text"])
            metas.append(_meta(c, paper))
        if ids:
            col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    return col.count()


def build_bm25(chunks) -> int:
    ids = [c["chunk_id"] for c, _ in chunks]
    corpus = [_tokens(c["text"]) for c, _ in chunks]
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"ids": ids, "bm25": BM25Okapi(corpus)}, f)
    return len(ids)


def build_sqlite() -> int:
    con = sqlite3.connect(config.DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY, title TEXT, year INTEGER, source TEXT, doi TEXT,
        anatomical_target TEXT, imaging_modality TEXT, query_bucket TEXT,
        status TEXT, n_chunks INTEGER)""")
    con.execute("DELETE FROM papers")
    for r in manifest.load_manifest():
        con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (r["paper_id"], r.get("title"), r.get("year"), r.get("source"),
                     r.get("doi"), r.get("anatomical_target"), r.get("imaging_modality"),
                     r.get("query_bucket"), r.get("status"), r.get("n_chunks")))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    con.close()
    return n


def run() -> None:
    chunks = load_all_chunks()
    cache = embed._load_cache()
    print(f"indexing {len(chunks)} chunks ...")
    nc = build_chroma(chunks, cache)
    nb = build_bm25(chunks)
    ns = build_sqlite()
    manifest.update_manifest({r["paper_id"]: {"stage_completed": "indexed"}
                              for r in manifest.active_papers()})
    print("\n----- index summary -----")
    print(f"  chroma vectors : {nc}")
    print(f"  bm25 chunks    : {nb}")
    print(f"  sqlite papers  : {ns}")
    print(f"  stores in      : {config.INDEX_DIR}")


if __name__ == "__main__":
    run()
