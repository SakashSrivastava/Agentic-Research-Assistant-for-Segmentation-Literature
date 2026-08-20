"""Retrieval: vector, BM25, hybrid (reciprocal-rank fusion), and reranking.

- vector : semantic similarity over BGE-small embeddings (ChromaDB)
- bm25   : exact keyword matching (rank_bm25) for metric/architecture names
- hybrid : fuse the two ranked lists with reciprocal rank fusion (RRF)
- rerank : hybrid -> take 25 candidates -> cross-encoder rerank -> keep top 5

search(query, k, method, where) is the single entry point every caller uses.
"""
from __future__ import annotations

import json
import pickle
import re

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config, embed, manifest

_embedder = _reranker = _col = _bm = _passages_cache = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
    return _reranker


def _collection():
    global _col
    if _col is None:
        _col = chromadb.PersistentClient(
            path=str(config.INDEX_DIR / "chroma")).get_collection("chunks")
    return _col


def _bm25():
    global _bm
    if _bm is None:
        _bm = pickle.load(open(config.INDEX_DIR / "bm25.pkl", "rb"))
    return _bm["ids"], _bm["bm25"]


def _passages() -> dict[str, str]:
    """chunk_id -> title+section+text (same grounding the embedder saw), for the reranker."""
    global _passages_cache
    if _passages_cache is None:
        titles = {r["paper_id"]: r.get("title", "") for r in manifest.load_manifest()}
        _passages_cache = {}
        for row in manifest.active_papers():
            pid = row["paper_id"]
            f = config.CHUNKS_DIR / f"{pid}.json"
            if not f.exists():
                continue
            for c in json.loads(f.read_text(encoding="utf-8")):
                _passages_cache[c["chunk_id"]] = embed.embed_input(
                    titles.get(pid, ""), c["section"], c["text"])
    return _passages_cache


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def vector_search(query: str, k: int = 25, where=None):
    qv = _get_embedder().encode([query], normalize_embeddings=True)[0].tolist()
    r = _collection().query(query_embeddings=[qv], n_results=k, where=where)
    return list(zip(r["ids"][0], r["distances"][0]))


def bm25_search(query: str, k: int = 25):
    ids, bm = _bm25()
    scores = bm.get_scores(_tokens(query))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(ids[i], float(scores[i])) for i in top]


def _rrf(rankings, k: int):
    """Reciprocal rank fusion: score = sum 1/(RRF_K + rank). Rank position only,
    so vector and BM25 scores (different scales) never need normalizing."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


def hybrid_search(query: str, k: int = 25, where=None):
    return _rrf([vector_search(query, 50, where), bm25_search(query, 50)], k)


def rerank_search(query: str, k: int = 5, where=None):
    # Rerank the vector candidates (dense retrieval is the strongest base here);
    # the cross-encoder then reorders them.
    cand = vector_search(query, config.RERANK_CANDIDATES, where)
    passages = _passages()
    scores = _get_reranker().predict([(query, passages.get(cid, "")) for cid, _ in cand])
    ranked = sorted(zip([cid for cid, _ in cand], (float(s) for s in scores)),
                    key=lambda x: -x[1])
    return ranked[:k]


def search(query: str, k: int = 5, method: str = "rerank", where=None):
    if method == "vector":
        return vector_search(query, k, where)
    if method == "bm25":
        return bm25_search(query, k)
    if method == "hybrid":
        return hybrid_search(query, k, where)
    if method == "rerank":
        return rerank_search(query, k=k, where=where)
    raise ValueError(f"unknown method: {method}")
