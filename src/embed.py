"""Contextual enrichment + embedding.

Before embedding, prepend the paper title and section name to each
chunk. The stored display text is unchanged; this only grounds the vector, since
a bare results chunk is otherwise context-free.
Embed with BGE-small in batches of 64, L2-normalized. Vectors are cached
by content hash, so a re-run after a chunking tweak only embeds what changed.
The model name + dimension are recorded so two embedding models can never mix.

  python -m src.embed --limit 20   # smoke test
  python -m src.embed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config, manifest

CACHE = config.INDEX_DIR / "embed_cache.npz"
META = config.INDEX_DIR / "embed_meta.json"

_model: SentenceTransformer | None = None


def model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_input(title: str, section: str, text: str) -> str:
    """Contextual prefix, prepended for embedding only."""
    return f"{title}\nSection: {section}\n{text}"


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    d = np.load(CACHE, allow_pickle=False)
    hashes, vectors = d["hashes"], d["vectors"]   # load each array once, not per-row
    return {h: vectors[i] for i, h in enumerate(hashes)}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    hashes = np.array(list(cache.keys()))
    vectors = (np.stack(list(cache.values())) if cache
               else np.zeros((0, 384), dtype=np.float32))
    np.savez(CACHE, hashes=hashes, vectors=vectors.astype(np.float32))


def all_chunk_inputs(limit: int | None = None):
    """Yield (chunk_dict, embed_input_text, content_hash) for every active chunk."""
    titles = {r["paper_id"]: r.get("title", "") for r in manifest.load_manifest()}
    items = []
    for row in manifest.active_papers():
        pid = row["paper_id"]
        f = config.CHUNKS_DIR / f"{pid}.json"
        if not f.exists():
            continue
        for c in json.loads(f.read_text(encoding="utf-8")):
            ei = embed_input(titles.get(pid, ""), c["section"], c["text"])
            items.append((c, ei, _hash(ei)))
            if limit and len(items) >= limit:
                return items
    return items


def run(limit: int | None = None, force: bool = False) -> None:
    items = all_chunk_inputs(limit)
    wanted = {h for _, _, h in items}
    cache = {} if force else {h: v for h, v in _load_cache().items() if h in wanted}
    todo = [(ei, h) for _, ei, h in items if h not in cache]
    print(f"chunks: {len(items)} | cached: {len(items) - len(todo)} | to embed: {len(todo)}")

    t0 = time.time()
    if todo:
        vecs = model().encode([ei for ei, _ in todo], batch_size=64,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=True)
        for (_, h), v in zip(todo, vecs):
            cache[h] = v.astype(np.float32)
        _save_cache(cache)

    dim = int(next(iter(cache.values())).shape[0]) if cache else 0
    META.write_text(json.dumps({"model": config.EMBED_MODEL, "dim": dim,
                                "n_vectors": len(cache)}, indent=2), encoding="utf-8")
    print(f"\n----- embed summary -----")
    print(f"  embedded new : {len(todo)} in {time.time() - t0:.1f}s")
    print(f"  cache total  : {len(cache)} vectors | dim {dim} | model {config.EMBED_MODEL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Contextual embedding.")
    ap.add_argument("--limit", type=int, default=None, help="embed only first N chunks (smoke)")
    ap.add_argument("--force", action="store_true", help="ignore cache, re-embed all")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
