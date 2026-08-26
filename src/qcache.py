"""Semantic answer cache with a feedback loop.

When a new question is close (cosine >= SIM_THRESHOLD) to a previously answered one
that hasn't been net-downvoted or gone stale, we serve the stored answer instantly
for zero tokens. Feedback (up/down) decides what stays reusable. The cache is
global: one user's answer benefits everyone. It reuses the same BGE-small model as
retrieval, so there is no second model in memory.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src import db, retrieve

SIM_THRESHOLD = 0.93   # question-to-question cosine; high, to avoid wrong matches
TTL_DAYS = 21          # answers older than this are not reused (corpus drift)


def _embed(question: str) -> np.ndarray:
    v = retrieve._get_embedder().encode([question], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)


def embed_bytes(question: str) -> bytes:
    return _embed(question).tobytes()


def _age_days(created_at: str) -> float:
    try:
        t = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400.0


def find(question: str):
    """Best reusable cache entry for this question, or None. Returns (row, similarity).
    Skips net-downvoted and stale entries; both embeddings are unit vectors, so the
    dot product is cosine similarity."""
    cands = db.cache_candidates()
    if not cands:
        return None
    q = _embed(question)
    best, best_sim = None, -1.0
    for c in cands:
        if c["down"] > c["up"] or _age_days(c["created_at"]) > TTL_DAYS:
            continue
        e = np.frombuffer(c["embedding"], dtype=np.float32)
        if e.shape != q.shape:
            continue
        sim = float(np.dot(q, e))
        if sim > best_sim:
            best, best_sim = c, sim
    if best is not None and best_sim >= SIM_THRESHOLD:
        return best, best_sim
    return None


def store(question: str, answer: str, steps: int, tokens: int) -> int:
    return db.cache_insert(question, answer, steps, tokens, embed_bytes(question))
