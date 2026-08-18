"""Shared paths and constants for the pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CLEAN_DIR = DATA_DIR / "clean"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "app.db"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

for _d in (RAW_DIR, PARSED_DIR, CLEAN_DIR, CHUNKS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com")

RATE_LIMIT_SECONDS = 3.0   # arXiv: max 1 request / 3s
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
USER_AGENT = f"MedSegLitBot/0.1 (mailto:{CONTACT_EMAIL})"

# --- Embedding + chunking ---
# BGE-small-en-v1.5 has a 512-token max input, so chunks are sized to fit it.
# Bound: a chunk is at most CHUNK_TOKENS, or (overlap + one max paragraph) in the
# edge case = 60 + 400 = 460, leaving ~50 tokens for the title+section prefix that
# Stage 7 prepends before embedding. Chunking at the spec's 800 would silently
# truncate a third of each chunk at embed time.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 60
MIN_CHUNK_TOKENS = 100
