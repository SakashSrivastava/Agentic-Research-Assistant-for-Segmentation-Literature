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
DB_PATH = DATA_DIR / "app.db"                       # corpus (read-only in prod)
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

for _d in (RAW_DIR, PARSED_DIR, CLEAN_DIR, CHUNKS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com")

# --- Web app (accounts + history) ---
# User data is read-write and must persist, so it lives OUTSIDE the read-only
# corpus mount. In prod, point USER_DB_PATH at a writable volume.
USER_DB_PATH = Path(os.getenv("USER_DB_PATH", DATA_DIR / "users.db"))
USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
# Default target for `python -m src.manage_admin` (the primary way to grant admin).
# Signup never grants admin, so knowing this email is not enough to become admin.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()
# Shell-less bootstrap (hosts with no server shell): when true, logging in as
# ADMIN_EMAIL promotes that account to admin. Needs this server-set flag AND
# control of the admin email, so it is not a public backdoor. Turn it off again
# once you are admin. (Locally, prefer `python -m src.manage_admin grant`.)
ADMIN_BOOTSTRAP = os.getenv("ADMIN_BOOTSTRAP", "false").lower() == "true"
# Signs session cookies. MUST be set to a fixed random value in prod, or sessions
# reset on every restart. A per-process fallback keeps local dev working.
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or os.urandom(32).hex()
# Free searches/user/day on the shared key before they must add their own (BYOK).
FREE_DAILY_QUERIES = int(os.getenv("FREE_DAILY_QUERIES", "5"))

# --- Security / deployment ---
SESSION_HOURS = int(os.getenv("SESSION_HOURS", "12"))          # idle session lifetime
REQUIRE_EMAIL_VERIFICATION = os.getenv("REQUIRE_EMAIL_VERIFICATION", "true").lower() == "true"
HTTPS_ONLY = os.getenv("HTTPS_ONLY", "false").lower() == "true"  # set true in production
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")       # for links in emails
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
# Optional SMTP for verification / password-reset email. If unset, the app runs in
# dev mode: it shows the verification/reset link in-app and logs it (no real email).
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")

RATE_LIMIT_SECONDS = 3.0   # arXiv: max 1 request / 3s
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
USER_AGENT = f"MedSegLitBot/0.1 (mailto:{CONTACT_EMAIL})"

# --- Embedding + chunking ---
# BGE-small-en-v1.5 has a 512-token max input, so chunks are sized to fit it.
# Bound: a chunk is at most CHUNK_TOKENS, or (overlap + one max paragraph) in the
# edge case = 60 + 400 = 460, leaving ~50 tokens for the title+section prefix that
# The embedder prepends title and section. Chunking at the spec's 800 would silently
# truncate a third of each chunk at embed time.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 60
MIN_CHUNK_TOKENS = 100

# --- Retrieval ---
RERANK_MODEL = "BAAI/bge-reranker-base"   # cross-encoder for reranking
RRF_K = 60                                # reciprocal-rank-fusion constant
RERANK_CANDIDATES = 25                    # retrieve 25, rerank, keep top 5
