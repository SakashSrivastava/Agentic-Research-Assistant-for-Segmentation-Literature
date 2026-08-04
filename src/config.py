"""Central configuration: paths and constants shared across the whole pipeline.

Keeping these in one place means no later stage has to guess where data lives,
and changing a directory or a rate limit is a one-line edit instead of a
search-and-replace across the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains `src/`. Resolve from this file's
# location so it works no matter where the script is launched from.
ROOT = Path(__file__).resolve().parent.parent

# Load secrets from .env once, here. Any module that imports `config` gets them.
load_dotenv(ROOT / ".env")

# --- Directory layout (mirrors section 7 of the spec) ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        # immutable source PDFs + API metadata
PARSED_DIR = DATA_DIR / "parsed"  # page-level blocks with layout info
CLEAN_DIR = DATA_DIR / "clean"    # normalised text with sections
CHUNKS_DIR = DATA_DIR / "chunks"  # final chunks with metadata
INDEX_DIR = DATA_DIR / "index"    # chroma collection + bm25 index
DB_PATH = DATA_DIR / "app.db"           # sqlite: manifest + extracted metrics
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

# Create the data directories if missing, so a fresh clone just works.
for _d in (RAW_DIR, PARSED_DIR, CLEAN_DIR, CHUNKS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Secrets / contact ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# arXiv / PMC ask for a contact address so they can reach you if a script
# misbehaves. Falls back to a placeholder if .env is not set yet.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com")

# --- Politeness / networking ---
RATE_LIMIT_SECONDS = 3.0   # arXiv requests at most 1 hit every 3 seconds
MAX_RETRIES = 5            # exponential backoff attempts before giving up
REQUEST_TIMEOUT = 60       # seconds before a single HTTP call is abandoned
USER_AGENT = f"MedSegLitBot/0.1 (mailto:{CONTACT_EMAIL})"
