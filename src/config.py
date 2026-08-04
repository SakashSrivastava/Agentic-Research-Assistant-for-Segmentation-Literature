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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com")

RATE_LIMIT_SECONDS = 3.0   # arXiv: max 1 request / 3s
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
USER_AGENT = f"MedSegLitBot/0.1 (mailto:{CONTACT_EMAIL})"
