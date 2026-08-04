"""Stage 0 (arXiv): download segmentation papers and their metadata.

For each accepted paper we write two files into data/raw/ :
  {paper_id}.pdf        the immutable source document
  {paper_id}.meta.json  the API's metadata (title, authors, year, DOI, ...)

Why trust the API for metadata instead of parsing the PDF?
  The arXiv API IS the ground truth for title/authors/year/DOI. PDF-parsed
  author lists and titles are noisy (ligatures, column order, footnote marks).
  We only parse the PDF later for its *body text*, never its bibliographic data.

Why arXiv first (not PubMed Central)?
  arXiv gives a direct PDF link and clean Atom-XML metadata, so we can prove the
  whole download -> raw/ path end to end today. PMC serves papers as tar.gz
  packages and is added later as a separate source module.

Run a smoke test (5 papers) before the full pull:
  python -m src.acquire_arxiv --limit 5
Full run (targets ~150, weighted toward head-and-neck / small structures):
  python -m src.acquire_arxiv
"""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from src import config

ARXIV_API = "http://export.arxiv.org/api/query"

# XML namespaces used in arXiv's Atom feed. ElementTree needs these to find tags.
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Restrict to categories where medical-imaging segmentation actually lives, so
# we do not pull unrelated CS papers that merely mention "segmentation".
CATEGORY_FILTER = "(cat:eess.IV OR cat:cs.CV OR cat:physics.med-ph)"

# The corpus is deliberately weighted toward head-and-neck and small-structure
# work (that is the user's domain and the target query's subject). Targets sum
# to ~150. Each bucket is a separate arXiv search; buckets are de-duplicated
# against each other so a paper matching two buckets is only downloaded once.
QUERY_PLAN = [
    {
        "name": "head_neck",
        "query": 'abs:"head and neck" AND abs:segmentation',
        "target": 45,
    },
    {
        "name": "organs_at_risk",
        "query": 'abs:"organs at risk" AND abs:segmentation',
        "target": 25,
    },
    {
        "name": "small_structures",
        "query": '(abs:lesion OR abs:nodule OR abs:vessel OR abs:"small structure") '
                 'AND abs:segmentation',
        "target": 25,
    },
    {
        "name": "seg_architectures",
        "query": '(abs:"U-Net" OR abs:nnU-Net OR abs:transformer) '
                 'AND abs:"medical image segmentation"',
        "target": 30,
    },
    {
        "name": "general_medseg",
        "query": 'abs:"medical image segmentation"',
        "target": 25,
    },
]

# One shared clock for rate limiting: every call to arxiv.org (API query OR PDF
# download) waits so that no two hits are closer than RATE_LIMIT_SECONDS apart.
_last_request_at = 0.0


def _rate_limit() -> None:
    """Sleep just long enough to honour arXiv's 1-request-per-3-seconds rule."""
    global _last_request_at
    elapsed = time.time() - _last_request_at
    wait = config.RATE_LIMIT_SECONDS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


def _get(url: str, params: dict | None = None) -> requests.Response:
    """HTTP GET with rate limiting and exponential-backoff retry.

    On a transient failure (network error, 5xx, 429) we wait 2^attempt seconds
    and try again, up to MAX_RETRIES. This is what makes a long pull survive a
    flaky connection instead of dying halfway through.
    """
    headers = {"User-Agent": config.USER_AGENT}
    last_err: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        _rate_limit()
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                return resp
            # 429 = too many requests, 5xx = server hiccup: both worth retrying.
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {resp.status_code}")
            else:
                resp.raise_for_status()
        except requests.RequestException as e:
            last_err = e
        backoff = 2 ** attempt
        print(f"    retry {attempt + 1}/{config.MAX_RETRIES} after {backoff}s "
              f"({last_err})")
        time.sleep(backoff)
    raise RuntimeError(f"GET failed after {config.MAX_RETRIES} attempts: {last_err}")


def _canonical_id(raw_id: str) -> str:
    """Turn an arXiv <id> URL into a stable, filesystem-safe paper_id.

    'http://arxiv.org/abs/2103.12345v2' -> 'arxiv_2103.12345'
    The version suffix (v2) is stripped so re-running never treats a revised
    version as a brand-new paper.
    """
    tail = raw_id.rsplit("/", 1)[-1]          # '2103.12345v2'
    if "v" in tail:                            # strip trailing version
        base, _, ver = tail.rpartition("v")
        if ver.isdigit():
            tail = base
    return "arxiv_" + tail.replace("/", "_")   # old-style ids contain '/'


def parse_atom(xml_text: str) -> list[dict]:
    """Parse one page of arXiv Atom XML into a list of metadata dicts."""
    root = ET.fromstring(xml_text)
    entries = []
    for e in root.findall("atom:entry", NS):
        raw_id = e.findtext("atom:id", default="", namespaces=NS)
        pdf_url = ""
        for link in e.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        published = e.findtext("atom:published", default="", namespaces=NS)
        year = int(published[:4]) if published[:4].isdigit() else None
        entries.append({
            "paper_id": _canonical_id(raw_id),
            "arxiv_id": raw_id.rsplit("/", 1)[-1],
            "title": " ".join(
                e.findtext("atom:title", default="", namespaces=NS).split()
            ),
            "authors": [
                a.findtext("atom:name", default="", namespaces=NS)
                for a in e.findall("atom:author", NS)
            ],
            "year": year,
            "doi": e.findtext("arxiv:doi", default=None, namespaces=NS),
            "primary_category": (
                e.find("arxiv:primary_category", NS).get("term")
                if e.find("arxiv:primary_category", NS) is not None else None
            ),
            "categories": [c.get("term") for c in e.findall("atom:category", NS)],
            "abstract": " ".join(
                e.findtext("atom:summary", default="", namespaces=NS).split()
            ),
            "pdf_url": pdf_url,
        })
    return entries


def download_pdf(url: str, dest) -> bool:
    """Download a PDF, verifying it really is a PDF (magic bytes '%PDF')."""
    resp = _get(url)
    if not resp.content.startswith(b"%PDF"):
        print(f"    NOT a PDF (starts with {resp.content[:8]!r}) - skipping")
        return False
    dest.write_bytes(resp.content)
    return True


def _existing_ids() -> set[str]:
    """paper_ids already downloaded, so re-runs skip completed work."""
    return {p.stem for p in config.RAW_DIR.glob("*.pdf")}


def acquire(limit: int | None = None) -> None:
    seen: set[str] = _existing_ids()
    print(f"Starting acquisition. {len(seen)} papers already in data/raw/.")
    totals = {"downloaded": 0, "skipped": 0, "failed": 0}

    for bucket in QUERY_PLAN:
        full_query = f'{bucket["query"]} AND {CATEGORY_FILTER}'
        got = 0
        start = 0
        page_size = 50
        print(f"\n=== bucket '{bucket['name']}' (target {bucket['target']}) ===")

        while got < bucket["target"]:
            if limit is not None and totals["downloaded"] >= limit:
                print("\nReached --limit, stopping early.")
                _report(totals, seen)
                return

            xml_text = _get(ARXIV_API, params={
                "search_query": full_query,
                "start": start,
                "max_results": page_size,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }).text
            page = parse_atom(xml_text)
            if not page:
                print("    no more results for this bucket")
                break

            for entry in page:
                if got >= bucket["target"]:
                    break
                if limit is not None and totals["downloaded"] >= limit:
                    break
                pid = entry["paper_id"]
                if pid in seen:
                    totals["skipped"] += 1
                    continue
                if not entry["pdf_url"]:
                    totals["failed"] += 1
                    continue

                pdf_path = config.RAW_DIR / f"{pid}.pdf"
                print(f"  [{totals['downloaded'] + 1}] {pid}  {entry['title'][:60]}")
                try:
                    if download_pdf(entry["pdf_url"], pdf_path):
                        entry["source"] = "arxiv"
                        entry["query_bucket"] = bucket["name"]
                        entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
                        (config.RAW_DIR / f"{pid}.meta.json").write_text(
                            json.dumps(entry, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        seen.add(pid)
                        got += 1
                        totals["downloaded"] += 1
                    else:
                        totals["failed"] += 1
                except Exception as ex:  # noqa: BLE001 - log and keep going
                    print(f"    ERROR downloading {pid}: {ex}")
                    totals["failed"] += 1

            start += page_size

    _report(totals, seen)


def _report(totals: dict, seen: set[str]) -> None:
    print("\n----- acquisition summary -----")
    print(f"  downloaded this run : {totals['downloaded']}")
    print(f"  skipped (already had): {totals['skipped']}")
    print(f"  failed              : {totals['failed']}")
    print(f"  total PDFs in raw/  : {len(seen)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download arXiv segmentation papers.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N successful downloads (for smoke tests)")
    args = ap.parse_args()
    acquire(limit=args.limit)
