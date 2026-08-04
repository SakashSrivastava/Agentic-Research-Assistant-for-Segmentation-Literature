"""Stage 0 (arXiv): download papers as data/raw/{paper_id}.pdf + {paper_id}.meta.json.

Metadata comes from the API (ground truth), never parsed from the PDF.

  python -m src.acquire_arxiv --limit 5   # smoke test
  python -m src.acquire_arxiv             # full pull (~200, majority medical)
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
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Per-bucket category filters: medical work lives in eess.IV/physics.med-ph,
# broader-AI work in cs.LG/cs.AI/stat.ML. A single filter would exclude one side.
MED_CATS = "(cat:eess.IV OR cat:cs.CV OR cat:physics.med-ph)"
CV_ML_CATS = "(cat:cs.CV OR cat:cs.LG OR cat:stat.ML)"
AI_CATS = "(cat:cs.CV OR cat:cs.LG OR cat:cs.AI OR cat:eess.IV)"
DEFAULT_CATEGORY_FILTER = MED_CATS

# Majority medical, minority broader-AI. Buckets are de-duplicated against each
# other. Tune any target, or cap a run with --limit N. (~250 total)
QUERY_PLAN = [
    {"name": "head_neck",
     "query": 'abs:"head and neck" AND abs:segmentation',
     "target": 40, "categories": MED_CATS},
    {"name": "organs_at_risk",
     "query": 'abs:"organs at risk" AND abs:segmentation',
     "target": 20, "categories": MED_CATS},
    {"name": "orbital_ocular",
     "query": '(abs:orbital OR abs:orbit OR abs:ocular OR abs:ophthalmic '
              'OR abs:retinal OR abs:"eye socket") AND abs:segmentation',
     "target": 20, "categories": MED_CATS},
    {"name": "brain_neuro",
     "query": '(abs:brain OR abs:"brain tumor" OR abs:neuroimaging OR abs:glioma) '
              'AND abs:segmentation',
     "target": 15, "categories": MED_CATS},
    {"name": "cardiac_abdominal",
     "query": '(abs:cardiac OR abs:"whole heart" OR abs:abdominal OR abs:liver '
              'OR abs:kidney OR abs:pancreas) AND abs:segmentation',
     "target": 15, "categories": MED_CATS},
    {"name": "small_structures",
     "query": '(abs:lesion OR abs:nodule OR abs:vessel OR abs:"small structure") '
              'AND abs:segmentation',
     "target": 20, "categories": MED_CATS},
    {"name": "seg_architectures",
     "query": '(abs:"U-Net" OR abs:nnU-Net OR abs:transformer) '
              'AND abs:"medical image segmentation"',
     "target": 20, "categories": MED_CATS},
    {"name": "general_medseg",
     "query": 'abs:"medical image segmentation"',
     "target": 20, "categories": MED_CATS},
    {"name": "diffusion_models",
     "query": 'abs:"diffusion model" AND '
              '(abs:segmentation OR abs:"image generation" OR abs:generative)',
     "target": 20, "categories": CV_ML_CATS},
    {"name": "foundation_segmentation",
     "query": '(abs:"segment anything" OR abs:"foundation model" OR abs:SAM) '
              'AND abs:segmentation',
     "target": 15, "categories": CV_ML_CATS},
    {"name": "general_segmentation",
     "query": 'abs:"semantic segmentation" OR abs:"instance segmentation" '
              'OR abs:"panoptic segmentation"',
     "target": 15, "categories": CV_ML_CATS},
    {"name": "ai_medtech",
     "query": '(abs:healthcare OR abs:"digital health" OR abs:"clinical decision" '
              'OR abs:"medical device") '
              'AND (abs:"deep learning" OR abs:"machine learning")',
     "target": 15, "categories": AI_CATS},
    {"name": "dl_approaches",
     "query": '(abs:"self-supervised" OR abs:"vision transformer" '
              'OR abs:"generative adversarial") '
              'AND (abs:segmentation OR abs:"representation learning")',
     "target": 15, "categories": CV_ML_CATS},
]

_last_request_at = 0.0


def _rate_limit() -> None:
    global _last_request_at
    wait = config.RATE_LIMIT_SECONDS - (time.time() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


def _get(url: str, params: dict | None = None) -> requests.Response:
    """Rate-limited GET with exponential-backoff retry on transient failures."""
    headers = {"User-Agent": config.USER_AGENT}
    last_err: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        _rate_limit()
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {resp.status_code}")
            else:
                resp.raise_for_status()
        except requests.RequestException as e:
            last_err = e
        backoff = 2 ** attempt
        print(f"    retry {attempt + 1}/{config.MAX_RETRIES} after {backoff}s ({last_err})")
        time.sleep(backoff)
    raise RuntimeError(f"GET failed after {config.MAX_RETRIES} attempts: {last_err}")


def _canonical_id(raw_id: str) -> str:
    """'…/abs/2103.12345v2' -> 'arxiv_2103.12345' (version stripped for stability)."""
    tail = raw_id.rsplit("/", 1)[-1]
    base, _, ver = tail.rpartition("v")
    if ver.isdigit():
        tail = base
    return "arxiv_" + tail.replace("/", "_")


def parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = []
    for e in root.findall("atom:entry", NS):
        raw_id = e.findtext("atom:id", default="", namespaces=NS)
        pdf_url = next((l.get("href", "") for l in e.findall("atom:link", NS)
                        if l.get("title") == "pdf"), "")
        published = e.findtext("atom:published", default="", namespaces=NS)
        prim = e.find("arxiv:primary_category", NS)
        entries.append({
            "paper_id": _canonical_id(raw_id),
            "arxiv_id": raw_id.rsplit("/", 1)[-1],
            "title": " ".join(e.findtext("atom:title", default="", namespaces=NS).split()),
            "authors": [a.findtext("atom:name", default="", namespaces=NS)
                        for a in e.findall("atom:author", NS)],
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "doi": e.findtext("arxiv:doi", default=None, namespaces=NS),
            "primary_category": prim.get("term") if prim is not None else None,
            "categories": [c.get("term") for c in e.findall("atom:category", NS)],
            "abstract": " ".join(e.findtext("atom:summary", default="", namespaces=NS).split()),
            "pdf_url": pdf_url,
        })
    return entries


def download_pdf(url: str, dest) -> bool:
    resp = _get(url)
    if not resp.content.startswith(b"%PDF"):
        print(f"    NOT a PDF (starts with {resp.content[:8]!r}) - skipping")
        return False
    dest.write_bytes(resp.content)
    return True


def _existing_ids() -> set[str]:
    return {p.stem for p in config.RAW_DIR.glob("*.pdf")}


def acquire(limit: int | None = None) -> None:
    seen = _existing_ids()
    print(f"Starting acquisition. {len(seen)} papers already in data/raw/.")
    totals = {"downloaded": 0, "skipped": 0, "failed": 0}

    for bucket in QUERY_PLAN:
        cats = bucket.get("categories", DEFAULT_CATEGORY_FILTER)
        full_query = f'({bucket["query"]}) AND {cats}'
        got, start, page_size = 0, 0, 50
        print(f"\n=== bucket '{bucket['name']}' (target {bucket['target']}) ===")

        while got < bucket["target"]:
            if limit is not None and totals["downloaded"] >= limit:
                print("\nReached --limit, stopping early.")
                return _report(totals, seen)

            page = parse_atom(_get(ARXIV_API, params={
                "search_query": full_query, "start": start, "max_results": page_size,
                "sortBy": "relevance", "sortOrder": "descending",
            }).text)
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

                print(f"  [{totals['downloaded'] + 1}] {pid}  {entry['title'][:60]}")
                try:
                    if download_pdf(entry["pdf_url"], config.RAW_DIR / f"{pid}.pdf"):
                        entry["source"] = "arxiv"
                        entry["query_bucket"] = bucket["name"]
                        entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
                        (config.RAW_DIR / f"{pid}.meta.json").write_text(
                            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
                        seen.add(pid)
                        got += 1
                        totals["downloaded"] += 1
                    else:
                        totals["failed"] += 1
                except Exception as ex:  # noqa: BLE001
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
                    help="stop after N successful downloads (smoke test)")
    acquire(limit=ap.parse_args().limit)
