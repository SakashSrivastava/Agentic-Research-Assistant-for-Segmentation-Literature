"""Stage 5: metadata enrichment via one cheap LLM call per paper.

From each paper's title + abstract, label anatomical_target and imaging_modality
(controlled vocab), stored on the manifest for filtered retrieval. Uses the
Groq wrapper (Llama 3.3 70B, free tier). Idempotent: skips already-labelled papers.

  python -m src.enrich --limit 3
  python -m src.enrich
"""
from __future__ import annotations

import argparse
import json

from src import config, llm, manifest

ANATOMY = ["head and neck", "brain", "eye/orbital", "lung/chest", "cardiac",
           "liver", "kidney", "pancreas", "prostate", "breast",
           "abdominal multi-organ", "musculoskeletal", "vessel", "skin",
           "cell/microscopy", "general/multiple", "other"]
MODALITY = ["CT", "MRI", "PET", "PET/CT", "X-ray", "ultrasound", "fundus", "OCT",
            "histopathology", "dermoscopy", "microscopy", "endoscopy",
            "mammography", "multiple", "none"]

SYSTEM = (
    "You label medical-imaging papers. Read the title and abstract and respond with a "
    "JSON object with exactly two string keys:\n"
    f'  "anatomical_target": one of {ANATOMY}\n'
    f'  "imaging_modality": one of {MODALITY}\n'
    "Pick the single best fit. Use 'general/multiple' or 'multiple' when the paper is "
    "not specific to one, and 'other'/'none' when nothing fits (e.g. a non-medical "
    "methods paper). Respond with only the JSON object."
)


def classify(title: str, abstract: str):
    data, usage = llm.chat_json(
        SYSTEM, f"Title: {title}\n\nAbstract: {abstract[:4000]}", max_tokens=512)
    at = data.get("anatomical_target")
    im = data.get("imaging_modality")
    return {
        "anatomical_target": at if at in ANATOMY else "other",
        "imaging_modality": im if im in MODALITY else "none",
    }, usage


def run(limit: int | None = None, force: bool = False) -> None:
    rows = manifest.active_papers()
    if limit:
        rows = rows[:limit]
    updates: dict[str, dict] = {}
    tot_in = tot_out = done = skipped = failed = 0

    for row in rows:
        pid = row["paper_id"]
        if row.get("anatomical_target") and not force:
            skipped += 1
            continue
        meta = json.loads((config.RAW_DIR / f"{pid}.meta.json").read_text(encoding="utf-8"))
        try:
            data, usage = classify(meta.get("title", ""), meta.get("abstract", ""))
        except Exception as ex:  # noqa: BLE001
            print(f"  FAIL {pid}: {ex}")
            failed += 1
            continue
        tot_in += usage.prompt_tokens
        tot_out += usage.completion_tokens
        updates[pid] = {**data, "enriched": True}
        done += 1
        print(f"  {pid}: {data['anatomical_target']} | {data['imaging_modality']}")

    if updates:
        manifest.update_manifest(updates)
    print("\n----- enrichment summary -----")
    print(f"  enriched: {done} | skipped: {skipped} | failed: {failed}")
    print(f"  tokens: {tot_in} in / {tot_out} out | cost: $0.00 (Groq free tier)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 5 metadata enrichment.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
