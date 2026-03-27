"""
benchmarks/olmocr/dataset.py

Downloads the olmOCR-bench dataset from HuggingFace and returns a list
of OlmTestCase objects ready for inference + scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from benchmarks.base import OlmTestCase


# All available splits in allenai/olmOCR-bench
OLM_SPLITS = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
]

# Maps each split to the leaderboard column header used in reporting
SPLIT_LABELS = {
    "arxiv_math":      "ArXiv",
    "headers_footers": "HdrFtr",
    "long_tiny_text":  "TinyTxt",
    "multi_column":    "MultCol",
    "old_scans":       "OldScan",
    "old_scans_math":  "OldMath",
    "table_tests":     "Tables",
}

# Aliases used in some records' `type` field
_TYPE_ALIASES = {
    "absent":  "text_absent",
    "present": "text_present",
    "order":   "reading_order",
    "math":    "math",
    "table":   "table",
}


def load_olmocr_bench(
    splits: List[str],
    max_per_split: Optional[int] = None,
) -> List[OlmTestCase]:
    """
    Download allenai/olmOCR-bench from HuggingFace (cached after first run)
    and return one OlmTestCase per test case across the requested splits.

    Args:
        splits:        List of split names to load (subset of OLM_SPLITS).
        max_per_split: If set, load at most this many cases per split.
    """
    import os
    import huggingface_hub

    # Disable HF transfer for the snapshot download — avoids occasional issues
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    print("Downloading olmOCR-bench dataset (snapshot)...")
    local_dir = Path(huggingface_hub.snapshot_download(
        repo_id="allenai/olmOCR-bench",
        repo_type="dataset",
    ))
    print(f"  Dataset at: {local_dir}")

    cases: List[OlmTestCase] = []
    pdf_cache: dict = {}

    for split in splits:
        jsonl_path = local_dir / "bench_data" / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"  [warn] JSONL not found: {jsonl_path}")
            continue

        split_cases: List[OlmTestCase] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                # Resolve check_type from field or infer from payload keys
                check_type = rec.get("check_type", rec.get("type", ""))
                check_type = _TYPE_ALIASES.get(check_type, check_type)

                if not check_type:
                    if "math" in rec:
                        check_type = "math"
                    elif "before" in rec and "after" in rec:
                        check_type = "reading_order"
                    elif "cell" in rec:
                        check_type = "table"
                    elif "text" in rec:
                        check_type = "text_absent" if rec.get("absent", False) else "text_present"

                pdf_rel = rec.get("pdf", "")
                page_num = int(rec.get("page", 0))

                # PDFs live at bench_data/pdfs/<pdf_rel>
                pdf_path = local_dir / "bench_data" / "pdfs" / pdf_rel
                if not pdf_path.exists():
                    print(f"  [skip] not found: {pdf_path}")
                    continue

                pdf_key = str(pdf_path)
                if pdf_key not in pdf_cache:
                    pdf_cache[pdf_key] = pdf_path.read_bytes()

                case_id = (
                    f"{split}/{Path(pdf_rel).stem}"
                    f"/p{page_num}/{check_type}/{len(split_cases)}"
                )
                split_cases.append(OlmTestCase(
                    pdf_bytes=pdf_cache[pdf_key],
                    page_num=page_num,
                    check_type=check_type,
                    split=split,
                    case_id=case_id,
                    payload=rec,
                ))

                if max_per_split and len(split_cases) >= max_per_split:
                    break

        print(f"  → {len(split_cases)} test cases from {split}")
        cases.extend(split_cases)

    print(f"  Total: {len(cases)} test cases across {len(splits)} splits")
    return cases