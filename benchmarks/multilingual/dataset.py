"""
benchmarks/multilingual/dataset.py

Downloads NayanaOCRBench from HuggingFace and returns per-language lists of
PageSample objects (image bytes + image name) ready for inference, along with
the per-language GT JSON in OmniDocBench format.

Dataset: v1v1d1/NayanaOCRBench_Natural_final_transformed
  - 22 language subsets
  - Each row: image (PIL) + id (str) + omnidocbench (GT dict, OmniDocBench schema)
"""

from __future__ import annotations

import io
import json
from typing import Dict, List, Optional, Tuple

from benchmarks.base import PageSample

ALL_LANGUAGES = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "gu",
    "hi",
    "it",
    "ja",
    "kn",
    "ko",
    "ml",
    "mr",
    "or",
    "pa",
    "ru",
    "sa",
    "ta",
    "te",
    "th",
    "zh",
]


def load_multilingual(
    languages: List[str],
    max_per_language: Optional[int] = None,
) -> Tuple[Dict[str, List[PageSample]], Dict[str, List[dict]]]:
    """
    Download NayanaOCRBench from HuggingFace (cached after first run) and
    return samples and GT records, keyed by language.

    Args:
        languages:         List of language codes to load (subset of ALL_LANGUAGES).
        max_per_language:  If set, only load this many pages per language.

    Returns:
        samples_by_lang:  lang -> list of PageSample (image_bytes + image_name)
        gt_by_lang:       lang -> list of GT dicts (OmniDocBench schema, with
                          table html field populated from text field)
    """
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = None

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("The 'datasets' package is required. Install with: pip install datasets") from e

    samples_by_lang: Dict[str, List[PageSample]] = {}
    gt_by_lang: Dict[str, List[dict]] = {}

    for lang in languages:
        print(f"\n  [{lang}] Loading dataset subset...")
        try:
            ds = load_dataset(
                "v1v1d1/NayanaOCRBench_Natural_final_transformed",
                name=lang,
                split="train",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [{lang}] SKIP — could not load: {e}")
            continue

        samples: List[PageSample] = []
        gt_records: List[dict] = []
        loaded = 0

        for row in ds:
            if max_per_language is not None and loaded >= max_per_language:
                break

            image_name = row["id"]
            pil_image = row["image"].convert("RGB")

            # Serialise image to bytes for PageSample (matches omnidocbench dataset.py pattern)
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")

            samples.append(
                PageSample(
                    image_bytes=buf.getvalue(),
                    image_name=image_name,
                )
            )

            # GT record — may arrive as a string or already-parsed dict
            gt_record = row["omnidocbench"]
            if isinstance(gt_record, str):
                gt_record = json.loads(gt_record)

            # Fix: OmniDocBench eval pipeline reads `html` for TEDS scoring, but
            # NayanaOCRBench stores table HTML in the `text` field with html=null.
            # Populate `html` from `text` for every table element so TEDS works.
            for element in gt_record.get("layout_dets", []):
                if element.get("category_type") == "table" and element.get("html") is None:
                    text_val = element.get("text", "") or ""
                    if text_val.strip().startswith("<table"):
                        element["html"] = text_val

            gt_records.append(gt_record)
            loaded += 1

        samples_by_lang[lang] = samples
        gt_by_lang[lang] = gt_records
        print(f"  [{lang}] Done — {loaded} pages loaded")

    return samples_by_lang, gt_by_lang
