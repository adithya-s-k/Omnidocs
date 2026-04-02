"""
benchmarks/omnidocbench/dataset.py

Downloads OmniDocBench from HuggingFace and returns a list of PageSample
objects (image bytes + image name) ready for inference.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

from benchmarks.base import PageSample


def load_omnidocbench(max_samples: Optional[int] = None) -> List[PageSample]:
    """
    Download OmniDocBench from HuggingFace (cached after first run) and
    return one PageSample per page image.

    Args:
        max_samples: If set, only load this many pages (useful for quick
                     iteration during development).
    """
    import json

    import huggingface_hub
    from PIL import Image as PILImage

    # Avoid PIL refusing large document scans
    PILImage.MAX_IMAGE_PIXELS = None

    print("Downloading OmniDocBench annotations...")
    json_path = huggingface_hub.hf_hub_download(
        repo_id="opendatalab/OmniDocBench",
        filename="OmniDocBench.json",
        repo_type="dataset",
    )
    annotations = json.loads(Path(json_path).read_text(encoding="utf-8"))

    samples: List[PageSample] = []

    for ann in annotations:
        if max_samples is not None and len(samples) >= max_samples:
            break

        img_name = Path(ann["page_info"]["image_path"]).name
        try:
            img_path = huggingface_hub.hf_hub_download(
                repo_id="opendatalab/OmniDocBench",
                filename=f"images/{img_name}",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  [skip] {img_name}: {e}")
            continue

        img = PILImage.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        samples.append(PageSample(image_bytes=buf.getvalue(), image_name=img_name))
        print(f"  [{len(samples)}] loaded {img_name}")

    print(f"  → {len(samples)} pages loaded")
    return samples
