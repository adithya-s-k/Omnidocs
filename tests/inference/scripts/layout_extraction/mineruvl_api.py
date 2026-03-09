#!/usr/bin/env python3
"""MinerU VL layout detection - API backend (LiteLLM)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_layout_result

check_api_keys("OPENROUTER_API_KEY", "OPENAI_API_KEY")

img = create_test_image()

from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutAPIConfig

with Timer("Model load") as t_load:
    detector = MinerUVLLayoutDetector(backend=MinerUVLLayoutAPIConfig())

with Timer("Inference") as t_infer:
    result = detector.extract(img)

verify_layout_result(result)
print_result(
    "mineruvl_layout_api",
    {
        "model": "MinerU2.5-2509-1.2B",
        "num_boxes": len(result.bboxes),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
