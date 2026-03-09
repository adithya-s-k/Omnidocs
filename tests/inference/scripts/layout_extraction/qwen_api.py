#!/usr/bin/env python3
"""Qwen layout detection - API backend (LiteLLM/OpenRouter)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_layout_result

check_api_keys("OPENROUTER_API_KEY")

img = create_test_image()

from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

with Timer("Model load") as t_load:
    detector = QwenLayoutDetector(
        backend=QwenLayoutAPIConfig(
            model="openrouter/qwen/qwen3-vl-8b",
        )
    )

with Timer("Inference") as t_infer:
    result = detector.extract(img)

verify_layout_result(result)
print_result("qwen_layout_api", {
    "model": "Qwen3-VL-8B",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
