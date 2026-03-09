#!/usr/bin/env python3
"""VLM layout detection - API backend (LiteLLM / Gemini)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_layout_result

check_api_keys("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")

img = create_test_image()

from omnidocs.tasks.layout_extraction import VLMLayoutDetector
from omnidocs.vlm import VLMAPIConfig

with Timer("Model load") as t_load:
    detector = VLMLayoutDetector(
        config=VLMAPIConfig(
            model="gemini/gemini-2.5-flash",
        )
    )

with Timer("Inference") as t_infer:
    result = detector.extract(img)

verify_layout_result(result)
print_result("vlm_layout_api", {
    "model": "gemini-2.5-flash",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
