#!/usr/bin/env python3
"""Qwen text extraction - API backend (LiteLLM/OpenRouter)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_text_result

check_api_keys("OPENROUTER_API_KEY")

img = create_test_image()

from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

with Timer("Model load") as t_load:
    extractor = QwenTextExtractor(
        backend=QwenTextAPIConfig(
            model="openrouter/qwen/qwen3-vl-8b",
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result("qwen_text_api", {
    "model": result.model_name,
    "content_length": len(result.content),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
