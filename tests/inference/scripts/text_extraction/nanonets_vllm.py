#!/usr/bin/env python3
"""Nanonets text extraction - VLLM backend."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import setup_vllm_env

setup_vllm_env()

from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import NanonetsTextExtractor
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextVLLMConfig

with Timer("Model load") as t_load:
    extractor = NanonetsTextExtractor(
        backend=NanonetsTextVLLMConfig(
            model="nanonets/Nanonets-OCR-s",
            gpu_memory_utilization=0.85,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result("nanonets_text_vllm", {
    "model": result.model_name,
    "content_length": len(result.content),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
