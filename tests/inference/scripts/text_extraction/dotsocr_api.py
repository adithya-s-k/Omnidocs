#!/usr/bin/env python3
"""DotsOCR text extraction - API backend (LiteLLM)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_text_result

check_api_keys("OPENROUTER_API_KEY", "OPENAI_API_KEY")

img = create_test_image()

from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

with Timer("Model load") as t_load:
    extractor = DotsOCRTextExtractor(backend=DotsOCRAPIConfig())

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "dotsocr_text_api",
    {
        "model": "dots.ocr",
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
