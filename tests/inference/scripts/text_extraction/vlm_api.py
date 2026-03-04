#!/usr/bin/env python3
"""VLM text extraction - API backend (LiteLLM / Gemini)."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_text_result

check_api_keys("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")

img = create_test_image()

from omnidocs.tasks.text_extraction import VLMTextExtractor
from omnidocs.vlm import VLMAPIConfig

with Timer("Model load") as t_load:
    extractor = VLMTextExtractor(
        config=VLMAPIConfig(
            model="gemini/gemini-2.5-flash",
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result("vlm_text_api", {
    "model": result.model_name,
    "content_length": len(result.content),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
