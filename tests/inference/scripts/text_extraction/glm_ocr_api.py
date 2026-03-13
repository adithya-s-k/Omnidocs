#!/usr/bin/env python3
"""GLM-OCR text extraction - API backend (self-hosted vLLM or ZhipuAI)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_api_keys, create_test_image, print_result, verify_text_result

check_api_keys("ZAI_API_KEY", "OPENAI_API_KEY")

img = create_test_image()

from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

with Timer("Model load") as t_load:
    extractor = GLMOCRTextExtractor(
        backend=GLMOCRAPIConfig(
            model="zai-org/GLM-OCR",
            api_key=os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("GLM_OCR_API_BASE"),
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "glm_ocr_text_api",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
