#!/usr/bin/env python3
"""GLM-OCR text extraction - VLLM backend."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import setup_vllm_env

setup_vllm_env()

from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

with Timer("Model load") as t_load:
    extractor = GLMOCRTextExtractor(
        backend=GLMOCRVLLMConfig(
            model="zai-org/GLM-OCR",
            gpu_memory_utilization=0.85,
            max_model_len=8192,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "glm_ocr_text_vllm",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
        "result":f"{result.content}",
    },
)
