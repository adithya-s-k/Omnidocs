#!/usr/bin/env python3
"""DeepSeek-OCR text extraction - VLLM backend (~2500 tokens/s on A100)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import setup_vllm_env

setup_vllm_env()

from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

with Timer("Model load") as t_load:
    extractor = DeepSeekOCRTextExtractor(
        backend=DeepSeekOCRTextVLLMConfig(
            model="deepseek-ai/DeepSeek-OCR",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            enforce_eager=True,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "deepseek_ocr_text_vllm",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
