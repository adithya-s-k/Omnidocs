#!/usr/bin/env python3
"""DeepSeek-OCR-2 text extraction - PyTorch backend."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_gpu_available, create_test_image, print_result, verify_text_result

check_gpu_available()

img = create_test_image()

from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

with Timer("Model load") as t_load:
    extractor = DeepSeekOCRTextExtractor(
        backend=DeepSeekOCRTextPyTorchConfig(
            model="unsloth/DeepSeek-OCR-2",
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
            crop_mode=True,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "deepseek_ocr_text_pytorch",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
