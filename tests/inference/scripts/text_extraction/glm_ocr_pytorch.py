#!/usr/bin/env python3
"""GLM-OCR text extraction - PyTorch backend (zai-org/GLM-OCR, 0.9B, #1 OmniDocBench)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, check_gpu_available, create_test_image, print_result, verify_text_result

check_gpu_available()

img = create_test_image()

from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

with Timer("Model load") as t_load:
    extractor = GLMOCRTextExtractor(
        backend=GLMOCRPyTorchConfig(
            model="zai-org/GLM-OCR",
            device="cuda",
            torch_dtype="bfloat16",
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "glm_ocr_text_pytorch",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
        "result":f"{result.content}",
    },
)
