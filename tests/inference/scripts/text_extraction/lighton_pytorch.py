#!/usr/bin/env python3
"""LightOn text extraction - PyTorch backend."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

with Timer("Model load") as t_load:
    extractor = LightOnTextExtractor(
        backend=LightOnTextPyTorchConfig(
            model="lightonai/LightOnOCR-2-1B",
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "lighton_text_pytorch",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s", 
    },
)