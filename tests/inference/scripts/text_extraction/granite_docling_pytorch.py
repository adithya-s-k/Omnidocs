#!/usr/bin/env python3
"""Granite Docling text extraction - PyTorch backend."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextPyTorchConfig

with Timer("Model load") as t_load:
    extractor = GraniteDoclingTextExtractor(
        backend=GraniteDoclingTextPyTorchConfig(
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "granite_docling_text_pytorch",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
