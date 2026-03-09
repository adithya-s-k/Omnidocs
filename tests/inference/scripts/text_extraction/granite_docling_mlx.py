#!/usr/bin/env python3
"""Granite Docling text extraction - MLX backend (Apple Silicon only)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextMLXConfig

with Timer("Model load") as t_load:
    extractor = GraniteDoclingTextExtractor(backend=GraniteDoclingTextMLXConfig())

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "granite_docling_text_mlx",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
