#!/usr/bin/env python3
"""LightOn text extraction - MLX backend (Apple Silicon only)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextMLXConfig

with Timer("Model load") as t_load:
    extractor = LightOnTextExtractor(
        backend=LightOnTextMLXConfig(
            model="lightonai/LightOnOCR-2-1B",
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "lighton_text_mlx",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)