#!/usr/bin/env python3
"""DeepSeek-OCR text extraction - MLX backend (Apple Silicon only).

Note: MLX variants currently available for DeepSeek-OCR v1.
Check mlx-community for DeepSeek-OCR-2 as it becomes available.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

with Timer("Model load") as t_load:
    extractor = DeepSeekOCRTextExtractor(
        backend=DeepSeekOCRTextMLXConfig(
            model="mlx-community/DeepSeek-OCR-4bit",
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result(
    "deepseek_ocr_text_mlx",
    {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
