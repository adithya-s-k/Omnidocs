#!/usr/bin/env python3
"""EasyOCR - GPU."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_ocr_result

img = create_test_image()

from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

with Timer("Model load") as t_load:
    extractor = EasyOCR(config=EasyOCRConfig(gpu=True))

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_ocr_result(result)
print_result("easyocr_gpu", {
    "model": "EasyOCR",
    "num_blocks": len(result.text_blocks),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
