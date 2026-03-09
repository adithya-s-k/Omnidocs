#!/usr/bin/env python3
"""Qwen layout detection - MLX backend (Apple Silicon only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_layout_result

img = create_test_image()

from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutMLXConfig

with Timer("Model load") as t_load:
    detector = QwenLayoutDetector(
        backend=QwenLayoutMLXConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
        )
    )

with Timer("Inference") as t_infer:
    result = detector.extract(img)

verify_layout_result(result)
print_result("qwen_layout_mlx", {
    "model": "Qwen3-VL-4B",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
