#!/usr/bin/env python3
"""MinerU VL layout detection - PyTorch backend."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_layout_result

img = create_test_image()

from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

with Timer("Model load") as t_load:
    detector = MinerUVLLayoutDetector(
        backend=MinerUVLLayoutPyTorchConfig(
            device="cuda",
            torch_dtype="float16",
            use_flash_attention=False,
        )
    )

with Timer("Inference") as t_infer:
    result = detector.extract(img)

verify_layout_result(result)
print_result("mineruvl_layout_pytorch", {
    "model": "MinerU2.5-2509-1.2B",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
