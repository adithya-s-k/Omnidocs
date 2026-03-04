#!/usr/bin/env python3
"""RT-DETR layout detection - GPU."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_layout_result

img = create_test_image()

from omnidocs.tasks.layout_extraction import RTDETRConfig, RTDETRLayoutExtractor

with Timer("Model load") as t_load:
    extractor = RTDETRLayoutExtractor(config=RTDETRConfig(device="cuda"))

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_layout_result(result)
print_result("rtdetr_gpu", {
    "model": "RTDETR",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
