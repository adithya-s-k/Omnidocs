#!/usr/bin/env python3
"""DocLayoutYOLO layout detection - GPU."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_layout_result

img = create_test_image()

from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

with Timer("Model load") as t_load:
    extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_layout_result(result)
print_result("doclayout_yolo_gpu", {
    "model": "DocLayoutYOLO",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
