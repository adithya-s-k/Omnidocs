#!/usr/bin/env python3
"""MinerU VL layout detection - VLLM backend."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import setup_vllm_env

setup_vllm_env()

from helpers import Timer, create_test_image, print_result, verify_layout_result

img = create_test_image()

from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutVLLMConfig

with Timer("Model load") as t_load:
    detector = MinerUVLLayoutDetector(
        backend=MinerUVLLayoutVLLMConfig(
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            enforce_eager=True,
        )
    )

with Timer("Inference") as t_infer:
    result = detector.extract(img)

verify_layout_result(result)
print_result("mineruvl_layout_vllm", {
    "model": "MinerU2.5-2509-1.2B",
    "num_boxes": len(result.bboxes),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
