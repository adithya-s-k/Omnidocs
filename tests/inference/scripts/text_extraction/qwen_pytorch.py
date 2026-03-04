#!/usr/bin/env python3
"""Qwen text extraction - PyTorch backend."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

with Timer("Model load") as t_load:
    extractor = QwenTextExtractor(
        backend=QwenTextPyTorchConfig(
            model="Qwen/Qwen3-VL-2B-Instruct",
            device="cuda",
            torch_dtype="bfloat16",
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result("qwen_text_pytorch", {
    "model": result.model_name,
    "content_length": len(result.content),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
