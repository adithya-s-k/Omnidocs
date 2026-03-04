#!/usr/bin/env python3
"""MinerU VL text extraction - PyTorch backend."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

img = create_test_image()

from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

with Timer("Model load") as t_load:
    extractor = MinerUVLTextExtractor(
        backend=MinerUVLTextPyTorchConfig(
            device="cuda",
            torch_dtype="float16",
            use_flash_attention=False,
        )
    )

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_text_result(result)
print_result("mineruvl_text_pytorch", {
    "model": result.model_name,
    "content_length": len(result.content),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
