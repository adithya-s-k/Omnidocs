#!/usr/bin/env python3
"""TATR table extraction - PyTorch CPU."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_table_image, print_result, verify_table_result

img = create_table_image(rows=4, cols=3)

from omnidocs.tasks.table_extraction import TATRExtractor
from omnidocs.tasks.table_extraction.tatr import TATRPyTorchConfig, TATRVariant

with Timer("Model load") as t_load:
    extractor = TATRExtractor(backend=TATRPyTorchConfig(device="cpu", variant=TATRVariant.ALL))

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_table_result(result)
print_result(
    "tatr_pytorch_cpu",
    {
        "model": f"TATR-{TATRVariant.ALL.value}",
        "backend": "pytorch",
        "num_cells": len(result.cells),
        "grid": f"{result.num_rows}x{result.num_cols}",
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    },
)
