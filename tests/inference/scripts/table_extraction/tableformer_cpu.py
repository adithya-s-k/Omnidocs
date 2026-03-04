#!/usr/bin/env python3
"""TableFormer table extraction - CPU."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_table_image, print_result, verify_table_result

img = create_table_image(rows=4, cols=3)

from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerExtractor

with Timer("Model load") as t_load:
    extractor = TableFormerExtractor(config=TableFormerConfig(device="cpu"))

with Timer("Inference") as t_infer:
    result = extractor.extract(img)

verify_table_result(result)
print_result("tableformer_cpu", {
    "model": "TableFormer",
    "num_tables": len(result.tables),
    "load_time": f"{t_load.elapsed:.2f}s",
    "inference_time": f"{t_infer.elapsed:.2f}s",
})
