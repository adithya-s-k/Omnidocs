#!/usr/bin/env python3
"""Rule-based reading order prediction."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_reading_order_result

img = create_test_image()

from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor

# Step 1: Layout detection
with Timer("Layout detection") as t_layout:
    layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu"))
    layout_result = layout_extractor.extract(img)

print(f"Detected {len(layout_result.bboxes)} layout boxes")

# Step 2: OCR extraction
with Timer("OCR extraction") as t_ocr:
    ocr_extractor = EasyOCR(config=EasyOCRConfig(gpu=False))
    ocr_result = ocr_extractor.extract(img)

print(f"Detected {len(ocr_result.text_blocks)} text blocks")

# Step 3: Reading order prediction
with Timer("Reading order") as t_order:
    predictor = RuleBasedReadingOrderPredictor()
    result = predictor.extract(
        layout_output=layout_result,
        ocr_output=ocr_result,
        image=img,
    )

verify_reading_order_result(result)
print_result("reading_order_rule_based", {
    "model": "RuleBased",
    "num_elements": len(result.elements),
    "layout_time": f"{t_layout.elapsed:.2f}s",
    "ocr_time": f"{t_ocr.elapsed:.2f}s",
    "order_time": f"{t_order.elapsed:.2f}s",
})
