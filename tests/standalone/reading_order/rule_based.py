"""
Rule-Based Reading Order Prediction - CPU

Standalone test script. Run locally on CPU.
Requires layout extraction and OCR results as input.

Usage:
    python -m tests.standalone.reading_order.rule_based path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class RuleBasedReadingOrderTest(BaseOmnidocsTest):
    """Test rule-based reading order prediction."""

    @property
    def test_name(self) -> str:
        return "reading_order_rule_based"

    @property
    def backend_name(self) -> str:
        return "pytorch_cpu"

    @property
    def task_name(self) -> str:
        return "reading_order"

    def create_extractor(self) -> Any:
        # For reading order, we need to create all components
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
        from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor

        return {
            "layout": DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu")),
            "ocr": EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False)),
            "predictor": RuleBasedReadingOrderPredictor(),
        }

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        # Run layout extraction
        layout_result = extractor["layout"].extract(image)

        # Run OCR
        ocr_result = extractor["ocr"].extract(image)

        # Predict reading order
        return extractor["predictor"].predict(layout_result, ocr_result)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_elements": len(result.ordered_elements),
            "element_types": [e.element_type.value for e in result.ordered_elements],
        }


if __name__ == "__main__":
    run_standalone_test(RuleBasedReadingOrderTest)
