"""
RT-DETR Layout Detection - CPU

Standalone test script. Run locally on CPU.

Usage:
    python -m tests.standalone.layout_extraction.rtdetr_cpu path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class RTDETRCPUTest(BaseOmnidocsTest):
    """Test RT-DETR layout detection with CPU."""

    @property
    def test_name(self) -> str:
        return "rtdetr_cpu"

    @property
    def backend_name(self) -> str:
        return "pytorch_cpu"

    @property
    def task_name(self) -> str:
        return "layout_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.layout_extraction import RTDETRConfig, RTDETRLayoutExtractor

        return RTDETRLayoutExtractor(config=RTDETRConfig(device="cpu"))

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_boxes": len(result.bboxes),
            "labels": [box.label.value for box in result.bboxes],
        }


if __name__ == "__main__":
    run_standalone_test(RTDETRCPUTest)
