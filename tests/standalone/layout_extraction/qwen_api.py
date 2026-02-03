"""
Qwen Layout Detection - API Backend

Standalone test script. Run locally via API (OpenRouter, etc.).

Usage:
    python -m tests.standalone.layout_extraction.qwen_api path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class QwenLayoutAPITest(BaseOmnidocsTest):
    """Test Qwen layout detection with API backend."""

    @property
    def test_name(self) -> str:
        return "qwen_layout_api"

    @property
    def backend_name(self) -> str:
        return "api"

    @property
    def task_name(self) -> str:
        return "layout_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

        return QwenLayoutDetector(backend=QwenLayoutAPIConfig())

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_boxes": len(result.bboxes),
            "labels": [box.label.value for box in result.bboxes],
        }


if __name__ == "__main__":
    run_standalone_test(QwenLayoutAPITest)
