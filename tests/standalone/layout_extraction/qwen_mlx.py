"""
Qwen Layout Detection - MLX Backend (Apple Silicon)

Standalone test script. Run locally on Apple Silicon Mac.

Usage:
    python -m tests.standalone.layout_extraction.qwen_mlx path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class QwenLayoutMLXTest(BaseOmnidocsTest):
    """Test Qwen layout detection with MLX backend."""

    @property
    def test_name(self) -> str:
        return "qwen_layout_mlx"

    @property
    def backend_name(self) -> str:
        return "mlx"

    @property
    def task_name(self) -> str:
        return "layout_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutMLXConfig

        return QwenLayoutDetector(backend=QwenLayoutMLXConfig())

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_boxes": len(result.bboxes),
            "labels": [box.label.value for box in result.bboxes],
        }


if __name__ == "__main__":
    run_standalone_test(QwenLayoutMLXTest)
