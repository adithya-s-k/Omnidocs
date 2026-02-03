"""
Qwen Layout Detection - VLLM Backend

Standalone test script. Run locally with GPU or via Modal runner.

Usage:
    python -m tests.standalone.layout_extraction.qwen_vllm path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class QwenLayoutVLLMTest(BaseOmnidocsTest):
    """Test Qwen layout detection with VLLM backend."""

    @property
    def test_name(self) -> str:
        return "qwen_layout_vllm"

    @property
    def backend_name(self) -> str:
        return "vllm"

    @property
    def task_name(self) -> str:
        return "layout_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

        return QwenLayoutDetector(
            backend=QwenLayoutVLLMConfig(gpu_memory_utilization=0.85)
        )

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_boxes": len(result.bboxes),
            "labels": [box.label.value for box in result.bboxes],
        }


if __name__ == "__main__":
    run_standalone_test(QwenLayoutVLLMTest)
