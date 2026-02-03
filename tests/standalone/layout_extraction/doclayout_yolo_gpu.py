"""
DocLayout-YOLO Layout Detection - GPU

Standalone test script. Run locally with GPU or via Modal runner.

Usage:
    python -m tests.standalone.layout_extraction.doclayout_yolo_gpu path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class DocLayoutYOLOGPUTest(BaseOmnidocsTest):
    """Test DocLayout-YOLO with GPU."""

    @property
    def test_name(self) -> str:
        return "doclayout_yolo_gpu"

    @property
    def backend_name(self) -> str:
        return "pytorch_gpu"

    @property
    def task_name(self) -> str:
        return "layout_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

        return DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_boxes": len(result.bboxes),
            "labels": [box.label.value for box in result.bboxes],
        }


if __name__ == "__main__":
    run_standalone_test(DocLayoutYOLOGPUTest)
