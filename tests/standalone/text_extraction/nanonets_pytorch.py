"""
Nanonets Text Extraction - PyTorch Backend

Standalone test script. Run locally with GPU or via Modal runner.

Usage:
    python -m tests.standalone.text_extraction.nanonets_pytorch path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class NanonetsTextPyTorchTest(BaseOmnidocsTest):
    """Test Nanonets text extraction with PyTorch backend."""

    @property
    def test_name(self) -> str:
        return "nanonets_text_pytorch"

    @property
    def backend_name(self) -> str:
        return "pytorch_gpu"

    @property
    def task_name(self) -> str:
        return "text_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.text_extraction import NanonetsTextExtractor
        from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

        return NanonetsTextExtractor(backend=NanonetsTextPyTorchConfig())

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "content_length": len(result.content),
            "model": result.model_name,
        }


if __name__ == "__main__":
    run_standalone_test(NanonetsTextPyTorchTest)
