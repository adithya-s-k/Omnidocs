"""
Qwen Text Extraction - API Backend

Standalone test script. Run locally via API (OpenRouter, etc.).

Usage:
    python -m tests.standalone.text_extraction.qwen_api path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class QwenTextAPITest(BaseOmnidocsTest):
    """Test Qwen text extraction with API backend."""

    @property
    def test_name(self) -> str:
        return "qwen_text_api"

    @property
    def backend_name(self) -> str:
        return "api"

    @property
    def task_name(self) -> str:
        return "text_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

        return QwenTextExtractor(backend=QwenTextAPIConfig())

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image, output_format="markdown")

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "content_length": len(result.content),
            "model": result.model_name,
        }


if __name__ == "__main__":
    run_standalone_test(QwenTextAPITest)
