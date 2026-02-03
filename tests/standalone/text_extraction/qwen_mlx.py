"""
Qwen Text Extraction - MLX Backend (Apple Silicon)

Standalone test script. Run locally on Apple Silicon Mac.

Usage:
    python -m tests.standalone.text_extraction.qwen_mlx path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class QwenTextMLXTest(BaseOmnidocsTest):
    """Test Qwen text extraction with MLX backend."""

    @property
    def test_name(self) -> str:
        return "qwen_text_mlx"

    @property
    def backend_name(self) -> str:
        return "mlx"

    @property
    def task_name(self) -> str:
        return "text_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

        return QwenTextExtractor(backend=QwenTextMLXConfig())

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image, output_format="markdown")

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "content_length": len(result.content),
            "model": result.model_name,
        }


if __name__ == "__main__":
    run_standalone_test(QwenTextMLXTest)
